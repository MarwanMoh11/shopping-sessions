#pragma once
#include "common.hpp"
#include "parquet_writer.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace SS_ENGINE_NS {

// ============================================================================
// Cgroup + Affinity-Aware CPU Detection (Never Oversubscribes Pinned vCPUs!)
// ============================================================================
static int detect_ncpu() {
    if (const char* e = std::getenv("NCPU")) {
        int v = std::atoi(e);
        if (v > 0) return v > 64 ? 64 : v;
    }
    int n = int(std::thread::hardware_concurrency());
    if (n < 1) n = 1;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (::sched_getaffinity(0, sizeof(set), &set) == 0) {
        int aff = CPU_COUNT(&set);
        if (aff > 0 && aff < n) n = aff;
    }
    // Check cgroup v2 cpu.max ("<quota> <period>" or "max <period>")
    if (FILE* f = std::fopen("/sys/fs/cgroup/cpu.max", "r")) {
        char qbuf[64]{};
        long long period = 0;
        if (std::fscanf(f, "%63s %lld", qbuf, &period) == 2 && period > 0 && qbuf[0] != 'm') {
            long long quota = std::atoll(qbuf);
            if (quota > 0) {
                int cg = int((quota + period - 1) / period);
                if (cg < 1) cg = 1;
                if (cg < n) n = cg;
            }
        }
        std::fclose(f);
    } else if (FILE* fq = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r")) {
        long long quota = -1, period = 100000;
        if (std::fscanf(fq, "%lld", &quota) == 1 && quota > 0) {
            if (FILE* fp = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r")) {
                long long p = 0;
                if (std::fscanf(fp, "%lld", &p) == 1 && p > 0) period = p;
                std::fclose(fp);
            }
            int cg = int((quota + period - 1) / period);
            if (cg < 1) cg = 1;
            if (cg < n) n = cg;
        }
        std::fclose(fq);
    }
#endif
    if (n < 1) n = 1;
    if (n > 64) n = 64;
    return n;
}

// ============================================================================
// Persistent Low-Latency Thread Pool (Lock-Free Fast Wakeup when Spinning!)
// ============================================================================
class ThreadPool {
public:
    int nthreads = 1;

    explicit ThreadPool(int n) : nthreads(n < 1 ? 1 : (n > 64 ? 64 : n)) {
        workers.reserve(static_cast<size_t>(nthreads - 1));
        for (int t = 1; t < nthreads; ++t) {
            workers.emplace_back([this, t]() { worker_loop(t); });
        }
    }

    template <typename Fn>
    void run(Fn&& fn) {
        if (nthreads == 1) {
            fn(0);
            return;
        }
        std::function<void(int)> f = std::ref(fn);
        current_fn = &f;
        done_count.store(0, std::memory_order_relaxed);
        gen.fetch_add(1, std::memory_order_seq_cst);
        if (__builtin_expect(sleeping.load(std::memory_order_seq_cst) > 0, 0)) {
            std::lock_guard<std::mutex> lk(mu);
            cv.notify_all();
        }
        fn(0);
        done_count.fetch_add(1, std::memory_order_acq_rel);
        while (done_count.load(std::memory_order_acquire) < nthreads) {
            #if defined(__x86_64__)
            __builtin_ia32_pause();
            #elif defined(__aarch64__)
            asm volatile("yield");
            #endif
        }
        current_fn = nullptr;
    }

private:
    std::vector<std::thread> workers;
    std::mutex mu;
    std::condition_variable cv;
    std::atomic<uint32_t> gen{0};
    std::atomic<int> done_count{0};
    std::atomic<int> sleeping{0};
    std::function<void(int)>* current_fn = nullptr;

    void worker_loop(int tid) {
        uint32_t my_gen = 0;
        while (true) {
            uint32_t target = my_gen + 1;
            int spins = 0;
            while (gen.load(std::memory_order_acquire) < target) {
                if (++spins > 60000) {
                    sleeping.fetch_add(1, std::memory_order_seq_cst);
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&]() { return gen.load(std::memory_order_acquire) >= target; });
                    sleeping.fetch_sub(1, std::memory_order_relaxed);
                    break;
                }
                #if defined(__x86_64__)
                __builtin_ia32_pause();
                #elif defined(__aarch64__)
                asm volatile("yield");
                #endif
            }
            my_gen = target;
            (*current_fn)(tid);
            done_count.fetch_add(1, std::memory_order_acq_rel);
        }
    }
};

// ============================================================================
// L1 Write-Combining Slab-Backed Partitioned Store (64B Aligned + NT Stores!)
// ============================================================================
template <typename T, uint32_t CAP, int NUM_B, uint32_t WC_CAP, bool UseNT = false>
struct PartitionedStore {
    struct alignas(64) Block {
        Block* next;
        uint32_t count;
        alignas(64) T items[CAP];
    };
    static constexpr size_t SLAB_BYTES = 16ULL << 20; // 16 MB slabs

    struct alignas(64) ThreadState {
        alignas(64) T wc_buf[NUM_B][WC_CAP];
        uint8_t wc_cnt[NUM_B]{};
        Block* heads[NUM_B]{};
        size_t counts[NUM_B]{};
        std::vector<void*> slabs;
        char* slab = nullptr;
        size_t slab_pos = SLAB_BYTES;

        inline Block* alloc_block() {
            constexpr size_t bsz = (sizeof(Block) + 63) & ~size_t(63);
            if (__builtin_expect(slab_pos + bsz > SLAB_BYTES, 0)) {
                slab = static_cast<char*>(alloc_mmap(SLAB_BYTES));
                slabs.push_back(slab);
                slab_pos = 0;
            }
            auto* b = reinterpret_cast<Block*>(slab + slab_pos);
            slab_pos += bsz;
            b->next = nullptr;
            b->count = 0;
            return b;
        }

        inline void flush_bucket_full(uint32_t b) {
            Block* h = heads[b];
            if (__builtin_expect(h == nullptr || h->count + WC_CAP > CAP, 0)) {
                Block* nb = alloc_block();
                nb->next = h;
                heads[b] = nb;
                h = nb;
            }
            if constexpr (UseNT && ((WC_CAP * sizeof(T)) % 64 == 0)) {
                stream_copy_64b_aligned(h->items + h->count, wc_buf[b], WC_CAP * sizeof(T));
            } else {
                std::memcpy(h->items + h->count, wc_buf[b], WC_CAP * sizeof(T));
            }
            h->count += WC_CAP;
            counts[b] += WC_CAP;
            wc_cnt[b] = 0;
        }

        inline void flush_bucket_partial(uint32_t b) {
            uint32_t n = wc_cnt[b];
            if (n == 0) return;
            Block* h = heads[b];
            if (__builtin_expect(h == nullptr || h->count + n > CAP, 0)) {
                Block* nb = alloc_block();
                nb->next = h;
                heads[b] = nb;
                h = nb;
            }
            std::memcpy(h->items + h->count, wc_buf[b], size_t(n) * sizeof(T));
            h->count += n;
            counts[b] += n;
            wc_cnt[b] = 0;
        }

        inline void push(uint32_t bucket, const T& val) {
            uint8_t c = wc_cnt[bucket];
            wc_buf[bucket][c++] = val;
            if (__builtin_expect(c == WC_CAP, 0)) {
                flush_bucket_full(bucket);
            } else {
                wc_cnt[bucket] = c;
            }
        }

        void flush_all() {
            for (uint32_t b = 0; b < NUM_B; ++b) {
                if (wc_cnt[b]) flush_bucket_partial(b);
            }
            if constexpr (UseNT) stream_fence();
        }
    };

    int nthreads = 1;
    std::vector<ThreadState*> ts;

    void init(int nt) {
        nthreads = nt;
        ts.resize(static_cast<size_t>(nt));
        for (int t = 0; t < nt; ++t) {
            void* mem = alloc_mmap((sizeof(ThreadState) + 4095) & ~size_t(4095));
            ts[size_t(t)] = new (mem) ThreadState();
        }
    }

    size_t bucket_count(int b) const {
        size_t s = 0;
        for (int t = 0; t < nthreads; ++t) s += ts[size_t(t)]->counts[b];
        return s;
    }
};

// ============================================================================
// Zero-Copy MMAP Parallel File Opener & Non-Overlapping 64B Block CSV Scanner
// ============================================================================
struct MmapCsvFile {
    const char* ptr = nullptr;
    size_t file_size = 0;
    size_t data_start = 0;
    uint8_t kind = 0; // 0 = click, 1 = event, 2 = link
};

struct MmapTask {
    const char* base;
    size_t file_size;
    size_t data_start;
    size_t start;
    size_t end;
    uint8_t kind;
};

static void list_csv_paths(const std::string& dir, uint8_t kind, std::vector<std::pair<std::string, uint8_t>>& out) {
    DIR* d = ::opendir(dir.c_str());
    if (!d) return;
    size_t start_idx = out.size();
    while (auto* ent = ::readdir(d)) {
        std::string_view name(ent->d_name);
        if (name.size() >= 4 && name.substr(name.size() - 4) == ".csv") {
#ifdef DT_REG
            if (ent->d_type != DT_REG && ent->d_type != DT_UNKNOWN) continue;
#endif
            std::string full = dir + "/" + std::string(name);
#ifdef DT_UNKNOWN
            if (ent->d_type == DT_UNKNOWN) {
                struct stat st{};
                if (::stat(full.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
            }
#endif
            out.emplace_back(std::move(full), kind);
        }
    }
    ::closedir(d);
    std::sort(out.begin() + ptrdiff_t(start_idx), out.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
}

static MmapCsvFile open_and_mmap_csv(const std::string& path, uint8_t kind) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return {};
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        return {};
    }
    size_t fsz = size_t(st.st_size);
    void* m = ::mmap(nullptr, fsz, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd); // POSIX keeps mmap valid after close -> zero EMFILE risk!
    if (m == MAP_FAILED) return {};
    const char* cp = static_cast<const char*>(m);
    const char* nl = find_nl_simd(cp, cp + fsz);
    size_t ds = nl ? (size_t(nl - cp) + 1) : fsz;
    if (ds >= fsz) {
        ::munmap(m, fsz);
        return {};
    }
    return {cp, fsz, ds, kind};
}

static void build_mmap_tasks(
    const std::vector<MmapCsvFile>& files,
    uint8_t kind_mask, // bitmask: (1<<kind)
    int nthreads,
    std::vector<MmapTask>& tasks
) {
    size_t total_bytes = 0;
    for (const auto& f : files) {
        if (f.ptr && ((kind_mask >> f.kind) & 1U)) total_bytes += (f.file_size - f.data_start);
    }
    if (total_bytes == 0) return;
    size_t chunk_sz = std::max<size_t>(131072, std::min<size_t>(4ULL << 20, total_bytes / size_t(nthreads * 8) + 1));
    for (const auto& f : files) {
        if (!f.ptr || !((kind_mask >> f.kind) & 1U)) continue;
        size_t pos = f.data_start;
        while (pos < f.file_size) {
            size_t end = std::min<size_t>(f.file_size, pos + chunk_sz);
            tasks.push_back({f.ptr, f.file_size, f.data_start, pos, end, f.kind});
            pos = end;
        }
    }
}

// Non-overlapping 64-byte SIMD block-bitmask CSV scanner directly on mmap!
// Computes (nl, comma, quote) ONCE per 64 bytes (ZERO redundant loads, ZERO pread copy!).
// 100% exact across all line lengths (1..127 fast path, >=128 scalar, EOF without '\n').
template <int N, typename OnFields>
static inline void scan_mmap_csv_simd(const MmapTask& task, OnFields&& on_fields) {
    const char* base = task.base;
    const size_t n = task.file_size;
    const size_t limit = task.end;
    size_t cur = task.start;

    if (cur > task.data_start && base[cur - 1] != '\n') {
        const char* nl = find_nl_simd(base + cur, base + n);
        if (!nl) return;
        cur = size_t(nl - base) + 1;
        if (cur >= limit) return;
    }

    Slice f[N];
    uint64_t nls[64], cms[66], qms[66];

    while (cur < limit) {
        size_t avail = n - cur;
        size_t nblks = std::min<size_t>(64, avail >> 6);
        if (__builtin_expect(nblks == 0, 0)) break;

        const char* bptr = base + cur;
        for (size_t bi = 0; bi < nblks; ++bi) {
            Vec64 v = Vec64::load(bptr + (bi << 6));
            nls[bi] = v.eq('\n');
            cms[bi] = v.eq(',');
            qms[bi] = v.eq('"');
        }
        cms[nblks] = cms[nblks + 1] = 0;
        qms[nblks] = qms[nblks + 1] = 0;

        size_t ls = 0;
        const size_t limit_rel = limit - cur;
        for (size_t bi = 0; bi < nblks; ++bi) {
            uint64_t nm = nls[bi];
            while (nm) {
                size_t nl_rel = (bi << 6) + size_t(__builtin_ctzll(nm));
                nm &= nm - 1;
                size_t L = nl_rel - ls;
                if (__builtin_expect(L > 0 && L < 128, 1)) {
                    size_t sbi = ls >> 6;
                    unsigned sh = unsigned(ls & 63);
                    uint64_t c0, c1, q0, q1;
                    if (__builtin_expect(sh != 0, 1)) {
                        unsigned rsh = 64U - sh;
                        uint64_t nc = cms[sbi + 1], nnc = cms[sbi + 2];
                        c0 = (cms[sbi] >> sh) | (nc << rsh);
                        c1 = (nc >> sh) | (nnc << rsh);
                        uint64_t nq = qms[sbi + 1], nnq = qms[sbi + 2];
                        q0 = (qms[sbi] >> sh) | (nq << rsh);
                        q1 = (nq >> sh) | (nnq << rsh);
                    } else {
                        c0 = cms[sbi]; c1 = cms[sbi + 1];
                        q0 = qms[sbi]; q1 = qms[sbi + 1];
                    }
                    unsigned __int128 lm128 = (static_cast<unsigned __int128>(1) << L) - 1;
                    c0 &= uint64_t(lm128);
                    c1 &= uint64_t(lm128 >> 64);
                    q0 &= uint64_t(lm128);
                    q1 &= uint64_t(lm128 >> 64);
                    if (__builtin_expect(extract_fields_branchless<N>(bptr + ls, L, c0, c1, q0, q1, f), 1)) {
                        on_fields(f);
                    }
                } else if (L >= 128) {
                    if (parse_csv_line_scalar<N>(bptr + ls, bptr + nl_rel, f)) {
                        on_fields(f);
                    }
                }
                ls = nl_rel + 1;
                if (ls >= limit_rel) return;
            }
        }

        if (__builtin_expect(ls == 0, 0)) {
            // Rare line >= 4 KB (no '\n' in all nblks)
            const char* nl = find_nl_simd(bptr + (nblks << 6), base + n);
            const char* le = nl ? nl : (base + n);
            if (parse_csv_line_scalar<N>(bptr, le, f)) {
                on_fields(f);
            }
            if (!nl) return;
            cur = size_t(nl - base) + 1;
        } else {
            cur += ls;
        }
    }

    // Scalar tail for final < 64 bytes of file (including EOF without '\n')
    while (cur < limit) {
        const char* line = base + cur;
        const char* nl = find_nl_simd(line, base + n);
        const char* le = nl ? nl : (base + n);
        if (parse_csv_line_scalar<N>(line, le, f)) {
            on_fields(f);
        }
        if (!nl) break;
        cur = size_t(nl - base) + 1;
    }
}

// ============================================================================
// Small File Helper (EINTR-safe pread) + Products / Campaigns Lookups
// ============================================================================
static std::string read_small_file(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return {};
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        return {};
    }
    std::string s(size_t(st.st_size), '\0');
    size_t off = 0;
    while (off < s.size()) {
        ssize_t n = ::pread(fd, s.data() + off, s.size() - off, off_t(off));
        if (n < 0) { if (errno == EINTR) continue; break; }
        if (n == 0) break;
        off += size_t(n);
    }
    ::close(fd);
    return s;
}

static std::string extract_json_string(std::string_view json, std::string_view key) {
    std::string needle = "\"" + std::string(key) + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string_view::npos) return {};
    pos = json.find(':', pos + needle.size());
    if (pos == std::string_view::npos) return {};
    size_t q1 = json.find('"', pos + 1);
    if (q1 == std::string_view::npos) return {};
    size_t q2 = json.find('"', q1 + 1);
    if (q2 == std::string_view::npos) return {};
    return std::string(json.substr(q1 + 1, q2 - q1 - 1));
}

struct FastStringIdMap {
    struct Entry {
        uint64_t hash;
        uint32_t str_off;
        uint32_t str_len;
        uint32_t val;
    };
    std::vector<char> pool;
    std::vector<Entry> table;
    uint32_t mask = 0;
    static constexpr uint32_t NOT_FOUND = UINT32_MAX;

    void init(size_t expected) {
        size_t cap = 64;
        while (cap < expected * 2 + 16) cap <<= 1;
        table.assign(cap, {0, 0, 0, NOT_FOUND});
        mask = uint32_t(cap - 1);
        pool.reserve(expected * 10 + 64);
    }

    void insert(std::string_view s, uint32_t val) {
        uint64_t h = hash_bytes(s.data(), s.size());
        if (h == 0) h = 1;
        uint32_t slot = uint32_t(h) & mask;
        while (table[slot].val != NOT_FOUND) {
            if (table[slot].hash == h && table[slot].str_len == s.size() &&
                std::memcmp(pool.data() + table[slot].str_off, s.data(), s.size()) == 0) {
                table[slot].val = val;
                return;
            }
            slot = (slot + 1) & mask;
        }
        uint32_t off = uint32_t(pool.size());
        pool.insert(pool.end(), s.begin(), s.end());
        table[slot] = {h, off, uint32_t(s.size()), val};
    }

    inline uint32_t find(const char* s, size_t len) const {
        uint64_t h = hash_bytes(s, len);
        if (h == 0) h = 1;
        uint32_t slot = uint32_t(h) & mask;
        while (table[slot].val != NOT_FOUND) {
            if (table[slot].hash == h && table[slot].str_len == len &&
                std::memcmp(pool.data() + table[slot].str_off, s, len) == 0) {
                return table[slot].val;
            }
            slot = (slot + 1) & mask;
        }
        return NOT_FOUND;
    }
};

struct Campaigns {
    std::vector<std::string> ids;
    std::vector<std::string> channels;
    FastStringIdMap map;
    static constexpr uint32_t FAST_CAP = 65536;
    std::vector<uint32_t> fast_c;
    uint32_t organic_idx = 0;

    void load(const std::string& path) {
        std::string data = read_small_file(path);
        std::vector<std::pair<std::string, std::string>> rows;
        if (!data.empty()) {
            const char* nl = find_nl_simd(data.data(), data.data() + data.size());
            if (nl) {
                const char* p = nl + 1;
                const char* end = data.data() + data.size();
                Slice f[2];
                while (p < end) {
                    const char* eol = find_nl_simd(p, end);
                    const char* le = eol ? eol : end;
                    if (parse_csv_line<2>(p, le, f)) {
                        rows.emplace_back(std::string(f[0].ptr, f[0].len), std::string(f[1].ptr, f[1].len));
                    }
                    p = eol ? (eol + 1) : end;
                }
            }
        }
        ids.reserve(rows.size() + 1);
        channels.reserve(rows.size() + 1);
        map.init(rows.size() + 1);
        fast_c.assign(FAST_CAP, FastStringIdMap::NOT_FOUND);
        for (uint32_t i = 0; i < uint32_t(rows.size()); ++i) {
            ids.push_back(std::move(rows[i].first));
            channels.push_back(std::move(rows[i].second));
            map.insert(ids[i], i);
            const std::string& s = ids[i];
            if (s.size() >= 3 && s[0] == 'c' && s[1] == '_' && s[2] >= '1' && s[2] <= '9') {
                uint32_t n = 0;
                bool ok = true;
                for (size_t k = 2; k < s.size(); ++k) {
                    unsigned d = uint8_t(s[k]) - uint8_t('0');
                    if (d > 9 || n > 100000) { ok = false; break; }
                    n = n * 10 + d;
                }
                if (ok && n < FAST_CAP) fast_c[n] = i;
            }
        }
        organic_idx = uint32_t(ids.size());
        ids.emplace_back("organic");
        channels.emplace_back("organic");
    }

    inline uint32_t lookup(const char* s, size_t len) const {
        if (__builtin_expect(len >= 3 && s[0] == 'c' && s[1] == '_' && s[2] >= '1' && s[2] <= '9', 1)) {
            uint32_t n = 0;
            bool ok = true;
            for (size_t k = 2; k < len; ++k) {
                unsigned d = uint8_t(s[k]) - uint8_t('0');
                if (__builtin_expect(d > 9, 0)) { ok = false; break; }
                n = n * 10 + d;
                if (__builtin_expect(n >= FAST_CAP, 0)) { ok = false; break; }
            }
            if (__builtin_expect(ok, 1)) return fast_c[n];
        }
        return map.find(s, len);
    }
};

struct Products {
    std::vector<std::string> cat_names;
    uint32_t unknown_cat_id = 0;
    size_t max_cat_len = 8;
    FastStringIdMap map;
    static constexpr uint32_t FAST_CAP = 131072;
    std::vector<uint32_t> fast_p;

    void load(const std::string& path) {
        std::string data = read_small_file(path);
        std::vector<std::pair<std::string_view, std::string_view>> rows;
        rows.reserve(52000);
        if (!data.empty()) {
            const char* nl = find_nl_simd(data.data(), data.data() + data.size());
            if (nl) {
                const char* p = nl + 1;
                const char* end = data.data() + data.size();
                Slice f[2];
                while (p < end) {
                    const char* eol = find_nl_simd(p, end);
                    const char* le = eol ? eol : end;
                    if (parse_csv_line<2>(p, le, f)) {
                        rows.push_back({{f[0].ptr, f[0].len}, {f[1].ptr, f[1].len}});
                    }
                    p = eol ? (eol + 1) : end;
                }
            }
        }
        std::vector<std::string_view> cats;
        cats.reserve(64);
        cats.push_back("unknown");
        for (const auto& r : rows) {
            bool found = false;
            for (auto c : cats) { if (c == r.second) { found = true; break; } }
            if (!found) cats.push_back(r.second);
        }
        std::sort(cats.begin(), cats.end());
        cat_names.reserve(cats.size());
        max_cat_len = 8;
        for (auto c : cats) {
            if (c.size() > max_cat_len) max_cat_len = c.size();
            cat_names.emplace_back(c);
        }

        FastStringIdMap cat_map;
        cat_map.init(cat_names.size());
        for (uint32_t i = 0; i < uint32_t(cat_names.size()); ++i) {
            cat_map.insert(cat_names[i], i);
            if (cat_names[i] == "unknown") unknown_cat_id = i;
        }

        fast_p.assign(FAST_CAP, unknown_cat_id);
        size_t non_fast = 0;
        std::vector<std::pair<std::string_view, uint32_t>> slow_rows;
        for (const auto& r : rows) {
            uint32_t cid = cat_map.find(r.second.data(), r.second.size());
            std::string_view s = r.first;
            bool is_fast = false;
            if (s.size() >= 2 && s[0] == 'p' && s[1] >= '1' && s[1] <= '9') {
                uint32_t n = 0;
                bool ok = true;
                for (size_t k = 1; k < s.size(); ++k) {
                    unsigned d = uint8_t(s[k]) - uint8_t('0');
                    if (d > 9 || n > 200000) { ok = false; break; }
                    n = n * 10 + d;
                }
                if (ok && n < FAST_CAP) { fast_p[n] = cid; is_fast = true; }
            }
            if (!is_fast) { ++non_fast; slow_rows.push_back({s, cid}); }
        }
        map.init(non_fast + 1);
        for (const auto& sr : slow_rows) map.insert(sr.first, sr.second);
    }

    inline uint32_t lookup(const char* s, size_t len) const {
        if (__builtin_expect(len >= 2 && s[0] == 'p' && s[1] >= '1' && s[1] <= '9', 1)) {
            uint32_t n = 0;
            bool ok = true;
            for (size_t k = 1; k < len; ++k) {
                unsigned d = uint8_t(s[k]) - uint8_t('0');
                if (__builtin_expect(d > 9, 0)) { ok = false; break; }
                n = n * 10 + d;
                if (__builtin_expect(n >= FAST_CAP, 0)) { ok = false; break; }
            }
            if (__builtin_expect(ok, 1)) return fast_p[n];
        }
        uint32_t r = map.find(s, len);
        return r == FastStringIdMap::NOT_FOUND ? unknown_cat_id : r;
    }
};

// ============================================================================
// Stage 1: 4-Barrier Lock-Free Direct Hash-DSU + 1-Word 3-Bit Blocked Bloom!
// ============================================================================
struct LinkMap {
    struct Slot {
        uint64_t key;
        uint64_t val;
    };
    Slot* slots = nullptr;
    size_t cap = 0;
    size_t mask = 0;

    // 1-word (64-bit register-blocked) 3-bit Bloom filter -> 1 cacheline, 1 prefetch, ~0.5% FP!
    uint64_t* bloom = nullptr;
    size_t bloom_word_mask = 0;
    bool has_fallback = false;

    static inline uint64_t bloom_mask3(uint64_t h) {
        return (1ULL << (h & 63U)) | (1ULL << ((h >> 6) & 63U)) | (1ULL << ((h >> 12) & 63U));
    }

    inline bool bloom_might_contain(uint64_t h) const {
        uint64_t word = bloom[(h >> 18) & bloom_word_mask];
        uint64_t m = bloom_mask3(h);
        return (word & m) == m;
    }

    inline void prefetch_bloom(uint64_t h) const {
        if (__builtin_expect(bloom != nullptr, 1)) {
            __builtin_prefetch(&bloom[(h >> 18) & bloom_word_mask], 0, 1);
        }
    }

    inline void prefetch_slot(uint64_t h) const {
        if (__builtin_expect(slots != nullptr, 1)) {
            __builtin_prefetch(&slots[size_t(h) & mask], 0, 1);
        }
    }

    inline uint64_t resolve_slot_only(uint64_t k, uint64_t h, const StringArenaPool& arenas) const {
        size_t idx = size_t(h) & mask;
        while (true) {
            uint64_t cur = slots[idx].key;
            if (cur == 0) return k;
            if (key_eq(cur, k, arenas)) return slots[idx].val;
            idx = (idx + 1) & mask;
        }
    }

    inline uint64_t resolve_with_hash(uint64_t k, uint64_t h, const StringArenaPool& arenas) const {
        if (__builtin_expect(cap == 0, 0)) return k;
        if (__builtin_expect(!has_fallback && (k & FALLBACK_BIT) != 0, 0)) return k;
        if (__builtin_expect(bloom != nullptr && !bloom_might_contain(h), 1)) return k;
        return resolve_slot_only(k, h, arenas);
    }

    void build(const std::vector<MmapTask>& tasks, ThreadPool& pool, StringArenaPool& arenas) {
        if (tasks.empty()) return;

        struct Edge { uint64_t a, b; };
        std::vector<std::vector<Edge>> thread_edges(static_cast<size_t>(pool.nthreads));
        std::atomic<size_t> next_t{0};
        std::atomic<bool> any_fallback{false};

        // Pass 1: Parallel zero-copy mmap SIMD scan of links/*.csv
        pool.run([&](int tid) {
            auto& edges = thread_edges[size_t(tid)];
            edges.reserve(65536);
            bool local_fb = false;
            while (true) {
                size_t ti = next_t.fetch_add(1, std::memory_order_relaxed);
                if (ti >= tasks.size()) break;
                scan_mmap_csv_simd<2>(tasks[ti], [&](const Slice f[2]) {
                    if (__builtin_expect(f[0].len > 0 && f[1].len > 0, 1)) {
                        uint64_t ka = encode_key64(f[0].ptr, f[0].len, arenas, tid);
                        uint64_t kb = encode_key64(f[1].ptr, f[1].len, arenas, tid);
                        if (__builtin_expect(ka == kb, 0)) return;
                        if (__builtin_expect(((ka | kb) & FALLBACK_BIT) != 0, 0)) local_fb = true;
                        edges.push_back({ka, kb});
                    }
                });
            }
            if (local_fb) any_fallback.store(true, std::memory_order_relaxed);
        });

        size_t total_edges = 0;
        for (int t = 0; t < pool.nthreads; ++t) total_edges += thread_edges[size_t(t)].size();
        if (total_edges == 0) return;

        has_fallback = any_fallback.load(std::memory_order_relaxed);

        if (!has_fallback && total_edges >= 2048) {
            // Direct lock-free flat Hash-DSU: parent == 0 means root, parent == r + 1 links to r < s!
            // inv_min[r] stores ~min_key (initialized to 0 by mmap -> ZERO init pass!).
            struct DsuSlot {
                uint64_t key;
                uint32_t parent;
                uint32_t _pad;
            };
            size_t hcap = 64;
            while (hcap < total_edges * 4) hcap <<= 1;
            size_t hmask = hcap - 1;
            auto* dsu = static_cast<DsuSlot*>(alloc_mmap(hcap * sizeof(DsuSlot)));
            auto* inv_min = static_cast<uint64_t*>(alloc_mmap(hcap * sizeof(uint64_t)));

            auto insert_key = [&](uint64_t k) -> uint32_t {
                size_t s = size_t(mix64(k)) & hmask;
                while (true) {
                    std::atomic_ref<uint64_t> ak(dsu[s].key);
                    uint64_t cur = ak.load(std::memory_order_relaxed);
                    if (cur == k) return uint32_t(s);
                    if (cur == 0) {
                        uint64_t exp = 0;
                        if (ak.compare_exchange_strong(exp, k, std::memory_order_relaxed) || exp == k) {
                            return uint32_t(s);
                        }
                    }
                    s = (s + 1) & hmask;
                }
            };
            auto find_root = [&](uint32_t x) -> uint32_t {
                while (true) {
                    std::atomic_ref<uint32_t> ap(dsu[x].parent);
                    uint32_t p1 = ap.load(std::memory_order_relaxed);
                    if (p1 == 0) return x;
                    uint32_t p = p1 - 1;
                    uint32_t gp1 = std::atomic_ref<uint32_t>(dsu[p].parent).load(std::memory_order_relaxed);
                    if (gp1 != 0) {
                        ap.compare_exchange_weak(p1, gp1, std::memory_order_relaxed);
                        x = gp1 - 1;
                    } else {
                        return p;
                    }
                }
            };

            // Pass 2: Parallel fused Hash-Insert + Lock-Free Union!
            pool.run([&](int tid) {
                const auto& ev = thread_edges[size_t(tid)];
                for (const auto& e : ev) {
                    uint32_t u = insert_key(e.a);
                    uint32_t v = insert_key(e.b);
                    while (u != v) {
                        u = find_root(u);
                        v = find_root(v);
                        if (u == v) break;
                        if (u < v) std::swap(u, v); // always link larger slot -> smaller slot!
                        uint32_t exp = 0;
                        if (std::atomic_ref<uint32_t>(dsu[u].parent).compare_exchange_weak(
                                exp, v + 1, std::memory_order_relaxed)) {
                            break;
                        }
                    }
                }
            });

            // Pass 3: Parallel path compression + component min_key (~max) + exact non-minimal count!
            std::vector<size_t> t_nonmin(static_cast<size_t>(pool.nthreads), 0);
            pool.run([&](int tid) {
                size_t s = (hcap * size_t(tid)) / size_t(pool.nthreads);
                size_t e = (hcap * size_t(tid + 1)) / size_t(pool.nthreads);
                size_t c = 0;
                for (size_t i = s; i < e; ++i) {
                    uint64_t k = dsu[i].key;
                    if (k == 0) continue;
                    uint32_t r = uint32_t(i);
                    while (true) {
                        uint32_t p1 = std::atomic_ref<uint32_t>(dsu[r].parent).load(std::memory_order_relaxed);
                        if (p1 == 0) break;
                        r = p1 - 1;
                    }
                    if (r != uint32_t(i)) {
                        std::atomic_ref<uint32_t>(dsu[i].parent).store(r + 1, std::memory_order_relaxed);
                        ++c;
                    }
                    uint64_t inv_k = ~k; // k has bit 63 == 0 -> inv_k >= 0x8000... > 0!
                    std::atomic_ref<uint64_t> am(inv_min[r]);
                    uint64_t cur = am.load(std::memory_order_relaxed);
                    while (inv_k > cur && !am.compare_exchange_weak(cur, inv_k, std::memory_order_relaxed)) {}
                }
                t_nonmin[size_t(tid)] = c;
            });

            size_t total_nonmin = 0;
            for (size_t c : t_nonmin) total_nonmin += c;
            if (total_nonmin == 0) return;

            cap = 64;
            while (cap < total_nonmin * 2) cap <<= 1;
            mask = cap - 1;
            slots = static_cast<Slot*>(alloc_mmap(cap * sizeof(Slot)));

            // 1-word 3-bit blocked Bloom: ~16 bits/key (2x total_nonmin bytes / 8)
            size_t bwords = 4096;
            while (bwords * 4 < total_nonmin && bwords < (1ULL << 20)) bwords <<= 1;
            bloom_word_mask = bwords - 1;
            bloom = static_cast<uint64_t*>(alloc_mmap(bwords * sizeof(uint64_t)));

            // Pass 4: Populate compact non-minimal-only slots + 1-word 3-bit Bloom!
            pool.run([&](int tid) {
                size_t s = (hcap * size_t(tid)) / size_t(pool.nthreads);
                size_t e = (hcap * size_t(tid + 1)) / size_t(pool.nthreads);
                for (size_t i = s; i < e; ++i) {
                    uint64_t k = dsu[i].key;
                    if (k == 0) continue;
                    uint32_t p1 = dsu[i].parent;
                    uint32_t r = (p1 == 0) ? uint32_t(i) : (p1 - 1);
                    uint64_t mk = ~inv_min[r];
                    if (k == mk) continue;
                    uint64_t h = mix64(k);
                    std::atomic_ref<uint64_t>(bloom[(h >> 18) & bloom_word_mask]).fetch_or(
                        bloom_mask3(h), std::memory_order_relaxed);
                    size_t idx = size_t(h) & mask;
                    while (true) {
                        uint64_t exp = 0;
                        if (std::atomic_ref<uint64_t>(slots[idx].key).compare_exchange_strong(
                                exp, k, std::memory_order_relaxed)) {
                            slots[idx].val = mk;
                            break;
                        }
                        idx = (idx + 1) & mask;
                    }
                }
            });
        } else {
            cap = 64;
            while (cap < total_edges * 4) cap <<= 1;
            mask = cap - 1;
            slots = static_cast<Slot*>(alloc_mmap(cap * sizeof(Slot)));

            std::vector<uint64_t> node_key, min_key;
            std::vector<uint32_t> parent;
            node_key.reserve(total_edges * 2);
            parent.reserve(total_edges * 2);
            min_key.reserve(total_edges * 2);

            auto get_or_insert = [&](uint64_t k) -> uint32_t {
                size_t idx = size_t(hash_key64(k, arenas)) & mask;
                while (true) {
                    uint64_t cur = slots[idx].key;
                    if (cur == 0) {
                        uint32_t nid = uint32_t(node_key.size());
                        node_key.push_back(k);
                        parent.push_back(nid);
                        min_key.push_back(k);
                        slots[idx].key = k;
                        slots[idx].val = uint64_t(nid) + 1;
                        return nid;
                    }
                    if (key_eq(cur, k, arenas)) return uint32_t(slots[idx].val - 1);
                    idx = (idx + 1) & mask;
                }
            };
            auto find_root = [&](uint32_t x) -> uint32_t {
                uint32_t r = x;
                while (parent[r] != r) r = parent[r];
                while (parent[x] != r) { uint32_t nxt = parent[x]; parent[x] = r; x = nxt; }
                return r;
            };
            for (const auto& vec : thread_edges) {
                for (const auto& e : vec) {
                    uint32_t u = find_root(get_or_insert(e.a));
                    uint32_t v = find_root(get_or_insert(e.b));
                    if (u != v) {
                        if (u > v) std::swap(u, v);
                        parent[v] = u;
                        if (key_less(min_key[v], min_key[u], arenas)) min_key[u] = min_key[v];
                    }
                }
            }
            for (size_t i = 0; i < cap; ++i) {
                if (slots[i].val != 0) {
                    uint32_t r = find_root(uint32_t(slots[i].val - 1));
                    slots[i].val = min_key[r];
                }
            }
        }
    }
};

// ============================================================================
// Compact Record Structs (32b ts_off + 30b ingest_rel + Exact ms SideEvent!)
// ============================================================================
static constexpr int EV_BUCKETS = 256;     // 256b x WC_CAP=8 x 24B = 48KB, 192B = 3x64B EXACT NT store!
static constexpr int PERSON_BUCKETS = 512; // 512b x WC_CAP=4 x 16B = 32KB L1D-hot, 64B EXACT NT store!

struct SideEvent {
    int64_t ts_ms;
    int64_t ingest_ms;
    uint32_t price;
    uint32_t cat_id;
};

struct RawEvent {
    uint64_t event_id;
    uint64_t user_id;
    uint64_t meta;

    // meta layout:
    //   bit 63:     valid
    //   bit 62:     has_side
    //   bits 61..32 (30b): ingest_rel (whole-second offset + REL_BIAS, 0 = read exact ingest_ms from SideEvent!)
    //   bits 31..0  (32b): if !has_side: ts_off in ms (covers 49.7 DAYS inline!)
    //                      if has_side:  (tid << 26) | side_idx (67M/thread!)
    static constexpr int64_t REL_BIAS = 1LL << 29;
    static constexpr int64_t REL_MAX  = (1LL << 30) - 1;

    inline bool is_valid() const { return (meta >> 63) != 0; }
    inline bool has_side() const { return ((meta >> 62) & 1ULL) != 0; }
    inline uint32_t ingest_rel() const { return uint32_t((meta >> 32) & 0x3FFFFFFFULL); }
    inline uint32_t low32() const { return uint32_t(meta & 0xFFFFFFFFULL); }

    static inline uint64_t pack_inline(bool valid, uint32_t ingest_rel30, uint32_t ts_off32) {
        return (uint64_t(valid) << 63) | (uint64_t(ingest_rel30) << 32) | uint64_t(ts_off32);
    }
    static inline uint64_t pack_side(bool valid, uint32_t ingest_rel30, int tid, uint32_t side_idx) {
        return (uint64_t(valid) << 63) | (1ULL << 62) | (uint64_t(ingest_rel30) << 32) |
               (uint64_t(tid) << 26) | uint64_t(side_idx);
    }
};

static inline int64_t get_exact_ingest_ms(
    const RawEvent& ev,
    int64_t win_start_sec,
    const std::vector<std::vector<SideEvent>>& sides
) {
    uint32_t ir = ev.ingest_rel();
    if (__builtin_expect(ir != 0, 1)) return (win_start_sec + int64_t(ir) - RawEvent::REL_BIAS) * 1000LL;
    uint32_t lo = ev.low32();
    return sides[lo >> 26][lo & 0x03FFFFFFU].ingest_ms;
}

static inline bool ingest_strictly_earlier(
    const RawEvent& a,
    const RawEvent& b,
    int64_t win_start_sec,
    const std::vector<std::vector<SideEvent>>& sides
) {
    uint32_t ia = a.ingest_rel();
    uint32_t ib = b.ingest_rel();
    if (__builtin_expect(ia != 0 && ib != 0, 1)) return ia < ib;
    return get_exact_ingest_ms(a, win_start_sec, sides) < get_exact_ingest_ms(b, win_start_sec, sides);
}

struct CompactPersonEvent {
    uint64_t person_key;
    uint64_t packed; // [32b: ts_off] [24b: price_cents] [8b: cat_id]
};

struct WidePersonEvent {
    uint64_t person_key;
    int64_t ts_ms;
    uint32_t price;
    uint32_t cat_id;
};

struct PersonClick {
    uint64_t person_key;
    uint64_t click_id;
    uint64_t ts_camp; // [44b: ts - click_start] [20b: camp_idx]
};

// Packed 16-byte LocalClick (down from 24B -> 1.5x faster scatter & sort!)
struct LocalClick {
    uint64_t ts_camp; // [44b: ts_off] [20b: camp_idx]
    uint64_t click_id;
    inline uint64_t ts_off() const { return ts_camp >> 20; }
    inline uint32_t camp_idx() const { return uint32_t(ts_camp & 0xFFFFFULL); }
};

struct CampCell {
    uint32_t sessions;
    uint32_t purchasing_sessions;
    uint32_t assisted_sessions;
    uint32_t _pad;
    uint64_t revenue_cents;
};

// Fused per-campaign state (1 cache line instead of 3 separate arrays!)
struct CampState {
    uint32_t count;
    uint32_t enter_sess;
    uint32_t active_pos;
};

// ============================================================================
// Bounds-Safe Slab-Recycled Growable UserDailyBuffers (0 Faults + 0 Overflow!)
// ============================================================================
struct UserDailyBuffers {
    uint8_t* uid_plain = nullptr;
    size_t uid_sz = 0, uid_cap = 0;
    bool uid_owned = false;

    int32_t* day_vals = nullptr;
    int32_t* sess_vals = nullptr;
    int32_t* ev_vals = nullptr;
    int64_t* rev_vals = nullptr;
    uint8_t* top_cat_bitmap = nullptr;
    size_t num_rows = 0, row_cap = 0;
    bool rows_owned = false;

    uint8_t* top_cat_plain = nullptr;
    size_t top_cat_sz = 0, top_cat_cap = 0;
    bool cat_owned = false;

    void init(size_t est_rows, size_t max_cat_len, const std::vector<void*>& dead_slabs, size_t slab_bytes) {
        // Try to carve initial buffers directly out of a dead already-faulted 16 MB event_store slab!
        // Layout inside one 16 MB slab (up to 480K rows -> covers 100% of 5M & 20M threads with ZERO page faults!):
        //   day(4) + sess(4) + ev(4) + rev(8) + bm(1/8) = 20.125 B/row
        //   uid_plain ~ 14 B/row, top_cat ~ 2 B/row -> ~36 B/row -> 400K rows = 14.4 MB < 16 MB!
        if (!dead_slabs.empty() && slab_bytes >= (16ULL << 20)) {
            char* s = static_cast<char*>(dead_slabs[0]);
            size_t rcap = 393216; // 384K rows (fits 5M & 20M/8 threads with room to spare!)
            day_vals       = reinterpret_cast<int32_t*>(s);                 s += rcap * 4;
            sess_vals      = reinterpret_cast<int32_t*>(s);                 s += rcap * 4;
            ev_vals        = reinterpret_cast<int32_t*>(s);                 s += rcap * 4;
            rev_vals       = reinterpret_cast<int64_t*>(s);                 s += rcap * 8;
            size_t bm_sz   = ((rcap >> 3) + 64) & ~size_t(63);
            top_cat_bitmap = reinterpret_cast<uint8_t*>(s);                 s += bm_sz;
            std::memset(top_cat_bitmap, 0, bm_sz);
            size_t rem     = slab_bytes - size_t(s - static_cast<char*>(dead_slabs[0]));
            uid_cap        = (rem * 3) / 4;
            top_cat_cap    = rem - uid_cap;
            uid_plain      = reinterpret_cast<uint8_t*>(s);
            top_cat_plain  = reinterpret_cast<uint8_t*>(s + uid_cap);
            row_cap        = rcap;
            rows_owned = uid_owned = cat_owned = false;
            return;
        }
        size_t rcap = std::max<size_t>(est_rows * 2, 131072);
        row_cap = rcap;
        uid_cap = rcap * 20 + 4096;
        top_cat_cap = rcap * (4 + max_cat_len) + 4096;
        day_vals = static_cast<int32_t*>(alloc_mmap(rcap * 4));
        sess_vals = static_cast<int32_t*>(alloc_mmap(rcap * 4));
        ev_vals = static_cast<int32_t*>(alloc_mmap(rcap * 4));
        rev_vals = static_cast<int64_t*>(alloc_mmap(rcap * 8));
        top_cat_bitmap = static_cast<uint8_t*>(alloc_mmap((rcap >> 3) + 64));
        uid_plain = static_cast<uint8_t*>(alloc_mmap(uid_cap));
        top_cat_plain = static_cast<uint8_t*>(alloc_mmap(top_cat_cap));
        rows_owned = uid_owned = cat_owned = true;
    }

    __attribute__((noinline)) void grow_rows() {
        size_t new_cap = row_cap * 2;
        auto* nd = static_cast<int32_t*>(alloc_mmap(new_cap * 4));
        auto* ns = static_cast<int32_t*>(alloc_mmap(new_cap * 4));
        auto* ne = static_cast<int32_t*>(alloc_mmap(new_cap * 4));
        auto* nr = static_cast<int64_t*>(alloc_mmap(new_cap * 8));
        auto* nb = static_cast<uint8_t*>(alloc_mmap((new_cap >> 3) + 64));
        std::memcpy(nd, day_vals, num_rows * 4);
        std::memcpy(ns, sess_vals, num_rows * 4);
        std::memcpy(ne, ev_vals, num_rows * 4);
        std::memcpy(nr, rev_vals, num_rows * 8);
        std::memcpy(nb, top_cat_bitmap, (num_rows + 7) >> 3);
        day_vals = nd; sess_vals = ns; ev_vals = ne; rev_vals = nr; top_cat_bitmap = nb;
        row_cap = new_cap;
        rows_owned = true;
    }

    __attribute__((noinline)) void grow_uid(size_t need) {
        size_t new_cap = std::max(uid_cap * 2, uid_sz + need + 65536);
        auto* np = static_cast<uint8_t*>(alloc_mmap(new_cap));
        if (uid_sz) std::memcpy(np, uid_plain, uid_sz);
        uid_plain = np;
        uid_cap = new_cap;
        uid_owned = true;
    }

    __attribute__((noinline)) void grow_cat(size_t need) {
        size_t new_cap = std::max(top_cat_cap * 2, top_cat_sz + need + 65536);
        auto* np = static_cast<uint8_t*>(alloc_mmap(new_cap));
        if (top_cat_sz) std::memcpy(np, top_cat_plain, top_cat_sz);
        top_cat_plain = np;
        top_cat_cap = new_cap;
        cat_owned = true;
    }

    inline void push_row_key(
        uint64_t person_key,
        const StringArenaPool& arenas,
        int32_t day,
        int32_t sessions,
        int32_t events,
        int64_t rev,
        int32_t best_cat,
        const std::vector<std::string>& cat_names
    ) {
        if (__builtin_expect(num_rows >= row_cap, 0)) grow_rows();
        if (__builtin_expect((person_key & FALLBACK_BIT) == 0, 1)) {
            if (__builtin_expect(uid_sz + 16 > uid_cap, 0)) grow_uid(16);
            uint8_t* dst = uid_plain + uid_sz;
            uint32_t len = uint32_t(decode_inline(person_key, reinterpret_cast<char*>(dst + 4)));
            std::memcpy(dst, &len, 4);
            uid_sz += 4 + len;
        } else {
            auto sv = arenas.get_fallback(person_key);
            uint32_t len = uint32_t(sv.size());
            if (__builtin_expect(uid_sz + 4 + len > uid_cap, 0)) grow_uid(4 + len);
            uint8_t* dst = uid_plain + uid_sz;
            std::memcpy(dst, &len, 4);
            if (len) std::memcpy(dst + 4, sv.data(), len);
            uid_sz += 4 + len;
        }
        size_t r = num_rows++;
        day_vals[r] = day;
        sess_vals[r] = sessions;
        ev_vals[r] = events;
        rev_vals[r] = rev;
        if (best_cat >= 0) {
            top_cat_bitmap[r >> 3] |= uint8_t(1U << (r & 7));
            const std::string& cs = cat_names[size_t(best_cat)];
            uint32_t len = uint32_t(cs.size());
            if (__builtin_expect(top_cat_sz + 4 + len > top_cat_cap, 0)) grow_cat(4 + len);
            uint8_t* dst = top_cat_plain + top_cat_sz;
            std::memcpy(dst, &len, 4);
            std::memcpy(dst + 4, cs.data(), len);
            top_cat_sz += 4 + len;
        }
    }
};

// Fast L1-hot 256-bucket (uint16_t) LSD radix sort on 32-bit ts_off for en in [40, 65535].
// Skips uniform bytes automatically (0 work if high bytes are identical!), ZERO branch mispredicts!
static inline void radix_sort_256_u64(uint64_t* arr, uint32_t n, std::vector<uint64_t>& tmp) {
    tmp.resize(n);
    uint16_t c0[256]{}, c1[256]{}, c2[256]{}, c3[256]{};
    uint32_t first = uint32_t(arr[0] >> 32);
    uint32_t diff = 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t k = uint32_t(arr[i] >> 32);
        diff |= (k ^ first);
        c0[k & 255U]++;
        c1[(k >> 8) & 255U]++;
        c2[(k >> 16) & 255U]++;
        c3[k >> 24]++;
    }
    if (diff == 0) return;
    uint64_t* src = arr;
    uint64_t* dst = tmp.data();
    auto pass = [&](uint16_t* cnt, uint32_t shift) {
        uint16_t acc = 0;
        for (int i = 0; i < 256; ++i) { uint16_t t = cnt[i]; cnt[i] = acc; acc = uint16_t(acc + t); }
        for (uint32_t i = 0; i < n; ++i) {
            uint64_t v = src[i];
            dst[cnt[(uint32_t(v >> 32) >> shift) & 255U]++] = v;
        }
        std::swap(src, dst);
    };
    if (diff & 0x000000FFU) pass(c0, 0);
    if (diff & 0x0000FF00U) pass(c1, 8);
    if (diff & 0x00FF0000U) pass(c2, 16);
    if (diff & 0xFF000000U) pass(c3, 24);
    if (src != arr) std::memcpy(arr, src, size_t(n) * sizeof(uint64_t));
}

// Full 3-pass LSD radix sort on all 32 bits of ts_off (used only for rare en >= 65536 whales/kiosks!)
static inline void radix_sort_ts_u64(uint64_t* arr, uint32_t n, std::vector<uint64_t>& tmp) {
    tmp.resize(n);
    uint32_t c0[2048]{}, c1[2048]{}, c2[1024]{};
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t k = uint32_t(arr[i] >> 32);
        c0[k & 2047U]++;
        c1[(k >> 11) & 2047U]++;
        c2[k >> 22]++;
    }
    uint32_t s0 = 0, s1 = 0, s2 = 0;
    for (int i = 0; i < 2048; ++i) { uint32_t t = c0[i]; c0[i] = s0; s0 += t; }
    for (int i = 0; i < 2048; ++i) { uint32_t t = c1[i]; c1[i] = s1; s1 += t; }
    for (int i = 0; i < 1024; ++i) { uint32_t t = c2[i]; c2[i] = s2; s2 += t; }

    uint64_t* dst = tmp.data();
    for (uint32_t i = 0; i < n; ++i) { uint64_t v = arr[i]; dst[c0[uint32_t(v >> 32) & 2047U]++] = v; }
    for (uint32_t i = 0; i < n; ++i) { uint64_t v = dst[i]; arr[c1[uint32_t(v >> 43) & 2047U]++] = v; }
    for (uint32_t i = 0; i < n; ++i) { uint64_t v = arr[i]; dst[c2[uint32_t(v >> 54)]++] = v; }
    std::memcpy(arr, dst, size_t(n) * sizeof(uint64_t));
}

static void mkdir_p(const std::string& path) {
    size_t pos = 0;
    while ((pos = path.find('/', pos + 1)) != std::string::npos) {
        ::mkdir(path.substr(0, pos).c_str(), 0755);
    }
    ::mkdir(path.c_str(), 0755);
}

// ============================================================================
// Stage 2C + Stage 3 + Stage 4 FUSED: Dedup -> Coarse-Radix Sessionize -> writev()
// ============================================================================
template <bool Compact>
bool run_dedup_sessionize_and_write(
    PartitionedStore<RawEvent, 256, EV_BUCKETS, 8, true>& event_store,
    const std::vector<std::vector<SideEvent>>& sides,
    PartitionedStore<PersonClick, 128, PERSON_BUCKETS, 4, false>& click_store,
    const LinkMap& links,
    const Products& products,
    const Campaigns& campaigns,
    const StringArenaPool& arenas,
    ThreadPool& pool,
    int64_t win_start_ms,
    int64_t win_end_ms,
    int64_t win_start_sec,
    int64_t click_start_ms,
    int32_t win_start_day,
    int32_t num_days,
    const std::string& ud_dir,
    const std::string& cd_dir,
    bool prof
) {
    using PEv = std::conditional_t<Compact, CompactPersonEvent, WidePersonEvent>;
    // WC_CAP = 4 for CompactPersonEvent (16B) -> 64B EXACT NT streaming store + 32 KB L1D fit!
    PartitionedStore<PEv, 256, PERSON_BUCKETS, 4, Compact> person_ev_store;
    person_ev_store.init(pool.nthreads);

    auto now_ms = []() {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    double td0 = now_ms();

    // Stage 2C: IN-PLACE L2 Fingerprinted Dedup with 8-Ahead Single-Hash PREFETCHW + 1-Word Bloom Ring!
    using EvBlock = typename PartitionedStore<RawEvent, 256, EV_BUCKETS, 8, true>::Block;
    std::atomic<int> next_eb{0};
    pool.run([&](int tid) {
        struct DedupSlot8 { uint32_t fp; uint32_t handle1; };
        std::vector<uint32_t> ht4;
        std::vector<DedupSlot8> ht8;
        std::vector<EvBlock*> blks;
        ht4.reserve(262144);
        blks.reserve(512);

        auto& out_ts = *person_ev_store.ts[size_t(tid)];
        const bool links_active = (links.cap > 0);
        const bool links_has_bloom = (links.bloom != nullptr);

        struct RingItem {
            const RawEvent* ev;
            uint64_t uh;
            bool probe_ht;
        };
        constexpr size_t RING_SZ = 64;
        constexpr size_t RING_MASK = RING_SZ - 1;
        RingItem ring[RING_SZ];

        auto emit_resolved = [&](const RawEvent& ev, uint64_t uh, bool probe_ht) {
            uint64_t pkey = probe_ht ? links.resolve_slot_only(ev.user_id, uh, arenas) : ev.user_id;
            uint64_t ph = (pkey == ev.user_id) ? uh : hash_key64(pkey, arenas);
            uint32_t pb = uint32_t(ph >> 55);
            if constexpr (Compact) {
                uint64_t packed;
                if (__builtin_expect(!ev.has_side(), 1)) {
                    packed = uint64_t(ev.low32()) << 32;
                } else {
                    uint32_t lo = ev.low32();
                    const SideEvent& se = sides[lo >> 26][lo & 0x03FFFFFFU];
                    packed = (uint64_t(se.ts_ms - win_start_ms) << 32) | (uint64_t(se.price) << 8) | uint64_t(se.cat_id);
                }
                out_ts.push(pb, CompactPersonEvent{pkey, packed});
            } else {
                if (!ev.has_side()) {
                    out_ts.push(pb, WidePersonEvent{pkey, win_start_ms + int64_t(ev.low32()), 0, 0});
                } else {
                    uint32_t lo = ev.low32();
                    const SideEvent& se = sides[lo >> 26][lo & 0x03FFFFFFU];
                    out_ts.push(pb, WidePersonEvent{pkey, se.ts_ms, se.price, se.cat_id});
                }
            }
        };

        while (true) {
            int b = next_eb.fetch_add(1, std::memory_order_relaxed);
            if (b >= EV_BUCKETS) break;
            size_t nb = event_store.bucket_count(b);
            if (nb == 0) continue;

            blks.clear();
            for (int st = 0; st < pool.nthreads; ++st) {
                for (auto* blk = event_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                    blks.push_back(blk);
                }
            }

            size_t hcap = 64;
            while (hcap < nb + (nb >> 1) + 16) hcap <<= 1;
            size_t hmask = hcap - 1;
            const uint32_t nblks = uint32_t(blks.size());

            if (__builtin_expect(nblks < 4095, 1)) {
                ht4.assign(hcap, 0);
                uint32_t* htp = ht4.data();
                uint64_t hring[8];
                for (uint32_t bi = 0; bi < nblks; ++bi) {
                    EvBlock* blk = blks[bi];
                    const uint32_t cnt = blk->count;
                    RawEvent* items = blk->items;
                    const uint32_t pre = cnt < 8 ? cnt : 8;
                    for (uint32_t p = 0; p < pre; ++p) {
                        uint64_t hp = hash_key64(items[p].event_id, arenas);
                        hring[p] = hp;
                        __builtin_prefetch(&htp[size_t(hp >> 12) & hmask], 1, 1);
                    }
                    for (uint32_t i = 0; i < cnt; ++i) {
                        uint64_t h = hring[i & 7U];
                        if (__builtin_expect(i + 8 < cnt, 1)) {
                            uint64_t hp = hash_key64(items[i + 8].event_id, arenas);
                            hring[(i + 8) & 7U] = hp;
                            __builtin_prefetch(&htp[size_t(hp >> 12) & hmask], 1, 1);
                        }
                        RawEvent& ev = items[i];
                        uint32_t fp12 = uint32_t(h & 0xFFFULL) << 20;
                        size_t slot = size_t(h >> 12) & hmask;
                        while (true) {
                            uint32_t entry = htp[slot];
                            if (__builtin_expect(entry == 0, 1)) {
                                htp[slot] = fp12 | (((bi << 8) | i) + 1);
                                break;
                            }
                            if (__builtin_expect((entry & 0xFFF00000U) == fp12, 0)) {
                                uint32_t prev = (entry & 0x000FFFFFU) - 1;
                                RawEvent& existing = blks[prev >> 8]->items[prev & 255U];
                                if (key_eq(existing.event_id, ev.event_id, arenas)) {
                                    if (ingest_strictly_earlier(ev, existing, win_start_sec, sides)) {
                                        existing = ev;
                                    }
                                    ev.meta = 0;
                                    break;
                                }
                            }
                            slot = (slot + 1) & hmask;
                        }
                    }
                }
            } else {
                ht8.assign(hcap, {0, 0});
                for (uint32_t bi = 0; bi < nblks; ++bi) {
                    EvBlock* blk = blks[bi];
                    const uint32_t cnt = blk->count;
                    RawEvent* items = blk->items;
                    for (uint32_t i = 0; i < cnt; ++i) {
                        RawEvent& ev = items[i];
                        uint64_t h = hash_key64(ev.event_id, arenas);
                        uint32_t fp = uint32_t(h);
                        size_t slot = size_t(h >> 16) & hmask;
                        while (true) {
                            uint32_t h1 = ht8[slot].handle1;
                            if (__builtin_expect(h1 == 0, 1)) {
                                ht8[slot] = {fp, ((bi << 8) | i) + 1};
                                break;
                            }
                            if (__builtin_expect(ht8[slot].fp == fp, 0)) {
                                uint32_t prev = h1 - 1;
                                RawEvent& existing = blks[prev >> 8]->items[prev & 255U];
                                if (key_eq(existing.event_id, ev.event_id, arenas)) {
                                    if (ingest_strictly_earlier(ev, existing, win_start_sec, sides)) {
                                        existing = ev;
                                    }
                                    ev.meta = 0;
                                    break;
                                }
                            }
                            slot = (slot + 1) & hmask;
                        }
                    }
                }
            }

            if (__builtin_expect(!links_active, 0)) {
                for (uint32_t bi = 0; bi < nblks; ++bi) {
                    const EvBlock* blk = blks[bi];
                    const uint32_t cnt = blk->count;
                    const RawEvent* items = blk->items;
                    for (uint32_t i = 0; i < cnt; ++i) {
                        if (__builtin_expect(!items[i].is_valid(), 0)) continue;
                        uint64_t uh = hash_key64(items[i].user_id, arenas);
                        emit_resolved(items[i], uh, false);
                    }
                }
            } else {
                size_t pushed = 0;
                for (uint32_t bi = 0; bi < nblks; ++bi) {
                    const EvBlock* blk = blks[bi];
                    const uint32_t cnt = blk->count;
                    const RawEvent* items = blk->items;
                    for (uint32_t i = 0; i < cnt; ++i) {
                        if (__builtin_expect(!items[i].is_valid(), 0)) continue;
                        uint64_t uh = hash_key64(items[i].user_id, arenas);
                        ring[pushed & RING_MASK] = {&items[i], uh, false};
                        if (links_has_bloom) links.prefetch_bloom(uh);

                        if (pushed >= 16) {
                            auto& r1 = ring[(pushed - 16) & RING_MASK];
                            bool may_link = (links.has_fallback || (r1.ev->user_id & FALLBACK_BIT) == 0) &&
                                            (!links_has_bloom || links.bloom_might_contain(r1.uh));
                            r1.probe_ht = may_link;
                            if (may_link) links.prefetch_slot(r1.uh);
                        }
                        if (pushed >= 32) {
                            const auto& r2 = ring[(pushed - 32) & RING_MASK];
                            emit_resolved(*r2.ev, r2.uh, r2.probe_ht);
                        }
                        ++pushed;
                    }
                }
                size_t s1_start = (pushed > 16) ? (pushed - 16) : 0;
                for (size_t k = s1_start; k < pushed; ++k) {
                    auto& r1 = ring[k & RING_MASK];
                    bool may_link = (links.has_fallback || (r1.ev->user_id & FALLBACK_BIT) == 0) &&
                                    (!links_has_bloom || links.bloom_might_contain(r1.uh));
                    r1.probe_ht = may_link;
                    if (may_link) links.prefetch_slot(r1.uh);
                }
                size_t s2_start = (pushed > 32) ? (pushed - 32) : 0;
                for (size_t k = s2_start; k < pushed; ++k) {
                    const auto& r2 = ring[k & RING_MASK];
                    emit_resolved(*r2.ev, r2.uh, r2.probe_ht);
                }
            }
        }
        out_ts.flush_all();
    });
    double td1 = now_ms();

    // Stage 3 + Stage 4 FUSED: LPT 512-Bucket Coarse-Radix Sessionize + Immediate writev()!
    int bucket_order[PERSON_BUCKETS];
    size_t bucket_sizes[PERSON_BUCKETS];
    size_t total_surviving = 0;
    for (int b = 0; b < PERSON_BUCKETS; ++b) {
        bucket_order[b] = b;
        bucket_sizes[b] = person_ev_store.bucket_count(b);
        total_surviving += bucket_sizes[b];
    }
    std::sort(bucket_order, bucket_order + PERSON_BUCKETS, [&](int a, int b) {
        return bucket_sizes[a] > bucket_sizes[b];
    });

    static constexpr pq::ColSpec UD_COLS[6] = {
        {"user_id",       pq::ColKind::STRING_REQ},
        {"day",           pq::ColKind::DATE_REQ},
        {"sessions",      pq::ColKind::INT32_REQ},
        {"events",        pq::ColKind::INT32_REQ},
        {"revenue_cents", pq::ColKind::INT64_REQ},
        {"top_category",  pq::ColKind::STRING_OPT},
    };
    static constexpr pq::ColSpec CD_COLS[7] = {
        {"campaign_id",         pq::ColKind::STRING_REQ},
        {"channel",             pq::ColKind::STRING_REQ},
        {"day",                 pq::ColKind::DATE_REQ},
        {"sessions",            pq::ColKind::INT32_REQ},
        {"purchasing_sessions", pq::ColKind::INT32_REQ},
        {"revenue_cents",       pq::ColKind::INT64_REQ},
        {"assisted_sessions",   pq::ColKind::INT32_REQ},
    };

    std::vector<UserDailyBuffers> thread_ud(static_cast<size_t>(pool.nthreads));
    std::vector<std::vector<CampCell>> thread_cd(static_cast<size_t>(pool.nthreads));
    std::atomic<size_t> total_ud_rows_written{0};
    std::atomic<int> finished_workers{0};
    std::atomic<bool> write_ok{true};

    const uint32_t num_camps = campaigns.organic_idx;
    const uint32_t camp_stride = num_camps + 1;
    const uint32_t num_cats = uint32_t(products.cat_names.size());
    std::atomic<int> next_pb{0};

    pool.run([&](int tid) {
        auto& ud = thread_ud[size_t(tid)];
        ud.init(total_surviving / size_t(pool.nthreads * 6) + 4096,
                products.max_cat_len,
                event_store.ts[size_t(tid)]->slabs,
                PartitionedStore<RawEvent, 256, EV_BUCKETS, 8, true>::SLAB_BYTES);

        auto& cd = thread_cd[size_t(tid)];
        cd.assign(size_t(num_days) * size_t(camp_stride), CampCell{0, 0, 0, 0, 0});

        std::vector<uint64_t> pkeys;
        std::vector<uint32_t> pht;
        std::vector<uint32_t> ev_cnt, ev_off, ev_pids;
        std::vector<uint32_t> clk_cnt, clk_off, clk_pids;

        using EvItem = std::conditional_t<Compact, uint64_t, WidePersonEvent>;
        std::vector<EvItem> grouped_evs;
        std::vector<uint64_t> radix_tmp;
        std::vector<LocalClick> grouped_clks;
        std::vector<uint64_t> cat_rev(num_cats, 0);
        std::vector<uint32_t> used_cats;
        used_cats.reserve(32);

        std::vector<CampState> cstate(num_camps, {0, 0, 0});
        std::vector<uint32_t> active_camps;
        active_camps.reserve(num_camps);

        while (true) {
            int ord = next_pb.fetch_add(1, std::memory_order_relaxed);
            if (ord >= PERSON_BUCKETS) break;
            int b = bucket_order[ord];
            size_t nev = bucket_sizes[b];
            if (nev == 0) continue;

            pkeys.clear();
            ev_cnt.clear();
            ev_pids.clear();
            if (ev_pids.capacity() < nev) ev_pids.reserve(nev);

            size_t est_people = std::max<size_t>(64, nev / 4);
            size_t hcap = 64;
            while (hcap < est_people * 2) hcap <<= 1;
            pht.assign(hcap, 0);
            size_t hmask = hcap - 1;

            auto rehash_pht = [&]() {
                hcap <<= 1;
                hmask = hcap - 1;
                pht.assign(hcap, 0);
                for (uint32_t pid = 0; pid < uint32_t(pkeys.size()); ++pid) {
                    size_t slot = size_t(hash_key64(pkeys[pid], arenas)) & hmask;
                    while (pht[slot] != 0) slot = (slot + 1) & hmask;
                    pht[slot] = pid + 1;
                }
            };

            for (int st = 0; st < pool.nthreads; ++st) {
                for (auto* blk = person_ev_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                    const uint32_t cnt = blk->count;
                    const PEv* items = blk->items;
                    for (uint32_t i = 0; i < cnt; ++i) {
                        uint64_t pk = items[i].person_key;
                        if (__builtin_expect(pkeys.size() * 2 >= hcap, 0)) rehash_pht();
                        size_t slot = size_t(hash_key64(pk, arenas)) & hmask;
                        uint32_t pid;
                        while (true) {
                            uint32_t id1 = pht[slot];
                            if (id1 == 0) {
                                pid = uint32_t(pkeys.size());
                                pkeys.push_back(pk);
                                ev_cnt.push_back(0);
                                pht[slot] = pid + 1;
                                break;
                            }
                            if (key_eq(pkeys[id1 - 1], pk, arenas)) { pid = id1 - 1; break; }
                            slot = (slot + 1) & hmask;
                        }
                        ev_cnt[pid]++;
                        ev_pids.push_back(pid);
                    }
                }
            }

            uint32_t num_people = uint32_t(pkeys.size());
            ev_off.resize(num_people + 1);
            uint32_t acc = 0;
            for (uint32_t p = 0; p < num_people; ++p) { ev_off[p] = acc; acc += ev_cnt[p]; }
            ev_off[num_people] = acc;
            grouped_evs.resize(acc);

            size_t flat_i = 0;
            for (int st = 0; st < pool.nthreads; ++st) {
                for (auto* blk = person_ev_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                    const uint32_t cnt = blk->count;
                    const PEv* items = blk->items;
                    for (uint32_t i = 0; i < cnt; ++i) {
                        uint32_t pos = ev_off[ev_pids[flat_i++]]++;
                        if constexpr (Compact) grouped_evs[pos] = items[i].packed;
                        else grouped_evs[pos] = items[i];
                    }
                }
            }
            acc = 0;
            for (uint32_t p = 0; p < num_people; ++p) { ev_off[p] = acc; acc += ev_cnt[p]; }

            size_t nclk_bucket = click_store.bucket_count(b);
            clk_cnt.assign(num_people, 0);
            clk_pids.clear();
            if (nclk_bucket > 0) {
                if (clk_pids.capacity() < nclk_bucket) clk_pids.reserve(nclk_bucket);
                for (int st = 0; st < pool.nthreads; ++st) {
                    for (auto* blk = click_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                        const uint32_t cnt = blk->count;
                        const PersonClick* items = blk->items;
                        for (uint32_t i = 0; i < cnt; ++i) {
                            uint64_t pk = items[i].person_key;
                            size_t slot = size_t(hash_key64(pk, arenas)) & hmask;
                            uint32_t pid = UINT32_MAX;
                            while (true) {
                                uint32_t id1 = pht[slot];
                                if (id1 == 0) break;
                                if (key_eq(pkeys[id1 - 1], pk, arenas)) { pid = id1 - 1; break; }
                                slot = (slot + 1) & hmask;
                            }
                            clk_pids.push_back(pid);
                            if (pid != UINT32_MAX) clk_cnt[pid]++;
                        }
                    }
                }
            }
            clk_off.resize(num_people + 1);
            uint32_t cacc = 0;
            for (uint32_t p = 0; p < num_people; ++p) { clk_off[p] = cacc; cacc += clk_cnt[p]; }
            clk_off[num_people] = cacc;
            grouped_clks.resize(cacc);
            if (cacc > 0) {
                size_t ci = 0;
                for (int st = 0; st < pool.nthreads; ++st) {
                    for (auto* blk = click_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                        const uint32_t cnt = blk->count;
                        const PersonClick* items = blk->items;
                        for (uint32_t i = 0; i < cnt; ++i) {
                            uint32_t pid = clk_pids[ci++];
                            if (pid != UINT32_MAX) {
                                grouped_clks[clk_off[pid]++] = {items[i].ts_camp, items[i].click_id};
                            }
                        }
                    }
                }
                cacc = 0;
                for (uint32_t p = 0; p < num_people; ++p) { clk_off[p] = cacc; cacc += clk_cnt[p]; }
            }

            for (uint32_t pid = 0; pid < num_people; ++pid) {
                uint32_t es = ev_off[pid];
                uint32_t ee = ev_off[pid + 1];
                uint32_t en = ee - es;
                if (en == 0) continue;

                if constexpr (Compact) {
                    uint64_t* arr = grouped_evs.data() + es;
                    if (en <= 24) {
                        for (uint32_t i = 1; i < en; ++i) {
                            uint64_t x = arr[i];
                            uint32_t j = i;
                            while (j > 0 && arr[j - 1] > x) { arr[j] = arr[j - 1]; --j; }
                            arr[j] = x;
                        }
                    } else if (en < 256) {
                        std::sort(arr, arr + en);
                    } else if (en < 65536) {
                        radix_sort_256_u64(arr, en, radix_tmp);
                    } else {
                        radix_sort_ts_u64(arr, en, radix_tmp);
                    }
                } else {
                    auto* arr = grouped_evs.data() + es;
                    std::sort(arr, arr + en, [](const WidePersonEvent& a, const WidePersonEvent& b) {
                        return a.ts_ms < b.ts_ms;
                    });
                }

                uint32_t cs = clk_off[pid];
                uint32_t ce = clk_off[pid + 1];
                uint32_t cn = ce - cs;
                LocalClick* clks = grouped_clks.data() + cs;
                if (cn > 1) {
                    auto clk_less = [&](const LocalClick& a, const LocalClick& b) -> bool {
                        uint64_t ta = a.ts_off(), tb = b.ts_off();
                        if (ta != tb) return ta < tb;
                        return key_less(b.click_id, a.click_id, arenas);
                    };
                    if (cn <= 24) {
                        for (uint32_t i = 1; i < cn; ++i) {
                            LocalClick x = clks[i];
                            uint32_t j = i;
                            while (j > 0 && clk_less(x, clks[j - 1])) { clks[j] = clks[j - 1]; --j; }
                            clks[j] = x;
                        }
                    } else {
                        std::sort(clks, clks + cn, clk_less);
                    }
                    uint32_t un = 1;
                    for (uint32_t k = 1; k < cn; ++k) {
                        if (__builtin_expect(clks[k].ts_off() == clks[un - 1].ts_off() &&
                                             key_eq(clks[k].click_id, clks[un - 1].click_id, arenas), 0)) continue;
                        clks[un++] = clks[k];
                    }
                    cn = un;
                }

                int32_t cur_day = -1;
                int64_t cur_day_end_ms = INT64_MIN;
                uint32_t day_sessions = 0, day_events = 0;
                uint64_t day_rev = 0;
                uint32_t L = 0, R = 0;

                auto flush_day = [&]() {
                    if (day_sessions == 0) return;
                    if (!active_camps.empty()) {
                        CampCell* day_row = cd.data() + size_t(cur_day) * camp_stride;
                        for (uint32_t ac : active_camps) {
                            day_row[ac].assisted_sessions += (day_sessions - cstate[ac].enter_sess);
                            cstate[ac].enter_sess = 0;
                        }
                    }
                    int32_t best_c = -1;
                    if (!used_cats.empty()) {
                        uint32_t bc = used_cats[0];
                        uint64_t br = cat_rev[bc];
                        for (size_t k = 1; k < used_cats.size(); ++k) {
                            uint32_t c = used_cats[k];
                            uint64_t r = cat_rev[c];
                            if (r > br || (r == br && c < bc)) { br = r; bc = c; }
                        }
                        best_c = int32_t(bc);
                        for (uint32_t c : used_cats) cat_rev[c] = 0;
                        used_cats.clear();
                    }
                    ud.push_row_key(pkeys[pid], arenas, win_start_day + cur_day,
                                    int32_t(day_sessions), int32_t(day_events), int64_t(day_rev),
                                    best_c, products.cat_names);
                    day_sessions = 0; day_events = 0; day_rev = 0;
                };

                auto get_ev = [&](uint32_t idx, int64_t& ts, uint32_t& price, uint32_t& cat) {
                    if constexpr (Compact) {
                        uint64_t p = grouped_evs[idx];
                        ts = win_start_ms + int64_t(p >> 32);
                        price = uint32_t((p >> 8) & 0xFFFFFFULL);
                        cat = uint32_t(p & 0xFFULL);
                    } else {
                        const auto& p = grouped_evs[idx];
                        ts = p.ts_ms; price = p.price; cat = p.cat_id;
                    }
                };

                uint32_t i = es;
                while (i < ee) {
                    int64_t s_start, prev_ts;
                    uint32_t price, cat;
                    get_ev(i, s_start, price, cat);
                    prev_ts = s_start;
                    // Monotonic day check (0 division on same day!)
                    if (__builtin_expect(s_start >= cur_day_end_ms, 0)) {
                        int32_t s_day = ms_to_day(s_start) - win_start_day;
                        if (s_day != cur_day) {
                            flush_day();
                            cur_day = s_day;
                            cur_day_end_ms = int64_t(win_start_day + cur_day + 1) * DAY_MS;
                        }
                    }

                    uint32_t s_ev_cnt = 0;
                    uint64_t s_rev = 0;
                    while (i < ee) {
                        int64_t ts;
                        get_ev(i, ts, price, cat);
                        if (ts - prev_ts > SESSION_GAP_MS) break;
                        prev_ts = ts;
                        ++s_ev_cnt;
                        if (__builtin_expect(price > 0, 0)) {
                            s_rev += price;
                            if (cat_rev[cat] == 0) used_cats.push_back(cat);
                            cat_rev[cat] += price;
                        }
                        ++i;
                    }

                    CampCell* day_row = cd.data() + size_t(cur_day) * camp_stride;
                    if (cn == 0) {
                        auto& cell = day_row[campaigns.organic_idx];
                        cell.sessions++;
                        cell.purchasing_sessions += (s_rev > 0);
                        cell.revenue_cents += s_rev;
                    } else {
                        uint64_t s_rel = uint64_t(s_start - click_start_ms);
                        uint64_t win_lo = s_rel >= uint64_t(CLICK_WINDOW_MS) ? (s_rel - uint64_t(CLICK_WINDOW_MS)) : 0;
                        while (R < cn && clks[R].ts_off() <= s_rel) {
                            uint32_t c = clks[R].camp_idx();
                            if (cstate[c].count++ == 0) {
                                cstate[c].enter_sess = day_sessions;
                                cstate[c].active_pos = uint32_t(active_camps.size());
                                active_camps.push_back(c);
                            }
                            ++R;
                        }
                        while (L < R && clks[L].ts_off() < win_lo) {
                            uint32_t c = clks[L].camp_idx();
                            if (--cstate[c].count == 0) {
                                if (day_sessions > cstate[c].enter_sess) {
                                    day_row[c].assisted_sessions += (day_sessions - cstate[c].enter_sess);
                                }
                                uint32_t pos = cstate[c].active_pos;
                                uint32_t back = active_camps.back();
                                active_camps[pos] = back;
                                cstate[back].active_pos = pos;
                                active_camps.pop_back();
                            }
                            ++L;
                        }
                        uint32_t credit_c = (L < R) ? clks[R - 1].camp_idx() : campaigns.organic_idx;
                        auto& cell = day_row[credit_c];
                        cell.sessions++;
                        cell.purchasing_sessions += (s_rev > 0);
                        cell.revenue_cents += s_rev;
                    }
                    day_sessions++;
                    day_events += s_ev_cnt;
                    day_rev += s_rev;
                }
                flush_day();
                for (uint32_t ac : active_camps) cstate[ac].count = 0;
                active_camps.clear();
            }
        }

        // FUSED STAGE 4: Write this worker's user_daily/part-{tid}.parquet immediately via writev()!
        if (ud.num_rows > 0) {
            total_ud_rows_written.fetch_add(ud.num_rows, std::memory_order_relaxed);
            std::string path = ud_dir + "/part-" + std::to_string(tid) + ".parquet";
            pq::ColData data[6] = {
                {ud.uid_plain, ud.uid_sz, nullptr},
                {reinterpret_cast<const uint8_t*>(ud.day_vals), ud.num_rows * 4, nullptr},
                {reinterpret_cast<const uint8_t*>(ud.sess_vals), ud.num_rows * 4, nullptr},
                {reinterpret_cast<const uint8_t*>(ud.ev_vals), ud.num_rows * 4, nullptr},
                {reinterpret_cast<const uint8_t*>(ud.rev_vals), ud.num_rows * 8, nullptr},
                {ud.top_cat_plain, ud.top_cat_sz, ud.top_cat_bitmap},
            };
            if (!pq::write_parquet_file(path.c_str(), UD_COLS, data, 6, int64_t(ud.num_rows))) {
                write_ok.store(false, std::memory_order_relaxed);
            }
        }

        // Last worker to finish writes campaign_daily (and empty user_daily if 0 total rows)!
        if (finished_workers.fetch_add(1, std::memory_order_acq_rel) == pool.nthreads - 1) {
            if (total_ud_rows_written.load(std::memory_order_acquire) == 0) {
                std::string path = ud_dir + "/part-0.parquet";
                pq::ColData empty_ud[6]{};
                if (!pq::write_parquet_file(path.c_str(), UD_COLS, empty_ud, 6, 0)) {
                    write_ok.store(false, std::memory_order_relaxed);
                }
            }

            size_t total_cells = size_t(num_days) * size_t(camp_stride);
            auto& agg = thread_cd[0];
            for (int t = 1; t < pool.nthreads; ++t) {
                const auto& src = thread_cd[size_t(t)];
                for (size_t i = 0; i < total_cells; ++i) {
                    agg[i].sessions += src[i].sessions;
                    agg[i].purchasing_sessions += src[i].purchasing_sessions;
                    agg[i].assisted_sessions += src[i].assisted_sessions;
                    agg[i].revenue_cents += src[i].revenue_cents;
                }
            }

            std::vector<uint8_t> cid_plain, ch_plain;
            std::vector<int32_t> day_v, sess_v, psess_v, assist_v;
            std::vector<int64_t> rev_v;
            cid_plain.reserve(total_cells * 10);
            ch_plain.reserve(total_cells * 10);
            day_v.reserve(total_cells);
            sess_v.reserve(total_cells);
            psess_v.reserve(total_cells);
            rev_v.reserve(total_cells);
            assist_v.reserve(total_cells);

            auto push_s = [](std::vector<uint8_t>& buf, const std::string& s) {
                uint32_t len = uint32_t(s.size());
                size_t old = buf.size();
                buf.resize(old + 4 + len);
                std::memcpy(buf.data() + old, &len, 4);
                std::memcpy(buf.data() + old + 4, s.data(), len);
            };

            for (int32_t d = 0; d < num_days; ++d) {
                int32_t abs_day = win_start_day + d;
                for (uint32_t c = 0; c < camp_stride; ++c) {
                    const auto& cell = agg[size_t(d) * camp_stride + c];
                    if (cell.sessions == 0 && cell.assisted_sessions == 0) continue;
                    push_s(cid_plain, campaigns.ids[c]);
                    push_s(ch_plain, campaigns.channels[c]);
                    day_v.push_back(abs_day);
                    sess_v.push_back(int32_t(cell.sessions));
                    psess_v.push_back(int32_t(cell.purchasing_sessions));
                    rev_v.push_back(int64_t(cell.revenue_cents));
                    assist_v.push_back(int32_t(cell.assisted_sessions));
                }
            }

            std::string path = cd_dir + "/part-0.parquet";
            pq::ColData data[7] = {
                {cid_plain.data(), cid_plain.size(), nullptr},
                {ch_plain.data(), ch_plain.size(), nullptr},
                {reinterpret_cast<const uint8_t*>(day_v.data()), day_v.size() * 4, nullptr},
                {reinterpret_cast<const uint8_t*>(sess_v.data()), sess_v.size() * 4, nullptr},
                {reinterpret_cast<const uint8_t*>(psess_v.data()), psess_v.size() * 4, nullptr},
                {reinterpret_cast<const uint8_t*>(rev_v.data()), rev_v.size() * 8, nullptr},
                {reinterpret_cast<const uint8_t*>(assist_v.data()), assist_v.size() * 4, nullptr},
            };
            if (!pq::write_parquet_file(path.c_str(), CD_COLS, data, 7, int64_t(day_v.size()))) {
                write_ok.store(false, std::memory_order_relaxed);
            }
        }
    });

    double td2 = now_ms();
    if (prof) {
        std::fprintf(stderr, "  sub-profile: dedup=%.1fms sess+write=%.1fms\n", td1 - td0, td2 - td1);
    }
    return write_ok.load(std::memory_order_acquire);
}

__attribute__((visibility("default")))
int run_engine(int argc, char** argv) {
    std::string data_dir = "/data";
    std::string out_dir = "/output";
    if (const char* e = std::getenv("DATA_DIR")) data_dir = e;
    if (const char* e = std::getenv("OUTPUT_DIR")) out_dir = e;
    if (argc >= 3) {
        data_dir = argv[argc - 2];
        out_dir = argv[argc - 1];
    }

    int ncpu = detect_ncpu();
    bool prof = std::getenv("PROFILE") != nullptr;
    auto now_ms = []() {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    double t0 = now_ms();

    DateLut lut;
    int64_t win_start_ms = 0, win_end_ms = 0;
    {
        std::string jv = read_small_file(data_dir + "/manifest.json");
        if (jv.empty()) {
            std::fprintf(stderr, "Failed to open manifest.json in %s\n", data_dir.c_str());
            return 1;
        }
        std::string ws = extract_json_string(jv, "window_start");
        std::string we = extract_json_string(jv, "window_end");
        if (!parse_ts(ws.data(), ws.size(), win_start_ms, lut) ||
            !parse_ts(we.data(), we.size(), win_end_ms, lut)) {
            std::fprintf(stderr, "Failed to parse manifest timestamps\n");
            return 1;
        }
        lut.init(win_start_ms - CLICK_WINDOW_MS, win_end_ms);
    }
    int64_t win_start_sec = win_start_ms / 1000LL;
    int64_t click_start_ms = win_start_ms - CLICK_WINDOW_MS;
    int32_t win_start_day = ms_to_day(win_start_ms);
    int32_t win_end_day = (win_end_ms > win_start_ms) ? ms_to_day(win_end_ms - 1) : win_start_day;
    int32_t num_days = std::max<int32_t>(1, win_end_day - win_start_day + 1);

    std::string ud_dir = out_dir + "/user_daily";
    std::string cd_dir = out_dir + "/campaign_daily";
    mkdir_p(ud_dir);
    mkdir_p(cd_dir);

    ThreadPool pool(ncpu);
    StringArenaPool arenas;
    arenas.init(ncpu);

    // Stage 0: Overlap products/campaigns load (Thread 0) with parallel mmap of ALL 3 CSV dirs!
    std::vector<std::pair<std::string, uint8_t>> all_paths;
    all_paths.reserve(256);
    list_csv_paths(data_dir + "/links", 2, all_paths);
    list_csv_paths(data_dir + "/clicks", 0, all_paths);
    list_csv_paths(data_dir + "/events", 1, all_paths);

    std::vector<MmapCsvFile> mapped_files(all_paths.size());
    Products products;
    Campaigns campaigns;
    std::atomic<size_t> next_f{0};

    pool.run([&](int tid) {
        if (tid == 0) {
            products.load(data_dir + "/products.csv");
            campaigns.load(data_dir + "/campaigns.csv");
        }
        while (true) {
            size_t i = next_f.fetch_add(1, std::memory_order_relaxed);
            if (i >= all_paths.size()) break;
            mapped_files[i] = open_and_mmap_csv(all_paths[i].first, all_paths[i].second);
        }
    });

    std::vector<MmapTask> link_tasks, io_tasks;
    build_mmap_tasks(mapped_files, 1U << 2, ncpu, link_tasks);
    build_mmap_tasks(mapped_files, (1U << 0) | (1U << 1), ncpu, io_tasks);
    double t_init = now_ms();

    // Stage 1: Blazing 4-Barrier Lock-Free Direct Hash-DSU + 1-Word Bloom
    LinkMap links;
    links.build(link_tasks, pool, arenas);
    double t_links = now_ms();

    // Stage 2: Fused Non-Overlapping 64B SIMD Scan of clicks/ + events/
    PartitionedStore<PersonClick, 128, PERSON_BUCKETS, 4, false> click_store;
    PartitionedStore<RawEvent, 256, EV_BUCKETS, 8, true> event_store;
    click_store.init(ncpu);
    event_store.init(ncpu);

    std::vector<std::vector<SideEvent>> thread_sides(static_cast<size_t>(ncpu));

    if (!io_tasks.empty()) {
        std::atomic<size_t> next_t{0};
        uint32_t view_u32, cart_u32;
        uint64_t pur_u64;
        std::memcpy(&view_u32, "view", 4);
        std::memcpy(&cart_u32, "cart", 4);
        std::memcpy(&pur_u64, "purchase", 8);

        const bool links_active = (links.cap > 0);
        const bool links_has_bloom = (links.bloom != nullptr);

        pool.run([&](int tid) {
            auto& clk_out = *click_store.ts[size_t(tid)];
            auto& ev_out = *event_store.ts[size_t(tid)];
            auto& side_vec = thread_sides[size_t(tid)];
            side_vec.reserve(65536);

            // 64-entry L1 prefetch ring for clicks LinkMap resolution!
            struct ClickRing {
                uint64_t ukey;
                uint64_t uh;
                uint64_t cid;
                uint64_t ts_camp;
                bool probe_ht;
            };
            constexpr size_t CR_SZ = 64;
            constexpr size_t CR_MASK = CR_SZ - 1;
            ClickRing cring[CR_SZ];
            size_t cpushed = 0;

            auto emit_click = [&](const ClickRing& r) {
                uint64_t pkey = r.probe_ht ? links.resolve_slot_only(r.ukey, r.uh, arenas) : r.ukey;
                uint64_t ph = (pkey == r.ukey) ? r.uh : hash_key64(pkey, arenas);
                uint32_t pb = uint32_t(ph >> 55);
                clk_out.push(pb, PersonClick{pkey, r.cid, r.ts_camp});
            };

            while (true) {
                size_t ti = next_t.fetch_add(1, std::memory_order_relaxed);
                if (ti >= io_tasks.size()) break;
                const auto& task = io_tasks[ti];
                if (task.kind == 0) {
                    // Clicks with 2-stage prefetched LinkMap resolution!
                    scan_mmap_csv_simd<4>(task, [&](const Slice f[4]) {
                        if (__builtin_expect(f[0].len > 0 && f[1].len > 0 && f[2].len > 0 && f[3].len > 0, 1)) {
                            uint32_t camp_idx = campaigns.lookup(f[2].ptr, f[2].len);
                            if (__builtin_expect(camp_idx != FastStringIdMap::NOT_FOUND, 1)) {
                                int64_t ts = 0;
                                if (__builtin_expect(parse_ts(f[3].ptr, f[3].len, ts, lut) &&
                                                     ts >= click_start_ms && ts < win_end_ms, 1)) {
                                    uint64_t ukey = encode_key64(f[1].ptr, f[1].len, arenas, tid);
                                    uint64_t cid = encode_key64(f[0].ptr, f[0].len, arenas, tid);
                                    uint64_t ts_camp = (uint64_t(ts - click_start_ms) << 20) | uint64_t(camp_idx);
                                    uint64_t uh = hash_key64(ukey, arenas);
                                    if (__builtin_expect(!links_active, 0)) {
                                        uint32_t pb = uint32_t(uh >> 55);
                                        clk_out.push(pb, PersonClick{ukey, cid, ts_camp});
                                    } else {
                                        cring[cpushed & CR_MASK] = {ukey, uh, cid, ts_camp, false};
                                        if (links_has_bloom) links.prefetch_bloom(uh);
                                        if (cpushed >= 16) {
                                            auto& r1 = cring[(cpushed - 16) & CR_MASK];
                                            bool ml = (links.has_fallback || (r1.ukey & FALLBACK_BIT) == 0) &&
                                                      (!links_has_bloom || links.bloom_might_contain(r1.uh));
                                            r1.probe_ht = ml;
                                            if (ml) links.prefetch_slot(r1.uh);
                                        }
                                        if (cpushed >= 32) {
                                            emit_click(cring[(cpushed - 32) & CR_MASK]);
                                        }
                                        ++cpushed;
                                    }
                                }
                            }
                        }
                    });
                } else {
                    // Events: zero-pext event_id + same-10B date reuse + exact ms ingest_ts!
                    scan_mmap_csv_simd<7>(task, [&](const Slice f[7]) {
                        if (__builtin_expect(f[0].len > 0, 1)) {
                            int64_t ingest_ms = 0;
                            int64_t ing_day_sec = 0;
                            bool ing_is_f1 = false;
                            bool ing_whole_sec = false;
                            if (__builtin_expect(f[3].len == 20 && f[3].ptr[19] == 'Z', 1)) {
                                int64_t isec;
                                if (__builtin_expect(!parse_iso19_sec_day(f[3].ptr, isec, ing_day_sec, lut), 0)) return;
                                ingest_ms = isec * 1000LL;
                                ing_is_f1 = true;
                                ing_whole_sec = true;
                            } else {
                                if (!parse_ts(f[3].ptr, f[3].len, ingest_ms, lut)) return;
                                ing_whole_sec = (ingest_ms % 1000LL == 0);
                            }

                            uint64_t eid = encode_event_id(f[0].ptr, f[0].len, arenas, tid);
                            uint32_t eb = uint32_t(hash_key64(eid, arenas) >> 56); // 256 event buckets!

                            uint32_t irel30 = 0U;
                            if (__builtin_expect(ing_whole_sec, 1)) {
                                int64_t irel_s = (ingest_ms / 1000LL - win_start_sec) + RawEvent::REL_BIAS;
                                if (__builtin_expect(irel_s > 0 && irel_s <= RawEvent::REL_MAX, 1)) {
                                    irel30 = uint32_t(irel_s);
                                }
                            }

                            bool ok = (f[1].len > 0 && f[5].len > 0);
                            int etype = 0;
                            if (__builtin_expect(ok, 1)) {
                                if (f[4].len == 4) {
                                    uint32_t v;
                                    std::memcpy(&v, f[4].ptr, 4);
                                    if (v == view_u32 || v == cart_u32) etype = 1;
                                    else ok = false;
                                } else if (f[4].len == 8) {
                                    uint64_t v;
                                    std::memcpy(&v, f[4].ptr, 8);
                                    if (v == pur_u64) etype = 2;
                                    else ok = false;
                                } else {
                                    ok = false;
                                }
                            }

                            int64_t ts = 0;
                            if (__builtin_expect(ok, 1)) {
                                bool ts_ok = false;
                                if (f[2].len == 20 && f[2].ptr[19] == 'Z') {
                                    int64_t tsec = 0;
                                    if (__builtin_expect(ing_is_f1, 1)) {
                                        uint64_t a8, b8;
                                        uint16_t a2, b2;
                                        std::memcpy(&a8, f[2].ptr, 8);
                                        std::memcpy(&b8, f[3].ptr, 8);
                                        std::memcpy(&a2, f[2].ptr + 8, 2);
                                        std::memcpy(&b2, f[3].ptr + 8, 2);
                                        if (__builtin_expect(a8 == b8 && a2 == b2, 1)) {
                                            ts_ok = parse_hms_only(f[2].ptr, ing_day_sec, tsec);
                                        } else {
                                            ts_ok = parse_iso19_sec(f[2].ptr, tsec, lut);
                                        }
                                    } else {
                                        ts_ok = parse_iso19_sec(f[2].ptr, tsec, lut);
                                    }
                                    ts = tsec * 1000LL;
                                } else if (f[2].len == 13) {
                                    ts_ok = parse_13d_swar(f[2].ptr, ts);
                                } else {
                                    ts_ok = parse_ts(f[2].ptr, f[2].len, ts, lut);
                                }
                                if (__builtin_expect(!ts_ok || ts < win_start_ms || ts >= win_end_ms ||
                                                     ingest_ms - ts > DAY_MS, 0)) {
                                    ok = false;
                                }
                            }

                            uint32_t price = 0;
                            uint32_t cat_id = 0;
                            if (__builtin_expect(ok && etype == 2, 0)) {
                                if (!parse_price_cents(f[6].ptr, f[6].len, price)) {
                                    ok = false;
                                } else {
                                    cat_id = products.lookup(f[5].ptr, f[5].len);
                                }
                            }

                            if (__builtin_expect(ok, 1)) {
                                uint64_t ukey = encode_key64(f[1].ptr, f[1].len, arenas, tid);
                                uint64_t ts_off = uint64_t(ts - win_start_ms);
                                if (__builtin_expect(etype == 1 && irel30 != 0 && ts_off <= 0xFFFFFFFFULL, 1)) {
                                    ev_out.push(eb, RawEvent{eid, ukey, RawEvent::pack_inline(true, irel30, uint32_t(ts_off))});
                                } else {
                                    uint32_t sidx = uint32_t(side_vec.size());
                                    if (__builtin_expect(sidx >= (1U << 26), 0)) std::abort();
                                    side_vec.push_back({ts, ingest_ms, price, cat_id});
                                    ev_out.push(eb, RawEvent{eid, ukey, RawEvent::pack_side(true, irel30, tid, sidx)});
                                }
                            } else {
                                if (__builtin_expect(irel30 != 0, 1)) {
                                    ev_out.push(eb, RawEvent{eid, 0, RawEvent::pack_inline(false, irel30, 0)});
                                } else {
                                    uint32_t sidx = uint32_t(side_vec.size());
                                    if (__builtin_expect(sidx >= (1U << 26), 0)) std::abort();
                                    side_vec.push_back({0, ingest_ms, 0, 0});
                                    ev_out.push(eb, RawEvent{eid, 0, RawEvent::pack_side(false, 0, tid, sidx)});
                                }
                            }
                        }
                    });
                }
            }

            // Drain remaining <= 32 clicks in prefetch ring!
            if (cpushed > 0) {
                size_t s1 = (cpushed > 16) ? (cpushed - 16) : 0;
                for (size_t k = s1; k < cpushed; ++k) {
                    auto& r1 = cring[k & CR_MASK];
                    bool ml = (links.has_fallback || (r1.ukey & FALLBACK_BIT) == 0) &&
                              (!links_has_bloom || links.bloom_might_contain(r1.uh));
                    r1.probe_ht = ml;
                    if (ml) links.prefetch_slot(r1.uh);
                }
                size_t s2 = (cpushed > 32) ? (cpushed - 32) : 0;
                for (size_t k = s2; k < cpushed; ++k) {
                    emit_click(cring[k & CR_MASK]);
                }
            }
            clk_out.flush_all();
            ev_out.flush_all();
        });
    }
    double t_scan = now_ms();

    bool use_compact = (win_end_ms - win_start_ms < (1LL << 32)) && (products.cat_names.size() < 256);
    bool write_ok = false;
    if (use_compact) {
        write_ok = run_dedup_sessionize_and_write<true>(
            event_store, thread_sides, click_store, links, products, campaigns, arenas, pool,
            win_start_ms, win_end_ms, win_start_sec, click_start_ms, win_start_day, num_days, ud_dir, cd_dir, prof);
    } else {
        write_ok = run_dedup_sessionize_and_write<false>(
            event_store, thread_sides, click_store, links, products, campaigns, arenas, pool,
            win_start_ms, win_end_ms, win_start_sec, click_start_ms, win_start_day, num_days, ud_dir, cd_dir, prof);
    }
    double t_end = now_ms();

    if (prof) {
        std::fprintf(stderr,
            "PROFILE (ms): init+open=%.1f links=%.1f scan=%.1f dedup+sess+write=%.1f TOTAL=%.1f\n",
            t_init - t0, t_links - t_init, t_scan - t_links, t_end - t_scan, t_end - t0);
    }
    if (!write_ok) {
        std::fprintf(stderr, "Parquet write failed\n");
        ::_exit(1);
    }
    ::_exit(0);
}

} // namespace SS_ENGINE_NS
