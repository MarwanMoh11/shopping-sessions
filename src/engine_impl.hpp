#pragma once
#include "common.hpp"
#include "parquet_writer.hpp"

#include <algorithm>
#include <atomic>
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

namespace SS_ENGINE_NS {

using namespace ss;

// ============================================================================
// Persistent Low-Latency Thread Pool (spawn once, hybrid spin+condvar barrier)
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
        {
            std::lock_guard<std::mutex> lk(mu);
            gen.fetch_add(1, std::memory_order_release);
        }
        cv.notify_all();
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
    std::function<void(int)>* current_fn = nullptr;

    void worker_loop(int tid) {
        uint32_t my_gen = 0;
        while (true) {
            uint32_t target = my_gen + 1;
            int spins = 0;
            while (gen.load(std::memory_order_acquire) < target) {
                if (++spins > 40000) {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&]() { return gen.load(std::memory_order_acquire) >= target; });
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
// L1 Write-Combining Slab-Backed Partitioned Store (ZERO DTLB Thrashing!)
// ============================================================================
template <typename T, uint32_t CAP, int NUM_B, uint32_t WC_CAP>
struct PartitionedStore {
    struct Block {
        Block* next;
        uint32_t count;
        uint32_t _pad;
        T items[CAP];
    };
    static constexpr size_t SLAB_BYTES = 16ULL << 20; // 16 MB THP-friendly slabs

    struct alignas(64) ThreadState {
        // Contiguous L1/L2-hot software write-combining mini-buffers!
        T wc_buf[NUM_B][WC_CAP];
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

        inline void flush_bucket(uint32_t b) {
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
                wc_cnt[bucket] = WC_CAP;
                flush_bucket(bucket);
            } else {
                wc_cnt[bucket] = c;
            }
        }

        void flush_all() {
            for (uint32_t b = 0; b < NUM_B; ++b) {
                if (wc_cnt[b]) flush_bucket(b);
            }
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

    void free_slabs() {
        for (auto* s : ts) {
            for (void* p : s->slabs) ::munmap(p, SLAB_BYTES);
            s->slabs.clear();
            s->slab = nullptr;
            s->slab_pos = SLAB_BYTES;
        }
    }
};

// ============================================================================
// PREAD Zero-Page-Fault Chunked Single-Pass SIMD CSV Scanner
// ============================================================================
struct OpenCsvFile {
    int fd = -1;
    size_t file_size = 0;
    size_t data_start = 0;
};

struct PreadTask {
    int fd;
    size_t file_size;
    size_t data_start;
    size_t start;
    size_t end;
    uint8_t kind; // 0 = click, 1 = event, 2 = link
};

static std::vector<std::string> list_csv_files(const std::string& dir) {
    std::vector<std::string> out;
    DIR* d = ::opendir(dir.c_str());
    if (!d) return out;
    while (auto* ent = ::readdir(d)) {
        std::string_view name(ent->d_name);
        if (name.size() >= 4 && name.substr(name.size() - 4) == ".csv") {
            std::string full = dir + "/" + std::string(name);
            struct stat st{};
            if (::stat(full.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
                out.push_back(std::move(full));
            }
        }
    }
    ::closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

static size_t find_header_end_fd(int fd, size_t file_size) {
    char buf[4096];
    size_t off = 0;
    while (off < file_size) {
        size_t to_read = std::min<size_t>(sizeof(buf), file_size - off);
        ssize_t n = ::pread(fd, buf, to_read, off_t(off));
        if (n <= 0) break;
        const char* nl = find_nl_simd(buf, buf + n);
        if (nl) return off + size_t(nl - buf) + 1;
        off += size_t(n);
    }
    return file_size;
}

static void open_pread_dir(
    const std::string& dir,
    uint8_t kind,
    ThreadPool& pool,
    std::vector<OpenCsvFile>& open_files,
    std::vector<PreadTask>& tasks
) {
    auto paths = list_csv_files(dir);
    if (paths.empty()) return;

    std::vector<OpenCsvFile> results(paths.size());
    std::atomic<size_t> next_f{0};
    pool.run([&](int) {
        while (true) {
            size_t i = next_f.fetch_add(1, std::memory_order_relaxed);
            if (i >= paths.size()) break;
            int fd = ::open(paths[i].c_str(), O_RDONLY);
            if (fd < 0) continue;
            struct stat st{};
            if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
                ::close(fd);
                continue;
            }
            size_t fsz = size_t(st.st_size);
            size_t ds = find_header_end_fd(fd, fsz);
            if (ds >= fsz) {
                ::close(fd);
                continue;
            }
            results[i] = {fd, fsz, ds};
        }
    });

    size_t total_bytes = 0;
    size_t start_idx = open_files.size();
    for (const auto& f : results) {
        if (f.fd >= 0) {
            open_files.push_back(f);
            total_bytes += (f.file_size - f.data_start);
        }
    }
    if (open_files.size() == start_idx) return;

    size_t chunk_sz = std::max<size_t>(65536, std::min<size_t>(4ULL << 20, total_bytes / size_t(pool.nthreads * 6) + 1));
    for (size_t i = start_idx; i < open_files.size(); ++i) {
        const auto& f = open_files[i];
        size_t pos = f.data_start;
        while (pos < f.file_size) {
            size_t end = std::min<size_t>(f.file_size, pos + chunk_sz);
            tasks.push_back({f.fd, f.file_size, f.data_start, pos, end, kind});
            pos = end;
        }
    }
}

static void close_pread_files(std::vector<OpenCsvFile>& open_files) {
    for (auto& f : open_files) {
        if (f.fd >= 0) ::close(f.fd);
    }
    open_files.clear();
}

// Single-pass 64-byte SIMD CSV task scanner (ZERO memchr calls, ZERO redundant byte loads!)
// Calls `on_fields(const Slice f[N])` for every well-formed line owned by `task`.
template <int N, typename OnFields>
static inline void scan_pread_csv_simd(const PreadTask& task, std::vector<char>& buf, OnFields&& on_fields) {
    constexpr size_t PAD = 256;
    if (buf.size() < (512ULL << 10) + PAD) {
        buf.resize((512ULL << 10) + PAD);
    }
    size_t cap = buf.size() - PAD;
    size_t cur = task.start;

    // Align to first line starting in [task.start, task.end)
    if (cur > task.data_start) {
        // Read 1 byte before start + first block in ONE syscall!
        size_t to_read = std::min<size_t>(cap, task.file_size - (cur - 1));
        ssize_t got = ::pread(task.fd, buf.data(), to_read, off_t(cur - 1));
        if (got <= 1) return;
        if (buf[0] != '\n') {
            const char* nl = find_nl_simd(buf.data() + 1, buf.data() + got);
            if (nl) {
                cur = (cur - 1) + size_t(nl - buf.data()) + 1;
            } else {
                cur = (cur - 1) + size_t(got);
                while (cur < task.end) {
                    size_t tr = std::min<size_t>(cap, task.file_size - cur);
                    ssize_t gn = ::pread(task.fd, buf.data(), tr, off_t(cur));
                    if (gn <= 0) return;
                    const char* nl2 = find_nl_simd(buf.data(), buf.data() + gn);
                    if (nl2) { cur += size_t(nl2 - buf.data()) + 1; break; }
                    cur += size_t(gn);
                }
            }
            if (cur >= task.end) return;
        }
    }

    Slice f[N];
    while (cur < task.end) {
        size_t to_read = std::min<size_t>(cap, task.file_size - cur);
        ssize_t got = ::pread(task.fd, buf.data(), to_read, off_t(cur));
        if (got <= 0) break;
        size_t n = size_t(got);
        std::memset(buf.data() + n, 0, PAD); // 0x00 != '\n', ',', '"'

        const char* base = buf.data();
        size_t pos = 0;
        const size_t limit = std::min<size_t>(n, task.end - cur);

        while (pos < limit) {
            const char* line = base + pos;
            Vec64 v0 = Vec64::load(line);
            uint64_t nl0 = v0.eq('\n');
            if (nl0 != 0) {
                size_t L = size_t(__builtin_ctzll(nl0));
                if (__builtin_expect(L > 0, 1)) {
                    uint64_t lmask = (1ULL << L) - 1ULL;
                    uint64_t cm0 = v0.eq(',') & lmask;
                    uint64_t qm0 = v0.eq('"') & lmask;
                    if (__builtin_expect(extract_fields_from_masks<N>(line, L, cm0, 0, qm0, 0, f), 1)) {
                        on_fields(f);
                    }
                }
                pos += L + 1;
                continue;
            }
            if (__builtin_expect(pos + 64 >= n, 0)) break;

            Vec64 v1 = Vec64::load(line + 64);
            uint64_t nl1 = v1.eq('\n');
            if (__builtin_expect(nl1 != 0, 1)) {
                size_t k = size_t(__builtin_ctzll(nl1));
                size_t L = 64 + k;
                uint64_t cm0 = v0.eq(',');
                uint64_t qm0 = v0.eq('"');
                uint64_t lmask1 = (k == 0) ? 0ULL : ((1ULL << k) - 1ULL);
                uint64_t cm1 = v1.eq(',') & lmask1;
                uint64_t qm1 = v1.eq('"') & lmask1;
                if (__builtin_expect(extract_fields_from_masks<N>(line, L, cm0, cm1, qm0, qm1, f), 1)) {
                    on_fields(f);
                }
                pos += L + 1;
                continue;
            }
            if (__builtin_expect(pos + 128 >= n, 0)) break;

            // Slow path: line >= 128 bytes (rare adversarial long line)
            const char* nl = find_nl_simd(line + 128, base + n);
            if (!nl) break;
            if (parse_csv_line_scalar<N>(line, nl, f)) {
                on_fields(f);
            }
            pos = size_t(nl - base) + 1;
        }

        if (pos < limit) {
            // Reached end of buffer with an incomplete line (no '\n' in [pos, n))
            if (cur + n == task.file_size) {
                if (parse_csv_line_scalar<N>(base + pos, base + n, f)) {
                    on_fields(f);
                }
                return;
            }
            if (pos == 0) {
                cap <<= 1;
                buf.resize(cap + PAD);
            }
        }
        cur += pos;
    }
}

// ============================================================================
// Small File Helper (manifest.json, products.csv, campaigns.csv)
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
        if (n <= 0) break;
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
        for (auto c : cats) cat_names.emplace_back(c);

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
// Stage 1: L2-Bucketed Links DSU + Non-Minimal Map + Bloom Bitset (0 munmaps!)
// ============================================================================
struct LinkMap {
    struct Slot {
        uint64_t key;
        uint64_t val;
    };
    Slot* slots = nullptr;
    size_t cap = 0;
    size_t mask = 0;

    uint64_t* bloom = nullptr;
    size_t bloom_bits = 0;
    size_t bloom_mask = 0;
    bool has_fallback = false;

    inline bool bloom_might_contain(uint64_t h) const {
        size_t b1 = size_t(h) & bloom_mask;
        size_t b2 = size_t(h >> 32) & bloom_mask;
        return ((bloom[b1 >> 6] >> (b1 & 63)) & 1ULL) & ((bloom[b2 >> 6] >> (b2 & 63)) & 1ULL);
    }

    inline void prefetch_bloom(uint64_t h) const {
        if (__builtin_expect(bloom != nullptr, 1)) {
            __builtin_prefetch(&bloom[(size_t(h) & bloom_mask) >> 6], 0, 1);
            __builtin_prefetch(&bloom[(size_t(h >> 32) & bloom_mask) >> 6], 0, 1);
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

    inline uint64_t resolve(uint64_t k, const StringArenaPool& arenas) const {
        if (__builtin_expect(cap == 0, 0)) return k;
        if (__builtin_expect(!has_fallback && (k & FALLBACK_BIT) != 0, 0)) return k;
        return resolve_with_hash(k, hash_key64(k, arenas), arenas);
    }

    void build(const std::string& links_dir, ThreadPool& pool, StringArenaPool& arenas) {
        std::vector<OpenCsvFile> open_files;
        std::vector<PreadTask> tasks;
        open_pread_dir(links_dir, 2, pool, open_files, tasks);
        if (tasks.empty()) return;

        struct Edge { uint64_t a, b; };
        std::vector<std::vector<Edge>> thread_edges(static_cast<size_t>(pool.nthreads));
        std::atomic<size_t> next_t{0};
        std::atomic<bool> any_fallback{false};

        pool.run([&](int tid) {
            auto& edges = thread_edges[size_t(tid)];
            edges.reserve(131072);
            std::vector<char> buf;
            bool local_fb = false;
            while (true) {
                size_t ti = next_t.fetch_add(1, std::memory_order_relaxed);
                if (ti >= tasks.size()) break;
                scan_pread_csv_simd<2>(tasks[ti], buf, [&](const Slice f[2]) {
                    if (__builtin_expect(f[0].len > 0 && f[1].len > 0, 1)) {
                        uint64_t ka = encode_key64(f[0].ptr, f[0].len, arenas, tid);
                        uint64_t kb = encode_key64(f[1].ptr, f[1].len, arenas, tid);
                        if (__builtin_expect(((ka | kb) & FALLBACK_BIT) != 0, 0)) local_fb = true;
                        edges.push_back({ka, kb});
                    }
                });
            }
            if (local_fb) any_fallback.store(true, std::memory_order_relaxed);
        });
        close_pread_files(open_files);

        size_t total_edges = 0;
        std::vector<size_t> edge_base(static_cast<size_t>(pool.nthreads) + 1, 0);
        for (int t = 0; t < pool.nthreads; ++t) {
            edge_base[size_t(t)] = total_edges;
            total_edges += thread_edges[size_t(t)].size();
        }
        edge_base[size_t(pool.nthreads)] = total_edges;
        if (total_edges == 0) return;

        has_fallback = any_fallback.load(std::memory_order_relaxed);

        if (!has_fallback && total_edges >= 4096) {
            static constexpr int LB = 64;
            struct KeyRef { uint64_t key; uint32_t eside; };
            PartitionedStore<KeyRef, 256, LB, 8> kstore;
            kstore.init(pool.nthreads);

            pool.run([&](int tid) {
                auto& out = *kstore.ts[size_t(tid)];
                const auto& ev = thread_edges[size_t(tid)];
                uint32_t eidx = uint32_t(edge_base[size_t(tid)]);
                for (const auto& e : ev) {
                    if (e.a == e.b) { ++eidx; continue; }
                    out.push(uint32_t(mix64(e.a) >> 58), {e.a, (eidx << 1)});
                    out.push(uint32_t(mix64(e.b) >> 58), {e.b, (eidx << 1) | 1U});
                    ++eidx;
                }
                out.flush_all();
            });

            std::vector<std::vector<uint64_t>> bucket_keys(LB);
            std::vector<std::vector<std::pair<uint32_t, uint32_t>>> bucket_refs(LB);
            std::atomic<int> next_lb{0};
            pool.run([&](int) {
                struct LSlot { uint64_t k; uint32_t id1; };
                std::vector<LSlot> lht;
                while (true) {
                    int b = next_lb.fetch_add(1, std::memory_order_relaxed);
                    if (b >= LB) break;
                    size_t cnt = kstore.bucket_count(b);
                    if (cnt == 0) continue;
                    size_t hcap = 64;
                    while (hcap < cnt * 2) hcap <<= 1;
                    lht.assign(hcap, {0, 0});
                    size_t hmask = hcap - 1;

                    auto& bkeys = bucket_keys[size_t(b)];
                    auto& brefs = bucket_refs[size_t(b)];
                    bkeys.reserve(cnt);
                    brefs.reserve(cnt);

                    for (int st = 0; st < pool.nthreads; ++st) {
                        for (auto* blk = kstore.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                            for (uint32_t i = 0; i < blk->count; ++i) {
                                uint64_t k = blk->items[i].key;
                                size_t slot = size_t(mix64(k)) & hmask;
                                uint32_t lid;
                                while (true) {
                                    if (lht[slot].id1 == 0) {
                                        lid = uint32_t(bkeys.size());
                                        bkeys.push_back(k);
                                        lht[slot] = {k, lid + 1};
                                        break;
                                    }
                                    if (lht[slot].k == k) { lid = lht[slot].id1 - 1; break; }
                                    slot = (slot + 1) & hmask;
                                }
                                brefs.push_back({blk->items[i].eside, lid});
                            }
                        }
                    }
                }
            });

            uint32_t b_base[LB + 1];
            uint32_t num_nodes = 0;
            for (int b = 0; b < LB; ++b) {
                b_base[b] = num_nodes;
                num_nodes += uint32_t(bucket_keys[size_t(b)].size());
            }
            b_base[LB] = num_nodes;
            if (num_nodes == 0) return;

            uint32_t* edge_ends = static_cast<uint32_t*>(alloc_mmap(total_edges * 2 * sizeof(uint32_t)));
            uint64_t* node_key = static_cast<uint64_t*>(alloc_mmap(size_t(num_nodes) * sizeof(uint64_t)));
            std::atomic<uint32_t>* parent = static_cast<std::atomic<uint32_t>*>(alloc_mmap(size_t(num_nodes) * sizeof(std::atomic<uint32_t>)));

            std::atomic<int> next_sb{0};
            pool.run([&](int) {
                while (true) {
                    int b = next_sb.fetch_add(1, std::memory_order_relaxed);
                    if (b >= LB) break;
                    uint32_t base = b_base[b];
                    const auto& bk = bucket_keys[size_t(b)];
                    if (!bk.empty()) {
                        std::memcpy(node_key + base, bk.data(), bk.size() * sizeof(uint64_t));
                        for (uint32_t i = 0; i < uint32_t(bk.size()); ++i) {
                            parent[base + i].store(base + i, std::memory_order_relaxed);
                        }
                    }
                    for (const auto& r : bucket_refs[size_t(b)]) {
                        edge_ends[r.first] = base + r.second;
                    }
                }
            });

            auto find_root = [&](uint32_t x) -> uint32_t {
                while (true) {
                    uint32_t p = parent[x].load(std::memory_order_relaxed);
                    if (p == x) return x;
                    uint32_t gp = parent[p].load(std::memory_order_relaxed);
                    if (gp != p) parent[x].compare_exchange_weak(p, gp, std::memory_order_relaxed);
                    x = gp;
                }
            };
            pool.run([&](int tid) {
                size_t es = (total_edges * size_t(tid)) / size_t(pool.nthreads);
                size_t ee = (total_edges * size_t(tid + 1)) / size_t(pool.nthreads);
                for (size_t e = es; e < ee; ++e) {
                    uint32_t u = edge_ends[e * 2];
                    uint32_t v = edge_ends[e * 2 + 1];
                    if (u == v) continue;
                    while (true) {
                        u = find_root(u);
                        v = find_root(v);
                        if (u == v) break;
                        if (u < v) std::swap(u, v);
                        uint32_t expected = u;
                        if (parent[u].compare_exchange_weak(expected, v, std::memory_order_relaxed)) break;
                    }
                }
            });

            std::atomic<uint64_t>* min_key = static_cast<std::atomic<uint64_t>*>(alloc_mmap(size_t(num_nodes) * sizeof(std::atomic<uint64_t>)));
            pool.run([&](int tid) {
                uint32_t s = uint32_t((uint64_t(num_nodes) * uint64_t(tid)) / uint64_t(pool.nthreads));
                uint32_t e = uint32_t((uint64_t(num_nodes) * uint64_t(tid + 1)) / uint64_t(pool.nthreads));
                for (uint32_t i = s; i < e; ++i) min_key[i].store(UINT64_MAX, std::memory_order_relaxed);
            });
            pool.run([&](int tid) {
                uint32_t s = uint32_t((uint64_t(num_nodes) * uint64_t(tid)) / uint64_t(pool.nthreads));
                uint32_t e = uint32_t((uint64_t(num_nodes) * uint64_t(tid + 1)) / uint64_t(pool.nthreads));
                for (uint32_t i = s; i < e; ++i) {
                    uint32_t r = i;
                    while (true) {
                        uint32_t p = parent[r].load(std::memory_order_relaxed);
                        if (p == r) break;
                        r = p;
                    }
                    parent[i].store(r, std::memory_order_relaxed);
                    uint64_t k = node_key[i];
                    uint64_t cur = min_key[r].load(std::memory_order_relaxed);
                    while (k < cur && !min_key[r].compare_exchange_weak(cur, k, std::memory_order_relaxed)) {}
                }
            });

            std::vector<size_t> t_nonmin(static_cast<size_t>(pool.nthreads), 0);
            pool.run([&](int tid) {
                uint32_t s = uint32_t((uint64_t(num_nodes) * uint64_t(tid)) / uint64_t(pool.nthreads));
                uint32_t e = uint32_t((uint64_t(num_nodes) * uint64_t(tid + 1)) / uint64_t(pool.nthreads));
                size_t c = 0;
                for (uint32_t i = s; i < e; ++i) {
                    if (node_key[i] != min_key[parent[i].load(std::memory_order_relaxed)].load(std::memory_order_relaxed)) ++c;
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

            bloom_bits = 1ULL << 19;
            while (bloom_bits < total_nonmin * 16 && bloom_bits < (64ULL << 20)) bloom_bits <<= 1;
            bloom_mask = bloom_bits - 1;
            bloom = static_cast<uint64_t*>(alloc_mmap(bloom_bits >> 3));

            auto* atomic_slots = reinterpret_cast<std::atomic<uint64_t>*>(slots);
            auto* atomic_bloom = reinterpret_cast<std::atomic<uint64_t>*>(bloom);

            pool.run([&](int tid) {
                uint32_t s = uint32_t((uint64_t(num_nodes) * uint64_t(tid)) / uint64_t(pool.nthreads));
                uint32_t e = uint32_t((uint64_t(num_nodes) * uint64_t(tid + 1)) / uint64_t(pool.nthreads));
                for (uint32_t i = s; i < e; ++i) {
                    uint64_t k = node_key[i];
                    uint64_t mk = min_key[parent[i].load(std::memory_order_relaxed)].load(std::memory_order_relaxed);
                    if (k == mk) continue;
                    uint64_t h = mix64(k);
                    size_t b1 = size_t(h) & bloom_mask;
                    size_t b2 = size_t(h >> 32) & bloom_mask;
                    atomic_bloom[b1 >> 6].fetch_or(1ULL << (b1 & 63), std::memory_order_relaxed);
                    atomic_bloom[b2 >> 6].fetch_or(1ULL << (b2 & 63), std::memory_order_relaxed);

                    size_t idx = size_t(h) & mask;
                    while (true) {
                        uint64_t expected = 0;
                        if (atomic_slots[idx * 2].compare_exchange_strong(expected, k, std::memory_order_relaxed)) {
                            slots[idx].val = mk;
                            break;
                        }
                        idx = (idx + 1) & mask;
                    }
                }
            });
            // Skip mid-run munmap TLB shootdowns (< 180 MB even at 60M)!
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
// Compact Record Structs (24B RawEvent -> 16B PersonEvent, 512 Buckets!)
// ============================================================================
static constexpr int EV_BUCKETS = 512;     // ~40K events/bucket @ 20M -> 512KB L2 hash table!
static constexpr int PERSON_BUCKETS = 512; // ~1.3K people/bucket @ 20M -> 16KB L1 hash table!

struct SideEvent {
    int64_t ts_ms;
    int64_t ingest_sec;
    uint32_t price;
    uint32_t cat_id;
};

struct RawEvent {
    uint64_t event_id;
    uint64_t user_id;
    uint64_t meta;

    static constexpr int64_t REL_BIAS = 1LL << 30;
    static constexpr int64_t REL_MAX = (1LL << 31) - 1;

    inline bool is_valid() const { return (meta >> 63) != 0; }
    inline bool has_side() const { return ((meta >> 62) & 1ULL) != 0; }
    inline uint32_t ingest_rel() const { return uint32_t((meta >> 31) & 0x7FFFFFFFULL); }
    inline uint32_t low31() const { return uint32_t(meta & 0x7FFFFFFFULL); }

    static inline uint64_t pack_inline(bool valid, uint32_t ingest_rel31, uint32_t ts_rel31) {
        return (uint64_t(valid) << 63) | (uint64_t(ingest_rel31) << 31) | uint64_t(ts_rel31);
    }
    static inline uint64_t pack_side(bool valid, uint32_t ingest_rel31, int tid, uint32_t side_idx) {
        return (uint64_t(valid) << 63) | (1ULL << 62) | (uint64_t(ingest_rel31) << 31) |
               (uint64_t(tid) << 25) | uint64_t(side_idx);
    }
};

static inline int64_t get_exact_ingest(const RawEvent& ev, int64_t win_start_sec, const std::vector<std::vector<SideEvent>>& sides) {
    uint32_t ir = ev.ingest_rel();
    if (__builtin_expect(ir != 0, 1)) return win_start_sec + int64_t(ir) - RawEvent::REL_BIAS;
    uint32_t lo = ev.low31();
    return sides[lo >> 25][lo & 0x1FFFFFFU].ingest_sec;
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
    return get_exact_ingest(a, win_start_sec, sides) < get_exact_ingest(b, win_start_sec, sides);
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

struct LocalClick {
    uint64_t ts_off;
    uint64_t click_id;
    uint32_t camp_idx;
};

struct CampCell {
    uint32_t sessions;
    uint32_t purchasing_sessions;
    uint32_t assisted_sessions;
    uint32_t _pad;
    uint64_t revenue_cents;
};

// Zero-realloc mmap-backed column buffer for Stage 3 user_daily streaming!
struct UserDailyBuffers {
    uint8_t* uid_plain = nullptr;
    size_t uid_sz = 0, uid_cap = 0;
    int32_t* day_vals = nullptr;
    int32_t* sess_vals = nullptr;
    int32_t* ev_vals = nullptr;
    int64_t* rev_vals = nullptr;
    uint8_t* top_cat_bitmap = nullptr;
    uint8_t* top_cat_plain = nullptr;
    size_t top_cat_sz = 0, top_cat_cap = 0;
    size_t num_rows = 0, row_cap = 0;

    void init(size_t est_rows) {
        row_cap = std::max<size_t>(4096, est_rows);
        uid_cap = row_cap * 16 + 4096;
        top_cat_cap = row_cap * 4 + 4096;
        // Virtual reserve generous max (committed on demand -> 0 wasted physical RAM!)
        size_t max_rows = std::max<size_t>(row_cap * 4, 4ULL << 20); // up to 4M rows/thread (32M total!)
        row_cap = max_rows;
        uid_cap = max_rows * 16;
        top_cat_cap = max_rows * 12;
        uid_plain = static_cast<uint8_t*>(alloc_mmap(uid_cap));
        day_vals = static_cast<int32_t*>(alloc_mmap(max_rows * 4));
        sess_vals = static_cast<int32_t*>(alloc_mmap(max_rows * 4));
        ev_vals = static_cast<int32_t*>(alloc_mmap(max_rows * 4));
        rev_vals = static_cast<int64_t*>(alloc_mmap(max_rows * 8));
        top_cat_bitmap = static_cast<uint8_t*>(alloc_mmap((max_rows >> 3) + 64));
        top_cat_plain = static_cast<uint8_t*>(alloc_mmap(top_cat_cap));
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
        // Direct inline decode straight into uid_plain (ZERO resize, ZERO tmp copy!)
        if (__builtin_expect((person_key & FALLBACK_BIT) == 0, 1)) {
            uint8_t* dst = uid_plain + uid_sz;
            uint32_t len = uint32_t(decode_inline(person_key, reinterpret_cast<char*>(dst + 4)));
            std::memcpy(dst, &len, 4);
            uid_sz += 4 + len;
        } else {
            auto sv = arenas.get_fallback(person_key);
            uint32_t len = uint32_t(sv.size());
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
            uint8_t* dst = top_cat_plain + top_cat_sz;
            std::memcpy(dst, &len, 4);
            std::memcpy(dst + 4, cs.data(), len);
            top_cat_sz += 4 + len;
        }
    }
};

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

// ============================================================================
// Stage 2C + Stage 3: IN-PLACE L2 Fingerprinted Dedup + 512-Bucket Sessionize
// ============================================================================
template <bool Compact>
void run_dedup_and_sessionize(
    PartitionedStore<RawEvent, 256, EV_BUCKETS, 4>& event_store,
    const std::vector<std::vector<SideEvent>>& sides,
    PartitionedStore<PersonClick, 128, PERSON_BUCKETS, 4>& click_store,
    const LinkMap& links,
    const Products& products,
    const Campaigns& campaigns,
    const StringArenaPool& arenas,
    ThreadPool& pool,
    int64_t win_start_ms,
    int64_t win_start_sec,
    int64_t click_start_ms,
    int32_t win_start_day,
    int32_t num_days,
    std::vector<UserDailyBuffers>& thread_ud,
    std::vector<std::vector<CampCell>>& thread_cd,
    bool prof
) {
    using PEv = std::conditional_t<Compact, CompactPersonEvent, WidePersonEvent>;
    PartitionedStore<PEv, 256, PERSON_BUCKETS, 8> person_ev_store;
    person_ev_store.init(pool.nthreads);

    auto now_ms = []() {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    double td0 = now_ms();

    // Stage 2C: IN-PLACE L2 Fingerprinted Dedup (256KB 4-byte slots!) + 2-Stage Prefetched Ring Buffer!
    using EvBlock = typename PartitionedStore<RawEvent, 256, EV_BUCKETS, 4>::Block;
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

        // 64-entry L1 stack ring buffer for 2-stage software prefetching (0 heap traffic!)
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
                    packed = uint64_t(ev.low31()) << 32;
                } else {
                    uint32_t lo = ev.low31();
                    const SideEvent& se = sides[lo >> 25][lo & 0x1FFFFFFU];
                    packed = (uint64_t(se.ts_ms - win_start_ms) << 32) | (uint64_t(se.price) << 8) | uint64_t(se.cat_id);
                }
                out_ts.push(pb, CompactPersonEvent{pkey, packed});
            } else {
                if (!ev.has_side()) {
                    out_ts.push(pb, WidePersonEvent{pkey, win_start_ms + int64_t(ev.low31()), 0, 0});
                } else {
                    uint32_t lo = ev.low31();
                    const SideEvent& se = sides[lo >> 25][lo & 0x1FFFFFFU];
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
                // Ultra-compact 4-byte slot: [12b fp | 20b handle1] -> 256 KB L2-resident @ 20M!
                ht4.assign(hcap, 0);
                uint32_t* htp = ht4.data();
                for (uint32_t bi = 0; bi < nblks; ++bi) {
                    EvBlock* blk = blks[bi];
                    const uint32_t cnt = blk->count;
                    RawEvent* items = blk->items;
                    for (uint32_t i = 0; i < cnt; ++i) {
                        RawEvent& ev = items[i];
                        uint64_t h = hash_key64(ev.event_id, arenas);
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

            // Stream surviving valid events through 2-stage prefetch ring (Bloom -> LinkHT -> Emit)!
            size_t pushed = 0;
            for (uint32_t bi = 0; bi < nblks; ++bi) {
                EvBlock* blk = blks[bi];
                const uint32_t cnt = blk->count;
                const RawEvent* items = blk->items;
                for (uint32_t i = 0; i < cnt; ++i) {
                    if (__builtin_expect(!items[i].is_valid(), 0)) continue;

                    // Stage 0: compute hash + prefetch Bloom bitset
                    uint64_t uh = hash_key64(items[i].user_id, arenas);
                    ring[pushed & RING_MASK] = {&items[i], uh, false};
                    if (links_has_bloom) links.prefetch_bloom(uh);

                    // Stage 1 (+16 behind): test Bloom, prefetch link_ht slot if hit!
                    if (pushed >= 16) {
                        auto& r1 = ring[(pushed - 16) & RING_MASK];
                        bool may_link = links_active &&
                                        (links.has_fallback || (r1.ev->user_id & FALLBACK_BIT) == 0) &&
                                        (!links_has_bloom || links.bloom_might_contain(r1.uh));
                        r1.probe_ht = may_link;
                        if (may_link) links.prefetch_slot(r1.uh);
                    }

                    // Stage 2 (+32 behind): resolve & scatter!
                    if (pushed >= 32) {
                        const auto& r2 = ring[(pushed - 32) & RING_MASK];
                        emit_resolved(*r2.ev, r2.uh, r2.probe_ht);
                    }
                    ++pushed;
                }
            }
            // Drain final <= 32 items in ring buffer
            size_t s1_start = (pushed > 16) ? (pushed - 16) : 0;
            for (size_t k = s1_start; k < pushed; ++k) {
                auto& r1 = ring[k & RING_MASK];
                bool may_link = links_active &&
                                (links.has_fallback || (r1.ev->user_id & FALLBACK_BIT) == 0) &&
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
        out_ts.flush_all();
    });

    // Skip mid-run munmap on <= 30M datasets; only free at huge 60M+ scale if desired (even at 60M peak is < 3.6 GB!)
    double td1 = now_ms();

    // Stage 3: Sort 512 person buckets largest-first (LPT)
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

    const uint32_t num_camps = campaigns.organic_idx;
    const uint32_t camp_stride = num_camps + 1;
    const uint32_t num_cats = uint32_t(products.cat_names.size());
    std::atomic<int> next_pb{0};

    pool.run([&](int tid) {
        auto& ud = thread_ud[size_t(tid)];
        ud.init(total_surviving / size_t(pool.nthreads * 6) + 4096);
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

        std::vector<uint32_t> camp_count(num_camps, 0);
        std::vector<uint32_t> enter_sess(num_camps, 0);
        std::vector<uint32_t> active_pos(num_camps, 0);
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

            size_t est_people = std::max<size_t>(64, nev / 10);
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
                                uint32_t pos = clk_off[pid]++;
                                grouped_clks[pos] = {items[i].ts_camp >> 20, items[i].click_id, uint32_t(items[i].ts_camp & 0xFFFFFULL)};
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
                    } else if (en <= 256) {
                        std::sort(arr, arr + en);
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
                    std::sort(clks, clks + cn, [&](const LocalClick& a, const LocalClick& b) {
                        if (a.ts_off != b.ts_off) return a.ts_off < b.ts_off;
                        return key_less(b.click_id, a.click_id, arenas);
                    });
                    uint32_t un = 1;
                    for (uint32_t k = 1; k < cn; ++k) {
                        if (clks[k].ts_off == clks[un - 1].ts_off && key_eq(clks[k].click_id, clks[un - 1].click_id, arenas)) continue;
                        clks[un++] = clks[k];
                    }
                    cn = un;
                }

                int32_t cur_day = -1;
                uint32_t day_sessions = 0, day_events = 0;
                uint64_t day_rev = 0;
                uint32_t L = 0, R = 0;

                auto flush_day = [&]() {
                    if (day_sessions == 0) return;
                    if (!active_camps.empty()) {
                        CampCell* day_row = cd.data() + size_t(cur_day) * camp_stride;
                        for (uint32_t ac : active_camps) {
                            day_row[ac].assisted_sessions += (day_sessions - enter_sess[ac]);
                            enter_sess[ac] = 0;
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
                    int32_t s_day = ms_to_day(s_start) - win_start_day;
                    if (s_day != cur_day) { flush_day(); cur_day = s_day; }

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
                        while (R < cn && clks[R].ts_off <= s_rel) {
                            uint32_t c = clks[R].camp_idx;
                            if (camp_count[c]++ == 0) {
                                enter_sess[c] = day_sessions;
                                active_pos[c] = uint32_t(active_camps.size());
                                active_camps.push_back(c);
                            }
                            ++R;
                        }
                        while (L < R && clks[L].ts_off < win_lo) {
                            uint32_t c = clks[L].camp_idx;
                            if (--camp_count[c] == 0) {
                                if (day_sessions > enter_sess[c]) {
                                    day_row[c].assisted_sessions += (day_sessions - enter_sess[c]);
                                }
                                uint32_t pos = active_pos[c];
                                uint32_t back = active_camps.back();
                                active_camps[pos] = back;
                                active_pos[back] = pos;
                                active_camps.pop_back();
                            }
                            ++L;
                        }
                        uint32_t credit_c = (L < R) ? clks[R - 1].camp_idx : campaigns.organic_idx;
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
                for (uint32_t ac : active_camps) camp_count[ac] = 0;
                active_camps.clear();
            }
        }
    });

    double td2 = now_ms();
    if (prof) {
        std::fprintf(stderr, "  sub-profile: dedup=%.1fms sessionize=%.1fms\n", td1 - td0, td2 - td1);
    }
}

static void mkdir_p(const std::string& path) {
    size_t pos = 0;
    while ((pos = path.find('/', pos + 1)) != std::string::npos) {
        ::mkdir(path.substr(0, pos).c_str(), 0755);
    }
    ::mkdir(path.c_str(), 0755);
}

int run_engine(int argc, char** argv) {
    std::string data_dir = "/data";
    std::string out_dir = "/output";
    if (const char* e = std::getenv("DATA_DIR")) data_dir = e;
    if (const char* e = std::getenv("OUTPUT_DIR")) out_dir = e;
    if (argc >= 3) {
        data_dir = argv[argc - 2];
        out_dir = argv[argc - 1];
    }

    int ncpu = int(std::thread::hardware_concurrency());
    if (const char* e = std::getenv("NCPU")) {
        int v = std::atoi(e);
        if (v > 0) ncpu = v;
    }
    if (ncpu < 1) ncpu = 1;
    if (ncpu > 64) ncpu = 64;

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

    ThreadPool pool(ncpu);
    StringArenaPool arenas;
    arenas.init(ncpu);

    Products products;
    Campaigns campaigns;
    products.load(data_dir + "/products.csv");
    campaigns.load(data_dir + "/campaigns.csv");

    double t_init = now_ms();
    LinkMap links;
    links.build(data_dir + "/links", pool, arenas);
    double t_links = now_ms();

    // Discover & open clicks/ and events/ into ONE fused task queue (0 barrier between clicks & events!)
    std::vector<OpenCsvFile> io_files;
    std::vector<PreadTask> io_tasks;
    open_pread_dir(data_dir + "/clicks", 0, pool, io_files, io_tasks);
    open_pread_dir(data_dir + "/events", 1, pool, io_files, io_tasks);
    double t_chunks = now_ms();

    PartitionedStore<PersonClick, 128, PERSON_BUCKETS, 4> click_store;
    PartitionedStore<RawEvent, 256, EV_BUCKETS, 4> event_store;
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

        pool.run([&](int tid) {
            auto& clk_out = *click_store.ts[size_t(tid)];
            auto& ev_out = *event_store.ts[size_t(tid)];
            auto& side_vec = thread_sides[size_t(tid)];
            side_vec.reserve(65536);
            std::vector<char> buf;

            while (true) {
                size_t ti = next_t.fetch_add(1, std::memory_order_relaxed);
                if (ti >= io_tasks.size()) break;
                const auto& task = io_tasks[ti];
                if (task.kind == 0) {
                    // Clicks
                    scan_pread_csv_simd<4>(task, buf, [&](const Slice f[4]) {
                        if (__builtin_expect(f[0].len > 0 && f[1].len > 0 && f[2].len > 0 && f[3].len > 0, 1)) {
                            uint32_t camp_idx = campaigns.lookup(f[2].ptr, f[2].len);
                            if (__builtin_expect(camp_idx != FastStringIdMap::NOT_FOUND, 1)) {
                                int64_t ts = 0;
                                if (__builtin_expect(parse_ts(f[3].ptr, f[3].len, ts, lut) &&
                                                     ts >= click_start_ms && ts < win_end_ms, 1)) {
                                    uint64_t ukey = encode_key64(f[1].ptr, f[1].len, arenas, tid);
                                    uint64_t pkey = links.resolve(ukey, arenas);
                                    uint64_t cid = encode_key64(f[0].ptr, f[0].len, arenas, tid);
                                    uint32_t pb = uint32_t(hash_key64(pkey, arenas) >> 55);
                                    uint64_t ts_camp = (uint64_t(ts - click_start_ms) << 20) | uint64_t(camp_idx);
                                    clk_out.push(pb, PersonClick{pkey, cid, ts_camp});
                                }
                            }
                        }
                    });
                } else {
                    // Events
                    scan_pread_csv_simd<7>(task, buf, [&](const Slice f[7]) {
                        if (__builtin_expect(f[0].len > 0, 1)) {
                            int64_t ingest_sec = 0;
                            if (__builtin_expect(parse_ingest_sec(f[3].ptr, f[3].len, ingest_sec, lut), 1)) {
                                uint64_t eid = encode_key64(f[0].ptr, f[0].len, arenas, tid);
                                uint32_t eb = uint32_t(hash_key64(eid, arenas) >> 55); // 512 event buckets!

                                int64_t irel_s = (ingest_sec - win_start_sec) + RawEvent::REL_BIAS;
                                uint32_t irel31 = (irel_s > 0 && irel_s <= RawEvent::REL_MAX) ? uint32_t(irel_s) : 0U;

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
                                    if (__builtin_expect(!parse_ts(f[2].ptr, f[2].len, ts, lut) ||
                                                         ts < win_start_ms || ts >= win_end_ms, 0)) {
                                        ok = false;
                                    } else if (__builtin_expect(ingest_sec * 1000LL - ts > DAY_MS, 0)) {
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
                                    if (__builtin_expect(etype == 1 && irel31 != 0 && ts_off < (1ULL << 31), 1)) {
                                        ev_out.push(eb, RawEvent{eid, ukey, RawEvent::pack_inline(true, irel31, uint32_t(ts_off))});
                                    } else {
                                        uint32_t sidx = uint32_t(side_vec.size());
                                        side_vec.push_back({ts, ingest_sec, price, cat_id});
                                        ev_out.push(eb, RawEvent{eid, ukey, RawEvent::pack_side(true, irel31, tid, sidx)});
                                    }
                                } else {
                                    if (__builtin_expect(irel31 != 0, 1)) {
                                        ev_out.push(eb, RawEvent{eid, 0, RawEvent::pack_inline(false, irel31, 0)});
                                    } else {
                                        uint32_t sidx = uint32_t(side_vec.size());
                                        side_vec.push_back({0, ingest_sec, 0, 0});
                                        ev_out.push(eb, RawEvent{eid, 0, RawEvent::pack_side(false, 0, tid, sidx)});
                                    }
                                }
                            }
                        }
                    });
                }
            }
            clk_out.flush_all();
            ev_out.flush_all();
        });
        close_pread_files(io_files);
    }
    double t_scan = now_ms();

    std::vector<UserDailyBuffers> thread_ud(static_cast<size_t>(ncpu));
    std::vector<std::vector<CampCell>> thread_cd(static_cast<size_t>(ncpu));

    bool use_compact = (win_end_ms - win_start_ms < (1LL << 32)) && (products.cat_names.size() < 256);
    if (use_compact) {
        run_dedup_and_sessionize<true>(
            event_store, thread_sides, click_store, links, products, campaigns, arenas, pool,
            win_start_ms, win_start_sec, click_start_ms, win_start_day, num_days, thread_ud, thread_cd, prof);
    } else {
        run_dedup_and_sessionize<false>(
            event_store, thread_sides, click_store, links, products, campaigns, arenas, pool,
            win_start_ms, win_start_sec, click_start_ms, win_start_day, num_days, thread_ud, thread_cd, prof);
    }
    double t_sess = now_ms();

    // Stage 4: Parallel Parquet Output
    std::string ud_dir = out_dir + "/user_daily";
    std::string cd_dir = out_dir + "/campaign_daily";
    mkdir_p(ud_dir);
    mkdir_p(cd_dir);

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

    size_t total_ud_rows = 0;
    for (int t = 0; t < ncpu; ++t) total_ud_rows += thread_ud[size_t(t)].num_rows;

    pool.run([&](int tid) {
        const auto& ud = thread_ud[size_t(tid)];
        if (ud.num_rows > 0 || (total_ud_rows == 0 && tid == 0)) {
            std::string path = ud_dir + "/part-" + std::to_string(tid) + ".parquet";
            pq::ColData data[6] = {
                {ud.uid_plain, ud.uid_sz, nullptr},
                {reinterpret_cast<const uint8_t*>(ud.day_vals), ud.num_rows * 4, nullptr},
                {reinterpret_cast<const uint8_t*>(ud.sess_vals), ud.num_rows * 4, nullptr},
                {reinterpret_cast<const uint8_t*>(ud.ev_vals), ud.num_rows * 4, nullptr},
                {reinterpret_cast<const uint8_t*>(ud.rev_vals), ud.num_rows * 8, nullptr},
                {ud.top_cat_plain, ud.top_cat_sz, ud.top_cat_bitmap},
            };
            pq::write_parquet_file(path.c_str(), UD_COLS, data, 6, int64_t(ud.num_rows));
        }

        if (tid == 0) {
            uint32_t camp_stride = campaigns.organic_idx + 1;
            size_t total_cells = size_t(num_days) * size_t(camp_stride);
            auto& agg = thread_cd[0];
            for (int t = 1; t < ncpu; ++t) {
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
            pq::write_parquet_file(path.c_str(), CD_COLS, data, 7, int64_t(day_v.size()));
        }
    });

    double t_end = now_ms();
    if (prof) {
        std::fprintf(stderr,
            "PROFILE (ms): init=%.1f links=%.1f chunks=%.1f scan=%.1f dedup+sess=%.1f write=%.1f TOTAL=%.1f\n",
            t_init - t0, t_links - t_init, t_chunks - t_links, t_scan - t_chunks,
            t_sess - t_scan, t_end - t_sess, t_end - t0);
    }
    ::_exit(0);
}

} // namespace SS_ENGINE_NS
