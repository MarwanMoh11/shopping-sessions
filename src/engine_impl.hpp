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
// Cgroup + Affinity-Aware CPU Detection & Thread Pinning
// ============================================================================
struct CpuTopology {
    int ncpu = 1;
    std::vector<int> cpus;
};

static CpuTopology detect_topology() {
    CpuTopology topo;
    int n = int(std::thread::hardware_concurrency());
    if (n < 1) n = 1;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (::sched_getaffinity(0, sizeof(set), &set) == 0) {
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (CPU_ISSET(c, &set)) topo.cpus.push_back(c);
        }
        int aff = int(topo.cpus.size());
        if (aff > 0 && aff < n) n = aff;
    }
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
    if (const char* e = std::getenv("NCPU")) {
        int v = std::atoi(e);
        if (v > 0) n = v;
    }
    if (n < 1) n = 1;
    if (n > 64) n = 64;
    topo.ncpu = n;
    return topo;
}

static void pin_current_thread(int tid, const std::vector<int>& cpus) {
#if defined(__linux__)
    if (!cpus.empty()) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpus[size_t(tid) % cpus.size()], &set);
        ::sched_setaffinity(0, sizeof(set), &set);
    }
#else
    (void)tid; (void)cpus;
#endif
}

// ============================================================================
// Persistent Low-Latency Pinned Thread Pool
// ============================================================================
class ThreadPool {
public:
    int nthreads = 1;

    explicit ThreadPool(const CpuTopology& topo)
        : nthreads(topo.ncpu < 1 ? 1 : (topo.ncpu > 64 ? 64 : topo.ncpu)), allowed_cpus(topo.cpus) {
        pin_current_thread(0, allowed_cpus);
        workers.reserve(static_cast<size_t>(nthreads - 1));
        for (int t = 1; t < nthreads; ++t) {
            workers.emplace_back([this, t]() {
                pin_current_thread(t, allowed_cpus);
                worker_loop(t);
            });
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
    std::vector<int> allowed_cpus;
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
                if (++spins > 80000) {
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
// Huge-Page Slab-Backed L1 Write-Combining Partitioned Store (Zero WC-Thrash!)
// ============================================================================
template <typename T, uint32_t CAP, int NUM_B, uint32_t WC_CAP, bool UseNT = false>
struct PartitionedStore {
    struct alignas(64) Block {
        Block* next;
        uint32_t count;
        alignas(64) T items[CAP];
    };
    static constexpr size_t SLAB_BYTES = 16ULL << 20; // 16 MB huge-page slabs

    struct alignas(64) ThreadState {
        alignas(64) T wc_buf[NUM_B][WC_CAP];
        uint8_t wc_cnt[NUM_B]{};
        Block* heads[NUM_B]{};
        size_t counts[NUM_B]{};
        std::vector<void*> slabs;
        char* slab = nullptr;
        size_t slab_pos = SLAB_BYTES;
        size_t slab_cap = SLAB_BYTES;

        SS_COLD_NOINLINE Block* alloc_block() {
            constexpr size_t bsz = (sizeof(Block) + 63) & ~size_t(63);
            if (__builtin_expect(slab_pos + bsz > slab_cap, 0)) {
                slab = static_cast<char*>(alloc_mmap(SLAB_BYTES, true));
                slabs.push_back(slab);
                slab_pos = 0;
                slab_cap = SLAB_BYTES;
            }
            auto* b = reinterpret_cast<Block*>(slab + slab_pos);
            slab_pos += bsz;
            b->next = nullptr;
            b->count = 0;
            return b;
        }

        void donate_slab(void* mem, size_t bytes) {
            if (!mem || bytes < sizeof(Block) * 4) return;
            slab = static_cast<char*>(mem);
            slab_pos = 0;
            slab_cap = bytes;
            if (bytes >= SLAB_BYTES) slabs.push_back(mem);
        }

        SS_ALWAYS_INLINE void flush_bucket_full(uint32_t b) {
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

        SS_COLD_NOINLINE void flush_bucket_partial(uint32_t b) {
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

        SS_ALWAYS_INLINE void push(uint32_t bucket, const T& val) {
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
            void* mem = alloc_mmap((sizeof(ThreadState) + 4095) & ~size_t(4095), false);
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
// Zero-Fault L2-Warm PREAD(256KB) Chunked File Scanner + 64B Pad
// ============================================================================
struct CsvFile {
    std::string path;
    int fd = -1;
    size_t file_size = 0;
    uint8_t kind = 0; // 0 = click, 1 = event, 2 = link
};

struct CsvTask {
    const CsvFile* file;
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

static ssize_t pread_full(int fd, char* dst, size_t len, size_t off) {
    size_t got = 0;
    while (got < len) {
        ssize_t n = ::pread(fd, dst + got, len - got, off_t(off + got));
        if (n < 0) {
            if (errno == EINTR) continue;
            return got > 0 ? ssize_t(got) : -1;
        }
        if (n == 0) break;
        got += size_t(n);
    }
    return ssize_t(got);
}

static void build_csv_tasks(
    const std::vector<CsvFile>& files,
    uint8_t kind_mask,
    int nthreads,
    std::vector<CsvTask>& tasks
) {
    size_t total_bytes = 0;
    for (const auto& f : files) {
        if (f.file_size > 0 && ((kind_mask >> f.kind) & 1U)) total_bytes += f.file_size;
    }
    if (total_bytes == 0) return;
    size_t chunk_sz = std::max<size_t>(262144, std::min<size_t>(4ULL << 20, total_bytes / size_t(nthreads * 8) + 1));
    for (const auto& f : files) {
        if (f.file_size == 0 || !((kind_mask >> f.kind) & 1U)) continue;
        size_t pos = 0;
        while (pos < f.file_size) {
            size_t end = std::min<size_t>(f.file_size, pos + chunk_sz);
            tasks.push_back({&f, pos, end, f.kind});
            pos = end;
        }
    }
}

// Reusable per-thread L2-warm pread buffer (256 KB + carry + 64B zero-pad).
struct ReadBuf {
    static constexpr size_t READ_CHUNK = 262144; // 256 KB L2-warm!
    char* data = nullptr;
    size_t cap = 0;

    void init(size_t initial = READ_CHUNK + 65536) {
        cap = initial;
        data = static_cast<char*>(alloc_mmap((cap + 4095) & ~size_t(4095), false));
    }

    SS_COLD_NOINLINE void ensure(size_t need) {
        if (need <= cap) return;
        size_t nc = std::max(cap * 2, need + 65536);
        char* nd = static_cast<char*>(alloc_mmap((nc + 4095) & ~size_t(4095), false));
        std::memcpy(nd, data, cap);
        ::munmap(data, (cap + 4095) & ~size_t(4095));
        data = nd;
        cap = nc;
    }
};

// Zero-fault pread + non-overlapping 64B block-SIMD CSV scanner!
// Handles: header skip (start==0), mid-file chunk alignment (zero wasted read!),
// arbitrary line lengths (including >= 4 KB), EOF without '\n', and 64B safe zero-padding!
template <int N, typename OnFields>
static SS_ALWAYS_INLINE void scan_pread_csv(const CsvTask& task, ReadBuf& rbuf, OnFields&& on_fields) {
    const CsvFile& file = *task.file;
    int fd = file.fd;
    bool close_after = false;
    if (__builtin_expect(fd < 0, 0)) {
        fd = ::open(file.path.c_str(), O_RDONLY);
        if (fd < 0) return;
        close_after = true;
    }

    const size_t fsz = file.file_size;
    const size_t limit = task.end;
    // If start > 0, start reading at start - 1 to check if previous byte was '\n' with ZERO extra syscall!
    size_t file_pos = (task.start > 0) ? (task.start - 1) : 0;
    size_t carry = 0;
    bool skip_first_line = true; // start==0 skips header; start>0 skips partial line if buf[0]!='\n'
    bool first_read = true;
    bool hit_limit = false;

    Slice f[N];
    uint64_t nls[64], cms[66], qms[66];

    while (true) {
        if (file_pos >= fsz) break;
        // Read up to READ_CHUNK (or just enough to finish straddling line after limit)
        size_t to_read = hit_limit ? std::min<size_t>(4096, fsz - file_pos)
                                   : std::min<size_t>(ReadBuf::READ_CHUNK, fsz - file_pos);
        if (__builtin_expect(carry + to_read + 128 > rbuf.cap, 0)) {
            rbuf.ensure(carry + to_read + 128);
        }
        ssize_t got = pread_full(fd, rbuf.data + carry, to_read, file_pos);
        if (got <= 0) break;
        size_t buf_file_off = file_pos - carry;
        file_pos += size_t(got);
        size_t total = carry + size_t(got);
        // 64B zero-pad guarantees Vec64::load & 8B padded encoders NEVER overread!
        std::memset(rbuf.data + total, 0, 64);

        size_t cur = 0;
        if (__builtin_expect(first_read, 0)) {
            first_read = false;
            if (task.start > 0) {
                if (rbuf.data[0] == '\n') {
                    cur = 1; // task.start begins right after '\n'
                    skip_first_line = false;
                } else {
                    cur = 1; // skip remainder of line belonging to previous chunk
                    skip_first_line = true;
                }
            }
        }

        if (__builtin_expect(skip_first_line, 0)) {
            const char* nl = find_nl_simd(rbuf.data + cur, rbuf.data + total);
            if (!nl) {
                if (file_pos >= limit) break;
                carry = 0;
                continue;
            }
            cur = size_t(nl - rbuf.data) + 1;
            skip_first_line = false;
            if (buf_file_off + cur >= limit) break;
        }

        // Process all complete lines starting strictly before `limit` inside `rbuf.data[cur..total]`!
        const size_t limit_in_buf = (limit > buf_file_off) ? std::min<size_t>(limit - buf_file_off, total) : 0;
        bool done = false;

        while (cur < limit_in_buf) {
            size_t avail = total - cur;
            size_t nblks = std::min<size_t>(64, avail >> 6);
            if (__builtin_expect(nblks == 0, 0)) break;

            const char* bptr = rbuf.data + cur;
            for (size_t bi = 0; bi < nblks; ++bi) {
                Vec64 v = Vec64::load(bptr + (bi << 6));
                nls[bi] = v.eq('\n');
                cms[bi] = v.eq(',');
                qms[bi] = v.eq('"');
            }
            cms[nblks] = cms[nblks + 1] = 0;
            qms[nblks] = qms[nblks + 1] = 0;

            size_t ls = 0;
            const size_t lrel = limit_in_buf - cur;
            for (size_t bi = 0; bi < nblks; ++bi) {
                uint64_t nm = nls[bi];
                while (nm) {
                    size_t nl_rel = (bi << 6) + size_t(__builtin_ctzll(nm));
                    nm &= nm - 1;
                    uint32_t L = uint32_t(nl_rel - ls);
                    if constexpr (N == 7) {
                        if (__builtin_expect(L >= 65U && L < 128U, 1)) {
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
                            if (__builtin_expect(extract7_hi64(bptr + ls, L, c0, c1, q0, q1, f), 1)) {
                                on_fields(f);
                            }
                        } else if (L > 0) {
                            if (parse_csv_line_scalar<7>(bptr + ls, bptr + nl_rel, f)) on_fields(f);
                        }
                    } else {
                        if (__builtin_expect(L > 0 && L < 64U, 1)) {
                            size_t sbi = ls >> 6;
                            unsigned sh = unsigned(ls & 63);
                            uint64_t c0, q0;
                            if (__builtin_expect(sh != 0, 1)) {
                                unsigned rsh = 64U - sh;
                                c0 = (cms[sbi] >> sh) | (cms[sbi + 1] << rsh);
                                q0 = (qms[sbi] >> sh) | (qms[sbi + 1] << rsh);
                            } else {
                                c0 = cms[sbi];
                                q0 = qms[sbi];
                            }
                            if (__builtin_expect(extract_short64<N>(bptr + ls, L, c0, q0, f), 1)) {
                                on_fields(f);
                            }
                        } else if (L >= 64U) {
                            if (parse_csv_line_scalar<N>(bptr + ls, bptr + nl_rel, f)) on_fields(f);
                        }
                    }
                    ls = nl_rel + 1;
                    if (ls >= lrel) {
                        cur += ls;
                        done = true;
                        break;
                    }
                }
                if (done) break;
            }
            if (done) break;
            if (__builtin_expect(ls == 0, 0)) {
                // Rare line >= 4 KB inside buffer
                const char* nl = find_nl_simd(bptr + (nblks << 6), rbuf.data + total);
                if (!nl) break; // need more bytes!
                if (parse_csv_line_scalar<N>(bptr, nl, f)) on_fields(f);
                cur = size_t(nl - rbuf.data) + 1;
                if (cur >= limit_in_buf) { done = true; break; }
            } else {
                cur += ls;
            }
        }

        if (done && buf_file_off + cur >= limit) break;

        // Finish any remaining complete lines in tail (< 64B) before carrying over partial line
        while (cur < limit_in_buf) {
            const char* line = rbuf.data + cur;
            const char* nl = find_nl_simd(line, rbuf.data + total);
            if (!nl) break;
            if (parse_csv_line_scalar<N>(line, nl, f)) on_fields(f);
            cur = size_t(nl - rbuf.data) + 1;
        }

        if (buf_file_off + cur >= limit) break;

        // Handle EOF without trailing '\n'
        if (file_pos >= fsz) {
            if (cur < total && buf_file_off + cur < limit) {
                if (parse_csv_line_scalar<N>(rbuf.data + cur, rbuf.data + total, f)) on_fields(f);
            }
            break;
        }

        // Carry partial line to next read; if we already reached `limit`, read small tail to finish it!
        carry = total - cur;
        if (carry > 0) std::memmove(rbuf.data, rbuf.data + cur, carry);
        if (file_pos >= limit) hit_limit = true;
    }

    if (close_after) ::close(fd);
}

// ============================================================================
// Small File Helper + L1D/L2-Hot Branchless Campaigns & Products Lookups
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
    ssize_t n = pread_full(fd, s.data(), s.size(), 0);
    ::close(fd);
    if (n >= 0) s.resize(size_t(n));
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

    SS_COLD_NOINLINE uint32_t find(const char* s, size_t len) const {
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
    // 8 KB L1D-hot direct LUT for c_1 .. c_4095!
    static constexpr uint32_t FAST_CAP = 4096;
    static constexpr uint16_t NF16 = 0xFFFFU;
    uint16_t fast_c[FAST_CAP];
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
                    if (parse_csv_line_scalar<2>(p, le, f)) {
                        rows.emplace_back(std::string(f[0].ptr, f[0].len), std::string(f[1].ptr, f[1].len));
                    }
                    p = eol ? (eol + 1) : end;
                }
            }
        }
        ids.reserve(rows.size() + 1);
        channels.reserve(rows.size() + 1);
        map.init(rows.size() + 1);
        std::fill(std::begin(fast_c), std::end(fast_c), NF16);
        for (uint32_t i = 0; i < uint32_t(rows.size()); ++i) {
            ids.push_back(std::move(rows[i].first));
            channels.push_back(std::move(rows[i].second));
            map.insert(ids[i], i);
            const std::string& s = ids[i];
            if (s.size() >= 3 && s.size() <= 6 && s[0] == 'c' && s[1] == '_' && s[2] >= '1' && s[2] <= '9') {
                uint32_t n = 0;
                bool ok = true;
                for (size_t k = 2; k < s.size(); ++k) {
                    unsigned d = uint8_t(s[k]) - uint8_t('0');
                    if (d > 9) { ok = false; break; }
                    n = n * 10 + d;
                }
                if (ok && n < FAST_CAP && i < NF16) fast_c[n] = uint16_t(i);
            }
        }
        organic_idx = uint32_t(ids.size());
        ids.emplace_back("organic");
        channels.emplace_back("organic");
    }

    // Branchless 1..4 digit c_XXXX lookup (requires >= 4 readable pad bytes after s+2!)
    SS_ALWAYS_INLINE uint32_t lookup_padded(const char* s, size_t len) const {
        if (__builtin_expect((len - 3) <= 3 && s[0] == 'c' && s[1] == '_', 1)) {
            uint32_t w;
            std::memcpy(&w, s + 2, 4);
            // Shift digits into top `ndig` bytes, fill low bytes with '0' (leading zeros!)
            uint32_t sh = uint32_t(6 - len) << 3; // 0 for len=6, 8 for 5, 16 for 4, 24 for 3
            uint32_t b0_ok = uint8_t(w) - uint8_t('1'); // first digit must be '1'..'9' (rejects c_0123!)
            uint32_t shifted = (w << sh) | (0x30303030U & ~((~0U) << sh));
            uint32_t bad = ((shifted & 0xF0F0F0F0U) ^ 0x30303030U) |
                           (((shifted + 0x06060606U) & 0xF0F0F0F0U) ^ 0x30303030U);
            if (__builtin_expect((bad == 0) & (b0_ok < 9U), 1)) {
                uint32_t d = shifted & 0x0F0F0F0FU;
                uint32_t p = (d * 10U + (d >> 8)) & 0x00FF00FFU;
                uint32_t n = (p & 0xFFU) * 100U + (p >> 16);
                if (__builtin_expect(n < FAST_CAP, 1)) {
                    uint16_t r = fast_c[n];
                    return __builtin_expect(r != NF16, 1) ? uint32_t(r) : FastStringIdMap::NOT_FOUND;
                }
            }
        }
        return map.find(s, len);
    }
};

struct Products {
    struct CatEntry {
        uint32_t total_len; // 4 + str_len
        char bytes[64];     // [4B len][chars] for len <= 60
    };
    std::vector<std::string> cat_names;
    std::vector<CatEntry> cat_pre;
    uint32_t unknown_cat_id = 0;
    size_t max_cat_len = 8;
    FastStringIdMap map;
    static constexpr uint32_t FAST_CAP = 65536;
    uint8_t fast_p8[FAST_CAP];
    bool use_fast8 = false;

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
                    if (parse_csv_line_scalar<2>(p, le, f)) {
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
        cat_pre.resize(cats.size());
        max_cat_len = 8;
        for (size_t i = 0; i < cats.size(); ++i) {
            auto c = cats[i];
            if (c.size() > max_cat_len) max_cat_len = c.size();
            cat_names.emplace_back(c);
            if (c.size() <= 60) {
                uint32_t l = uint32_t(c.size());
                cat_pre[i].total_len = 4 + l;
                std::memcpy(cat_pre[i].bytes, &l, 4);
                std::memcpy(cat_pre[i].bytes + 4, c.data(), l);
            } else {
                cat_pre[i].total_len = 0;
            }
        }

        FastStringIdMap cat_map;
        cat_map.init(cat_names.size());
        for (uint32_t i = 0; i < uint32_t(cat_names.size()); ++i) {
            cat_map.insert(cat_names[i], i);
            if (cat_names[i] == "unknown") unknown_cat_id = i;
        }

        use_fast8 = (cat_names.size() < 256);
        if (use_fast8) std::memset(fast_p8, uint8_t(unknown_cat_id), FAST_CAP);
        size_t non_fast = 0;
        std::vector<std::pair<std::string_view, uint32_t>> slow_rows;
        for (const auto& r : rows) {
            uint32_t cid = cat_map.find(r.second.data(), r.second.size());
            std::string_view s = r.first;
            bool is_fast = false;
            if (use_fast8 && s.size() >= 2 && s.size() <= 6 && s[0] == 'p' && s[1] >= '1' && s[1] <= '9') {
                uint32_t n = 0;
                bool ok = true;
                for (size_t k = 1; k < s.size(); ++k) {
                    unsigned d = uint8_t(s[k]) - uint8_t('0');
                    if (d > 9) { ok = false; break; }
                    n = n * 10 + d;
                }
                if (ok && n < FAST_CAP) { fast_p8[n] = uint8_t(cid); is_fast = true; }
            }
            if (!is_fast) { ++non_fast; slow_rows.push_back({s, cid}); }
        }
        map.init(non_fast + 1);
        for (const auto& sr : slow_rows) map.insert(sr.first, sr.second);
    }

    SS_ALWAYS_INLINE uint32_t lookup(const char* s, size_t len) const {
        if (__builtin_expect(use_fast8 && (len - 2) <= 4 && s[0] == 'p' && s[1] >= '1' && s[1] <= '9', 1)) {
            uint32_t n = 0;
            bool ok = true;
            for (size_t k = 1; k < len; ++k) {
                unsigned d = uint8_t(s[k]) - uint8_t('0');
                if (__builtin_expect(d > 9, 0)) { ok = false; break; }
                n = n * 10 + d;
            }
            if (__builtin_expect(ok && n < FAST_CAP, 1)) return fast_p8[n];
        }
        uint32_t r = map.find(s, len);
        return r == FastStringIdMap::NOT_FOUND ? unknown_cat_id : r;
    }
};

// ============================================================================
// Stage 1: 3-Barrier Union-by-Key Lock-Free Huge-Page Hash-DSU + 1-Word Bloom
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
    size_t bloom_word_mask = 0;
    bool has_fallback = false;

    // Recycled huge-page DSU buffer to hand directly to event_store!
    void* recycled_dsu = nullptr;
    size_t recycled_dsu_bytes = 0;

    static SS_ALWAYS_INLINE uint64_t bloom_mask3(uint64_t h) {
        return (1ULL << (h & 63U)) | (1ULL << ((h >> 6) & 63U)) | (1ULL << ((h >> 12) & 63U));
    }

    SS_ALWAYS_INLINE bool bloom_might_contain(uint64_t h) const {
        uint64_t word = bloom[(h >> 18) & bloom_word_mask];
        uint64_t m = bloom_mask3(h);
        return (word & m) == m;
    }

    SS_ALWAYS_INLINE void prefetch_bloom(uint64_t h) const {
        if (__builtin_expect(bloom != nullptr, 1)) {
            __builtin_prefetch(&bloom[(h >> 18) & bloom_word_mask], 0, 1);
        }
    }

    SS_ALWAYS_INLINE void prefetch_slot(uint64_t h) const {
        if (__builtin_expect(slots != nullptr, 1)) {
            __builtin_prefetch(&slots[size_t(h) & mask], 0, 1);
        }
    }

    SS_ALWAYS_INLINE uint64_t resolve_slot_only(uint64_t k, uint64_t h, const StringArenaPool& arenas) const {
        size_t idx = size_t(h) & mask;
        while (true) {
            uint64_t cur = slots[idx].key;
            if (cur == 0) return k;
            if (key_eq(cur, k, arenas)) return slots[idx].val;
            idx = (idx + 1) & mask;
        }
    }

    void build(const std::vector<CsvTask>& tasks, std::vector<ReadBuf>& rbufs, ThreadPool& pool, StringArenaPool& arenas) {
        if (tasks.empty()) return;

        struct Edge { uint64_t a, b; };
        std::vector<std::vector<Edge>> thread_edges(static_cast<size_t>(pool.nthreads));
        std::atomic<size_t> next_t{0};
        std::atomic<bool> any_fallback{false};

        // Pass 1: Parallel zero-fault pread single-word SIMD scan of links/*.csv
        pool.run([&](int tid) {
            auto& edges = thread_edges[size_t(tid)];
            edges.reserve(65536);
            auto& rbuf = rbufs[size_t(tid)];
            bool local_fb = false;
            while (true) {
                size_t ti = next_t.fetch_add(1, std::memory_order_relaxed);
                if (ti >= tasks.size()) break;
                scan_pread_csv<2>(tasks[ti], rbuf, [&](const Slice f[2]) SS_LAMBDA_INLINE {
                    if (__builtin_expect(f[0].len > 0 && f[1].len > 0, 1)) {
                        uint64_t ka = encode_key64_padded(f[0].ptr, f[0].len, arenas, tid);
                        uint64_t kb = encode_key64_padded(f[1].ptr, f[1].len, arenas, tid);
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
            // UNION-BY-KEY Lock-Free Hash-DSU:
            //   - Always links larger key -> smaller key (strict total order => acyclic!)
            //   - Root (parent == 0) IS AUTOMATICALLY the exact byte-wise min_key of the component!
            //   - ZERO inv_min array, ZERO Pass 3!
            //   - Exact total_nonmin == total successful union CAS ops in Pass 2!
            struct DsuSlot {
                uint64_t key;
                uint32_t parent; // 0 = root (min_key!), r + 1 = parent slot r
                uint32_t _pad;
            };
            // Size with fastrange: 2.5x total_edges rounded to 2 MB multiple -> ~50% typical load, <=80% worst-case!
            size_t min_slots = total_edges * 2 + (total_edges >> 1) + 64;
            size_t dsu_bytes = ((min_slots * sizeof(DsuSlot)) + (2ULL << 20) - 1) & ~((2ULL << 20) - 1);
            size_t hcap = dsu_bytes / sizeof(DsuSlot);
            auto* dsu = static_cast<DsuSlot*>(alloc_mmap(dsu_bytes, true));

            auto slot_of = [hcap](uint64_t k) SS_LAMBDA_INLINE -> size_t {
                return size_t((static_cast<unsigned __int128>(mix64(k)) * hcap) >> 64);
            };
            auto insert_key = [&](uint64_t k) SS_LAMBDA_INLINE -> uint32_t {
                size_t s = slot_of(k);
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
                    if (++s == hcap) s = 0;
                }
            };
            auto find_root = [&](uint32_t x) SS_LAMBDA_INLINE -> uint32_t {
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

            // Pass 2: Parallel 16-ahead prefetched Hash-Insert + Lock-Free Union-by-Key!
            std::vector<size_t> t_unions(static_cast<size_t>(pool.nthreads), 0);
            pool.run([&](int tid) {
                const auto& ev = thread_edges[size_t(tid)];
                const size_t n = ev.size();
                size_t local_unions = 0;
                for (size_t i = 0; i < n; ++i) {
                    if (__builtin_expect(i + 16 < n, 1)) {
                        __builtin_prefetch(&dsu[slot_of(ev[i + 16].a)], 1, 1);
                        __builtin_prefetch(&dsu[slot_of(ev[i + 16].b)], 1, 1);
                    }
                    uint32_t u = insert_key(ev[i].a);
                    uint32_t v = insert_key(ev[i].b);
                    while (u != v) {
                        u = find_root(u);
                        v = find_root(v);
                        if (u == v) break;
                        // Union-by-Key: ALWAYS link LARGER key -> SMALLER key!
                        if (dsu[u].key < dsu[v].key) std::swap(u, v);
                        uint32_t exp = 0;
                        if (std::atomic_ref<uint32_t>(dsu[u].parent).compare_exchange_weak(
                                exp, v + 1, std::memory_order_relaxed)) {
                            ++local_unions;
                            break;
                        }
                    }
                }
                t_unions[size_t(tid)] = local_unions;
            });

            size_t total_nonmin = 0;
            for (size_t u : t_unions) total_nonmin += u;
            if (total_nonmin == 0) {
                recycled_dsu = dsu;
                recycled_dsu_bytes = dsu_bytes;
                return;
            }

            cap = 64;
            while (cap < total_nonmin * 2) cap <<= 1;
            mask = cap - 1;
            slots = static_cast<Slot*>(alloc_mmap(cap * sizeof(Slot), true));

            // 1-word 3-bit blocked Bloom (~12-16 bits/key -> 1 MB at 20M!)
            size_t bwords = 4096;
            while (bwords * 6 < total_nonmin && bwords < (1ULL << 20)) bwords <<= 1;
            bloom_word_mask = bwords - 1;
            bloom = static_cast<uint64_t*>(alloc_mmap(bwords * sizeof(uint64_t), false));

            // Pass 3 (FINAL!): Compress non-root paths & populate compact LinkMap::slots + Bloom!
            pool.run([&](int tid) {
                size_t s = (hcap * size_t(tid)) / size_t(pool.nthreads);
                size_t e = (hcap * size_t(tid + 1)) / size_t(pool.nthreads);
                for (size_t i = s; i < e; ++i) {
                    uint32_t p1 = std::atomic_ref<uint32_t>(dsu[i].parent).load(std::memory_order_relaxed);
                    if (p1 == 0) continue; // empty OR min-key root -> skip!
                    uint32_t r = p1 - 1;
                    while (true) {
                        uint32_t np1 = std::atomic_ref<uint32_t>(dsu[r].parent).load(std::memory_order_relaxed);
                        if (np1 == 0) break;
                        r = np1 - 1;
                    }
                    uint64_t k = dsu[i].key;
                    uint64_t mk = dsu[r].key;
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

            recycled_dsu = dsu;
            recycled_dsu_bytes = dsu_bytes;
        } else {
            cap = 64;
            while (cap < total_edges * 4) cap <<= 1;
            mask = cap - 1;
            slots = static_cast<Slot*>(alloc_mmap(cap * sizeof(Slot), false));

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
// Ultra-Compact 16-Byte RawEvent16 (Zero mix64 in Dedup + 100% L2-Hot Buckets!)
// ============================================================================
static constexpr int EV_BUCKETS = 512;     // 512b x WC_CAP=4 x 16B = 32 KB L1D! Bucket ~672KB+256KB = L2-hot!
static constexpr int PERSON_BUCKETS = 512; // 512b x WC_CAP=4 x 16B = 32 KB L1D! Bucket ~600KB+16KB pht = L2-hot!

struct SideEvent {
    uint64_t eid_key64;
    uint64_t user_key64;
    int64_t ts_ms;
    int64_t ingest_ms;
    uint32_t price;
    uint32_t cat_id;
};

struct RawEvent16 {
    uint64_t w0;
    uint64_t w1;

    // w0 layout:
    //   bit 63:     is_side
    //   bit 62:     valid
    //   bits 61..41 (21b): ingest_rel21 = (ingest_sec - win_start_sec) + ING_BIAS (0 = read exact ingest_ms from SideEvent!)
    //   bit 40:     if !is_side: delta_hi1 (bit 16 of delta = (ingest_sec - ts_sec) + 8, in [0..131071])
    //               if  is_side: 0
    //   bit 39:     eid_in_side (0 = exact 39-bit mix48 remainder in bits 38..0, 1 = fallback eid_key64 in SideEvent!)
    //   bits 38..0  (39b): eid_rem39 (bottom 39 bits of mix48(eid48), OR bottom 39 bits of hash_key64 if eid_in_side==1)
    //
    // w1 layout:
    //   if !is_side: [bits 63..16: uid48 (48b)] [bits 15..0: delta_lo16 (16b)]
    //   if  is_side: (tid << 26) | side_idx
    static constexpr int64_t ING_BIAS = 1LL << 19;      // -524,288s (-6d) to +1,572,863s (+18.2d)
    static constexpr int64_t ING_MAX  = (1LL << 21) - 1;
    static constexpr uint64_t EID40_MASK = (1ULL << 40) - 1ULL;
    static constexpr uint64_t REM39_MASK = (1ULL << 39) - 1ULL;

    SS_ALWAYS_INLINE bool is_side() const { return (w0 >> 63) != 0; }
    SS_ALWAYS_INLINE bool is_valid() const { return ((w0 >> 62) & 1ULL) != 0; }
    SS_ALWAYS_INLINE uint32_t ingest_rel() const { return uint32_t((w0 >> 41) & 0x1FFFFFULL); }
    SS_ALWAYS_INLINE uint64_t eid40() const { return w0 & EID40_MASK; }
    SS_ALWAYS_INLINE bool eid_in_side() const { return ((w0 >> 39) & 1ULL) != 0; }

    static SS_ALWAYS_INLINE RawEvent16 make_inline(uint32_t ing_rel21, uint32_t delta17, uint64_t rem39, uint64_t uid48) {
        uint64_t a = (1ULL << 62) | (uint64_t(ing_rel21) << 41) | (uint64_t((delta17 >> 16) & 1U) << 40) | rem39;
        uint64_t b = (uid48 << 16) | uint64_t(delta17 & 0xFFFFU);
        return {a, b};
    }

    static SS_ALWAYS_INLINE RawEvent16 make_side(bool valid, uint32_t ing_rel21, bool eid_side, uint64_t rem39, int tid, uint32_t sidx) {
        uint64_t a = (1ULL << 63) | (uint64_t(valid) << 62) | (uint64_t(ing_rel21) << 41) |
                     (uint64_t(eid_side) << 39) | rem39;
        uint64_t b = (uint64_t(tid) << 26) | uint64_t(sidx);
        return {a, b};
    }
};

static SS_COLD_NOINLINE int64_t get_exact_ingest_ms_slow(
    const RawEvent16& ev,
    int64_t win_start_sec,
    const std::vector<std::vector<SideEvent>>& sides
) {
    uint32_t ir = ev.ingest_rel();
    if (ir != 0) return (win_start_sec + int64_t(ir) - RawEvent16::ING_BIAS) * 1000LL;
    return sides[ev.w1 >> 26][ev.w1 & 0x03FFFFFFU].ingest_ms;
}

static SS_ALWAYS_INLINE bool ingest_strictly_earlier(
    const RawEvent16& a,
    const RawEvent16& b,
    int64_t win_start_sec,
    const std::vector<std::vector<SideEvent>>& sides
) {
    uint32_t ia = a.ingest_rel();
    uint32_t ib = b.ingest_rel();
    if (__builtin_expect((ia != 0) & (ib != 0), 1)) return ia < ib;
    return get_exact_ingest_ms_slow(a, win_start_sec, sides) < get_exact_ingest_ms_slow(b, win_start_sec, sides);
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
    uint64_t ts_camp;
    uint64_t click_id;
    SS_ALWAYS_INLINE uint64_t ts_off() const { return ts_camp >> 20; }
    SS_ALWAYS_INLINE uint32_t camp_idx() const { return uint32_t(ts_camp & 0xFFFFFULL); }
};

// Split & transposed campaign accumulators:
//   cd_assist[camp * num_days + day]: 4B/cell (112 KB total -> 14 days = 1 cache line!)
//   cd_cred  [camp * num_days + day]: 16B/cell (448 KB total!)
struct CredCell {
    uint32_t sessions;
    uint32_t purchasing_sessions;
    uint64_t revenue_cents;
};

struct CampState {
    uint32_t count;
    uint32_t enter_sess;
    uint32_t active_pos;
};

// ============================================================================
// Bounds-Safe 2-Slab-Recycled UserDailyBuffers (Zero Faults + Zero Grows at 60M!)
// ============================================================================
struct UserDailyBuffers {
    uint8_t* uid_plain = nullptr;
    size_t uid_sz = 0, uid_cap = 0;

    int32_t* day_vals = nullptr;
    int32_t* sess_vals = nullptr;
    int32_t* ev_vals = nullptr;
    int64_t* rev_vals = nullptr;
    uint8_t* top_cat_bitmap = nullptr;
    size_t num_rows = 0, row_cap = 0;

    uint8_t* top_cat_plain = nullptr;
    size_t top_cat_sz = 0, top_cat_cap = 0;

    void init(size_t est_rows, size_t max_cat_len, const std::vector<void*>& dead_slabs, size_t slab_bytes) {
        if (dead_slabs.size() >= 2 && slab_bytes >= (16ULL << 20)) {
            // Slab 0 (16 MB): fixed numeric columns for up to 786,432 rows (handles 60M Final with 0 grows!)
            char* s0 = static_cast<char*>(dead_slabs[0]);
            size_t rcap = 786432; // 786K * 20.125 B = 15.83 MB < 16 MB!
            day_vals       = reinterpret_cast<int32_t*>(s0); s0 += rcap * 4;
            sess_vals      = reinterpret_cast<int32_t*>(s0); s0 += rcap * 4;
            ev_vals        = reinterpret_cast<int32_t*>(s0); s0 += rcap * 4;
            rev_vals       = reinterpret_cast<int64_t*>(s0); s0 += rcap * 8;
            size_t bm_sz   = ((rcap >> 3) + 64) & ~size_t(63);
            top_cat_bitmap = reinterpret_cast<uint8_t*>(s0);
            std::memset(top_cat_bitmap, 0, bm_sz);
            row_cap        = rcap;
            // Slab 1 (16 MB): string columns (12 MB uid_plain + 4 MB top_cat_plain)
            char* s1 = static_cast<char*>(dead_slabs[1]);
            uid_cap        = (slab_bytes * 3) / 4;
            top_cat_cap    = slab_bytes - uid_cap;
            uid_plain      = reinterpret_cast<uint8_t*>(s1);
            top_cat_plain  = reinterpret_cast<uint8_t*>(s1 + uid_cap);
            return;
        }
        if (!dead_slabs.empty() && slab_bytes >= (16ULL << 20)) {
            char* s = static_cast<char*>(dead_slabs[0]);
            size_t rcap = 393216;
            day_vals       = reinterpret_cast<int32_t*>(s); s += rcap * 4;
            sess_vals      = reinterpret_cast<int32_t*>(s); s += rcap * 4;
            ev_vals        = reinterpret_cast<int32_t*>(s); s += rcap * 4;
            rev_vals       = reinterpret_cast<int64_t*>(s); s += rcap * 8;
            size_t bm_sz   = ((rcap >> 3) + 64) & ~size_t(63);
            top_cat_bitmap = reinterpret_cast<uint8_t*>(s); s += bm_sz;
            std::memset(top_cat_bitmap, 0, bm_sz);
            size_t rem     = slab_bytes - size_t(s - static_cast<char*>(dead_slabs[0]));
            uid_cap        = (rem * 3) / 4;
            top_cat_cap    = rem - uid_cap;
            uid_plain      = reinterpret_cast<uint8_t*>(s);
            top_cat_plain  = reinterpret_cast<uint8_t*>(s + uid_cap);
            row_cap        = rcap;
            return;
        }
        size_t rcap = std::max<size_t>(est_rows * 2, 131072);
        row_cap = rcap;
        uid_cap = rcap * 20 + 4096;
        top_cat_cap = rcap * (4 + max_cat_len) + 4096;
        day_vals = static_cast<int32_t*>(alloc_mmap(rcap * 4, true));
        sess_vals = static_cast<int32_t*>(alloc_mmap(rcap * 4, true));
        ev_vals = static_cast<int32_t*>(alloc_mmap(rcap * 4, true));
        rev_vals = static_cast<int64_t*>(alloc_mmap(rcap * 8, true));
        top_cat_bitmap = static_cast<uint8_t*>(alloc_mmap((rcap >> 3) + 64, false));
        uid_plain = static_cast<uint8_t*>(alloc_mmap(uid_cap, true));
        top_cat_plain = static_cast<uint8_t*>(alloc_mmap(top_cat_cap, true));
    }

    SS_COLD_NOINLINE void grow_rows() {
        size_t new_cap = row_cap * 2;
        auto* nd = static_cast<int32_t*>(alloc_mmap(new_cap * 4, true));
        auto* ns = static_cast<int32_t*>(alloc_mmap(new_cap * 4, true));
        auto* ne = static_cast<int32_t*>(alloc_mmap(new_cap * 4, true));
        auto* nr = static_cast<int64_t*>(alloc_mmap(new_cap * 8, true));
        auto* nb = static_cast<uint8_t*>(alloc_mmap((new_cap >> 3) + 64, false));
        std::memcpy(nd, day_vals, num_rows * 4);
        std::memcpy(ns, sess_vals, num_rows * 4);
        std::memcpy(ne, ev_vals, num_rows * 4);
        std::memcpy(nr, rev_vals, num_rows * 8);
        std::memcpy(nb, top_cat_bitmap, (num_rows + 7) >> 3);
        day_vals = nd; sess_vals = ns; ev_vals = ne; rev_vals = nr; top_cat_bitmap = nb;
        row_cap = new_cap;
    }

    SS_COLD_NOINLINE void grow_uid(size_t need) {
        size_t new_cap = std::max(uid_cap * 2, uid_sz + need + 65536);
        auto* np = static_cast<uint8_t*>(alloc_mmap(new_cap, true));
        if (uid_sz) std::memcpy(np, uid_plain, uid_sz);
        uid_plain = np;
        uid_cap = new_cap;
    }

    SS_COLD_NOINLINE void grow_cat(size_t need) {
        size_t new_cap = std::max(top_cat_cap * 2, top_cat_sz + need + 65536);
        auto* np = static_cast<uint8_t*>(alloc_mmap(new_cap, true));
        if (top_cat_sz) std::memcpy(np, top_cat_plain, top_cat_sz);
        top_cat_plain = np;
        top_cat_cap = new_cap;
    }

    SS_ALWAYS_INLINE void push_row_pre(
        const char* uid_pre, uint32_t uid_pre_len,
        int32_t day, int32_t sessions, int32_t events, int64_t rev,
        int32_t best_cat, const Products& products
    ) {
        if (__builtin_expect(num_rows >= row_cap, 0)) grow_rows();
        if (__builtin_expect(uid_sz + uid_pre_len + 16 > uid_cap, 0)) grow_uid(uid_pre_len + 16);
        if (__builtin_expect(uid_pre_len <= 16, 1)) {
            std::memcpy(uid_plain + uid_sz, uid_pre, 16); // single 16B register store!
        } else {
            std::memcpy(uid_plain + uid_sz, uid_pre, uid_pre_len);
        }
        uid_sz += uid_pre_len;
        size_t r = num_rows++;
        day_vals[r] = day;
        sess_vals[r] = sessions;
        ev_vals[r] = events;
        rev_vals[r] = rev;
        if (__builtin_expect(best_cat >= 0, 0)) {
            top_cat_bitmap[r >> 3] |= uint8_t(1U << (r & 7));
            const auto& cp = products.cat_pre[size_t(best_cat)];
            if (__builtin_expect(cp.total_len > 0, 1)) {
                if (__builtin_expect(top_cat_sz + cp.total_len > top_cat_cap, 0)) grow_cat(cp.total_len);
                std::memcpy(top_cat_plain + top_cat_sz, cp.bytes, cp.total_len);
                top_cat_sz += cp.total_len;
            } else {
                const std::string& cs = products.cat_names[size_t(best_cat)];
                uint32_t len = uint32_t(cs.size());
                if (__builtin_expect(top_cat_sz + 4 + len > top_cat_cap, 0)) grow_cat(4 + len);
                std::memcpy(top_cat_plain + top_cat_sz, &len, 4);
                std::memcpy(top_cat_plain + top_cat_sz + 4, cs.data(), len);
                top_cat_sz += 4 + len;
            }
        }
    }
};

// ============================================================================
// Fast Zero-Mispredict Event Sorting (Branchless Small + 1-Pass MSD Bucket Sort!)
// ============================================================================
static SS_ALWAYS_INLINE void cswap_u64(uint64_t& a, uint64_t& b) {
    uint64_t lo = a < b ? a : b;
    uint64_t hi = a < b ? b : a;
    a = lo; b = hi;
}

static SS_ALWAYS_INLINE void sort_small_u64(uint64_t* a, uint32_t n) {
    if (n <= 1) return;
    if (n == 2) { cswap_u64(a[0], a[1]); return; }
    if (n == 3) { cswap_u64(a[0], a[1]); cswap_u64(a[1], a[2]); cswap_u64(a[0], a[1]); return; }
    if (n == 4) {
        cswap_u64(a[0], a[1]); cswap_u64(a[2], a[3]);
        cswap_u64(a[0], a[2]); cswap_u64(a[1], a[3]);
        cswap_u64(a[1], a[2]);
        return;
    }
    for (uint32_t i = 1; i < n; ++i) {
        uint64_t x = a[i];
        uint32_t j = i;
        while (j > 0 && a[j - 1] > x) { a[j] = a[j - 1]; --j; }
        a[j] = x;
    }
}

static SS_COLD_NOINLINE void radix_sort_ts_u64(uint64_t* arr, uint32_t n, std::vector<uint64_t>& tmp) {
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

// 1-pass top-8-bit MSD bucket sort (256 buckets of ~2.33h for 14d windows!) -> ~2-6 items/bucket!
// Splits 25..65535 events in 1 pass instead of 6 passes, crushing std::sort branch mispredicts!
static SS_ALWAYS_INLINE void sort_events_compact_u64(uint64_t* arr, uint32_t n, uint32_t shift_top, std::vector<uint64_t>& tmp) {
    if (n <= 20) {
        sort_small_u64(arr, n);
        return;
    }
    if (__builtin_expect(n >= 65536, 0)) {
        radix_sort_ts_u64(arr, n, tmp);
        return;
    }
    if (tmp.size() < n) tmp.resize(n);
    uint16_t cnt[256]{};
    uint32_t bmin = 255, bmax = 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t b = uint32_t((arr[i] >> shift_top) & 255U);
        cnt[b]++;
        if (b < bmin) bmin = b;
        if (b > bmax) bmax = b;
    }
    if (__builtin_expect(bmin == bmax, 0)) {
        std::sort(arr, arr + n);
        return;
    }
    uint16_t off[256];
    uint16_t acc = 0;
    for (uint32_t b = bmin; b <= bmax; ++b) {
        off[b] = acc;
        acc = uint16_t(acc + cnt[b]);
    }
    uint64_t* dst = tmp.data();
    for (uint32_t i = 0; i < n; ++i) {
        uint64_t v = arr[i];
        dst[off[(v >> shift_top) & 255U]++] = v;
    }
    uint32_t start = 0;
    for (uint32_t b = bmin; b <= bmax; ++b) {
        uint32_t c = cnt[b];
        if (c == 0) continue;
        uint64_t* p = dst + start;
        if (c <= 20) {
            sort_small_u64(p, c);
        } else {
            std::sort(p, p + c);
        }
        start += c;
    }
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
// Stage 2C + Stage 3 + Stage 4 FUSED: Zero-mix64 L2-Hot Dedup -> Sessionize -> writev()
// ============================================================================
template <bool Compact>
bool run_dedup_sessionize_and_write(
    PartitionedStore<RawEvent16, 256, EV_BUCKETS, 4, false>& event_store,
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
    PartitionedStore<PEv, 256, PERSON_BUCKETS, 4, false> person_ev_store;
    person_ev_store.init(pool.nthreads);

    auto now_ms = []() {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    double td0 = now_ms();

    // Compute exact top-8-bit shift for 1-pass MSD sort of Compact ts_off!
    uint32_t msd_shift = 55; // default (ts_off >> 23 in bits 32..63 -> shift 55)
    {
        uint64_t dur = (win_end_ms > win_start_ms) ? uint64_t(win_end_ms - win_start_ms) : 1ULL;
        uint32_t bits = 64U - uint32_t(__builtin_clzll(dur));
        msd_shift = (bits > 8U) ? (32U + bits - 8U) : 32U;
    }

    // Stage 2C: ZERO-mix64 L2-Hot Fingerprinted Dedup on RawEvent16 + 2-Stage Bloom Ring!
    using EvBlock = typename PartitionedStore<RawEvent16, 256, EV_BUCKETS, 4, false>::Block;
    std::atomic<int> next_eb{0};
    pool.run([&](int tid) {
        struct DedupSlot8 { uint32_t fp; uint32_t handle1; };
        std::vector<uint32_t> ht4;
        std::vector<DedupSlot8> ht8;
        std::vector<EvBlock*> blks;
        ht4.reserve(131072);
        blks.reserve(512);

        auto& out_ts = *person_ev_store.ts[size_t(tid)];
        const bool links_active = (links.cap > 0);
        const bool links_has_bloom = (links.bloom != nullptr);

        struct RingItem {
            uint64_t ukey;
            uint64_t uh;
            uint64_t packed; // Compact: packed; Wide: pointer index (handled separately)
            const RawEvent16* ev;
            bool probe_ht;
        };
        constexpr size_t RING_SZ = 64;
        constexpr size_t RING_MASK = RING_SZ - 1;
        RingItem ring[RING_SZ];

        auto emit_resolved = [&](const RingItem& r) SS_LAMBDA_INLINE {
            uint64_t pkey = r.probe_ht ? links.resolve_slot_only(r.ukey, r.uh, arenas) : r.ukey;
            uint64_t ph = (pkey == r.ukey) ? r.uh : hash_key64(pkey, arenas);
            uint32_t pb = uint32_t(ph >> 55);
            if constexpr (Compact) {
                out_ts.push(pb, CompactPersonEvent{pkey, r.packed});
            } else {
                const RawEvent16& ev = *r.ev;
                if (__builtin_expect(!ev.is_side(), 1)) {
                    int64_t ing_rel_sec = int64_t(ev.ingest_rel()) - RawEvent16::ING_BIAS;
                    int64_t delta_sec = int64_t(((ev.w0 >> 24) & 0x10000ULL) | (ev.w1 & 0xFFFFULL)) - 8LL;
                    int64_t ts_ms = win_start_ms + (ing_rel_sec - delta_sec) * 1000LL;
                    out_ts.push(pb, WidePersonEvent{pkey, ts_ms, 0, 0});
                } else {
                    const SideEvent& se = sides[ev.w1 >> 26][ev.w1 & 0x03FFFFFFU];
                    out_ts.push(pb, WidePersonEvent{pkey, se.ts_ms, se.price, se.cat_id});
                }
            }
        };

        auto unpack_ukey_and_packed = [&](const RawEvent16& ev, uint64_t& ukey, uint64_t& packed) SS_LAMBDA_INLINE {
            if (__builtin_expect(!ev.is_side(), 1)) {
                ukey = decode_uid48_to_key64(ev.w1 >> 16);
                if constexpr (Compact) {
                    int64_t ing_rel_sec = int64_t(ev.ingest_rel()) - RawEvent16::ING_BIAS;
                    int64_t delta_sec = int64_t(((ev.w0 >> 24) & 0x10000ULL) | (ev.w1 & 0xFFFFULL)) - 8LL;
                    uint64_t ts_off = uint64_t((ing_rel_sec - delta_sec) * 1000LL);
                    packed = ts_off << 32;
                }
            } else {
                const SideEvent& se = sides[ev.w1 >> 26][ev.w1 & 0x03FFFFFFU];
                ukey = se.user_key64;
                if constexpr (Compact) {
                    packed = (uint64_t(se.ts_ms - win_start_ms) << 32) | (uint64_t(se.price) << 8) | uint64_t(se.cat_id);
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

            // Loop 1: ZERO-mix64 dedup touching ONLY w0 (8 bytes)!
            if (__builtin_expect(nblks < 4095, 1)) {
                ht4.assign(hcap, 0);
                uint32_t* htp = ht4.data();
                for (uint32_t bi = 0; bi < nblks; ++bi) {
                    EvBlock* blk = blks[bi];
                    const uint32_t cnt = blk->count;
                    RawEvent16* items = blk->items;
                    for (uint32_t i = 0; i < cnt; ++i) {
                        if (__builtin_expect(i + 8 < cnt, 1)) {
                            __builtin_prefetch(&htp[size_t(items[i + 8].w0) & hmask], 1, 1);
                        }
                        RawEvent16& ev = items[i];
                        uint64_t eid40 = ev.eid40();
                        uint32_t fp12 = uint32_t((eid40 >> 27) & 0xFFFULL) << 20;
                        size_t slot = size_t(eid40) & hmask;
                        while (true) {
                            uint32_t entry = htp[slot];
                            if (__builtin_expect(entry == 0, 1)) {
                                htp[slot] = fp12 | (((bi << 8) | i) + 1);
                                break;
                            }
                            if (__builtin_expect((entry & 0xFFF00000U) == fp12, 0)) {
                                uint32_t prev = (entry & 0x000FFFFFU) - 1;
                                RawEvent16& existing = blks[prev >> 8]->items[prev & 255U];
                                if (existing.eid40() == eid40) {
                                    bool same = true;
                                    if (__builtin_expect(ev.eid_in_side(), 0)) {
                                        const SideEvent& sa = sides[existing.w1 >> 26][existing.w1 & 0x03FFFFFFU];
                                        const SideEvent& sb = sides[ev.w1 >> 26][ev.w1 & 0x03FFFFFFU];
                                        same = key_eq(sa.eid_key64, sb.eid_key64, arenas);
                                    }
                                    if (same) {
                                        if (ingest_strictly_earlier(ev, existing, win_start_sec, sides)) {
                                            existing = ev;
                                        }
                                        ev.w0 = 0; // clears valid bit!
                                        break;
                                    }
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
                    RawEvent16* items = blk->items;
                    for (uint32_t i = 0; i < cnt; ++i) {
                        RawEvent16& ev = items[i];
                        uint64_t eid40 = ev.eid40();
                        uint32_t fp = uint32_t(eid40 >> 8);
                        size_t slot = size_t(eid40) & hmask;
                        while (true) {
                            uint32_t h1 = ht8[slot].handle1;
                            if (__builtin_expect(h1 == 0, 1)) {
                                ht8[slot] = {fp, ((bi << 8) | i) + 1};
                                break;
                            }
                            if (__builtin_expect(ht8[slot].fp == fp, 0)) {
                                uint32_t prev = h1 - 1;
                                RawEvent16& existing = blks[prev >> 8]->items[prev & 255U];
                                if (existing.eid40() == eid40) {
                                    bool same = true;
                                    if (__builtin_expect(ev.eid_in_side(), 0)) {
                                        const SideEvent& sa = sides[existing.w1 >> 26][existing.w1 & 0x03FFFFFFU];
                                        const SideEvent& sb = sides[ev.w1 >> 26][ev.w1 & 0x03FFFFFFU];
                                        same = key_eq(sa.eid_key64, sb.eid_key64, arenas);
                                    }
                                    if (same) {
                                        if (ingest_strictly_earlier(ev, existing, win_start_sec, sides)) {
                                            existing = ev;
                                        }
                                        ev.w0 = 0;
                                        break;
                                    }
                                }
                            }
                            slot = (slot + 1) & hmask;
                        }
                    }
                }
            }

            // Loop 2 (100% L2 HIT!): Resolve LinkMap & scatter CompactPersonEvent!
            if (__builtin_expect(!links_active, 0)) {
                for (uint32_t bi = 0; bi < nblks; ++bi) {
                    const EvBlock* blk = blks[bi];
                    const uint32_t cnt = blk->count;
                    const RawEvent16* items = blk->items;
                    for (uint32_t i = 0; i < cnt; ++i) {
                        if (__builtin_expect(!items[i].is_valid(), 0)) continue;
                        uint64_t ukey, packed = 0;
                        unpack_ukey_and_packed(items[i], ukey, packed);
                        uint64_t uh = hash_key64(ukey, arenas);
                        emit_resolved({ukey, uh, packed, &items[i], false});
                    }
                }
            } else {
                size_t pushed = 0;
                for (uint32_t bi = 0; bi < nblks; ++bi) {
                    const EvBlock* blk = blks[bi];
                    const uint32_t cnt = blk->count;
                    const RawEvent16* items = blk->items;
                    for (uint32_t i = 0; i < cnt; ++i) {
                        if (__builtin_expect(!items[i].is_valid(), 0)) continue;
                        uint64_t ukey, packed = 0;
                        unpack_ukey_and_packed(items[i], ukey, packed);
                        uint64_t uh = hash_key64(ukey, arenas);
                        ring[pushed & RING_MASK] = {ukey, uh, packed, &items[i], false};
                        if (links_has_bloom) links.prefetch_bloom(uh);

                        if (pushed >= 16) {
                            auto& r1 = ring[(pushed - 16) & RING_MASK];
                            bool may_link = (links.has_fallback || (r1.ukey & FALLBACK_BIT) == 0) &&
                                            (!links_has_bloom || links.bloom_might_contain(r1.uh));
                            r1.probe_ht = may_link;
                            if (may_link) links.prefetch_slot(r1.uh);
                        }
                        if (pushed >= 32) {
                            emit_resolved(ring[(pushed - 32) & RING_MASK]);
                        }
                        ++pushed;
                    }
                }
                size_t s1_start = (pushed > 16) ? (pushed - 16) : 0;
                for (size_t k = s1_start; k < pushed; ++k) {
                    auto& r1 = ring[k & RING_MASK];
                    bool may_link = (links.has_fallback || (r1.ukey & FALLBACK_BIT) == 0) &&
                                    (!links_has_bloom || links.bloom_might_contain(r1.uh));
                    r1.probe_ht = may_link;
                    if (may_link) links.prefetch_slot(r1.uh);
                }
                size_t s2_start = (pushed > 32) ? (pushed - 32) : 0;
                for (size_t k = s2_start; k < pushed; ++k) {
                    emit_resolved(ring[k & RING_MASK]);
                }
            }
        }
        out_ts.flush_all();
    });
    double td1 = now_ms();

    // Stage 3 + Stage 4 FUSED: LPT 512-Bucket L1D-Hot Sessionize + Immediate writev()!
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
    std::vector<std::vector<uint32_t>> thread_cd_assist(static_cast<size_t>(pool.nthreads));
    std::vector<std::vector<CredCell>> thread_cd_cred(static_cast<size_t>(pool.nthreads));
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
                PartitionedStore<RawEvent16, 256, EV_BUCKETS, 4, false>::SLAB_BYTES);

        const size_t n_cells = size_t(camp_stride) * size_t(num_days);
        auto& cd_assist = thread_cd_assist[size_t(tid)];
        auto& cd_cred = thread_cd_cred[size_t(tid)];
        cd_assist.assign(n_cells, 0);
        cd_cred.assign(n_cells, CredCell{0, 0, 0});

        std::vector<uint64_t> pkeys;
        std::vector<uint32_t> pht;
        std::vector<uint32_t> ev_cnt, ev_off, ev_pids;
        std::vector<uint32_t> clk_cnt, clk_off, clk_pids;

        using EvItem = std::conditional_t<Compact, uint64_t, WidePersonEvent>;
        std::vector<EvItem> grouped_evs;
        std::vector<uint64_t> sort_tmp;
        std::vector<LocalClick> grouped_clks;
        std::vector<uint64_t> cat_rev(num_cats, 0);
        std::vector<uint32_t> used_cats;
        used_cats.reserve(32);

        std::vector<CampState> cstate(num_camps, {0, 0, 0});
        std::vector<uint32_t> active_camps;
        active_camps.reserve(num_camps);

        std::vector<char> fb_uid_buf;

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

            // L1D-hot pht sizing: avg person has ~30 events (not 4!) -> 16 KB instead of 128 KB–1 MB!
            size_t est_people = std::max<size_t>(64, nev / 20);
            size_t hcap = 64;
            while (hcap < est_people * 2) hcap <<= 1;
            pht.assign(hcap, 0);
            size_t hmask = hcap - 1;

            auto rehash_pht = [&]() SS_COLD_NOINLINE {
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

            const size_t nd = size_t(num_days);

            for (uint32_t pid = 0; pid < num_people; ++pid) {
                uint32_t es = ev_off[pid];
                uint32_t ee = ev_off[pid + 1];
                uint32_t en = ee - es;
                if (en == 0) continue;

                if constexpr (Compact) {
                    sort_events_compact_u64(grouped_evs.data() + es, en, msd_shift, sort_tmp);
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
                    auto clk_less = [&](const LocalClick& a, const LocalClick& b) SS_LAMBDA_INLINE -> bool {
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

                // Decode person's user_id ONCE per person (not once per day -> 4.2x fewer decodes!)
                alignas(16) char uid_pre_inline[16]{};
                const char* uid_pre_ptr = uid_pre_inline;
                uint32_t uid_pre_len = 0;
                uint64_t pk = pkeys[pid];
                if (__builtin_expect((pk & FALLBACK_BIT) == 0, 1)) {
                    uint32_t ulen = uint32_t(decode_inline(pk, uid_pre_inline + 4));
                    std::memcpy(uid_pre_inline, &ulen, 4);
                    uid_pre_len = 4 + ulen;
                } else {
                    auto sv = arenas.get_fallback(pk);
                    uint32_t ulen = uint32_t(sv.size());
                    fb_uid_buf.resize(4 + ulen);
                    std::memcpy(fb_uid_buf.data(), &ulen, 4);
                    if (ulen) std::memcpy(fb_uid_buf.data() + 4, sv.data(), ulen);
                    uid_pre_ptr = fb_uid_buf.data();
                    uid_pre_len = 4 + ulen;
                }

                int32_t cur_day = -1;
                int64_t cur_day_end_ms = INT64_MIN;
                uint32_t day_sessions = 0, day_events = 0;
                uint64_t day_rev = 0;
                uint32_t L = 0, R = 0;

                auto flush_day = [&]() SS_LAMBDA_INLINE {
                    if (day_sessions == 0) return;
                    if (!active_camps.empty()) {
                        for (uint32_t ac : active_camps) {
                            cd_assist[size_t(ac) * nd + size_t(cur_day)] += (day_sessions - cstate[ac].enter_sess);
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
                    ud.push_row_pre(uid_pre_ptr, uid_pre_len, win_start_day + cur_day,
                                    int32_t(day_sessions), int32_t(day_events), int64_t(day_rev),
                                    best_c, products);
                    day_sessions = 0; day_events = 0; day_rev = 0;
                };

                auto get_ev = [&](uint32_t idx, int64_t& ts, uint32_t& price, uint32_t& cat) SS_LAMBDA_INLINE {
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

                    if (cn == 0) {
                        auto& cell = cd_cred[size_t(campaigns.organic_idx) * nd + size_t(cur_day)];
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
                                    cd_assist[size_t(c) * nd + size_t(cur_day)] += (day_sessions - cstate[c].enter_sess);
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
                        auto& cell = cd_cred[size_t(credit_c) * nd + size_t(cur_day)];
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

        // FUSED STAGE 4: Write worker's user_daily/part-{tid}.parquet via writev()!
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

        if (finished_workers.fetch_add(1, std::memory_order_acq_rel) == pool.nthreads - 1) {
            if (total_ud_rows_written.load(std::memory_order_acquire) == 0) {
                std::string path = ud_dir + "/part-0.parquet";
                pq::ColData empty_ud[6]{};
                if (!pq::write_parquet_file(path.c_str(), UD_COLS, empty_ud, 6, 0)) {
                    write_ok.store(false, std::memory_order_relaxed);
                }
            }

            auto& agg_a = thread_cd_assist[0];
            auto& agg_c = thread_cd_cred[0];
            for (int t = 1; t < pool.nthreads; ++t) {
                const auto& sa = thread_cd_assist[size_t(t)];
                const auto& sc = thread_cd_cred[size_t(t)];
                for (size_t i = 0; i < n_cells; ++i) {
                    agg_a[i] += sa[i];
                    agg_c[i].sessions += sc[i].sessions;
                    agg_c[i].purchasing_sessions += sc[i].purchasing_sessions;
                    agg_c[i].revenue_cents += sc[i].revenue_cents;
                }
            }

            std::vector<uint8_t> cid_plain, ch_plain;
            std::vector<int32_t> day_v, sess_v, psess_v, assist_v;
            std::vector<int64_t> rev_v;
            cid_plain.reserve(n_cells * 10);
            ch_plain.reserve(n_cells * 10);
            day_v.reserve(n_cells);
            sess_v.reserve(n_cells);
            psess_v.reserve(n_cells);
            rev_v.reserve(n_cells);
            assist_v.reserve(n_cells);

            auto push_s = [](std::vector<uint8_t>& buf, const std::string& s) {
                uint32_t len = uint32_t(s.size());
                size_t old = buf.size();
                buf.resize(old + 4 + len);
                std::memcpy(buf.data() + old, &len, 4);
                std::memcpy(buf.data() + old + 4, s.data(), len);
            };

            const size_t nd = size_t(num_days);
            for (int32_t d = 0; d < num_days; ++d) {
                int32_t abs_day = win_start_day + d;
                for (uint32_t c = 0; c < camp_stride; ++c) {
                    size_t idx = size_t(c) * nd + size_t(d);
                    uint32_t a = agg_a[idx];
                    const auto& cell = agg_c[idx];
                    if (cell.sessions == 0 && a == 0) continue;
                    push_s(cid_plain, campaigns.ids[c]);
                    push_s(ch_plain, campaigns.channels[c]);
                    day_v.push_back(abs_day);
                    sess_v.push_back(int32_t(cell.sessions));
                    psess_v.push_back(int32_t(cell.purchasing_sessions));
                    rev_v.push_back(int64_t(cell.revenue_cents));
                    assist_v.push_back(int32_t(a));
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

    auto now_ms = []() {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    double t0 = now_ms();
    bool prof = std::getenv("PROFILE") != nullptr;

    // Spawn & pin worker pool FIRST so thread creation overlaps with manifest/dir setup!
    CpuTopology topo = detect_topology();
    int ncpu = topo.ncpu;
    ThreadPool pool(topo);

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
        lut.init(win_start_ms - CLICK_WINDOW_MS, win_start_ms, win_end_ms);
    }
    int64_t win_start_sec = win_start_ms / 1000LL;
    int64_t win_dur_ms = win_end_ms - win_start_ms;
    int64_t click_start_ms = win_start_ms - CLICK_WINDOW_MS;
    int32_t win_start_day = ms_to_day(win_start_ms);
    int32_t win_end_day = (win_end_ms > win_start_ms) ? ms_to_day(win_end_ms - 1) : win_start_day;
    int32_t num_days = std::max<int32_t>(1, win_end_day - win_start_day + 1);

    std::string ud_dir = out_dir + "/user_daily";
    std::string cd_dir = out_dir + "/campaign_daily";
    mkdir_p(ud_dir);
    mkdir_p(cd_dir);

    StringArenaPool arenas;
    arenas.init(ncpu);
    std::vector<ReadBuf> rbufs(static_cast<size_t>(ncpu));
    for (int t = 0; t < ncpu; ++t) rbufs[size_t(t)].init();

    // Stage 0: Overlap products/campaigns load with parallel open+fstat of CSV files!
    std::vector<std::pair<std::string, uint8_t>> all_paths;
    all_paths.reserve(256);
    list_csv_paths(data_dir + "/links", 2, all_paths);
    list_csv_paths(data_dir + "/clicks", 0, all_paths);
    list_csv_paths(data_dir + "/events", 1, all_paths);

    const bool keep_fds = (all_paths.size() < 480);
    std::vector<CsvFile> csv_files(all_paths.size());
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
            int fd = ::open(all_paths[i].first.c_str(), O_RDONLY);
            if (fd < 0) continue;
            struct stat st{};
            size_t fsz = (::fstat(fd, &st) == 0 && st.st_size > 0) ? size_t(st.st_size) : 0;
            if (!keep_fds || fsz == 0) { ::close(fd); fd = -1; }
            csv_files[i] = {all_paths[i].first, fd, fsz, all_paths[i].second};
        }
    });

    std::vector<CsvTask> link_tasks, io_tasks;
    build_csv_tasks(csv_files, 1U << 2, ncpu, link_tasks);
    build_csv_tasks(csv_files, (1U << 0) | (1U << 1), ncpu, io_tasks);
    double t_init = now_ms();

    // Stage 1: Blazing 3-Barrier Union-by-Key Huge-Page Hash-DSU + 1-Word Bloom
    LinkMap links;
    links.build(link_tasks, rbufs, pool, arenas);
    double t_links = now_ms();

    // Stage 2: Zero-Fault pread(256KB) + Shifted-Remainder 6-tzcnt SIMD Scan + RawEvent16!
    PartitionedStore<PersonClick, 128, PERSON_BUCKETS, 4, false> click_store;
    PartitionedStore<RawEvent16, 256, EV_BUCKETS, 4, false> event_store;
    click_store.init(ncpu);
    event_store.init(ncpu);

    // Recycle LinkMap's already-faulted huge-page DSU memory into event_store (0 new page faults!)
    if (links.recycled_dsu && links.recycled_dsu_bytes >= (4ULL << 20)) {
        size_t slice = (links.recycled_dsu_bytes / size_t(ncpu)) & ~size_t(4095);
        char* p = static_cast<char*>(links.recycled_dsu);
        for (int t = 0; t < ncpu; ++t) {
            event_store.ts[size_t(t)]->donate_slab(p + size_t(t) * slice, slice);
        }
    }

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
            auto& rbuf = rbufs[size_t(tid)];
            side_vec.reserve(65536);

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
            bool clicks_flushed = false;

            auto emit_click = [&](const ClickRing& r) SS_LAMBDA_INLINE {
                uint64_t pkey = r.probe_ht ? links.resolve_slot_only(r.ukey, r.uh, arenas) : r.ukey;
                uint64_t ph = (pkey == r.ukey) ? r.uh : hash_key64(pkey, arenas);
                uint32_t pb = uint32_t(ph >> 55);
                clk_out.push(pb, PersonClick{pkey, r.cid, r.ts_camp});
            };

            auto drain_clicks = [&]() SS_COLD_NOINLINE {
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
                    cpushed = 0;
                }
                clk_out.flush_all();
                clicks_flushed = true;
            };

            auto push_side_ev = [&](bool valid, uint32_t irel21, bool eid_in_side, uint32_t eb, uint64_t rem39,
                                    uint64_t eid_k64, uint64_t ukey, int64_t ts_ms, int64_t ing_ms,
                                    uint32_t price, uint32_t cat_id) SS_COLD_NOINLINE {
                uint32_t sidx = uint32_t(side_vec.size());
                if (__builtin_expect(sidx >= (1U << 26), 0)) std::abort();
                side_vec.push_back({eid_k64, ukey, ts_ms, ing_ms, price, cat_id});
                ev_out.push(eb, RawEvent16::make_side(valid, irel21, eid_in_side, rem39, tid, sidx));
            };

            while (true) {
                size_t ti = next_t.fetch_add(1, std::memory_order_relaxed);
                if (ti >= io_tasks.size()) break;
                const auto& task = io_tasks[ti];
                if (task.kind == 0) {
                    // Clicks: single-word L<64 SIMD + 8KB L1D branchless campaign lookup!
                    scan_pread_csv<4>(task, rbuf, [&](const Slice f[4]) SS_LAMBDA_INLINE {
                        if (__builtin_expect((f[0].len > 0) & (f[1].len > 0) & (f[2].len > 0) & (f[3].len > 0), 1)) {
                            uint32_t camp_idx = campaigns.lookup_padded(f[2].ptr, f[2].len);
                            if (__builtin_expect(camp_idx != FastStringIdMap::NOT_FOUND, 1)) {
                                int64_t ts = 0;
                                if (__builtin_expect(parse_ts(f[3].ptr, f[3].len, ts, lut) &&
                                                     ts >= click_start_ms && ts < win_end_ms, 1)) {
                                    uint64_t ukey = encode_key64_padded(f[1].ptr, f[1].len, arenas, tid);
                                    uint64_t cid = encode_key64_padded(f[0].ptr, f[0].len, arenas, tid);
                                    uint64_t ts_camp = (uint64_t(ts - click_start_ms) << 20) | uint64_t(camp_idx);
                                    uint64_t uh = hash_key64(ukey, arenas);
                                    if (__builtin_expect(!links_active, 0)) {
                                        clk_out.push(uint32_t(uh >> 55), PersonClick{ukey, cid, ts_camp});
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
                                        if (cpushed >= 32) emit_click(cring[(cpushed - 32) & CR_MASK]);
                                        ++cpushed;
                                    }
                                }
                            }
                        }
                    });
                } else {
                    if (__builtin_expect(!clicks_flushed && cpushed > 0, 0)) drain_clicks();
                    // Events: 6-tzcnt shifted-remainder SIMD + FastDay[256] + SWAR + RawEvent16!
                    scan_pread_csv<7>(task, rbuf, [&](const Slice f[7]) SS_LAMBDA_INLINE {
                        if (__builtin_expect(f[0].len == 0, 0)) return;

                        // 1. Parse ingest_ts (99.99% F1!)
                        int64_t ing_rel_sec = 0;
                        int64_t ing_ms = 0;
                        bool ing_whole_sec = true;
                        if (__builtin_expect(f[3].len == 20 && f[3].ptr[19] == 'Z', 1)) {
                            if (__builtin_expect(!lut.parse_iso19_rel_sec(f[3].ptr, ing_rel_sec), 0)) return;
                            ing_ms = (win_start_sec + ing_rel_sec) * 1000LL;
                        } else {
                            if (!parse_ts(f[3].ptr, f[3].len, ing_ms, lut)) return;
                            ing_whole_sec = (ing_ms % 1000LL == 0);
                            ing_rel_sec = ing_ms / 1000LL - win_start_sec;
                        }

                        // 2. Encode event_id -> 48b bijection (eb = top 9b, rem39 = low 39b)
                        uint64_t eid48 = 0, eid_k64 = 0;
                        uint32_t eb = 0;
                        uint64_t rem39 = 0;
                        bool eid_in_side = false;
                        if (__builtin_expect(encode_eid48_padded(f[0].ptr, f[0].len, eid48), 1)) {
                            uint64_t h48 = mix48(eid48);
                            eb = uint32_t(h48 >> 39);
                            rem39 = h48 & RawEvent16::REM39_MASK;
                        } else {
                            eid_in_side = true;
                            eid_k64 = encode_key64_padded(f[0].ptr, f[0].len, arenas, tid);
                            uint64_t h64 = hash_key64(eid_k64, arenas);
                            eb = uint32_t(h64 >> 55);
                            rem39 = h64 & RawEvent16::REM39_MASK;
                        }

                        uint32_t irel21 = 0U;
                        if (__builtin_expect(ing_whole_sec, 1)) {
                            int64_t ir = ing_rel_sec + RawEvent16::ING_BIAS;
                            if (__builtin_expect(ir > 0 && ir <= RawEvent16::ING_MAX, 1)) {
                                irel21 = uint32_t(ir);
                            }
                        }

                        // 3. Branchless event_type + non-empty user_id/product_id check
                        uint32_t v4 = 0;
                        uint64_t v8 = 0;
                        std::memcpy(&v4, f[4].ptr, 4); // safe: 64B zero-pad!
                        std::memcpy(&v8, f[4].ptr, 8);
                        bool is_vc = (f[4].len == 4) & ((v4 == view_u32) | (v4 == cart_u32));
                        bool is_pur = (f[4].len == 8) & (v8 == pur_u64);
                        bool ok = (f[1].len > 0) & (f[5].len > 0) & (is_vc | is_pur);

                        // 4. Parse ts (F1 / F3 / F2)
                        int64_t ts_ms = 0;
                        int64_t ts_rel_sec = 0;
                        bool ts_whole_sec = false;
                        if (__builtin_expect(ok, 1)) {
                            bool ts_ok = false;
                            if (f[2].len == 20 && f[2].ptr[19] == 'Z') {
                                ts_ok = lut.parse_iso19_rel_sec(f[2].ptr, ts_rel_sec);
                                ts_ms = (win_start_sec + ts_rel_sec) * 1000LL;
                                ts_whole_sec = true;
                            } else if (f[2].len == 13) {
                                ts_ok = parse_13d_swar(f[2].ptr, ts_ms);
                                // Fast check if whole second (95% of F3!)
                                if (__builtin_expect(f[2].ptr[10] == '0' && f[2].ptr[11] == '0' && f[2].ptr[12] == '0', 1)) {
                                    ts_whole_sec = true;
                                    ts_rel_sec = ts_ms / 1000LL - win_start_sec;
                                }
                            } else if (f[2].len == 25) {
                                ts_ok = lut.parse_f2_rel_sec(f[2].ptr, ts_rel_sec);
                                ts_ms = (win_start_sec + ts_rel_sec) * 1000LL;
                                ts_whole_sec = true;
                            } else {
                                ts_ok = parse_ts_slow(f[2].ptr, f[2].len, ts_ms, lut);
                            }
                            if (__builtin_expect(!ts_ok || uint64_t(ts_ms - win_start_ms) >= uint64_t(win_dur_ms) ||
                                                 ing_ms - ts_ms > DAY_MS, 0)) {
                                ok = false;
                            }
                        }

                        // 5. FAST PATH (92%+ of all rows!): valid view/cart, whole-sec, base36 IDs, delta in [-8..131063]
                        if (__builtin_expect(ok & is_vc & ts_whole_sec & (!eid_in_side) & (irel21 != 0), 1)) {
                            int64_t delta = (ing_rel_sec - ts_rel_sec) + 8LL;
                            uint64_t uid48 = 0;
                            if (__builtin_expect(uint64_t(delta) < 131072ULL &&
                                                 encode_uid48_padded(f[1].ptr, f[1].len, uid48), 1)) {
                                ev_out.push(eb, RawEvent16::make_inline(irel21, uint32_t(delta), rem39, uid48));
                                return;
                            }
                        }

                        // 6. SIDE PATH (~7-8%: purchase 3%, ms-ts 1.5%, invalid/late 2.5%, uppercase/long/clock-skew)
                        uint32_t price = 0, cat_id = 0;
                        if (ok && is_pur) {
                            if (!parse_price_cents(f[6].ptr, f[6].len, price)) {
                                ok = false;
                            } else {
                                cat_id = products.lookup(f[5].ptr, f[5].len);
                            }
                        }
                        uint64_t ukey = ok ? encode_key64_padded(f[1].ptr, f[1].len, arenas, tid) : 0ULL;
                        push_side_ev(ok, irel21, eid_in_side, eb, rem39, eid_k64, ukey, ts_ms, ing_ms, price, cat_id);
                    });
                }
            }

            if (!clicks_flushed) drain_clicks();
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
