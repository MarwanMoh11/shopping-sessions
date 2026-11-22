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
// Cgroup + Affinity-Aware CPU Detection & Safe Thread Pinning
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

static void pin_current_thread(int tid, int ncpu, const std::vector<int>& cpus) {
#if defined(__linux__)
    // ONLY pin 1-to-1 when the container actually has exactly `ncpu` pinned CPUs!
    if (cpus.size() == size_t(ncpu)) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpus[size_t(tid)], &set);
        ::sched_setaffinity(0, sizeof(set), &set);
    }
#else
    (void)tid; (void)ncpu; (void)cpus;
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
        pin_current_thread(0, nthreads, allowed_cpus);
        workers.reserve(static_cast<size_t>(nthreads - 1));
        for (int t = 1; t < nthreads; ++t) {
            workers.emplace_back([this, t]() {
                pin_current_thread(t, nthreads, allowed_cpus);
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
// Huge-Page Slab-Backed L1 Write-Combining Partitioned Store
//   - 64B Aligned Non-Temporal Stores (_mm256_stream_si256 / _mm_stream_si128)
//     whenever sizeof(T) * WC_CAP == 64 (+ _mm_sfence() in flush_all)
//   - 4 KB Block Recycling (free_head) for zero-fault CompactPersonEv8 reuse!
// ============================================================================
template <typename T, uint32_t CAP, int NUM_B, uint32_t WC_CAP>
struct PartitionedStore {
    struct alignas(64) Block {
        Block* next;
        uint32_t count;
        alignas(64) T items[CAP];
    };
    struct FreeNode { FreeNode* next; };
    static constexpr size_t SLAB_BYTES = 16ULL << 20;

    struct alignas(64) ThreadState {
        alignas(64) T wc_buf[NUM_B][WC_CAP];
        uint8_t wc_cnt[NUM_B]{};
        Block* heads[NUM_B]{};
        size_t counts[NUM_B]{};
        FreeNode* free_head = nullptr;
        std::vector<std::pair<void*, size_t>> slabs;
        char* slab = nullptr;
        size_t slab_pos = SLAB_BYTES;
        size_t slab_cap = SLAB_BYTES;
        char* pre_slab = nullptr;
        size_t pre_slab_cap = 0;

        SS_ALWAYS_INLINE void recycle_block(void* p) {
            auto* fn = static_cast<FreeNode*>(p);
            fn->next = free_head;
            free_head = fn;
        }

        SS_COLD_NOINLINE Block* alloc_block() {
            if (__builtin_expect(free_head != nullptr, 1)) {
                auto* b = reinterpret_cast<Block*>(free_head);
                free_head = free_head->next;
                b->next = nullptr;
                b->count = 0;
                return b;
            }
            constexpr size_t bsz = (sizeof(Block) + 63) & ~size_t(63);
            if (__builtin_expect(slab_pos + bsz > slab_cap, 0)) {
                if (__builtin_expect(pre_slab != nullptr && pre_slab_cap >= bsz, 1)) {
                    slab = pre_slab;
                    slab_pos = 0;
                    slab_cap = pre_slab_cap;
                    pre_slab = nullptr;
                    pre_slab_cap = 0;
                } else {
                    slab = static_cast<char*>(alloc_mmap(SLAB_BYTES, true));
                    slabs.emplace_back(slab, alloc_mmap_actual_size(SLAB_BYTES, true));
                    slab_pos = 0;
                    slab_cap = SLAB_BYTES;
                }
            }
            auto* b = reinterpret_cast<Block*>(slab + slab_pos);
            slab_pos += bsz;
            b->next = nullptr;
            b->count = 0;
            return b;
        }

        void donate_slab(void* mem, size_t bytes) {
            if (!mem || bytes < sizeof(Block) * 4) return;
            if (slab != nullptr && slab_pos == 0 && slab_cap > bytes && pre_slab == nullptr) {
                pre_slab = slab;
                pre_slab_cap = slab_cap;
            }
            slab = static_cast<char*>(mem);
            slab_pos = 0;
            slab_cap = bytes;
        }

        SS_ALWAYS_INLINE void flush_bucket_full(uint32_t b) {
            Block* h = heads[b];
            if (__builtin_expect(h == nullptr || h->count + WC_CAP > CAP, 0)) {
                Block* nb = alloc_block();
                nb->next = h;
                heads[b] = nb;
                h = nb;
            }
#if defined(__AVX2__)
            if constexpr (sizeof(T) * WC_CAP == 64) {
                auto* dst = reinterpret_cast<__m256i*>(h->items + h->count);
                const auto* src = reinterpret_cast<const __m256i*>(wc_buf[b]);
                _mm256_stream_si256(dst,     _mm256_load_si256(src));
                _mm256_stream_si256(dst + 1, _mm256_load_si256(src + 1));
            } else {
                std::memcpy(h->items + h->count, wc_buf[b], WC_CAP * sizeof(T));
            }
#elif defined(__SSE2__)
            if constexpr (sizeof(T) * WC_CAP == 64) {
                auto* dst = reinterpret_cast<__m128i*>(h->items + h->count);
                const auto* src = reinterpret_cast<const __m128i*>(wc_buf[b]);
                _mm_stream_si128(dst,     _mm_load_si128(src));
                _mm_stream_si128(dst + 1, _mm_load_si128(src + 1));
                _mm_stream_si128(dst + 2, _mm_load_si128(src + 2));
                _mm_stream_si128(dst + 3, _mm_load_si128(src + 3));
            } else {
                std::memcpy(h->items + h->count, wc_buf[b], WC_CAP * sizeof(T));
            }
#else
            std::memcpy(h->items + h->count, wc_buf[b], WC_CAP * sizeof(T));
#endif
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
#if defined(__x86_64__)
            if constexpr (sizeof(T) * WC_CAP == 64) {
                _mm_sfence();
            }
#endif
        }
    };

    int nthreads = 1;
    std::vector<ThreadState*> ts;

    void init(int nt, size_t pre_reserve_bytes = 0) {
        nthreads = nt;
        ts.resize(static_cast<size_t>(nt));
        const bool hp = (pre_reserve_bytes >= (2ULL << 20));
        for (int t = 0; t < nt; ++t) {
            void* mem = alloc_mmap((sizeof(ThreadState) + 4095) & ~size_t(4095), false);
            auto* st = new (mem) ThreadState();
            if (pre_reserve_bytes > 0) {
                char* p = static_cast<char*>(alloc_mmap(pre_reserve_bytes, hp));
                st->slab = p;
                st->slab_pos = 0;
                st->slab_cap = pre_reserve_bytes;
                st->slabs.emplace_back(p, alloc_mmap_actual_size(pre_reserve_bytes, hp));
            }
            ts[size_t(t)] = st;
        }
    }

    size_t bucket_count(int b) const {
        size_t s = 0;
        for (int t = 0; t < nthreads; ++t) s += ts[size_t(t)]->counts[b];
        return s;
    }

    void dontneed_all_tid(int tid) {
#if defined(__linux__) && defined(MADV_DONTNEED)
        auto* st = ts[size_t(tid)];
        for (auto [p, sz] : st->slabs) {
            if (p && sz) ::madvise(p, sz, MADV_DONTNEED);
        }
#else
        (void)tid;
#endif
    }
};

// ============================================================================
// Zero-Copy Guard-Paged MMAP + Bounded PREAD Chunked File Scanner + Guided Tasks
// ============================================================================
struct CsvFile {
    std::string path;
    int fd = -1;
    size_t file_size = 0;
    const char* map_base = nullptr;
    size_t map_len = 0;
    uint8_t kind = 0;
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
            return -1;
        }
        if (n == 0) break;
        got += size_t(n);
    }
    return ssize_t(got);
}

// Guided chunking: 4 MB chunks for first 85%, 512 KB for next 12%, 128 KB for final 3% (eliminates tail!)
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

    size_t big_sz = std::max<size_t>(262144, std::min<size_t>(4ULL << 20, total_bytes / size_t(nthreads * 8) + 1));
    size_t med_sz = std::max<size_t>(131072, big_sz >> 3);
    size_t sml_sz = std::max<size_t>(65536,  big_sz >> 5);
    size_t thr1 = (total_bytes * 85ULL) / 100ULL;
    size_t thr2 = (total_bytes * 97ULL) / 100ULL;
    size_t done_bytes = 0;

    for (const auto& f : files) {
        if (f.file_size == 0 || !((kind_mask >> f.kind) & 1U)) continue;
        size_t pos = 0;
        while (pos < f.file_size) {
            size_t csz = (done_bytes < thr1) ? big_sz : ((done_bytes < thr2) ? med_sz : sml_sz);
            size_t end = std::min<size_t>(f.file_size, pos + csz);
            tasks.push_back({&f, pos, end, f.kind});
            done_bytes += (end - pos);
            pos = end;
        }
    }
}

struct ReadBuf {
    static constexpr size_t READ_CHUNK = 262144;
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

template <int N, typename OnFields>
static SS_ALWAYS_INLINE void scan_mmap_csv(const CsvTask& task, OnFields&& on_fields) {
    const CsvFile& file = *task.file;
    const char* base = file.map_base;
    const size_t fsz = file.file_size;
    const size_t limit = task.end;
    if (__builtin_expect(!base || fsz == 0 || task.start >= limit, 0)) return;

#if defined(__linux__) && defined(MADV_POPULATE_READ)
    {
        size_t lo = task.start & ~size_t(4095);
        size_t hi = (std::min<size_t>(fsz, limit + 4096) + 4095) & ~size_t(4095);
        if (hi > lo) ::madvise(const_cast<char*>(base + lo), hi - lo, MADV_POPULATE_READ);
    }
#endif

    size_t cur = task.start;
    if (task.start == 0) {
        const char* nl = find_nl_simd(base, base + fsz);
        if (!nl) return;
        cur = size_t(nl - base) + 1;
        if (cur >= limit) return;
    } else if (base[task.start - 1] != '\n') {
        const char* nl = find_nl_simd(base + task.start, base + fsz);
        if (!nl) return;
        cur = size_t(nl - base) + 1;
        if (cur >= limit) return;
    }

    Slice f[N];
    uint64_t nls[64], cms[66], qms[66];
    bool done = false;

    while (cur < limit) {
        size_t avail = fsz - cur;
        size_t nblks = std::min<size_t>(64, avail >> 6);
        if (__builtin_expect(nblks == 0, 0)) break;

        const char* bptr = base + cur;
        for (size_t bi = 0; bi < nblks; ++bi) {
            if ((bi & 7U) == 0) __builtin_prefetch(bptr + (bi << 6) + 2048, 0, 1);
            Vec64 v = Vec64::load(bptr + (bi << 6));
            nls[bi] = v.eq('\n');
            cms[bi] = v.eq(',');
            qms[bi] = v.eq('"');
        }
        cms[nblks] = cms[nblks + 1] = 0;
        qms[nblks] = qms[nblks + 1] = 0;

        size_t ls = 0;
        const size_t lrel = limit - cur;
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
            const char* nl = find_nl_simd(bptr + (nblks << 6), base + fsz);
            if (!nl) break;
            if (parse_csv_line_scalar<N>(bptr, nl, f)) on_fields(f);
            cur = size_t(nl - base) + 1;
        } else {
            cur += ls;
        }
    }

    while (cur < limit && cur < fsz) {
        const char* line = base + cur;
        const char* nl = find_nl_simd(line, base + fsz);
        const char* le = nl ? nl : (base + fsz);
        if (parse_csv_line_scalar<N>(line, le, f)) on_fields(f);
        cur = nl ? (size_t(nl - base) + 1) : fsz;
    }
}

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
    size_t file_pos = (task.start > 0) ? (task.start - 1) : 0;
    size_t carry = 0;
    bool skip_first_line = true;
    bool first_read = true;
    bool hit_limit = false;

    Slice f[N];
    uint64_t nls[64], cms[66], qms[66];

    while (true) {
        if (file_pos >= fsz) break;
        // Bounded read: never over-read 256 KiB across task boundaries!
        size_t bound = hit_limit ? fsz : std::min<size_t>(fsz, limit + 4096);
        size_t to_read = hit_limit ? std::min<size_t>(4096, fsz - file_pos)
                                   : std::min<size_t>(ReadBuf::READ_CHUNK, bound > file_pos ? (bound - file_pos) : std::min<size_t>(4096, fsz - file_pos));
        if (__builtin_expect(carry + to_read + 128 > rbuf.cap, 0)) {
            rbuf.ensure(carry + to_read + 128);
        }
        ssize_t got = pread_full(fd, rbuf.data + carry, to_read, file_pos);
        if (got <= 0) break;
        size_t buf_file_off = file_pos - carry;
        file_pos += size_t(got);
        size_t total = carry + size_t(got);
        std::memset(rbuf.data + total, 0, 64);

        size_t cur = 0;
        if (__builtin_expect(first_read, 0)) {
            first_read = false;
            if (task.start > 0) {
                cur = 1;
                skip_first_line = (rbuf.data[0] != '\n');
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
                const char* nl = find_nl_simd(bptr + (nblks << 6), rbuf.data + total);
                if (!nl) break;
                if (parse_csv_line_scalar<N>(bptr, nl, f)) on_fields(f);
                cur = size_t(nl - rbuf.data) + 1;
                if (cur >= limit_in_buf) { done = true; break; }
            } else {
                cur += ls;
            }
        }

        if (done && buf_file_off + cur >= limit) break;

        while (cur < limit_in_buf) {
            const char* line = rbuf.data + cur;
            const char* nl = find_nl_simd(line, rbuf.data + total);
            if (!nl) break;
            if (parse_csv_line_scalar<N>(line, nl, f)) on_fields(f);
            cur = size_t(nl - rbuf.data) + 1;
        }

        if (buf_file_off + cur >= limit) break;

        if (file_pos >= fsz) {
            if (cur < total && buf_file_off + cur < limit) {
                if (parse_csv_line_scalar<N>(rbuf.data + cur, rbuf.data + total, f)) on_fields(f);
            }
            break;
        }

        carry = total - cur;
        if (carry > 0) std::memmove(rbuf.data, rbuf.data + cur, carry);
        if (file_pos >= limit) hit_limit = true;
    }

    if (close_after) ::close(fd);
}

template <int N, typename OnFields>
static SS_ALWAYS_INLINE void scan_csv(const CsvTask& task, ReadBuf& rbuf, OnFields&& on_fields) {
    if (__builtin_expect(task.file->map_base != nullptr, 1)) {
        scan_mmap_csv<N>(task, std::forward<OnFields>(on_fields));
    } else {
        scan_pread_csv<N>(task, rbuf, std::forward<OnFields>(on_fields));
    }
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

    SS_ALWAYS_INLINE uint32_t lookup_padded(const char* s, size_t len) const {
        if (__builtin_expect((len - 3) <= 3 && s[0] == 'c' && s[1] == '_', 1)) {
            uint32_t w;
            std::memcpy(&w, s + 2, 4);
            uint32_t sh = uint32_t(6 - len) << 3;
            uint32_t b0_ok = uint8_t(w) - uint8_t('1');
            uint32_t shifted = (w << sh) | (0x30303030U & ~((~0U) << sh));
            uint32_t bad = ((shifted & 0xF0F0F0F0U) ^ 0x30303030U) |
                           (((shifted + 0x06060606U) & 0xF0F0F0F0U) ^ 0x30303030U);
            if (__builtin_expect((bad == 0) & (b0_ok < 9U), 1)) {
                uint32_t d = shifted & 0x0F0F0F0FU;
                uint32_t p = (d * 10U + (d >> 8)) & 0x00FF00FFU;
                uint32_t n = (p & 0xFFU) * 100U + (p >> 16);
                if (__builtin_expect(n < FAST_CAP, 1)) {
                    uint16_t r = fast_c[n];
                    if (__builtin_expect(r != NF16, 1)) return uint32_t(r);
                    if (__builtin_expect(organic_idx <= NF16, 1)) return FastStringIdMap::NOT_FOUND;
                }
            }
        }
        return map.find(s, len);
    }
};

struct Products {
    struct CatEntry {
        uint32_t total_len;
        char bytes[64];
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
        // Fast hashed category deduplication (O(1) per product instead of linear scan!)
        std::vector<std::string_view> cats;
        cats.reserve(64);
        cats.push_back("unknown");
        FastStringIdMap seen_cats;
        seen_cats.init(64);
        seen_cats.insert("unknown", 0);
        size_t seen_cap_half = 32;
        for (const auto& r : rows) {
            if (seen_cats.find(r.second.data(), r.second.size()) == FastStringIdMap::NOT_FOUND) {
                uint32_t id = uint32_t(cats.size());
                cats.push_back(r.second);
                if (cats.size() >= seen_cap_half) {
                    seen_cats.init(cats.size() * 2);
                    for (uint32_t k = 0; k < uint32_t(cats.size()); ++k) seen_cats.insert(cats[k], k);
                    seen_cap_half = (seen_cats.mask + 1) >> 1;
                } else {
                    seen_cats.insert(r.second, id);
                }
            }
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
// Stage 1: Total User Dictionary + 1-Pass CAS Union-by-Key Hash-DSU!
//   - fast_dict_mode == true (no FALLBACK_BIT in links):
//       * uslots[ucap]: 8B/slot [tag32 = h48>>16][disp1 (1..254):8][pid:24]
//       * Holds BOTH linked uid48s (pid = comp_id) AND unlinked uid48s (insert-if-absent CAS in scan!)
//       * pid_name[pid]: exact canonical name (bits 63..56==0 => raw uid48; else key64/arena!)
//       * fb_slots: read-only linked non-uid48 keys -> comp_id
//       * extra_slots (mutex): ultra-rare unlinked non-uid48 keys or d > 254 overflows -> pid
//     => ZERO LinkMap probes in Dedup Loop 2, ZERO pht/ev_pids in Stage 3, 8B person records!
//   - fast_dict_mode == false (FALLBACK_BIT in links or wide window or overflow):
//       * Classic 16B slots[cap] (key64 -> min_key64), 100% exact for all strings!
// ============================================================================
struct UserDict {
    struct Slot {
        uint64_t key;
        uint64_t val;
    };

    static constexpr uint32_t MAX_DICT_PIDS = (1U << 24) - 4096U;
    static constexpr size_t   PID_NAME_CAP  = 1ULL << 24;

    bool fast_dict_mode = false;
    uint64_t* uslots = nullptr;
    size_t ucap = 0;
    size_t umask = 0;
    size_t uslots_bytes = 0;

    uint64_t* pid_name = nullptr;
    size_t pid_name_bytes = 0;
    std::atomic<uint32_t> next_pid{1};
    std::atomic<bool> dict_overflow{false};

    // Read-only linked non-uid48 keys -> comp_id
    Slot* fb_slots = nullptr;
    size_t fb_mask = 0;
    size_t fb_bytes = 0;

    // Mutex-guarded overflow/non-uid48 unlinked user map (hit < 0.001% of rows!)
    mutable std::mutex extra_mu;
    Slot* extra_slots = nullptr;
    size_t extra_cap = 0;
    size_t extra_mask = 0;
    size_t extra_cnt = 0;

    // Classic 16B-slot fallback mode
    Slot* slots = nullptr;
    size_t cap = 0;
    size_t mask = 0;
    size_t slots_bytes = 0;
    bool has_fallback = false;

    void* recycled_dsu = nullptr;
    size_t recycled_dsu_bytes = 0;

    SS_ALWAYS_INLINE void prefetch_uslot(uint64_t h48) const {
        __builtin_prefetch(&uslots[size_t(h48) & umask], 0, 1);
    }

    SS_COLD_NOINLINE uint32_t get_or_insert_extra(uint64_t k, const StringArenaPool* arenas) {
        std::lock_guard<std::mutex> lk(extra_mu);
        if (extra_cap == 0) {
            extra_cap = 256;
            extra_mask = extra_cap - 1;
            extra_slots = static_cast<Slot*>(alloc_mmap(extra_cap * sizeof(Slot), false));
        } else if ((extra_cnt + 1) * 2 >= extra_cap) {
            size_t nc = extra_cap * 2;
            size_t nm = nc - 1;
            auto* ns = static_cast<Slot*>(alloc_mmap(nc * sizeof(Slot), false));
            for (size_t i = 0; i < extra_cap; ++i) {
                uint64_t ok = extra_slots[i].key;
                if (ok == 0) continue;
                uint64_t h = (arenas && (ok & FALLBACK_BIT)) ? hash_key64(ok, *arenas) : mix64(ok);
                size_t idx = size_t(h) & nm;
                while (ns[idx].key != 0) idx = (idx + 1) & nm;
                ns[idx] = extra_slots[i];
            }
            ::munmap(extra_slots, (extra_cap * sizeof(Slot) + 4095) & ~size_t(4095));
            extra_slots = ns;
            extra_cap = nc;
            extra_mask = nm;
        }
        uint64_t h = (arenas && (k & FALLBACK_BIT)) ? hash_key64(k, *arenas) : mix64(k);
        size_t idx = size_t(h) & extra_mask;
        while (true) {
            uint64_t cur = extra_slots[idx].key;
            if (cur == 0) {
                uint32_t np = next_pid.fetch_add(1, std::memory_order_relaxed);
                if (__builtin_expect(np >= MAX_DICT_PIDS, 0)) {
                    dict_overflow.store(true, std::memory_order_relaxed);
                    return 0;
                }
                pid_name[np] = k;
                extra_slots[idx] = {k, uint64_t(np)};
                ++extra_cnt;
                return np;
            }
            if (cur == k || (arenas && ((cur & k) & FALLBACK_BIT) && key_eq(cur, k, *arenas))) {
                return uint32_t(extra_slots[idx].val);
            }
            idx = (idx + 1) & extra_mask;
        }
    }

    SS_COLD_NOINLINE void preinsert_extra_linked(uint64_t k, uint32_t cid) {
        std::lock_guard<std::mutex> lk(extra_mu);
        if (extra_cap == 0) {
            extra_cap = 256;
            extra_mask = extra_cap - 1;
            extra_slots = static_cast<Slot*>(alloc_mmap(extra_cap * sizeof(Slot), false));
        } else if ((extra_cnt + 1) * 2 >= extra_cap) {
            size_t nc = extra_cap * 2;
            size_t nm = nc - 1;
            auto* ns = static_cast<Slot*>(alloc_mmap(nc * sizeof(Slot), false));
            for (size_t i = 0; i < extra_cap; ++i) {
                uint64_t ok = extra_slots[i].key;
                if (ok == 0) continue;
                size_t idx = size_t(mix64(ok)) & nm;
                while (ns[idx].key != 0) idx = (idx + 1) & nm;
                ns[idx] = extra_slots[i];
            }
            ::munmap(extra_slots, (extra_cap * sizeof(Slot) + 4095) & ~size_t(4095));
            extra_slots = ns;
            extra_cap = nc;
            extra_mask = nm;
        }
        size_t idx = size_t(mix64(k)) & extra_mask;
        while (extra_slots[idx].key != 0) {
            if (extra_slots[idx].key == k) return;
            idx = (idx + 1) & extra_mask;
        }
        extra_slots[idx] = {k, uint64_t(cid)};
        ++extra_cnt;
    }

    // Lock-free, carry-safe (d = 1..254) lookup/insert of uid48 -> dense 24-bit pid!
    SS_ALWAYS_INLINE uint32_t get_or_insert_uid48(
        uint64_t u48, uint64_t h48, uint32_t& tl_cur, uint32_t& tl_end
    ) {
        size_t idx = size_t(h48) & umask;
        const uint64_t tag_hi = (h48 >> 16) << 32;
        for (uint32_t d = 1; d <= 254; ++d) {
            std::atomic_ref<uint64_t> slot(uslots[idx]);
            uint64_t sl = slot.load(std::memory_order_relaxed);
            const uint64_t want_hi = tag_hi | (uint64_t(d) << 24);
            if (__builtin_expect((sl & 0xFFFFFFFFFF000000ULL) == want_hi, 1)) {
                return uint32_t(sl & 0xFFFFFFULL);
            }
            if (__builtin_expect(sl == 0, 1)) {
                if (__builtin_expect(tl_cur >= tl_end, 0)) {
                    uint32_t base = next_pid.fetch_add(1024, std::memory_order_relaxed);
                    if (__builtin_expect(base >= MAX_DICT_PIDS, 0)) {
                        dict_overflow.store(true, std::memory_order_relaxed);
                        return 0;
                    }
                    tl_cur = base;
                    tl_end = base + 1024;
                }
                uint32_t np = tl_cur;
                uint64_t exp = 0;
                uint64_t val = want_hi | uint64_t(np);
                if (slot.compare_exchange_strong(exp, val, std::memory_order_relaxed)) {
                    pid_name[np] = u48;
                    ++tl_cur;
                    return np;
                }
                if ((exp & 0xFFFFFFFFFF000000ULL) == want_hi) {
                    return uint32_t(exp & 0xFFFFFFULL);
                }
            }
            idx = (idx + 1) & umask;
        }
        return get_or_insert_extra(u48, nullptr);
    }

    SS_COLD_NOINLINE uint32_t get_or_insert_non_uid48(uint64_t k, const StringArenaPool& arenas) {
        if (fb_slots && (k & FALLBACK_BIT) == 0) {
            uint64_t h = mix64(k);
            size_t idx = size_t(h) & fb_mask;
            while (true) {
                uint64_t cur = fb_slots[idx].key;
                if (cur == 0) break;
                if (cur == k) return uint32_t(fb_slots[idx].val);
                idx = (idx + 1) & fb_mask;
            }
        }
        return get_or_insert_extra(k, &arenas);
    }

    SS_ALWAYS_INLINE uint64_t resolve_classic(uint64_t k, const StringArenaPool& arenas) const {
        if (__builtin_expect(cap == 0, 0)) return k;
        if (__builtin_expect(!has_fallback && (k & FALLBACK_BIT) != 0, 0)) return k;
        size_t idx = size_t(hash_key64(k, arenas)) & mask;
        while (true) {
            uint64_t cur = slots[idx].key;
            if (cur == 0) return k;
            if (key_eq(cur, k, arenas)) return slots[idx].val;
            idx = (idx + 1) & mask;
        }
    }

    void dontneed_tables() {
#if defined(__linux__) && defined(MADV_DONTNEED)
        if (uslots && uslots_bytes) ::madvise(uslots, uslots_bytes, MADV_DONTNEED);
        if (pid_name && pid_name_bytes) {
            size_t used = (size_t(next_pid.load(std::memory_order_relaxed)) + 2048) * sizeof(uint64_t);
            used = std::min(used, pid_name_bytes);
            used = (used + 4095) & ~size_t(4095);
            if (used) ::madvise(pid_name, used, MADV_DONTNEED);
        }
        if (slots && slots_bytes) ::madvise(slots, slots_bytes, MADV_DONTNEED);
#endif
    }

    void build_and_load_dims(
        const std::vector<CsvTask>& tasks,
        std::vector<ReadBuf>& rbufs,
        ThreadPool& pool,
        StringArenaPool& arenas,
        Products& products,
        Campaigns& campaigns,
        const std::string& data_dir,
        size_t total_event_bytes,
        bool allow_fast_dict
    ) {
        struct Edge { uint64_t a, b; };
        std::vector<std::vector<Edge>> thread_edges(static_cast<size_t>(pool.nthreads));
        std::atomic<size_t> next_t{0};
        std::atomic<bool> any_fallback{false};

        // Pass 1: Overlap products.csv + campaigns.csv loads with parallel links/*.csv scan!
        pool.run([&](int tid) {
            if (tid == 0) {
                products.load(data_dir + "/products.csv");
                if (pool.nthreads == 1) campaigns.load(data_dir + "/campaigns.csv");
            } else if (tid == 1) {
                campaigns.load(data_dir + "/campaigns.csv");
            }
            if (tasks.empty()) return;
            auto& edges = thread_edges[size_t(tid)];
            edges.reserve(65536);
            auto& rbuf = rbufs[size_t(tid)];
            bool local_fb = false;
            while (true) {
                size_t ti = next_t.fetch_add(1, std::memory_order_relaxed);
                if (ti >= tasks.size()) break;
                scan_csv<2>(tasks[ti], rbuf, [&](const Slice f[2]) SS_LAMBDA_INLINE {
                    if (__builtin_expect((f[0].len > 0) & (f[1].len > 0), 1)) {
                        uint64_t ka = encode_key64_padded(f[0].ptr, f[0].len, arenas, tid);
                        uint64_t kb = encode_key64_padded(f[1].ptr, f[1].len, arenas, tid);
                        if (__builtin_expect(ka == kb, 0)) return;
                        if (__builtin_expect(((ka | kb) & FALLBACK_BIT) != 0, 0)) {
                            if (key_eq(ka, kb, arenas)) return;
                            local_fb = true;
                        }
                        edges.push_back({ka, kb});
                    }
                });
            }
            if (local_fb) any_fallback.store(true, std::memory_order_relaxed);
        });

        size_t total_edges = 0;
        for (int t = 0; t < pool.nthreads; ++t) total_edges += thread_edges[size_t(t)].size();
        has_fallback = any_fallback.load(std::memory_order_relaxed);

        // Allocate Total User Dictionary (32 MB L3-resident at 20M, ~35% load -> avg_d=1.20!)
        if (allow_fast_dict && !has_fallback) {
            size_t est_users = std::max<size_t>(total_edges * 2, total_edges + total_event_bytes / 1500);
            ucap = 65536;
            while (ucap * 3 < est_users * 5) ucap <<= 1;
            umask = ucap - 1;
            uslots_bytes = alloc_mmap_actual_size(ucap * sizeof(uint64_t), ucap * sizeof(uint64_t) >= (2ULL << 20));
            uslots = static_cast<uint64_t*>(alloc_mmap(ucap * sizeof(uint64_t), ucap * sizeof(uint64_t) >= (2ULL << 20)));
            pid_name_bytes = PID_NAME_CAP * sizeof(uint64_t);
            pid_name = static_cast<uint64_t*>(alloc_mmap(pid_name_bytes, false));
            next_pid.store(1, std::memory_order_relaxed);

            if (total_edges == 0) {
                fast_dict_mode = true;
                return;
            }

            struct DsuSlot {
                uint64_t key;
                uint32_t parent;
                uint32_t comp_id;
            };
            size_t min_slots = total_edges * 2 + (total_edges >> 1) + 64;
            size_t raw_dsu_bytes = min_slots * sizeof(DsuSlot);
            bool dsu_hp = (raw_dsu_bytes >= (2ULL << 20));
            size_t dsu_bytes = dsu_hp
                ? ((raw_dsu_bytes + (2ULL << 20) - 1) & ~((2ULL << 20) - 1))
                : ((raw_dsu_bytes + 4095) & ~size_t(4095));
            size_t hcap = dsu_bytes / sizeof(DsuSlot);
            auto* dsu = static_cast<DsuSlot*>(alloc_mmap(dsu_bytes, dsu_hp));

            auto slot_of = [hcap](uint64_t k) SS_LAMBDA_INLINE -> size_t {
                return size_t((static_cast<unsigned __int128>(mix64(k)) * hcap) >> 64);
            };
            // Pass 2a: Relaxed lock-free interning of all edge endpoints into dsu[]!
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
            std::vector<std::vector<std::pair<uint32_t, uint32_t>>> thread_idx_edges(static_cast<size_t>(pool.nthreads));
            pool.run([&](int tid) {
                const auto& ev = thread_edges[size_t(tid)];
                auto& ie = thread_idx_edges[size_t(tid)];
                const size_t n = ev.size();
                ie.resize(n);
                for (size_t i = 0; i < n; ++i) {
                    if (__builtin_expect(i + 16 < n, 1)) {
                        __builtin_prefetch(&dsu[slot_of(ev[i + 16].a)], 1, 1);
                        __builtin_prefetch(&dsu[slot_of(ev[i + 16].b)], 1, 1);
                    }
                    ie[i] = {insert_key(ev[i].a), insert_key(ev[i].b)};
                }
            });

            // Pass 2b: Union-by-Key over 100% IMMUTABLE dsu[].key (zero acq_rel, zero data race!)
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
            pool.run([&](int tid) {
                const auto& ie = thread_idx_edges[size_t(tid)];
                const size_t n = ie.size();
                for (size_t i = 0; i < n; ++i) {
                    if (__builtin_expect(i + 16 < n, 1)) {
                        __builtin_prefetch(&dsu[ie[i + 16].first], 1, 1);
                        __builtin_prefetch(&dsu[ie[i + 16].second], 1, 1);
                    }
                    uint32_t u = ie[i].first;
                    uint32_t v = ie[i].second;
                    while (u != v) {
                        u = find_root(u);
                        v = find_root(v);
                        if (u == v) break;
                        if (dsu[u].key < dsu[v].key) std::swap(u, v);
                        uint32_t exp = 0;
                        if (std::atomic_ref<uint32_t>(dsu[u].parent).compare_exchange_weak(
                                exp, v + 1, std::memory_order_relaxed)) {
                            break;
                        }
                    }
                }
            });

            // Pass 3 (SINGLE PASS!): CAS dense comp_id on roots + populate uslots + collect non-uid48/overflow!
            std::vector<std::vector<std::pair<uint64_t, uint32_t>>> thread_fb(static_cast<size_t>(pool.nthreads));
            std::vector<std::vector<std::pair<uint64_t, uint32_t>>> thread_ovf(static_cast<size_t>(pool.nthreads));

            pool.run([&](int tid) {
                size_t s = (hcap * size_t(tid)) / size_t(pool.nthreads);
                size_t e = (hcap * size_t(tid + 1)) / size_t(pool.nthreads);
                auto& lfb = thread_fb[size_t(tid)];
                auto& lov = thread_ovf[size_t(tid)];
                uint32_t tl_cur = 0, tl_end = 0;
                for (size_t i = s; i < e; ++i) {
                    uint64_t k = dsu[i].key;
                    if (k == 0) continue;
                    uint32_t r = uint32_t(i);
                    while (dsu[r].parent != 0) r = dsu[r].parent - 1;
                    std::atomic_ref<uint32_t> ac(dsu[r].comp_id);
                    uint32_t cid = ac.load(std::memory_order_relaxed);
                    if (cid == 0) {
                        if (__builtin_expect(tl_cur >= tl_end, 0)) {
                            uint32_t base = next_pid.fetch_add(256, std::memory_order_relaxed);
                            if (__builtin_expect(base >= MAX_DICT_PIDS, 0)) {
                                dict_overflow.store(true, std::memory_order_relaxed);
                                continue;
                            }
                            tl_cur = base;
                            tl_end = base + 256;
                        }
                        uint32_t cand = tl_cur;
                        uint32_t exp = 0;
                        if (ac.compare_exchange_strong(exp, cand, std::memory_order_relaxed)) {
                            pid_name[cand] = dsu[r].key;
                            cid = cand;
                            ++tl_cur;
                        } else {
                            cid = exp;
                        }
                    }
                    char tmp[16]{};
                    size_t len = decode_inline(k, tmp);
                    uint64_t u48 = 0;
                    if (__builtin_expect(encode_uid48_padded(tmp, len, u48), 1)) {
                        uint64_t h48 = mix48(u48);
                        size_t home = size_t(h48) & umask;
                        const uint64_t tag_hi = (h48 >> 16) << 32;
                        bool placed = false;
                        for (uint32_t d = 1; d <= 254; ++d) {
                            size_t idx = (home + (d - 1)) & umask;
                            uint64_t val = tag_hi | (uint64_t(d) << 24) | uint64_t(cid);
                            uint64_t exp = 0;
                            if (std::atomic_ref<uint64_t>(uslots[idx]).compare_exchange_strong(
                                    exp, val, std::memory_order_relaxed)) {
                                placed = true;
                                break;
                            }
                        }
                        if (__builtin_expect(!placed, 0)) lov.emplace_back(u48, cid);
                    } else {
                        lfb.emplace_back(k, cid);
                    }
                }
            });

            if (__builtin_expect(!dict_overflow.load(std::memory_order_relaxed), 1)) {
                size_t total_fb = 0;
                for (const auto& v : thread_fb) total_fb += v.size();
                if (total_fb > 0) {
                    size_t fcap = 64;
                    while (fcap < total_fb * 4) fcap <<= 1;
                    fb_mask = fcap - 1;
                    fb_bytes = (fcap * sizeof(Slot) + 4095) & ~size_t(4095);
                    fb_slots = static_cast<Slot*>(alloc_mmap(fcap * sizeof(Slot), false));
                    for (const auto& v : thread_fb) {
                        for (auto [k, cid] : v) {
                            size_t idx = size_t(mix64(k)) & fb_mask;
                            while (fb_slots[idx].key != 0) idx = (idx + 1) & fb_mask;
                            fb_slots[idx] = {k, uint64_t(cid)};
                        }
                    }
                }
                for (const auto& v : thread_ovf) {
                    for (auto [u48, cid] : v) preinsert_extra_linked(u48, cid);
                }
                fast_dict_mode = true;
                if (dsu_bytes >= (4ULL << 20)) {
                    recycled_dsu = dsu;
                    recycled_dsu_bytes = dsu_bytes;
                } else {
                    ::munmap(dsu, dsu_bytes);
                }
                return;
            }

            // Ultra-rare >16.7M link components -> release fast tables & fall back to classic!
            ::munmap(uslots, uslots_bytes); uslots = nullptr; uslots_bytes = 0;
            ::munmap(pid_name, pid_name_bytes); pid_name = nullptr; pid_name_bytes = 0;
            ::munmap(dsu, dsu_bytes);
            dict_overflow.store(false, std::memory_order_relaxed);
        }

        // Classic 16B-slot DSU fallback (supports arbitrary UTF-8 / >16M components / wide windows)
        if (total_edges == 0) return;
        cap = 64;
        while (cap < total_edges * 4) cap <<= 1;
        mask = cap - 1;
        slots_bytes = alloc_mmap_actual_size(cap * sizeof(Slot), cap * sizeof(Slot) >= (2ULL << 20));
        slots = static_cast<Slot*>(alloc_mmap(cap * sizeof(Slot), cap * sizeof(Slot) >= (2ULL << 20)));

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
};

// ============================================================================
// Compact Record Structs (512-Bucket Cascade-Lake L2-Fit RawEvent16 + 8B PEv8 + 16B Purchase/Click)
// ============================================================================
static constexpr int EV_BUCKETS     = 512;
static constexpr int EV_SHIFT       = 9;
static constexpr int PERSON_BUCKETS = 512;
static constexpr int PERSON_SHIFT   = 9;

struct SideEvent {
    uint64_t eid_key64;
    uint64_t user_key64; // used in wide/classic mode
    int64_t ts_ms;
    int64_t ingest_ms;
    uint32_t pid;        // used in fast_dict_mode!
    uint32_t price;
    uint32_t cat_id;
    uint32_t _pad;
};

struct RawEvent16 {
    uint64_t w0;
    uint64_t w1;

    static constexpr int64_t ING_BIAS = 1LL << 19;
    static constexpr int64_t ING_MAX  = (1LL << 21) - 1;
    // 9 bucket bits (eb = h48 >> 39) + 39 remainder bits (rem39) = exact 48-bit bijection!
    // Bit 39 = eid_in_side (separates standard eid48 namespace from generic eid_key64 namespace!)
    static constexpr uint64_t EID40_MASK = (1ULL << 40) - 1ULL;
    static constexpr uint64_t REM39_MASK = (1ULL << 39) - 1ULL;

    SS_ALWAYS_INLINE bool is_side() const { return (w0 >> 63) != 0; }
    SS_ALWAYS_INLINE bool is_valid() const { return ((w0 >> 62) & 1ULL) != 0; }
    SS_ALWAYS_INLINE uint32_t ingest_rel() const { return uint32_t((w0 >> 41) & 0x1FFFFFULL); }
    SS_ALWAYS_INLINE uint64_t eid40() const { return w0 & EID40_MASK; }
    SS_ALWAYS_INLINE bool eid_in_side() const { return ((w0 >> 39) & 1ULL) != 0; }

    // Fast-dict inline: w1 = [ts_rel_ms:32][pid:24][0:8] (covers ALL F3-ms rows inline too!)
    static SS_ALWAYS_INLINE RawEvent16 make_inline_pid(uint32_t ing_rel21, uint64_t rem39, uint32_t ts_rel_ms, uint32_t pid) {
        uint64_t a = (1ULL << 62) | (uint64_t(ing_rel21) << 41) | rem39;
        uint64_t b = (uint64_t(ts_rel_ms) << 32) | (uint64_t(pid) << 8);
        return {a, b};
    }

    // Invalid-inline: zero SideEvent allocation for invalid rows with standard eid48 + valid ingest!
    static SS_ALWAYS_INLINE RawEvent16 make_invalid_inline(uint32_t ing_rel21, uint64_t rem39) {
        uint64_t a = (uint64_t(ing_rel21) << 41) | rem39;
        return {a, 0ULL};
    }

    static SS_ALWAYS_INLINE RawEvent16 make_side(bool valid, uint32_t ing_rel21, bool eid_side, uint64_t rem39, int tid, uint32_t sidx) {
        uint64_t a = (1ULL << 63) | (uint64_t(valid) << 62) | (uint64_t(ing_rel21) << 41) |
                     (uint64_t(eid_side) << 39) | rem39;
        uint64_t b = (uint64_t(tid) << 26) | uint64_t(sidx);
        return {a, b};
    }

    SS_ALWAYS_INLINE uint32_t inline_ts_rel_ms() const { return uint32_t(w1 >> 32); }
    SS_ALWAYS_INLINE uint32_t inline_pid() const { return uint32_t((w1 >> 8) & 0xFFFFFFULL); }
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

// 8-byte sortable person event for fast_dict_mode: [pid_hi:32][ts_rel_ms:32] (8 per 64B NT store!)
struct CompactPersonEv8 {
    uint64_t v;
    static SS_ALWAYS_INLINE CompactPersonEv8 make(uint32_t pid_hi, uint32_t ts_rel_ms) {
        return {(uint64_t(pid_hi) << 32) | uint64_t(ts_rel_ms)};
    }
    SS_ALWAYS_INLINE uint32_t pid_hi() const { return uint32_t(v >> 32); }
    SS_ALWAYS_INLINE uint32_t ts_rel_ms() const { return uint32_t(v); }
};

// 16-byte purchase record (only 2.97% of events -> ~18 KB per bucket -> L1D hot!)
struct CompactPurchase {
    uint32_t pid_hi;
    uint32_t ts_rel_ms;
    uint32_t price;
    uint32_t cat_id;
};

struct LocalPurchase {
    uint32_t ts_rel_ms;
    uint32_t price;
    uint32_t cat_id;
};

// 16-byte compact click for fast_dict_mode: [pid_hi:20][ts_rel_ms:32][camp_idx:12] + click_id (64B NT store!)
struct CompactClick {
    uint64_t click_id;
    uint64_t meta;
    static SS_ALWAYS_INLINE CompactClick make(uint32_t pid_hi, uint32_t ts_rel_ms, uint32_t camp_idx, uint64_t cid) {
        return {cid, (uint64_t(pid_hi) << 44) | (uint64_t(ts_rel_ms) << 12) | uint64_t(camp_idx)};
    }
    SS_ALWAYS_INLINE uint32_t pid_hi() const { return uint32_t(meta >> 44); }
    SS_ALWAYS_INLINE uint32_t ts_off() const { return uint32_t((meta >> 12) & 0xFFFFFFFFULL); }
    SS_ALWAYS_INLINE uint32_t camp_idx() const { return uint32_t(meta & 0xFFFULL); }
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
    uint64_t ts_camp;
};

struct LocalClick {
    uint64_t ts_camp;
    uint64_t click_id;
    SS_ALWAYS_INLINE uint64_t ts_off() const { return ts_camp >> 20; }
    SS_ALWAYS_INLINE uint32_t camp_idx() const { return uint32_t(ts_camp & 0xFFFFFULL); }
};

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
// Bounds-Safe THP UserDailyBuffers (Clean Separate Mapping + Parallel Teardown)
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

    std::vector<std::pair<void*, size_t>> allocs;

    void* track_alloc(size_t bytes, bool hp) {
        void* p = alloc_mmap(bytes, hp);
        allocs.emplace_back(p, alloc_mmap_actual_size(bytes, hp));
        return p;
    }

    void dontneed_all() {
#if defined(__linux__) && defined(MADV_DONTNEED)
        for (auto [p, sz] : allocs) {
            if (p && sz) ::madvise(p, sz, MADV_DONTNEED);
        }
#endif
    }

    void init(size_t est_rows, size_t max_cat_len) {
        size_t rcap = std::max<size_t>(est_rows * 2, 131072);
        row_cap = rcap;
        uid_cap = rcap * 16 + 4096;
        top_cat_cap = rcap * (4 + std::min<size_t>(max_cat_len, 24)) + 4096;
        // Pack fixed columns into one contiguous THP slab for minimal page faults!
        size_t bm_sz = ((rcap >> 3) + 64) & ~size_t(63);
        size_t fixed_bytes = rcap * 20 + bm_sz;
        char* fmem = static_cast<char*>(track_alloc(fixed_bytes, fixed_bytes >= (2ULL << 20)));
        day_vals       = reinterpret_cast<int32_t*>(fmem); fmem += rcap * 4;
        sess_vals      = reinterpret_cast<int32_t*>(fmem); fmem += rcap * 4;
        ev_vals        = reinterpret_cast<int32_t*>(fmem); fmem += rcap * 4;
        rev_vals       = reinterpret_cast<int64_t*>(fmem); fmem += rcap * 8;
        top_cat_bitmap = reinterpret_cast<uint8_t*>(fmem);
        uid_plain      = static_cast<uint8_t*>(track_alloc(uid_cap, uid_cap >= (2ULL << 20)));
        top_cat_plain  = static_cast<uint8_t*>(track_alloc(top_cat_cap, top_cat_cap >= (2ULL << 20)));
    }

    SS_COLD_NOINLINE void grow_rows() {
        size_t new_cap = row_cap * 2;
        size_t bm_sz = ((new_cap >> 3) + 64) & ~size_t(63);
        size_t fixed_bytes = new_cap * 20 + bm_sz;
        char* fmem = static_cast<char*>(track_alloc(fixed_bytes, fixed_bytes >= (2ULL << 20)));
        auto* nd = reinterpret_cast<int32_t*>(fmem); fmem += new_cap * 4;
        auto* ns = reinterpret_cast<int32_t*>(fmem); fmem += new_cap * 4;
        auto* ne = reinterpret_cast<int32_t*>(fmem); fmem += new_cap * 4;
        auto* nr = reinterpret_cast<int64_t*>(fmem); fmem += new_cap * 8;
        auto* nb = reinterpret_cast<uint8_t*>(fmem);
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
        auto* np = static_cast<uint8_t*>(track_alloc(new_cap, new_cap >= (2ULL << 20)));
        if (uid_sz) std::memcpy(np, uid_plain, uid_sz);
        uid_plain = np;
        uid_cap = new_cap;
    }

    SS_COLD_NOINLINE void grow_cat(size_t need) {
        size_t new_cap = std::max(top_cat_cap * 2, top_cat_sz + need + 65536);
        auto* np = static_cast<uint8_t*>(track_alloc(new_cap, new_cap >= (2ULL << 20)));
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
            std::memcpy(uid_plain + uid_sz, uid_pre, 16);
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
// Dictionary-Encoded THP FastUserDailyBuffers (3.4x fewer user_id copies, -43% Parquet bytes!)
//   - user_id:      RLE_DICTIONARY (each active person string written ONCE + 1 RLE run!)
//   - day:          RLE_DICTIONARY (4-bit packed for 14-day windows!)
//   - top_category: RLE_DICTIONARY (5-bit packed for ~21 categories!)
// ============================================================================
struct FastUserDailyBuffers {
    uint8_t* uid_dict = nullptr;
    size_t uid_dict_sz = 0, uid_dict_cap = 0;
    std::vector<uint32_t> uid_runs; // 1 entry per active person emitting >=1 row!

    uint16_t* day_idx = nullptr;
    int32_t* sess_vals = nullptr;
    int32_t* ev_vals = nullptr;
    int64_t* rev_vals = nullptr;
    uint8_t* top_cat_bitmap = nullptr;
    size_t num_rows = 0, row_cap = 0;

    uint16_t* cat_idx = nullptr;
    size_t num_cats_nonnull = 0, cat_cap = 0;

    std::vector<std::pair<void*, size_t>> allocs;

    void* track_alloc(size_t bytes, bool hp) {
        void* p = alloc_mmap(bytes, hp);
        allocs.emplace_back(p, alloc_mmap_actual_size(bytes, hp));
        return p;
    }

    void dontneed_all() {
#if defined(__linux__) && defined(MADV_DONTNEED)
        for (auto [p, sz] : allocs) {
            if (p && sz) ::madvise(p, sz, MADV_DONTNEED);
        }
#endif
    }

    void init(size_t est_rows) {
        size_t rcap = std::max<size_t>(est_rows * 2, 131072);
        row_cap = rcap;
        uid_dict_cap = ((rcap / 2) * 16 + 65536 + 63) & ~size_t(63);
        cat_cap = (rcap / 2) + 4096;
        uid_runs.reserve(rcap / 2 + 1024);
        size_t bm_sz = ((rcap >> 3) + 64) & ~size_t(63);
        size_t c_sz  = (cat_cap * 2 + 63) & ~size_t(63);
        // Pack ALL columns into ONE contiguous 2MB-aligned THP slab (1 mmap instead of 3!)
        size_t d_off = 0;
        size_t s_off = (rcap * 2 + 63) & ~size_t(63);
        size_t e_off = s_off + rcap * 4;
        size_t r_off = e_off + rcap * 4;
        size_t b_off = r_off + rcap * 8;
        size_t c_off = b_off + bm_sz;
        size_t u_off = c_off + c_sz;
        size_t total_bytes = u_off + uid_dict_cap;
        char* fmem = static_cast<char*>(track_alloc(total_bytes, total_bytes >= (2ULL << 20)));
        day_idx        = reinterpret_cast<uint16_t*>(fmem + d_off);
        sess_vals      = reinterpret_cast<int32_t*>(fmem + s_off);
        ev_vals        = reinterpret_cast<int32_t*>(fmem + e_off);
        rev_vals       = reinterpret_cast<int64_t*>(fmem + r_off);
        top_cat_bitmap = reinterpret_cast<uint8_t*>(fmem + b_off);
        cat_idx        = reinterpret_cast<uint16_t*>(fmem + c_off);
        uid_dict       = reinterpret_cast<uint8_t*>(fmem + u_off);
    }

    SS_COLD_NOINLINE void grow_rows() {
        size_t new_cap = row_cap * 2;
        size_t bm_sz = ((new_cap >> 3) + 64) & ~size_t(63);
        size_t d_off = 0;
        size_t s_off = (new_cap * 2 + 63) & ~size_t(63);
        size_t e_off = s_off + new_cap * 4;
        size_t r_off = e_off + new_cap * 4;
        size_t b_off = r_off + new_cap * 8;
        size_t fixed_bytes = b_off + bm_sz;
        char* fmem = static_cast<char*>(track_alloc(fixed_bytes, fixed_bytes >= (2ULL << 20)));
        auto* nd = reinterpret_cast<uint16_t*>(fmem + d_off);
        auto* ns = reinterpret_cast<int32_t*>(fmem + s_off);
        auto* ne = reinterpret_cast<int32_t*>(fmem + e_off);
        auto* nr = reinterpret_cast<int64_t*>(fmem + r_off);
        auto* nb = reinterpret_cast<uint8_t*>(fmem + b_off);
        std::memcpy(nd, day_idx, num_rows * 2);
        std::memcpy(ns, sess_vals, num_rows * 4);
        std::memcpy(ne, ev_vals, num_rows * 4);
        std::memcpy(nr, rev_vals, num_rows * 8);
        std::memcpy(nb, top_cat_bitmap, (num_rows + 7) >> 3);
        day_idx = nd; sess_vals = ns; ev_vals = ne; rev_vals = nr; top_cat_bitmap = nb;
        row_cap = new_cap;
    }

    SS_COLD_NOINLINE void grow_uid_dict(size_t need) {
        size_t new_cap = std::max(uid_dict_cap * 2, uid_dict_sz + need + 65536);
        auto* np = static_cast<uint8_t*>(track_alloc(new_cap, new_cap >= (2ULL << 20)));
        if (uid_dict_sz) std::memcpy(np, uid_dict, uid_dict_sz);
        uid_dict = np;
        uid_dict_cap = new_cap;
    }

    SS_COLD_NOINLINE void grow_cat_idx() {
        size_t new_cap = cat_cap * 2;
        auto* np = static_cast<uint16_t*>(track_alloc(new_cap * 2, new_cap * 2 >= (2ULL << 20)));
        if (num_cats_nonnull) std::memcpy(np, cat_idx, num_cats_nonnull * 2);
        cat_idx = np;
        cat_cap = new_cap;
    }

    // Write canonical user_id ONCE per person + run count!
    SS_ALWAYS_INLINE void emit_person_uid(const char* uid_pre, uint32_t uid_pre_len, uint32_t run_rows) {
        if (__builtin_expect(uid_dict_sz + uid_pre_len + 16 > uid_dict_cap, 0)) grow_uid_dict(uid_pre_len + 16);
        if (__builtin_expect(uid_pre_len <= 16, 1)) {
            std::memcpy(uid_dict + uid_dict_sz, uid_pre, 16);
        } else {
            std::memcpy(uid_dict + uid_dict_sz, uid_pre, uid_pre_len);
        }
        uid_dict_sz += uid_pre_len;
        uid_runs.push_back(run_rows);
    }

    SS_ALWAYS_INLINE void push_day_row(
        uint16_t rel_day, int32_t sessions, int32_t events, int64_t rev, int32_t best_cat
    ) {
        if (__builtin_expect(num_rows >= row_cap, 0)) grow_rows();
        size_t r = num_rows++;
        day_idx[r] = rel_day;
        sess_vals[r] = sessions;
        ev_vals[r] = events;
        rev_vals[r] = rev;
        if (__builtin_expect(best_cat >= 0, 0)) {
            top_cat_bitmap[r >> 3] |= uint8_t(1U << (r & 7));
            if (__builtin_expect(num_cats_nonnull >= cat_cap, 0)) grow_cat_idx();
            cat_idx[num_cats_nonnull++] = uint16_t(best_cat);
        }
    }
};

// ============================================================================
// Ultra-Fast 32-Bit Event Sorting: Branchless CMOV Networks <=8 + Ins <=20 + MSD32 + MSD256 + LSD
// ============================================================================
static SS_ALWAYS_INLINE void cswap_u32(uint32_t& a, uint32_t& b) {
    uint32_t lo = a < b ? a : b;
    uint32_t hi = a < b ? b : a;
    a = lo; b = hi;
}

static SS_ALWAYS_INLINE void sort_small_u32(uint32_t* a, uint32_t n) {
    switch (n) {
        case 0: case 1: return;
        case 2: cswap_u32(a[0], a[1]); return;
        case 3:
            cswap_u32(a[0], a[1]); cswap_u32(a[1], a[2]); cswap_u32(a[0], a[1]); return;
        case 4:
            cswap_u32(a[0], a[1]); cswap_u32(a[2], a[3]);
            cswap_u32(a[0], a[2]); cswap_u32(a[1], a[3]);
            cswap_u32(a[1], a[2]); return;
        case 5:
            cswap_u32(a[0], a[1]); cswap_u32(a[3], a[4]);
            cswap_u32(a[2], a[4]); cswap_u32(a[2], a[3]);
            cswap_u32(a[1], a[4]); cswap_u32(a[0], a[3]);
            cswap_u32(a[0], a[2]); cswap_u32(a[1], a[3]);
            cswap_u32(a[1], a[2]); return;
        case 6:
            cswap_u32(a[1], a[2]); cswap_u32(a[4], a[5]);
            cswap_u32(a[0], a[2]); cswap_u32(a[3], a[5]);
            cswap_u32(a[0], a[1]); cswap_u32(a[3], a[4]);
            cswap_u32(a[2], a[5]); cswap_u32(a[0], a[3]);
            cswap_u32(a[1], a[4]); cswap_u32(a[2], a[4]);
            cswap_u32(a[1], a[3]); cswap_u32(a[2], a[3]); return;
        case 7:
            cswap_u32(a[1], a[2]); cswap_u32(a[3], a[4]); cswap_u32(a[5], a[6]);
            cswap_u32(a[0], a[2]); cswap_u32(a[3], a[5]); cswap_u32(a[4], a[6]);
            cswap_u32(a[0], a[1]); cswap_u32(a[4], a[5]); cswap_u32(a[2], a[6]);
            cswap_u32(a[0], a[4]); cswap_u32(a[1], a[5]); cswap_u32(a[0], a[3]);
            cswap_u32(a[2], a[5]); cswap_u32(a[1], a[3]); cswap_u32(a[2], a[4]);
            cswap_u32(a[2], a[3]); return;
        case 8:
            cswap_u32(a[0], a[1]); cswap_u32(a[2], a[3]); cswap_u32(a[4], a[5]); cswap_u32(a[6], a[7]);
            cswap_u32(a[0], a[2]); cswap_u32(a[1], a[3]); cswap_u32(a[4], a[6]); cswap_u32(a[5], a[7]);
            cswap_u32(a[1], a[2]); cswap_u32(a[5], a[6]); cswap_u32(a[0], a[4]); cswap_u32(a[3], a[7]);
            cswap_u32(a[1], a[5]); cswap_u32(a[2], a[6]); cswap_u32(a[1], a[4]); cswap_u32(a[3], a[6]);
            cswap_u32(a[2], a[4]); cswap_u32(a[3], a[5]); cswap_u32(a[3], a[4]); return;
        default:
            for (uint32_t i = 1; i < n; ++i) {
                uint32_t x = a[i];
                uint32_t j = i;
                while (j > 0 && a[j - 1] > x) { a[j] = a[j - 1]; --j; }
                a[j] = x;
            }
            return;
    }
}

static SS_COLD_NOINLINE void radix_sort_u32(uint32_t* arr, uint32_t n, std::vector<uint32_t>& tmp) {
    tmp.resize(n);
    uint32_t c0[2048]{}, c1[2048]{}, c2[1024]{};
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t k = arr[i];
        c0[k & 2047U]++;
        c1[(k >> 11) & 2047U]++;
        c2[k >> 22]++;
    }
    uint32_t s0 = 0, s1 = 0, s2 = 0;
    for (int i = 0; i < 2048; ++i) { uint32_t t = c0[i]; c0[i] = s0; s0 += t; }
    for (int i = 0; i < 2048; ++i) { uint32_t t = c1[i]; c1[i] = s1; s1 += t; }
    for (int i = 0; i < 1024; ++i) { uint32_t t = c2[i]; c2[i] = s2; s2 += t; }
    uint32_t* dst = tmp.data();
    for (uint32_t i = 0; i < n; ++i) { uint32_t v = arr[i]; dst[c0[v & 2047U]++] = v; }
    for (uint32_t i = 0; i < n; ++i) { uint32_t v = dst[i]; arr[c1[(v >> 11) & 2047U]++] = v; }
    for (uint32_t i = 0; i < n; ++i) { uint32_t v = arr[i]; dst[c2[v >> 22]++] = v; }
    std::memcpy(arr, dst, size_t(n) * sizeof(uint32_t));
}

static SS_NOINLINE void sort_msd256_u32(uint32_t* arr, uint32_t n, uint32_t shift8, std::vector<uint32_t>& tmp) {
    if (__builtin_expect(n >= 65536, 0)) {
        radix_sort_u32(arr, n, tmp);
        return;
    }
    if (tmp.size() < n) tmp.resize(n);
    uint16_t cnt[256]{};
    uint32_t bmin = 255, bmax = 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t b = (arr[i] >> shift8) & 255U;
        cnt[b]++;
        if (b < bmin) bmin = b;
        if (b > bmax) bmax = b;
    }
    if (__builtin_expect(bmin == bmax, 0)) { std::sort(arr, arr + n); return; }
    uint16_t off[256];
    uint16_t acc = 0;
    for (uint32_t b = bmin; b <= bmax; ++b) { off[b] = acc; acc = uint16_t(acc + cnt[b]); }
    uint32_t* dst = tmp.data();
    for (uint32_t i = 0; i < n; ++i) { uint32_t v = arr[i]; dst[off[(v >> shift8) & 255U]++] = v; }
    uint32_t st = 0;
    for (uint32_t b = bmin; b <= bmax; ++b) {
        uint32_t c = cnt[b];
        if (c > 1) {
            if (c <= 20) sort_small_u32(dst + st, c);
            else std::sort(dst + st, dst + st + c);
        }
        st += c;
    }
    std::memcpy(arr, dst, size_t(n) * sizeof(uint32_t));
}

static SS_ALWAYS_INLINE void sort_events_u32(uint32_t* arr, uint32_t n, uint32_t shift8, uint32_t shift5, std::vector<uint32_t>& tmp) {
    if (n <= 20) {
        sort_small_u32(arr, n);
        return;
    }
    if (__builtin_expect(n < 256, 1)) {
        uint8_t cnt[32]{};
        uint32_t non_empty = 0;
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t b = (arr[i] >> shift5) & 31U;
            cnt[b]++;
            non_empty |= (1U << b);
        }
        uint8_t off[32];
        uint8_t acc = 0;
        uint32_t m = non_empty;
        while (m) {
            uint32_t b = uint32_t(__builtin_ctz(m));
            m &= m - 1;
            off[b] = acc;
            acc = uint8_t(acc + cnt[b]);
        }
        uint32_t dst[256];
        for (uint32_t i = 0; i < n; ++i) { uint32_t v = arr[i]; dst[off[(v >> shift5) & 31U]++] = v; }
        uint32_t st = 0;
        m = non_empty;
        while (m) {
            uint32_t b = uint32_t(__builtin_ctz(m));
            m &= m - 1;
            uint32_t c = cnt[b];
            if (c > 1) {
                if (c <= 20) sort_small_u32(dst + st, c);
                else std::sort(dst + st, dst + st + c);
            }
            st += c;
        }
        std::memcpy(arr, dst, size_t(n) * sizeof(uint32_t));
        return;
    }
    sort_msd256_u32(arr, n, shift8, tmp);
}

static void mkdir_p(const std::string& path) {
    size_t pos = 0;
    while ((pos = path.find('/', pos + 1)) != std::string::npos) {
        ::mkdir(path.substr(0, pos).c_str(), 0755);
    }
    ::mkdir(path.c_str(), 0755);
}

// ============================================================================
// Split L1I-Hot Per-Task Scanners (Fast-Dict + Wide/Classic Paths!)
// ============================================================================
struct ClickRingFast {
    uint64_t u48;
    uint64_t h48;
    uint64_t cid;
    uint64_t ts_camp44; // (uint64_t(ts - click_start_ms) << 12) | camp_idx
};
struct EvRingFast {
    uint64_t u48;
    uint64_t h48;
    uint64_t rem40;
    uint32_t eb;
    uint32_t irel21;
    uint32_t ts_rel_ms;
};
static constexpr size_t RING_SZ   = 32;
static constexpr size_t RING_MASK = RING_SZ - 1;
static constexpr size_t RING_LAG  = 24;

static SS_NOINLINE void process_click_task_fast(
    const CsvTask& task,
    ReadBuf& rbuf,
    int tid,
    const Campaigns& campaigns,
    const DateLut& lut,
    UserDict& dict,
    StringArenaPool& arenas,
    int64_t click_start_ms,
    int64_t win_end_ms,
    PartitionedStore<CompactClick, 128, PERSON_BUCKETS, 4>::ThreadState& clk_out,
    ClickRingFast cring[RING_SZ],
    size_t& cpushed,
    uint32_t& tl_cur,
    uint32_t& tl_end
) {
    auto emit_click = [&](const ClickRingFast& r) SS_LAMBDA_INLINE {
        uint32_t pid = dict.get_or_insert_uid48(r.u48, r.h48, tl_cur, tl_end);
        uint32_t pb = pid & uint32_t(PERSON_BUCKETS - 1);
        uint32_t pid_hi = pid >> PERSON_SHIFT;
        clk_out.push(pb, CompactClick{r.cid, (uint64_t(pid_hi) << 44) | r.ts_camp44});
    };

    scan_csv<4>(task, rbuf, [&](const Slice f[4]) SS_LAMBDA_INLINE {
        if (__builtin_expect((f[0].len > 0) & (f[1].len > 0) & (f[2].len > 0) & (f[3].len > 0), 1)) {
            uint32_t camp_idx = campaigns.lookup_padded(f[2].ptr, f[2].len);
            if (__builtin_expect(camp_idx != FastStringIdMap::NOT_FOUND, 1)) {
                int64_t ts = 0;
                if (__builtin_expect(parse_ts(f[3].ptr, f[3].len, ts, lut) &&
                                     ts >= click_start_ms && ts < win_end_ms, 1)) {
                    uint64_t cid = encode_key64_padded(f[0].ptr, f[0].len, arenas, tid);
                    uint64_t ts_camp44 = (uint64_t(uint32_t(ts - click_start_ms)) << 12) | uint64_t(camp_idx);
                    uint64_t u48 = 0;
                    if (__builtin_expect(encode_uid48_padded(f[1].ptr, f[1].len, u48), 1)) {
                        uint64_t h48 = mix48(u48);
                        cring[cpushed & RING_MASK] = {u48, h48, cid, ts_camp44};
                        dict.prefetch_uslot(h48);
                        if (__builtin_expect(cpushed >= RING_LAG, 1)) {
                            emit_click(cring[(cpushed - RING_LAG) & RING_MASK]);
                        }
                        ++cpushed;
                    } else {
                        uint64_t ukey = encode_key64_padded(f[1].ptr, f[1].len, arenas, tid);
                        uint32_t pid = dict.get_or_insert_non_uid48(ukey, arenas);
                        uint32_t pb = pid & uint32_t(PERSON_BUCKETS - 1);
                        clk_out.push(pb, CompactClick{cid, (uint64_t(pid >> PERSON_SHIFT) << 44) | ts_camp44});
                    }
                }
            }
        }
    });
}

static SS_NOINLINE void process_event_task_fast(
    const CsvTask& task,
    ReadBuf& rbuf,
    int tid,
    const Products& products,
    const DateLut& lut,
    UserDict& dict,
    StringArenaPool& arenas,
    int64_t win_start_ms,
    int64_t win_start_sec,
    int64_t win_dur_ms,
    int64_t win_dur_sec,
    PartitionedStore<RawEvent16, 256, EV_BUCKETS, 4>::ThreadState& ev_out,
    std::vector<SideEvent>& side_vec,
    EvRingFast ering[RING_SZ],
    size_t& epushed,
    uint32_t& tl_cur,
    uint32_t& tl_end
) {
    uint32_t view_u32, cart_u32;
    uint64_t pur_u64;
    std::memcpy(&view_u32, "view", 4);
    std::memcpy(&cart_u32, "cart", 4);
    std::memcpy(&pur_u64, "purchase", 8);

    auto emit_ev = [&](const EvRingFast& r) SS_LAMBDA_INLINE {
        uint32_t pid = dict.get_or_insert_uid48(r.u48, r.h48, tl_cur, tl_end);
        ev_out.push(r.eb, RawEvent16::make_inline_pid(r.irel21, r.rem40, r.ts_rel_ms, pid));
    };

    auto push_side_ev = [&](bool valid, uint32_t irel21, bool eid_in_side, uint32_t eb, uint64_t rem40,
                            uint64_t eid_k64, uint32_t pid, int64_t ts_ms, int64_t ing_ms,
                            uint32_t price, uint32_t cat_id) SS_COLD_NOINLINE {
        uint32_t sidx = uint32_t(side_vec.size());
        if (__builtin_expect(sidx >= (1U << 26), 0)) std::abort();
        side_vec.push_back({eid_k64, 0ULL, ts_ms, ing_ms, pid, price, cat_id, 0U});
        ev_out.push(eb, RawEvent16::make_side(valid, irel21, eid_in_side, rem40, tid, sidx));
    };

    scan_csv<7>(task, rbuf, [&](const Slice f[7]) SS_LAMBDA_INLINE {
        if (__builtin_expect(f[0].len == 0, 0)) return;

        // 1. Parse ingest_ts + cache 10-byte YYYY-MM-DD for same-day F1 ts reuse!
        int64_t ing_rel_sec = 0;
        int64_t ing_ms = 0;
        bool ing_whole_sec = true;
        uint64_t ing_ym = 0;
        uint16_t ing_dd16 = 0;
        int64_t ing_day_rel = 0;
        if (__builtin_expect(f[3].len == 20 && f[3].ptr[19] == 'Z', 1)) {
            if (__builtin_expect(!lut.parse_ingest_f1_rel_sec_day(f[3].ptr, ing_rel_sec, ing_ym, ing_dd16, ing_day_rel), 0)) return;
        } else {
            if (!parse_ts(f[3].ptr, f[3].len, ing_ms, lut)) return;
            ing_whole_sec = (ing_ms % 1000LL == 0);
            ing_rel_sec = ing_ms / 1000LL - win_start_sec;
        }

        // 2. Encode event_id -> 48b bijection (eb = top 9b, rem39 = low 39b)
        uint64_t eid48 = 0, eid_k64 = 0, rem40 = 0;
        uint32_t eb = 0;
        bool eid_in_side = false;
        if (__builtin_expect(encode_eid48_padded(f[0].ptr, f[0].len, eid48), 1)) {
            uint64_t h48 = mix48(eid48);
            eb = uint32_t(h48 >> 39);
            rem40 = h48 & RawEvent16::REM39_MASK;
        } else {
            eid_in_side = true;
            eid_k64 = encode_key64_padded(f[0].ptr, f[0].len, arenas, tid);
            uint64_t h64 = hash_key64(eid_k64, arenas);
            eb = uint32_t(h64 >> 55);
            rem40 = h64 & RawEvent16::REM39_MASK;
        }

        uint32_t irel21 = 0U;
        if (__builtin_expect(ing_whole_sec, 1)) {
            int64_t ir = ing_rel_sec + RawEvent16::ING_BIAS;
            if (__builtin_expect(ir > 0 && ir <= RawEvent16::ING_MAX, 1)) irel21 = uint32_t(ir);
        }

        // 3. Branchless event_type + non-empty user_id/product_id check
        uint32_t v4 = 0;
        uint64_t v8 = 0;
        std::memcpy(&v4, f[4].ptr, 4);
        std::memcpy(&v8, f[4].ptr, 8);
        bool is_vc = (f[4].len == 4) & ((v4 == view_u32) | (v4 == cart_u32));
        bool is_pur = (f[4].len == 8) & (v8 == pur_u64);
        bool ok = (f[1].len > 0) & (f[5].len > 0) & (is_vc | is_pur);

        // 4. Parse ts (date-reused F1, zero-div F3 with exact whole-sec late check, sign-checked F2!)
        int64_t ts_rel_sec = 0;
        int64_t ts_ms = 0;
        uint32_t ts_ms_rem = 0;
        bool ts_from_sec = true;
        if (__builtin_expect(ok, 1)) {
            bool ts_ok = false;
            if (f[2].len == 20 && f[2].ptr[19] == 'Z') {
                ts_ok = lut.parse_iso19_rel_sec_reuse(f[2].ptr, ing_ym, ing_dd16, ing_day_rel, ts_rel_sec);
            } else if (f[2].len == 13) {
                int64_t ts_sec = 0;
                ts_ok = parse_13d_sec_ms(f[2].ptr, ts_sec, ts_ms_rem);
                ts_rel_sec = ts_sec - win_start_sec;
            } else if (f[2].len == 25 && (f[2].ptr[19] == '+' || f[2].ptr[19] == '-')) {
                ts_ok = lut.parse_f2_rel_sec(f[2].ptr, ts_rel_sec);
            } else {
                ts_ok = parse_ts_slow(f[2].ptr, f[2].len, ts_ms, lut);
                ts_from_sec = false;
            }
            if (__builtin_expect(ts_ok & ts_from_sec & ing_whole_sec, 1)) {
                // Exact even with ts_ms_rem in [0..999] when win_start_ms & win_end_ms are whole-second
                // AND ingest_ts is whole-second: ing_sec*1000 - (ts_sec*1000 + rem) > 86400000 <=> ing_sec - ts_sec > 86400!
                if (__builtin_expect(uint64_t(ts_rel_sec) >= uint64_t(win_dur_sec) ||
                                     ing_rel_sec - ts_rel_sec > 86400LL, 0)) {
                    ok = false;
                }
            } else {
                if (!ts_ok) {
                    ok = false;
                } else {
                    if (ts_from_sec) ts_ms = (win_start_sec + ts_rel_sec) * 1000LL + int64_t(ts_ms_rem);
                    if (ing_whole_sec) ing_ms = (win_start_sec + ing_rel_sec) * 1000LL;
                    if (uint64_t(ts_ms - win_start_ms) >= uint64_t(win_dur_ms) || ing_ms - ts_ms > DAY_MS) {
                        ok = false;
                    }
                }
            }
        }

        // 5. FAST PATH (~96% of all rows — INCLUDES ALL F3 millisecond rows!!):
        if (__builtin_expect(ok & is_vc & (!eid_in_side) & (irel21 != 0), 1)) {
            uint32_t ts_rel_ms = ts_from_sec
                ? uint32_t(ts_rel_sec * 1000LL + int64_t(ts_ms_rem))
                : uint32_t(ts_ms - win_start_ms);
            uint64_t u48 = 0;
            if (__builtin_expect(encode_uid48_padded(f[1].ptr, f[1].len, u48), 1)) {
                uint64_t h48 = mix48(u48);
                ering[epushed & RING_MASK] = {u48, h48, rem40, eb, irel21, ts_rel_ms};
                dict.prefetch_uslot(h48);
                if (__builtin_expect(epushed >= RING_LAG, 1)) {
                    emit_ev(ering[(epushed - RING_LAG) & RING_MASK]);
                }
                ++epushed;
                return;
            }
            uint64_t ukey = encode_key64_padded(f[1].ptr, f[1].len, arenas, tid);
            uint32_t pid = dict.get_or_insert_non_uid48(ukey, arenas);
            ev_out.push(eb, RawEvent16::make_inline_pid(irel21, rem40, ts_rel_ms, pid));
            return;
        }

        // 5b. Invalid-inline shortcut (zero SideEvent allocation for invalid standard rows!)
        if (!ok && !eid_in_side && irel21 != 0) {
            ev_out.push(eb, RawEvent16::make_invalid_inline(irel21, rem40));
            return;
        }

        // 6. SIDE PATH (~3% — purchases + rare non-eid48 / out-of-range ingest!)
        if (ing_whole_sec) ing_ms = (win_start_sec + ing_rel_sec) * 1000LL;
        if (ts_from_sec) ts_ms = (win_start_sec + ts_rel_sec) * 1000LL + int64_t(ts_ms_rem);
        uint32_t price = 0, cat_id = 0;
        if (ok && is_pur) {
            if (!parse_price_cents(f[6].ptr, f[6].len, price)) {
                ok = false;
                if (!eid_in_side && irel21 != 0) {
                    ev_out.push(eb, RawEvent16::make_invalid_inline(irel21, rem40));
                    return;
                }
            } else {
                cat_id = products.lookup(f[5].ptr, f[5].len);
            }
        }
        uint32_t pid = 0;
        if (ok) {
            uint64_t u48 = 0;
            if (__builtin_expect(encode_uid48_padded(f[1].ptr, f[1].len, u48), 1)) {
                pid = dict.get_or_insert_uid48(u48, mix48(u48), tl_cur, tl_end);
            } else {
                uint64_t ukey = encode_key64_padded(f[1].ptr, f[1].len, arenas, tid);
                pid = dict.get_or_insert_non_uid48(ukey, arenas);
            }
        }
        push_side_ev(ok, irel21, eid_in_side, eb, rem40, eid_k64, pid, ts_ms, ing_ms, price, cat_id);
    });
}

// Wide / Classic fallback scanners (arbitrary UTF-8 links, >49.7d windows, >4096 campaigns, >16.7M users!)
static SS_NOINLINE void process_click_task_wide(
    const CsvTask& task,
    ReadBuf& rbuf,
    int tid,
    const Campaigns& campaigns,
    const DateLut& lut,
    const UserDict& dict,
    StringArenaPool& arenas,
    int64_t click_start_ms,
    int64_t win_end_ms,
    PartitionedStore<PersonClick, 128, PERSON_BUCKETS, 4>::ThreadState& clk_out
) {
    const bool classic_links = (dict.cap > 0);
    scan_csv<4>(task, rbuf, [&](const Slice f[4]) SS_LAMBDA_INLINE {
        if (__builtin_expect((f[0].len > 0) & (f[1].len > 0) & (f[2].len > 0) & (f[3].len > 0), 1)) {
            uint32_t camp_idx = campaigns.lookup_padded(f[2].ptr, f[2].len);
            if (__builtin_expect(camp_idx != FastStringIdMap::NOT_FOUND, 1)) {
                int64_t ts = 0;
                if (__builtin_expect(parse_ts(f[3].ptr, f[3].len, ts, lut) &&
                                     ts >= click_start_ms && ts < win_end_ms, 1)) {
                    uint64_t cid = encode_key64_padded(f[0].ptr, f[0].len, arenas, tid);
                    uint64_t ts_camp = (uint64_t(ts - click_start_ms) << 20) | uint64_t(camp_idx);
                    uint64_t ukey = encode_key64_padded(f[1].ptr, f[1].len, arenas, tid);
                    uint64_t pkey = classic_links ? dict.resolve_classic(ukey, arenas) : ukey;
                    uint32_t pb = uint32_t(hash_key64(pkey, arenas) >> (64 - PERSON_SHIFT));
                    clk_out.push(pb, PersonClick{pkey, cid, ts_camp});
                }
            }
        }
    });
}

static SS_NOINLINE void process_event_task_wide(
    const CsvTask& task,
    ReadBuf& rbuf,
    int tid,
    const Products& products,
    const DateLut& lut,
    StringArenaPool& arenas,
    int64_t win_start_ms,
    int64_t win_start_sec,
    int64_t win_dur_ms,
    PartitionedStore<RawEvent16, 256, EV_BUCKETS, 4>::ThreadState& ev_out,
    std::vector<SideEvent>& side_vec
) {
    uint32_t view_u32, cart_u32;
    uint64_t pur_u64;
    std::memcpy(&view_u32, "view", 4);
    std::memcpy(&cart_u32, "cart", 4);
    std::memcpy(&pur_u64, "purchase", 8);

    scan_csv<7>(task, rbuf, [&](const Slice f[7]) SS_LAMBDA_INLINE {
        if (__builtin_expect(f[0].len == 0, 0)) return;
        int64_t ing_ms = 0;
        if (!parse_ts(f[3].ptr, f[3].len, ing_ms, lut)) return;
        uint32_t irel21 = 0U;
        if (ing_ms % 1000LL == 0) {
            int64_t ir = (ing_ms / 1000LL - win_start_sec) + RawEvent16::ING_BIAS;
            if (ir > 0 && ir <= RawEvent16::ING_MAX) irel21 = uint32_t(ir);
        }
        uint64_t eid48 = 0, eid_k64 = 0, rem40 = 0;
        uint32_t eb = 0;
        bool eid_in_side = false;
        if (encode_eid48_padded(f[0].ptr, f[0].len, eid48)) {
            uint64_t h48 = mix48(eid48);
            eb = uint32_t(h48 >> 39);
            rem40 = h48 & RawEvent16::REM39_MASK;
        } else {
            eid_in_side = true;
            eid_k64 = encode_key64_padded(f[0].ptr, f[0].len, arenas, tid);
            uint64_t h64 = hash_key64(eid_k64, arenas);
            eb = uint32_t(h64 >> 55);
            rem40 = h64 & RawEvent16::REM39_MASK;
        }
        uint32_t v4 = 0; uint64_t v8 = 0;
        std::memcpy(&v4, f[4].ptr, 4); std::memcpy(&v8, f[4].ptr, 8);
        bool is_vc = (f[4].len == 4) & ((v4 == view_u32) | (v4 == cart_u32));
        bool is_pur = (f[4].len == 8) & (v8 == pur_u64);
        bool ok = (f[1].len > 0) & (f[5].len > 0) & (is_vc | is_pur);
        int64_t ts_ms = 0;
        if (ok) {
            if (!parse_ts(f[2].ptr, f[2].len, ts_ms, lut) ||
                ts_ms < win_start_ms || ts_ms >= win_start_ms + win_dur_ms ||
                ing_ms - ts_ms > DAY_MS) {
                ok = false;
            }
        }
        uint32_t price = 0, cat_id = 0;
        if (ok && is_pur) {
            if (!parse_price_cents(f[6].ptr, f[6].len, price)) ok = false;
            else cat_id = products.lookup(f[5].ptr, f[5].len);
        }
        uint64_t ukey = ok ? encode_key64_padded(f[1].ptr, f[1].len, arenas, tid) : 0ULL;
        uint32_t sidx = uint32_t(side_vec.size());
        if (sidx >= (1U << 26)) std::abort();
        side_vec.push_back({eid_k64, ukey, ts_ms, ing_ms, 0U, price, cat_id, 0U});
        ev_out.push(eb, RawEvent16::make_side(ok, irel21, eid_in_side, rem40, tid, sidx));
    });
}

// ============================================================================
// Split Out-of-Line Campaign Daily Merger + Writer (Shrinks sess_worker L1I!)
// ============================================================================
static SS_NOINLINE bool write_campaign_daily_and_empty_ud(
    const std::string& ud_dir,
    const std::string& cd_dir,
    bool need_empty_ud,
    int nthreads,
    int32_t win_start_day,
    int32_t num_days,
    const Campaigns& campaigns,
    std::vector<std::vector<uint32_t>>& thread_cd_assist,
    std::vector<std::vector<CredCell>>& thread_cd_cred
) {
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

    bool ok = true;
    if (need_empty_ud) {
        std::string path = ud_dir + "/part-0.parquet";
        pq::ColData empty_ud[6]{};
        if (!pq::write_parquet_file(path.c_str(), UD_COLS, empty_ud, 6, 0)) ok = false;
    }

    const uint32_t camp_stride = campaigns.organic_idx + 1;
    const size_t nd = size_t(num_days);
    const size_t n_cells = size_t(camp_stride) * nd;
    auto& agg_a = thread_cd_assist[0];
    auto& agg_c = thread_cd_cred[0];
    for (int t = 1; t < nthreads; ++t) {
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
    if (!pq::write_parquet_file(path.c_str(), CD_COLS, data, 7, int64_t(day_v.size()))) ok = false;
    return ok;
}

// ============================================================================
// Stage 2C + Stage 3 + Stage 4 FUSED — Fast-Dict Mode (Dense pid, 8B PEv, 4B Sort, Zero pht!)
// ============================================================================
struct FastLocalClick {
    uint32_t ts_off;
    uint32_t camp_idx;
    uint64_t click_id;
};

static SS_NOINLINE bool run_fast_dict_dedup_sessionize_and_write(
    PartitionedStore<RawEvent16, 256, EV_BUCKETS, 4>& event_store,
    const std::vector<std::vector<SideEvent>>& sides,
    PartitionedStore<CompactClick, 128, PERSON_BUCKETS, 4>& click_store,
    UserDict& dict,
    const Products& products,
    const Campaigns& campaigns,
    const StringArenaPool& arenas,
    ThreadPool& pool,
    int64_t win_start_ms,
    int64_t win_end_ms,
    int64_t win_start_sec,
    int32_t win_start_day,
    int32_t num_days,
    const std::string& ud_dir,
    const std::string& cd_dir,
    bool prof
) {
    using EvStore   = PartitionedStore<RawEvent16, 256, EV_BUCKETS, 4>;
    using PEv8Store = PartitionedStore<CompactPersonEv8, 512, PERSON_BUCKETS, 8>;
    using PurStore  = PartitionedStore<CompactPurchase, 64, PERSON_BUCKETS, 1>;
    static_assert(sizeof(EvStore::Block) == sizeof(PEv8Store::Block), "Block sizes must match for zero-fault recycling");

    PEv8Store person_ev_store;
    PurStore purchase_store;
    person_ev_store.init(pool.nthreads);
    purchase_store.init(pool.nthreads, 4ULL << 20);

    auto now_ms = []() {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    double td0 = now_ms();

    uint32_t msd_shift8 = 23, msd_shift5 = 26;
    {
        uint64_t dur = (win_end_ms > win_start_ms) ? uint64_t(win_end_ms - win_start_ms) : 1ULL;
        uint32_t bits = 64U - uint32_t(__builtin_clzll(dur));
        msd_shift8 = (bits > 8U) ? (bits - 8U) : 0U;
        msd_shift5 = (bits > 5U) ? (bits - 5U) : 0U;
    }

    // Stage 2C: ZERO-mix64 L2-Hot Dedup + ZERO-LinkMap 8B Filtered NT Copy + 4KB Block Recycling!
    using EvBlock = typename EvStore::Block;
    std::atomic<int> next_eb{0};
    pool.run([&](int tid) {
        struct DedupSlot8 { uint32_t fp; uint32_t handle1; };
        std::vector<uint32_t> ht4;
        std::vector<DedupSlot8> ht8;
        std::vector<EvBlock*> blks;
        ht4.reserve(131072);
        blks.reserve(512);

        auto& out_ev = *person_ev_store.ts[size_t(tid)];
        auto& out_pur = *purchase_store.ts[size_t(tid)];

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

            // Loop 1: Exact dedup on RawEvent16::w0
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
                        uint32_t fp12 = uint32_t((eid40 >> 28) & 0xFFFULL) << 20;
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
                                        ev.w0 = 0;
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

            // Loop 2: ZERO LinkMap probes, ZERO hashing — 10-insn 8B filtered NT copy!
            for (uint32_t bi = 0; bi < nblks; ++bi) {
                EvBlock* blk = blks[bi];
                const uint32_t cnt = blk->count;
                const RawEvent16* items = blk->items;
                for (uint32_t i = 0; i < cnt; ++i) {
                    const RawEvent16& ev = items[i];
                    if (__builtin_expect(!ev.is_valid(), 0)) continue;
                    if (__builtin_expect(!ev.is_side(), 1)) {
                        uint32_t pid = ev.inline_pid();
                        uint32_t ts_rel = ev.inline_ts_rel_ms();
                        out_ev.push(pid & uint32_t(PERSON_BUCKETS - 1), CompactPersonEv8::make(pid >> PERSON_SHIFT, ts_rel));
                    } else {
                        const SideEvent& se = sides[ev.w1 >> 26][ev.w1 & 0x03FFFFFFU];
                        uint32_t pid = se.pid;
                        uint32_t pb = pid & uint32_t(PERSON_BUCKETS - 1);
                        uint32_t pid_hi = pid >> PERSON_SHIFT;
                        uint32_t ts_rel = uint32_t(se.ts_ms - win_start_ms);
                        out_ev.push(pb, CompactPersonEv8::make(pid_hi, ts_rel));
                        if (__builtin_expect(se.price > 0, 1)) {
                            out_pur.push(pb, CompactPurchase{pid_hi, ts_rel, se.price, se.cat_id});
                        }
                    }
                }
                // Recycle dead 4 KB RawEvent16 block directly into person_ev_store free list!
                out_ev.recycle_block(blk);
            }
        }
        out_ev.flush_all();
        out_pur.flush_all();
    });
    double td1 = now_ms();

    // Stage 3 + Stage 4 FUSED: Direct L1D Array Grouping (NO pht, NO ev_pids!) + 4B Sort + 1-Session Fast Path!
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

    const uint32_t max_pid = dict.next_pid.load(std::memory_order_acquire);
    const uint32_t n_local = (max_pid > 0) ? (((max_pid - 1U) >> PERSON_SHIFT) + 1U) : 1U;

    static constexpr pq::ColSpec UD_COLS[6] = {
        {"user_id",       pq::ColKind::STRING_REQ},
        {"day",           pq::ColKind::DATE_REQ},
        {"sessions",      pq::ColKind::INT32_REQ},
        {"events",        pq::ColKind::INT32_REQ},
        {"revenue_cents", pq::ColKind::INT64_REQ},
        {"top_category",  pq::ColKind::STRING_OPT},
    };

    std::vector<FastUserDailyBuffers> thread_ud(static_cast<size_t>(pool.nthreads));
    std::vector<std::vector<uint32_t>> thread_cd_assist(static_cast<size_t>(pool.nthreads));
    std::vector<std::vector<CredCell>> thread_cd_cred(static_cast<size_t>(pool.nthreads));
    std::atomic<size_t> total_ud_rows_written{0};
    std::atomic<int> finished_workers{0};
    std::atomic<bool> write_ok{true};

    const uint32_t num_camps = campaigns.organic_idx;
    const uint32_t camp_stride = num_camps + 1;
    const uint32_t num_cats = uint32_t(products.cat_names.size());
    std::atomic<int> next_pb{0};

    // Precompute tiny shared day & category PLAIN dictionaries for all writer threads!
    std::vector<int32_t> day_dict(static_cast<size_t>(num_days));
    for (int32_t d = 0; d < num_days; ++d) day_dict[size_t(d)] = win_start_day + d;
    std::vector<uint8_t> cat_dict;
    cat_dict.reserve(size_t(num_cats) * 16 + 64);
    for (uint32_t c = 0; c < num_cats; ++c) {
        const std::string& cs = products.cat_names[c];
        uint32_t cl = uint32_t(cs.size());
        size_t old = cat_dict.size();
        cat_dict.resize(old + 4 + cl);
        std::memcpy(cat_dict.data() + old, &cl, 4);
        if (cl) std::memcpy(cat_dict.data() + old + 4, cs.data(), cl);
    }

    const size_t ud_est_rows = total_surviving / size_t(pool.nthreads * 6) + 4096;
    for (int t = 0; t < pool.nthreads; ++t) thread_ud[size_t(t)].init(ud_est_rows);

    pool.run([&](int tid) {
        auto& ud = thread_ud[size_t(tid)];

        const size_t nd = size_t(num_days);
        const size_t n_cells = size_t(camp_stride) * nd;
        auto& cd_assist = thread_cd_assist[size_t(tid)];
        auto& cd_cred = thread_cd_cred[size_t(tid)];
        cd_assist.assign(n_cells, 0);
        cd_cred.assign(n_cells, CredCell{0, 0, 0});

        std::vector<uint32_t> ev_cnt(n_local, 0), ev_off(n_local + 1, 0);
        std::vector<uint32_t> pur_cnt(n_local, 0), pur_off(n_local + 1, 0);
        std::vector<uint32_t> clk_cnt(n_local, 0), clk_off(n_local + 1, 0);
        std::vector<uint32_t> active_pids;
        active_pids.reserve(std::min<uint32_t>(n_local, 8192));

        std::vector<uint32_t> grouped_evs;
        std::vector<uint32_t> sort_tmp;
        std::vector<LocalPurchase> grouped_purs;
        std::vector<FastLocalClick> grouped_clks;

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

            // 1. Direct L1D counting of 8B CompactPersonEv8 (ZERO hash table!)
            active_pids.clear();
            uint32_t* ec = ev_cnt.data();
            for (int st = 0; st < pool.nthreads; ++st) {
                for (auto* blk = person_ev_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                    const uint32_t cnt = blk->count;
                    const CompactPersonEv8* items = blk->items;
                    for (uint32_t i = 0; i < cnt; ++i) {
                        uint32_t p = items[i].pid_hi();
                        if (ec[p]++ == 0) active_pids.push_back(p);
                    }
                }
            }
            uint32_t acc = 0;
            uint32_t* eo = ev_off.data();
            for (uint32_t p : active_pids) { eo[p] = acc; acc += ec[p]; }
            grouped_evs.resize(acc);

            // 2. Direct scatter to 4-byte uint32_t ts_rel_ms!
            uint32_t* ge = grouped_evs.data();
            for (int st = 0; st < pool.nthreads; ++st) {
                for (auto* blk = person_ev_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                    const uint32_t cnt = blk->count;
                    const CompactPersonEv8* items = blk->items;
                    for (uint32_t i = 0; i < cnt; ++i) {
                        CompactPersonEv8 item = items[i];
                        ge[eo[item.pid_hi()]++] = item.ts_rel_ms();
                    }
                }
            }
            acc = 0;
            for (uint32_t p : active_pids) { eo[p] = acc; acc += ec[p]; }

            // 3. Direct L1D grouping of purchases (~3% of events -> ~18 KB/bucket!)
            size_t npur_bucket = purchase_store.bucket_count(b);
            uint32_t* pc = pur_cnt.data();
            uint32_t* po = pur_off.data();
            if (npur_bucket > 0) {
                for (int st = 0; st < pool.nthreads; ++st) {
                    for (auto* blk = purchase_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                        const uint32_t cnt = blk->count;
                        const CompactPurchase* items = blk->items;
                        for (uint32_t i = 0; i < cnt; ++i) pc[items[i].pid_hi]++;
                    }
                }
                uint32_t pacc = 0;
                for (uint32_t p : active_pids) { po[p] = pacc; pacc += pc[p]; }
                grouped_purs.resize(pacc);
                LocalPurchase* gp = grouped_purs.data();
                for (int st = 0; st < pool.nthreads; ++st) {
                    for (auto* blk = purchase_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                        const uint32_t cnt = blk->count;
                        const CompactPurchase* items = blk->items;
                        for (uint32_t i = 0; i < cnt; ++i) {
                            const auto& it = items[i];
                            gp[po[it.pid_hi]++] = {it.ts_rel_ms, it.price, it.cat_id};
                        }
                    }
                }
                pacc = 0;
                for (uint32_t p : active_pids) { po[p] = pacc; pacc += pc[p]; }
            }

            // 4. Direct L1D grouping of clicks (1-load filter skips click-only users, ZERO hash table!)
            size_t nclk_bucket = click_store.bucket_count(b);
            uint32_t* cc = clk_cnt.data();
            uint32_t* co = clk_off.data();
            if (nclk_bucket > 0) {
                for (int st = 0; st < pool.nthreads; ++st) {
                    for (auto* blk = click_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                        const uint32_t cnt = blk->count;
                        const CompactClick* items = blk->items;
                        for (uint32_t i = 0; i < cnt; ++i) {
                            uint32_t p = items[i].pid_hi();
                            if (__builtin_expect(ec[p] != 0, 1)) cc[p]++;
                        }
                    }
                }
                uint32_t cacc = 0;
                for (uint32_t p : active_pids) { co[p] = cacc; cacc += cc[p]; }
                grouped_clks.resize(cacc);
                if (cacc > 0) {
                    FastLocalClick* gc = grouped_clks.data();
                    for (int st = 0; st < pool.nthreads; ++st) {
                        for (auto* blk = click_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                            const uint32_t cnt = blk->count;
                            const CompactClick* items = blk->items;
                            for (uint32_t i = 0; i < cnt; ++i) {
                                uint32_t p = items[i].pid_hi();
                                if (__builtin_expect(ec[p] != 0, 1)) {
                                    gc[co[p]++] = {items[i].ts_off(), items[i].camp_idx(), items[i].click_id};
                                }
                            }
                        }
                    }
                    cacc = 0;
                    for (uint32_t p : active_pids) { co[p] = cacc; cacc += cc[p]; }
                }
            }

            // 5. Sessionize + Attribute + Emit for each active person in bucket b!
            for (uint32_t p : active_pids) {
                uint32_t es = eo[p];
                uint32_t en = ec[p];
                uint32_t ps = (npur_bucket > 0) ? po[p] : 0U;
                uint32_t pn = (npur_bucket > 0) ? pc[p] : 0U;
                uint32_t cs = (nclk_bucket > 0) ? co[p] : 0U;
                uint32_t cn = (nclk_bucket > 0) ? cc[p] : 0U;

                // Reset per-pid counters right away (stays L1D hot, zero full-array memset!)
                ec[p] = 0;
                if (pn) pc[p] = 0;
                if (cn) cc[p] = 0;

                // Decode person's canonical user_id ONCE from pid_name!
                alignas(16) char uid_pre_inline[16]{};
                const char* uid_pre_ptr = uid_pre_inline;
                uint32_t uid_pre_len = 0;
                uint32_t full_pid = (p << PERSON_SHIFT) | uint32_t(b);
                uint64_t nm = dict.pid_name[full_pid];
                if (__builtin_expect((nm >> 56) == 0, 1)) {
                    uint32_t ulen = uint32_t(decode_uid48_inline(nm, uid_pre_inline + 4));
                    std::memcpy(uid_pre_inline, &ulen, 4);
                    uid_pre_len = 4 + ulen;
                } else if (__builtin_expect((nm & FALLBACK_BIT) == 0, 1)) {
                    uint32_t ulen = uint32_t(decode_inline(nm, uid_pre_inline + 4));
                    std::memcpy(uid_pre_inline, &ulen, 4);
                    uid_pre_len = 4 + ulen;
                } else {
                    auto sv = arenas.get_fallback(nm);
                    uint32_t ulen = uint32_t(sv.size());
                    fb_uid_buf.resize(4 + ulen + 16);
                    std::memcpy(fb_uid_buf.data(), &ulen, 4);
                    if (ulen) std::memcpy(fb_uid_buf.data() + 4, sv.data(), ulen);
                    uid_pre_ptr = fb_uid_buf.data();
                    uid_pre_len = 4 + ulen;
                }

                // Sort 4-byte uint32_t timestamps!
                uint32_t* evs = ge + es;
                if (en > 1) sort_events_u32(evs, en, msd_shift8, msd_shift5, sort_tmp);

                uint32_t first_ts = evs[0];
                uint32_t last_ts  = evs[en - 1];

                // ULTRA-FAST 1-SESSION PATH: if span <= 30 min -> ZERO event loop, ZERO purchase sort, ZERO click sort!
                if (last_ts - first_ts <= uint32_t(SESSION_GAP_MS)) {
                    int32_t s_day = ms_to_day(win_start_ms + int64_t(first_ts)) - win_start_day;
                    uint64_t s_rev = 0;
                    int32_t best_c = -1;
                    if (__builtin_expect(pn > 0, 0)) {
                        const LocalPurchase* purs = grouped_purs.data() + ps;
                        for (uint32_t k = 0; k < pn; ++k) {
                            uint32_t pr = purs[k].price;
                            uint32_t ct = purs[k].cat_id;
                            s_rev += pr;
                            if (cat_rev[ct] == 0) used_cats.push_back(ct);
                            cat_rev[ct] += pr;
                        }
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
                    if (cn == 0) {
                        auto& cell = cd_cred[size_t(campaigns.organic_idx) * nd + size_t(s_day)];
                        cell.sessions++;
                        cell.purchasing_sessions += (s_rev > 0);
                        cell.revenue_cents += s_rev;
                    } else {
                        const FastLocalClick* clks = grouped_clks.data() + cs;
                        uint32_t s_rel = first_ts + uint32_t(CLICK_WINDOW_MS);
                        uint32_t win_lo = first_ts;
                        bool has_best = false;
                        uint32_t best_ts = 0, best_camp = 0;
                        uint64_t best_cid = 0;
                        for (uint32_t k = 0; k < cn; ++k) {
                            uint32_t cts = clks[k].ts_off;
                            if (cts > s_rel || cts < win_lo) continue;
                            uint32_t camp = clks[k].camp_idx;
                            if (!has_best || cts > best_ts ||
                                (cts == best_ts && key_less(clks[k].click_id, best_cid, arenas))) {
                                has_best = true;
                                best_ts = cts;
                                best_cid = clks[k].click_id;
                                best_camp = camp;
                            }
                            if (cstate[camp].count == 0) {
                                cstate[camp].count = 1;
                                active_camps.push_back(camp);
                            }
                        }
                        uint32_t credit_c = has_best ? best_camp : campaigns.organic_idx;
                        auto& cell = cd_cred[size_t(credit_c) * nd + size_t(s_day)];
                        cell.sessions++;
                        cell.purchasing_sessions += (s_rev > 0);
                        cell.revenue_cents += s_rev;
                        for (uint32_t ac : active_camps) {
                            cd_assist[size_t(ac) * nd + size_t(s_day)] += 1;
                            cstate[ac].count = 0;
                        }
                        active_camps.clear();
                    }
                    ud.emit_person_uid(uid_pre_ptr, uid_pre_len, 1U);
                    ud.push_day_row(uint16_t(s_day), 1, int32_t(en), int64_t(s_rev), best_c);
                    continue;
                }

                // Multi-session path: sort purchases (tiny!) and clicks (no physical dedup compaction needed!)
                LocalPurchase* purs = (pn > 0) ? (grouped_purs.data() + ps) : nullptr;
                if (pn > 1) {
                    if (pn <= 20) {
                        for (uint32_t i = 1; i < pn; ++i) {
                            LocalPurchase x = purs[i];
                            uint32_t j = i;
                            while (j > 0 && purs[j - 1].ts_rel_ms > x.ts_rel_ms) { purs[j] = purs[j - 1]; --j; }
                            purs[j] = x;
                        }
                    } else {
                        std::sort(purs, purs + pn, [](const LocalPurchase& a, const LocalPurchase& b) {
                            return a.ts_rel_ms < b.ts_rel_ms;
                        });
                    }
                }

                FastLocalClick* clks = (cn > 0) ? (grouped_clks.data() + cs) : nullptr;
                if (cn > 1) {
                    auto clk_less = [&](const FastLocalClick& a, const FastLocalClick& b) SS_LAMBDA_INLINE -> bool {
                        if (__builtin_expect(a.ts_off != b.ts_off, 1)) return a.ts_off < b.ts_off;
                        if (__builtin_expect(((a.click_id | b.click_id) & FALLBACK_BIT) == 0, 1)) return a.click_id > b.click_id;
                        return key_less(b.click_id, a.click_id, arenas);
                    };
                    if (cn == 2) {
                        if (clk_less(clks[1], clks[0])) std::swap(clks[0], clks[1]);
                    } else if (cn <= 24) {
                        for (uint32_t i = 1; i < cn; ++i) {
                            FastLocalClick x = clks[i];
                            uint32_t j = i;
                            while (j > 0 && clk_less(x, clks[j - 1])) { clks[j] = clks[j - 1]; --j; }
                            clks[j] = x;
                        }
                    } else {
                        std::sort(clks, clks + cn, clk_less);
                    }
                }

                const uint32_t person_rows_start = uint32_t(ud.num_rows);
                int32_t cur_day = -1;
                uint32_t cur_day_end_rel = 0;
                uint32_t day_sessions = 0, day_events = 0;
                uint64_t day_rev = 0;
                uint32_t L = 0, R = 0;
                uint32_t pk = 0;

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
                    ud.push_day_row(uint16_t(cur_day), int32_t(day_sessions), int32_t(day_events),
                                    int64_t(day_rev), best_c);
                    day_sessions = 0; day_events = 0; day_rev = 0;
                };

                uint32_t i = 0;
                while (i < en) {
                    uint32_t s_start = evs[i];
                    if (__builtin_expect(s_start >= cur_day_end_rel, 0)) {
                        int32_t s_day = ms_to_day(win_start_ms + int64_t(s_start)) - win_start_day;
                        if (s_day != cur_day) {
                            flush_day();
                            cur_day = s_day;
                            int64_t end_rel = int64_t(win_start_day + cur_day + 1) * DAY_MS - win_start_ms;
                            cur_day_end_rel = (end_rel >= (1LL << 32)) ? UINT32_MAX : uint32_t(end_rel);
                        }
                    }

                    // Ultra-tight 4-byte uint32_t session loop (ZERO price/category branches inside!)
                    uint32_t prev = s_start;
                    uint32_t j = i + 1;
                    while (j < en && evs[j] - prev <= uint32_t(SESSION_GAP_MS)) {
                        prev = evs[j];
                        ++j;
                    }
                    uint32_t s_ev_cnt = j - i;
                    i = j;

                    // Lockstep purchase merge ONCE per session (only touches purchases!)
                    uint64_t s_rev = 0;
                    if (__builtin_expect(pk < pn && purs[pk].ts_rel_ms <= prev, 0)) {
                        do {
                            uint32_t pr = purs[pk].price;
                            uint32_t ct = purs[pk].cat_id;
                            s_rev += pr;
                            if (cat_rev[ct] == 0) used_cats.push_back(ct);
                            cat_rev[ct] += pr;
                            ++pk;
                        } while (pk < pn && purs[pk].ts_rel_ms <= prev);
                    }

                    if (cn == 0) {
                        auto& cell = cd_cred[size_t(campaigns.organic_idx) * nd + size_t(cur_day)];
                        cell.sessions++;
                        cell.purchasing_sessions += (s_rev > 0);
                        cell.revenue_cents += s_rev;
                    } else {
                        uint32_t s_rel = s_start + uint32_t(CLICK_WINDOW_MS);
                        uint32_t win_lo = s_start;
                        while (R < cn && clks[R].ts_off <= s_rel) {
                            uint32_t c = clks[R].camp_idx;
                            if (cstate[c].count++ == 0) {
                                cstate[c].enter_sess = day_sessions;
                                cstate[c].active_pos = uint32_t(active_camps.size());
                                active_camps.push_back(c);
                            }
                            ++R;
                        }
                        while (L < R && clks[L].ts_off < win_lo) {
                            uint32_t c = clks[L].camp_idx;
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
                        uint32_t credit_c = (L < R) ? clks[R - 1].camp_idx : campaigns.organic_idx;
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
                ud.emit_person_uid(uid_pre_ptr, uid_pre_len, uint32_t(ud.num_rows) - person_rows_start);
                for (uint32_t ac : active_camps) cstate[ac].count = 0;
                active_camps.clear();
            }
        }

        if (ud.num_rows > 0) {
            total_ud_rows_written.fetch_add(ud.num_rows, std::memory_order_relaxed);
            std::string path = ud_dir + "/part-" + std::to_string(tid) + ".parquet";

            // 1. Encode user_id RLE_DICTIONARY indices (1 RLE run per active person!)
            pq::Buf uid_idx_buf;
            const uint32_t n_uid_dict = uint32_t(ud.uid_runs.size());
            const uint8_t uid_bw = pq::bits_for_count(n_uid_dict);
            const size_t uid_val_bytes = (size_t(uid_bw) + 7U) >> 3;
            uid_idx_buf.reserve(1 + size_t(n_uid_dict) * (2 + uid_val_bytes));
            uid_idx_buf.u8(uid_bw);
            for (uint32_t idx = 0; idx < n_uid_dict; ++idx) {
                uid_idx_buf.varint_u64(uint64_t(ud.uid_runs[idx]) << 1);
                if (uid_val_bytes) uid_idx_buf.raw(&idx, uid_val_bytes);
            }

            // 2. Encode day RLE_DICTIONARY indices (4-bit packed for 14-day window!)
            pq::Buf day_idx_buf;
            pq::encode_u16_indices_rle_dict(day_idx_buf, ud.day_idx, ud.num_rows,
                                            pq::bits_for_count(uint32_t(num_days)));

            // 3. Encode top_category RLE_DICTIONARY indices (5-bit packed for ~21 categories!)
            pq::Buf cat_idx_buf;
            pq::ColData cat_col{};
            if (ud.num_cats_nonnull == 0) {
                cat_col = {nullptr, 0, ud.top_cat_bitmap, nullptr, 0, 0};
            } else {
                pq::encode_u16_indices_rle_dict(cat_idx_buf, ud.cat_idx, ud.num_cats_nonnull,
                                                pq::bits_for_count(num_cats));
                cat_col = {cat_idx_buf.data.data(), cat_idx_buf.size(), ud.top_cat_bitmap,
                           cat_dict.data(), cat_dict.size(), int32_t(num_cats)};
            }

            pq::ColData data[6] = {
                {uid_idx_buf.data.data(), uid_idx_buf.size(), nullptr, ud.uid_dict, ud.uid_dict_sz, int32_t(n_uid_dict)},
                {day_idx_buf.data.data(), day_idx_buf.size(), nullptr, reinterpret_cast<const uint8_t*>(day_dict.data()), day_dict.size() * 4, num_days},
                {reinterpret_cast<const uint8_t*>(ud.sess_vals), ud.num_rows * 4, nullptr, nullptr, 0, 0},
                {reinterpret_cast<const uint8_t*>(ud.ev_vals), ud.num_rows * 4, nullptr, nullptr, 0, 0},
                {reinterpret_cast<const uint8_t*>(ud.rev_vals), ud.num_rows * 8, nullptr, nullptr, 0, 0},
                cat_col,
            };
            if (!pq::write_parquet_file(path.c_str(), UD_COLS, data, 6, int64_t(ud.num_rows))) {
                write_ok.store(false, std::memory_order_relaxed);
            }
        }

        if (finished_workers.fetch_add(1, std::memory_order_acq_rel) == pool.nthreads - 1) {
            bool need_empty = (total_ud_rows_written.load(std::memory_order_acquire) == 0);
            if (!write_campaign_daily_and_empty_ud(
                    ud_dir, cd_dir, need_empty, pool.nthreads, win_start_day, num_days,
                    campaigns, thread_cd_assist, thread_cd_cred)) {
                write_ok.store(false, std::memory_order_relaxed);
            }
        }
    });

    double td2 = now_ms();

    if (prof) {
        std::fprintf(stderr, "  sub-profile: dedup=%.1fms sess+write=%.1fms (pids=%u)\n",
                     td1 - td0, td2 - td1, max_pid);
    }
    return write_ok.load(std::memory_order_acquire);
}

// ============================================================================
// Wide / Classic Exact Fallback (UTF-8 links, >49.7d windows, >4096 campaigns, >16M users)
// ============================================================================
static SS_NOINLINE bool run_wide_dedup_sessionize_and_write(
    PartitionedStore<RawEvent16, 256, EV_BUCKETS, 4>& event_store,
    const std::vector<std::vector<SideEvent>>& sides,
    PartitionedStore<PersonClick, 128, PERSON_BUCKETS, 4>& click_store,
    const UserDict& dict,
    const Products& products,
    const Campaigns& campaigns,
    const StringArenaPool& arenas,
    ThreadPool& pool,
    int64_t /*win_start_ms*/,
    int64_t win_start_sec,
    int64_t click_start_ms,
    int32_t win_start_day,
    int32_t num_days,
    const std::string& ud_dir,
    const std::string& cd_dir
) {
    PartitionedStore<WidePersonEvent, 256, PERSON_BUCKETS, 4> person_ev_store;
    person_ev_store.init(pool.nthreads);

    using EvBlock = typename PartitionedStore<RawEvent16, 256, EV_BUCKETS, 4>::Block;
    std::atomic<int> next_eb{0};
    pool.run([&](int tid) {
        struct DedupSlot8 { uint32_t fp; uint32_t handle1; };
        std::vector<DedupSlot8> ht8;
        std::vector<EvBlock*> blks;
        auto& out_ts = *person_ev_store.ts[size_t(tid)];
        const bool classic_links = (dict.cap > 0);

        while (true) {
            int b = next_eb.fetch_add(1, std::memory_order_relaxed);
            if (b >= EV_BUCKETS) break;
            size_t nb = event_store.bucket_count(b);
            if (nb == 0) continue;

            blks.clear();
            for (int st = 0; st < pool.nthreads; ++st) {
                for (auto* blk = event_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) blks.push_back(blk);
            }
            size_t hcap = 64;
            while (hcap < nb * 2 + 16) hcap <<= 1;
            size_t hmask = hcap - 1;
            const uint32_t nblks = uint32_t(blks.size());

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
                        if (h1 == 0) {
                            ht8[slot] = {fp, ((bi << 8) | i) + 1};
                            break;
                        }
                        if (ht8[slot].fp == fp) {
                            uint32_t prev = h1 - 1;
                            RawEvent16& existing = blks[prev >> 8]->items[prev & 255U];
                            if (existing.eid40() == eid40) {
                                bool same = true;
                                if (ev.eid_in_side()) {
                                    const SideEvent& sa = sides[existing.w1 >> 26][existing.w1 & 0x03FFFFFFU];
                                    const SideEvent& sb = sides[ev.w1 >> 26][ev.w1 & 0x03FFFFFFU];
                                    same = key_eq(sa.eid_key64, sb.eid_key64, arenas);
                                }
                                if (same) {
                                    if (ingest_strictly_earlier(ev, existing, win_start_sec, sides)) existing = ev;
                                    ev.w0 = 0;
                                    break;
                                }
                            }
                        }
                        slot = (slot + 1) & hmask;
                    }
                }
            }
            for (uint32_t bi = 0; bi < nblks; ++bi) {
                const EvBlock* blk = blks[bi];
                for (uint32_t i = 0; i < blk->count; ++i) {
                    const RawEvent16& ev = blk->items[i];
                    if (!ev.is_valid()) continue;
                    const SideEvent& se = sides[ev.w1 >> 26][ev.w1 & 0x03FFFFFFU];
                    uint64_t pkey = classic_links ? dict.resolve_classic(se.user_key64, arenas) : se.user_key64;
                    uint32_t pb = uint32_t(hash_key64(pkey, arenas) >> (64 - PERSON_SHIFT));
                    out_ts.push(pb, WidePersonEvent{pkey, se.ts_ms, se.price, se.cat_id});
                }
            }
        }
        out_ts.flush_all();
    });

    static constexpr pq::ColSpec UD_COLS[6] = {
        {"user_id",       pq::ColKind::STRING_REQ},
        {"day",           pq::ColKind::DATE_REQ},
        {"sessions",      pq::ColKind::INT32_REQ},
        {"events",        pq::ColKind::INT32_REQ},
        {"revenue_cents", pq::ColKind::INT64_REQ},
        {"top_category",  pq::ColKind::STRING_OPT},
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
        ud.init(4096, products.max_cat_len);
        const size_t nd = size_t(num_days);
        const size_t n_cells = size_t(camp_stride) * nd;
        auto& cd_assist = thread_cd_assist[size_t(tid)];
        auto& cd_cred = thread_cd_cred[size_t(tid)];
        cd_assist.assign(n_cells, 0);
        cd_cred.assign(n_cells, CredCell{0, 0, 0});

        std::vector<uint64_t> pkeys;
        std::vector<uint32_t> pht, ev_cnt, ev_off, ev_pids, clk_cnt, clk_off, clk_pids;
        std::vector<WidePersonEvent> grouped_evs;
        std::vector<LocalClick> grouped_clks;
        std::vector<uint64_t> cat_rev(num_cats, 0);
        std::vector<uint32_t> used_cats;
        std::vector<CampState> cstate(num_camps, {0, 0, 0});
        std::vector<uint32_t> active_camps;
        std::vector<char> fb_uid_buf;

        while (true) {
            int b = next_pb.fetch_add(1, std::memory_order_relaxed);
            if (b >= PERSON_BUCKETS) break;
            size_t nev = person_ev_store.bucket_count(b);
            if (nev == 0) continue;

            pkeys.clear(); ev_cnt.clear(); ev_pids.clear();
            size_t hcap = 64;
            while (hcap < std::max<size_t>(64, nev / 10) * 2) hcap <<= 1;
            pht.assign(hcap, 0);
            size_t hmask = hcap - 1;
            auto rehash_pht = [&]() {
                hcap <<= 1; hmask = hcap - 1; pht.assign(hcap, 0);
                for (uint32_t pid = 0; pid < uint32_t(pkeys.size()); ++pid) {
                    size_t slot = hash_pkey_local(pkeys[pid], arenas) & hmask;
                    while (pht[slot] != 0) slot = (slot + 1) & hmask;
                    pht[slot] = pid + 1;
                }
            };
            for (int st = 0; st < pool.nthreads; ++st) {
                for (auto* blk = person_ev_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                    for (uint32_t i = 0; i < blk->count; ++i) {
                        uint64_t pk = blk->items[i].person_key;
                        if (pkeys.size() * 2 >= hcap) rehash_pht();
                        size_t slot = hash_pkey_local(pk, arenas) & hmask;
                        uint32_t pid;
                        while (true) {
                            uint32_t id1 = pht[slot];
                            if (id1 == 0) {
                                pid = uint32_t(pkeys.size());
                                pkeys.push_back(pk); ev_cnt.push_back(0);
                                pht[slot] = pid + 1; break;
                            }
                            if (key_eq(pkeys[id1 - 1], pk, arenas)) { pid = id1 - 1; break; }
                            slot = (slot + 1) & hmask;
                        }
                        ev_cnt[pid]++; ev_pids.push_back(pid);
                    }
                }
            }
            uint32_t num_people = uint32_t(pkeys.size());
            ev_off.resize(num_people + 1);
            uint32_t acc = 0;
            for (uint32_t p = 0; p < num_people; ++p) { ev_off[p] = acc; acc += ev_cnt[p]; }
            ev_off[num_people] = acc;
            grouped_evs.resize(acc);
            size_t fi = 0;
            for (int st = 0; st < pool.nthreads; ++st) {
                for (auto* blk = person_ev_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                    for (uint32_t i = 0; i < blk->count; ++i) grouped_evs[ev_off[ev_pids[fi++]]++] = blk->items[i];
                }
            }
            acc = 0; for (uint32_t p = 0; p < num_people; ++p) { ev_off[p] = acc; acc += ev_cnt[p]; }

            clk_cnt.assign(num_people, 0); clk_pids.clear();
            for (int st = 0; st < pool.nthreads; ++st) {
                for (auto* blk = click_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                    for (uint32_t i = 0; i < blk->count; ++i) {
                        uint64_t pk = blk->items[i].person_key;
                        size_t slot = hash_pkey_local(pk, arenas) & hmask;
                        uint32_t pid = UINT32_MAX;
                        while (pht[slot] != 0) {
                            if (key_eq(pkeys[pht[slot] - 1], pk, arenas)) { pid = pht[slot] - 1; break; }
                            slot = (slot + 1) & hmask;
                        }
                        clk_pids.push_back(pid);
                        if (pid != UINT32_MAX) clk_cnt[pid]++;
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
                        for (uint32_t i = 0; i < blk->count; ++i) {
                            uint32_t pid = clk_pids[ci++];
                            if (pid != UINT32_MAX) grouped_clks[clk_off[pid]++] = {blk->items[i].ts_camp, blk->items[i].click_id};
                        }
                    }
                }
                cacc = 0; for (uint32_t p = 0; p < num_people; ++p) { clk_off[p] = cacc; cacc += clk_cnt[p]; }
            }

            for (uint32_t pid = 0; pid < num_people; ++pid) {
                uint32_t es = ev_off[pid], ee = ev_off[pid + 1];
                if (es == ee) continue;
                std::sort(grouped_evs.data() + es, grouped_evs.data() + ee,
                          [](const WidePersonEvent& a, const WidePersonEvent& b) { return a.ts_ms < b.ts_ms; });
                uint32_t cs = clk_off[pid], ce = clk_off[pid + 1], cn = ce - cs;
                LocalClick* clks = grouped_clks.data() + cs;
                if (cn > 1) {
                    std::sort(clks, clks + cn, [&](const LocalClick& a, const LocalClick& b) {
                        if (a.ts_off() != b.ts_off()) return a.ts_off() < b.ts_off();
                        return key_less(b.click_id, a.click_id, arenas);
                    });
                }
                alignas(16) char uid_pre_inline[16]{};
                const char* uid_pre_ptr = uid_pre_inline;
                uint32_t uid_pre_len = 0;
                uint64_t pk = pkeys[pid];
                if ((pk & FALLBACK_BIT) == 0) {
                    uint32_t ulen = uint32_t(decode_inline(pk, uid_pre_inline + 4));
                    std::memcpy(uid_pre_inline, &ulen, 4);
                    uid_pre_len = 4 + ulen;
                } else {
                    auto sv = arenas.get_fallback(pk);
                    uint32_t ulen = uint32_t(sv.size());
                    fb_uid_buf.resize(4 + ulen + 16);
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

                auto flush_day = [&]() {
                    if (day_sessions == 0) return;
                    for (uint32_t ac : active_camps) {
                        cd_assist[size_t(ac) * nd + size_t(cur_day)] += (day_sessions - cstate[ac].enter_sess);
                        cstate[ac].enter_sess = 0;
                    }
                    int32_t best_c = -1;
                    if (!used_cats.empty()) {
                        uint32_t bc = used_cats[0]; uint64_t br = cat_rev[bc];
                        for (size_t k = 1; k < used_cats.size(); ++k) {
                            uint32_t c = used_cats[k]; uint64_t r = cat_rev[c];
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

                uint32_t i = es;
                while (i < ee) {
                    int64_t s_start = grouped_evs[i].ts_ms, prev_ts = s_start;
                    if (s_start >= cur_day_end_ms) {
                        int32_t s_day = ms_to_day(s_start) - win_start_day;
                        if (s_day != cur_day) {
                            flush_day();
                            cur_day = s_day;
                            cur_day_end_ms = int64_t(win_start_day + cur_day + 1) * DAY_MS;
                        }
                    }
                    uint32_t s_ev_cnt = 0; uint64_t s_rev = 0;
                    while (i < ee && grouped_evs[i].ts_ms - prev_ts <= SESSION_GAP_MS) {
                        prev_ts = grouped_evs[i].ts_ms;
                        ++s_ev_cnt;
                        uint32_t pr = grouped_evs[i].price;
                        if (pr > 0) {
                            uint32_t ct = grouped_evs[i].cat_id;
                            s_rev += pr;
                            if (cat_rev[ct] == 0) used_cats.push_back(ct);
                            cat_rev[ct] += pr;
                        }
                        ++i;
                    }
                    if (cn == 0) {
                        auto& cell = cd_cred[size_t(campaigns.organic_idx) * nd + size_t(cur_day)];
                        cell.sessions++; cell.purchasing_sessions += (s_rev > 0); cell.revenue_cents += s_rev;
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
                        cell.sessions++; cell.purchasing_sessions += (s_rev > 0); cell.revenue_cents += s_rev;
                    }
                    day_sessions++; day_events += s_ev_cnt; day_rev += s_rev;
                }
                flush_day();
                for (uint32_t ac : active_camps) cstate[ac].count = 0;
                active_camps.clear();
            }
        }
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
            bool need_empty = (total_ud_rows_written.load(std::memory_order_acquire) == 0);
            if (!write_campaign_daily_and_empty_ud(
                    ud_dir, cd_dir, need_empty, pool.nthreads, win_start_day, num_days,
                    campaigns, thread_cd_assist, thread_cd_cred)) {
                write_ok.store(false, std::memory_order_relaxed);
            }
        }
    });

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
    bool use_pread = std::getenv("USE_PREAD") != nullptr;
    bool force_wide = std::getenv("FORCE_WIDE") != nullptr;

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
    int64_t win_dur_sec = win_dur_ms / 1000LL;
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
    if (use_pread) {
        for (int t = 0; t < ncpu; ++t) rbufs[size_t(t)].init();
    }

    std::vector<std::pair<std::string, uint8_t>> all_paths;
    all_paths.reserve(256);
    list_csv_paths(data_dir + "/links", 2, all_paths);
    list_csv_paths(data_dir + "/clicks", 0, all_paths);
    list_csv_paths(data_dir + "/events", 1, all_paths);

    const bool keep_fds = use_pread && (all_paths.size() < 480);
    std::vector<CsvFile> csv_files(all_paths.size());
    std::atomic<size_t> next_f{0};

    // Parallel open + guard-paged mmap (zero fd leak, zero SIGBUS even when fsz % 4096 == 0!)
    pool.run([&](int) {
        while (true) {
            size_t i = next_f.fetch_add(1, std::memory_order_relaxed);
            if (i >= all_paths.size()) break;
            int fd = ::open(all_paths[i].first.c_str(), O_RDONLY);
            if (fd < 0) continue;
            struct stat st{};
            size_t fsz = (::fstat(fd, &st) == 0 && st.st_size > 0) ? size_t(st.st_size) : 0;
            const char* mbase = nullptr;
            size_t mlen_tot = 0;
            if (!use_pread && fsz > 0) {
                size_t mlen = (fsz + 4095) & ~size_t(4095);
                void* res = ::mmap(nullptr, mlen + 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (res != MAP_FAILED) {
                    void* mapped = ::mmap(res, fsz, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0);
                    if (mapped != MAP_FAILED) {
                        mbase = static_cast<const char*>(mapped);
                        mlen_tot = mlen + 4096;
                    } else {
                        ::munmap(res, mlen + 4096);
                    }
                }
            }
            if (mbase != nullptr || !keep_fds || fsz == 0) { ::close(fd); fd = -1; }
            csv_files[i] = {all_paths[i].first, fd, fsz, mbase, mlen_tot, all_paths[i].second};
        }
    });

    size_t total_event_bytes = 0;
    bool need_readbufs = use_pread;
    for (const auto& f : csv_files) {
        if (f.kind == 1) total_event_bytes += f.file_size;
        if (f.file_size > 0 && f.map_base == nullptr) need_readbufs = true;
    }
    if (need_readbufs && rbufs[0].data == nullptr) {
        for (int t = 0; t < ncpu; ++t) rbufs[size_t(t)].init();
    }

    std::vector<CsvTask> link_tasks, io_tasks;
    build_csv_tasks(csv_files, 1U << 2, ncpu, link_tasks);
    build_csv_tasks(csv_files, (1U << 0) | (1U << 1), ncpu, io_tasks);
    double t_init = now_ms();

    // Exact manifest-bound gate for fast_dict_mode (whole-second bounds + < 49.7d span!)
    bool manifest_compact_ok = !force_wide &&
                               (win_start_ms % 1000LL == 0) &&
                               (win_end_ms % 1000LL == 0) &&
                               (win_end_ms >= win_start_ms) &&
                               (uint64_t(win_end_ms - win_start_ms) < (1ULL << 32)) &&
                               (uint64_t(win_end_ms - click_start_ms) < (1ULL << 32));

    Products products;
    Campaigns campaigns;
    UserDict dict;
    dict.build_and_load_dims(link_tasks, rbufs, pool, arenas, products, campaigns,
                             data_dir, total_event_bytes, manifest_compact_ok);
    double t_links = now_ms();

    bool use_fast_dict = manifest_compact_ok && dict.fast_dict_mode &&
                         (campaigns.organic_idx < 4096) &&
                         (num_days <= 65535) &&
                         (products.cat_names.size() <= 65535);

    bool write_ok = false;
    double t_scan = t_links;

    if (use_fast_dict) {
        PartitionedStore<CompactClick, 128, PERSON_BUCKETS, 4> click_store;
        PartitionedStore<RawEvent16, 256, EV_BUCKETS, 4> event_store;
        click_store.init(ncpu, 64ULL << 20);
        event_store.init(ncpu, 256ULL << 20);

        if (dict.recycled_dsu && dict.recycled_dsu_bytes >= (4ULL << 20)) {
            size_t slice = (dict.recycled_dsu_bytes / size_t(ncpu)) & ~size_t(4095);
            char* p = static_cast<char*>(dict.recycled_dsu);
            for (int t = 0; t < ncpu; ++t) {
                event_store.ts[size_t(t)]->donate_slab(p + size_t(t) * slice, slice);
            }
        }

        std::vector<std::vector<SideEvent>> thread_sides(static_cast<size_t>(ncpu));

        if (!io_tasks.empty()) {
            std::atomic<size_t> next_t{0};
            pool.run([&](int tid) {
                auto& clk_out = *click_store.ts[size_t(tid)];
                auto& ev_out = *event_store.ts[size_t(tid)];
                auto& side_vec = thread_sides[size_t(tid)];
                auto& rbuf = rbufs[size_t(tid)];
                side_vec.reserve(32768);

                ClickRingFast cring[RING_SZ];
                EvRingFast    ering[RING_SZ];
                size_t cpushed = 0, epushed = 0;
                uint32_t tl_cur = 0, tl_end = 0;

                auto drain_clicks = [&]() SS_COLD_NOINLINE {
                    if (cpushed > 0) {
                        size_t s = (cpushed > RING_LAG) ? (cpushed - RING_LAG) : 0;
                        for (size_t k = s; k < cpushed; ++k) {
                            const auto& r = cring[k & RING_MASK];
                            uint32_t pid = dict.get_or_insert_uid48(r.u48, r.h48, tl_cur, tl_end);
                            uint32_t pb = pid & uint32_t(PERSON_BUCKETS - 1);
                            clk_out.push(pb, CompactClick{r.cid, (uint64_t(pid >> PERSON_SHIFT) << 44) | r.ts_camp44});
                        }
                        cpushed = 0;
                    }
                };
                auto drain_events = [&]() SS_COLD_NOINLINE {
                    if (epushed > 0) {
                        size_t s = (epushed > RING_LAG) ? (epushed - RING_LAG) : 0;
                        for (size_t k = s; k < epushed; ++k) {
                            const auto& r = ering[k & RING_MASK];
                            uint32_t pid = dict.get_or_insert_uid48(r.u48, r.h48, tl_cur, tl_end);
                            ev_out.push(r.eb, RawEvent16::make_inline_pid(r.irel21, r.rem40, r.ts_rel_ms, pid));
                        }
                        epushed = 0;
                    }
                };

                while (true) {
                    size_t ti = next_t.fetch_add(1, std::memory_order_relaxed);
                    if (ti >= io_tasks.size()) break;
                    const auto& task = io_tasks[ti];
                    if (task.kind == 0) {
                        process_click_task_fast(task, rbuf, tid, campaigns, lut, dict, arenas,
                                                click_start_ms, win_end_ms, clk_out, cring, cpushed, tl_cur, tl_end);
                    } else {
                        process_event_task_fast(task, rbuf, tid, products, lut, dict, arenas,
                                                win_start_ms, win_start_sec, win_dur_ms, win_dur_sec,
                                                ev_out, side_vec, ering, epushed, tl_cur, tl_end);
                    }
                }
                drain_clicks();
                drain_events();
                clk_out.flush_all();
                ev_out.flush_all();
            });
        }
        t_scan = now_ms();

        if (__builtin_expect(!dict.dict_overflow.load(std::memory_order_acquire), 1)) {
            write_ok = run_fast_dict_dedup_sessionize_and_write(
                event_store, thread_sides, click_store, dict, products, campaigns, arenas, pool,
                win_start_ms, win_end_ms, win_start_sec, win_start_day, num_days, ud_dir, cd_dir, prof);
        } else {
            // Ultra-rare >16.7M distinct users -> rebuild links classic & run wide fallback!
            std::fprintf(stderr, "UserDict overflow (>16.7M users); falling back to wide mode\n");
            use_fast_dict = false;
        }
    }

    if (!use_fast_dict) {
        if (dict.fast_dict_mode) {
            dict.fast_dict_mode = false;
            dict.build_and_load_dims(link_tasks, rbufs, pool, arenas, products, campaigns,
                                     data_dir, total_event_bytes, false);
        }
        PartitionedStore<PersonClick, 128, PERSON_BUCKETS, 4> click_store;
        PartitionedStore<RawEvent16, 256, EV_BUCKETS, 4> event_store;
        click_store.init(ncpu);
        event_store.init(ncpu);
        std::vector<std::vector<SideEvent>> thread_sides(static_cast<size_t>(ncpu));

        if (!io_tasks.empty()) {
            std::atomic<size_t> next_t{0};
            pool.run([&](int tid) {
                auto& clk_out = *click_store.ts[size_t(tid)];
                auto& ev_out = *event_store.ts[size_t(tid)];
                auto& side_vec = thread_sides[size_t(tid)];
                auto& rbuf = rbufs[size_t(tid)];
                while (true) {
                    size_t ti = next_t.fetch_add(1, std::memory_order_relaxed);
                    if (ti >= io_tasks.size()) break;
                    const auto& task = io_tasks[ti];
                    if (task.kind == 0) {
                        process_click_task_wide(task, rbuf, tid, campaigns, lut, dict, arenas,
                                                click_start_ms, win_end_ms, clk_out);
                    } else {
                        process_event_task_wide(task, rbuf, tid, products, lut, arenas,
                                                win_start_ms, win_start_sec, win_dur_ms, ev_out, side_vec);
                    }
                }
                clk_out.flush_all();
                ev_out.flush_all();
            });
        }
        t_scan = now_ms();
        write_ok = run_wide_dedup_sessionize_and_write(
            event_store, thread_sides, click_store, dict, products, campaigns, arenas, pool,
            win_start_ms, win_start_sec, click_start_ms, win_start_day, num_days, ud_dir, cd_dir);
    }

    double t_end = now_ms();

    if (prof) {
        std::fprintf(stderr,
            "PROFILE (ms): init+open=%.1f links+dims=%.1f scan=%.1f dedup+sess+write=%.1f TOTAL=%.1f\n",
            t_init - t0, t_links - t_init, t_scan - t_links, t_end - t_scan, t_end - t0);
    }
    if (!write_ok) {
        std::fprintf(stderr, "Parquet write failed\n");
        ::_exit(1);
    }
    ::_exit(0);
}

} // namespace SS_ENGINE_NS
