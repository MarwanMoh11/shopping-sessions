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
    uint32_t seg_id = UINT32_MAX;
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
    static constexpr size_t READ_CHUNK = 131072; // 128 KiB: stays 100% L2-hot per HT on Cascade Lake!
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

    uint64_t cms[68];
    uint16_t npos[4098];
    bool done = false;

    auto on_cold = [&](const Slice* fs) SS_COLD_NOINLINE {
        on_fields(fs);
    };

    while (cur < limit) {
        size_t avail = fsz - cur;
        size_t nblks = std::min<size_t>(64, avail >> 6);
        if (__builtin_expect(nblks == 0, 0)) break;

        const char* bptr = base + cur;
        uint64_t any_bad = 0;
        uint64_t s_carry = 1, rq_carry = 0, inq = 0;
        uint32_t nnl = 0;
        for (size_t bi = 0; bi < nblks; ++bi) {
            if ((bi & 7U) == 0) __builtin_prefetch(bptr + (bi << 6) + 2048, 0, 1);
            Vec64 v = Vec64::load(bptr + (bi << 6));
            uint64_t nm = v.eq('\n');
            uint64_t cm = v.eq(',');
            uint64_t qm = v.eq('"');
            cms[bi] = cm;
            // Block-level quote validation with ZERO pmull loop-carried chain (1-cycle popcnt carry)!
            uint64_t S  = cm | nm;
            uint64_t FS = (S << 1) | s_carry;
            uint64_t RQ = qm & ~FS;
            uint64_t PX = prefix_xor(qm) ^ (0ULL - inq);
            any_bad |= (((RQ << 1) | rq_carry) & ~S) | (PX & S);
            s_carry  = S >> 63;
            rq_carry = RQ >> 63;
            inq ^= (uint64_t(__builtin_popcountll(qm)) & 1ULL);

            // 100% branchless 2-slot newline flattening (zero branch on nm==0 / nm2==0 via ctz64_safe!)
            uint32_t base_off = uint32_t(bi << 6);
            uint32_t c = uint32_t(__builtin_popcountll(nm));
            npos[nnl] = uint16_t(base_off + ctz64_safe(nm));
            uint64_t nm2 = nm & (nm - 1);
            npos[nnl + 1] = uint16_t(base_off + ctz64_safe(nm2));
            if (__builtin_expect(c <= 2U, 1)) {
                nnl += c;
            } else {
                nnl += 2;
                nm2 &= nm2 - 1;
                while (nm2) {
                    npos[nnl++] = uint16_t(base_off + uint32_t(__builtin_ctzll(nm2)));
                    nm2 &= nm2 - 1;
                }
            }
        }
        cms[nblks] = cms[nblks + 1] = cms[nblks + 2] = 0;

        size_t ls = 0;
        const size_t lrel = limit - cur;
        if (__builtin_expect(any_bad == 0, 1)) {
            for (uint32_t ri = 0; ri < nnl; ++ri) {
                size_t nl_rel = npos[ri];
                uint32_t L = uint32_t(nl_rel - ls);
                Slice f[N];
                int er = extract_clean_csv<N>(bptr + ls, ls, L, cms, f);
                if (__builtin_expect(er > 0, 1)) {
                    on_fields(f);
                } else if (__builtin_expect(er < 0, 0)) {
                    Slice fs[N];
                    if (parse_csv_line_scalar<N>(bptr + ls, bptr + nl_rel, fs)) on_cold(fs);
                }
                ls = nl_rel + 1;
                if (__builtin_expect(ls >= lrel, 0)) {
                    cur += ls;
                    done = true;
                    break;
                }
            }
        } else {
            Slice fs[N];
            for (uint32_t ri = 0; ri < nnl; ++ri) {
                size_t nl_rel = npos[ri];
                if (parse_csv_line_scalar<N>(bptr + ls, bptr + nl_rel, fs)) on_cold(fs);
                ls = nl_rel + 1;
                if (__builtin_expect(ls >= lrel, 0)) {
                    cur += ls;
                    done = true;
                    break;
                }
            }
        }
        if (done) break;
        if (__builtin_expect(ls == 0, 0)) {
            const char* nl = find_nl_simd(bptr + (nblks << 6), base + fsz);
            if (!nl) break;
            Slice fs[N];
            if (parse_csv_line_scalar<N>(bptr, nl, fs)) on_cold(fs);
            cur = size_t(nl - base) + 1;
        } else {
            cur += ls;
        }
    }

    while (cur < limit && cur < fsz) {
        const char* line = base + cur;
        const char* nl = find_nl_simd(line, base + fsz);
        const char* le = nl ? nl : (base + fsz);
        Slice fs[N];
        if (parse_csv_line_scalar<N>(line, le, fs)) on_cold(fs);
        cur = nl ? (size_t(nl - base) + 1) : fsz;
    }
}

template <int N, typename OnFields>
static SS_NOINLINE void scan_pread_csv(const CsvTask& task, ReadBuf& rbuf, OnFields&& on_fields) {
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

    uint64_t cms[68];
    uint16_t npos[4098];

    auto on_cold = [&](const Slice* fs) SS_COLD_NOINLINE {
        on_fields(fs);
    };

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
            uint64_t any_bad = 0;
            uint64_t s_carry = 1, rq_carry = 0, inq = 0;
            uint32_t nnl = 0;
            for (size_t bi = 0; bi < nblks; ++bi) {
                Vec64 v = Vec64::load(bptr + (bi << 6));
                uint64_t nm = v.eq('\n');
                uint64_t cm = v.eq(',');
                uint64_t qm = v.eq('"');
                cms[bi] = cm;
                uint64_t S  = cm | nm;
                uint64_t FS = (S << 1) | s_carry;
                uint64_t RQ = qm & ~FS;
                uint64_t PX = prefix_xor(qm) ^ (0ULL - inq);
                any_bad |= (((RQ << 1) | rq_carry) & ~S) | (PX & S);
                s_carry  = S >> 63;
                rq_carry = RQ >> 63;
                inq ^= (uint64_t(__builtin_popcountll(qm)) & 1ULL);

                uint32_t base_off = uint32_t(bi << 6);
                uint32_t c = uint32_t(__builtin_popcountll(nm));
                npos[nnl] = uint16_t(base_off + ctz64_safe(nm));
                uint64_t nm2 = nm & (nm - 1);
                npos[nnl + 1] = uint16_t(base_off + ctz64_safe(nm2));
                if (__builtin_expect(c <= 2U, 1)) {
                    nnl += c;
                } else {
                    nnl += 2;
                    nm2 &= nm2 - 1;
                    while (nm2) {
                        npos[nnl++] = uint16_t(base_off + uint32_t(__builtin_ctzll(nm2)));
                        nm2 &= nm2 - 1;
                    }
                }
            }
            cms[nblks] = cms[nblks + 1] = cms[nblks + 2] = 0;

            size_t ls = 0;
            const size_t lrel = limit_in_buf - cur;
            if (__builtin_expect(any_bad == 0, 1)) {
                for (uint32_t ri = 0; ri < nnl; ++ri) {
                    size_t nl_rel = npos[ri];
                    uint32_t L = uint32_t(nl_rel - ls);
                    Slice f[N];
                    int er = extract_clean_csv<N>(bptr + ls, ls, L, cms, f);
                    if (__builtin_expect(er > 0, 1)) {
                        on_fields(f);
                    } else if (__builtin_expect(er < 0, 0)) {
                        Slice fs[N];
                        if (parse_csv_line_scalar<N>(bptr + ls, bptr + nl_rel, fs)) on_cold(fs);
                    }
                    ls = nl_rel + 1;
                    if (__builtin_expect(ls >= lrel, 0)) {
                        cur += ls;
                        done = true;
                        break;
                    }
                }
            } else {
                Slice fs[N];
                for (uint32_t ri = 0; ri < nnl; ++ri) {
                    size_t nl_rel = npos[ri];
                    if (parse_csv_line_scalar<N>(bptr + ls, bptr + nl_rel, fs)) on_cold(fs);
                    ls = nl_rel + 1;
                    if (__builtin_expect(ls >= lrel, 0)) {
                        cur += ls;
                        done = true;
                        break;
                    }
                }
            }
            if (done) break;
            if (__builtin_expect(ls == 0, 0)) {
                const char* nl = find_nl_simd(bptr + (nblks << 6), rbuf.data + total);
                if (!nl) break;
                Slice fs[N];
                if (parse_csv_line_scalar<N>(bptr, nl, fs)) on_cold(fs);
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
            Slice fs[N];
            if (parse_csv_line_scalar<N>(line, nl, fs)) on_cold(fs);
            cur = size_t(nl - rbuf.data) + 1;
        }

        if (buf_file_off + cur >= limit) break;

        if (file_pos >= fsz) {
            if (cur < total && buf_file_off + cur < limit) {
                Slice fs[N];
                if (parse_csv_line_scalar<N>(rbuf.data + cur, rbuf.data + total, fs)) on_cold(fs);
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

// Register-only lazy-field CSV scanner: passes PackedCsvRow<N>{lp, pk, L} (3 GPRs!) on the clean hot loop,
// and routes dirty/long/tail lines through a single SS_COLD_NOINLINE SliceCsvRow callback!
template <int N, typename OnRow>
static SS_ALWAYS_INLINE void scan_mmap_csv_rows(const CsvTask& task, OnRow&& on_row) {
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

    uint64_t cms[68];
    uint16_t npos[4098];
    bool done = false;

    auto on_cold = [&](const Slice* fs) SS_COLD_NOINLINE {
        on_row(SliceCsvRow{fs});
    };

    while (cur < limit) {
        size_t avail = fsz - cur;
        size_t nblks = std::min<size_t>(64, avail >> 6);
        if (__builtin_expect(nblks == 0, 0)) break;

        const char* bptr = base + cur;
        uint64_t any_bad = 0;
        uint64_t s_carry = 1, rq_carry = 0, inq = 0;
        uint32_t nnl = 0;
        for (size_t bi = 0; bi < nblks; ++bi) {
            if ((bi & 7U) == 0) __builtin_prefetch(bptr + (bi << 6) + 2048, 0, 1);
            Vec64 v = Vec64::load(bptr + (bi << 6));
            uint64_t nm = v.eq('\n');
            uint64_t cm = v.eq(',');
            uint64_t qm = v.eq('"');
            cms[bi] = cm;
            uint64_t S  = cm | nm;
            uint64_t FS = (S << 1) | s_carry;
            uint64_t RQ = qm & ~FS;
            uint64_t PX = prefix_xor(qm) ^ (0ULL - inq);
            any_bad |= (((RQ << 1) | rq_carry) & ~S) | (PX & S);
            s_carry  = S >> 63;
            rq_carry = RQ >> 63;
            inq ^= (uint64_t(__builtin_popcountll(qm)) & 1ULL);

            uint32_t base_off = uint32_t(bi << 6);
            uint32_t c = uint32_t(__builtin_popcountll(nm));
            npos[nnl] = uint16_t(base_off + ctz64_safe(nm));
            uint64_t nm2 = nm & (nm - 1);
            npos[nnl + 1] = uint16_t(base_off + ctz64_safe(nm2));
            if (__builtin_expect(c <= 2U, 1)) {
                nnl += c;
            } else {
                nnl += 2;
                nm2 &= nm2 - 1;
                while (nm2) {
                    npos[nnl++] = uint16_t(base_off + uint32_t(__builtin_ctzll(nm2)));
                    nm2 &= nm2 - 1;
                }
            }
        }
        cms[nblks] = cms[nblks + 1] = cms[nblks + 2] = 0;

        size_t ls = 0;
        const size_t lrel = limit - cur;
        if (__builtin_expect(any_bad == 0, 1)) {
            for (uint32_t ri = 0; ri < nnl; ++ri) {
                size_t nl_rel = npos[ri];
                uint32_t L = uint32_t(nl_rel - ls);
                uint64_t pk = 0;
                int er = extract_clean_packed<N>(ls, L, cms, pk);
                if (__builtin_expect(er > 0, 1)) {
                    on_row(PackedCsvRow<N>{bptr + ls, pk, L});
                } else if (__builtin_expect(er < 0, 0)) {
                    Slice fs[N];
                    if (parse_csv_line_scalar<N>(bptr + ls, bptr + nl_rel, fs)) on_cold(fs);
                }
                ls = nl_rel + 1;
                if (__builtin_expect(ls >= lrel, 0)) {
                    cur += ls;
                    done = true;
                    break;
                }
            }
        } else {
            Slice fs[N];
            for (uint32_t ri = 0; ri < nnl; ++ri) {
                size_t nl_rel = npos[ri];
                if (parse_csv_line_scalar<N>(bptr + ls, bptr + nl_rel, fs)) on_cold(fs);
                ls = nl_rel + 1;
                if (__builtin_expect(ls >= lrel, 0)) {
                    cur += ls;
                    done = true;
                    break;
                }
            }
        }
        if (done) break;
        if (__builtin_expect(ls == 0, 0)) {
            const char* nl = find_nl_simd(bptr + (nblks << 6), base + fsz);
            if (!nl) break;
            Slice fs[N];
            if (parse_csv_line_scalar<N>(bptr, nl, fs)) on_cold(fs);
            cur = size_t(nl - base) + 1;
        } else {
            cur += ls;
        }
    }

    while (cur < limit && cur < fsz) {
        const char* line = base + cur;
        const char* nl = find_nl_simd(line, base + fsz);
        const char* le = nl ? nl : (base + fsz);
        Slice fs[N];
        if (parse_csv_line_scalar<N>(line, le, fs)) on_cold(fs);
        cur = nl ? (size_t(nl - base) + 1) : fsz;
    }
}

template <int N, typename OnRow>
static SS_NOINLINE void scan_pread_csv_rows(const CsvTask& task, ReadBuf& rbuf, OnRow&& on_row) {
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

    uint64_t cms[68];
    uint16_t npos[4098];

    auto on_cold = [&](const Slice* fs) SS_COLD_NOINLINE {
        on_row(SliceCsvRow{fs});
    };

    while (true) {
        if (file_pos >= fsz) break;
        size_t bound = hit_limit ? fsz : std::min<size_t>(fsz, limit + 4096);
        size_t chunk_sz = (first_read && task.start > 0) ? (ReadBuf::READ_CHUNK + 1) : ReadBuf::READ_CHUNK;
        size_t to_read = hit_limit ? std::min<size_t>(4096, fsz - file_pos)
                                   : std::min<size_t>(chunk_sz, bound > file_pos ? (bound - file_pos) : std::min<size_t>(4096, fsz - file_pos));
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
            uint64_t any_bad = 0;
            uint64_t s_carry = 1, rq_carry = 0, inq = 0;
            uint32_t nnl = 0;
            for (size_t bi = 0; bi < nblks; ++bi) {
                Vec64 v = Vec64::load(bptr + (bi << 6));
                uint64_t nm = v.eq('\n');
                uint64_t cm = v.eq(',');
                uint64_t qm = v.eq('"');
                cms[bi] = cm;
                uint64_t S  = cm | nm;
                uint64_t FS = (S << 1) | s_carry;
                uint64_t RQ = qm & ~FS;
                uint64_t PX = prefix_xor(qm) ^ (0ULL - inq);
                any_bad |= (((RQ << 1) | rq_carry) & ~S) | (PX & S);
                s_carry  = S >> 63;
                rq_carry = RQ >> 63;
                inq ^= (uint64_t(__builtin_popcountll(qm)) & 1ULL);

                uint32_t base_off = uint32_t(bi << 6);
                uint32_t c = uint32_t(__builtin_popcountll(nm));
                npos[nnl] = uint16_t(base_off + ctz64_safe(nm));
                uint64_t nm2 = nm & (nm - 1);
                npos[nnl + 1] = uint16_t(base_off + ctz64_safe(nm2));
                if (__builtin_expect(c <= 2U, 1)) {
                    nnl += c;
                } else {
                    nnl += 2;
                    nm2 &= nm2 - 1;
                    while (nm2) {
                        npos[nnl++] = uint16_t(base_off + uint32_t(__builtin_ctzll(nm2)));
                        nm2 &= nm2 - 1;
                    }
                }
            }
            cms[nblks] = cms[nblks + 1] = cms[nblks + 2] = 0;

            size_t ls = 0;
            const size_t lrel = limit_in_buf - cur;
            if (__builtin_expect(any_bad == 0, 1)) {
                for (uint32_t ri = 0; ri < nnl; ++ri) {
                    size_t nl_rel = npos[ri];
                    uint32_t L = uint32_t(nl_rel - ls);
                    uint64_t pk = 0;
                    int er = extract_clean_packed<N>(ls, L, cms, pk);
                    if (__builtin_expect(er > 0, 1)) {
                        on_row(PackedCsvRow<N>{bptr + ls, pk, L});
                    } else if (__builtin_expect(er < 0, 0)) {
                        Slice fs[N];
                        if (parse_csv_line_scalar<N>(bptr + ls, bptr + nl_rel, fs)) on_cold(fs);
                    }
                    ls = nl_rel + 1;
                    if (__builtin_expect(ls >= lrel, 0)) {
                        cur += ls;
                        done = true;
                        break;
                    }
                }
            } else {
                Slice fs[N];
                for (uint32_t ri = 0; ri < nnl; ++ri) {
                    size_t nl_rel = npos[ri];
                    if (parse_csv_line_scalar<N>(bptr + ls, bptr + nl_rel, fs)) on_cold(fs);
                    ls = nl_rel + 1;
                    if (__builtin_expect(ls >= lrel, 0)) {
                        cur += ls;
                        done = true;
                        break;
                    }
                }
            }
            if (done) break;
            if (__builtin_expect(ls == 0, 0)) {
                const char* nl = find_nl_simd(bptr + (nblks << 6), rbuf.data + total);
                if (!nl) break;
                Slice fs[N];
                if (parse_csv_line_scalar<N>(bptr, nl, fs)) on_cold(fs);
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
            Slice fs[N];
            if (parse_csv_line_scalar<N>(line, nl, fs)) on_cold(fs);
            cur = size_t(nl - rbuf.data) + 1;
        }

        if (buf_file_off + cur >= limit) break;

        if (file_pos >= fsz) {
            if (cur < total && buf_file_off + cur < limit) {
                Slice fs[N];
                if (parse_csv_line_scalar<N>(rbuf.data + cur, rbuf.data + total, fs)) on_cold(fs);
            }
            break;
        }

        carry = total - cur;
        if (carry > 0) std::memmove(rbuf.data, rbuf.data + cur, carry);
        if (file_pos >= limit) hit_limit = true;
    }

    if (close_after) ::close(fd);
}

template <int N, typename OnRow>
static SS_ALWAYS_INLINE void scan_csv_rows(const CsvTask& task, ReadBuf& rbuf, OnRow&& on_row) {
    if (__builtin_expect(task.file->map_base != nullptr, 1)) {
        scan_mmap_csv_rows<N>(task, std::forward<OnRow>(on_row));
    } else {
        scan_pread_csv_rows<N>(task, rbuf, std::forward<OnRow>(on_row));
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
    static constexpr uint32_t MAX_GENTRIES  = (1U << 24) - 4096U;

    struct GenEntry {
        uint64_t key;
        uint32_t pid;
        uint32_t _pad;
    };

    // 1-cache-line 3-way associative inline 128-bit user table for 'u' + 32 lowercase hex!
    // Eliminates the gslots -> gentries -> 2GB mmap page walk -> memcmp chain (~3.6M lookups/run)!
    struct alignas(64) HexLine {
        uint64_t k[3][2];  // 48 B exact 128-bit keys
        uint32_t pid[3];   // 12 B dense PIDs
        uint32_t ctrl;     // 4 B atomic_ref: bits 0..2 claimed, bits 8..10 ready
    };
    static_assert(sizeof(HexLine) == 64, "HexLine must be 64 bytes");

    bool fast_dict_mode = false;
    uint64_t* uslots = nullptr;
    size_t ucap = 0;
    size_t umask = 0;
    size_t uslots_bytes = 0;

    // Inline 128-bit HexDict (THP-backed 64B cache-line table)
    HexLine* hlines = nullptr;
    size_t hcap = 0;
    size_t hmask = 0;
    size_t hlines_bytes = 0;

    // Lock-free general/fallback user dictionary (rare non-hex long, UTF-8, uppercase & d>254 overflows!)
    // gslots[i] = [hash_tag32 : 32][entry_idx_plus_1 : 32] (0 = empty)
    uint64_t* gslots = nullptr;
    size_t gcap = 0;
    size_t gmask = 0;
    size_t gslots_bytes = 0;
    GenEntry* gentries = nullptr;
    size_t gentries_bytes = 0;
    std::atomic<uint32_t> next_gentry{0};

    uint64_t* pid_name = nullptr;
    size_t pid_name_bytes = 0;
    std::atomic<uint32_t> next_pid{1};
    std::atomic<bool> dict_overflow{false};

    // Classic 16B-slot fallback mode (only for >49.7d windows, fractional manifest, >4096 campaigns, or >16M users)
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

    SS_ALWAYS_INLINE void prefetch_hline(uint64_t h) const {
        __builtin_prefetch(&hlines[size_t(h) & hmask], 0, 1);
    }

    SS_ALWAYS_INLINE void prefetch_gslot(uint64_t h) const {
        __builtin_prefetch(&gslots[size_t(h) & gmask], 0, 1);
    }

    SS_COLD_NOINLINE bool lease_pid_slow(uint32_t& tl_cur, uint32_t& tl_end) {
        uint32_t base = next_pid.fetch_add(1024, std::memory_order_relaxed);
        if (__builtin_expect(base >= MAX_DICT_PIDS, 0)) {
            dict_overflow.store(true, std::memory_order_relaxed);
            return false;
        }
        tl_cur = base;
        tl_end = base + 1024;
        return true;
    }

    // Lock-free insert of pre-assigned linked 128-bit hex user -> cid during Pass 3!
    SS_NOINLINE void insert_linked_hex(uint64_t lo, uint64_t hi, uint64_t h, uint32_t cid) {
        size_t li = size_t(h) & hmask;
        for (size_t pr = 0; pr < hcap; ++pr) {
            HexLine& line = hlines[li];
            std::atomic_ref<uint32_t> ac(line.ctrl);
            for (;;) {
                uint32_t c = ac.load(std::memory_order_acquire);
                uint32_t claimed = c & 7U;
                uint32_t ready   = (c >> 8) & 7U;
                if (ready & 1U) { if (((line.k[0][0] ^ lo) | (line.k[0][1] ^ hi)) == 0) return; }
                if (ready & 2U) { if (((line.k[1][0] ^ lo) | (line.k[1][1] ^ hi)) == 0) return; }
                if (ready & 4U) { if (((line.k[2][0] ^ lo) | (line.k[2][1] ^ hi)) == 0) return; }
                if (__builtin_expect(claimed != ready, 0)) {
                    #if defined(__x86_64__)
                    __builtin_ia32_pause();
                    #elif defined(__aarch64__)
                    asm volatile("yield");
                    #endif
                    continue;
                }
                if (claimed == 7U) break;
                uint32_t e = uint32_t(__builtin_ctz(~claimed & 7U));
                uint32_t exp = c;
                if (ac.compare_exchange_weak(exp, c | (1U << e),
                                             std::memory_order_acq_rel, std::memory_order_acquire)) {
                    line.k[e][0] = lo;
                    line.k[e][1] = hi;
                    line.pid[e]  = cid;
                    ac.fetch_or(1U << (8U + e), std::memory_order_release);
                    return;
                }
            }
            li = (li + 1) & hmask;
        }
        dict_overflow.store(true, std::memory_order_relaxed);
    }

    // Single-cache-line lock-free lookup/insert of 128-bit 'u' + 32 hex user -> dense pid!
    // Pre-leases PID & decodes name BEFORE claiming the HexLine slot so critical section is 3 stores (~3 ns)!
    SS_NOINLINE uint32_t get_or_insert_hex(
        uint64_t lo, uint64_t hi, uint64_t h,
        StringArenaPool& arenas, int tid,
        uint32_t& tl_cur, uint32_t& tl_end
    ) {
        size_t li = size_t(h) & hmask;
        const size_t max_lines = std::min<size_t>(hcap, 4096);
        uint64_t prepared_name = 0;
        for (size_t pr = 0; pr < max_lines; ++pr) {
            HexLine& line = hlines[li];
            std::atomic_ref<uint32_t> ac(line.ctrl);
            for (;;) {
                uint32_t c = ac.load(std::memory_order_acquire);
                uint32_t claimed = c & 7U;
                uint32_t ready   = (c >> 8) & 7U;
                if (ready & 1U) { if (((line.k[0][0] ^ lo) | (line.k[0][1] ^ hi)) == 0) return line.pid[0]; }
                if (ready & 2U) { if (((line.k[1][0] ^ lo) | (line.k[1][1] ^ hi)) == 0) return line.pid[1]; }
                if (ready & 4U) { if (((line.k[2][0] ^ lo) | (line.k[2][1] ^ hi)) == 0) return line.pid[2]; }
                if (__builtin_expect(claimed != ready, 0)) {
                    #if defined(__x86_64__)
                    __builtin_ia32_pause();
                    #elif defined(__aarch64__)
                    asm volatile("yield");
                    #endif
                    continue;
                }
                if (__builtin_expect(claimed == 7U, 0)) break;
                // Prepare PID lease & decoded name OUTSIDE the claimed critical section!
                if (__builtin_expect(tl_cur >= tl_end, 0)) {
                    if (!lease_pid_slow(tl_cur, tl_end)) return 0;
                }
                if (prepared_name == 0) {
                    char tmp[33];
                    decode_uhex128(lo, hi, tmp);
                    prepared_name = arenas.store_fallback(tid, tmp, 33);
                }
                uint32_t np = tl_cur;
                uint32_t e = uint32_t(__builtin_ctz(~claimed & 7U));
                uint32_t exp = c;
                if (ac.compare_exchange_weak(exp, c | (1U << e),
                                             std::memory_order_acq_rel, std::memory_order_acquire)) {
                    ++tl_cur;
                    pid_name[np] = prepared_name;
                    line.k[e][0] = lo;
                    line.k[e][1] = hi;
                    line.pid[e]  = np;
                    ac.fetch_or(1U << (8U + e), std::memory_order_release);
                    return np;
                }
            }
            li = (li + 1) & hmask;
        }
        dict_overflow.store(true, std::memory_order_relaxed);
        return 0;
    }

    // Lock-free insert of pre-assigned linked (KeyRef -> cid) into gslots during Pass 3!
    // Always hashes raw UTF-8/ASCII bytes via hash_bytes() so it matches scan lookups 100% identically!
    SS_COLD_NOINLINE void insert_linked_general(
        uint64_t k, uint32_t cid, const StringArenaPool& arenas, uint32_t& g_cur, uint32_t& g_end
    ) {
        char tmp[10];
        auto sv = key_to_sv(k, tmp, arenas);
        uint64_t h = hash_bytes(sv.data(), sv.size());
        uint32_t tag = uint32_t(h >> 32);
        size_t idx = size_t(h) & gmask;
        if (__builtin_expect(g_cur >= g_end, 0)) {
            uint32_t base = next_gentry.fetch_add(256, std::memory_order_relaxed);
            if (__builtin_expect(base >= MAX_GENTRIES, 0)) {
                dict_overflow.store(true, std::memory_order_relaxed);
                return;
            }
            g_cur = base;
            g_end = base + 256;
        }
        uint32_t ei = g_cur;
        gentries[ei] = {k, cid, 0U};
        uint64_t want = (uint64_t(tag) << 32) | uint64_t(ei + 1U);
        size_t max_probes = gcap >> 1;
        for (size_t pr = 0; pr < max_probes; ++pr) {
            std::atomic_ref<uint64_t> slot(gslots[idx]);
            uint64_t cur = slot.load(std::memory_order_acquire);
            if (cur == 0) {
                uint64_t exp = 0;
                if (slot.compare_exchange_strong(exp, want, std::memory_order_release, std::memory_order_acquire)) {
                    ++g_cur;
                    return;
                }
                cur = exp;
            }
            if (uint32_t(cur >> 32) == tag) {
                uint32_t oei = uint32_t(cur) - 1U;
                if (key_eq(gentries[oei].key, k, arenas)) return;
            }
            idx = (idx + 1) & gmask;
        }
        dict_overflow.store(true, std::memory_order_relaxed);
    }

    // Cold lock-free lookup/insert of rare general user_id (uppercase, UTF-8, unusual length, d>254 overflow) -> pid!
    SS_COLD_NOINLINE uint32_t get_or_insert_general(
        const char* s, size_t len, uint64_t h,
        uint32_t seg_id, const char* seg_base_ptr,
        StringArenaPool& arenas, int tid,
        uint32_t& tl_cur, uint32_t& tl_end,
        uint32_t& g_cur, uint32_t& g_end
    ) {
        uint32_t tag = uint32_t(h >> 32);
        size_t idx = size_t(h) & gmask;
        size_t max_probes = gcap >> 1;
        for (size_t pr = 0; pr < max_probes; ++pr) {
            std::atomic_ref<uint64_t> slot(gslots[idx]);
            uint64_t cur = slot.load(std::memory_order_acquire);
            if (cur != 0) {
                if (__builtin_expect(uint32_t(cur >> 32) == tag, 1)) {
                    uint32_t ei = uint32_t(cur) - 1U;
                    if (__builtin_expect(key_eq_raw(gentries[ei].key, s, len, arenas), 1)) {
                        return gentries[ei].pid;
                    }
                }
            } else {
                if (__builtin_expect(tl_cur >= tl_end, 0)) {
                    if (!lease_pid_slow(tl_cur, tl_end)) return 0;
                }
                if (__builtin_expect(g_cur >= g_end, 0)) {
                    uint32_t base = next_gentry.fetch_add(256, std::memory_order_relaxed);
                    if (__builtin_expect(base >= MAX_GENTRIES, 0)) {
                        dict_overflow.store(true, std::memory_order_relaxed);
                        return 0;
                    }
                    g_cur = base;
                    g_end = base + 256;
                }
                uint32_t np = tl_cur;
                uint32_t ei = g_cur;
                uint64_t kref = encode_keyref_padded(s, len, seg_id, seg_base_ptr, arenas, tid);
                gentries[ei] = {kref, np, 0U};
                pid_name[np] = kref;
                uint64_t want = (uint64_t(tag) << 32) | uint64_t(ei + 1U);
                uint64_t exp = 0;
                if (slot.compare_exchange_strong(exp, want, std::memory_order_release, std::memory_order_acquire)) {
                    ++tl_cur;
                    ++g_cur;
                    return np;
                }
                if (uint32_t(exp >> 32) == tag) {
                    uint32_t oei = uint32_t(exp) - 1U;
                    if (key_eq_raw(gentries[oei].key, s, len, arenas)) {
                        return gentries[oei].pid;
                    }
                }
            }
            idx = (idx + 1) & gmask;
        }
        dict_overflow.store(true, std::memory_order_relaxed);
        return 0;
    }

    SS_COLD_NOINLINE uint32_t get_or_insert_uid48_overflow(
        uint64_t u48,
        StringArenaPool& arenas, int tid,
        uint32_t& tl_cur, uint32_t& tl_end,
        uint32_t& g_cur, uint32_t& g_end
    ) {
        char tmp[16]{};
        size_t ulen = decode_uid48_inline(u48, tmp);
        uint64_t gh = hash_bytes(tmp, ulen);
        return get_or_insert_general(tmp, ulen, gh, UINT32_MAX, nullptr, arenas, tid, tl_cur, tl_end, g_cur, g_end);
    }

    // Lock-free, carry-safe (d = 1..254) lookup/insert of uid48 -> dense 24-bit pid!
    // Cold lease refill & d>254 overflow moved out-of-line for minimal DSB footprint!
    SS_ALWAYS_INLINE uint32_t get_or_insert_uid48(
        uint64_t u48, uint64_t h48,
        StringArenaPool& arenas, int tid,
        uint32_t& tl_cur, uint32_t& tl_end,
        uint32_t& g_cur, uint32_t& g_end
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
                    if (!lease_pid_slow(tl_cur, tl_end)) return 0;
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
        return get_or_insert_uid48_overflow(u48, arenas, tid, tl_cur, tl_end, g_cur, g_end);
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
        if (hlines && hlines_bytes) ::madvise(hlines, hlines_bytes, MADV_DONTNEED);
        if (gslots && gslots_bytes) ::madvise(gslots, gslots_bytes, MADV_DONTNEED);
        if (slots && slots_bytes) ::madvise(slots, slots_bytes, MADV_DONTNEED);
#endif
    }

    void build(
        const std::vector<CsvTask>& tasks,
        std::vector<ReadBuf>& rbufs,
        ThreadPool& pool,
        StringArenaPool& arenas,
        size_t total_event_bytes,
        bool allow_fast_dict
    ) {
        struct Edge { uint64_t a, b; };
        std::vector<std::vector<Edge>> thread_edges(static_cast<size_t>(pool.nthreads));
        std::atomic<size_t> next_t{0};
        std::atomic<size_t> total_hex_links{0};
        std::atomic<size_t> total_nonhex_links{0};

        // Pass 1: Pure parallel links/*.csv scan (products/campaigns load concurrently on dedicated thread!)
        if (!tasks.empty()) {
            pool.run([&](int tid) {
                auto& edges = thread_edges[size_t(tid)];
                edges.reserve(65536);
                auto& rbuf = rbufs[size_t(tid)];
                size_t local_hex = 0, local_nonhex = 0;
                while (true) {
                    size_t ti = next_t.fetch_add(1, std::memory_order_relaxed);
                    if (ti >= tasks.size()) break;
                    const CsvTask& task = tasks[ti];
                    const uint32_t seg_id = task.file->seg_id;
                    const char* seg_base = task.file->map_base;
                    scan_csv<2>(task, rbuf, [&](const Slice f[2]) SS_LAMBDA_INLINE {
                        if (__builtin_expect((f[0].len > 0) & (f[1].len > 0), 1)) {
                            uint64_t ka = encode_keyref_padded(f[0].ptr, f[0].len, seg_id, seg_base, arenas, tid);
                            uint64_t kb = encode_keyref_padded(f[1].ptr, f[1].len, seg_id, seg_base, arenas, tid);
                            if (__builtin_expect(ka == kb, 0)) return;
                            if (__builtin_expect(((ka | kb) & FALLBACK_BIT) != 0, 0)) {
                                if (key_eq(ka, kb, arenas)) return;
                                if ((ka & FALLBACK_BIT) != 0) {
                                    if (f[0].len == 33 && f[0].ptr[0] == 'u') ++local_hex; else ++local_nonhex;
                                }
                                if ((kb & FALLBACK_BIT) != 0) {
                                    if (f[1].len == 33 && f[1].ptr[0] == 'u') ++local_hex; else ++local_nonhex;
                                }
                            }
                            edges.push_back({ka, kb});
                        }
                    });
                }
                if (local_hex) total_hex_links.fetch_add(local_hex, std::memory_order_relaxed);
                if (local_nonhex) total_nonhex_links.fetch_add(local_nonhex, std::memory_order_relaxed);
            });
        }

        size_t total_edges = 0;
        for (int t = 0; t < pool.nthreads; ++t) total_edges += thread_edges[size_t(t)].size();
        size_t hex_cnt = total_hex_links.load(std::memory_order_relaxed);
        size_t nonhex_cnt = total_nonhex_links.load(std::memory_order_relaxed);
        has_fallback = ((hex_cnt + nonhex_cnt) > 0);

        // Allocate Unified Lock-Free Total User Dictionary (uid48 + 3-way 64B HexDict + general fallback!)
        if (allow_fast_dict) {
            size_t est_users = std::max<size_t>(total_edges * 2, total_edges + total_event_bytes / 1500);
            ucap = 65536;
            while (ucap * 3 < est_users * 5) ucap <<= 1;
            umask = ucap - 1;
            uslots_bytes = alloc_mmap_actual_size(ucap * sizeof(uint64_t), ucap * sizeof(uint64_t) >= (2ULL << 20));
            uslots = static_cast<uint64_t*>(alloc_mmap(ucap * sizeof(uint64_t), ucap * sizeof(uint64_t) >= (2ULL << 20)));

            // Size 3-way 64B HexDict cleanly: <= 0.6 entry load (256K lines = 16 MB at 20M, 512K lines = 32 MB at 60M!)
            size_t est_hex = std::max<size_t>((hex_cnt >> 1) + total_event_bytes / 12000, 4096);
            if (total_edges > 0 && hex_cnt * 4 > total_edges * 2) {
                est_hex = std::max(est_hex, est_users);
            }
            hcap = 4096;
            while (hcap * 9 < est_hex * 5) hcap <<= 1;
            hmask = hcap - 1;
            bool h_hp = (hcap * sizeof(HexLine) >= (2ULL << 20));
            hlines_bytes = alloc_mmap_actual_size(hcap * sizeof(HexLine), h_hp);
            hlines = static_cast<HexLine*>(alloc_mmap(hcap * sizeof(HexLine), h_hp));

            // gslots now handles only rare non-hex / UTF-8 / uppercase IDs -> 4 MB VA non-THP (almost 0 RAM touched!),
            // scaling up automatically if a dataset actually has many non-hex long links!
            size_t est_gen = std::max<size_t>(nonhex_cnt * 2 + 131072, 131072);
            if (total_edges > 0 && nonhex_cnt * 4 > total_edges * 2) {
                est_gen = std::max(est_gen, est_users);
            }
            gcap = 262144;
            while (gcap * 2 < est_gen * 5) gcap <<= 1;
            gmask = gcap - 1;
            bool g_hp = (nonhex_cnt >= 65536) && (gcap * sizeof(uint64_t) >= (2ULL << 20));
            gslots_bytes = alloc_mmap_actual_size(gcap * sizeof(uint64_t), g_hp);
            gslots = static_cast<uint64_t*>(alloc_mmap(gcap * sizeof(uint64_t), g_hp));

            gentries_bytes = size_t(MAX_GENTRIES) * sizeof(GenEntry);
            gentries = static_cast<GenEntry*>(alloc_mmap(gentries_bytes, false));
            next_gentry.store(0, std::memory_order_relaxed);

            pid_name_bytes = alloc_mmap_actual_size(PID_NAME_CAP * sizeof(uint64_t), true);
            pid_name = static_cast<uint64_t*>(alloc_mmap(PID_NAME_CAP * sizeof(uint64_t), true));
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
            size_t dcap = dsu_bytes / sizeof(DsuSlot);
            auto* dsu = static_cast<DsuSlot*>(alloc_mmap(dsu_bytes, dsu_hp));

            auto slot_of = [&](uint64_t k) SS_LAMBDA_INLINE -> size_t {
                return size_t((static_cast<unsigned __int128>(hash_key64(k, arenas)) * dcap) >> 64);
            };
            // Pass 2a: Lock-free interning of ALL KeyRef endpoints into dsu[]!
            auto insert_key = [&](uint64_t k) SS_LAMBDA_INLINE -> uint32_t {
                size_t s = slot_of(k);
                while (true) {
                    std::atomic_ref<uint64_t> ak(dsu[s].key);
                    uint64_t cur = ak.load(std::memory_order_acquire);
                    if (cur != 0) {
                        if (key_eq(cur, k, arenas)) return uint32_t(s);
                    } else {
                        uint64_t exp = 0;
                        if (ak.compare_exchange_strong(exp, k, std::memory_order_release, std::memory_order_acquire) ||
                            key_eq(exp, k, arenas)) {
                            return uint32_t(s);
                        }
                    }
                    if (++s == dcap) s = 0;
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

            // Pass 2b: Parallel Union-by-Exact-Key over 100% IMMUTABLE dsu[].key
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
                        if (key_less(dsu[u].key, dsu[v].key, arenas)) std::swap(u, v);
                        uint32_t exp = 0;
                        if (std::atomic_ref<uint32_t>(dsu[u].parent).compare_exchange_weak(
                                exp, v + 1, std::memory_order_relaxed)) {
                            break;
                        }
                    }
                }
            });

            // Pass 3 (64-SLOT SOFTWARE-PIPELINED PARALLEL PUBLISH!):
            // Batched root-walk prefetch + write-intent uslots/hlines home prefetch!
            pool.run([&](int tid) {
                size_t s = (dcap * size_t(tid)) / size_t(pool.nthreads);
                size_t e = (dcap * size_t(tid + 1)) / size_t(pool.nthreads);
                uint32_t tl_cur = 0, tl_end = 0;
                uint32_t g_cur = 0, g_end = 0;

                static constexpr uint32_t WIN = 64;
                uint64_t b_key[WIN];
                uint64_t b_v0[WIN];
                uint64_t b_v1[WIN];
                uint64_t b_vh[WIN];
                uint32_t b_root[WIN];
                uint32_t b_cid[WIN];
                uint8_t  b_dom[WIN]; // 0 = uid48, 1 = uhex128, 2 = general
                uint8_t  act[WIN];

                size_t pos = s;
                while (pos < e) {
                    uint32_t cnt = 0;
                    uint32_t na = 0;
                    while (pos < e && cnt < WIN) {
                        uint64_t k = dsu[pos].key;
                        if (k != 0) {
                            b_key[cnt] = k;
                            uint32_t p1 = dsu[pos].parent;
                            if (p1 == 0) {
                                b_root[cnt] = uint32_t(pos);
                            } else {
                                uint32_t nxt = p1 - 1;
                                b_root[cnt] = nxt;
                                __builtin_prefetch(&dsu[nxt], 0, 1);
                                act[na++] = uint8_t(cnt);
                            }
                            ++cnt;
                        }
                        ++pos;
                    }
                    if (cnt == 0) break;

                    // Step A: Lockstep prefetched root-walk rounds for the batch
                    while (na > 0) {
                        uint32_t nna = 0;
                        for (uint32_t ai = 0; ai < na; ++ai) {
                            uint8_t j = act[ai];
                            uint32_t p1 = dsu[b_root[j]].parent;
                            if (p1 != 0) {
                                uint32_t nxt = p1 - 1;
                                b_root[j] = nxt;
                                __builtin_prefetch(&dsu[nxt], 0, 1);
                                act[nna++] = j;
                            }
                        }
                        na = nna;
                    }

                    // Step B: Assign root comp_id + pure-byte domain dispatch + write-intent home prefetch
                    for (uint32_t j = 0; j < cnt; ++j) {
                        uint32_t r = b_root[j];
                        std::atomic_ref<uint32_t> ac(dsu[r].comp_id);
                        uint32_t cid = ac.load(std::memory_order_relaxed);
                        if (cid == 0) {
                            if (__builtin_expect(tl_cur >= tl_end, 0)) {
                                uint32_t base = next_pid.fetch_add(256, std::memory_order_relaxed);
                                if (__builtin_expect(base >= MAX_DICT_PIDS, 0)) {
                                    dict_overflow.store(true, std::memory_order_relaxed);
                                    b_dom[j] = 3;
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
                        b_cid[j] = cid;

                        uint64_t k = b_key[j];
                        if (__builtin_expect((k & FALLBACK_BIT) == 0, 1)) {
                            char tmp[16]{};
                            size_t len = decode_inline(k, tmp);
                            uint64_t u48 = 0;
                            if (__builtin_expect(encode_uid48_padded(tmp, len, u48), 1)) {
                                uint64_t h48 = mix48(u48);
                                b_dom[j] = 0;
                                b_v0[j] = h48;
                                __builtin_prefetch(&uslots[size_t(h48) & umask], 1, 1);
                                continue;
                            }
                            b_dom[j] = 2;
                        } else {
                            auto sv = arenas.get_fallback(k);
                            uint64_t lo = 0, hi = 0;
                            if (__builtin_expect(encode_uhex128(sv.data(), sv.size(), lo, hi), 1)) {
                                uint64_t h = hash_uhex128(lo, hi);
                                b_dom[j] = 1;
                                b_v0[j] = lo;
                                b_v1[j] = hi;
                                b_vh[j] = h;
                                __builtin_prefetch(&hlines[size_t(h) & hmask], 1, 1);
                            } else {
                                b_dom[j] = 2;
                            }
                        }
                    }

                    // Step C: Execute prefetched CAS inserts into uslots / hlines / gslots!
                    for (uint32_t j = 0; j < cnt; ++j) {
                        uint8_t dm = b_dom[j];
                        if (__builtin_expect(dm == 0, 1)) {
                            uint64_t h48 = b_v0[j];
                            uint32_t cid = b_cid[j];
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
                            if (__builtin_expect(!placed, 0)) {
                                insert_linked_general(b_key[j], cid, arenas, g_cur, g_end);
                            }
                        } else if (dm == 1) {
                            insert_linked_hex(b_v0[j], b_v1[j], b_vh[j], b_cid[j]);
                        } else if (dm == 2) {
                            insert_linked_general(b_key[j], b_cid[j], arenas, g_cur, g_end);
                        }
                    }
                }
            });

            if (__builtin_expect(!dict_overflow.load(std::memory_order_relaxed), 1)) {
                fast_dict_mode = true;
                if (dsu_bytes >= (4ULL << 20)) {
                    recycled_dsu = dsu;
                    recycled_dsu_bytes = dsu_bytes;
                }
                return;
            }

            // Ultra-rare >16.7M link components -> fall back to classic!
            uslots = nullptr; uslots_bytes = 0;
            hlines = nullptr; hlines_bytes = 0;
            gslots = nullptr; gslots_bytes = 0;
            gentries = nullptr; gentries_bytes = 0;
            pid_name = nullptr; pid_name_bytes = 0;
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

// Compact 32-byte SideEvent for fast_dict_mode (exact half cache line! Covers 12.5% long EID + 3% purchases!)
struct FastSideEvent {
    uint64_t eid_key64;
    uint32_t ts_rel_ms;
    uint32_t pid;
    int64_t  ingest_ms;
    uint32_t price;
    uint32_t cat_id;
};
static_assert(sizeof(FastSideEvent) == 32, "FastSideEvent must be exactly 32 bytes");

// Pre-reserved 2MB-THP bump slab for FastSideEvent (zero realloc copies, zero 4KB page fault storms!)
struct FastSideSlab {
    static constexpr uint32_t MAX_ITEMS = 1U << 26;
    FastSideEvent* data = nullptr;
    uint32_t count = 0;
    uint32_t cap = 0;

    void init(size_t reserve_bytes) {
        size_t aligned = (reserve_bytes + (2ULL << 20) - 1) & ~((2ULL << 20) - 1);
        if (aligned < (4ULL << 20)) aligned = (4ULL << 20);
        size_t max_bytes = size_t(MAX_ITEMS) * sizeof(FastSideEvent);
        if (aligned > max_bytes) aligned = max_bytes;
        data = static_cast<FastSideEvent*>(alloc_mmap(aligned, true));
        count = 0;
        cap = uint32_t(aligned / sizeof(FastSideEvent));
    }

    SS_COLD_NOINLINE void grow() {
        size_t new_cap = std::min<size_t>(size_t(MAX_ITEMS), size_t(cap) * 2);
        if (new_cap <= cap) std::abort();
        size_t new_bytes = new_cap * sizeof(FastSideEvent);
        auto* nd = static_cast<FastSideEvent*>(alloc_mmap(new_bytes, new_bytes >= (2ULL << 20)));
        if (count) std::memcpy(nd, data, size_t(count) * sizeof(FastSideEvent));
        data = nd;
        cap = uint32_t(new_cap);
    }

    SS_ALWAYS_INLINE uint32_t push(const FastSideEvent& ev) {
        if (__builtin_expect(count >= cap, 0)) grow();
        uint32_t idx = count++;
        data[idx] = ev;
        return idx;
    }

    SS_ALWAYS_INLINE const FastSideEvent& operator[](size_t i) const { return data[i]; }
};

struct SideEvent {
    uint64_t eid_key64;
    uint64_t user_key64; // used in wide/classic mode
    int64_t ts_ms;
    int64_t ingest_ms;
    uint32_t pid;
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

// Direct 32-byte inline bucket record for all 'e' + 32 lowercase hex events (2 per 64B AVX2 NT flush!).
// Eliminates FastSideSlab random DRAM reads, StringArena copies, and mmap memcmp for 99.99% of long EIDs!
struct LongEv32 {
    uint64_t k_lo, k_hi;
    uint64_t ing_pid;   // [(ing_abs_sec + (1LL << 39)) : 40][pid : 24]
    uint64_t payload;   // [ts_rel_ms : 32][price : 24][cat_id : 7][valid : 1]

    static constexpr int64_t ABS_BIAS = 1LL << 39;

    static SS_ALWAYS_INLINE LongEv32 make_valid(
        uint64_t lo, uint64_t hi, int64_t ing_abs_sec, uint32_t pid,
        uint32_t ts_rel_ms, uint32_t price, uint32_t cat_id
    ) {
        uint64_t a = (uint64_t(ing_abs_sec + ABS_BIAS) << 24) | uint64_t(pid);
        uint64_t b = (uint64_t(ts_rel_ms) << 32) | (uint64_t(price) << 8) | (uint64_t(cat_id) << 1) | 1ULL;
        return {lo, hi, a, b};
    }

    static SS_ALWAYS_INLINE LongEv32 make_invalid(uint64_t lo, uint64_t hi, int64_t ing_abs_sec) {
        uint64_t a = uint64_t(ing_abs_sec + ABS_BIAS) << 24;
        return {lo, hi, a, 0ULL};
    }

    SS_ALWAYS_INLINE uint64_t ing40() const { return ing_pid >> 24; }
    SS_ALWAYS_INLINE uint32_t pid() const { return uint32_t(ing_pid & 0xFFFFFFULL); }
    SS_ALWAYS_INLINE bool is_valid() const { return (payload & 1ULL) != 0; }
    SS_ALWAYS_INLINE uint32_t ts_rel_ms() const { return uint32_t(payload >> 32); }
    SS_ALWAYS_INLINE uint32_t price() const { return uint32_t((payload >> 8) & 0xFFFFFFULL); }
    SS_ALWAYS_INLINE uint32_t cat_id() const { return uint32_t((payload >> 1) & 0x7FULL); }
};
static_assert(sizeof(LongEv32) == 32, "LongEv32 must be 32 bytes");

template <typename SideContainer>
static SS_COLD_NOINLINE int64_t get_exact_ingest_ms_slow(
    const RawEvent16& ev,
    int64_t win_start_sec,
    const std::vector<SideContainer>& sides
) {
    uint32_t ir = ev.ingest_rel();
    if (ir != 0) return (win_start_sec + int64_t(ir) - RawEvent16::ING_BIAS) * 1000LL;
    return sides[ev.w1 >> 26][ev.w1 & 0x03FFFFFFU].ingest_ms;
}

template <typename SideContainer>
static SS_ALWAYS_INLINE bool ingest_strictly_earlier(
    const RawEvent16& a,
    const RawEvent16& b,
    int64_t win_start_sec,
    const std::vector<SideContainer>& sides
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
        uid_dict_cap = ((rcap / 2) * 20 + 65536 + 63) & ~size_t(63);
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

    // Direct zero-intermediate-copy emit for external KeyRef (e.g., 33B hex user_ids)!
    SS_ALWAYS_INLINE void emit_person_uid_sv(const char* s, size_t len, uint32_t run_rows) {
        uint32_t ulen = uint32_t(len);
        size_t tot = 4 + len;
        if (__builtin_expect(uid_dict_sz + tot + 16 > uid_dict_cap, 0)) grow_uid_dict(tot + 16);
        std::memcpy(uid_dict + uid_dict_sz, &ulen, 4);
        if (len) std::memcpy(uid_dict + uid_dict_sz + 4, s, len);
        uid_dict_sz += tot;
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
// Split L1I-Hot Per-Task Scanners (Zero-Spill Lazy Rows + 24B Core Ring + LongEv32!)
// ============================================================================
static constexpr size_t RING_SZ   = 32;
static constexpr size_t RING_MASK = RING_SZ - 1;
static constexpr size_t RING_LAG  = 24;

static constexpr uint64_t EV_F_SHORT_U   = 1ULL << 0;
static constexpr uint64_t EV_F_HEX_U     = 1ULL << 1;
static constexpr uint64_t EV_F_NEED_SIDE = 1ULL << 2;
static constexpr uint64_t EV_F_EID_SIDE  = 1ULL << 3;
static constexpr uint64_t EV_F_LONG_EID  = 1ULL << 4;

struct EvRingHex {
    uint64_t lo, hi;
};
struct EvRingGen {
    const char* ptr;
    const char* seg_base;
    uint32_t len;
    uint32_t seg_id;
};
struct EvRingSide {
    uint64_t eid_key64;
    int64_t  ing_ms;
    uint32_t price;
    uint32_t cat_id;
};
struct EvRingLong {
    uint64_t elo, ehi;
    int64_t  ing_abs_sec;
    uint32_t price;
    uint32_t cat_id;
};

// 24 B core (3 stores/row! Recomputes mix48(u48) in 3 uops at emit!)
struct EvRingCore {
    uint64_t q0;        // [u48:48 (bits 63..16)][eb:9 (bits 15..7)][flags:7 (bits 6..0)]
    uint64_t q1;        // [rem39:39 (bits 63..25)][irel21:21 (bits 24..4)][0:4]
    uint32_t ts_rel_ms;
    uint32_t _pad;
};
static_assert(sizeof(EvRingCore) == 24, "EvRingCore must be 24 bytes");

struct EvRingState {
    EvRingCore core[RING_SZ];
    EvRingHex  hex[RING_SZ];
    EvRingLong long_ev[RING_SZ];
    EvRingSide side[RING_SZ];
    EvRingGen  gen[RING_SZ];
};

// 24 B click core (3 stores/row!)
// u48_kind = (u48 << 2) | kind, where kind: 0 = uid48, 1 = uhex128, 2 = general
struct ClickRingCore {
    uint64_t u48_kind;
    uint64_t cid;
    uint64_t ts_camp44;
};
static_assert(sizeof(ClickRingCore) == 24, "ClickRingCore must be 24 bytes");

struct ClickRingState {
    ClickRingCore core[RING_SZ];
    EvRingHex     hex[RING_SZ];
    EvRingGen     gen[RING_SZ];
};

static SS_NOINLINE void emit_ev_extra_slow(
    uint64_t q0, uint32_t eb, uint32_t irel21, uint64_t rem39,
    uint32_t ts_rel_ms, uint32_t pid,
    const EvRingLong& lg, const EvRingSide& sd, int tid,
    PartitionedStore<RawEvent16, 256, EV_BUCKETS, 4>::ThreadState& ev_out,
    PartitionedStore<LongEv32, 128, EV_BUCKETS, 2>::ThreadState& long_ev_out,
    FastSideSlab& side_slab
) {
    if (__builtin_expect((q0 & EV_F_LONG_EID) != 0, 1)) {
        long_ev_out.push(eb, LongEv32::make_valid(lg.elo, lg.ehi, lg.ing_abs_sec, pid, ts_rel_ms, lg.price, lg.cat_id));
    } else {
        uint32_t sidx = side_slab.push({sd.eid_key64, ts_rel_ms, pid, sd.ing_ms, sd.price, sd.cat_id});
        ev_out.push(eb, RawEvent16::make_side(true, irel21, (q0 & EV_F_EID_SIDE) != 0, rem39, tid, sidx));
    }
}

static SS_COLD_NOINLINE void stage_gen_user_ev_cold(
    const char* uptr, size_t ulen, uint32_t seg_id, const char* seg_base,
    EvRingGen& dst, const UserDict& dict
) {
    dst = {uptr, seg_base, uint32_t(ulen), seg_id};
    dict.prefetch_gslot(hash_bytes(uptr, ulen));
}

static SS_COLD_NOINLINE void emit_ev_direct_gen_cold(
    const char* uptr, size_t ulen,
    uint32_t eb, uint32_t irel21, uint64_t rem39, bool is_long32, bool need_side, bool eid_in_side,
    uint32_t ts_rel_ms, uint64_t elo, uint64_t ehi, int64_t ing_abs_sec,
    uint64_t eid_k64, int64_t ing_ms, uint32_t price, uint32_t cat_id,
    UserDict& dict, StringArenaPool& arenas, int tid,
    PartitionedStore<RawEvent16, 256, EV_BUCKETS, 4>::ThreadState& ev_out,
    PartitionedStore<LongEv32, 128, EV_BUCKETS, 2>::ThreadState& long_ev_out,
    FastSideSlab& side_slab,
    uint32_t& tl_cur, uint32_t& tl_end, uint32_t& g_cur, uint32_t& g_end
) {
    uint64_t h = hash_bytes(uptr, ulen);
    uint32_t pid = dict.get_or_insert_general(uptr, ulen, h, UINT32_MAX, nullptr, arenas, tid, tl_cur, tl_end, g_cur, g_end);
    if (is_long32) {
        long_ev_out.push(eb, LongEv32::make_valid(elo, ehi, ing_abs_sec, pid, ts_rel_ms, price, cat_id));
    } else if (!need_side) {
        ev_out.push(eb, RawEvent16::make_inline_pid(irel21, rem39, ts_rel_ms, pid));
    } else {
        uint32_t sidx = side_slab.push({eid_k64, ts_rel_ms, pid, ing_ms, price, cat_id});
        ev_out.push(eb, RawEvent16::make_side(true, irel21, eid_in_side, rem39, tid, sidx));
    }
}

static SS_COLD_NOINLINE void emit_clk_direct_gen_cold(
    const char* uptr, size_t ulen, uint64_t cid, uint64_t ts_camp44,
    UserDict& dict, StringArenaPool& arenas, int tid,
    PartitionedStore<CompactClick, 128, PERSON_BUCKETS, 4>::ThreadState& clk_out,
    uint32_t& tl_cur, uint32_t& tl_end, uint32_t& g_cur, uint32_t& g_end
) {
    uint64_t h = hash_bytes(uptr, ulen);
    uint32_t pid = dict.get_or_insert_general(uptr, ulen, h, UINT32_MAX, nullptr, arenas, tid, tl_cur, tl_end, g_cur, g_end);
    uint32_t pb = pid & uint32_t(PERSON_BUCKETS - 1);
    clk_out.push(pb, CompactClick{cid, (uint64_t(pid >> PERSON_SHIFT) << 44) | ts_camp44});
}

struct PurRes {
    uint32_t price;
    uint32_t cat_id;
    bool ok;
};

static SS_COLD_NOINLINE PurRes parse_purchase_cold(
    const char* price_ptr, size_t price_len,
    const char* prod_ptr, size_t prod_len,
    const Products& products
) {
    uint32_t pr = 0;
    if (!parse_price_cents(price_ptr, price_len, pr)) return {0U, 0U, false};
    return {pr, products.lookup(prod_ptr, prod_len), true};
}

struct LongEidRes {
    uint64_t elo;
    uint64_t ehi;
    uint64_t k64;
    uint64_t rem39;
    uint32_t eb;
    bool is_long32;
};

static SS_NOINLINE LongEidRes encode_long_eid_slow(
    const char* s, size_t len, bool use_long_ev32, bool ing_whole_sec,
    uint32_t seg_id, const char* seg_base,
    StringArenaPool& arenas, int tid, UserDict& dict
) {
    uint64_t elo = 0, ehi = 0;
    if (__builtin_expect(use_long_ev32 && encode_ehex128(s, len, elo, ehi), 1)) {
        if (__builtin_expect(!ing_whole_sec, 0)) {
            dict.dict_overflow.store(true, std::memory_order_relaxed);
            return {0ULL, 0ULL, 0ULL, 0ULL, 0U, false};
        }
        uint64_t h = hash_uhex128(elo, ehi);
        return {elo, ehi, 0ULL, 0ULL, uint32_t(h >> (64 - EV_SHIFT)), true};
    }
    uint64_t k64 = encode_keyref_padded(s, len, seg_id, seg_base, arenas, tid);
    uint64_t h64 = ((k64 & FALLBACK_BIT) == 0) ? mix64(k64) : hash_bytes(s, len);
    return {0ULL, 0ULL, k64, h64 & RawEvent16::REM39_MASK, uint32_t(h64 >> 55), false};
}

struct IngSlowRes {
    int64_t ing_ms;
    int64_t ing_rel_sec;
    bool ok;
    bool whole_sec;
};

static SS_COLD_NOINLINE IngSlowRes parse_ingest_slow_ev(
    const char* s, size_t len, const DateLut& lut, int64_t win_start_sec
) {
    int64_t ims = 0;
    if (!parse_ts(s, len, ims, lut)) return {0LL, 0LL, false, false};
    return {ims, ims / 1000LL - win_start_sec, true, (ims % 1000LL == 0)};
}

struct TsCheckSlowRes {
    int64_t ing_ms;
    int64_t ts_ms;
    bool ok;
    bool from_sec;
};

static SS_COLD_NOINLINE TsCheckSlowRes check_ts_slow_ev(
    const char* s, size_t len, const DateLut& lut,
    bool ts_ok, bool ts_from_sec, bool ing_whole_sec,
    int64_t win_start_sec, int64_t win_start_ms, int64_t win_dur_ms,
    int64_t ts_rel_sec, uint32_t ts_ms_rem, int64_t ing_rel_sec,
    int64_t ing_ms, int64_t ts_ms
) {
    if (!ts_ok) {
        ts_ok = parse_ts_slow(s, len, ts_ms, lut);
        ts_from_sec = false;
    }
    if (!ts_ok) return {ing_ms, ts_ms, false, ts_from_sec};
    if (ts_from_sec) ts_ms = (win_start_sec + ts_rel_sec) * 1000LL + int64_t(ts_ms_rem);
    if (ing_whole_sec) ing_ms = (win_start_sec + ing_rel_sec) * 1000LL;
    bool ok = !(uint64_t(ts_ms - win_start_ms) >= uint64_t(win_dur_ms) || ing_ms - ts_ms > DAY_MS);
    return {ing_ms, ts_ms, ok, ts_from_sec};
}

static SS_COLD_NOINLINE void push_invalid_ev_cold(
    bool is_long32, uint64_t elo, uint64_t ehi,
    bool eid_in_side, uint32_t irel21, uint32_t eb, uint64_t rem39,
    uint64_t eid_k64, bool ing_whole_sec, int64_t win_start_sec, int64_t ing_rel_sec, int64_t ing_ms,
    int tid,
    PartitionedStore<RawEvent16, 256, EV_BUCKETS, 4>::ThreadState& ev_out,
    PartitionedStore<LongEv32, 128, EV_BUCKETS, 2>::ThreadState& long_ev_out,
    FastSideSlab& side_slab
) {
    if (is_long32) {
        long_ev_out.push(eb, LongEv32::make_invalid(elo, ehi, win_start_sec + ing_rel_sec));
        return;
    }
    if (!eid_in_side && irel21 != 0) {
        ev_out.push(eb, RawEvent16::make_invalid_inline(irel21, rem39));
        return;
    }
    if (ing_whole_sec) ing_ms = (win_start_sec + ing_rel_sec) * 1000LL;
    uint32_t sidx = side_slab.push({eid_k64, 0U, 0U, ing_ms, 0U, 0U});
    ev_out.push(eb, RawEvent16::make_side(false, irel21, eid_in_side, rem39, tid, sidx));
}

static SS_COLD_NOINLINE uint32_t resolve_gen_user_cold(
    const EvRingGen& gn, UserDict& dict, StringArenaPool& arenas, int tid,
    uint32_t& tl_cur, uint32_t& tl_end, uint32_t& g_cur, uint32_t& g_end
) {
    return dict.get_or_insert_general(gn.ptr, gn.len, hash_bytes(gn.ptr, gn.len),
                                      gn.seg_id, gn.seg_base, arenas, tid, tl_cur, tl_end, g_cur, g_end);
}

static SS_ALWAYS_INLINE void emit_click_ring_entry(
    const ClickRingState& st, size_t slot,
    UserDict& dict,
    StringArenaPool& arenas,
    int tid,
    PartitionedStore<CompactClick, 128, PERSON_BUCKETS, 4>::ThreadState& clk_out,
    uint32_t& tl_cur, uint32_t& tl_end,
    uint32_t& g_cur,  uint32_t& g_end
) {
    const ClickRingCore& r = st.core[slot];
    uint64_t uk = r.u48_kind;
    uint32_t kind = uint32_t(uk & 3ULL);
    uint32_t pid;
    if (__builtin_expect(kind == 0U, 1)) {
        uint64_t u48 = uk >> 2;
        pid = dict.get_or_insert_uid48(u48, mix48(u48), arenas, tid, tl_cur, tl_end, g_cur, g_end);
    } else if (__builtin_expect(kind == 1U, 1)) {
        const EvRingHex& hx = st.hex[slot];
        pid = dict.get_or_insert_hex(hx.lo, hx.hi, hash_uhex128(hx.lo, hx.hi), arenas, tid, tl_cur, tl_end);
    } else {
        pid = resolve_gen_user_cold(st.gen[slot], dict, arenas, tid, tl_cur, tl_end, g_cur, g_end);
    }
    uint32_t pb = pid & uint32_t(PERSON_BUCKETS - 1);
    clk_out.push(pb, CompactClick{r.cid, (uint64_t(pid >> PERSON_SHIFT) << 44) | r.ts_camp44});
}

static SS_ALWAYS_INLINE void emit_ev_ring_entry(
    const EvRingState& st, size_t slot,
    UserDict& dict,
    StringArenaPool& arenas,
    int tid,
    PartitionedStore<RawEvent16, 256, EV_BUCKETS, 4>::ThreadState& ev_out,
    PartitionedStore<LongEv32, 128, EV_BUCKETS, 2>::ThreadState& long_ev_out,
    FastSideSlab& side_slab,
    uint32_t& tl_cur, uint32_t& tl_end,
    uint32_t& g_cur,  uint32_t& g_end
) {
    const EvRingCore& r = st.core[slot];
    const uint64_t q0 = r.q0;
    const uint64_t q1 = r.q1;
    const uint32_t ts_rel_ms = r.ts_rel_ms;
    uint32_t pid;
    if (__builtin_expect((q0 & EV_F_SHORT_U) != 0, 1)) {
        uint64_t u48 = q0 >> 16;
        pid = dict.get_or_insert_uid48(u48, mix48(u48), arenas, tid, tl_cur, tl_end, g_cur, g_end);
    } else if (__builtin_expect((q0 & EV_F_HEX_U) != 0, 1)) {
        const EvRingHex& hx = st.hex[slot];
        pid = dict.get_or_insert_hex(hx.lo, hx.hi, hash_uhex128(hx.lo, hx.hi), arenas, tid, tl_cur, tl_end);
    } else {
        pid = resolve_gen_user_cold(st.gen[slot], dict, arenas, tid, tl_cur, tl_end, g_cur, g_end);
    }
    const uint32_t eb = uint32_t((q0 >> 7) & 0x1FFULL);
    const uint64_t rem39 = q1 >> 25;
    const uint32_t irel21 = uint32_t((q1 >> 4) & 0x1FFFFFULL);
    if (__builtin_expect((q0 & (EV_F_NEED_SIDE | EV_F_LONG_EID)) == 0, 1)) {
        ev_out.push(eb, RawEvent16::make_inline_pid(irel21, rem39, ts_rel_ms, pid));
    } else {
        emit_ev_extra_slow(q0, eb, irel21, rem39, ts_rel_ms, pid,
                           st.long_ev[slot], st.side[slot], tid, ev_out, long_ev_out, side_slab);
    }
}

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
    ClickRingState& cring,
    size_t& cpushed,
    uint32_t& tl_cur, uint32_t& tl_end,
    uint32_t& g_cur,  uint32_t& g_end
) {
    const uint32_t seg_id = task.file->seg_id;
    const char* seg_base = task.file->map_base;
    const bool is_mmap = (seg_id < StringArenaPool::ARENA_SEG_BASE);

    scan_csv_rows<4>(task, rbuf, [&](auto row) SS_LAMBDA_INLINE {
        Slice f2 = row.template get<2>();
        if (__builtin_expect(f2.len == 0, 0)) return;
        uint32_t camp_idx = campaigns.lookup_padded(f2.ptr, f2.len);
        if (__builtin_expect(camp_idx == FastStringIdMap::NOT_FOUND, 0)) return;
        Slice f3 = row.template get<3>();
        int64_t ts = 0;
        if (__builtin_expect(f3.len == 0 || !parse_ts(f3.ptr, f3.len, ts, lut) ||
                             ts < click_start_ms || ts >= win_end_ms, 0)) return;
        Slice f0 = row.template get<0>();
        Slice f1 = row.template get<1>();
        if (__builtin_expect((f0.len == 0) | (f1.len == 0), 0)) return;

        uint64_t cid = encode_keyref_padded(f0.ptr, f0.len, seg_id, seg_base, arenas, tid);
        uint64_t ts_camp44 = (uint64_t(uint32_t(ts - click_start_ms)) << 12) | uint64_t(camp_idx);
        const size_t slot = cpushed & RING_MASK;
        uint64_t u48 = 0;
        uint64_t uk = 0;
        if (__builtin_expect(encode_uid48_padded(f1.ptr, f1.len, u48), 1)) {
            uk = (u48 << 2);
            dict.prefetch_uslot(mix48(u48));
        } else {
            uint64_t lo = 0, hi = 0;
            if (__builtin_expect(encode_uhex128(f1.ptr, f1.len, lo, hi), 1)) {
                uk = 1ULL;
                cring.hex[slot] = {lo, hi};
                dict.prefetch_hline(hash_uhex128(lo, hi));
            } else if (__builtin_expect(is_mmap, 1)) {
                uk = 2ULL;
                stage_gen_user_ev_cold(f1.ptr, f1.len, seg_id, seg_base, cring.gen[slot], dict);
            } else {
                emit_clk_direct_gen_cold(f1.ptr, f1.len, cid, ts_camp44,
                                         dict, arenas, tid, clk_out, tl_cur, tl_end, g_cur, g_end);
                return;
            }
        }
        cring.core[slot] = {uk, cid, ts_camp44};
        if (__builtin_expect(cpushed >= RING_LAG, 1)) {
            emit_click_ring_entry(cring, (cpushed - RING_LAG) & RING_MASK,
                                  dict, arenas, tid, clk_out, tl_cur, tl_end, g_cur, g_end);
        }
        ++cpushed;
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
    bool use_long_ev32,
    PartitionedStore<RawEvent16, 256, EV_BUCKETS, 4>::ThreadState& ev_out,
    PartitionedStore<LongEv32, 128, EV_BUCKETS, 2>::ThreadState& long_ev_out,
    FastSideSlab& side_slab,
    EvRingState& ering,
    size_t& epushed,
    uint32_t& tl_cur, uint32_t& tl_end,
    uint32_t& g_cur,  uint32_t& g_end
) {
    static constexpr uint32_t VIEW_U32 = 0x77656976U; // "view" LE
    static constexpr uint32_t CART_U32 = 0x74726163U; // "cart" LE
    static constexpr uint64_t PUR_U64  = 0x6573616863727570ULL; // "purchase" LE

    const uint32_t seg_id = task.file->seg_id;
    const char* seg_base = task.file->map_base;
    const bool is_mmap = (seg_id < StringArenaPool::ARENA_SEG_BASE);

    scan_csv_rows<7>(task, rbuf, [&](auto row) SS_LAMBDA_INLINE {
        Slice f0 = row.template get<0>();
        if (__builtin_expect(f0.len == 0, 0)) return;

        // 1. Parse ingest_ts lazily (zero stack-escaping out-params!)
        Slice f3 = row.template get<3>();
        int64_t ing_rel_sec = 0;
        int64_t ing_ms = 0;
        bool ing_whole_sec = true;
        uint64_t ing_ym = 0;
        uint16_t ing_dd16 = 0;
        int64_t ing_day_rel = 0;
        if (__builtin_expect(f3.len == 20 && f3.ptr[19] == 'Z', 1)) {
            if (__builtin_expect(!lut.parse_ingest_f1_rel_sec_day(f3.ptr, ing_rel_sec, ing_ym, ing_dd16, ing_day_rel), 0)) return;
        } else {
            auto ir = parse_ingest_slow_ev(f3.ptr, f3.len, lut, win_start_sec);
            if (!ir.ok) return;
            ing_ms = ir.ing_ms;
            ing_rel_sec = ir.ing_rel_sec;
            ing_whole_sec = ir.whole_sec;
        }

        // 2. Encode event_id -> 48b bijection (short) OR direct 128b LongEv32 / general fallback (by-value return!)
        uint64_t eid48 = 0, eid_k64 = 0, rem39 = 0, elo = 0, ehi = 0;
        uint32_t eb = 0;
        bool is_long32 = false;
        bool eid_in_side = false;
        if (__builtin_expect(encode_eid48_padded(f0.ptr, f0.len, eid48), 1)) {
            uint64_t h48 = mix48(eid48);
            eb = uint32_t(h48 >> 39);
            rem39 = h48 & RawEvent16::REM39_MASK;
        } else {
            auto lr = encode_long_eid_slow(f0.ptr, f0.len, use_long_ev32, ing_whole_sec, seg_id, seg_base,
                                           arenas, tid, dict);
            is_long32 = lr.is_long32;
            elo = lr.elo; ehi = lr.ehi; eid_k64 = lr.k64; eb = lr.eb; rem39 = lr.rem39;
            eid_in_side = !is_long32;
        }

        uint32_t irel21 = 0U;
        if (__builtin_expect(ing_whole_sec, 1)) {
            int64_t ir = ing_rel_sec + RawEvent16::ING_BIAS;
            if (__builtin_expect(ir > 0 && ir <= RawEvent16::ING_MAX, 1)) irel21 = uint32_t(ir);
        }

        // 3. Lazy event_type (immediate constants, zero stack loads!) + zero-unpack non-empty check
        Slice f4 = row.template get<4>();
        uint64_t v8 = 0;
        std::memcpy(&v8, f4.ptr, 8);
        uint32_t v4 = uint32_t(v8);
        bool is_vc = (f4.len == 4) & ((v4 == VIEW_U32) | (v4 == CART_U32));
        bool is_pur = (f4.len == 8) & (v8 == PUR_U64);
        bool ok = row.template nonempty<1>() & row.template nonempty<5>() & (is_vc | is_pur);

        // 4. Format-selected fast F1 / F3 + by-value out-of-line F2 (zero stack-escaping out-params!)
        int64_t ts_rel_sec = 0;
        int64_t ts_ms = 0;
        uint32_t ts_ms_rem = 0;
        bool ts_from_sec = true;
        if (__builtin_expect(ok, 1)) {
            Slice f2 = row.template get<2>();
            bool ts_ok;
            if (__builtin_expect(f2.len == 20 && f2.ptr[19] == 'Z', 1)) {
                ts_ok = lut.parse_iso19_rel_sec_reuse(f2.ptr, ing_ym, ing_dd16, ing_day_rel, ts_rel_sec);
            } else if (__builtin_expect(f2.len == 13, 1)) {
                int64_t f3_sec = 0;
                ts_ok = parse_13d_sec_ms(f2.ptr, f3_sec, ts_ms_rem);
                ts_rel_sec = f3_sec - win_start_sec;
            } else {
                auto fr = parse_ts_f2_or_slow_rel(f2.ptr, f2.len, lut);
                ts_ok = fr.ok;
                ts_rel_sec = fr.rel_sec;
                ts_ms = fr.ms;
                ts_from_sec = fr.from_sec;
            }
            if (__builtin_expect(ts_ok & ts_from_sec & ing_whole_sec, 1)) {
                if (__builtin_expect(uint64_t(ts_rel_sec) >= uint64_t(win_dur_sec) ||
                                     ing_rel_sec - ts_rel_sec > 86400LL, 0)) {
                    ok = false;
                }
            } else {
                auto cr = check_ts_slow_ev(f2.ptr, f2.len, lut, ts_ok, ts_from_sec, ing_whole_sec,
                                           win_start_sec, win_start_ms, win_dur_ms,
                                           ts_rel_sec, ts_ms_rem, ing_rel_sec, ing_ms, ts_ms);
                ing_ms = cr.ing_ms;
                ts_ms = cr.ts_ms;
                ok = cr.ok;
                ts_from_sec = cr.from_sec;
            }
        }

        // 5. Cold out-of-line purchase validation (by-value return, zero stack-escaping out-params!)
        uint32_t price = 0, cat_id = 0;
        if (__builtin_expect(ok & is_pur, 0)) {
            Slice f6 = row.template get<6>();
            Slice f5 = row.template get<5>();
            auto pr = parse_purchase_cold(f6.ptr, f6.len, f5.ptr, f5.len, products);
            if (!pr.ok) {
                ok = false;
            } else {
                price = pr.price;
                cat_id = pr.cat_id;
            }
        }

        // 6. Cold out-of-line invalid row shortcut (routes LongEv32 invalid tombstones directly to long_ev_out!)
        if (__builtin_expect(!ok, 0)) {
            push_invalid_ev_cold(is_long32, elo, ehi, eid_in_side, irel21, eb, rem39, eid_k64,
                                 ing_whole_sec, win_start_sec, ing_rel_sec, ing_ms,
                                 tid, ev_out, long_ev_out, side_slab);
            return;
        }

        // 7. ALL VALID ROWS -> 24 B Core + Sparse Side/Long 24-Ahead Prefetch Ring!
        uint32_t ts_rel_ms = ts_from_sec
            ? uint32_t(ts_rel_sec * 1000LL + int64_t(ts_ms_rem))
            : uint32_t(ts_ms - win_start_ms);
        bool need_side = (!is_long32) & (is_pur | eid_in_side | (irel21 == 0));
        const size_t slot = epushed & RING_MASK;
        uint64_t base_q0 = (uint64_t(eb) << 7) | (uint64_t(need_side) << 2) |
                           (uint64_t(eid_in_side) << 3) | (uint64_t(is_long32) << 4);

        Slice f1 = row.template get<1>();
        uint64_t u48 = 0;
        uint64_t q0 = base_q0;
        if (__builtin_expect(encode_uid48_padded(f1.ptr, f1.len, u48), 1)) {
            q0 |= (u48 << 16) | EV_F_SHORT_U;
            dict.prefetch_uslot(mix48(u48));
        } else {
            uint64_t lo = 0, hi = 0;
            if (__builtin_expect(encode_uhex128(f1.ptr, f1.len, lo, hi), 1)) {
                q0 |= EV_F_HEX_U;
                ering.hex[slot] = {lo, hi};
                dict.prefetch_hline(hash_uhex128(lo, hi));
            } else if (__builtin_expect(is_mmap, 1)) {
                stage_gen_user_ev_cold(f1.ptr, f1.len, seg_id, seg_base, ering.gen[slot], dict);
            } else {
                if (need_side && ing_whole_sec && irel21 == 0) ing_ms = (win_start_sec + ing_rel_sec) * 1000LL;
                emit_ev_direct_gen_cold(f1.ptr, f1.len, eb, irel21, rem39, is_long32, need_side, eid_in_side,
                                        ts_rel_ms, elo, ehi, win_start_sec + ing_rel_sec,
                                        eid_k64, ing_ms, price, cat_id,
                                        dict, arenas, tid, ev_out, long_ev_out, side_slab, tl_cur, tl_end, g_cur, g_end);
                return;
            }
        }

        if (__builtin_expect(is_long32, 0)) {
            ering.long_ev[slot] = {elo, ehi, win_start_sec + ing_rel_sec, price, cat_id};
        } else if (__builtin_expect(need_side, 0)) {
            if (__builtin_expect(ing_whole_sec && irel21 == 0, 0)) {
                ing_ms = (win_start_sec + ing_rel_sec) * 1000LL;
            }
            ering.side[slot] = {eid_k64, ing_ms, price, cat_id};
        }

        EvRingCore& c = ering.core[slot];
        c.q0 = q0;
        c.q1 = (rem39 << 25) | (uint64_t(irel21) << 4);
        c.ts_rel_ms = ts_rel_ms;

        if (__builtin_expect(epushed >= RING_LAG, 1)) {
            emit_ev_ring_entry(ering, (epushed - RING_LAG) & RING_MASK,
                               dict, arenas, tid, ev_out, long_ev_out, side_slab, tl_cur, tl_end, g_cur, g_end);
        }
        ++epushed;
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
    PartitionedStore<LongEv32, 128, EV_BUCKETS, 2>& long_ev_store,
    const std::vector<FastSideSlab>& sides,
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
    using EvStore     = PartitionedStore<RawEvent16, 256, EV_BUCKETS, 4>;
    using LongEvStore = PartitionedStore<LongEv32, 128, EV_BUCKETS, 2>;
    using PEv8Store   = PartitionedStore<CompactPersonEv8, 512, PERSON_BUCKETS, 8>;
    using PurStore    = PartitionedStore<CompactPurchase, 64, PERSON_BUCKETS, 1>;
    static_assert(sizeof(EvStore::Block) == sizeof(PEv8Store::Block), "Block sizes must match for zero-fault recycling");
    static_assert(sizeof(LongEvStore::Block) == sizeof(PEv8Store::Block), "LongEv32 block size must match for recycling");

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

    // Stage 2C: ZERO-mix64 L2-Hot Short Dedup + L1D-Hot Direct LongEv32 Dedup + 4KB Block Recycling!
    using EvBlock = typename EvStore::Block;
    using LongBlock = typename LongEvStore::Block;
    std::atomic<int> next_eb{0};
    pool.run([&](int tid) {
        struct DedupSlot8 { uint32_t fp; uint32_t handle1; };
        std::vector<uint32_t> ht4;
        std::vector<DedupSlot8> ht8;
        std::vector<EvBlock*> blks;
        std::vector<LongBlock*> lblks;
        ht4.reserve(131072);
        blks.reserve(512);
        lblks.reserve(128);

        auto& out_ev = *person_ev_store.ts[size_t(tid)];
        auto& out_pur = *purchase_store.ts[size_t(tid)];

        while (true) {
            int b = next_eb.fetch_add(1, std::memory_order_relaxed);
            if (b >= EV_BUCKETS) break;
            size_t nb = event_store.bucket_count(b);
            size_t nlb = long_ev_store.bucket_count(b);
            if (nb == 0 && nlb == 0) continue;

            if (nb > 0) {
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
                                            const FastSideEvent& sa = sides[existing.w1 >> 26][existing.w1 & 0x03FFFFFFU];
                                            const FastSideEvent& sb = sides[ev.w1 >> 26][ev.w1 & 0x03FFFFFFU];
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
                                            const FastSideEvent& sa = sides[existing.w1 >> 26][existing.w1 & 0x03FFFFFFU];
                                            const FastSideEvent& sb = sides[ev.w1 >> 26][ev.w1 & 0x03FFFFFFU];
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

                // Loop 2 (short): ZERO LinkMap probes — 8B filtered NT copy + 8-ahead FastSideEvent prefetch!
                for (uint32_t bi = 0; bi < nblks; ++bi) {
                    EvBlock* blk = blks[bi];
                    const uint32_t cnt = blk->count;
                    const RawEvent16* items = blk->items;
                    for (uint32_t i = 0; i < cnt; ++i) {
                        if (__builtin_expect(i + 8 < cnt, 1)) {
                            const RawEvent16& ahead = items[i + 8];
                            if (__builtin_expect(ahead.is_side() && ahead.is_valid(), 0)) {
                                __builtin_prefetch(&sides[ahead.w1 >> 26][ahead.w1 & 0x03FFFFFFU], 0, 1);
                            }
                        }
                        const RawEvent16& ev = items[i];
                        if (__builtin_expect(!ev.is_valid(), 0)) continue;
                        if (__builtin_expect(!ev.is_side(), 1)) {
                            uint32_t pid = ev.inline_pid();
                            uint32_t ts_rel = ev.inline_ts_rel_ms();
                            out_ev.push(pid & uint32_t(PERSON_BUCKETS - 1), CompactPersonEv8::make(pid >> PERSON_SHIFT, ts_rel));
                        } else {
                            const FastSideEvent& se = sides[ev.w1 >> 26][ev.w1 & 0x03FFFFFFU];
                            uint32_t pid = se.pid;
                            uint32_t pb = pid & uint32_t(PERSON_BUCKETS - 1);
                            uint32_t pid_hi = pid >> PERSON_SHIFT;
                            uint32_t ts_rel = se.ts_rel_ms;
                            out_ev.push(pb, CompactPersonEv8::make(pid_hi, ts_rel));
                            if (__builtin_expect(se.price > 0, 0)) {
                                out_pur.push(pb, CompactPurchase{pid_hi, ts_rel, se.price, se.cat_id});
                            }
                        }
                    }
                    out_ev.recycle_block(blk);
                }
            }

            // Direct L1D-hot inline 128-bit LongEv32 dedup (~5K rows/bucket = 32 KB table, ZERO random side reads!)
            if (nlb > 0) {
                lblks.clear();
                for (int st = 0; st < pool.nthreads; ++st) {
                    for (auto* blk = long_ev_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                        lblks.push_back(blk);
                    }
                }
                size_t lcap = 64;
                while (lcap < nlb + (nlb >> 1) + 16) lcap <<= 1;
                size_t lmask = lcap - 1;
                const uint32_t nl = uint32_t(lblks.size());

                if (__builtin_expect(nl < 8191, 1)) {
                    ht4.assign(lcap, 0);
                    uint32_t* htp = ht4.data();
                    for (uint32_t bi = 0; bi < nl; ++bi) {
                        LongBlock* blk = lblks[bi];
                        const uint32_t cnt = blk->count;
                        LongEv32* items = blk->items;
                        for (uint32_t i = 0; i < cnt; ++i) {
                            LongEv32& ev = items[i];
                            uint64_t h = hash_uhex128(ev.k_lo, ev.k_hi);
                            uint32_t fp12 = uint32_t((h >> 40) & 0xFFFULL) << 20;
                            size_t slot = size_t(h) & lmask;
                            while (true) {
                                uint32_t entry = htp[slot];
                                if (__builtin_expect(entry == 0, 1)) {
                                    htp[slot] = fp12 | (((bi << 7) | i) + 1);
                                    break;
                                }
                                if (__builtin_expect((entry & 0xFFF00000U) == fp12, 0)) {
                                    uint32_t prev = (entry & 0x000FFFFFU) - 1;
                                    LongEv32& existing = lblks[prev >> 7]->items[prev & 127U];
                                    if (((existing.k_lo ^ ev.k_lo) | (existing.k_hi ^ ev.k_hi)) == 0) {
                                        if (ev.ing40() < existing.ing40()) existing = ev;
                                        ev.payload = 0;
                                        break;
                                    }
                                }
                                slot = (slot + 1) & lmask;
                            }
                        }
                    }
                } else {
                    ht8.assign(lcap, {0, 0});
                    for (uint32_t bi = 0; bi < nl; ++bi) {
                        LongBlock* blk = lblks[bi];
                        const uint32_t cnt = blk->count;
                        LongEv32* items = blk->items;
                        for (uint32_t i = 0; i < cnt; ++i) {
                            LongEv32& ev = items[i];
                            uint64_t h = hash_uhex128(ev.k_lo, ev.k_hi);
                            uint32_t fp = uint32_t(h >> 24);
                            size_t slot = size_t(h) & lmask;
                            while (true) {
                                uint32_t h1 = ht8[slot].handle1;
                                if (__builtin_expect(h1 == 0, 1)) {
                                    ht8[slot] = {fp, ((bi << 7) | i) + 1};
                                    break;
                                }
                                if (__builtin_expect(ht8[slot].fp == fp, 0)) {
                                    uint32_t prev = h1 - 1;
                                    LongEv32& existing = lblks[prev >> 7]->items[prev & 127U];
                                    if (((existing.k_lo ^ ev.k_lo) | (existing.k_hi ^ ev.k_hi)) == 0) {
                                        if (ev.ing40() < existing.ing40()) existing = ev;
                                        ev.payload = 0;
                                        break;
                                    }
                                }
                                slot = (slot + 1) & lmask;
                            }
                        }
                    }
                }

                for (uint32_t bi = 0; bi < nl; ++bi) {
                    LongBlock* blk = lblks[bi];
                    const uint32_t cnt = blk->count;
                    const LongEv32* items = blk->items;
                    for (uint32_t i = 0; i < cnt; ++i) {
                        const LongEv32& ev = items[i];
                        if (__builtin_expect(!ev.is_valid(), 0)) continue;
                        uint32_t pid = ev.pid();
                        uint32_t pb = pid & uint32_t(PERSON_BUCKETS - 1);
                        uint32_t pid_hi = pid >> PERSON_SHIFT;
                        uint32_t ts_rel = ev.ts_rel_ms();
                        out_ev.push(pb, CompactPersonEv8::make(pid_hi, ts_rel));
                        uint32_t pr = ev.price();
                        if (__builtin_expect(pr > 0, 0)) {
                            out_pur.push(pb, CompactPurchase{pid_hi, ts_rel, pr, ev.cat_id()});
                        }
                    }
                    out_ev.recycle_block(blk);
                }
            }
        }
        out_ev.flush_all();
        out_pur.flush_all();
    });
    double td1 = now_ms();

    // Stage 3 + Stage 4 FUSED: Weighted Bucket Scheduling (events + 2*clicks + 2*purchases) + Direct L1D Grouping!
    int bucket_order[PERSON_BUCKETS];
    size_t bucket_sizes[PERSON_BUCKETS];
    size_t bucket_weight[PERSON_BUCKETS];
    size_t total_surviving = 0;
    for (int b = 0; b < PERSON_BUCKETS; ++b) {
        bucket_order[b] = b;
        bucket_sizes[b] = person_ev_store.bucket_count(b);
        bucket_weight[b] = bucket_sizes[b] + 2 * click_store.bucket_count(b) + 2 * purchase_store.bucket_count(b);
        total_surviving += bucket_sizes[b];
    }
    std::sort(bucket_order, bucket_order + PERSON_BUCKETS, [&](int a, int b) {
        return bucket_weight[a] > bucket_weight[b];
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

                // Helper to emit person's canonical user_id ONCE (zero intermediate copy for external 33B hex IDs!)
                uint32_t full_pid = (p << PERSON_SHIFT) | uint32_t(b);
                uint64_t nm = dict.pid_name[full_pid];
                auto emit_this_person_uid = [&](uint32_t run_rows) SS_LAMBDA_INLINE {
                    if (__builtin_expect((nm & FALLBACK_BIT) == 0, 1)) {
                        alignas(16) char uid_pre_inline[16]{};
                        uint32_t ulen = __builtin_expect((nm >> 56) == 0, 1)
                            ? uint32_t(decode_uid48_inline(nm, uid_pre_inline + 4))
                            : uint32_t(decode_inline(nm, uid_pre_inline + 4));
                        std::memcpy(uid_pre_inline, &ulen, 4);
                        ud.emit_person_uid(uid_pre_inline, 4 + ulen, run_rows);
                    } else {
                        auto sv = arenas.get_fallback(nm);
                        ud.emit_person_uid_sv(sv.data(), sv.size(), run_rows);
                    }
                };

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
                    emit_this_person_uid(1U);
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

                    // Ultra-tight 4-byte uint32_t session loop + AVX2 8-wide unsigned gap scan!
                    uint32_t j = i + 1;
#if defined(__AVX2__)
                    if (__builtin_expect(j + 8 <= en, 0)) {
                        const __m256i v_sign   = _mm256_set1_epi32(int32_t(0x80000000U));
                        const __m256i v_thresh = _mm256_set1_epi32(int32_t(uint32_t(SESSION_GAP_MS) ^ 0x80000000U));
                        while (j + 8 <= en) {
                            __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(evs + j - 1));
                            __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(evs + j));
                            __m256i d  = _mm256_xor_si256(_mm256_sub_epi32(vb, va), v_sign);
                            uint32_t m = uint32_t(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(d, v_thresh))));
                            if (m != 0) {
                                j += uint32_t(__builtin_ctz(m));
                                goto sess_gap_done;
                            }
                            j += 8;
                        }
                    }
#endif
                    while (j < en && evs[j] - evs[j - 1] <= uint32_t(SESSION_GAP_MS)) {
                        ++j;
                    }
#if defined(__AVX2__)
                sess_gap_done:
#endif
                    uint32_t prev = evs[j - 1];
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
                emit_this_person_uid(uint32_t(ud.num_rows) - person_rows_start);
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
    bool force_all_pread = std::getenv("USE_PREAD") != nullptr;
    bool force_all_mmap = std::getenv("USE_MMAP") != nullptr;
    // Mixed I/O default: mmap small 24 MB links (zero-copy KeyRef) + L2-hot 128 KB pread for 2.06 GB events+clicks!
    // Eliminates 503K file PTE faults, 1,403 MADV_POPULATE_READ calls, and 503K serial exit_mmap PTE zaps!
    bool pread_events_clicks = force_all_pread || !force_all_mmap;
    bool force_wide = std::getenv("FORCE_WIDE") != nullptr;

    // Launch products + campaigns loader BEFORE ThreadPool pins main to CPU 0 (inherits full affinity mask!)
    Products products;
    Campaigns campaigns;
    struct ThreadJoiner {
        std::thread t;
        void join() { if (t.joinable()) t.join(); }
        ~ThreadJoiner() { join(); }
    } dims_thr{std::thread([&]() {
        products.load(data_dir + "/products.csv");
        campaigns.load(data_dir + "/campaigns.csv");
    })};

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
    if (pread_events_clicks) {
        for (int t = 0; t < ncpu; ++t) rbufs[size_t(t)].init();
    }

    std::vector<std::pair<std::string, uint8_t>> all_paths;
    all_paths.reserve(256);
    list_csv_paths(data_dir + "/links", 2, all_paths);
    list_csv_paths(data_dir + "/clicks", 0, all_paths);
    list_csv_paths(data_dir + "/events", 1, all_paths);

    const bool keep_fds = pread_events_clicks && (all_paths.size() < 480);
    std::vector<CsvFile> csv_files(all_paths.size());
    std::atomic<size_t> next_f{0};

    // Parallel open + selective mmap (links mmap by default; events+clicks pread with persistent fds!)
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
            const bool want_mmap = !force_all_pread && (force_all_mmap || all_paths[i].second == 2);
            if (want_mmap && fsz > 0) {
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
            uint32_t sid = mbase ? arenas.register_mmap_seg(mbase) : UINT32_MAX;
            csv_files[i] = {all_paths[i].first, fd, fsz, mbase, mlen_tot, sid, all_paths[i].second};
        }
    });

    size_t total_event_bytes = 0;
    bool need_readbufs = pread_events_clicks;
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

    UserDict dict;
    dict.build(link_tasks, rbufs, pool, arenas, total_event_bytes, manifest_compact_ok);
    dims_thr.join();
    double t_links = now_ms();

    bool use_fast_dict = manifest_compact_ok && dict.fast_dict_mode &&
                         (campaigns.organic_idx < 4096) &&
                         (num_days <= 65535) &&
                         (products.cat_names.size() <= 65535);

    bool write_ok = false;
    double t_scan = t_links;

    if (use_fast_dict) {
        const bool use_long_ev32 = (products.cat_names.size() <= 127);
        PartitionedStore<CompactClick, 128, PERSON_BUCKETS, 4> click_store;
        PartitionedStore<RawEvent16, 256, EV_BUCKETS, 4> event_store;
        PartitionedStore<LongEv32, 128, EV_BUCKETS, 2> long_ev_store;
        click_store.init(ncpu, 64ULL << 20);
        event_store.init(ncpu, 256ULL << 20);
        long_ev_store.init(ncpu, use_long_ev32 ? (32ULL << 20) : 0ULL);

        if (dict.recycled_dsu && dict.recycled_dsu_bytes >= (4ULL << 20)) {
            size_t slice = (dict.recycled_dsu_bytes / size_t(ncpu)) & ~size_t(4095);
            char* p = static_cast<char*>(dict.recycled_dsu);
            for (int t = 0; t < ncpu; ++t) {
                event_store.ts[size_t(t)]->donate_slab(p + size_t(t) * slice, slice);
            }
        }

        size_t side_reserve = use_long_ev32
            ? (16ULL << 20)
            : std::max<size_t>(48ULL << 20, std::min<size_t>(256ULL << 20, total_event_bytes / size_t(ncpu * 2)));
        std::vector<FastSideSlab> thread_sides(static_cast<size_t>(ncpu));
        for (int t = 0; t < ncpu; ++t) thread_sides[size_t(t)].init(side_reserve);

        if (!io_tasks.empty()) {
            std::atomic<size_t> next_t{0};
            pool.run([&](int tid) {
                auto& clk_out = *click_store.ts[size_t(tid)];
                auto& ev_out = *event_store.ts[size_t(tid)];
                auto& long_ev_out = *long_ev_store.ts[size_t(tid)];
                auto& side_slab = thread_sides[size_t(tid)];
                auto& rbuf = rbufs[size_t(tid)];

                ClickRingState cring;
                EvRingState    ering;
                size_t cpushed = 0, epushed = 0;
                uint32_t tl_cur = 0, tl_end = 0;
                uint32_t g_cur = 0, g_end = 0;

                auto drain_clicks = [&]() SS_COLD_NOINLINE {
                    if (cpushed > 0) {
                        size_t s = (cpushed > RING_LAG) ? (cpushed - RING_LAG) : 0;
                        for (size_t k = s; k < cpushed; ++k) {
                            emit_click_ring_entry(cring, k & RING_MASK, dict, arenas, tid, clk_out, tl_cur, tl_end, g_cur, g_end);
                        }
                        cpushed = 0;
                    }
                };
                auto drain_events = [&]() SS_COLD_NOINLINE {
                    if (epushed > 0) {
                        size_t s = (epushed > RING_LAG) ? (epushed - RING_LAG) : 0;
                        for (size_t k = s; k < epushed; ++k) {
                            emit_ev_ring_entry(ering, k & RING_MASK, dict, arenas, tid, ev_out, long_ev_out, side_slab, tl_cur, tl_end, g_cur, g_end);
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
                                                click_start_ms, win_end_ms, clk_out, cring, cpushed, tl_cur, tl_end, g_cur, g_end);
                    } else {
                        process_event_task_fast(task, rbuf, tid, products, lut, dict, arenas,
                                                win_start_ms, win_start_sec, win_dur_ms, win_dur_sec, use_long_ev32,
                                                ev_out, long_ev_out, side_slab, ering, epushed, tl_cur, tl_end, g_cur, g_end);
                    }
                }
                drain_clicks();
                drain_events();
                clk_out.flush_all();
                ev_out.flush_all();
                long_ev_out.flush_all();
            });
        }
        t_scan = now_ms();

        if (__builtin_expect(!dict.dict_overflow.load(std::memory_order_acquire), 1)) {
            write_ok = run_fast_dict_dedup_sessionize_and_write(
                event_store, long_ev_store, thread_sides, click_store, dict, products, campaigns, arenas, pool,
                win_start_ms, win_end_ms, win_start_sec, win_start_day, num_days, ud_dir, cd_dir, prof);
        } else {
            // Ultra-rare >16.7M distinct users or fractional F3 ingest on hex EID -> rebuild classic & run wide fallback!
            std::fprintf(stderr, "UserDict/LongEv fallback triggered; running wide mode\n");
            use_fast_dict = false;
        }
    }

    if (!use_fast_dict) {
        if (dict.fast_dict_mode) {
            dict.fast_dict_mode = false;
            dict.build(link_tasks, rbufs, pool, arenas, total_event_bytes, false);
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
