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
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif
#if defined(__GLIBC__)
#include <malloc.h>
#endif

extern "C" [[noreturn]] void ss_on_exit(int code);

namespace SS_ENGINE_NS {

// 256-bit zero-ZMM uint32_t buffer zeroer (prevents glibc __memset_avx512_unaligned_erms License-2 downclock!)
static SS_ALWAYS_INLINE void zero_u32_no_zmm(uint32_t* __restrict p, size_t n) {
#if defined(__AVX2__)
    const __m256i vz = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p + i),      vz);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p + i + 8),  vz);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p + i + 16), vz);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p + i + 24), vz);
    }
    for (; i + 8 <= n; i += 8) _mm256_storeu_si256(reinterpret_cast<__m256i*>(p + i), vz);
    for (; i < n; ++i) p[i] = 0;
#else
    std::memset(p, 0, n * sizeof(uint32_t));
#endif
}

// ============================================================================
// Physical-Core-First Topology Detection + Cache-Line-Aligned True-Sleep ThreadPool
// ============================================================================
struct CpuTopology {
    int ncpu = 1;
    int nphys = 1;
    bool can_pin = false;
    std::vector<int> cpus; // ordered round-robin by physical core: [0..nphys-1] are distinct physical cores!
};

static CpuTopology detect_topology() {
    CpuTopology topo;
    int n = int(std::thread::hardware_concurrency());
    if (n < 1) n = 1;
    std::vector<int> raw_cpus;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (::sched_getaffinity(0, sizeof(set), &set) == 0) {
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (CPU_ISSET(c, &set)) raw_cpus.push_back(c);
        }
        int aff = int(raw_cpus.size());
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
    // Group allowed CPUs by physical core via thread_siblings_list -> interleave round-robin!
    if (!raw_cpus.empty()) {
        std::vector<std::pair<int, std::vector<int>>> groups;
        for (int c : raw_cpus) {
            int key = c;
            char path[128];
            std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
            if (FILE* sf = std::fopen(path, "r")) {
                int first_sib = c;
                if (std::fscanf(sf, "%d", &first_sib) == 1 && first_sib >= 0) key = first_sib;
                std::fclose(sf);
            }
            bool found = false;
            for (auto& g : groups) {
                if (g.first == key) { g.second.push_back(c); found = true; break; }
            }
            if (!found) groups.push_back({key, {c}});
        }
        size_t max_d = 0;
        for (const auto& g : groups) max_d = std::max(max_d, g.second.size());
        for (size_t d = 0; d < max_d; ++d) {
            for (const auto& g : groups) {
                if (d < g.second.size()) topo.cpus.push_back(g.second[d]);
            }
        }
        topo.nphys = std::max(1, int(groups.size()));
        topo.can_pin = (raw_cpus.size() == size_t(n));
    }
#endif
    if (const char* e = std::getenv("NCPU")) {
        int v = std::atoi(e);
        if (v > 0) n = v;
    }
    if (n < 1) n = 1;
    if (n > 64) n = 64;
    topo.ncpu = n;
    if (topo.nphys > n || topo.cpus.empty()) topo.nphys = std::max(1, std::min(n, topo.nphys));
    if (topo.cpus.empty()) topo.nphys = n;
    return topo;
}

static void pin_current_thread(int tid, const CpuTopology& topo) {
#if defined(__linux__)
    if (topo.can_pin && size_t(tid) < topo.cpus.size()) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(topo.cpus[size_t(tid)], &set);
        ::sched_setaffinity(0, sizeof(set), &set);
    }
#else
    (void)tid; (void)topo;
#endif
}

class ThreadPool {
public:
    int nthreads = 1;
    int nphys = 1;

    explicit ThreadPool(const CpuTopology& topo)
        : nthreads(topo.ncpu < 1 ? 1 : (topo.ncpu > 64 ? 64 : topo.ncpu)),
          nphys(std::max(1, std::min(nthreads, topo.nphys))),
          topology(topo) {
        pin_current_thread(0, topology);
        workers.reserve(static_cast<size_t>(nthreads - 1));
        for (int t = 1; t < nthreads; ++t) {
            workers.emplace_back([this, t]() {
                pin_current_thread(t, topology);
                worker_loop(t);
            });
        }
    }

    template <typename Fn>
    void run_n(int active_n, Fn&& fn) {
        int active = std::max(1, std::min(nthreads, active_n));
        if (active == 1) {
            fn(0);
            return;
        }
        using FnT = std::remove_reference_t<Fn>;
        current_fn_ptr = &fn;
        current_invoke = [](void* p, int t) { (*static_cast<FnT*>(p))(t); };
        active_limit.store(active, std::memory_order_relaxed);
        done_count.store(0, std::memory_order_relaxed);
        gen.fetch_add(1, std::memory_order_seq_cst);
        if (__builtin_expect(sleeping.load(std::memory_order_seq_cst) > 0, 0)) {
            std::lock_guard<std::mutex> lk(mu);
            cv.notify_all();
        }
        fn(0);
        done_count.fetch_add(1, std::memory_order_acq_rel);
        while (done_count.load(std::memory_order_acquire) < active) {
            #if defined(__x86_64__)
            __builtin_ia32_pause();
            #elif defined(__aarch64__)
            asm volatile("yield");
            #endif
        }
        current_fn_ptr = nullptr;
        current_invoke = nullptr;
    }

    template <typename Fn>
    void run(Fn&& fn) {
        run_n(nthreads, std::forward<Fn>(fn));
    }

private:
    CpuTopology topology;
    std::vector<std::thread> workers;
    std::mutex mu;
    std::condition_variable cv;
    alignas(64) std::atomic<uint32_t> gen{0};
    alignas(64) std::atomic<int> done_count{0};
    alignas(64) std::atomic<int> sleeping{0};
    alignas(64) std::atomic<int> active_limit{1};
    void (*current_invoke)(void*, int) = nullptr;
    void* current_fn_ptr = nullptr;

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
            if (tid < active_limit.load(std::memory_order_relaxed)) {
                current_invoke(current_fn_ptr, tid);
                done_count.fetch_add(1, std::memory_order_acq_rel);
            } else {
                // Immediately sleep (futex -> HLT) so sibling HT gives 100% ROB/LB/SB/L2 to physical core!
                sleeping.fetch_add(1, std::memory_order_seq_cst);
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&]() { return gen.load(std::memory_order_acquire) >= my_gen + 1; });
                sleeping.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    }
};

// Parallel 2MB-aligned page pre-population (kills THP fault storms & makes PREFETCHW 100% effective from iter 0!)
static void populate_write_parallel_ranges(
    ThreadPool& pool, const std::pair<void*, size_t>* ranges, size_t nranges
) {
    constexpr size_t HP = 2ULL << 20;
    size_t total_chunks = 0;
    for (size_t r = 0; r < nranges; ++r) {
        if (ranges[r].first && ranges[r].second >= (4ULL << 20)) {
            total_chunks += ranges[r].second / HP;
        }
    }
    if (total_chunks == 0) return;
    int nt = std::min<int>(pool.nphys, int(total_chunks));
    pool.run_n(nt, [&](int tid) {
        size_t cs = (total_chunks * size_t(tid)) / size_t(nt);
        size_t ce = (total_chunks * size_t(tid + 1)) / size_t(nt);
        if (ce <= cs) return;
        size_t off_c = 0;
        for (size_t r = 0; r < nranges; ++r) {
            if (!ranges[r].first || ranges[r].second < (4ULL << 20)) continue;
            size_t rc = ranges[r].second / HP;
            size_t lo = std::max(cs, off_c);
            size_t hi = std::min(ce, off_c + rc);
            if (hi > lo) {
                char* p = static_cast<char*>(ranges[r].first) + (lo - off_c) * HP;
                size_t len = (hi - lo) * HP;
#if defined(__linux__) && defined(MADV_POPULATE_WRITE)
                if (::madvise(p, len, MADV_POPULATE_WRITE) == 0) { off_c += rc; continue; }
#endif
                volatile char* vp = p;
                for (size_t off = 0; off < len; off += 4096) vp[off] = 0;
            }
            off_c += rc;
        }
    });
}

static void populate_write_parallel(ThreadPool& pool, void* ptr, size_t bytes) {
    std::pair<void*, size_t> r{ptr, bytes};
    populate_write_parallel_ranges(pool, &r, 1);
}

// ============================================================================
// Direct 4KB-Aligned Huge-Page Slab Partitioned Store (ZERO wc_buf, ZERO wc_cnt, ZERO c==WC_CAP mispredicts!)
//   - Block is alignas(4096) & sizeof(Block)==4096 -> end-of-block check is 1-cycle (uintptr_t(p) & 4095) == 0!
//   - Also catches initial nullptr automatically! 0.4% taken (1/252), 8KB/thread fits L1D even at 1024 buckets!
//   - FIFO 4KB Block Recycling (free_head/free_tail) for zero-fault CompactPersonEv8 reuse!
// ============================================================================
template <typename T, uint32_t CAP, int NUM_B, uint32_t /*UNUSED_WC*/ = 0>
struct PartitionedStore {
    struct alignas(4096) Block {
        Block* next;
        uint32_t count;
        uint32_t _pad;
        alignas(64) T items[CAP];
    };
    static_assert(sizeof(Block) == 4096, "Block must be exactly 4096 bytes for 1-cycle (p & 4095) boundary check");
    struct FreeNode { FreeNode* next; };
    static constexpr size_t SLAB_BYTES = 16ULL << 20;

    struct alignas(64) ThreadState {
        T* blk_cur[NUM_B]{};
        Block* heads[NUM_B]{};
        size_t counts[NUM_B]{};
        FreeNode* free_head = nullptr;
        FreeNode* free_tail = nullptr;
        std::vector<std::pair<void*, size_t>> slabs;
        char* slab = nullptr;
        size_t slab_pos = SLAB_BYTES;
        size_t slab_cap = SLAB_BYTES;
        char* pre_slab = nullptr;
        size_t pre_slab_cap = 0;

        // LIFO (stack) recycling: reuse the L1D/L2-hot block just read 200ns ago (ZERO DRAM RFO!)
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
            constexpr size_t bsz = sizeof(Block); // exact 4096
            if (__builtin_expect(slab_pos + bsz > slab_cap, 0)) {
                if (__builtin_expect(pre_slab != nullptr && pre_slab_cap >= bsz, 1)) {
                    slab = pre_slab;
                    slab_pos = 0;
                    slab_cap = pre_slab_cap;
                    pre_slab = nullptr;
                    pre_slab_cap = 0;
                } else {
                    slab = static_cast<char*>(alloc_mmap(SLAB_BYTES, false));
                    slabs.emplace_back(slab, alloc_mmap_actual_size(SLAB_BYTES, false));
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

        SS_COLD_NOINLINE T* alloc_bucket_block(uint32_t b) {
            Block* prev = heads[b];
            if (prev) {
                prev->count = CAP;
                counts[b] += CAP;
            }
            Block* nb = alloc_block();
            nb->next = prev;
            heads[b] = nb;
            return nb->items;
        }

        void donate_slab(void* mem, size_t bytes) {
            if (!mem) return;
            uintptr_t raw = reinterpret_cast<uintptr_t>(mem);
            uintptr_t aligned = (raw + 4095U) & ~uintptr_t(4095U);
            size_t delta = size_t(aligned - raw);
            if (bytes <= delta) return;
            size_t avail = (bytes - delta) & ~size_t(4095U);
            if (avail < sizeof(Block) * 4) return;
            if (slab != nullptr && slab_pos == 0 && slab_cap > avail && pre_slab == nullptr) {
                pre_slab = slab;
                pre_slab_cap = slab_cap;
            }
            slab = reinterpret_cast<char*>(aligned);
            slab_pos = 0;
            slab_cap = avail;
        }

        SS_ALWAYS_INLINE void push(uint32_t bucket, const T& val) {
            T* p = blk_cur[bucket];
            if (__builtin_expect((uintptr_t(p) & 4095U) == 0, 0)) {
                p = alloc_bucket_block(bucket);
            }
            *p = val;
            blk_cur[bucket] = p + 1;
        }

        // Branchless conditional push (keep01 in {0, 1}): eliminates duplicate/invalid mispredicts in Stage 2!
        SS_ALWAYS_INLINE void push_cond(uint32_t bucket, const T& val, uint64_t keep01) {
            T* p = blk_cur[bucket];
            if (__builtin_expect((uintptr_t(p) & 4095U) == 0, 0)) {
                p = alloc_bucket_block(bucket);
            }
            *p = val;
            blk_cur[bucket] = p + keep01;
        }

        void flush_all() {
            for (uint32_t b = 0; b < NUM_B; ++b) {
                if (heads[b] && blk_cur[b]) {
                    uint32_t c = uint32_t(blk_cur[b] - heads[b]->items);
                    heads[b]->count = c;
                    counts[b] += c;
                    blk_cur[b] = nullptr;
                }
            }
        }
    };

    int nthreads = 1;
    std::vector<ThreadState*> ts;

    void init(int nt, size_t pre_reserve_bytes = 0) {
        nthreads = nt;
        ts.resize(static_cast<size_t>(nt));
        constexpr bool hp = false; // 4KB pcp lock-free allocs avoid MADV_HUGEPAGE synchronous compaction stalls!
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
    static constexpr size_t READ_CHUNK = 131072; // 128 KiB: 25% of Cascade Lake L2 for 2 HTs + low seccomp syscalls!
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

// Unified single-base SepBuf with seps[-1] == 0xFFFF sentinel & 64-entry slack for 8B POS8 stores!
struct alignas(64) SepBuf {
    uint16_t pad0[31];
    uint16_t seps_sentinel; // seps[-1] == 0xFFFFU!
    uint16_t seps[4160];    // 4096 + 64 slack (exact 64B multiple; max speculative write <= 4091)
    uint64_t nl_bits[70];
    uint64_t q_bits[70];
    uint64_t any_bad;
};

// 2 KB (32 L1D lines) zero-overflow byte-index LUT for x86_64 vpmovzxbw, plus packed LUT for ARM64/scalar.
#if defined(__SSE4_1__) || defined(__AVX2__)
struct alignas(64) Pos8U8Lut {
    uint8_t e[256][8];
    constexpr Pos8U8Lut() : e{} {
        for (uint32_t m = 0; m < 256U; ++m) {
            uint32_t k = 0;
            for (uint32_t b = 0; b < 8U; ++b) {
                if ((m >> b) & 1U) e[m][k++] = uint8_t(b);
            }
        }
    }
};
static constexpr Pos8U8Lut POS8_U8_LUT{};
#else
struct alignas(64) Pos8CntLut {
    uint64_t e[256];
    constexpr Pos8CntLut() : e{} {
        for (uint32_t m = 0; m < 256U; ++m) {
            uint64_t packed = 0;
            uint32_t cnt = 0;
            for (uint32_t bit = 0; bit < 8U; ++bit) {
                if ((m >> bit) & 1U) {
                    if (cnt < 4U) packed |= uint64_t(bit) << (cnt * 16U);
                    ++cnt;
                }
            }
            uint64_t tag = (cnt <= 4U) ? uint64_t(cnt) : 8ULL;
            e[m] = packed | (tag << 60);
        }
    }
};
static constexpr Pos8CntLut POS8_CNT_LUT{};
#endif

// Port-balanced zero-overflow vpmovzxbw POS8_U8 + parallel popcnt + lazy nl/q flush (ZERO %zmm!)
template <bool DO_PREFETCH>
static SS_NOINLINE uint32_t build_seps_4kb(
    const char* __restrict bptr,
    size_t nblks,
    SepBuf* __restrict buf
) {
    buf->seps_sentinel = 0xFFFFU;
    uint64_t any_bad = 0, s_carry = 1, rq_carry = 0, inq = 0;
    uint64_t nl_acc = 0, q_acc = 0;
    uint32_t nsep = 0, widx = 0, bpos = 0;
#if !defined(__SSE4_1__) && !defined(__AVX2__)
    static constexpr uint64_t MASK60 = 0x0FFFFFFFFFFFFFFFULL;
    static constexpr uint64_t STEP8  = 0x0008000800080008ULL;
    uint64_t base_bcast = 0;
#else
    const __m128i v_s8 = _mm_set1_epi16(8);
#endif

    for (size_t bi = 0; bi < nblks; ++bi) {
        if constexpr (DO_PREFETCH) {
            if ((bi & 7U) == 0) __builtin_prefetch(bptr + (bi << 6) + 2048, 0, 1);
        }
        Vec64 v = Vec64::load(bptr + (bi << 6));
        uint64_t nm = v.eq('\n');      // Port 5 (AVX-512VL %k + kunpckdq)
        uint64_t cm = v.eq_p01(',');   // Port 0/1 (AVX2 %ymm + vpmovmskb)
        uint64_t qm = v.eq('"');       // Port 5 (AVX-512VL %k + kunpckdq)

        uint64_t S  = cm | nm;
        uint64_t FS = (S << 1) | s_carry;
        uint64_t RQ = qm & ~FS;
        uint64_t CQ = (RQ << 1) | rq_carry;
        uint64_t raw_px = prefix_xor(qm);
        uint64_t PX = raw_px ^ (0ULL - inq);
        any_bad |= (CQ & ~S) | (PX & S);
        s_carry  = S >> 63;
        rq_carry = RQ >> 63;
        inq     ^= (raw_px >> 63);

        uint64_t nlm = pext_u64_fast(nm, S);
        uint64_t qms = pext_u64_fast(CQ, S);

#if defined(__SSE4_1__) || defined(__AVX2__)
        uint32_t S_lo  = uint32_t(S), S_hi = uint32_t(S >> 32);
        uint32_t c_lo  = uint32_t(__builtin_popcount(S_lo));
        uint32_t c_tot = uint32_t(__builtin_popcountll(S));
        uint32_t b0 = S_lo & 0xFFU, b1 = (S_lo >> 8) & 0xFFU, b2 = (S_lo >> 16) & 0xFFU, b3 = S_lo >> 24;
        uint32_t b4 = S_hi & 0xFFU, b5 = (S_hi >> 8) & 0xFFU, b6 = (S_hi >> 16) & 0xFFU, b7 = S_hi >> 24;
        uint32_t c0  = uint32_t(__builtin_popcount(b0));
        uint32_t s01 = uint32_t(__builtin_popcount(S_lo & 0xFFFFU));
        uint32_t c2  = uint32_t(__builtin_popcount(b2));
        uint32_t c4  = uint32_t(__builtin_popcount(b4));
        uint32_t s45 = uint32_t(__builtin_popcount(S_hi & 0xFFFFU));
        uint32_t c6  = uint32_t(__builtin_popcount(b6));

        uint16_t* const sp_lo = buf->seps + nsep;
        uint16_t* const sp_hi = sp_lo + c_lo;
        __m128i vb = _mm_set1_epi16(int16_t(bi << 6));
        auto emit8 = [&](uint16_t* dst, uint32_t b) SS_LAMBDA_INLINE {
            __m128i off = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(POS8_U8_LUT.e[b])));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), _mm_add_epi16(off, vb));
            vb = _mm_add_epi16(vb, v_s8);
        };
        emit8(sp_lo,            b0); emit8(sp_lo + c0,       b1);
        emit8(sp_lo + s01,      b2); emit8(sp_lo + s01 + c2, b3);
        emit8(sp_hi,            b4); emit8(sp_hi + c4,       b5);
        emit8(sp_hi + s45,      b6); emit8(sp_hi + s45 + c6, b7);
#else
        uint16_t* const sp = buf->seps + nsep;
        uint32_t S_lo = uint32_t(S);
        uint32_t S_hi = uint32_t(S >> 32);

        uint64_t e0 = POS8_CNT_LUT.e[S_lo & 0xFFU];
        uint32_t c0 = uint32_t(e0 >> 60);
        uint64_t p0 = (e0 & MASK60) + base_bcast;
        std::memcpy(sp, &p0, 8);
        base_bcast += STEP8;

        uint64_t e1 = POS8_CNT_LUT.e[(S_lo >> 8) & 0xFFU];
        uint32_t c1 = uint32_t(e1 >> 60);
        uint64_t p1 = (e1 & MASK60) + base_bcast;
        std::memcpy(sp + c0, &p1, 8);
        base_bcast += STEP8;
        uint32_t s01 = c0 + c1;
        uint32_t ov  = c0 | c1;

        uint64_t e2 = POS8_CNT_LUT.e[(S_lo >> 16) & 0xFFU];
        uint32_t c2 = uint32_t(e2 >> 60);
        uint64_t p2 = (e2 & MASK60) + base_bcast;
        std::memcpy(sp + s01, &p2, 8);
        base_bcast += STEP8;

        uint64_t e3 = POS8_CNT_LUT.e[S_lo >> 24];
        uint32_t c3 = uint32_t(e3 >> 60);
        uint64_t p3 = (e3 & MASK60) + base_bcast;
        std::memcpy(sp + s01 + c2, &p3, 8);
        base_bcast += STEP8;
        uint32_t c_lo = s01 + (c2 + c3);
        ov |= c2 | c3;

        uint16_t* const sp_hi = sp + c_lo;
        uint64_t e4 = POS8_CNT_LUT.e[S_hi & 0xFFU];
        uint32_t c4 = uint32_t(e4 >> 60);
        uint64_t p4 = (e4 & MASK60) + base_bcast;
        std::memcpy(sp_hi, &p4, 8);
        base_bcast += STEP8;

        uint64_t e5 = POS8_CNT_LUT.e[(S_hi >> 8) & 0xFFU];
        uint32_t c5 = uint32_t(e5 >> 60);
        uint64_t p5 = (e5 & MASK60) + base_bcast;
        std::memcpy(sp_hi + c4, &p5, 8);
        base_bcast += STEP8;
        uint32_t s45 = c4 + c5;
        ov |= c4 | c5;

        uint64_t e6 = POS8_CNT_LUT.e[(S_hi >> 16) & 0xFFU];
        uint32_t c6 = uint32_t(e6 >> 60);
        uint64_t p6 = (e6 & MASK60) + base_bcast;
        std::memcpy(sp_hi + s45, &p6, 8);
        base_bcast += STEP8;

        uint64_t e7 = POS8_CNT_LUT.e[S_hi >> 24];
        uint32_t c7 = uint32_t(e7 >> 60);
        uint64_t p7 = (e7 & MASK60) + base_bcast;
        std::memcpy(sp_hi + s45 + c6, &p7, 8);
        base_bcast += STEP8;
        uint32_t c_hi = s45 + (c6 + c7);
        ov |= c6 | c7;

        uint32_t c_tot = c_lo + c_hi;
        if (__builtin_expect((ov & 8U) != 0U, 0)) {
            uint32_t base_off = uint32_t(bi << 6);
            uint64_t t = S;
            uint32_t k = 0;
            while (t) {
                sp[k++] = uint16_t(base_off + uint32_t(__builtin_ctzll(t)));
                t &= t - 1;
            }
            c_tot = k;
        }
#endif

        // Lazy nl_bits/q_bits flush: ONLY stores on 64-bit word rollover (~8.4% of blocks!)
        uint32_t nbpos = bpos + c_tot;
        nl_acc |= (nlm << bpos);
        q_acc  |= (qms << bpos);
        if (__builtin_expect(nbpos >= 64U, 0)) {
            buf->nl_bits[widx] = nl_acc;
            buf->q_bits[widx]  = q_acc;
            ++widx;
            uint32_t rsh = 63U - bpos;
            nl_acc = (nlm >> 1) >> rsh;
            q_acc  = (qms >> 1) >> rsh;
            nbpos -= 64U;
        }
        bpos = nbpos;
        nsep += c_tot;
    }

    buf->nl_bits[widx] = nl_acc; buf->nl_bits[widx + 1] = 0;
    buf->q_bits[widx]  = q_acc;  buf->q_bits[widx + 1]  = 0;
    buf->any_bad = any_bad;
    return nsep;
}

// Whole-window O(5-word) periodic nl_bits verifier (eliminates per-row nl_bytes load & ls>=lrel check on 99.5% of rows!)
template <int N>
static SS_ALWAYS_INLINE bool verify_periodic_nl_bits(const uint64_t* nl_bits, uint32_t nsep, uint32_t n_used) {
    static constexpr uint64_t PAT7[7] = {
        0x4081020408102040ULL, 0x2040810204081020ULL, 0x1020408102040810ULL,
        0x0810204081020408ULL, 0x0408102040810204ULL, 0x0204081020408102ULL,
        0x8102040810204081ULL
    };
    uint32_t full_w = n_used >> 6;
    uint32_t rem    = n_used & 63U;
    uint32_t last_w = nsep >> 6;
    uint64_t diff = 0;
    if constexpr (N == 7) {
        uint32_t p = 0;
        for (uint32_t w = 0; w < full_w; ++w) {
            diff |= nl_bits[w] ^ PAT7[p];
            if (++p == 7U) p = 0;
        }
        uint32_t wi = full_w;
        if (rem > 0) {
            diff |= nl_bits[wi] ^ (PAT7[p] & ((1ULL << rem) - 1ULL));
            ++wi;
        }
        while (wi <= last_w) { diff |= nl_bits[wi]; ++wi; }
    } else if constexpr (N == 4) {
        constexpr uint64_t PAT4 = 0x8888888888888888ULL;
        for (uint32_t w = 0; w < full_w; ++w) diff |= nl_bits[w] ^ PAT4;
        uint32_t wi = full_w;
        if (rem > 0) {
            diff |= nl_bits[wi] ^ (PAT4 & ((1ULL << rem) - 1ULL));
            ++wi;
        }
        while (wi <= last_w) { diff |= nl_bits[wi]; ++wi; }
    } else if constexpr (N == 2) {
        constexpr uint64_t PAT2 = 0xAAAAAAAAAAAAAAAAULL;
        for (uint32_t w = 0; w < full_w; ++w) diff |= nl_bits[w] ^ PAT2;
        uint32_t wi = full_w;
        if (rem > 0) {
            diff |= nl_bits[wi] ^ (PAT2 & ((1ULL << rem) - 1ULL));
            ++wi;
        }
        while (wi <= last_w) { diff |= nl_bits[wi]; ++wi; }
    } else {
        return false;
    }
    return diff == 0;
}

// Zero-quote-load BMI2 PEXT separator-indexed CSV scanner with periodic window callback & row fallback!
template <int N, typename OnWindow, typename OnRow, typename OnBatchEnd>
static SS_NOINLINE void scan_mmap_csv_core(
    const CsvTask& task, OnWindow&& on_window, OnRow&& on_row, OnBatchEnd&& on_batch_end
) {
    static constexpr uint32_t ROW_MASK = (1U << N) - 1U;
    static constexpr uint32_t ROW_WANT = 1U << (N - 1);

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

    SepBuf sbuf;
    bool done = false;

    auto on_cold = [&](const Slice* fs) SS_COLD_NOINLINE {
        on_row(SliceCsvRow{fs});
    };

    while (cur < limit) {
        size_t avail = fsz - cur;
        size_t nblks = std::min<size_t>(64, avail >> 6);
        if (__builtin_expect(nblks == 0, 0)) break;

        const char* bptr = base + cur;
        uint32_t nsep = build_seps_4kb<true>(bptr, nblks, &sbuf);
        const uint16_t* seps = sbuf.seps;
        const uint64_t* nl_bits = sbuf.nl_bits;
        const uint64_t* q_bits  = sbuf.q_bits;

        size_t ls = 0;
        const size_t lrel = limit - cur;
        if (__builtin_expect(sbuf.any_bad == 0, 1)) {
            const uint8_t* q_bytes = reinterpret_cast<const uint8_t*>(q_bits);
            uint32_t n_rows = nsep / uint32_t(N);
            uint32_t n_used = n_rows * uint32_t(N);
            if (__builtin_expect(n_rows > 0 && lrel >= (nblks << 6) &&
                                 verify_periodic_nl_bits<N>(nl_bits, nsep, n_used), 1)) {
                if constexpr (!std::is_same_v<std::decay_t<OnWindow>, std::nullptr_t>) {
                    on_window(bptr, seps, q_bytes, n_rows);
                } else {
                    for (uint32_t si = 0; si < n_used; si += uint32_t(N)) {
                        uint16_t qb16;
                        std::memcpy(&qb16, q_bytes + (si >> 3), 2);
                        uint32_t qm = uint32_t(qb16) >> (si & 7U);
                        on_row(SepsCsvRow<N>{bptr, seps + si, qm});
                    }
                }
                ls = size_t(seps[n_used - 1]) + 1;
            } else {
                uint32_t si = 0;
                const uint8_t* nl_bytes = reinterpret_cast<const uint8_t*>(nl_bits);
                while (si + uint32_t(N) <= nsep) {
                    uint16_t nb16;
                    std::memcpy(&nb16, nl_bytes + (si >> 3), 2);
                    uint32_t sh = si & 7U;
                    if (__builtin_expect(((nb16 >> sh) & ROW_MASK) == ROW_WANT, 1)) {
                        uint16_t qb16;
                        std::memcpy(&qb16, q_bytes + (si >> 3), 2);
                        uint32_t qm = uint32_t(qb16) >> sh;
                        on_row(SepsCsvRow<N>{bptr, seps + si, qm});
                        ls = size_t(seps[si + N - 1]) + 1;
                        si += uint32_t(N);
                        if (__builtin_expect(ls >= lrel, 0)) {
                            cur += ls;
                            done = true;
                            break;
                        }
                    } else {
                        while (si < nsep && !((nl_bits[si >> 6] >> (si & 63)) & 1ULL)) ++si;
                        if (si >= nsep) break;
                        ls = size_t(seps[si]) + 1;
                        ++si;
                        if (__builtin_expect(ls >= lrel, 0)) {
                            cur += ls;
                            done = true;
                            break;
                        }
                    }
                }
                while (!done && si < nsep) {
                    while (si < nsep && !((nl_bits[si >> 6] >> (si & 63)) & 1ULL)) ++si;
                    if (si >= nsep) break;
                    ls = size_t(seps[si]) + 1;
                    ++si;
                    if (__builtin_expect(ls >= lrel, 0)) {
                        cur += ls;
                        done = true;
                        break;
                    }
                }
            }
        } else {
            Slice fs[N];
            for (uint32_t si = 0; si < nsep; ++si) {
                if ((nl_bits[si >> 6] >> (si & 63)) & 1ULL) {
                    size_t nl_rel = seps[si];
                    if (parse_csv_line_scalar<N>(bptr + ls, bptr + nl_rel, fs)) on_cold(fs);
                    ls = nl_rel + 1;
                    if (__builtin_expect(ls >= lrel, 0)) {
                        cur += ls;
                        done = true;
                        break;
                    }
                }
            }
        }
        on_batch_end();
        if (done) break;
        if (__builtin_expect(ls == 0, 0)) {
            const char* nl = find_nl_simd(bptr + (nblks << 6), base + fsz);
            if (!nl) break;
            Slice fs[N];
            if (parse_csv_line_scalar<N>(bptr, nl, fs)) on_cold(fs);
            on_batch_end();
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
    on_batch_end();
}

template <int N, typename OnWindow, typename OnRow, typename OnBatchEnd>
static SS_NOINLINE void scan_pread_csv_core(
    const CsvTask& task, ReadBuf& rbuf, OnWindow&& on_window, OnRow&& on_row, OnBatchEnd&& on_batch_end
) {
    static constexpr uint32_t ROW_MASK = (1U << N) - 1U;
    static constexpr uint32_t ROW_WANT = 1U << (N - 1);

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

    SepBuf sbuf;

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
            uint32_t nsep = build_seps_4kb<false>(bptr, nblks, &sbuf);
            const uint16_t* seps = sbuf.seps;
            const uint64_t* nl_bits = sbuf.nl_bits;
            const uint64_t* q_bits  = sbuf.q_bits;

            size_t ls = 0;
            const size_t lrel = limit_in_buf - cur;
            if (__builtin_expect(sbuf.any_bad == 0, 1)) {
                const uint8_t* q_bytes = reinterpret_cast<const uint8_t*>(q_bits);
                uint32_t n_rows = nsep / uint32_t(N);
                uint32_t n_used = n_rows * uint32_t(N);
                if (__builtin_expect(n_rows > 0 && lrel >= (nblks << 6) &&
                                     verify_periodic_nl_bits<N>(nl_bits, nsep, n_used), 1)) {
                    if constexpr (!std::is_same_v<std::decay_t<OnWindow>, std::nullptr_t>) {
                        on_window(bptr, seps, q_bytes, n_rows);
                    } else {
                        for (uint32_t si = 0; si < n_used; si += uint32_t(N)) {
                            uint16_t qb16;
                            std::memcpy(&qb16, q_bytes + (si >> 3), 2);
                            uint32_t qm = uint32_t(qb16) >> (si & 7U);
                            on_row(SepsCsvRow<N>{bptr, seps + si, qm});
                        }
                    }
                    ls = size_t(seps[n_used - 1]) + 1;
                } else {
                    uint32_t si = 0;
                    const uint8_t* nl_bytes = reinterpret_cast<const uint8_t*>(nl_bits);
                    while (si + uint32_t(N) <= nsep) {
                        uint16_t nb16;
                        std::memcpy(&nb16, nl_bytes + (si >> 3), 2);
                        uint32_t sh = si & 7U;
                        if (__builtin_expect(((nb16 >> sh) & ROW_MASK) == ROW_WANT, 1)) {
                            uint16_t qb16;
                            std::memcpy(&qb16, q_bytes + (si >> 3), 2);
                            uint32_t qm = uint32_t(qb16) >> sh;
                            on_row(SepsCsvRow<N>{bptr, seps + si, qm});
                            ls = size_t(seps[si + N - 1]) + 1;
                            si += uint32_t(N);
                            if (__builtin_expect(ls >= lrel, 0)) {
                                cur += ls;
                                done = true;
                                break;
                            }
                        } else {
                            while (si < nsep && !((nl_bits[si >> 6] >> (si & 63)) & 1ULL)) ++si;
                            if (si >= nsep) break;
                            ls = size_t(seps[si]) + 1;
                            ++si;
                            if (__builtin_expect(ls >= lrel, 0)) {
                                cur += ls;
                                done = true;
                                break;
                            }
                        }
                    }
                    while (!done && si < nsep) {
                        while (si < nsep && !((nl_bits[si >> 6] >> (si & 63)) & 1ULL)) ++si;
                        if (si >= nsep) break;
                        ls = size_t(seps[si]) + 1;
                        ++si;
                        if (__builtin_expect(ls >= lrel, 0)) {
                            cur += ls;
                            done = true;
                            break;
                        }
                    }
                }
            } else {
                Slice fs[N];
                for (uint32_t si = 0; si < nsep; ++si) {
                    if ((nl_bits[si >> 6] >> (si & 63)) & 1ULL) {
                        size_t nl_rel = seps[si];
                        if (parse_csv_line_scalar<N>(bptr + ls, bptr + nl_rel, fs)) on_cold(fs);
                        ls = nl_rel + 1;
                        if (__builtin_expect(ls >= lrel, 0)) {
                            cur += ls;
                            done = true;
                            break;
                        }
                    }
                }
            }
            on_batch_end();
            if (done) break;
            if (__builtin_expect(ls == 0, 0)) {
                const char* nl = find_nl_simd(bptr + (nblks << 6), rbuf.data + total);
                if (!nl) break;
                Slice fs[N];
                if (parse_csv_line_scalar<N>(bptr, nl, fs)) on_cold(fs);
                on_batch_end();
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
        on_batch_end();

        if (buf_file_off + cur >= limit) break;

        if (file_pos >= fsz) {
            if (cur < total && buf_file_off + cur < limit) {
                Slice fs[N];
                if (parse_csv_line_scalar<N>(rbuf.data + cur, rbuf.data + total, fs)) on_cold(fs);
                on_batch_end();
            }
            break;
        }

        carry = total - cur;
        if (carry > 0) std::memmove(rbuf.data, rbuf.data + cur, carry);
        if (file_pos >= limit) hit_limit = true;
    }

    on_batch_end();
    if (close_after) ::close(fd);
}

template <int N, typename OnWindow, typename OnRow, typename OnBatchEnd>
static SS_ALWAYS_INLINE void scan_csv_windows(
    const CsvTask& task, ReadBuf& rbuf, OnWindow&& on_window, OnRow&& on_row, OnBatchEnd&& on_batch_end
) {
    if (__builtin_expect(task.file->map_base == nullptr, 1)) {
        scan_pread_csv_core<N>(task, rbuf, std::forward<OnWindow>(on_window),
                               std::forward<OnRow>(on_row), std::forward<OnBatchEnd>(on_batch_end));
    } else {
        scan_mmap_csv_core<N>(task, std::forward<OnWindow>(on_window),
                              std::forward<OnRow>(on_row), std::forward<OnBatchEnd>(on_batch_end));
    }
}

template <int N, typename OnRow, typename OnBatchEnd>
static SS_ALWAYS_INLINE void scan_csv_rows(const CsvTask& task, ReadBuf& rbuf, OnRow&& on_row, OnBatchEnd&& on_batch_end) {
    scan_csv_windows<N>(task, rbuf, nullptr, std::forward<OnRow>(on_row), std::forward<OnBatchEnd>(on_batch_end));
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

    SS_ALWAYS_INLINE uint32_t find_inline(const char* s, size_t len) const {
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

static SS_ALWAYS_INLINE bool parse_csv2_fast(const char* p, const char* le, Slice out[2]) {
    if (__builtin_expect(le > p && le[-1] == '\r', 0)) --le;
    const char* comma = nullptr;
    for (const char* q = p; q < le; ++q) {
        char c = *q;
        if (__builtin_expect(c == '"', 0)) return parse_csv_line_scalar<2>(p, le, out);
        if (c == ',') {
            if (__builtin_expect(comma != nullptr, 0)) return false;
            comma = q;
        }
    }
    if (__builtin_expect(comma == nullptr, 0)) return false;
    out[0] = {p, size_t(comma - p)};
    out[1] = {comma + 1, size_t(le - comma - 1)};
    return true;
}

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
        ids.clear();
        channels.clear();
        std::fill(std::begin(fast_c), std::end(fast_c), NF16);
        if (!data.empty()) {
            const char* end = data.data() + data.size();
            const char* nl = find_nl_simd(data.data(), end);
            if (nl) {
                ids.reserve(data.size() / 16 + 16);
                channels.reserve(data.size() / 16 + 16);
                const char* p = nl + 1;
                Slice f[2];
                while (p < end) {
                    const char* eol = static_cast<const char*>(std::memchr(p, '\n', size_t(end - p)));
                    const char* le = eol ? eol : end;
                    if (parse_csv2_fast(p, le, f)) {
                        ids.emplace_back(f[0].ptr, f[0].len);
                        channels.emplace_back(f[1].ptr, f[1].len);
                    }
                    p = eol ? (eol + 1) : end;
                }
            }
        }
        map.init(ids.size() + 1);
        for (uint32_t i = 0; i < uint32_t(ids.size()); ++i) {
            const std::string& s = ids[i];
            map.insert(s, i);
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

    // Leaf fast-path campaign lookup for (len - 3) <= 3: 1 = found, 0 = definitely absent, -1 = fallback to map.find()
    SS_ALWAYS_INLINE int try_lookup_fast_c(const char* s, size_t len, uint32_t& out_idx) const {
        if (__builtin_expect(((len - 3) <= 3) & (s[0] == 'c') & (s[1] == '_'), 1)) {
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
                    if (__builtin_expect(r != NF16, 1)) { out_idx = uint32_t(r); return 1; }
                    if (__builtin_expect(organic_idx <= NF16, 1)) return 0;
                }
            }
        }
        return -1;
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
        std::vector<std::pair<std::string, uint32_t>> cats;
        cats.reserve(64);
        cats.emplace_back("unknown", 0U);
        FastStringIdMap seen_cats;
        seen_cats.init(64);
        seen_cats.insert("unknown", 0U);
        size_t seen_cap_half = 32;

        use_fast8 = true;
        std::memset(fast_p8, 0, FAST_CAP); // raw_id 0 == "unknown"
        std::vector<std::pair<std::string, uint32_t>> slow_rows;

        const char* data_start = nullptr;
        const char* data_end = data.data() + data.size();
        if (!data.empty()) {
            const char* nl = find_nl_simd(data.data(), data_end);
            if (nl) data_start = nl + 1;
        }

        if (data_start) {
            const char* p = data_start;
            Slice f[2];
            while (p < data_end) {
                const char* eol = static_cast<const char*>(std::memchr(p, '\n', size_t(data_end - p)));
                const char* le = eol ? eol : data_end;
                if (__builtin_expect(parse_csv2_fast(p, le, f), 1)) {
                    uint32_t raw_cid = seen_cats.find_inline(f[1].ptr, f[1].len);
                    if (__builtin_expect(raw_cid == FastStringIdMap::NOT_FOUND, 0)) {
                        raw_cid = uint32_t(cats.size());
                        cats.emplace_back(std::string(f[1].ptr, f[1].len), raw_cid);
                        if (raw_cid >= 256U) use_fast8 = false;
                        if (cats.size() >= seen_cap_half) {
                            seen_cats.init(cats.size() * 2);
                            for (const auto& kv : cats) seen_cats.insert(kv.first, kv.second);
                            seen_cap_half = (seen_cats.mask + 1) >> 1;
                        } else {
                            seen_cats.insert({f[1].ptr, f[1].len}, raw_cid);
                        }
                    }
                    if (__builtin_expect(use_fast8, 1)) {
                        bool is_fast = false;
                        if (__builtin_expect((f[0].len - 2) <= 4 && f[0].ptr[0] == 'p' &&
                                             f[0].ptr[1] >= '1' && f[0].ptr[1] <= '9', 1)) {
                            uint32_t n = uint8_t(f[0].ptr[1]) - uint8_t('0');
                            bool ok = true;
                            for (size_t k = 2; k < f[0].len; ++k) {
                                unsigned d = uint8_t(f[0].ptr[k]) - uint8_t('0');
                                if (__builtin_expect(d > 9, 0)) { ok = false; break; }
                                n = n * 10U + d;
                            }
                            if (__builtin_expect(ok && n < FAST_CAP, 1)) {
                                fast_p8[n] = uint8_t(raw_cid);
                                is_fast = true;
                            }
                        }
                        if (__builtin_expect(!is_fast, 0)) {
                            slow_rows.emplace_back(std::string(f[0].ptr, f[0].len), raw_cid);
                        }
                    }
                }
                p = eol ? (eol + 1) : data_end;
            }
        }

        std::sort(cats.begin(), cats.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        cat_names.clear();
        cat_names.reserve(cats.size());
        cat_pre.resize(cats.size());
        max_cat_len = 8;
        std::vector<uint32_t> remap(cats.size(), 0);
        for (size_t i = 0; i < cats.size(); ++i) {
            const std::string& c = cats[i].first;
            remap[cats[i].second] = uint32_t(i);
            if (cats[i].second == 0U) unknown_cat_id = uint32_t(i);
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

        if (__builtin_expect(use_fast8, 1)) {
            uint8_t remap8[256];
            for (size_t i = 0; i < cats.size(); ++i) remap8[i] = uint8_t(remap[i]);
            for (uint32_t i = 0; i < FAST_CAP; ++i) fast_p8[i] = remap8[fast_p8[i]];
            map.init(slow_rows.size() + 1);
            for (const auto& sr : slow_rows) map.insert(sr.first, remap[sr.second]);
        } else {
            size_t est_rows = data.size() / 16 + 16;
            map.init(est_rows);
            if (data_start) {
                const char* p = data_start;
                Slice f[2];
                while (p < data_end) {
                    const char* eol = static_cast<const char*>(std::memchr(p, '\n', size_t(data_end - p)));
                    const char* le = eol ? eol : data_end;
                    if (parse_csv2_fast(p, le, f)) {
                        uint32_t raw_cid = seen_cats.find_inline(f[1].ptr, f[1].len);
                        map.insert({f[0].ptr, f[0].len}, remap[raw_cid]);
                    }
                    p = eol ? (eol + 1) : data_end;
                }
            }
        }
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

    // 8-way cache-line-associative uid48 table (K >= 14 => 16,384 lines = 1 MiB VA min):
    //   slot[w] = [tag34 = h48>>14 : 34][disp6 (1..63) : 6][pid24 : 24]  (0ULL = empty)
    //   100% lossless bijection (34 + >=14 = >=48 bits), 97-99% 1st-line hit, ZERO cache-line splits!
    static constexpr size_t   MIN_ULINES = 1ULL << 14;
    static constexpr uint32_t MAX_UDISP  = 63U;

    struct alignas(64) Uline {
        uint64_t s[8];
    };
    static_assert(sizeof(Uline) == 64, "Uline must be exactly 64 bytes");

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
    Uline* ulines = nullptr;
    size_t lcap = 0;
    size_t lmask = 0;
    size_t ulines_bytes = 0;

    // Inline 128-bit HexDict (THP-backed 64B cache-line table)
    HexLine* hlines = nullptr;
    size_t hcap = 0;
    size_t hmask = 0;
    size_t hlines_bytes = 0;

    // Lock-free general/fallback user dictionary (rare non-hex long, UTF-8, uppercase & d>63 overflows!)
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

    SS_ALWAYS_INLINE void prefetch_h48(uint64_t h48) const {
        __builtin_prefetch(&ulines[size_t(h48) & lmask], 0, 3);
    }

    SS_ALWAYS_INLINE void prefetch_h48_l2(uint64_t h48) const {
        __builtin_prefetch(&ulines[size_t(h48) & lmask], 0, 2);
    }

    SS_ALWAYS_INLINE void prefetch_uslot(uint64_t h48) const {
        prefetch_h48(h48);
    }

    SS_ALWAYS_INLINE void prefetch_uslot_l2(uint64_t h48) const {
        prefetch_h48_l2(h48);
    }

    SS_ALWAYS_INLINE void prefetch_hline(uint64_t h) const {
        __builtin_prefetch(&hlines[size_t(h) & hmask], 0, 3);
    }

    SS_ALWAYS_INLINE void prefetch_hline_l2(uint64_t h) const {
        __builtin_prefetch(&hlines[size_t(h) & hmask], 0, 2);
    }

    SS_ALWAYS_INLINE void prefetch_gslot(uint64_t h) const {
        __builtin_prefetch(&gslots[size_t(h) & gmask], 0, 1);
    }

    // 256-bit AVX-512VL / AVX2 / NEON / scalar 8-way Uline tag+disp matcher (ZERO %zmm!)
    static SS_ALWAYS_INLINE uint32_t uline_match_mask(const Uline& line, uint64_t want40) {
#if defined(__AVX512VL__)
        __m256i vw = _mm256_set1_epi64x(static_cast<long long>(want40));
        __m256i v0 = _mm256_srli_epi64(_mm256_load_si256(reinterpret_cast<const __m256i*>(line.s)), 24);
        __m256i v1 = _mm256_srli_epi64(_mm256_load_si256(reinterpret_cast<const __m256i*>(line.s + 4)), 24);
        __mmask8 k0 = _mm256_cmpeq_epi64_mask(v0, vw);
        __mmask8 k1 = _mm256_cmpeq_epi64_mask(v1, vw);
        return uint32_t(k0) | (uint32_t(k1) << 4);
#elif defined(__AVX2__)
        __m256i vw = _mm256_set1_epi64x(static_cast<long long>(want40));
        __m256i v0 = _mm256_srli_epi64(_mm256_load_si256(reinterpret_cast<const __m256i*>(line.s)), 24);
        __m256i v1 = _mm256_srli_epi64(_mm256_load_si256(reinterpret_cast<const __m256i*>(line.s + 4)), 24);
        uint32_t m0 = uint32_t(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(v0, vw))));
        uint32_t m1 = uint32_t(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(v1, vw))));
        return m0 | (m1 << 4);
#elif defined(__aarch64__)
        uint64x2_t vw = vdupq_n_u64(want40);
        uint64x2_t c0 = vceqq_u64(vshrq_n_u64(vld1q_u64(line.s + 0), 24), vw);
        uint64x2_t c1 = vceqq_u64(vshrq_n_u64(vld1q_u64(line.s + 2), 24), vw);
        uint64x2_t c2 = vceqq_u64(vshrq_n_u64(vld1q_u64(line.s + 4), 24), vw);
        uint64x2_t c3 = vceqq_u64(vshrq_n_u64(vld1q_u64(line.s + 6), 24), vw);
        uint32x4_t m01 = vuzp1q_u32(vreinterpretq_u32_u64(c0), vreinterpretq_u32_u64(c1));
        uint32x4_t m23 = vuzp1q_u32(vreinterpretq_u32_u64(c2), vreinterpretq_u32_u64(c3));
        uint16x8_t m8  = vuzp1q_u16(vreinterpretq_u16_u32(m01), vreinterpretq_u16_u32(m23));
        const uint16x8_t bits = {1, 2, 4, 8, 16, 32, 64, 128};
        return uint32_t(vaddvq_u16(vandq_u16(m8, bits)));
#else
        uint32_t mask = 0;
        for (uint32_t w = 0; w < 8; ++w) {
            uint64_t sl = std::atomic_ref<uint64_t>(const_cast<uint64_t&>(line.s[w])).load(std::memory_order_relaxed);
            mask |= uint32_t((sl >> 24) == want40) << w;
        }
        return mask;
#endif
    }

    // Single-snapshot 8-way match + empty masks for get_or_insert_h48_slow ((sl>>24)==0 <=> sl==0 because d>=1!)
    static SS_ALWAYS_INLINE void uline_probe_masks(
        const Uline& line, uint64_t want40, uint32_t& out_match, uint32_t& out_empty
    ) {
#if defined(__AVX512VL__)
        __m256i vw = _mm256_set1_epi64x(static_cast<long long>(want40));
        __m256i vz = _mm256_setzero_si256();
        __m256i v0 = _mm256_srli_epi64(_mm256_load_si256(reinterpret_cast<const __m256i*>(line.s)), 24);
        __m256i v1 = _mm256_srli_epi64(_mm256_load_si256(reinterpret_cast<const __m256i*>(line.s + 4)), 24);
        out_match = uint32_t(_mm256_cmpeq_epi64_mask(v0, vw)) | (uint32_t(_mm256_cmpeq_epi64_mask(v1, vw)) << 4);
        out_empty = uint32_t(_mm256_cmpeq_epi64_mask(v0, vz)) | (uint32_t(_mm256_cmpeq_epi64_mask(v1, vz)) << 4);
#elif defined(__AVX2__)
        __m256i vw = _mm256_set1_epi64x(static_cast<long long>(want40));
        __m256i vz = _mm256_setzero_si256();
        __m256i v0 = _mm256_srli_epi64(_mm256_load_si256(reinterpret_cast<const __m256i*>(line.s)), 24);
        __m256i v1 = _mm256_srli_epi64(_mm256_load_si256(reinterpret_cast<const __m256i*>(line.s + 4)), 24);
        uint32_t m0 = uint32_t(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(v0, vw))));
        uint32_t m1 = uint32_t(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(v1, vw))));
        uint32_t e0 = uint32_t(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(v0, vz))));
        uint32_t e1 = uint32_t(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(v1, vz))));
        out_match = m0 | (m1 << 4);
        out_empty = e0 | (e1 << 4);
#elif defined(__aarch64__)
        uint64x2_t vw = vdupq_n_u64(want40);
        uint64x2_t s0 = vshrq_n_u64(vld1q_u64(line.s + 0), 24);
        uint64x2_t s1 = vshrq_n_u64(vld1q_u64(line.s + 2), 24);
        uint64x2_t s2 = vshrq_n_u64(vld1q_u64(line.s + 4), 24);
        uint64x2_t s3 = vshrq_n_u64(vld1q_u64(line.s + 6), 24);
        const uint16x8_t bits = {1, 2, 4, 8, 16, 32, 64, 128};
        uint32x4_t m01 = vuzp1q_u32(vreinterpretq_u32_u64(vceqq_u64(s0, vw)), vreinterpretq_u32_u64(vceqq_u64(s1, vw)));
        uint32x4_t m23 = vuzp1q_u32(vreinterpretq_u32_u64(vceqq_u64(s2, vw)), vreinterpretq_u32_u64(vceqq_u64(s3, vw)));
        out_match = uint32_t(vaddvq_u16(vandq_u16(vuzp1q_u16(vreinterpretq_u16_u32(m01), vreinterpretq_u16_u32(m23)), bits)));
        uint32x4_t e01 = vuzp1q_u32(vreinterpretq_u32_u64(vceqzq_u64(s0)), vreinterpretq_u32_u64(vceqzq_u64(s1)));
        uint32x4_t e23 = vuzp1q_u32(vreinterpretq_u32_u64(vceqzq_u64(s2)), vreinterpretq_u32_u64(vceqzq_u64(s3)));
        out_empty = uint32_t(vaddvq_u16(vandq_u16(vuzp1q_u16(vreinterpretq_u16_u32(e01), vreinterpretq_u16_u32(e23)), bits)));
#else
        uint32_t mm = 0, em = 0;
        for (uint32_t w = 0; w < 8; ++w) {
            uint64_t hi = std::atomic_ref<uint64_t>(const_cast<uint64_t&>(line.s[w])).load(std::memory_order_relaxed) >> 24;
            mm |= uint32_t(hi == want40) << w;
            em |= uint32_t(hi == 0) << w;
        }
        out_match = mm;
        out_empty = em;
#endif
    }

    // 2-insn AVX-512VL empty-slot finder for Pass 3 unique linked-key publish!
    static SS_ALWAYS_INLINE uint32_t uline_empty_mask(const Uline& line) {
#if defined(__AVX512VL__)
        __m256i vz = _mm256_setzero_si256();
        __m256i v0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(line.s));
        __m256i v1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(line.s + 4));
        return uint32_t(_mm256_cmpeq_epi64_mask(v0, vz)) | (uint32_t(_mm256_cmpeq_epi64_mask(v1, vz)) << 4);
#elif defined(__AVX2__)
        __m256i vz = _mm256_setzero_si256();
        __m256i v0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(line.s));
        __m256i v1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(line.s + 4));
        uint32_t e0 = uint32_t(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(v0, vz))));
        uint32_t e1 = uint32_t(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(v1, vz))));
        return e0 | (e1 << 4);
#elif defined(__aarch64__)
        uint64x2_t e0 = vceqzq_u64(vld1q_u64(line.s + 0));
        uint64x2_t e1 = vceqzq_u64(vld1q_u64(line.s + 2));
        uint64x2_t e2 = vceqzq_u64(vld1q_u64(line.s + 4));
        uint64x2_t e3 = vceqzq_u64(vld1q_u64(line.s + 6));
        uint32x4_t e01 = vuzp1q_u32(vreinterpretq_u32_u64(e0), vreinterpretq_u32_u64(e1));
        uint32x4_t e23 = vuzp1q_u32(vreinterpretq_u32_u64(e2), vreinterpretq_u32_u64(e3));
        const uint16x8_t bits = {1, 2, 4, 8, 16, 32, 64, 128};
        return uint32_t(vaddvq_u16(vandq_u16(vuzp1q_u16(vreinterpretq_u16_u32(e01), vreinterpretq_u16_u32(e23)), bits)));
#else
        uint32_t em = 0;
        for (uint32_t w = 0; w < 8; ++w) {
            uint64_t sl = std::atomic_ref<uint64_t>(const_cast<uint64_t&>(line.s[w])).load(std::memory_order_relaxed);
            em |= uint32_t(sl == 0) << w;
        }
        return em;
#endif
    }

    // 4-uop AVX-512VL vpxorq/vptestnmq (ZERO vpsrlq!) + inline d=2 for 99.9% hit at 16 MiB L3-resident ulines!
    static SS_ALWAYS_INLINE uint32_t uline_match_mask_hi(const Uline& line, uint64_t want_hi) {
#if defined(__AVX512VL__)
        const __m256i vw    = _mm256_set1_epi64x(static_cast<long long>(want_hi));
        const __m256i vmask = _mm256_set1_epi64x(static_cast<long long>(0xFFFFFFFFFF000000ULL));
        __m256i x0 = _mm256_xor_si256(_mm256_load_si256(reinterpret_cast<const __m256i*>(line.s)),     vw);
        __m256i x1 = _mm256_xor_si256(_mm256_load_si256(reinterpret_cast<const __m256i*>(line.s + 4)), vw);
        return uint32_t(_mm256_testn_epi64_mask(x0, vmask))
             | (uint32_t(_mm256_testn_epi64_mask(x1, vmask)) << 4);
#elif defined(__AVX2__)
        const __m256i vw    = _mm256_set1_epi64x(static_cast<long long>(want_hi));
        const __m256i vmask = _mm256_set1_epi64x(static_cast<long long>(0xFFFFFFFFFF000000ULL));
        const __m256i vz    = _mm256_setzero_si256();
        __m256i x0 = _mm256_and_si256(_mm256_xor_si256(_mm256_load_si256(reinterpret_cast<const __m256i*>(line.s)),     vw), vmask);
        __m256i x1 = _mm256_and_si256(_mm256_xor_si256(_mm256_load_si256(reinterpret_cast<const __m256i*>(line.s + 4)), vw), vmask);
        return uint32_t(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(x0, vz))))
             | (uint32_t(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(x1, vz)))) << 4);
#else
        return uline_match_mask(line, want_hi >> 24);
#endif
    }

    static SS_ALWAYS_INLINE bool try_lookup_h48_fast(
        const Uline* __restrict lines_p, size_t lmask_v, uint64_t h48, uint32_t& out_pid
    ) {
        const size_t li = size_t(h48) & lmask_v;
        const Uline& line = lines_p[li];
        const uint64_t want_hi1 = ((h48 << 16) & 0xFFFFFFFFC0000000ULL) | (1ULL << 24);
        uint32_t mask = uline_match_mask_hi(line, want_hi1);
        if (__builtin_expect(mask != 0, 1)) {
            out_pid = uint32_t(line.s[__builtin_ctz(mask)] & 0xFFFFFFULL);
            return true;
        }
        const Uline& line2 = lines_p[(li + 1) & lmask_v];
        uint32_t mask2 = uline_match_mask_hi(line2, want_hi1 + (1ULL << 24));
        if (__builtin_expect(mask2 != 0, 1)) {
            out_pid = uint32_t(line2.s[__builtin_ctz(mask2)] & 0xFFFFFFULL);
            return true;
        }
        return false;
    }

    static SS_ALWAYS_INLINE uint32_t hexline_match_mask(const HexLine& line, uint64_t lo, uint64_t hi) {
        uint32_t ready = (std::atomic_ref<uint32_t>(const_cast<uint32_t&>(line.ctrl)).load(std::memory_order_acquire) >> 8) & 7U;
#if defined(__AVX512VL__)
        __m128i t128 = _mm_set_epi64x(static_cast<long long>(hi), static_cast<long long>(lo));
        __m256i t256 = _mm256_inserti128_si256(_mm256_castsi128_si256(t128), t128, 1);
        __m256i v01  = _mm256_load_si256(reinterpret_cast<const __m256i*>(line.k));
        __m128i v2   = _mm_load_si128(reinterpret_cast<const __m128i*>(line.k[2]));
        uint32_t m01 = uint32_t(_mm256_cmpeq_epi64_mask(v01, t256));
        uint32_t m2  = uint32_t(_mm_cmpeq_epi64_mask(v2, t128));
        uint32_t m6  = m01 | (m2 << 4);
#if defined(__BMI2__)
        return ready & uint32_t(_pext_u32(m6 & (m6 >> 1), 0x15U));
#else
        return ready & (uint32_t((m01 & 3U) == 3U) | (uint32_t((m01 & 12U) == 12U) << 1) | (uint32_t(m2 == 3U) << 2));
#endif
#elif defined(__AVX2__)
        __m128i t128 = _mm_set_epi64x(static_cast<long long>(hi), static_cast<long long>(lo));
        __m256i t256 = _mm256_inserti128_si256(_mm256_castsi128_si256(t128), t128, 1);
        __m256i v01  = _mm256_load_si256(reinterpret_cast<const __m256i*>(line.k));
        __m128i v2   = _mm_load_si128(reinterpret_cast<const __m128i*>(line.k[2]));
        uint32_t m01 = uint32_t(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(v01, t256))));
        uint32_t m2  = uint32_t(_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi64(v2, t128))));
        uint32_t m6  = m01 | (m2 << 4);
#if defined(__BMI2__)
        return ready & uint32_t(_pext_u32(m6 & (m6 >> 1), 0x15U));
#else
        return ready & (uint32_t((m01 & 3U) == 3U) | (uint32_t((m01 & 12U) == 12U) << 1) | (uint32_t(m2 == 3U) << 2));
#endif
#else
        return ready & (
            uint32_t(((line.k[0][0] ^ lo) | (line.k[0][1] ^ hi)) == 0) |
            (uint32_t(((line.k[1][0] ^ lo) | (line.k[1][1] ^ hi)) == 0) << 1) |
            (uint32_t(((line.k[2][0] ^ lo) | (line.k[2][1] ^ hi)) == 0) << 2)
        );
#endif
    }

    // Branchless 256b+128b SIMD 3-way HexLine lookup with inline d=2 (>97% hit at 8 MiB!)
    static SS_ALWAYS_INLINE bool try_lookup_hex_fast(
        const HexLine* __restrict lines_p, size_t mask_v, uint64_t lo, uint64_t hi, uint64_t h, uint32_t& out_pid
    ) {
        const size_t li = size_t(h) & mask_v;
        const HexLine& line = lines_p[li];
        uint32_t hit = hexline_match_mask(line, lo, hi);
        if (__builtin_expect(hit != 0, 1)) {
            out_pid = line.pid[__builtin_ctz(hit)];
            return true;
        }
        const HexLine& line2 = lines_p[(li + 1) & mask_v];
        uint32_t hit2 = hexline_match_mask(line2, lo, hi);
        if (__builtin_expect(hit2 != 0, 1)) {
            out_pid = line2.pid[__builtin_ctz(hit2)];
            return true;
        }
        return false;
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
                    std::atomic_ref<uint64_t>(line.k[e][0]).store(lo, std::memory_order_relaxed);
                    std::atomic_ref<uint64_t>(line.k[e][1]).store(hi, std::memory_order_relaxed);
                    std::atomic_ref<uint32_t>(line.pid[e]).store(cid, std::memory_order_relaxed);
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
                    std::atomic_ref<uint64_t>(line.k[e][0]).store(lo, std::memory_order_relaxed);
                    std::atomic_ref<uint64_t>(line.k[e][1]).store(hi, std::memory_order_relaxed);
                    std::atomic_ref<uint32_t>(line.pid[e]).store(np, std::memory_order_relaxed);
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

    SS_COLD_NOINLINE uint32_t get_or_insert_h48_overflow(
        uint64_t h48,
        StringArenaPool& arenas, int tid,
        uint32_t& tl_cur, uint32_t& tl_end,
        uint32_t& g_cur, uint32_t& g_end
    ) {
        char tmp[16]{};
        size_t ulen = decode_uid48_inline(unmix48(h48), tmp);
        uint64_t gh = hash_bytes(tmp, ulen);
        return get_or_insert_general(tmp, ulen, gh, UINT32_MAX, nullptr, arenas, tid, tl_cur, tl_end, g_cur, g_end);
    }

    // Lock-free, strictly linearizable 8-way Uline slow path (d = 1..63 lines = up to 504 slots!).
    SS_NOINLINE uint32_t get_or_insert_h48_slow(
        uint64_t h48,
        StringArenaPool& arenas, int tid,
        uint32_t& tl_cur, uint32_t& tl_end,
        uint32_t& g_cur, uint32_t& g_end
    ) {
        size_t li = size_t(h48) & lmask;
        const uint64_t base_tag40 = (h48 >> 8) & ~63ULL;
        for (uint32_t d = 1; d <= MAX_UDISP; ++d) {
            Uline& line = ulines[li];
            const uint64_t want40 = base_tag40 | uint64_t(d);
            uint32_t match_m = 0, empty_m = 0;
            uline_probe_masks(line, want40, match_m, empty_m);
            if (__builtin_expect(match_m != 0, 1)) {
                return uint32_t(line.s[__builtin_ctz(match_m)] & 0xFFFFFFULL);
            }
            if (__builtin_expect(empty_m != 0, 1)) {
                const uint64_t want_hi = want40 << 24;
                do {
                    uint32_t w = uint32_t(__builtin_ctz(empty_m));
                    empty_m &= empty_m - 1;
                    if (__builtin_expect(tl_cur >= tl_end, 0)) {
                        if (!lease_pid_slow(tl_cur, tl_end)) return 0;
                    }
                    uint32_t np = tl_cur;
                    uint64_t exp = 0;
                    uint64_t val = want_hi | uint64_t(np);
                    std::atomic_ref<uint64_t> slot(line.s[w]);
                    if (__builtin_expect(slot.compare_exchange_strong(exp, val, std::memory_order_relaxed), 1)) {
                        pid_name[np] = unmix48(h48);
                        ++tl_cur;
                        return np;
                    }
                    if (__builtin_expect((exp >> 24) == want40, 0)) {
                        return uint32_t(exp & 0xFFFFFFULL);
                    }
                } while (empty_m != 0);
            }
            li = (li + 1) & lmask;
        }
        return get_or_insert_h48_overflow(h48, arenas, tid, tl_cur, tl_end, g_cur, g_end);
    }

    SS_ALWAYS_INLINE uint32_t get_or_insert_h48(
        uint64_t h48,
        StringArenaPool& arenas, int tid,
        uint32_t& tl_cur, uint32_t& tl_end,
        uint32_t& g_cur, uint32_t& g_end
    ) {
        uint32_t pid;
        if (__builtin_expect(try_lookup_h48_fast(ulines, lmask, h48, pid), 1)) return pid;
        return get_or_insert_h48_slow(h48, arenas, tid, tl_cur, tl_end, g_cur, g_end);
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

    void dontneed_tables(bool include_pid_name = false) {
#if defined(__linux__) && defined(MADV_DONTNEED)
        if (ulines && ulines_bytes) ::madvise(ulines, ulines_bytes, MADV_DONTNEED);
        if (hlines && hlines_bytes) ::madvise(hlines, hlines_bytes, MADV_DONTNEED);
        if (gslots && gslots_bytes) ::madvise(gslots, gslots_bytes, MADV_DONTNEED);
        if (gentries && gentries_bytes) ::madvise(gentries, gentries_bytes, MADV_DONTNEED);
        if (slots && slots_bytes) ::madvise(slots, slots_bytes, MADV_DONTNEED);
        if (include_pid_name) {
            if (pid_name && pid_name_bytes) ::madvise(pid_name, pid_name_bytes, MADV_DONTNEED);
            if (recycled_dsu && recycled_dsu_bytes) ::madvise(recycled_dsu, recycled_dsu_bytes, MADV_DONTNEED);
        }
#else
        (void)include_pid_name;
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
        size_t total_link_bytes = 0;
        for (const auto& t : tasks) total_link_bytes += (t.end > t.start) ? (t.end - t.start) : 0;

        auto alloc_fast_tables = [&](size_t est_short_users, size_t est_hex_users, size_t est_gen_users) {
            lcap = MIN_ULINES;
            while (lcap * 6 < est_short_users) lcap <<= 1; // 16 MiB on v4 (alpha ~62%, 99.9% d<=2, L3-resident!)
            lmask = lcap - 1;
            size_t raw_ubytes = lcap * sizeof(Uline);
            bool u_hp = (raw_ubytes >= (2ULL << 20));
            ulines_bytes = alloc_mmap_actual_size(raw_ubytes, u_hp);
            ulines = static_cast<Uline*>(alloc_mmap(raw_ubytes, u_hp));

            hcap = 4096;
            while (hcap * 2 < est_hex_users) hcap <<= 1; // 8 MiB on v4 (alpha ~50%, >97% d<=2, L3-resident!)
            hmask = hcap - 1;
            bool h_hp = (hcap * sizeof(HexLine) >= (2ULL << 20));
            hlines_bytes = alloc_mmap_actual_size(hcap * sizeof(HexLine), h_hp);
            hlines = static_cast<HexLine*>(alloc_mmap(hcap * sizeof(HexLine), h_hp));

            gcap = 262144;
            while (gcap * 2 < est_gen_users * 5) gcap <<= 1;
            gmask = gcap - 1;
            gslots_bytes = alloc_mmap_actual_size(gcap * sizeof(uint64_t), false);
            gslots = static_cast<uint64_t*>(alloc_mmap(gcap * sizeof(uint64_t), false));

            gentries_bytes = size_t(MAX_GENTRIES) * sizeof(GenEntry);
            gentries = static_cast<GenEntry*>(alloc_mmap(gentries_bytes, false));
            next_gentry.store(0, std::memory_order_relaxed);

            pid_name_bytes = alloc_mmap_actual_size(PID_NAME_CAP * sizeof(uint64_t), false);
            pid_name = static_cast<uint64_t*>(alloc_mmap(PID_NAME_CAP * sizeof(uint64_t), false));
            next_pid.store(1, std::memory_order_relaxed);
        };

        // Fused 1-Pass Scan + Intern + Lock-Free Union-by-Exact-Key (ZERO thread_edges, ZERO Pass 2b!)
        if (allow_fast_dict) {
            if (tasks.empty() || total_link_bytes == 0) {
                has_fallback = false;
                size_t est_short = std::max<size_t>(total_event_bytes / 1200, 16384);
                size_t est_hex   = std::max<size_t>(total_event_bytes / 12000, 4096);
                alloc_fast_tables(est_short, est_hex, 131072);
                std::pair<void*, size_t> rs[2] = {{ulines, ulines_bytes}, {hlines, hlines_bytes}};
                populate_write_parallel_ranges(pool, rs, 2);
                fast_dict_mode = true;
                return;
            }

            size_t avg_line_b = 20;
            if (tasks[0].file->map_base && tasks[0].file->file_size > 64) {
                const char* p = tasks[0].file->map_base;
                size_t lim = std::min<size_t>(tasks[0].file->file_size, 4096);
                const char* nl = find_nl_simd(p, p + lim);
                if (nl && size_t(nl - p) + 1 < lim) {
                    const char* s = nl + 1;
                    size_t nl_cnt = 0;
                    const char* last_nl = s;
                    for (const char* q = s; q < p + lim; ++q) {
                        if (*q == '\n') { ++nl_cnt; last_nl = q + 1; }
                    }
                    if (nl_cnt >= 4) avg_line_b = std::clamp<size_t>(size_t(last_nl - s) / nl_cnt, 4, 48);
                }
            }
            size_t est_edges = total_link_bytes / avg_line_b + 4096;

            struct DsuSlot {
                uint64_t key;
                uint32_t parent;
                uint32_t comp_id;
            };
            DsuSlot* dsu = nullptr;
            size_t dsu_bytes = 0, dcap = 0;
            std::atomic<size_t> unique_short_links{0};
            std::atomic<size_t> unique_fb_links{0};
            std::atomic<size_t> unique_hex_links{0};
            std::atomic<size_t> unique_nonhex_links{0};

            for (int attempt = 0; attempt < 2; ++attempt) {
                size_t min_slots = (attempt == 0)
                    ? std::max<size_t>(est_edges * 2 + (est_edges >> 1) + 4096, 8192)
                    : std::max<size_t>(total_link_bytes + 8192, 16384);
                size_t raw_dsu_bytes = min_slots * sizeof(DsuSlot);
                bool dsu_hp = (raw_dsu_bytes >= (2ULL << 20));
                dsu_bytes = dsu_hp
                    ? ((raw_dsu_bytes + (2ULL << 20) - 1) & ~((2ULL << 20) - 1))
                    : ((raw_dsu_bytes + 4095) & ~size_t(4095));
                dcap = dsu_bytes / sizeof(DsuSlot);
                dsu = static_cast<DsuSlot*>(alloc_mmap(dsu_bytes, dsu_hp));
                populate_write_parallel(pool, dsu, dsu_bytes);

                unique_short_links.store(0, std::memory_order_relaxed);
                unique_fb_links.store(0, std::memory_order_relaxed);
                unique_hex_links.store(0, std::memory_order_relaxed);
                unique_nonhex_links.store(0, std::memory_order_relaxed);
                std::atomic<size_t> total_inserted{0};
                std::atomic<bool> dsu_overflow{false};
                std::atomic<size_t> next_t{0};
                const size_t max_nodes = (dcap * 3) >> 2;

                auto slot_of = [&](uint64_t k) SS_LAMBDA_INLINE -> size_t {
                    return size_t((static_cast<unsigned __int128>(hash_key64(k, arenas)) * dcap) >> 64);
                };
                auto find_root = [&](uint32_t x) SS_LAMBDA_INLINE -> uint32_t {
                    while (true) {
                        std::atomic_ref<uint32_t> ap(dsu[x].parent);
                        uint32_t p1 = ap.load(std::memory_order_acquire);
                        if (p1 == 0) return x;
                        uint32_t p = p1 - 1;
                        uint32_t gp1 = std::atomic_ref<uint32_t>(dsu[p].parent).load(std::memory_order_acquire);
                        if (gp1 != 0) {
                            ap.compare_exchange_weak(p1, gp1, std::memory_order_acq_rel, std::memory_order_acquire);
                            x = gp1 - 1;
                        } else {
                            return p;
                        }
                    }
                };

                pool.run([&](int tid) {
                    auto& rbuf = rbufs[size_t(tid)];
                    size_t l_short = 0, l_fb = 0, l_hex = 0, l_nonhex = 0, l_ins = 0;
                    struct BatchEdge { uint64_t ka, kb; size_t sa, sb; uint8_t fa, fb; };
                    BatchEdge batch[64];
                    uint32_t nb = 0;

                    auto insert_key_at = [&](uint64_t k, size_t s, uint8_t fk) SS_LAMBDA_INLINE -> uint32_t {
                        for (size_t pr = 0; pr < dcap; ++pr) {
                            std::atomic_ref<uint64_t> ak(dsu[s].key);
                            uint64_t cur = ak.load(std::memory_order_acquire);
                            if (cur != 0) {
                                if (key_eq(cur, k, arenas)) return uint32_t(s);
                            } else {
                                if (__builtin_expect(dsu_overflow.load(std::memory_order_relaxed), 0)) return UINT32_MAX;
                                uint64_t exp = 0;
                                if (ak.compare_exchange_strong(exp, k, std::memory_order_release, std::memory_order_acquire)) {
                                    if (fk == 0) ++l_short;
                                    else { ++l_fb; if (fk == 1) ++l_hex; else ++l_nonhex; }
                                    if (((++l_ins) & 63U) == 0) {
                                        if (total_inserted.fetch_add(64, std::memory_order_relaxed) + 64 > max_nodes) {
                                            dsu_overflow.store(true, std::memory_order_relaxed);
                                        }
                                    }
                                    return uint32_t(s);
                                }
                                if (key_eq(exp, k, arenas)) return uint32_t(s);
                            }
                            if (++s == dcap) s = 0;
                        }
                        dsu_overflow.store(true, std::memory_order_relaxed);
                        return UINT32_MAX;
                    };

                    auto flush_batch = [&]() SS_LAMBDA_INLINE {
                        if (__builtin_expect(nb == 0 || dsu_overflow.load(std::memory_order_relaxed), 0)) { nb = 0; return; }
                        for (uint32_t i = 0; i < nb; ++i) {
                            if (i + 16 < nb) {
                                __builtin_prefetch(&dsu[batch[i + 16].sa], 1, 1);
                                __builtin_prefetch(&dsu[batch[i + 16].sb], 1, 1);
                            }
                            uint32_t u = insert_key_at(batch[i].ka, batch[i].sa, batch[i].fa);
                            uint32_t v = insert_key_at(batch[i].kb, batch[i].sb, batch[i].fb);
                            if (__builtin_expect((u == UINT32_MAX) | (v == UINT32_MAX), 0)) { nb = 0; return; }
                            while (u != v) {
                                u = find_root(u);
                                v = find_root(v);
                                if (u == v) break;
                                if (key_less(dsu[u].key, dsu[v].key, arenas)) std::swap(u, v);
                                uint32_t exp = 0;
                                if (std::atomic_ref<uint32_t>(dsu[u].parent).compare_exchange_weak(
                                        exp, v + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
                                    break;
                                }
                            }
                        }
                        nb = 0;
                    };

                    while (true) {
                        if (__builtin_expect(dsu_overflow.load(std::memory_order_relaxed), 0)) break;
                        size_t ti = next_t.fetch_add(1, std::memory_order_relaxed);
                        if (ti >= tasks.size()) break;
                        const CsvTask& task = tasks[ti];
                        const uint32_t seg_id = task.file->seg_id;
                        const char* seg_base = task.file->map_base;
                        scan_csv_rows<2>(task, rbuf, [&](auto row) SS_LAMBDA_INLINE {
                            Slice f0 = row.template get<0>();
                            Slice f1 = row.template get<1>();
                            if (__builtin_expect((f0.len > 0) & (f1.len > 0), 1)) {
                                uint64_t ka = encode_keyref_padded(f0.ptr, f0.len, seg_id, seg_base, arenas, tid);
                                uint64_t kb = encode_keyref_padded(f1.ptr, f1.len, seg_id, seg_base, arenas, tid);
                                if (__builtin_expect(ka == kb, 0)) return;
                                uint8_t fa = 0, fb = 0;
                                if (__builtin_expect(((ka | kb) & FALLBACK_BIT) != 0, 0)) {
                                    if (key_eq(ka, kb, arenas)) return;
                                    if (ka & FALLBACK_BIT) fa = (f0.len == 33 && f0.ptr[0] == 'u') ? 1 : 2;
                                    if (kb & FALLBACK_BIT) fb = (f1.len == 33 && f1.ptr[0] == 'u') ? 1 : 2;
                                }
                                size_t sa = slot_of(ka), sb = slot_of(kb);
                                __builtin_prefetch(&dsu[sa], 1, 1);
                                __builtin_prefetch(&dsu[sb], 1, 1);
                                batch[nb++] = {ka, kb, sa, sb, fa, fb};
                                if (__builtin_expect(nb == 64, 0)) flush_batch();
                            }
                        }, flush_batch);
                    }
                    if (l_short)  unique_short_links.fetch_add(l_short, std::memory_order_relaxed);
                    if (l_fb)     unique_fb_links.fetch_add(l_fb, std::memory_order_relaxed);
                    if (l_hex)    unique_hex_links.fetch_add(l_hex, std::memory_order_relaxed);
                    if (l_nonhex) unique_nonhex_links.fetch_add(l_nonhex, std::memory_order_relaxed);
                });
                if (!dsu_overflow.load(std::memory_order_relaxed)) break;
            }

            size_t u_short  = unique_short_links.load(std::memory_order_relaxed);
            size_t u_fb     = unique_fb_links.load(std::memory_order_relaxed);
            size_t u_hex    = unique_hex_links.load(std::memory_order_relaxed);
            size_t u_nonhex = unique_nonhex_links.load(std::memory_order_relaxed);
            has_fallback = (u_fb > 0);
            if (u_short + u_fb == 0) {
                size_t est_short = std::max<size_t>(total_event_bytes / 1200, 16384);
                size_t est_hex   = std::max<size_t>(total_event_bytes / 12000, 4096);
                alloc_fast_tables(est_short, est_hex, 131072);
                std::pair<void*, size_t> rs[2] = {{ulines, ulines_bytes}, {hlines, hlines_bytes}};
                populate_write_parallel_ranges(pool, rs, 2);
                fast_dict_mode = true;
                return;
            }
            size_t est_short = u_short + std::max<size_t>(total_event_bytes / 3500, 16384);
            size_t est_hex   = u_hex + std::max<size_t>(total_event_bytes / 24000, 4096);
            size_t est_gen   = std::max<size_t>(u_nonhex * 2 + 131072, 131072);
            if (u_nonhex > 0 && u_nonhex * 2 > u_short + u_fb) est_gen = std::max(est_gen, u_fb + total_event_bytes / 1800);
            alloc_fast_tables(est_short, est_hex, est_gen);
            std::pair<void*, size_t> rs[2] = {{ulines, ulines_bytes}, {hlines, hlines_bytes}};
            populate_write_parallel_ranges(pool, rs, 2);

            // Pass 3 (64-SLOT SOFTWARE-PIPELINED PARALLEL PUBLISH!):
            // Register-only key64_to_uid48 + load-before-CAS + write-intent home prefetch!
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
                            if (__builtin_expect((k & FALLBACK_BIT) != 0, 0)) {
                                auto sv = arenas.get_fallback(k);
                                __builtin_prefetch(sv.data(), 0, 1);
                            }
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
                                uint64_t rk = dsu[r].key;
                                uint64_t ru48 = 0;
                                if (__builtin_expect((rk & FALLBACK_BIT) == 0 && key64_to_uid48(rk, ru48), 1)) {
                                    pid_name[cand] = ru48;
                                } else {
                                    pid_name[cand] = rk;
                                }
                                cid = cand;
                                ++tl_cur;
                            } else {
                                cid = exp;
                            }
                        }
                        b_cid[j] = cid;

                        uint64_t k = b_key[j];
                        if (__builtin_expect((k & FALLBACK_BIT) == 0, 1)) {
                            uint64_t u48 = 0;
                            if (__builtin_expect(key64_to_uid48(k, u48), 1)) {
                                uint64_t h48 = mix48(u48);
                                b_dom[j] = 0;
                                b_v0[j] = h48;
                                __builtin_prefetch(&ulines[size_t(h48) & lmask], 1, 1);
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

                    for (uint32_t j = 0; j < cnt; ++j) {
                        uint8_t dm = b_dom[j];
                        if (__builtin_expect(dm == 0, 1)) {
                            uint64_t h48 = b_v0[j];
                            uint32_t cid = b_cid[j];
                            size_t li = size_t(h48) & lmask;
                            const uint64_t base_tag40 = (h48 >> 8) & ~63ULL;
                            bool placed = false;
                            for (uint32_t d = 1; d <= MAX_UDISP; ++d) {
                                Uline& line = ulines[li];
                                uint32_t empty_m = uline_empty_mask(line);
                                if (__builtin_expect(empty_m != 0, 1)) {
                                    const uint64_t val = ((base_tag40 | uint64_t(d)) << 24) | uint64_t(cid);
                                    do {
                                        uint32_t w = uint32_t(__builtin_ctz(empty_m));
                                        empty_m &= empty_m - 1;
                                        uint64_t exp = 0;
                                        if (__builtin_expect(
                                                std::atomic_ref<uint64_t>(line.s[w]).compare_exchange_strong(
                                                    exp, val, std::memory_order_relaxed), 1)) {
                                            placed = true;
                                            break;
                                        }
                                    } while (empty_m != 0);
                                    if (__builtin_expect(placed, 1)) break;
                                }
                                li = (li + 1) & lmask;
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
            ulines = nullptr; ulines_bytes = 0; lcap = 0; lmask = 0;
            hlines = nullptr; hlines_bytes = 0;
            gslots = nullptr; gslots_bytes = 0;
            gentries = nullptr; gentries_bytes = 0;
            pid_name = nullptr; pid_name_bytes = 0;
            dict_overflow.store(false, std::memory_order_relaxed);
        }

        // Classic 16B-slot DSU fallback (supports arbitrary UTF-8 / >16M components / wide windows)
        struct Edge { uint64_t a, b; };
        std::vector<std::vector<Edge>> thread_edges(static_cast<size_t>(pool.nthreads));
        std::atomic<size_t> next_t_fb{0};
        std::atomic<size_t> fb_seen{0};
        if (!tasks.empty()) {
            const size_t per_thr_reserve = std::max<size_t>(4096, total_link_bytes / size_t(pool.nthreads * 20) + 1024);
            pool.run([&](int tid) {
                auto& edges = thread_edges[size_t(tid)];
                edges.reserve(per_thr_reserve);
                auto& rbuf = rbufs[size_t(tid)];
                size_t local_fb = 0;
                auto noop_end = []() SS_LAMBDA_INLINE {};
                while (true) {
                    size_t ti = next_t_fb.fetch_add(1, std::memory_order_relaxed);
                    if (ti >= tasks.size()) break;
                    const CsvTask& task = tasks[ti];
                    const uint32_t seg_id = task.file->seg_id;
                    const char* seg_base = task.file->map_base;
                    scan_csv_rows<2>(task, rbuf, [&](auto row) SS_LAMBDA_INLINE {
                        Slice f0 = row.template get<0>();
                        Slice f1 = row.template get<1>();
                        if (__builtin_expect((f0.len > 0) & (f1.len > 0), 1)) {
                            uint64_t ka = encode_keyref_padded(f0.ptr, f0.len, seg_id, seg_base, arenas, tid);
                            uint64_t kb = encode_keyref_padded(f1.ptr, f1.len, seg_id, seg_base, arenas, tid);
                            if (__builtin_expect(ka == kb, 0)) return;
                            if (__builtin_expect(((ka | kb) & FALLBACK_BIT) != 0, 0)) {
                                if (key_eq(ka, kb, arenas)) return;
                                ++local_fb;
                            }
                            edges.push_back({ka, kb});
                        }
                    }, noop_end);
                }
                if (local_fb) fb_seen.fetch_add(local_fb, std::memory_order_relaxed);
            });
        }
        has_fallback = (fb_seen.load(std::memory_order_relaxed) > 0);
        size_t total_edges = 0;
        for (int t = 0; t < pool.nthreads; ++t) total_edges += thread_edges[size_t(t)].size();
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
// Compact Record Structs (1024-Bucket Cascade-Lake L2-Fit + 8-PID Cache-Line Blocking!)
// ============================================================================
static constexpr int EV_BUCKETS      = 1024;
static constexpr int EV_SHIFT        = 10;
static constexpr int LONG_EV_BUCKETS = 512;
static constexpr int LONG_EV_SHIFT   = 9;
static constexpr int PERSON_BUCKETS  = 512;
static constexpr int PERSON_SHIFT    = 9;

// 8-PID cache-line blocking: 8 consecutive local PIDs live on ONE 64B cache line of dict.pid_name!
static SS_ALWAYS_INLINE uint32_t pid_to_bucket(uint32_t pid) {
    return (pid >> 3) & uint32_t(PERSON_BUCKETS - 1);
}
static SS_ALWAYS_INLINE uint32_t pid_to_local(uint32_t pid) {
    return ((pid >> (PERSON_SHIFT + 3)) << 3) | (pid & 7U);
}
static SS_ALWAYS_INLINE uint32_t local_to_pid(uint32_t p, uint32_t b) {
    return ((p & ~7U) << PERSON_SHIFT) | (b << 3) | (p & 7U);
}

static constexpr uint32_t EV_BLK_CAP   = 252; // 64 + 252*16 = 4096 exact
static constexpr uint32_t LEV_BLK_CAP  = 126; // 64 + 126*32 = 4096 exact
static constexpr uint32_t PEV8_BLK_CAP = 504; // 64 + 504*8  = 4096 exact
static constexpr uint32_t CLK_BLK_CAP  = 252; // 64 + 252*16 = 4096 exact
static constexpr uint32_t PUR_BLK_CAP  = 252; // 64 + 252*16 = 4096 exact
static constexpr uint32_t WIDE_BLK_CAP = 168; // 64 + 168*24 = 4096 exact

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
        data = static_cast<FastSideEvent*>(alloc_mmap(aligned, false));
        count = 0;
        cap = uint32_t(aligned / sizeof(FastSideEvent));
    }

    SS_COLD_NOINLINE void grow() {
        size_t new_cap = std::min<size_t>(size_t(MAX_ITEMS), size_t(cap) * 2);
        if (new_cap <= cap) std::abort();
        size_t new_bytes = new_cap * sizeof(FastSideEvent);
        auto* nd = static_cast<FastSideEvent*>(alloc_mmap(new_bytes, false));
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
    // 10 bucket bits (eb = h48 >> 38) + 38 remainder bits (rem38) = exact 48-bit bijection!
    // Bit 39 = eid_in_side (separates standard eid48 namespace from generic eid_key64 namespace!)
    static constexpr uint64_t EID40_MASK = (1ULL << 40) - 1ULL;
    static constexpr uint64_t REM39_MASK = (1ULL << 39) - 1ULL;
    static constexpr uint64_t REM38_MASK = (1ULL << 38) - 1ULL;

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

struct CampDayCell {
    uint32_t sessions;
    uint32_t purchasing_sessions;
    uint32_t assisted_sessions;
    uint32_t _pad;
    uint64_t revenue_cents;
};

struct CampState {
    uint32_t count;
    uint16_t enter_sess;
    uint16_t active_pos;
};
static_assert(sizeof(CampState) == 8, "CampState must be 8 bytes");

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

    void* track_alloc(size_t bytes, bool /*hp*/) {
        void* p = alloc_mmap(bytes, false);
        allocs.emplace_back(p, alloc_mmap_actual_size(bytes, false));
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
// Zero-Hash-Table Compact FastUserDailyBuffers (v22a: 100% Dict/RLE, ZERO THP stalls!)
//   - user_id:       RLE_DICTIONARY (uint8_t uid_runs + AVX2 8-wide RLE!)
//   - day:           RLE_DICTIONARY (uint8_t day_idx + AVX2 64-wide 4-bit packer!)
//   - sessions:      RLE_DICTIONARY (uint8_t sess_idx = sessions-1, max 48/day!)
//   - events:        RLE_DICTIONARY (uint16_t ev_u16 = events-1, static seq dict when max<=4096!)
//   - revenue_cents: RLE_DICTIONARY (64-bit tzcnt on top_cat_bitmap + rev_dict[0]=0, rev_dict[1..nz]!)
//   - top_category:  RLE_DICTIONARY (8-row cur_bm pure-store -> ZERO memset + zero-copy writev!)
// ============================================================================
struct FastUserDailyBuffers {
    uint8_t*  uid_dict = nullptr;
    size_t    uid_dict_sz = 0, uid_dict_cap = 0;
    uint8_t*  uid_runs = nullptr;
    uint32_t  n_uid_dict = 0;
    size_t    uid_runs_cap = 0;
    uint8_t   uid_run_or = 0;

    uint8_t*  day_idx = nullptr;
    uint8_t*  sess_idx = nullptr;
    uint16_t* ev_u16 = nullptr;
    uint8_t*  top_cat_bitmap = nullptr;
    uint8_t   cur_bm = 0;
    uint8_t   max_sess_idx = 0;
    uint32_t  max_ev = 0;
    size_t    num_rows = 0, row_cap = 0;

    uint16_t* cat_idx = nullptr;
    int64_t*  rev_dict = nullptr;
    size_t    num_cats_nonnull = 0, cat_cap = 0;
    std::vector<std::pair<uint32_t, int32_t>> ev_overflow;

    std::vector<std::pair<void*, size_t>> allocs;

    void* track_alloc(size_t bytes, bool /*hp*/) {
        void* p = alloc_mmap(bytes, false);
        allocs.emplace_back(p, alloc_mmap_actual_size(bytes, false));
        return p;
    }

    void dontneed_all() {
#if defined(__linux__) && defined(MADV_DONTNEED)
        for (auto [p, sz] : allocs) {
            if (p && sz) ::madvise(p, sz, MADV_DONTNEED);
        }
#endif
    }

    void init(size_t est_rows, size_t uid_dict_est = 0) {
        size_t rcap = std::max<size_t>((est_rows * 5) / 4 + 4096, 65536);
        rcap = (rcap + 63U) & ~size_t(63U); // Multiple of 64 (and 8 for cur_bm!)
        row_cap = rcap;
        uid_runs_cap = rcap;
        cat_cap = std::max<size_t>((rcap >> 2) + 4096, 8192);
        uid_dict_cap = (std::max<size_t>(uid_dict_est, (rcap >> 1) * 16 + 65536) + 63U) & ~size_t(63U);

        const size_t bm_sz = ((rcap >> 3) + 64) & ~size_t(63);
        const size_t c_sz  = (cat_cap * 2 + 63) & ~size_t(63);
        const size_t rv_sz = ((cat_cap + 1) * 8 + 63) & ~size_t(63);
        const size_t d_off  = 0;
        const size_t ur_off = d_off + rcap;
        const size_t s_off  = ur_off + rcap;
        const size_t e_off  = (s_off + rcap + 63) & ~size_t(63);
        const size_t b_off  = e_off + rcap * 2;
        const size_t c_off  = b_off + bm_sz;
        const size_t rv_off = c_off + c_sz;
        const size_t u_off  = rv_off + rv_sz;
        const size_t total_bytes = u_off + uid_dict_cap;

        char* fmem = static_cast<char*>(track_alloc(total_bytes, false));
        day_idx        = reinterpret_cast<uint8_t*>(fmem + d_off);
        uid_runs       = reinterpret_cast<uint8_t*>(fmem + ur_off);
        sess_idx       = reinterpret_cast<uint8_t*>(fmem + s_off);
        ev_u16         = reinterpret_cast<uint16_t*>(fmem + e_off);
        top_cat_bitmap = reinterpret_cast<uint8_t*>(fmem + b_off);
        cat_idx        = reinterpret_cast<uint16_t*>(fmem + c_off);
        rev_dict       = reinterpret_cast<int64_t*>(fmem + rv_off);
        uid_dict       = reinterpret_cast<uint8_t*>(fmem + u_off);
        rev_dict[0]    = 0LL;
    }

    SS_ALWAYS_INLINE void finish_rows() {
        if ((num_rows & 7U) != 0) {
            top_cat_bitmap[num_rows >> 3] = cur_bm;
            cur_bm = 0;
        }
    }

    SS_COLD_NOINLINE void grow_rows() {
        const size_t new_cap = row_cap * 2; // stays multiple of 64 -> num_rows & 7 == 0 and cur_bm == 0!
        const size_t bm_sz   = ((new_cap >> 3) + 64) & ~size_t(63);
        const size_t d_off   = 0;
        const size_t s_off   = new_cap;
        const size_t e_off   = (s_off + new_cap + 63) & ~size_t(63);
        const size_t b_off   = e_off + new_cap * 2;
        const size_t fbytes  = b_off + bm_sz;
        char* fmem = static_cast<char*>(track_alloc(fbytes, false));
        auto* nd = reinterpret_cast<uint8_t*>(fmem + d_off);
        auto* ns = reinterpret_cast<uint8_t*>(fmem + s_off);
        auto* ne = reinterpret_cast<uint16_t*>(fmem + e_off);
        auto* nb = reinterpret_cast<uint8_t*>(fmem + b_off);
        std::memcpy(nd, day_idx, num_rows);
        std::memcpy(ns, sess_idx, num_rows);
        std::memcpy(ne, ev_u16, num_rows * 2);
        std::memcpy(nb, top_cat_bitmap, num_rows >> 3);
        day_idx = nd; sess_idx = ns; ev_u16 = ne; top_cat_bitmap = nb;
        row_cap = new_cap;
    }

    SS_COLD_NOINLINE void grow_uid_runs() {
        size_t new_cap = uid_runs_cap * 2;
        auto* np = static_cast<uint8_t*>(track_alloc(new_cap, false));
        if (n_uid_dict) std::memcpy(np, uid_runs, n_uid_dict);
        uid_runs = np;
        uid_runs_cap = new_cap;
    }

    SS_COLD_NOINLINE void grow_uid_dict(size_t need) {
        size_t new_cap = std::max(uid_dict_cap * 2, uid_dict_sz + need + 65536);
        auto* np = static_cast<uint8_t*>(track_alloc(new_cap, false));
        if (uid_dict_sz) std::memcpy(np, uid_dict, uid_dict_sz);
        uid_dict = np;
        uid_dict_cap = new_cap;
    }

    SS_COLD_NOINLINE void grow_cat_idx() {
        size_t new_cap = cat_cap * 2;
        size_t c_sz  = (new_cap * 2 + 63) & ~size_t(63);
        size_t rv_sz = ((new_cap + 1) * 8 + 63) & ~size_t(63);
        char* mem = static_cast<char*>(track_alloc(c_sz + rv_sz, false));
        auto* nc = reinterpret_cast<uint16_t*>(mem);
        auto* nr = reinterpret_cast<int64_t*>(mem + c_sz);
        if (num_cats_nonnull) std::memcpy(nc, cat_idx, num_cats_nonnull * 2);
        std::memcpy(nr, rev_dict, (num_cats_nonnull + 1) * 8);
        cat_idx = nc;
        rev_dict = nr;
        cat_cap = new_cap;
    }

    SS_ALWAYS_INLINE void push_uid_run(uint32_t run_rows) {
        if (__builtin_expect(n_uid_dict >= uid_runs_cap, 0)) grow_uid_runs();
        uint8_t r8 = uint8_t(run_rows);
        uid_runs[n_uid_dict++] = r8;
        uid_run_or |= r8;
    }

    SS_ALWAYS_INLINE void emit_person_uid_inline(uint64_t nm, uint32_t run_rows) {
        if (__builtin_expect(uid_dict_sz + 32 > uid_dict_cap, 0)) grow_uid_dict(32);
        char* dst = reinterpret_cast<char*>(uid_dict + uid_dict_sz);
        uint32_t ulen = __builtin_expect((nm >> 56) == 0, 1)
            ? uint32_t(decode_uid48_inline(nm, dst + 4))
            : uint32_t(decode_inline(nm, dst + 4));
        std::memcpy(dst, &ulen, 4);
        uid_dict_sz += 4 + ulen;
        push_uid_run(run_rows);
    }

    SS_ALWAYS_INLINE void emit_person_uid_sv(const char* s, size_t len, uint32_t run_rows) {
        uint32_t ulen = uint32_t(len);
        size_t tot = 4 + len;
        if (__builtin_expect(uid_dict_sz + tot + 16 > uid_dict_cap, 0)) grow_uid_dict(tot + 16);
        std::memcpy(uid_dict + uid_dict_sz, &ulen, 4);
        if (len) std::memcpy(uid_dict + uid_dict_sz + 4, s, len);
        uid_dict_sz += tot;
        push_uid_run(run_rows);
    }

    SS_ALWAYS_INLINE void push_day_row(
        uint16_t rel_day, int32_t sessions, int32_t events, int64_t rev, int32_t best_cat
    ) {
        if (__builtin_expect(num_rows >= row_cap, 0)) grow_rows();
        const size_t r = num_rows++;
        day_idx[r] = uint8_t(rel_day);
        const uint8_t s_idx = uint8_t(sessions - 1);
        sess_idx[r] = s_idx;
        if (s_idx > max_sess_idx) max_sess_idx = s_idx;
        if (__builtin_expect(uint32_t(events) <= 65535U, 1)) {
            ev_u16[r] = uint16_t(events - 1);
        } else {
            ev_u16[r] = 0;
            ev_overflow.emplace_back(uint32_t(r), events);
        }
        if (uint32_t(events) > max_ev) max_ev = uint32_t(events);
        const uint32_t bit = uint32_t(r & 7U);
        if (__builtin_expect(best_cat >= 0, 0)) {
            cur_bm |= uint8_t(1U << bit);
            if (__builtin_expect(num_cats_nonnull >= cat_cap, 0)) grow_cat_idx();
            cat_idx[num_cats_nonnull] = uint16_t(best_cat);
            rev_dict[num_cats_nonnull + 1] = rev;
            ++num_cats_nonnull;
        }
        if (__builtin_expect(bit == 7U, 0)) {
            top_cat_bitmap[r >> 3] = cur_bm;
            cur_bm = 0;
        }
    }
};

// ============================================================================
// Ultra-Fast 32-Bit Event Sorting: AVX2 Bitonic <=16 + Zero-Memcpy MSD64 + MSD256 + LSD
// ============================================================================
static SS_ALWAYS_INLINE void cswap_u32(uint32_t& a, uint32_t& b) {
    uint32_t lo = a < b ? a : b;
    uint32_t hi = a < b ? b : a;
    a = lo; b = hi;
}

#if defined(__AVX2__)
static SS_ALWAYS_INLINE __m256i bitonic_merge8_asc_avx2(__m256i v) {
    __m256i p, mn, mx;
    p  = _mm256_permute2x128_si256(v, v, 0x01);
    mn = _mm256_min_epu32(v, p); mx = _mm256_max_epu32(v, p);
    v  = _mm256_blend_epi32(mn, mx, 0xF0);

    p  = _mm256_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
    mn = _mm256_min_epu32(v, p); mx = _mm256_max_epu32(v, p);
    v  = _mm256_blend_epi32(mn, mx, 0xCC);

    p  = _mm256_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1));
    mn = _mm256_min_epu32(v, p); mx = _mm256_max_epu32(v, p);
    return _mm256_blend_epi32(mn, mx, 0xAA);
}

static SS_ALWAYS_INLINE __m256i sort8_avx2(__m256i v) {
    __m256i p, mn, mx;
    p  = _mm256_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1));
    mn = _mm256_min_epu32(v, p); mx = _mm256_max_epu32(v, p);
    v  = _mm256_blend_epi32(mn, mx, 0x66);

    p  = _mm256_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
    mn = _mm256_min_epu32(v, p); mx = _mm256_max_epu32(v, p);
    v  = _mm256_blend_epi32(mn, mx, 0x3C);

    p  = _mm256_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1));
    mn = _mm256_min_epu32(v, p); mx = _mm256_max_epu32(v, p);
    v  = _mm256_blend_epi32(mn, mx, 0x5A);

    return bitonic_merge8_asc_avx2(v);
}

alignas(32) static constexpr int32_t MASK8_U32[9][8] = {
    { 0, 0, 0, 0, 0, 0, 0, 0},
    {-1, 0, 0, 0, 0, 0, 0, 0},
    {-1,-1, 0, 0, 0, 0, 0, 0},
    {-1,-1,-1, 0, 0, 0, 0, 0},
    {-1,-1,-1,-1, 0, 0, 0, 0},
    {-1,-1,-1,-1,-1, 0, 0, 0},
    {-1,-1,-1,-1,-1,-1, 0, 0},
    {-1,-1,-1,-1,-1,-1,-1, 0},
    {-1,-1,-1,-1,-1,-1,-1,-1},
};

static SS_ALWAYS_INLINE void sort16_regs_avx2(__m256i a, __m256i b, __m256i& out_lo, __m256i& out_hi) {
    a = sort8_avx2(a);
    b = sort8_avx2(b);
    b = _mm256_permute2x128_si256(b, b, 0x01);
    b = _mm256_shuffle_epi32(b, _MM_SHUFFLE(0, 1, 2, 3));
    __m256i lo = _mm256_min_epu32(a, b);
    __m256i hi = _mm256_max_epu32(a, b);
    out_lo = bitonic_merge8_asc_avx2(lo);
    out_hi = bitonic_merge8_asc_avx2(hi);
}
#endif

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
        default:
#if defined(__AVX512VL__)
            if (__builtin_expect(n <= 8, 1)) {
                __mmask8 m = __mmask8((1U << n) - 1U);
                __m256i v = sort8_avx2(_mm256_mask_loadu_epi32(_mm256_set1_epi32(-1), m, a));
                _mm256_mask_storeu_epi32(a, m, v);
                return;
            }
            if (__builtin_expect(n <= 16, 1)) {
                __mmask8 m2 = __mmask8((1U << (n - 8U)) - 1U);
                __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                __m256i vb = _mm256_mask_loadu_epi32(_mm256_set1_epi32(-1), m2, a + 8);
                __m256i lo, hi;
                sort16_regs_avx2(va, vb, lo, hi);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(a), lo);
                _mm256_mask_storeu_epi32(a + 8, m2, hi);
                return;
            }
#elif defined(__AVX2__)
            if (__builtin_expect(n <= 8, 1)) {
                __m256i m = _mm256_load_si256(reinterpret_cast<const __m256i*>(MASK8_U32[n]));
                __m256i raw = _mm256_maskload_epi32(reinterpret_cast<const int*>(a), m);
                __m256i v = sort8_avx2(_mm256_blendv_epi8(_mm256_set1_epi32(-1), raw, m));
                _mm256_maskstore_epi32(reinterpret_cast<int*>(a), m, v);
                return;
            }
            if (__builtin_expect(n <= 16, 1)) {
                __m256i m2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(MASK8_U32[n - 8U]));
                __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                __m256i rawb = _mm256_maskload_epi32(reinterpret_cast<const int*>(a + 8), m2);
                __m256i vb = _mm256_blendv_epi8(_mm256_set1_epi32(-1), rawb, m2);
                __m256i lo, hi;
                sort16_regs_avx2(va, vb, lo, hi);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(a), lo);
                _mm256_maskstore_epi32(reinterpret_cast<int*>(a + 8), m2, hi);
                return;
            }
#endif
            for (uint32_t i = 1; i < n; ++i) {
                uint32_t x = a[i];
                uint32_t j = i;
                while (j > 0 && a[j - 1] > x) { a[j] = a[j - 1]; --j; }
                a[j] = x;
            }
            return;
    }
}

static SS_COLD_NOINLINE const uint32_t* radix_sort_u32(uint32_t* arr, uint32_t n, std::vector<uint32_t>& tmp) {
    if (tmp.size() < size_t(n) + 16) tmp.resize(size_t(n) + 16);
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
    return dst;
}

static SS_NOINLINE const uint32_t* sort_msd256_u32(uint32_t* arr, uint32_t n, uint32_t shift8, std::vector<uint32_t>& tmp) {
    if (__builtin_expect(n >= 65536, 0)) {
        return radix_sort_u32(arr, n, tmp);
    }
    if (tmp.size() < size_t(n) + 16) tmp.resize(size_t(n) + 16);
    uint16_t cnt[256]{};
    uint32_t bmin = 255, bmax = 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t b = (arr[i] >> shift8) & 255U;
        cnt[b]++;
        if (b < bmin) bmin = b;
        if (b > bmax) bmax = b;
    }
    if (__builtin_expect(bmin == bmax, 0)) { std::sort(arr, arr + n); return arr; }
    uint16_t off[256];
    uint16_t acc = 0;
    for (uint32_t b = bmin; b <= bmax; ++b) { off[b] = acc; acc = uint16_t(acc + cnt[b]); }
    uint32_t* dst = tmp.data();
    for (uint32_t i = 0; i < n; ++i) { uint32_t v = arr[i]; dst[off[(v >> shift8) & 255U]++] = v; }
    uint32_t st = 0;
    for (uint32_t b = bmin; b <= bmax; ++b) {
        uint32_t c = cnt[b];
        if (c > 1) {
            if (c <= 16) sort_small_u32(dst + st, c);
            else std::sort(dst + st, dst + st + c);
        }
        st += c;
    }
    return dst; // ZERO memcpy back to grouped_evs!
}

// Zero-memcpy event sorter: returns pointer to sorted uint32_t[n] (either arr, sbuf, or tmp.data())!
static SS_ALWAYS_INLINE const uint32_t* sort_events_u32(
    uint32_t* arr, uint32_t n, uint32_t shift8, uint64_t mul64, uint32_t* sbuf, std::vector<uint32_t>& tmp
) {
    if (n <= 4) {
        sort_small_u32(arr, n);
        return arr;
    }
#if defined(__AVX512VL__)
    if (__builtin_expect(n <= 8, 1)) {
        __mmask8 m = __mmask8((1U << n) - 1U);
        __m256i v = _mm256_mask_loadu_epi32(_mm256_set1_epi32(-1), m, arr);
        _mm256_store_si256(reinterpret_cast<__m256i*>(sbuf), sort8_avx2(v));
        return sbuf;
    }
    if (__builtin_expect(n <= 16, 1)) {
        __mmask8 m2 = __mmask8((1U << (n - 8U)) - 1U);
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr));
        __m256i vb = _mm256_mask_loadu_epi32(_mm256_set1_epi32(-1), m2, arr + 8);
        __m256i lo, hi;
        sort16_regs_avx2(va, vb, lo, hi);
        _mm256_store_si256(reinterpret_cast<__m256i*>(sbuf),     lo);
        _mm256_store_si256(reinterpret_cast<__m256i*>(sbuf + 8), hi);
        return sbuf;
    }
#elif defined(__AVX2__)
    if (__builtin_expect(n <= 8, 1)) {
        __m256i m = _mm256_load_si256(reinterpret_cast<const __m256i*>(MASK8_U32[n]));
        __m256i v = _mm256_blendv_epi8(_mm256_set1_epi32(-1),
                                       _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr)), m);
        _mm256_store_si256(reinterpret_cast<__m256i*>(sbuf), sort8_avx2(v));
        return sbuf;
    }
    if (__builtin_expect(n <= 16, 1)) {
        __m256i m2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(MASK8_U32[n - 8U]));
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr));
        __m256i vb = _mm256_blendv_epi8(_mm256_set1_epi32(-1),
                                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr + 8)), m2);
        __m256i lo, hi;
        sort16_regs_avx2(va, vb, lo, hi);
        _mm256_store_si256(reinterpret_cast<__m256i*>(sbuf),     lo);
        _mm256_store_si256(reinterpret_cast<__m256i*>(sbuf + 8), hi);
        return sbuf;
    }
#else
    if (n <= 16) {
        sort_small_u32(arr, n);
        return arr;
    }
#endif
    if (__builtin_expect(n < 256, 1)) {
        uint8_t cnt[64]{};
        uint64_t non_empty = 0;
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t b = uint32_t((uint64_t(arr[i]) * mul64) >> 32) & 63U;
            cnt[b]++;
            non_empty |= (1ULL << b);
        }
        uint8_t off[64];
        uint8_t acc = 0;
        uint64_t m = non_empty;
        while (m) {
            uint32_t b = uint32_t(__builtin_ctzll(m));
            m &= m - 1;
            off[b] = acc;
            acc = uint8_t(acc + cnt[b]);
        }
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t v = arr[i];
            sbuf[off[uint32_t((uint64_t(v) * mul64) >> 32) & 63U]++] = v;
        }
        uint32_t st = 0;
        m = non_empty;
        while (m) {
            uint32_t b = uint32_t(__builtin_ctzll(m));
            m &= m - 1;
            uint32_t c = cnt[b];
            if (c > 1) {
                if (c <= 16) sort_small_u32(sbuf + st, c);
                else std::sort(sbuf + st, sbuf + st + c);
            }
            st += c;
        }
        return sbuf; // ZERO memcpy back!
    }
    return sort_msd256_u32(arr, n, shift8, tmp);
}

static void mkdir_p(const std::string& path) {
    size_t pos = 0;
    while ((pos = path.find('/', pos + 1)) != std::string::npos) {
        ::mkdir(path.substr(0, pos).c_str(), 0755);
    }
    ::mkdir(path.c_str(), 0755);
}

// ============================================================================
// v19a Slim DSB-Resident Scanners + 8-Way AVX-512VL Uline + L1D-Hot 64/32/32 Batching
//   + Same-Day Shortcut + 128-bit Dual-HMS + 128-bit SIMD F2 + Noinline Hex Helpers
// ============================================================================
static constexpr uint32_t EV_FAST_CAP  = 128;
static constexpr uint32_t EV_HEX_CAP   = 64;
static constexpr uint32_t EV_EXTRA_CAP = 64;
static constexpr uint32_t CLK_FAST_CAP = 128;
static constexpr uint32_t CLK_HEX_CAP  = 64;

struct FastEvItem {
    uint64_t h48;
    uint64_t w0;
    uint32_t ts_rel_ms;
    uint32_t eb;
};
struct HexEvItem {
    uint64_t lo, hi, h;
    uint64_t w0;
    uint32_t ts_rel_ms;
    uint32_t eb;
};
struct ExtraEvItem {
    uint64_t u_a, u_b, u_h;  // short: u_a=h48; hex: lo=u_a, hi=u_b, h=u_h
    uint64_t e_a, e_b;       // long32: elo=e_a, ehi=e_b; side: eid_k64=e_a, rem39=e_b
    int64_t  ing_val;        // long32: ing_abs_sec; side: ing_ms
    uint32_t ts_rel_ms;
    uint32_t eb;
    uint32_t price;
    uint32_t cat_id;
    uint32_t irel21;
    uint32_t flags;          // bit0=hex_u, bit1=long32, bit2=eid_in_side
};
struct alignas(64) EvBatchState {
    uint32_t nfast = 0, nhex = 0, nextra = 0;
    FastEvItem  fast[EV_FAST_CAP];
    HexEvItem   hex[EV_HEX_CAP];
    ExtraEvItem extra[EV_EXTRA_CAP];
};

struct FastClkItem {
    uint64_t h48;
    uint64_t cid;
    uint64_t ts_camp44;
};
struct HexClkItem {
    uint64_t lo, hi, h;
    uint64_t cid;
    uint64_t ts_camp44;
};
struct alignas(64) ClkBatchState {
    uint32_t nfast = 0, nhex = 0;
    FastClkItem fast[CLK_FAST_CAP];
    HexClkItem  hex[CLK_HEX_CAP];
};

struct EvTaskCtx {
    int tid;
    uint32_t seg_id;
    const char* seg_base;
    const Products& products;
    const DateLut& lut;
    UserDict& dict;
    StringArenaPool& arenas;
    int64_t win_start_ms;
    int64_t win_start_sec;
    int64_t win_dur_ms;
    int64_t win_dur_sec;
    bool use_long_ev32;
    PartitionedStore<RawEvent16, EV_BLK_CAP, EV_BUCKETS>::ThreadState& ev_out;
    PartitionedStore<LongEv32, LEV_BLK_CAP, LONG_EV_BUCKETS>::ThreadState& long_ev_out;
    FastSideSlab& side_slab;
    EvBatchState& ebatch;
    uint32_t& tl_cur;
    uint32_t& tl_end;
    uint32_t& g_cur;
    uint32_t& g_end;
};

struct ClkTaskCtx {
    int tid;
    uint32_t seg_id;
    const char* seg_base;
    UserDict& dict;
    StringArenaPool& arenas;
    PartitionedStore<CompactClick, CLK_BLK_CAP, PERSON_BUCKETS>::ThreadState& clk_out;
    ClkBatchState& cbatch;
    uint32_t& tl_cur;
    uint32_t& tl_end;
    uint32_t& g_cur;
    uint32_t& g_end;
};

// Independent Micro-Loop B flushers: Primed 8-ahead L2->L1D prefetcht0 + prefetchw blk_cur!
static SS_NOINLINE void flush_ev_fast(EvTaskCtx& __restrict ctx) {
    EvBatchState& __restrict st = ctx.ebatch;
    const uint32_t nf = st.nfast;
    if (__builtin_expect(nf == 0, 0)) return;
    UserDict& __restrict dict = ctx.dict;
    StringArenaPool& __restrict arenas = ctx.arenas;
    const int tid = ctx.tid;
    auto& __restrict ev_out = ctx.ev_out;
    uint32_t tl_cur = ctx.tl_cur, tl_end = ctx.tl_end;
    uint32_t g_cur = ctx.g_cur, g_end = ctx.g_end;

    const UserDict::Uline* const __restrict ulines_p = dict.ulines;
    const size_t lmask_v = dict.lmask;
    const FastEvItem* const __restrict items = st.fast;

    const uint32_t prime = (nf < 8U) ? nf : 8U;
    for (uint32_t k = 0; k < prime; ++k) {
        __builtin_prefetch(&ulines_p[size_t(items[k].h48) & lmask_v], 0, 3);
        __builtin_prefetch(ev_out.blk_cur[items[k].eb], 1, 1);
    }
    const uint32_t main_end = (nf > 8U) ? (nf - 8U) : 0U;
    uint32_t i = 0;
    for (; i < main_end; ++i) {
        __builtin_prefetch(&ulines_p[size_t(items[i + 8].h48) & lmask_v], 0, 3);
        __builtin_prefetch(ev_out.blk_cur[items[i + 8].eb], 1, 1);
        const FastEvItem& r = items[i];
        uint32_t pid;
        if (__builtin_expect(!UserDict::try_lookup_h48_fast(ulines_p, lmask_v, r.h48, pid), 0)) {
            pid = dict.get_or_insert_h48_slow(r.h48, arenas, tid, tl_cur, tl_end, g_cur, g_end);
        }
        ev_out.push(r.eb, RawEvent16{r.w0, (uint64_t(r.ts_rel_ms) << 32) | (uint64_t(pid) << 8)});
    }
    for (; i < nf; ++i) {
        const FastEvItem& r = items[i];
        uint32_t pid;
        if (__builtin_expect(!UserDict::try_lookup_h48_fast(ulines_p, lmask_v, r.h48, pid), 0)) {
            pid = dict.get_or_insert_h48_slow(r.h48, arenas, tid, tl_cur, tl_end, g_cur, g_end);
        }
        ev_out.push(r.eb, RawEvent16{r.w0, (uint64_t(r.ts_rel_ms) << 32) | (uint64_t(pid) << 8)});
    }
    st.nfast = 0;
    ctx.tl_cur = tl_cur; ctx.tl_end = tl_end;
    ctx.g_cur  = g_cur;  ctx.g_end  = g_end;
}

static SS_NOINLINE void flush_ev_hex(EvTaskCtx& __restrict ctx) {
    EvBatchState& __restrict st = ctx.ebatch;
    const uint32_t nh = st.nhex;
    if (__builtin_expect(nh == 0, 0)) return;
    UserDict& __restrict dict = ctx.dict;
    StringArenaPool& __restrict arenas = ctx.arenas;
    const int tid = ctx.tid;
    auto& __restrict ev_out = ctx.ev_out;
    uint32_t tl_cur = ctx.tl_cur, tl_end = ctx.tl_end;

    const UserDict::HexLine* const __restrict hlines_p = dict.hlines;
    const size_t hmask_v = dict.hmask;
    const HexEvItem* const __restrict items = st.hex;

    const uint32_t prime = (nh < 8U) ? nh : 8U;
    for (uint32_t k = 0; k < prime; ++k) {
        __builtin_prefetch(&hlines_p[size_t(items[k].h) & hmask_v], 0, 3);
        __builtin_prefetch(ev_out.blk_cur[items[k].eb], 1, 1);
    }
    const uint32_t main_end = (nh > 8U) ? (nh - 8U) : 0U;
    uint32_t i = 0;
    for (; i < main_end; ++i) {
        __builtin_prefetch(&hlines_p[size_t(items[i + 8].h) & hmask_v], 0, 3);
        __builtin_prefetch(ev_out.blk_cur[items[i + 8].eb], 1, 1);
        const HexEvItem& r = items[i];
        uint32_t pid;
        if (__builtin_expect(!UserDict::try_lookup_hex_fast(hlines_p, hmask_v, r.lo, r.hi, r.h, pid), 0)) {
            pid = dict.get_or_insert_hex(r.lo, r.hi, r.h, arenas, tid, tl_cur, tl_end);
        }
        ev_out.push(r.eb, RawEvent16{r.w0, (uint64_t(r.ts_rel_ms) << 32) | (uint64_t(pid) << 8)});
    }
    for (; i < nh; ++i) {
        const HexEvItem& r = items[i];
        uint32_t pid;
        if (__builtin_expect(!UserDict::try_lookup_hex_fast(hlines_p, hmask_v, r.lo, r.hi, r.h, pid), 0)) {
            pid = dict.get_or_insert_hex(r.lo, r.hi, r.h, arenas, tid, tl_cur, tl_end);
        }
        ev_out.push(r.eb, RawEvent16{r.w0, (uint64_t(r.ts_rel_ms) << 32) | (uint64_t(pid) << 8)});
    }
    st.nhex = 0;
    ctx.tl_cur = tl_cur; ctx.tl_end = tl_end;
}

static SS_NOINLINE void flush_ev_extra(EvTaskCtx& __restrict ctx) {
    EvBatchState& __restrict st = ctx.ebatch;
    const uint32_t ne = st.nextra;
    if (__builtin_expect(ne == 0, 0)) return;
    UserDict& __restrict dict = ctx.dict;
    StringArenaPool& __restrict arenas = ctx.arenas;
    const int tid = ctx.tid;
    auto& __restrict ev_out = ctx.ev_out;
    auto& __restrict long_ev_out = ctx.long_ev_out;
    auto& __restrict side_slab = ctx.side_slab;
    uint32_t tl_cur = ctx.tl_cur, tl_end = ctx.tl_end;
    uint32_t g_cur = ctx.g_cur, g_end = ctx.g_end;

    const UserDict::Uline* const __restrict ulines_p = dict.ulines;
    const size_t lmask_v = dict.lmask;
    const UserDict::HexLine* const __restrict hlines_p = dict.hlines;
    const size_t hmask_v = dict.hmask;

    const uint32_t prime = (ne < 8U) ? ne : 8U;
    for (uint32_t k = 0; k < prime; ++k) {
        const ExtraEvItem& ah = st.extra[k];
        if (ah.flags & 1U) __builtin_prefetch(&hlines_p[size_t(ah.u_h) & hmask_v], 0, 3);
        else               __builtin_prefetch(&ulines_p[size_t(ah.u_a) & lmask_v], 0, 3);
    }
    const uint32_t main_end = (ne > 8U) ? (ne - 8U) : 0U;
    uint32_t i = 0;
    for (; i < ne; ++i) {
        if (__builtin_expect(i < main_end, 1)) {
            const ExtraEvItem& ah = st.extra[i + 8];
            if (ah.flags & 1U) __builtin_prefetch(&hlines_p[size_t(ah.u_h) & hmask_v], 0, 3);
            else               __builtin_prefetch(&ulines_p[size_t(ah.u_a) & lmask_v], 0, 3);
        }
        const ExtraEvItem& r = st.extra[i];
        uint32_t pid;
        if (r.flags & 1U) {
            if (__builtin_expect(!UserDict::try_lookup_hex_fast(hlines_p, hmask_v, r.u_a, r.u_b, r.u_h, pid), 0)) {
                pid = dict.get_or_insert_hex(r.u_a, r.u_b, r.u_h, arenas, tid, tl_cur, tl_end);
            }
        } else {
            if (__builtin_expect(!UserDict::try_lookup_h48_fast(ulines_p, lmask_v, r.u_a, pid), 0)) {
                pid = dict.get_or_insert_h48_slow(r.u_a, arenas, tid, tl_cur, tl_end, g_cur, g_end);
            }
        }
        if (r.flags & 2U) {
            long_ev_out.push(r.eb, LongEv32::make_valid(r.e_a, r.e_b, r.ing_val, pid, r.ts_rel_ms, r.price, r.cat_id));
        } else {
            uint32_t sidx = side_slab.push({r.e_a, r.ts_rel_ms, pid, r.ing_val, r.price, r.cat_id});
            ev_out.push(r.eb, RawEvent16::make_side(true, r.irel21, (r.flags & 4U) != 0, r.e_b, tid, sidx));
        }
    }
    st.nextra = 0;
    ctx.tl_cur = tl_cur; ctx.tl_end = tl_end;
    ctx.g_cur  = g_cur;  ctx.g_end  = g_end;
}

static SS_ALWAYS_INLINE void flush_ev_all(EvTaskCtx& ctx) {
    if (ctx.ebatch.nfast)  flush_ev_fast(ctx);
    if (ctx.ebatch.nhex)   flush_ev_hex(ctx);
    if (ctx.ebatch.nextra) flush_ev_extra(ctx);
}

static SS_NOINLINE void flush_clk_fast(ClkTaskCtx& __restrict ctx) {
    ClkBatchState& __restrict st = ctx.cbatch;
    const uint32_t nf = st.nfast;
    if (__builtin_expect(nf == 0, 0)) return;
    UserDict& __restrict dict = ctx.dict;
    StringArenaPool& __restrict arenas = ctx.arenas;
    const int tid = ctx.tid;
    auto& __restrict clk_out = ctx.clk_out;
    uint32_t tl_cur = ctx.tl_cur, tl_end = ctx.tl_end;
    uint32_t g_cur = ctx.g_cur, g_end = ctx.g_end;

    const UserDict::Uline* const __restrict ulines_p = dict.ulines;
    const size_t lmask_v = dict.lmask;
    const FastClkItem* const __restrict items = st.fast;

    const uint32_t prime = (nf < 8U) ? nf : 8U;
    for (uint32_t k = 0; k < prime; ++k) {
        __builtin_prefetch(&ulines_p[size_t(items[k].h48) & lmask_v], 0, 3);
    }
    const uint32_t main_end = (nf > 8U) ? (nf - 8U) : 0U;
    uint32_t i = 0;
    for (; i < main_end; ++i) {
        __builtin_prefetch(&ulines_p[size_t(items[i + 8].h48) & lmask_v], 0, 3);
        const FastClkItem& r = items[i];
        uint32_t pid;
        if (__builtin_expect(!UserDict::try_lookup_h48_fast(ulines_p, lmask_v, r.h48, pid), 0)) {
            pid = dict.get_or_insert_h48_slow(r.h48, arenas, tid, tl_cur, tl_end, g_cur, g_end);
        }
        clk_out.push(pid_to_bucket(pid), CompactClick{r.cid, (uint64_t(pid_to_local(pid)) << 44) | r.ts_camp44});
    }
    for (; i < nf; ++i) {
        const FastClkItem& r = items[i];
        uint32_t pid;
        if (__builtin_expect(!UserDict::try_lookup_h48_fast(ulines_p, lmask_v, r.h48, pid), 0)) {
            pid = dict.get_or_insert_h48_slow(r.h48, arenas, tid, tl_cur, tl_end, g_cur, g_end);
        }
        clk_out.push(pid_to_bucket(pid), CompactClick{r.cid, (uint64_t(pid_to_local(pid)) << 44) | r.ts_camp44});
    }
    st.nfast = 0;
    ctx.tl_cur = tl_cur; ctx.tl_end = tl_end;
    ctx.g_cur  = g_cur;  ctx.g_end  = g_end;
}

static SS_NOINLINE void flush_clk_hex(ClkTaskCtx& __restrict ctx) {
    ClkBatchState& __restrict st = ctx.cbatch;
    const uint32_t nh = st.nhex;
    if (__builtin_expect(nh == 0, 0)) return;
    UserDict& __restrict dict = ctx.dict;
    StringArenaPool& __restrict arenas = ctx.arenas;
    const int tid = ctx.tid;
    auto& __restrict clk_out = ctx.clk_out;
    uint32_t tl_cur = ctx.tl_cur, tl_end = ctx.tl_end;

    const UserDict::HexLine* const __restrict hlines_p = dict.hlines;
    const size_t hmask_v = dict.hmask;
    const HexClkItem* const __restrict items = st.hex;

    const uint32_t prime = (nh < 8U) ? nh : 8U;
    for (uint32_t k = 0; k < prime; ++k) {
        __builtin_prefetch(&hlines_p[size_t(items[k].h) & hmask_v], 0, 3);
    }
    const uint32_t main_end = (nh > 8U) ? (nh - 8U) : 0U;
    uint32_t i = 0;
    for (; i < main_end; ++i) {
        __builtin_prefetch(&hlines_p[size_t(items[i + 8].h) & hmask_v], 0, 3);
        const HexClkItem& r = items[i];
        uint32_t pid;
        if (__builtin_expect(!UserDict::try_lookup_hex_fast(hlines_p, hmask_v, r.lo, r.hi, r.h, pid), 0)) {
            pid = dict.get_or_insert_hex(r.lo, r.hi, r.h, arenas, tid, tl_cur, tl_end);
        }
        clk_out.push(pid_to_bucket(pid), CompactClick{r.cid, (uint64_t(pid_to_local(pid)) << 44) | r.ts_camp44});
    }
    for (; i < nh; ++i) {
        const HexClkItem& r = items[i];
        uint32_t pid;
        if (__builtin_expect(!UserDict::try_lookup_hex_fast(hlines_p, hmask_v, r.lo, r.hi, r.h, pid), 0)) {
            pid = dict.get_or_insert_hex(r.lo, r.hi, r.h, arenas, tid, tl_cur, tl_end);
        }
        clk_out.push(pid_to_bucket(pid), CompactClick{r.cid, (uint64_t(pid_to_local(pid)) << 44) | r.ts_camp44});
    }
    st.nhex = 0;
    ctx.tl_cur = tl_cur; ctx.tl_end = tl_end;
}

static SS_ALWAYS_INLINE void flush_clk_all(ClkTaskCtx& ctx) {
    if (ctx.cbatch.nfast) flush_clk_fast(ctx);
    if (ctx.cbatch.nhex)  flush_clk_hex(ctx);
}

static SS_COLD_NOINLINE void emit_ev_direct_gen_cold(
    const char* uptr, size_t ulen, uint32_t seg_id, const char* seg_base,
    uint32_t eb, uint32_t irel21, uint64_t rem39, bool is_long32, bool need_side, bool eid_in_side,
    uint32_t ts_rel_ms, uint64_t elo, uint64_t ehi, int64_t ing_abs_sec,
    uint64_t eid_k64, int64_t ing_ms, uint32_t price, uint32_t cat_id,
    UserDict& dict, StringArenaPool& arenas, int tid,
    PartitionedStore<RawEvent16, EV_BLK_CAP, EV_BUCKETS>::ThreadState& ev_out,
    PartitionedStore<LongEv32, LEV_BLK_CAP, LONG_EV_BUCKETS>::ThreadState& long_ev_out,
    FastSideSlab& side_slab,
    uint32_t& tl_cur, uint32_t& tl_end, uint32_t& g_cur, uint32_t& g_end
) {
    uint64_t h = hash_bytes(uptr, ulen);
    uint32_t pid = dict.get_or_insert_general(uptr, ulen, h, seg_id, seg_base, arenas, tid, tl_cur, tl_end, g_cur, g_end);
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
    const char* uptr, size_t ulen, uint32_t seg_id, const char* seg_base,
    uint64_t cid, uint64_t ts_camp44,
    UserDict& dict, StringArenaPool& arenas, int tid,
    PartitionedStore<CompactClick, CLK_BLK_CAP, PERSON_BUCKETS>::ThreadState& clk_out,
    uint32_t& tl_cur, uint32_t& tl_end, uint32_t& g_cur, uint32_t& g_end
) {
    uint64_t h = hash_bytes(uptr, ulen);
    uint32_t pid = dict.get_or_insert_general(uptr, ulen, h, seg_id, seg_base, arenas, tid, tl_cur, tl_end, g_cur, g_end);
    clk_out.push(pid_to_bucket(pid), CompactClick{cid, (uint64_t(pid_to_local(pid)) << 44) | ts_camp44});
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
        return {elo, ehi, 0ULL, 0ULL, uint32_t(h >> (64 - LONG_EV_SHIFT)), true};
    }
    uint64_t k64 = encode_keyref_padded(s, len, seg_id, seg_base, arenas, tid);
    uint64_t h64 = ((k64 & FALLBACK_BIT) == 0) ? mix64(k64) : hash_bytes(s, len);
    return {0ULL, 0ULL, k64, h64 & RawEvent16::REM39_MASK, uint32_t(h64 >> (64 - EV_SHIFT)), false};
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
    PartitionedStore<RawEvent16, EV_BLK_CAP, EV_BUCKETS>::ThreadState& ev_out,
    PartitionedStore<LongEv32, LEV_BLK_CAP, LONG_EV_BUCKETS>::ThreadState& long_ev_out,
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

static SS_NOINLINE void emit_clk_hex_or_gen_slow(
    const char* uptr, size_t ulen, uint64_t cid, uint64_t ts_camp44, ClkTaskCtx& __restrict ctx
) {
    uint64_t lo = 0, hi = 0;
    if (__builtin_expect(encode_uhex128(uptr, ulen, lo, hi), 1)) {
        uint64_t hh = hash_uhex128(lo, hi);
        ctx.dict.prefetch_hline_l2(hh);
        auto& cbp = ctx.cbatch;
        cbp.hex[cbp.nhex++] = {lo, hi, hh, cid, ts_camp44};
        if (__builtin_expect(cbp.nhex == CLK_HEX_CAP, 0)) flush_clk_hex(ctx);
        return;
    }
    emit_clk_direct_gen_cold(uptr, ulen, ctx.seg_id, ctx.seg_base, cid, ts_camp44,
                             ctx.dict, ctx.arenas, ctx.tid, ctx.clk_out,
                             ctx.tl_cur, ctx.tl_end, ctx.g_cur, ctx.g_end);
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
    PartitionedStore<CompactClick, CLK_BLK_CAP, PERSON_BUCKETS>::ThreadState& clk_out,
    ClkBatchState& cbatch,
    uint32_t& tl_cur, uint32_t& tl_end,
    uint32_t& g_cur,  uint32_t& g_end
) {
    const uint32_t seg_id = task.file->seg_id;
    const char* seg_base = task.file->map_base;
    ClkTaskCtx cctx{tid, seg_id, seg_base, dict, arenas, clk_out, cbatch, tl_cur, tl_end, g_cur, g_end};
    ClkTaskCtx* const ctxp = &cctx;
    auto noop_end = []() SS_LAMBDA_INLINE {};

    const int64_t cs_ms = click_start_ms;
    const int64_t we_ms = win_end_ms;
    const int64_t ws_sec = lut.win_start_sec;
    const UserDict::Uline* const __restrict ulines_p = dict.ulines;
    const size_t lmask_v = dict.lmask;
    ClkBatchState* const __restrict cbp = &cbatch;

    scan_csv_rows<4>(task, rbuf, [&](auto row) SS_LAMBDA_INLINE {
        Slice f2 = row.template get<2>();
        if (__builtin_expect(f2.len == 0, 0)) return;
        uint32_t camp_idx = 0;
        int cr = campaigns.try_lookup_fast_c(f2.ptr, f2.len, camp_idx);
        if (__builtin_expect(cr <= 0, 0)) {
            if (cr == 0) return;
            camp_idx = campaigns.lookup_padded(f2.ptr, f2.len);
            if (__builtin_expect(camp_idx == FastStringIdMap::NOT_FOUND, 0)) return;
        }
        Slice f3 = row.template get<3>();
        int64_t ts = 0;
        bool ts_ok = false;
        if (__builtin_expect((f3.len == 20) & (f3.ptr[19] == 'Z'), 1)) {
            int64_t rel_s = 0;
            ts_ok = lut.parse_iso19_rel_sec(f3.ptr, rel_s);
            ts = (ws_sec + rel_s) * 1000LL;
        } else if (__builtin_expect(f3.len == 13, 1)) {
            int64_t s = 0; uint32_t msr = 0;
            ts_ok = parse_13d_sec_ms(f3.ptr, s, msr);
            ts = s * 1000LL + int64_t(msr);
        } else if (__builtin_expect(f3.len == 25, 1)) {
            int64_t rel_s = 0;
            ts_ok = lut.parse_f2_rel_sec(f3.ptr, rel_s);
            ts = (ws_sec + rel_s) * 1000LL;
        }
        if (__builtin_expect(!ts_ok, 0)) ts_ok = parse_ts_slow(f3.ptr, f3.len, ts, lut);
        if (__builtin_expect(!ts_ok || ts < cs_ms || ts >= we_ms, 0)) return;
        Slice f0 = row.template get<0>();
        Slice f1 = row.template get<1>();
        if (__builtin_expect((f0.len == 0) | (f1.len == 0), 0)) return;

        uint64_t cid = (f0.len > 9)
            ? encode_keyref_long(f0.ptr, f0.len, seg_id, seg_base, arenas, tid)
            : encode_keyref_padded(f0.ptr, f0.len, seg_id, seg_base, arenas, tid);
        uint64_t ts_camp44 = (uint64_t(uint32_t(ts - cs_ms)) << 12) | uint64_t(camp_idx);
        uint64_t u48 = 0;
        if (__builtin_expect(encode_uid48_padded(f1.ptr, f1.len, u48), 1)) {
            uint64_t h48 = mix48(u48);
            __builtin_prefetch(&ulines_p[size_t(h48) & lmask_v], 0, 2);
            cbp->fast[cbp->nfast++] = {h48, cid, ts_camp44};
            if (__builtin_expect(cbp->nfast == CLK_FAST_CAP, 0)) flush_clk_fast(*ctxp);
        } else {
            emit_clk_hex_or_gen_slow(f1.ptr, f1.len, cid, ts_camp44, *ctxp);
        }
    }, noop_end);
    flush_clk_all(cctx);
}

template <typename RowT>
static SS_ALWAYS_INLINE void process_event_row_full_impl(RowT row, EvTaskCtx& ctx) {
    static constexpr uint32_t VIEW_U32 = 0x77656976U; // "view" LE
    static constexpr uint32_t CART_U32 = 0x74726163U; // "cart" LE
    static constexpr uint64_t PUR_U64  = 0x6573616863727570ULL; // "purchase" LE

    Slice f0 = row.template get<0>();
    if (__builtin_expect(f0.len == 0, 0)) return;

    Slice f3 = row.template get<3>();
    int64_t ing_rel_sec = 0;
    int64_t ing_ms = 0;
    bool ing_whole_sec = true;
    if (__builtin_expect(f3.len == 20 && f3.ptr[19] == 'Z', 1)) {
        if (__builtin_expect(!ctx.lut.parse_ingest_f1_rel_sec(f3.ptr, ing_rel_sec), 0)) return;
    } else {
        auto ir = parse_ingest_slow_ev(f3.ptr, f3.len, ctx.lut, ctx.win_start_sec);
        if (!ir.ok) return;
        ing_ms = ir.ing_ms;
        ing_rel_sec = ir.ing_rel_sec;
        ing_whole_sec = ir.whole_sec;
    }

    uint64_t eid48 = 0, eid_k64 = 0, rem39 = 0, elo = 0, ehi = 0;
    uint32_t eb = 0;
    bool is_long32 = false;
    bool eid_in_side = false;
    if (encode_eid48_padded(f0.ptr, f0.len, eid48)) {
        uint64_t h48 = mix48(eid48);
        eb = uint32_t(h48 >> (48 - EV_SHIFT));
        rem39 = h48 & RawEvent16::REM38_MASK;
    } else {
        auto lr = encode_long_eid_slow(f0.ptr, f0.len, ctx.use_long_ev32, ing_whole_sec, ctx.seg_id, ctx.seg_base,
                                       ctx.arenas, ctx.tid, ctx.dict);
        is_long32 = lr.is_long32;
        elo = lr.elo; ehi = lr.ehi; eid_k64 = lr.k64; eb = lr.eb; rem39 = lr.rem39;
        eid_in_side = !is_long32;
    }

    uint32_t irel21 = 0U;
    if (__builtin_expect(ing_whole_sec, 1)) {
        int64_t ir = ing_rel_sec + RawEvent16::ING_BIAS;
        if (__builtin_expect(ir > 0 && ir <= RawEvent16::ING_MAX, 1)) irel21 = uint32_t(ir);
    }

    Slice f4 = row.template get<4>();
    uint64_t v8 = 0;
    std::memcpy(&v8, f4.ptr, 8);
    uint32_t v4 = uint32_t(v8);
    bool is_vc = (f4.len == 4) & ((v4 == VIEW_U32) | (v4 == CART_U32));
    bool is_pur = (f4.len == 8) & (v8 == PUR_U64);
    Slice f1 = row.template get<1>();
    bool ok = (f1.len != 0) & row.template nonempty<5>() & (is_vc | is_pur);

    int64_t ts_rel_sec = 0;
    int64_t ts_ms = 0;
    uint32_t ts_ms_rem = 0;
    bool ts_from_sec = true;
    if (__builtin_expect(ok, 1)) {
        Slice f2 = row.template get<2>();
        bool ts_ok;
        if (__builtin_expect(f2.len == 13, 0)) {
            int64_t f3_sec = 0;
            ts_ok = parse_13d_sec_ms(f2.ptr, f3_sec, ts_ms_rem);
            ts_rel_sec = f3_sec - ctx.win_start_sec;
        } else if (__builtin_expect(f2.len == 20 || f2.len == 25, 1)) {
            ts_ok = ctx.lut.parse_iso_f12_rel_sec(f2.ptr, f2.len, ts_rel_sec);
        } else {
            ts_ok = false;
        }
        if (__builtin_expect(ts_ok & ing_whole_sec, 1)) {
            if (__builtin_expect(uint64_t(ts_rel_sec) >= uint64_t(ctx.win_dur_sec) ||
                                 ing_rel_sec - ts_rel_sec > 86400LL, 0)) {
                ok = false;
            }
        } else {
            auto cr = check_ts_slow_ev(f2.ptr, f2.len, ctx.lut, ts_ok, true, ing_whole_sec,
                                       ctx.win_start_sec, ctx.win_start_ms, ctx.win_dur_ms,
                                       ts_rel_sec, ts_ms_rem, ing_rel_sec, ing_ms, ts_ms);
            ing_ms = cr.ing_ms;
            ts_ms = cr.ts_ms;
            ok = cr.ok;
            ts_from_sec = cr.from_sec;
        }
    }

    uint32_t price = 0, cat_id = 0;
    if (ok & is_pur) {
        Slice f6 = row.template get<6>();
        Slice f5 = row.template get<5>();
        auto pr = parse_purchase_cold(f6.ptr, f6.len, f5.ptr, f5.len, ctx.products);
        if (!pr.ok) {
            ok = false;
        } else {
            price = pr.price;
            cat_id = pr.cat_id;
        }
    }

    if (__builtin_expect(!ok, 0)) {
        push_invalid_ev_cold(is_long32, elo, ehi, eid_in_side, irel21, eb, rem39, eid_k64,
                             ing_whole_sec, ctx.win_start_sec, ing_rel_sec, ing_ms,
                             ctx.tid, ctx.ev_out, ctx.long_ev_out, ctx.side_slab);
        return;
    }

    uint32_t ts_rel_ms = ts_from_sec
        ? uint32_t(ts_rel_sec * 1000LL + int64_t(ts_ms_rem))
        : uint32_t(ts_ms - ctx.win_start_ms);
    bool need_side = (!is_long32) & (is_pur | eid_in_side | (irel21 == 0));

    uint64_t u48 = 0;
    if (__builtin_expect(encode_uid48_padded(f1.ptr, f1.len, u48), 1)) {
        uint64_t h48 = mix48(u48);
        ctx.dict.prefetch_uslot_l2(h48);
        if (__builtin_expect(!(is_long32 | need_side), 0)) {
            uint64_t w0 = (1ULL << 62) | (uint64_t(irel21) << 41) | rem39;
            ctx.ebatch.fast[ctx.ebatch.nfast++] = {h48, w0, ts_rel_ms, eb};
            if (__builtin_expect(ctx.ebatch.nfast == EV_FAST_CAP, 0)) flush_ev_fast(ctx);
        } else {
            int64_t iv = is_long32 ? (ctx.win_start_sec + ing_rel_sec)
                                   : ((ing_whole_sec && irel21 == 0) ? (ctx.win_start_sec + ing_rel_sec) * 1000LL : ing_ms);
            ctx.ebatch.extra[ctx.ebatch.nextra++] = {
                h48, 0ULL, 0ULL,
                is_long32 ? elo : eid_k64, is_long32 ? ehi : rem39,
                iv, ts_rel_ms, eb, price, cat_id, irel21,
                (uint32_t(is_long32) << 1) | (uint32_t(eid_in_side) << 2)
            };
            if (__builtin_expect(ctx.ebatch.nextra == EV_EXTRA_CAP, 0)) flush_ev_extra(ctx);
        }
    } else {
        uint64_t lo = 0, hi = 0;
        if (__builtin_expect(encode_uhex128(f1.ptr, f1.len, lo, hi), 1)) {
            uint64_t hh = hash_uhex128(lo, hi);
            ctx.dict.prefetch_hline_l2(hh);
            if (__builtin_expect(!(is_long32 | need_side), 0)) {
                uint64_t w0 = (1ULL << 62) | (uint64_t(irel21) << 41) | rem39;
                ctx.ebatch.hex[ctx.ebatch.nhex++] = {lo, hi, hh, w0, ts_rel_ms, eb};
                if (__builtin_expect(ctx.ebatch.nhex == EV_HEX_CAP, 0)) flush_ev_hex(ctx);
            } else {
                int64_t iv = is_long32 ? (ctx.win_start_sec + ing_rel_sec)
                                       : ((ing_whole_sec && irel21 == 0) ? (ctx.win_start_sec + ing_rel_sec) * 1000LL : ing_ms);
                ctx.ebatch.extra[ctx.ebatch.nextra++] = {
                    lo, hi, hh,
                    is_long32 ? elo : eid_k64, is_long32 ? ehi : rem39,
                    iv, ts_rel_ms, eb, price, cat_id, irel21,
                    1U | (uint32_t(is_long32) << 1) | (uint32_t(eid_in_side) << 2)
                };
                if (__builtin_expect(ctx.ebatch.nextra == EV_EXTRA_CAP, 0)) flush_ev_extra(ctx);
            }
        } else {
            if (need_side && ing_whole_sec && irel21 == 0) ing_ms = (ctx.win_start_sec + ing_rel_sec) * 1000LL;
            emit_ev_direct_gen_cold(f1.ptr, f1.len, ctx.seg_id, ctx.seg_base,
                                    eb, irel21, rem39, is_long32, need_side, eid_in_side,
                                    ts_rel_ms, elo, ehi, ctx.win_start_sec + ing_rel_sec,
                                    eid_k64, ing_ms, price, cat_id,
                                    ctx.dict, ctx.arenas, ctx.tid, ctx.ev_out, ctx.long_ev_out, ctx.side_slab,
                                    ctx.tl_cur, ctx.tl_end, ctx.g_cur, ctx.g_end);
        }
    }
}

static SS_NOINLINE void process_event_row_extra(
    const char* bptr, const uint16_t* row_seps, uint32_t qm, EvTaskCtx& ctx
) {
    process_event_row_full_impl(SepsCsvRow<7>{bptr, row_seps, qm}, ctx);
}

static SS_COLD_NOINLINE void process_event_row_slice(const Slice* fs, EvTaskCtx& ctx) {
    process_event_row_full_impl(SliceCsvRow{fs}, ctx);
}

// Out-of-line helper for ~10-14% short-eid + 33B hex-uid events: keeps YMM encode_uhex128 & hlines_p OUT of hot DSB!
static SS_NOINLINE bool emit_ev_short_eid_hex_u(
    const char* eptr, size_t elen,
    const char* uptr, size_t ulen,
    uint32_t ir, uint32_t ts_rel_ms,
    EvTaskCtx& __restrict ctx
) {
    uint64_t eid48 = 0, lo = 0, hi = 0;
    if (__builtin_expect(!encode_eid48_padded(eptr, elen, eid48) ||
                         !encode_uhex128(uptr, ulen, lo, hi), 0)) {
        return false;
    }
    uint64_t eh48 = mix48(eid48);
    uint32_t eb = uint32_t(eh48 >> (48 - EV_SHIFT));
    uint64_t w0 = (1ULL << 62) | (uint64_t(ir) << 41) | (eh48 & RawEvent16::REM38_MASK);
    uint64_t hh = hash_uhex128(lo, hi);
    ctx.dict.prefetch_hline_l2(hh);
    auto& ebp = ctx.ebatch;
    ebp.hex[ebp.nhex++] = {lo, hi, hh, w0, ts_rel_ms, eb};
    if (__builtin_expect(ebp.nhex == EV_HEX_CAP, 0)) flush_ev_hex(ctx);
    return true;
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
    PartitionedStore<RawEvent16, EV_BLK_CAP, EV_BUCKETS>::ThreadState& ev_out,
    PartitionedStore<LongEv32, LEV_BLK_CAP, LONG_EV_BUCKETS>::ThreadState& long_ev_out,
    FastSideSlab& side_slab,
    EvBatchState& ebatch,
    uint32_t& tl_cur, uint32_t& tl_end,
    uint32_t& g_cur,  uint32_t& g_end
) {
    static constexpr uint32_t VIEW_U32 = 0x77656976U; // "view" LE
    static constexpr uint32_t CART_U32 = 0x74726163U; // "cart" LE

    EvTaskCtx ctx{
        tid, task.file->seg_id, task.file->map_base,
        products, lut, dict, arenas,
        win_start_ms, win_start_sec, win_dur_ms, win_dur_sec, use_long_ev32,
        ev_out, long_ev_out, side_slab, ebatch,
        tl_cur, tl_end, g_cur, g_end
    };

    const int64_t ws_sec = win_start_sec;
    const int64_t wd_sec = win_dur_sec;
    const DateLut* const __restrict lutp = &lut;
    const UserDict::Uline* const __restrict ulines_p = dict.ulines;
    const size_t lmask_v = dict.lmask;
    EvBatchState* const __restrict ebp = &ebatch;
    EvTaskCtx* const ctxp = &ctx;
    auto noop_end = []() SS_LAMBDA_INLINE {};

    // Ultra-slim DSB-resident hot loop: Same-Day Dual-HMS + F1/F3/F2 split + L1D 128/64/64 + L2 prefetcht1!
    scan_csv_rows<7>(task, rbuf, [ws_sec, wd_sec, lutp, ulines_p, lmask_v, ebp, ctxp](auto row) SS_LAMBDA_INLINE {
        using RowT = std::decay_t<decltype(row)>;
        if constexpr (!std::is_same_v<RowT, SepsCsvRow<7>>) {
            process_event_row_slice(row.f, *ctxp);
            return;
        } else {
            Slice f0 = row.template get<0>();
            Slice f4 = row.template get<4>();
            uint32_t v4;
            std::memcpy(&v4, f4.ptr, 4);
            if (__builtin_expect(((f0.len - 2) > 7) | (f0.ptr[0] != 'e') |
                                 (f4.len != 4) | ((v4 != VIEW_U32) & (v4 != CART_U32)) |
                                 !row.template nonempty<5>(), 0)) {
                goto do_extra;
            }
            {
                Slice f2 = row.template get<2>();
                Slice f3 = row.template get<3>();
                int64_t ts_rel_sec = 0;
                int64_t ing_rel_sec = 0;
                uint32_t ts_ms_rem = 0;

                if (__builtin_expect((f2.len == 20) & (f3.len == 20) &
                                     (f2.ptr[19] == 'Z') & (f3.ptr[19] == 'Z'), 1)) {
                    // 57% F1+F1: Same-Day Shortcut + 128-bit Dual-HMS!
                    if (__builtin_expect(!lutp->parse_dual_f1_rel_sec(f2.ptr, f3.ptr, ts_rel_sec, ing_rel_sec), 0)) {
                        goto do_extra;
                    }
                } else {
                    if (__builtin_expect((f3.len != 20) | (f3.ptr[19] != 'Z') |
                                         !lutp->parse_ingest_f1_rel_sec(f3.ptr, ing_rel_sec), 0)) {
                        goto do_extra;
                    }
                    if (__builtin_expect(f2.len == 13, 1)) {
                        int64_t f3_sec = 0;
                        if (__builtin_expect(!parse_13d_sec_ms(f2.ptr, f3_sec, ts_ms_rem), 0)) goto do_extra;
                        ts_rel_sec = f3_sec - ws_sec;
                    } else if (__builtin_expect(f2.len == 25, 1)) {
                        if (__builtin_expect(!lutp->parse_f2_rel_sec(f2.ptr, ts_rel_sec), 0)) goto do_extra;
                    } else {
                        goto do_extra;
                    }
                }

                int64_t ir = ing_rel_sec + RawEvent16::ING_BIAS;
                if (__builtin_expect(
                        (uint64_t(ir - 1) >= uint64_t(RawEvent16::ING_MAX)) |
                        (uint64_t(ts_rel_sec) >= uint64_t(wd_sec)) |
                        (ing_rel_sec - ts_rel_sec > 86400LL), 0)) {
                    goto do_extra;
                }
                uint32_t ts_rel_ms = uint32_t(ts_rel_sec * 1000LL + int64_t(ts_ms_rem));

                Slice f1 = row.template get<1>();
                if (__builtin_expect(((f1.len - 2) <= 7) & (f1.ptr[0] == 'u'), 1)) {
                    uint64_t eid48 = 0, u48 = 0;
                    if (__builtin_expect(!encode_eid48_padded(f0.ptr, f0.len, eid48) ||
                                         !encode_uid48_padded(f1.ptr, f1.len, u48), 0)) {
                        goto do_extra;
                    }
                    uint64_t eh48 = mix48(eid48);
                    uint32_t eb = uint32_t(eh48 >> (48 - EV_SHIFT));
                    uint64_t w0 = (1ULL << 62) | (uint64_t(uint32_t(ir)) << 41) | (eh48 & RawEvent16::REM38_MASK);
                    uint64_t uh48 = mix48(u48);
                    __builtin_prefetch(&ulines_p[size_t(uh48) & lmask_v], 0, 2);
                    ebp->fast[ebp->nfast++] = {uh48, w0, ts_rel_ms, eb};
                    if (__builtin_expect(ebp->nfast == EV_FAST_CAP, 0)) flush_ev_fast(*ctxp);
                    return;
                }
                if (__builtin_expect((f1.len == 33) & (f1.ptr[0] == 'u'), 1)) {
                    if (__builtin_expect(emit_ev_short_eid_hex_u(f0.ptr, f0.len, f1.ptr, f1.len,
                                                                 uint32_t(ir), ts_rel_ms, *ctxp), 1)) {
                        return;
                    }
                }
            }
        do_extra:
            process_event_row_extra(row.bptr, row.sp, row.qm, *ctxp);
        }
    }, noop_end);
    flush_ev_all(ctx);
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
    PartitionedStore<PersonClick, WIDE_BLK_CAP, PERSON_BUCKETS>::ThreadState& clk_out
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
    PartitionedStore<RawEvent16, EV_BLK_CAP, EV_BUCKETS>::ThreadState& ev_out,
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
            eb = uint32_t(h48 >> (48 - EV_SHIFT));
            rem40 = h48 & RawEvent16::REM38_MASK;
        } else {
            eid_in_side = true;
            eid_k64 = encode_key64_padded(f[0].ptr, f[0].len, arenas, tid);
            uint64_t h64 = hash_key64(eid_k64, arenas);
            eb = uint32_t(h64 >> (64 - EV_SHIFT));
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
            size_t idx = size_t(d) * size_t(camp_stride) + size_t(c);
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

static SS_NOINLINE bool write_campaign_daily_and_empty_ud(
    const std::string& ud_dir,
    const std::string& cd_dir,
    bool need_empty_ud,
    int nthreads,
    int32_t win_start_day,
    int32_t num_days,
    const Campaigns& campaigns,
    std::vector<std::vector<CampDayCell>>& thread_cd
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
    auto& agg = thread_cd[0];
    for (int t = 1; t < nthreads; ++t) {
        const auto& src = thread_cd[size_t(t)];
        for (size_t i = 0; i < n_cells; ++i) {
            agg[i].sessions += src[i].sessions;
            agg[i].purchasing_sessions += src[i].purchasing_sessions;
            agg[i].assisted_sessions += src[i].assisted_sessions;
            agg[i].revenue_cents += src[i].revenue_cents;
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
            const auto& cell = agg[size_t(d) * size_t(camp_stride) + size_t(c)];
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
    if (!pq::write_parquet_file(path.c_str(), CD_COLS, data, 7, int64_t(day_v.size()))) ok = false;
    return ok;
}

// ============================================================================
// Stage 2C + Stage 3 + Stage 4 FUSED — Fast-Dict Mode (8-Way SIMD ht4 + Branchless Loop 2
// + DeadSlabPool Recycling + SIMD ev_cnt/end_mask + Day-Outer cd_cred + All-Dict Parquet!)
// ============================================================================
struct FastLocalClick {
    uint32_t ts_off;
    uint32_t camp_idx;
    uint64_t click_id;
};

struct AlignedU32Buf {
    uint32_t* ptr = nullptr;
    size_t cap = 0;
    void ensure_zero(size_t n) {
        if (n > cap) {
            if (ptr) std::free(ptr);
            cap = n;
            size_t bytes = ((n * sizeof(uint32_t)) + 63U) & ~size_t(63U);
            void* p = nullptr;
            if (::posix_memalign(&p, 64, bytes) != 0 || !p) std::abort();
            ptr = static_cast<uint32_t*>(p);
        }
        std::memset(ptr, 0, n * sizeof(uint32_t));
    }
    ~AlignedU32Buf() { if (ptr) std::free(ptr); }
};

#if defined(__AVX2__) && !defined(__AVX512VL__)
struct PackLut256 {
    alignas(64) uint8_t idx[256][8];
    constexpr PackLut256() : idx{} {
        for (uint32_t m = 0; m < 256; ++m) {
            uint32_t k = 0;
            for (uint32_t b = 0; b < 8; ++b) {
                if ((m >> b) & 1U) idx[m][k++] = uint8_t(b);
            }
        }
    }
};
static constexpr PackLut256 PACK_LUT8{};
#endif

static SS_NOINLINE bool run_fast_dict_dedup_sessionize_and_write(
    PartitionedStore<RawEvent16, EV_BLK_CAP, EV_BUCKETS>& event_store,
    PartitionedStore<LongEv32, LEV_BLK_CAP, LONG_EV_BUCKETS>& long_ev_store,
    const std::vector<FastSideSlab>& sides,
    PartitionedStore<CompactClick, CLK_BLK_CAP, PERSON_BUCKETS>& click_store,
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
    using EvStore     = PartitionedStore<RawEvent16, EV_BLK_CAP, EV_BUCKETS>;
    using LongEvStore = PartitionedStore<LongEv32, LEV_BLK_CAP, LONG_EV_BUCKETS>;
    using PEv8Store   = PartitionedStore<CompactPersonEv8, PEV8_BLK_CAP, PERSON_BUCKETS>;
    using PurStore    = PartitionedStore<CompactPurchase, PUR_BLK_CAP, PERSON_BUCKETS>;
    static_assert(sizeof(EvStore::Block) == 4096, "EvStore::Block must be 4096 bytes");
    static_assert(sizeof(LongEvStore::Block) == 4096, "LongEvStore::Block must be 4096 bytes");
    static_assert(sizeof(PEv8Store::Block) == 4096, "PEv8Store::Block must be 4096 bytes");
    static_assert(sizeof(PurStore::Block) == 4096, "PurStore::Block must be 4096 bytes");

    auto parse_threads_env = [&](const char* name, int def_val) -> int {
        const char* s = std::getenv(name);
        if (!s || !*s) return def_val;
        int v = std::atoi(s);
        if (v < 1) v = 1;
        if (v > pool.nthreads) v = pool.nthreads;
        return v;
    };
    // Default DEDUP_THREADS to pool.nthreads (8 HTs): maximizes Stage 2 L2/DRAM MLP!
    const int dedup_threads = parse_threads_env("DEDUP_THREADS", pool.nthreads);
    const bool par_free     = (std::getenv("PAR_FREE") != nullptr);

    PEv8Store person_ev_store;
    PurStore purchase_store;
    person_ev_store.init(dedup_threads, 2ULL << 20);
    purchase_store.init(dedup_threads, 2ULL << 20);

    auto now_ms = []() {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    double td0 = now_ms();

    const uint64_t win_dur = (win_end_ms > win_start_ms) ? uint64_t(win_end_ms - win_start_ms) : 1ULL;
    const uint32_t bits_dur = 64U - uint32_t(__builtin_clzll(win_dur));
    const uint32_t msd_shift8 = (bits_dur > 8U) ? (bits_dur - 8U) : 0U;
    const uint64_t msd_mul64 = (64ULL << 32) / win_dur;
    const uint64_t day0_off_ms = uint64_t(win_start_ms - int64_t(win_start_day) * DAY_MS);

    using EvBlock = typename EvStore::Block;
    using LongBlock = typename LongEvStore::Block;
    std::atomic<int> next_eb{0};
    std::atomic<int> next_leb{0};
    pool.run_n(dedup_threads, [&](int tid) {
        struct DedupSlot8 { uint32_t fp; uint32_t handle1; };
        std::vector<uint32_t> ht4;
        std::vector<DedupSlot8> ht8;
        std::vector<EvBlock*> blks;
        std::vector<LongBlock*> lblks;
        std::vector<uint32_t> sub_h[4];
        std::vector<uint64_t> sub_hash[4];
        ht4.reserve(65536);
        blks.reserve(512);
        lblks.reserve(512);

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
            const uint32_t nblks = uint32_t(blks.size());

            // Loop 1 (short): Lemire 40%-load ht4 + zero_u32_no_zmm (ZERO glibc %zmm memset!)
            if (__builtin_expect(nb <= 24000 && nblks < 255, 1)) {
                size_t hcap = (((nb * 5U) >> 1) + 127U) & ~size_t(63U);
                if (hcap < 128U) hcap = 128U;
                if (ht4.size() < hcap) ht4.resize(hcap);
                zero_u32_no_zmm(ht4.data(), hcap);
                uint32_t* const __restrict htp = ht4.data();
                const uint64_t hcap64 = uint64_t(hcap);
                for (uint32_t bi = 0; bi < nblks; ++bi) {
                    if (bi + 1 < nblks) __builtin_prefetch(blks[bi + 1], 0, 1);
                    EvBlock* blk = blks[bi];
                    const uint32_t cnt = blk->count;
                    RawEvent16* const __restrict items = blk->items;
                    const uint32_t bi8 = bi << 8;
                    for (uint32_t i = 0; i < cnt; ++i) {
                        if (__builtin_expect(i + 8 < cnt, 1)) {
                            size_t ps = size_t((uint64_t(uint32_t(items[i + 8].w0)) * hcap64) >> 32);
                            __builtin_prefetch(&htp[ps], 1, 1);
                        }
                        RawEvent16& ev = items[i];
                        uint64_t eid40 = ev.eid40();
                        uint32_t fp16 = (uint32_t(eid40) ^ uint32_t(eid40 >> 22)) << 16;
                        size_t slot = size_t((uint64_t(uint32_t(eid40)) * hcap64) >> 32);
                        while (true) {
                            uint32_t entry = htp[slot];
                            if (__builtin_expect(entry == 0, 1)) {
                                htp[slot] = fp16 | ((bi8 | i) + 1U);
                                break;
                            }
                            if (__builtin_expect((entry & 0xFFFF0000U) == fp16, 0)) {
                                uint32_t prev = (entry & 0x0000FFFFU) - 1U;
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
                            if (__builtin_expect(++slot == hcap, 0)) slot = 0;
                        }
                    }
                }
            } else {
                // Adaptive K=4 Radix Sub-Split on (eid40 >> 34) & 3
                for (int s = 0; s < 4; ++s) { sub_h[s].clear(); sub_h[s].reserve((nb >> 2) + 256); }
                for (uint32_t bi = 0; bi < nblks; ++bi) {
                    const EvBlock* blk = blks[bi];
                    const uint32_t cnt = blk->count;
                    const RawEvent16* items = blk->items;
                    const uint32_t bi8 = bi << 8;
                    for (uint32_t i = 0; i < cnt; ++i) {
                        uint32_t s = uint32_t((items[i].w0 >> 34) & 3ULL);
                        sub_h[s].push_back(bi8 | i);
                    }
                }
                for (int s = 0; s < 4; ++s) {
                    const uint32_t sn = uint32_t(sub_h[s].size());
                    if (sn == 0) continue;
                    const uint32_t* sh = sub_h[s].data();
                    size_t hcap = 64;
                    while (hcap < size_t(sn) + (size_t(sn) >> 1) + 64) hcap <<= 1;
                    const size_t hmask = hcap - 1;
                    if (__builtin_expect(sn < 65535, 1)) {
                        if (ht4.size() < hcap) ht4.resize(hcap);
                        zero_u32_no_zmm(ht4.data(), hcap);
                        uint32_t* const __restrict htp = ht4.data();
                        for (uint32_t k = 0; k < sn; ++k) {
                            if (__builtin_expect(k + 8 < sn, 1)) {
                                uint32_t ph = sh[k + 8];
                                __builtin_prefetch(&htp[size_t(blks[ph >> 8]->items[ph & 255U].w0) & hmask], 1, 1);
                            }
                            uint32_t h = sh[k];
                            RawEvent16& ev = blks[h >> 8]->items[h & 255U];
                            uint64_t eid40 = ev.eid40();
                            uint32_t fp16 = uint32_t((eid40 >> 18) & 0xFFFFULL) << 16;
                            size_t slot = size_t(eid40) & hmask;
                            while (true) {
                                uint32_t entry = htp[slot];
                                if (__builtin_expect(entry == 0, 1)) {
                                    htp[slot] = fp16 | (k + 1U);
                                    break;
                                }
                                if (__builtin_expect((entry & 0xFFFF0000U) == fp16, 0)) {
                                    uint32_t ph = sh[(entry & 0x0000FFFFU) - 1U];
                                    RawEvent16& existing = blks[ph >> 8]->items[ph & 255U];
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
                    } else {
                        ht8.assign(hcap, {0, 0});
                        for (uint32_t k = 0; k < sn; ++k) {
                            uint32_t h = sh[k];
                            RawEvent16& ev = blks[h >> 8]->items[h & 255U];
                            uint64_t eid40 = ev.eid40();
                            uint32_t fp = uint32_t(eid40 >> 8);
                            size_t slot = size_t(eid40) & hmask;
                            while (true) {
                                uint32_t h1 = ht8[slot].handle1;
                                if (__builtin_expect(h1 == 0, 1)) {
                                    ht8[slot] = {fp, k + 1U};
                                    break;
                                }
                                if (__builtin_expect(ht8[slot].fp == fp, 0)) {
                                    uint32_t ph = sh[h1 - 1U];
                                    RawEvent16& existing = blks[ph >> 8]->items[ph & 255U];
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
            }

            // Loop 2 (short): Branchless prefetchw + branchless push_cond (ZERO dup/invalid mispredicts!)
            for (uint32_t bi = 0; bi < nblks; ++bi) {
                EvBlock* blk = blks[bi];
                const uint32_t cnt = blk->count;
                const RawEvent16* const __restrict items = blk->items;
                for (uint32_t i = 0; i < cnt; ++i) {
                    if (__builtin_expect(i + 8 < cnt, 1)) {
                        const RawEvent16& ahead = items[i + 8];
                        if (__builtin_expect(int64_t(ahead.w0) >= 0, 1)) {
                            __builtin_prefetch(out_ev.blk_cur[pid_to_bucket(ahead.inline_pid())], 1, 1);
                        } else if (__builtin_expect((ahead.w0 & (1ULL << 62)) != 0, 0)) {
                            __builtin_prefetch(&sides[ahead.w1 >> 26][ahead.w1 & 0x03FFFFFFU], 0, 1);
                        }
                    }
                    const RawEvent16& ev = items[i];
                    if (__builtin_expect(int64_t(ev.w0) >= 0, 1)) {
                        uint32_t pid = ev.inline_pid();
                        out_ev.push_cond(pid_to_bucket(pid),
                                         CompactPersonEv8::make(pid_to_local(pid), ev.inline_ts_rel_ms()),
                                         (ev.w0 >> 62) & 1ULL);
                    } else if (__builtin_expect((ev.w0 & (1ULL << 62)) != 0, 0)) {
                        const FastSideEvent& se = sides[ev.w1 >> 26][ev.w1 & 0x03FFFFFFU];
                        uint32_t pid = se.pid;
                        uint32_t pb = pid_to_bucket(pid);
                        uint32_t ploc = pid_to_local(pid);
                        uint32_t ts_rel = se.ts_rel_ms;
                        out_ev.push(pb, CompactPersonEv8::make(ploc, ts_rel));
                        if (__builtin_expect(se.price > 0, 0)) {
                            out_pur.push(pb, CompactPurchase{ploc, ts_rel, se.price, se.cat_id});
                        }
                    }
                }
                out_ev.recycle_block(blk);
            }
        }

        // Direct L2-hot inline 128-bit LongEv32 dedup (Lemire 40% ht4 + zero_u32_no_zmm + branchless Loop 2!)
        while (true) {
            int lb = next_leb.fetch_add(1, std::memory_order_relaxed);
            if (lb >= LONG_EV_BUCKETS) break;
            size_t nlb = long_ev_store.bucket_count(lb);
            if (nlb == 0) continue;

            lblks.clear();
            for (int st = 0; st < pool.nthreads; ++st) {
                for (auto* blk = long_ev_store.ts[size_t(st)]->heads[lb]; blk; blk = blk->next) {
                    lblks.push_back(blk);
                }
            }
            const uint32_t nl = uint32_t(lblks.size());

            if (__builtin_expect(nlb <= 24000 && nl < 511, 1)) {
                size_t lcap = (((nlb * 5U) >> 1) + 127U) & ~size_t(63U);
                if (lcap < 128U) lcap = 128U;
                if (ht4.size() < lcap) ht4.resize(lcap);
                zero_u32_no_zmm(ht4.data(), lcap);
                uint32_t* const __restrict htp = ht4.data();
                const uint64_t lcap64 = uint64_t(lcap);
                for (uint32_t bi = 0; bi < nl; ++bi) {
                    if (bi + 1 < nl) __builtin_prefetch(lblks[bi + 1], 0, 1);
                    LongBlock* blk = lblks[bi];
                    const uint32_t cnt = blk->count;
                    LongEv32* const __restrict items = blk->items;
                    const uint32_t bi7 = bi << 7;
                    uint64_t hring[8];
                    const uint32_t prime = (cnt < 8U) ? cnt : 8U;
                    for (uint32_t k = 0; k < prime; ++k) {
                        uint64_t h = hash_uhex128(items[k].k_lo, items[k].k_hi);
                        hring[k] = h;
                        __builtin_prefetch(&htp[size_t((uint64_t(uint32_t(h)) * lcap64) >> 32)], 1, 1);
                    }
                    for (uint32_t i = 0; i < cnt; ++i) {
                        uint64_t h = hring[i & 7U];
                        if (__builtin_expect(i + 8 < cnt, 1)) {
                            uint64_t nh = hash_uhex128(items[i + 8].k_lo, items[i + 8].k_hi);
                            hring[i & 7U] = nh;
                            __builtin_prefetch(&htp[size_t((uint64_t(uint32_t(nh)) * lcap64) >> 32)], 1, 1);
                        }
                        LongEv32& ev = items[i];
                        uint32_t fp16 = (uint32_t(h) ^ uint32_t(h >> 32)) << 16;
                        size_t slot = size_t((uint64_t(uint32_t(h)) * lcap64) >> 32);
                        while (true) {
                            uint32_t entry = htp[slot];
                            if (__builtin_expect(entry == 0, 1)) {
                                htp[slot] = fp16 | ((bi7 | i) + 1U);
                                break;
                            }
                            if (__builtin_expect((entry & 0xFFFF0000U) == fp16, 0)) {
                                uint32_t prev = (entry & 0x0000FFFFU) - 1U;
                                LongEv32& existing = lblks[prev >> 7]->items[prev & 127U];
                                if (((existing.k_lo ^ ev.k_lo) | (existing.k_hi ^ ev.k_hi)) == 0) {
                                    if (ev.ing40() < existing.ing40()) {
                                        existing.ing_pid = ev.ing_pid;
                                        existing.payload = ev.payload;
                                    }
                                    ev.payload = 0;
                                    break;
                                }
                            }
                            if (__builtin_expect(++slot == lcap, 0)) slot = 0;
                        }
                    }
                }
            } else {
                // Adaptive K=4 Sub-Split on (h >> 54) & 3
                for (int s = 0; s < 4; ++s) {
                    sub_h[s].clear(); sub_h[s].reserve((nlb >> 2) + 128);
                    sub_hash[s].clear(); sub_hash[s].reserve((nlb >> 2) + 128);
                }
                for (uint32_t bi = 0; bi < nl; ++bi) {
                    const LongBlock* blk = lblks[bi];
                    const uint32_t cnt = blk->count;
                    const LongEv32* items = blk->items;
                    const uint32_t bi7 = bi << 7;
                    for (uint32_t i = 0; i < cnt; ++i) {
                        uint64_t h = hash_uhex128(items[i].k_lo, items[i].k_hi);
                        uint32_t s = uint32_t((h >> 54) & 3ULL);
                        sub_h[s].push_back(bi7 | i);
                        sub_hash[s].push_back(h);
                    }
                }
                for (int s = 0; s < 4; ++s) {
                    const uint32_t sn = uint32_t(sub_h[s].size());
                    if (sn == 0) continue;
                    const uint32_t* sh = sub_h[s].data();
                    const uint64_t* shash = sub_hash[s].data();
                    size_t lcap = 64;
                    while (lcap < size_t(sn) + (size_t(sn) >> 1) + 64) lcap <<= 1;
                    const size_t lmask = lcap - 1;
                    if (__builtin_expect(sn < 65535, 1)) {
                        if (ht4.size() < lcap) ht4.resize(lcap);
                        zero_u32_no_zmm(ht4.data(), lcap);
                        uint32_t* const __restrict htp = ht4.data();
                        for (uint32_t k = 0; k < sn; ++k) {
                            if (__builtin_expect(k + 8 < sn, 1)) {
                                __builtin_prefetch(&htp[size_t(shash[k + 8]) & lmask], 1, 1);
                            }
                            uint64_t h = shash[k];
                            uint32_t idx = sh[k];
                            LongEv32& ev = lblks[idx >> 7]->items[idx & 127U];
                            uint32_t fp16 = uint32_t((h >> 38) & 0xFFFFULL) << 16;
                            size_t slot = size_t(h) & lmask;
                            while (true) {
                                uint32_t entry = htp[slot];
                                if (__builtin_expect(entry == 0, 1)) {
                                    htp[slot] = fp16 | (k + 1U);
                                    break;
                                }
                                if (__builtin_expect((entry & 0xFFFF0000U) == fp16, 0)) {
                                    uint32_t pidx = sh[(entry & 0x0000FFFFU) - 1U];
                                    LongEv32& existing = lblks[pidx >> 7]->items[pidx & 127U];
                                    if (((existing.k_lo ^ ev.k_lo) | (existing.k_hi ^ ev.k_hi)) == 0) {
                                        if (ev.ing40() < existing.ing40()) {
                                            existing.ing_pid = ev.ing_pid;
                                            existing.payload = ev.payload;
                                        }
                                        ev.payload = 0;
                                        break;
                                    }
                                }
                                slot = (slot + 1) & lmask;
                            }
                        }
                    } else {
                        ht8.assign(lcap, {0, 0});
                        for (uint32_t k = 0; k < sn; ++k) {
                            uint64_t h = shash[k];
                            uint32_t idx = sh[k];
                            LongEv32& ev = lblks[idx >> 7]->items[idx & 127U];
                            uint32_t fp = uint32_t(h >> 22);
                            size_t slot = size_t(h) & lmask;
                            while (true) {
                                uint32_t h1 = ht8[slot].handle1;
                                if (__builtin_expect(h1 == 0, 1)) {
                                    ht8[slot] = {fp, k + 1U};
                                    break;
                                }
                                if (__builtin_expect(ht8[slot].fp == fp, 0)) {
                                    uint32_t pidx = sh[h1 - 1U];
                                    LongEv32& existing = lblks[pidx >> 7]->items[pidx & 127U];
                                    if (((existing.k_lo ^ ev.k_lo) | (existing.k_hi ^ ev.k_hi)) == 0) {
                                        if (ev.ing40() < existing.ing40()) {
                                            existing.ing_pid = ev.ing_pid;
                                            existing.payload = ev.payload;
                                        }
                                        ev.payload = 0;
                                        break;
                                    }
                                }
                                slot = (slot + 1) & lmask;
                            }
                        }
                    }
                }
            }

            // Loop 2 (LongEv32): Branchless prefetchw + branchless push_cond + LIFO recycle!
            for (uint32_t bi = 0; bi < nl; ++bi) {
                LongBlock* blk = lblks[bi];
                const uint32_t cnt = blk->count;
                const LongEv32* const __restrict items = blk->items;
                for (uint32_t i = 0; i < cnt; ++i) {
                    if (__builtin_expect(i + 8 < cnt, 1)) {
                        __builtin_prefetch(out_ev.blk_cur[pid_to_bucket(items[i + 8].pid())], 1, 1);
                    }
                    const LongEv32& ev = items[i];
                    uint32_t pid = ev.pid();
                    uint32_t pb = pid_to_bucket(pid);
                    uint32_t ploc = pid_to_local(pid);
                    uint32_t ts_rel = ev.ts_rel_ms();
                    out_ev.push_cond(pb, CompactPersonEv8::make(ploc, ts_rel), ev.payload & 1ULL);
                    uint32_t pr = ev.price();
                    if (__builtin_expect(pr > 0, 0)) {
                        out_pur.push(pb, CompactPurchase{ploc, ts_rel, pr, ev.cat_id()});
                    }
                }
                out_ev.recycle_block(blk);
            }
        }
        out_ev.flush_all();
        out_pur.flush_all();
    });
    double td1 = now_ms();

    // Stage 3 + Stage 4 FUSED: 512-Bucket Weighted Scheduling + 4-Physical-Core Zero-Hash-Table Compact Parquet!
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
    const uint32_t n_local = (max_pid > 0) ? ((((max_pid - 1U) >> (PERSON_SHIFT + 3)) + 1U) << 3) : 8U;
    const int sess_threads = parse_threads_env("SESS_THREADS", pool.nphys);

    static constexpr pq::ColSpec UD_COLS[6] = {
        {"user_id",       pq::ColKind::STRING_REQ},
        {"day",           pq::ColKind::DATE_REQ},
        {"sessions",      pq::ColKind::INT32_REQ},
        {"events",        pq::ColKind::INT32_REQ},
        {"revenue_cents", pq::ColKind::INT64_REQ},
        {"top_category",  pq::ColKind::STRING_OPT},
    };

    std::vector<FastUserDailyBuffers> thread_ud(static_cast<size_t>(sess_threads));
    std::vector<std::vector<CampDayCell>> thread_cd(static_cast<size_t>(sess_threads));
    std::atomic<size_t> total_ud_rows_written{0};
    std::atomic<int> cd_compute_done{0};
    std::atomic<bool> cd_claimed{false};
    std::atomic<bool> write_ok{true};

    const uint32_t num_camps = campaigns.organic_idx;
    const uint32_t camp_stride = num_camps + 1;
    const uint32_t num_cats = uint32_t(products.cat_names.size());
    std::atomic<int> next_pb{0};

    // Precompute tiny shared day, sessions, events, and category PLAIN dictionaries!
    std::vector<int32_t> day_dict(static_cast<size_t>(num_days));
    for (int32_t d = 0; d < num_days; ++d) day_dict[size_t(d)] = win_start_day + d;
    int32_t sess_seq_dict[256];
    for (int i = 0; i < 256; ++i) sess_seq_dict[i] = i + 1;
    std::vector<int32_t> ev_seq_dict(4096);
    for (int i = 0; i < 4096; ++i) ev_seq_dict[size_t(i)] = i + 1;
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

    const size_t ud_est_rows  = total_surviving / size_t(sess_threads * 3) + 8192;
    const size_t uid_dict_est = (size_t(max_pid) / size_t(sess_threads)) * 24 + 65536;
    for (int t = 0; t < sess_threads; ++t) thread_ud[size_t(t)].init(ud_est_rows, uid_dict_est);

    pool.run_n(sess_threads, [&](int tid) {
        auto& ud = thread_ud[size_t(tid)];

        const size_t nd = size_t(num_days);
        const size_t n_cells = size_t(camp_stride) * nd;
        auto& cd = thread_cd[size_t(tid)];
        cd.assign(n_cells, CampDayCell{0, 0, 0, 0, 0});

        std::vector<uint32_t> day_end_rel(nd);
        for (size_t d = 0; d < nd; ++d) {
            int64_t er = int64_t(win_start_day + int32_t(d) + 1) * DAY_MS - win_start_ms;
            day_end_rel[d] = (er >= (1LL << 32)) ? UINT32_MAX : uint32_t(er);
        }
        std::vector<CredCell> org_cred(nd, CredCell{0, 0, 0});
        CredCell* const __restrict org_p = org_cred.data();
        CampDayCell* const __restrict cd_p = cd.data(); // fused layout: d * camp_stride + c
        const bool small_camps = (num_camps <= 64);

        std::vector<uint32_t> ev_cnt(n_local, 0), ev_off(n_local + 8, 0);
        std::vector<uint32_t> pur_cnt(n_local, 0), pur_off(n_local + 1, 0);
        std::vector<uint32_t> clk_cnt(n_local, 0), clk_off(n_local + 1, 0);
        std::vector<uint32_t> active_pids(n_local + 8);

        std::vector<uint32_t> grouped_evs;
        std::vector<uint32_t> sort_tmp;
        std::vector<LocalPurchase> grouped_purs;
        std::vector<FastLocalClick> grouped_clks;
        alignas(32) uint32_t sbuf[272];

        std::vector<uint64_t> cat_rev(num_cats, 0);
        std::vector<uint32_t> used_cats;
        used_cats.reserve(32);

        std::vector<CampState> cstate(num_camps, {0, 0, 0});
        std::vector<uint32_t> active_camps;
        active_camps.reserve(num_camps);

        auto day_of = [&](uint32_t ts_rel) SS_LAMBDA_INLINE -> int32_t {
            return int32_t((uint64_t(ts_rel) + day0_off_ms) / uint64_t(DAY_MS));
        };

        while (true) {
            int ord = next_pb.fetch_add(1, std::memory_order_relaxed);
            if (ord >= PERSON_BUCKETS) break;
            int b = bucket_order[ord];
            size_t nev = bucket_sizes[b];
            if (nev == 0) continue;

            // 1. Branchless 1-array L1D counting (100% L1D hot, ZERO branches!)
            uint32_t* const __restrict ec = ev_cnt.data();
            for (int st = 0; st < dedup_threads; ++st) {
                for (auto* blk = person_ev_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                    const uint32_t cnt = blk->count;
                    const CompactPersonEv8* items = blk->items;
                    for (uint32_t i = 0; i < cnt; ++i) ec[items[i].pid_hi()]++;
                }
            }

            // Zero-branch 8-wide SIMD prefix-sum + active_pids compaction (vpcompressd / PACK_LUT8!)
            uint32_t* const __restrict eo = ev_off.data();
            uint32_t* const __restrict act = active_pids.data();
            uint32_t n_act = 0;
            uint32_t acc = 0;
#if defined(__AVX2__)
            const __m256i vzero   = _mm256_setzero_si256();
            const __m256i v8      = _mm256_set1_epi32(8);
            const __m256i v_seven = _mm256_set1_epi32(7);
            __m256i vp   = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256i vacc = vzero;
            for (uint32_t p = 0; p < n_local; p += 8) {
                __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ec + p));
                __m256i s = _mm256_add_epi32(v, _mm256_slli_si256(v, 4));
                s = _mm256_add_epi32(s, _mm256_slli_si256(s, 8));
                __m256i lo_sum = _mm256_shuffle_epi32(_mm256_permute2x128_si256(s, s, 0x08), 0xFF);
                __m256i inc = _mm256_add_epi32(s, lo_sum);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(eo + p), _mm256_add_epi32(_mm256_sub_epi32(inc, v), vacc));
                vacc = _mm256_add_epi32(vacc, _mm256_permutevar8x32_epi32(inc, v_seven));
#if defined(__AVX512VL__)
                __mmask8 m = _mm256_cmpneq_epi32_mask(v, vzero);
                _mm256_mask_compressstoreu_epi32(act + n_act, m, vp);
                n_act += uint32_t(__builtin_popcount(uint32_t(m)));
#else
                uint32_t m = uint32_t(~_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v, vzero)))) & 0xFFU;
                __m256i perm = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(PACK_LUT8.idx[m])));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(act + n_act), _mm256_permutevar8x32_epi32(vp, perm));
                n_act += uint32_t(__builtin_popcount(m));
#endif
                vp = _mm256_add_epi32(vp, v8);
            }
            acc = uint32_t(_mm_cvtsi128_si32(_mm256_castsi256_si128(vacc)));
#else
            for (uint32_t p = 0; p < n_local; ++p) {
                uint32_t c = ec[p];
                if (c != 0) { eo[p] = acc; acc += c; act[n_act++] = p; }
            }
#endif
            if (grouped_evs.size() < size_t(acc) + 16) grouped_evs.resize(size_t(acc) + 16);

            // 2. Branchless direct scatter to 4-byte uint32_t ts_rel_ms
            uint32_t* const __restrict ge = grouped_evs.data();
            for (int st = 0; st < dedup_threads; ++st) {
                for (auto* blk = person_ev_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                    const uint32_t cnt = blk->count;
                    const CompactPersonEv8* items = blk->items;
                    for (uint32_t i = 0; i < cnt; ++i) {
                        CompactPersonEv8 item = items[i];
                        ge[eo[item.pid_hi()]++] = item.ts_rel_ms();
                    }
                }
            }

            // 3. Direct L1D grouping of purchases
            size_t npur_bucket = purchase_store.bucket_count(b);
            uint32_t* const __restrict pc = pur_cnt.data();
            uint32_t* const __restrict po = pur_off.data();
            if (npur_bucket > 0) {
                for (int st = 0; st < dedup_threads; ++st) {
                    for (auto* blk = purchase_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                        const uint32_t cnt = blk->count;
                        const CompactPurchase* items = blk->items;
                        for (uint32_t i = 0; i < cnt; ++i) pc[items[i].pid_hi]++;
                    }
                }
                uint32_t pacc = 0;
                for (uint32_t ai = 0; ai < n_act; ++ai) { uint32_t p = act[ai]; po[p] = pacc; pacc += pc[p]; }
                grouped_purs.resize(pacc);
                LocalPurchase* const __restrict gp = grouped_purs.data();
                for (int st = 0; st < dedup_threads; ++st) {
                    for (auto* blk = purchase_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                        const uint32_t cnt = blk->count;
                        const CompactPurchase* items = blk->items;
                        for (uint32_t i = 0; i < cnt; ++i) {
                            const auto& it = items[i];
                            gp[po[it.pid_hi]++] = {it.ts_rel_ms, it.price, it.cat_id};
                        }
                    }
                }
            }

            // 4. Branchless direct L1D grouping of clicks (eliminates ~5M mispredicts on ec[p] != 0!)
            size_t nclk_bucket = click_store.bucket_count(b);
            uint32_t* const __restrict cc = clk_cnt.data();
            uint32_t* const __restrict co = clk_off.data();
            if (nclk_bucket > 0) {
                for (int st = 0; st < pool.nthreads; ++st) {
                    for (auto* blk = click_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                        const uint32_t cnt = blk->count;
                        const CompactClick* items = blk->items;
                        for (uint32_t i = 0; i < cnt; ++i) {
                            uint32_t p = items[i].pid_hi();
                            cc[p] += uint32_t(ec[p] != 0);
                        }
                    }
                }
                uint32_t cacc = 0;
                for (uint32_t ai = 0; ai < n_act; ++ai) { uint32_t p = act[ai]; co[p] = cacc; cacc += cc[p]; }
                if (cacc > 0) {
                    if (grouped_clks.size() < size_t(cacc) + 1) grouped_clks.resize(size_t(cacc) + 1);
                    FastLocalClick* const __restrict gc = grouped_clks.data();
                    for (int st = 0; st < pool.nthreads; ++st) {
                        for (auto* blk = click_store.ts[size_t(st)]->heads[b]; blk; blk = blk->next) {
                            const uint32_t cnt = blk->count;
                            const CompactClick* items = blk->items;
                            for (uint32_t i = 0; i < cnt; ++i) {
                                const CompactClick& it = items[i];
                                uint32_t p = it.pid_hi();
                                uint32_t active = uint32_t(ec[p] != 0);
                                uint32_t cur_co = co[p];
                                gc[active ? cur_co : cacc] = {it.ts_off(), it.camp_idx(), it.click_id};
                                co[p] = cur_co + active;
                            }
                        }
                    }
                }
            }

            // 5. Sessionize + Attribute + Emit with 2-stage L1-temporal pid_name/arena prefetch!
            const uint64_t* const __restrict pnames = dict.pid_name;
            const uint32_t ub = uint32_t(b);

            for (uint32_t ai = 0; ai < n_act; ++ai) {
                if (__builtin_expect(ai + 16 < n_act, 1)) {
                    __builtin_prefetch(&pnames[local_to_pid(act[ai + 16], ub)], 0, 3);
                }
                if (__builtin_expect(ai + 8 < n_act, 1)) {
                    uint64_t pf_nm = pnames[local_to_pid(act[ai + 8], ub)];
                    if (__builtin_expect((pf_nm & FALLBACK_BIT) != 0, 0)) {
                        auto sv = arenas.get_fallback(pf_nm);
                        __builtin_prefetch(sv.data(), 0, 1);
                    }
                }

                const uint32_t p = act[ai];
                const uint32_t en = ec[p];
                const uint32_t es = eo[p] - en;
                const uint32_t pn = (npur_bucket > 0) ? pc[p] : 0U;
                const uint32_t ps = (pn > 0) ? (po[p] - pn) : 0U;
                const uint32_t cn = (nclk_bucket > 0) ? cc[p] : 0U;
                const uint32_t cs = (cn > 0) ? (co[p] - cn) : 0U;

                // Reset per-pid counters right away (stays L1D hot, zero full-array memset!)
                ec[p] = 0;
                if (pn) pc[p] = 0;
                if (cn) cc[p] = 0;

                const uint32_t full_pid = local_to_pid(p, ub);
                const uint64_t nm = pnames[full_pid];
                auto emit_this_person_uid = [&](uint32_t run_rows) SS_LAMBDA_INLINE {
                    if (__builtin_expect((nm & FALLBACK_BIT) == 0, 1)) {
                        ud.emit_person_uid_inline(nm, run_rows);
                    } else {
                        auto sv = arenas.get_fallback(nm);
                        ud.emit_person_uid_sv(sv.data(), sv.size(), run_rows);
                    }
                };

                // 1. 2-point early-exit + overlapping-tail AVX2 min/max check for en in 2..64
                uint32_t* const ev_raw = ge + es;
                uint32_t first_ts = ev_raw[0];
                bool is_one_sess = (en == 1);
                constexpr uint32_t GAP = uint32_t(SESSION_GAP_MS);
                if (!is_one_sess && en <= 64 &&
                    uint32_t(ev_raw[en - 1] - first_ts + GAP) <= 2U * GAP &&
                    uint32_t(ev_raw[en >> 1] - first_ts + GAP) <= 2U * GAP) {
                    uint32_t mn = first_ts, mx = first_ts;
#if defined(__AVX2__)
                    if (en >= 8) {
                        __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ev_raw));
                        __m256i vmn = v0, vmx = v0;
                        uint32_t k = 8;
                        for (; k + 8 <= en; k += 8) {
                            __m256i vk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ev_raw + k));
                            vmn = _mm256_min_epu32(vmn, vk);
                            vmx = _mm256_max_epu32(vmx, vk);
                        }
                        if (k < en) {
                            __m256i vk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ev_raw + en - 8));
                            vmn = _mm256_min_epu32(vmn, vk);
                            vmx = _mm256_max_epu32(vmx, vk);
                        }
                        __m128i mn4 = _mm_min_epu32(_mm256_castsi256_si128(vmn), _mm256_extracti128_si256(vmn, 1));
                        mn4 = _mm_min_epu32(mn4, _mm_shuffle_epi32(mn4, _MM_SHUFFLE(1, 0, 3, 2)));
                        mn4 = _mm_min_epu32(mn4, _mm_shuffle_epi32(mn4, _MM_SHUFFLE(2, 3, 0, 1)));
                        mn = uint32_t(_mm_cvtsi128_si32(mn4));
                        __m128i mx4 = _mm_max_epu32(_mm256_castsi256_si128(vmx), _mm256_extracti128_si256(vmx, 1));
                        mx4 = _mm_max_epu32(mx4, _mm_shuffle_epi32(mx4, _MM_SHUFFLE(1, 0, 3, 2)));
                        mx4 = _mm_max_epu32(mx4, _mm_shuffle_epi32(mx4, _MM_SHUFFLE(2, 3, 0, 1)));
                        mx = uint32_t(_mm_cvtsi128_si32(mx4));
                    } else
#endif
                    {
                        for (uint32_t k = 1; k < en; ++k) {
                            uint32_t v = ev_raw[k];
                            mn = v < mn ? v : mn;
                            mx = v > mx ? v : mx;
                        }
                    }
                    if (mx - mn <= GAP) {
                        first_ts = mn;
                        is_one_sess = true;
                    }
                }

                if (is_one_sess) {
                    int32_t s_day = day_of(first_ts);
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
                        auto& cell = org_p[size_t(s_day)];
                        cell.sessions++;
                        cell.purchasing_sessions += (s_rev > 0);
                        cell.revenue_cents += s_rev;
                    } else {
                        const FastLocalClick* clks = grouped_clks.data() + cs;
                        const size_t day_off = size_t(s_day) * size_t(camp_stride);
                        if (__builtin_expect(cn == 1, 1)) {
                            if (clks[0].ts_off - first_ts <= uint32_t(CLICK_WINDOW_MS)) {
                                uint32_t camp = clks[0].camp_idx;
                                auto& cell = cd_p[day_off + camp];
                                cell.sessions++;
                                cell.purchasing_sessions += (s_rev > 0);
                                cell.assisted_sessions++;
                                cell.revenue_cents += s_rev;
                            } else {
                                auto& cell = org_p[size_t(s_day)];
                                cell.sessions++;
                                cell.purchasing_sessions += (s_rev > 0);
                                cell.revenue_cents += s_rev;
                            }
                        } else if (__builtin_expect(small_camps, 1)) {
                            int64_t best_ts = -1;
                            uint32_t best_camp = campaigns.organic_idx;
                            uint64_t best_cid = 0, seen_mask = 0;
                            for (uint32_t k = 0; k < cn; ++k) {
                                uint32_t cts = clks[k].ts_off;
                                if (cts - first_ts > uint32_t(CLICK_WINDOW_MS)) continue;
                                uint32_t camp = clks[k].camp_idx;
                                seen_mask |= (1ULL << camp);
                                if (int64_t(cts) > best_ts ||
                                    (int64_t(cts) == best_ts && key_less(clks[k].click_id, best_cid, arenas))) {
                                    best_ts = int64_t(cts);
                                    best_cid = clks[k].click_id;
                                    best_camp = camp;
                                }
                            }
                            if (best_ts >= 0) {
                                auto& cell = cd_p[day_off + best_camp];
                                cell.sessions++;
                                cell.purchasing_sessions += (s_rev > 0);
                                cell.revenue_cents += s_rev;
                            } else {
                                auto& cell = org_p[size_t(s_day)];
                                cell.sessions++;
                                cell.purchasing_sessions += (s_rev > 0);
                                cell.revenue_cents += s_rev;
                            }
                            CampDayCell* const da = cd_p + day_off;
                            while (seen_mask) {
                                uint32_t ac = uint32_t(__builtin_ctzll(seen_mask));
                                seen_mask &= seen_mask - 1;
                                da[ac].assisted_sessions += 1;
                            }
                        } else {
                            bool has_best = false;
                            uint32_t best_ts = 0, best_camp = 0;
                            uint64_t best_cid = 0;
                            for (uint32_t k = 0; k < cn; ++k) {
                                uint32_t cts = clks[k].ts_off;
                                if (cts - first_ts > uint32_t(CLICK_WINDOW_MS)) continue;
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
                            if (has_best) {
                                auto& cell = cd_p[day_off + best_camp];
                                cell.sessions++;
                                cell.purchasing_sessions += (s_rev > 0);
                                cell.revenue_cents += s_rev;
                            } else {
                                auto& cell = org_p[size_t(s_day)];
                                cell.sessions++;
                                cell.purchasing_sessions += (s_rev > 0);
                                cell.revenue_cents += s_rev;
                            }
                            for (uint32_t ac : active_camps) {
                                cd_p[day_off + ac].assisted_sessions += 1;
                                cstate[ac].count = 0;
                            }
                            active_camps.clear();
                        }
                    }
                    ud.push_day_row(uint16_t(s_day), 1, int32_t(en), int64_t(s_rev), best_c);
                    emit_this_person_uid(1U);
                    continue;
                }

                // Multi-session path: zero-memcpy AVX2/AVX-512VL bitonic/MSD64 sort + 64-bit end_mask!
                const uint32_t* evs = sort_events_u32(ev_raw, en, msd_shift8, msd_mul64, sbuf, sort_tmp);

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
                    auto clk_tie_less = [&](const FastLocalClick& a, const FastLocalClick& b) SS_LAMBDA_INLINE -> bool {
                        if (__builtin_expect(((a.click_id | b.click_id) & FALLBACK_BIT) == 0, 1)) return a.click_id > b.click_id;
                        return key_less(b.click_id, a.click_id, arenas);
                    };
                    if (cn == 2) {
                        if (clks[1].ts_off < clks[0].ts_off ||
                            (clks[1].ts_off == clks[0].ts_off && clk_tie_less(clks[1], clks[0]))) {
                            std::swap(clks[0], clks[1]);
                        }
                    } else if (cn <= 24) {
                        for (uint32_t i = 1; i < cn; ++i) {
                            FastLocalClick x = clks[i];
                            uint32_t x_ts = x.ts_off;
                            uint32_t j = i;
                            while (j > 0 && clks[j - 1].ts_off >= x_ts &&
                                   (clks[j - 1].ts_off > x_ts || clk_tie_less(x, clks[j - 1]))) {
                                clks[j] = clks[j - 1];
                                --j;
                            }
                            clks[j] = x;
                        }
                    } else {
                        std::sort(clks, clks + cn, [&](const FastLocalClick& a, const FastLocalClick& b) {
                            if (__builtin_expect(a.ts_off != b.ts_off, 1)) return a.ts_off < b.ts_off;
                            return clk_tie_less(a, b);
                        });
                    }
                }

                const uint32_t person_rows_start = uint32_t(ud.num_rows);
                int32_t cur_day = day_of(evs[0]);
                uint32_t cur_day_end_rel = day_end_rel[size_t(cur_day)];
                size_t cur_day_off = size_t(cur_day) * size_t(camp_stride);
                uint32_t day_sessions = 0, day_events = 0;
                uint64_t day_rev = 0;
                uint32_t L = 0, R = 0;
                uint32_t pk = 0;

                auto flush_day = [&]() SS_LAMBDA_INLINE {
                    if (day_sessions == 0) return;
                    if (!active_camps.empty()) {
                        for (uint32_t ac : active_camps) {
                            cd_p[cur_day_off + ac].assisted_sessions += (day_sessions - uint32_t(cstate[ac].enter_sess));
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
#if defined(__AVX512VL__)
                const __m256i v_gap = _mm256_set1_epi32(int32_t(uint32_t(SESSION_GAP_MS)));
#elif defined(__AVX2__)
                const __m256i v_sign   = _mm256_set1_epi32(int32_t(0x80000000U));
                const __m256i v_thresh = _mm256_set1_epi32(int32_t(uint32_t(SESSION_GAP_MS) ^ 0x80000000U));
#endif
                for (uint32_t base = 0; base < en; base += 64U) {
                    const uint32_t rem = en - base;
                    const uint32_t chunk_len = (rem < 64U) ? rem : 64U;
                    uint64_t end_mask = 0;
#if defined(__AVX2__)
                    for (uint32_t k = 0; k + 1U < chunk_len; k += 8U) {
                        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(evs + base + k));
                        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(evs + base + k + 1U));
#if defined(__AVX512VL__)
                        uint64_t m = uint64_t(_mm256_cmpgt_epu32_mask(_mm256_sub_epi32(vb, va), v_gap));
#else
                        __m256i d = _mm256_xor_si256(_mm256_sub_epi32(vb, va), v_sign);
                        uint64_t m = uint64_t(uint32_t(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(d, v_thresh)))));
#endif
                        end_mask |= (m << k);
                    }
#else
                    uint32_t cmp_end = (base + chunk_len < en) ? chunk_len : (chunk_len - 1U);
                    for (uint32_t k = 0; k < cmp_end; ++k) {
                        if (evs[base + k + 1U] - evs[base + k] > uint32_t(SESSION_GAP_MS)) end_mask |= (1ULL << k);
                    }
#endif
                    if (base + chunk_len == en) {
                        uint32_t lb = chunk_len - 1U;
                        end_mask = (end_mask & ((1ULL << lb) - 1ULL)) | (1ULL << lb);
                    }

                    while (end_mask) {
                        uint32_t last = base + uint32_t(__builtin_ctzll(end_mask));
                        end_mask &= end_mask - 1;
                        uint32_t s_start  = evs[i];
                        uint32_t prev     = evs[last];
                        uint32_t s_ev_cnt = last - i + 1U;
                        i = last + 1U;

                        if (__builtin_expect(s_start >= cur_day_end_rel, 0)) {
                            int32_t s_day = day_of(s_start);
                            flush_day();
                            cur_day = s_day;
                            cur_day_end_rel = day_end_rel[size_t(cur_day)];
                            cur_day_off = size_t(cur_day) * size_t(camp_stride);
                        }

                        // Lockstep purchase merge ONCE per session
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
                            auto& cell = org_p[size_t(cur_day)];
                            cell.sessions++;
                            cell.purchasing_sessions += (s_rev > 0);
                            cell.revenue_cents += s_rev;
                        } else {
                            uint32_t s_rel = s_start + uint32_t(CLICK_WINDOW_MS);
                            uint32_t win_lo = s_start;
                            while (R < cn && clks[R].ts_off <= s_rel) {
                                uint32_t c = clks[R].camp_idx;
                                if (cstate[c].count++ == 0) {
                                    cstate[c].enter_sess = uint16_t(day_sessions);
                                    cstate[c].active_pos = uint16_t(active_camps.size());
                                    active_camps.push_back(c);
                                }
                                ++R;
                            }
                            while (L < R && clks[L].ts_off < win_lo) {
                                uint32_t c = clks[L].camp_idx;
                                if (--cstate[c].count == 0) {
                                    if (day_sessions > uint32_t(cstate[c].enter_sess)) {
                                        cd_p[cur_day_off + c].assisted_sessions += (day_sessions - uint32_t(cstate[c].enter_sess));
                                    }
                                    uint32_t pos = cstate[c].active_pos;
                                    uint32_t back = active_camps.back();
                                    active_camps[pos] = back;
                                    cstate[back].active_pos = uint16_t(pos);
                                    active_camps.pop_back();
                                }
                                ++L;
                            }
                            if (L < R) {
                                auto& cell = cd_p[cur_day_off + clks[R - 1].camp_idx];
                                cell.sessions++;
                                cell.purchasing_sessions += (s_rev > 0);
                                cell.revenue_cents += s_rev;
                            } else {
                                auto& cell = org_p[size_t(cur_day)];
                                cell.sessions++;
                                cell.purchasing_sessions += (s_rev > 0);
                                cell.revenue_cents += s_rev;
                            }
                        }
                        day_sessions++;
                        day_events += s_ev_cnt;
                        day_rev += s_rev;
                    }
                }
                flush_day();
                emit_this_person_uid(uint32_t(ud.num_rows) - person_rows_start);
                for (uint32_t ac : active_camps) cstate[ac].count = 0;
                active_camps.clear();
            }
        }

        // Fold L1D-hot organic credits into fused CampDayCell BEFORE signaling cd_compute_done!
        for (size_t d = 0; d < nd; ++d) {
            auto& dst = cd_p[d * size_t(camp_stride) + campaigns.organic_idx];
            dst.sessions += org_p[d].sessions;
            dst.purchasing_sessions += org_p[d].purchasing_sessions;
            dst.revenue_cents += org_p[d].revenue_cents;
        }

        ud.finish_rows();
        if (ud.num_rows > 0) {
            total_ud_rows_written.fetch_add(ud.num_rows, std::memory_order_relaxed);
        }
        cd_compute_done.fetch_add(1, std::memory_order_acq_rel);

        if (ud.num_rows > 0) {
            std::string path = ud_dir + "/part-" + std::to_string(tid) + ".parquet";

            // 1. user_id: AVX2 8-wide RLE_DICTIONARY (1 RLE run per active person!)
            pq::Buf uid_idx_buf;
            const uint32_t n_uid_dict = ud.n_uid_dict;
            const uint8_t uid_bw = pq::bits_for_count(n_uid_dict);
            pq::encode_uid_runs_rle_dict(uid_idx_buf, ud.uid_runs, n_uid_dict, uid_bw,
                                         (ud.uid_run_or & 0xC0U) == 0);

            // 2. day: AVX2 64-wide 4-bit RLE_DICTIONARY (uint8_t day_idx!)
            pq::Buf day_idx_buf;
            pq::encode_u8_indices_rle_dict(day_idx_buf, ud.day_idx, ud.num_rows,
                                           pq::bits_for_count(uint32_t(num_days)));

            // 3. sessions: Zero-hash-table RLE_DICTIONARY (uint8_t sess_idx = sessions-1, max 48!)
            pq::Buf sess_idx_buf;
            const uint32_t n_sess_dict = uint32_t(ud.max_sess_idx) + 1U;
            pq::encode_u8_indices_rle_dict(sess_idx_buf, ud.sess_idx, ud.num_rows,
                                           pq::bits_for_count(n_sess_dict));

            // 4. events: Zero-hash-table RLE_DICTIONARY (static seq dict when max_ev<=4096, else PLAIN!)
            pq::Buf ev_idx_buf;
            std::vector<int32_t> ev_plain;
            pq::ColData ev_col{};
            if (__builtin_expect(ud.max_ev <= 4096U && ud.ev_overflow.empty(), 1)) {
                pq::encode_u16_indices_rle_dict(ev_idx_buf, ud.ev_u16, ud.num_rows,
                                                pq::bits_for_count(ud.max_ev));
                ev_col = {ev_idx_buf.data.data(), ev_idx_buf.size(), nullptr,
                          reinterpret_cast<const uint8_t*>(ev_seq_dict.data()),
                          size_t(ud.max_ev) * 4, int32_t(ud.max_ev)};
            } else {
                ev_plain.resize(ud.num_rows);
                for (size_t r = 0; r < ud.num_rows; ++r) ev_plain[r] = int32_t(ud.ev_u16[r]) + 1;
                for (auto [r, v] : ud.ev_overflow) ev_plain[r] = v;
                ev_col = {reinterpret_cast<const uint8_t*>(ev_plain.data()), ev_plain.size() * 4,
                          nullptr, nullptr, 0, 0};
            }

            // 5. revenue_cents: Zero-hash-table 64-bit tzcnt bitmap RLE_DICTIONARY (rev_dict[0]=0!)
            pq::Buf rev_idx_buf;
            pq::encode_rev_bitmap_rle_dict(rev_idx_buf, ud.top_cat_bitmap, ud.num_rows, ud.num_cats_nonnull);
            const size_t n_rev_dict = ud.num_cats_nonnull + 1;

            // 6. top_category: BMI2 PEXT 5-bit RLE_DICTIONARY + zero-copy def_bitmap!
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
                {uid_idx_buf.data.data(), uid_idx_buf.size(), nullptr,
                 ud.uid_dict, ud.uid_dict_sz, int32_t(n_uid_dict)},
                {day_idx_buf.data.data(), day_idx_buf.size(), nullptr,
                 reinterpret_cast<const uint8_t*>(day_dict.data()), day_dict.size() * 4, num_days},
                {sess_idx_buf.data.data(), sess_idx_buf.size(), nullptr,
                 reinterpret_cast<const uint8_t*>(sess_seq_dict), size_t(n_sess_dict) * 4, int32_t(n_sess_dict)},
                ev_col,
                {rev_idx_buf.data.data(), rev_idx_buf.size(), nullptr,
                 reinterpret_cast<const uint8_t*>(ud.rev_dict), n_rev_dict * 8, int32_t(n_rev_dict)},
                cat_col,
            };
            if (!pq::write_parquet_file(path.c_str(), UD_COLS, data, 6, int64_t(ud.num_rows))) {
                write_ok.store(false, std::memory_order_relaxed);
            }
        }

        // Overlapped campaign_daily write: runs in parallel with slower workers' user_daily Parquet writes!
        if (cd_compute_done.load(std::memory_order_acquire) == sess_threads &&
            !cd_claimed.exchange(true, std::memory_order_acq_rel)) {
            bool need_empty = (total_ud_rows_written.load(std::memory_order_acquire) == 0);
            if (!write_campaign_daily_and_empty_ud(
                    ud_dir, cd_dir, need_empty, sess_threads, win_start_day, num_days,
                    campaigns, thread_cd)) {
                write_ok.store(false, std::memory_order_relaxed);
            }
        }
    });

    double td2 = now_ms();

    if (prof) {
        std::fprintf(stderr, "  sub-profile: dedup=%.1fms sess+write=%.1fms (pids=%u)\n",
                     td1 - td0, td2 - td1, max_pid);
    }
    if (par_free) {
        pool.run_n(pool.nphys, [&](int tid) {
            for (int t = tid; t < dedup_threads; t += pool.nphys) {
                person_ev_store.dontneed_all_tid(t);
                purchase_store.dontneed_all_tid(t);
            }
        });
    }
    return write_ok.load(std::memory_order_acquire);
}

// ============================================================================
// Wide / Classic Exact Fallback (UTF-8 links, >49.7d windows, >4096 campaigns, >16M users)
// ============================================================================
static SS_NOINLINE bool run_wide_dedup_sessionize_and_write(
    PartitionedStore<RawEvent16, EV_BLK_CAP, EV_BUCKETS>& event_store,
    const std::vector<std::vector<SideEvent>>& sides,
    PartitionedStore<PersonClick, WIDE_BLK_CAP, PERSON_BUCKETS>& click_store,
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
    PartitionedStore<WidePersonEvent, WIDE_BLK_CAP, PERSON_BUCKETS> person_ev_store;
    person_ev_store.init(pool.nthreads);

    using EvBlock = typename PartitionedStore<RawEvent16, EV_BLK_CAP, EV_BUCKETS>::Block;
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
        struct WideCampState { uint32_t count, enter_sess, active_pos; };
        std::vector<WideCampState> cstate(num_camps, {0, 0, 0});
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
                        cd_assist[size_t(cur_day) * size_t(camp_stride) + size_t(ac)] += (day_sessions - cstate[ac].enter_sess);
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
                        auto& cell = cd_cred[size_t(cur_day) * size_t(camp_stride) + size_t(campaigns.organic_idx)];
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
                                    cd_assist[size_t(cur_day) * size_t(camp_stride) + size_t(c)] += (day_sessions - cstate[c].enter_sess);
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
                        auto& cell = cd_cred[size_t(cur_day) * size_t(camp_stride) + size_t(credit_c)];
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
#if defined(__GLIBC__)
    // Eliminate ~56 mid-run glibc munmap() write-lock TLB-shootdown IPIs from temporary std::vectors!
    ::mallopt(M_MMAP_THRESHOLD, 16 << 20);
    ::mallopt(M_TRIM_THRESHOLD, -1);
#endif
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

    Products products;
    Campaigns campaigns;

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
    std::atomic<size_t> next_item{0};
    const size_t total_open_items = all_paths.size() + 2;

    // Parallel products.load + campaigns.load + open/fstat/fadvise/mmap (ZERO 9th oversubscribed thread!)
    pool.run([&](int) {
        while (true) {
            size_t idx = next_item.fetch_add(1, std::memory_order_relaxed);
            if (idx >= total_open_items) break;
            if (idx == 0) { products.load(data_dir + "/products.csv"); continue; }
            if (idx == 1) { campaigns.load(data_dir + "/campaigns.csv"); continue; }
            size_t i = idx - 2;
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
            if (mbase != nullptr || !keep_fds || fsz == 0) {
                ::close(fd); fd = -1;
            } else {
#if defined(__linux__) && defined(POSIX_FADV_SEQUENTIAL)
                ::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
            }
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
    double t_links = now_ms();

    bool use_fast_dict = manifest_compact_ok && dict.fast_dict_mode &&
                         (campaigns.organic_idx < 4096) &&
                         (num_days <= 63) &&
                         (products.cat_names.size() <= 65535);

    bool write_ok = false;
    double t_scan = t_links;

    if (use_fast_dict) {
        const bool use_long_ev32 = (products.cat_names.size() <= 127);
        PartitionedStore<CompactClick, CLK_BLK_CAP, PERSON_BUCKETS> click_store;
        PartitionedStore<RawEvent16, EV_BLK_CAP, EV_BUCKETS> event_store;
        PartitionedStore<LongEv32, LEV_BLK_CAP, LONG_EV_BUCKETS> long_ev_store;
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

                ClkBatchState cbatch;
                EvBatchState  ebatch;
                uint32_t tl_cur = 0, tl_end = 0;
                uint32_t g_cur = 0, g_end = 0;

                while (true) {
                    size_t ti = next_t.fetch_add(1, std::memory_order_relaxed);
                    if (ti >= io_tasks.size()) break;
                    const auto& task = io_tasks[ti];
                    if (task.kind == 0) {
                        process_click_task_fast(task, rbuf, tid, campaigns, lut, dict, arenas,
                                                click_start_ms, win_end_ms, clk_out, cbatch, tl_cur, tl_end, g_cur, g_end);
                    } else {
                        process_event_task_fast(task, rbuf, tid, products, lut, dict, arenas,
                                                win_start_ms, win_start_sec, win_dur_ms, win_dur_sec, use_long_ev32,
                                                ev_out, long_ev_out, side_slab, ebatch, tl_cur, tl_end, g_cur, g_end);
                    }
                }
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
            double t_end = now_ms();
            if (prof) {
                std::fprintf(stderr,
                    "PROFILE (ms): init+open=%.1f links+dims=%.1f scan=%.1f dedup+sess+write=%.1f TOTAL=%.1f\n",
                    t_init - t0, t_links - t_init, t_scan - t_links, t_end - t_scan, t_end - t0);
            }
            if (!write_ok) {
                std::fprintf(stderr, "Parquet write failed\n");
                ss_on_exit(1);
            }
            // Pre-exit MADV_DONTNEED is opt-in (PAR_FREE=1): avoids 60-80 concurrent TLB shootdown IPIs!
            if (std::getenv("PAR_FREE") != nullptr) {
                pool.run_n(pool.nphys, [&](int tid) {
                    if (tid == 0) dict.dontneed_tables(true);
                    for (int t = tid; t < ncpu; t += pool.nphys) {
                        event_store.dontneed_all_tid(t);
                        long_ev_store.dontneed_all_tid(t);
                        click_store.dontneed_all_tid(t);
#if defined(__linux__) && defined(MADV_DONTNEED)
                        if (thread_sides[size_t(t)].data && thread_sides[size_t(t)].count > 0) {
                            size_t sb = (size_t(thread_sides[size_t(t)].count) * sizeof(FastSideEvent) + 4095) & ~size_t(4095);
                            ::madvise(thread_sides[size_t(t)].data, sb, MADV_DONTNEED);
                        }
                        if (arenas.base[t] && arenas.pos[t] > 0) {
                            size_t ab = (arenas.pos[t] + 4095) & ~size_t(4095);
                            ::madvise(arenas.base[t], ab, MADV_DONTNEED);
                        }
#endif
                    }
                });
            }
            ss_on_exit(0);
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
        PartitionedStore<PersonClick, WIDE_BLK_CAP, PERSON_BUCKETS> click_store;
        PartitionedStore<RawEvent16, EV_BLK_CAP, EV_BUCKETS> event_store;
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
        ss_on_exit(1);
    }
    ss_on_exit(0);
}

} // namespace SS_ENGINE_NS
