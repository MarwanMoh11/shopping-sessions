#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

#ifndef SS_ENGINE_NS
#define SS_ENGINE_NS ss_default
#endif

#define SS_ALWAYS_INLINE __attribute__((always_inline)) inline
#define SS_LAMBDA_INLINE __attribute__((always_inline))
#define SS_COLD_NOINLINE __attribute__((noinline, cold))
#define SS_NOINLINE      __attribute__((noinline))

namespace SS_ENGINE_NS {

static constexpr int64_t SECOND_MS       = 1000LL;
static constexpr int64_t MINUTE_MS       = 60LL * SECOND_MS;
static constexpr int64_t HOUR_MS         = 60LL * MINUTE_MS;
static constexpr int64_t DAY_MS          = 24LL * HOUR_MS;
static constexpr int64_t SESSION_GAP_MS  = 30LL * MINUTE_MS;
static constexpr int64_t CLICK_WINDOW_MS = 7LL * DAY_MS;

static constexpr uint64_t FALLBACK_BIT = 1ULL << 63;
static constexpr uint64_t COMP_BIT     = 1ULL << 55; // bits 63..56 == 0, bit 55 == 1 => linked comp_id!

// Anonymous mmap with optional 2MB-aligned MADV_HUGEPAGE (ONLY on actively-used >= 2 MB slabs/tables!)
static inline void* alloc_mmap(size_t bytes, bool hugepage = false) {
    if (bytes == 0) bytes = 4096;
#ifdef MADV_HUGEPAGE
    if (hugepage && bytes >= (2ULL << 20)) {
        constexpr size_t HP = 2ULL << 20;
        size_t aligned_bytes = (bytes + HP - 1) & ~(HP - 1);
        size_t req = aligned_bytes + HP;
        void* raw = ::mmap(nullptr, req, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw == MAP_FAILED) {
            std::fprintf(stderr, "mmap failed for %zu bytes\n", req);
            std::abort();
        }
        uintptr_t addr = reinterpret_cast<uintptr_t>(raw);
        uintptr_t aligned = (addr + HP - 1) & ~(HP - 1);
        void* p = reinterpret_cast<void*>(aligned);
        // Zero munmap() calls (untouched head/tail VA costs 0 physical RAM and avoids mmap_write_lock + TLB shootdowns!)
        ::madvise(p, aligned_bytes, MADV_HUGEPAGE);
        return p;
    }
#else
    (void)hugepage;
#endif
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        std::fprintf(stderr, "mmap failed for %zu bytes\n", bytes);
        std::abort();
    }
    return p;
}

static inline size_t alloc_mmap_actual_size(size_t bytes, bool hugepage) {
    if (bytes == 0) return 4096;
#ifdef MADV_HUGEPAGE
    if (hugepage && bytes >= (2ULL << 20)) {
        constexpr size_t HP = 2ULL << 20;
        return (bytes + HP - 1) & ~(HP - 1);
    }
#else
    (void)hugepage;
#endif
    return (bytes + 4095) & ~size_t(4095);
}

struct StringArenaPool {
    static constexpr int MAX_THREADS = 64;
    static constexpr uint32_t MAX_SEGS = 8192;
    static constexpr uint32_t ARENA_SEG_BASE = MAX_SEGS - MAX_THREADS; // 8064..8127? Wait: 8192 - 64 = 8128!
    static constexpr size_t   MAX_SEG_OFF = (1ULL << 34) - 1ULL;       // 16 GiB per segment
    static constexpr uint32_t MAX_EXT_LEN = 65535U;
    static constexpr size_t   RESERVATION = 2ULL << 30;                // 2 GiB VA (0 physical RAM until touched!)

    struct HugeStr {
        const char* ptr;
        size_t len;
    };

    const char* seg_base[MAX_SEGS]{};
    std::atomic<uint32_t> next_mmap_seg{0};
    char* base[MAX_THREADS]{};
    size_t pos[MAX_THREADS]{};
    mutable std::mutex huge_mu;
    std::vector<HugeStr> huge_strs;

    void init(int nthreads) {
        for (int t = 0; t < nthreads; ++t) {
            base[t] = static_cast<char*>(alloc_mmap(RESERVATION, false));
            seg_base[ARENA_SEG_BASE + uint32_t(t)] = base[t];
            pos[t] = 0;
        }
    }

    uint32_t register_mmap_seg(const char* ptr) {
        if (!ptr) return UINT32_MAX;
        uint32_t id = next_mmap_seg.fetch_add(1, std::memory_order_relaxed);
        if (id >= ARENA_SEG_BASE) return UINT32_MAX;
        seg_base[id] = ptr;
        return id;
    }

    SS_COLD_NOINLINE uint64_t store_huge(const char* ptr, size_t len) {
        std::lock_guard<std::mutex> lk(huge_mu);
        huge_strs.push_back({ptr, len});
        uint64_t idx1 = uint64_t(huge_strs.size());
        return FALLBACK_BIT | (idx1 << 16); // low 16 bits == 0 => huge descriptor!
    }

    SS_COLD_NOINLINE uint64_t store_fallback(int tid, const char* s, size_t len) {
        size_t off = pos[tid];
        if (__builtin_expect(off + len > RESERVATION, 0)) {
            std::fprintf(stderr, "StringArena overflow on thread %d (off=%zu len=%zu)\n", tid, off, len);
            std::abort();
        }
        if (len) std::memcpy(base[tid] + off, s, len);
        pos[tid] = off + len;
        if (__builtin_expect(len >= 1 && len <= MAX_EXT_LEN && off <= MAX_SEG_OFF, 1)) {
            return FALLBACK_BIT | (uint64_t(ARENA_SEG_BASE + uint32_t(tid)) << 50) | (uint64_t(off) << 16) | uint64_t(len);
        }
        return store_huge(base[tid] + off, len);
    }

    // Zero-copy KeyRef pointing directly into a persistent read-only mmap segment!
    SS_ALWAYS_INLINE uint64_t make_mmap_ref(uint32_t seg_id, size_t off, const char* s, size_t len, int tid) {
        if (__builtin_expect(seg_id < ARENA_SEG_BASE && off <= MAX_SEG_OFF && len >= 1 && len <= MAX_EXT_LEN, 1)) {
            return FALLBACK_BIT | (uint64_t(seg_id) << 50) | (uint64_t(off) << 16) | uint64_t(len);
        }
        if (seg_id < ARENA_SEG_BASE) return store_huge(s, len);
        return store_fallback(tid, s, len);
    }

    SS_ALWAYS_INLINE std::string_view get_fallback(uint64_t k) const {
        uint32_t len = uint32_t(k & 0xFFFFU);
        if (__builtin_expect(len != 0, 1)) {
            uint32_t seg = uint32_t((k >> 50) & 0x1FFFU);
            size_t   off = size_t((k >> 16) & 0x3FFFFFFFFULL);
            return {seg_base[seg] + off, len};
        }
        uint32_t idx1 = uint32_t((k >> 16) & 0xFFFFFFFFULL);
        std::lock_guard<std::mutex> lk(huge_mu);
        const auto& hs = huge_strs[idx1 - 1];
        return {hs.ptr, hs.len};
    }
};

// ============================================================================
// BMI2 PEXT/PDEP 56-bit (7-bit ASCII) & 48-bit (6-bit Base36) Order-Preserving Codecs
// ============================================================================
static SS_ALWAYS_INLINE uint64_t pext56(uint64_t be) {
#if defined(__BMI2__)
    return _pext_u64(be, 0x7F7F7F7F7F7F7F7FULL);
#else
    uint64_t t = ((be & 0x7F007F007F007F00ULL) >> 1) | (be & 0x007F007F007F007FULL);
    t = ((t & 0x3FFF00003FFF0000ULL) >> 2) | (t & 0x00003FFF00003FFFULL);
    return ((t & 0x0FFFFFFF00000000ULL) >> 4) | (t & 0x000000000FFFFFFFULL);
#endif
}

static SS_ALWAYS_INLINE uint64_t pdep56(uint64_t k) {
#if defined(__BMI2__)
    return _pdep_u64(k, 0x7F7F7F7F7F7F7F7FULL);
#else
    uint64_t t = ((k & 0x00FFFFFFF0000000ULL) << 4) | (k & 0x000000000FFFFFFFULL);
    t = ((t & 0x0FFFC0000FFFC000ULL) << 2) | (t & 0x00003FFF00003FFFULL);
    return ((t & 0x3F803F803F803F80ULL) << 1) | (t & 0x007F007F007F007FULL);
#endif
}

static SS_ALWAYS_INLINE uint64_t pext48(uint64_t v) {
#if defined(__BMI2__)
    return _pext_u64(v, 0x3F3F3F3F3F3F3F3FULL);
#else
    uint64_t t = ((v & 0x3F003F003F003F00ULL) >> 2) | (v & 0x003F003F003F003FULL);
    t = ((t & 0x0FFF00000FFF0000ULL) >> 4) | (t & 0x00000FFF00000FFFULL);
    return ((t & 0x00FFFFFF00000000ULL) >> 8) | (t & 0x0000000000FFFFFFULL);
#endif
}

static SS_ALWAYS_INLINE uint64_t pdep48(uint64_t k) {
#if defined(__BMI2__)
    return _pdep_u64(k, 0x3F3F3F3F3F3F3F3FULL);
#else
    uint64_t t = ((k & 0x0000FFFFFF000000ULL) << 8) | (k & 0x0000000000FFFFFFULL);
    t = ((t & 0x00FFF00000FFF000ULL) << 4) | (t & 0x00000FFF00000FFFULL);
    return ((t & 0x0FC00FC00FC00FC0ULL) << 2) | (t & 0x003F003F003F003FULL);
#endif
}

// Exact 48-bit bijection (1-mul + 2 xor-shifts mod 2^48): mix48(a)==mix48(b) <=> a==b!
static constexpr uint64_t MASK48 = (1ULL << 48) - 1ULL;
static SS_ALWAYS_INLINE uint64_t mix48(uint64_t x) {
    x ^= x >> 24;
    x = (x * 0xbf58476d1ce5ULL) & MASK48;
    x ^= x >> 24;
    return x;
}
static SS_ALWAYS_INLINE uint64_t unmix48(uint64_t x) {
    x ^= x >> 24;
    x = (x * 0x30b52c8740edULL) & MASK48;
    x ^= x >> 24;
    return x;
}

// Ultra-fast 4-instruction validator for 2..9 byte IDs starting with `pfx` whose active bytes
// lie in [0x30..0x3F] U [0x60..0x7F] (covers 100% of base36 '0'-'9','a'-'z'; rejects \0, 'A'-'Z', UTF-8!).
static SS_ALWAYS_INLINE bool encode_eid48_padded(const char* s, size_t len, uint64_t& out_eid48) {
    if (__builtin_expect((len - 2) <= 7 && s[0] == 'e', 1)) {
        uint64_t raw8;
        std::memcpy(&raw8, s + 1, 8);
        uint64_t mask = (~0ULL) >> ((9 - len) << 3);
        raw8 &= mask;
        uint64_t check = raw8 | (~mask & 0x3030303030303030ULL);
        uint64_t bad = ((check ^ 0x2020202020202020ULL) & 0xA0A0A0A0A0A0A0A0ULL)
                     | (~(check | (check >> 2)) & 0x1010101010101010ULL);
        if (__builtin_expect(bad == 0, 1)) {
            uint64_t x = ((raw8 & 0x4040404040404040ULL) >> 1) | (raw8 & 0x1F1F1F1F1F1F1F1FULL);
            out_eid48 = pext48(x);
            return true;
        }
    }
    return false;
}

// Zero-bswap little-endian 48-bit uid encoder (pid_name decodes directly via clzll, never order-compared!)
static SS_ALWAYS_INLINE bool encode_uid48_padded(const char* s, size_t len, uint64_t& out_uid48) {
    if (__builtin_expect((len - 2) <= 7 && s[0] == 'u', 1)) {
        uint64_t raw8;
        std::memcpy(&raw8, s + 1, 8);
        uint64_t mask = (~0ULL) >> ((9 - len) << 3);
        raw8 &= mask;
        uint64_t check = raw8 | (~mask & 0x3030303030303030ULL);
        uint64_t bad = ((check ^ 0x2020202020202020ULL) & 0xA0A0A0A0A0A0A0A0ULL)
                     | (~(check | (check >> 2)) & 0x1010101010101010ULL);
        if (__builtin_expect(bad == 0, 1)) {
            uint64_t x = ((raw8 & 0x4040404040404040ULL) >> 1) | (raw8 & 0x1F1F1F1F1F1F1F1FULL);
            out_uid48 = pext48(x);
            return true;
        }
    }
    return false;
}

// Decode little-endian uid48 directly into ASCII bytes ('u' + 1..8 chars), returns exact length in 2..9!
static SS_ALWAYS_INLINE size_t decode_uid48_inline(uint64_t uid48, char* out) {
    out[0] = 'u';
    uint64_t u = pdep48(uid48);
    uint64_t b5 = u & 0x2020202020202020ULL;
    uint64_t le = u | (b5 << 1) | ((~b5) & ((u & 0x1010101010101010ULL) << 1));
    std::memcpy(out + 1, &le, 8);
    return le ? (9U - (uint32_t(__builtin_clzll(le)) >> 3)) : 1U;
}

// Exact injective 128-bit encoder for PFX ('u' or 'e') + 32 lowercase hex chars (AVX2 ~12 uops + identical scalar!).
// Domain predicate depends ONLY on decoded bytes (disjoint from 2..9B inline codecs by length 33).
template <char PFX>
static SS_ALWAYS_INLINE bool encode_hex128(const char* s, size_t len, uint64_t& lo, uint64_t& hi) {
    if (__builtin_expect(len != 33 || s[0] != PFX, 0)) return false;
#if defined(__AVX2__)
    __m256i v   = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + 1));
    __m256i d   = _mm256_sub_epi8(v, _mm256_set1_epi8('0'));
    __m256i a   = _mm256_sub_epi8(v, _mm256_set1_epi8('a'));
    __m256i isd = _mm256_cmpeq_epi8(_mm256_min_epu8(d, _mm256_set1_epi8(9)), d);
    __m256i isa = _mm256_cmpeq_epi8(_mm256_min_epu8(a, _mm256_set1_epi8(5)), a);
    if (__builtin_expect(uint32_t(_mm256_movemask_epi8(_mm256_or_si256(isd, isa))) != 0xFFFFFFFFU, 0)) return false;
    __m256i nib = _mm256_blendv_epi8(_mm256_add_epi8(a, _mm256_set1_epi8(10)), d, isd);
    __m256i w   = _mm256_maddubs_epi16(nib, _mm256_set1_epi16(0x0110));
    __m256i p   = _mm256_permute4x64_epi64(_mm256_packus_epi16(w, w), 0xD8);
    __m128i k   = _mm256_castsi256_si128(p);
    lo = uint64_t(_mm_cvtsi128_si64(k));
    hi = uint64_t(_mm_extract_epi64(k, 1));
    return true;
#else
    uint8_t out[16];
    for (int i = 0; i < 16; ++i) {
        unsigned c0 = uint8_t(s[1 + 2 * i]);
        unsigned c1 = uint8_t(s[2 + 2 * i]);
        unsigned d0 = c0 - uint8_t('0'), a0 = c0 - uint8_t('a');
        unsigned d1 = c1 - uint8_t('0'), a1 = c1 - uint8_t('a');
        unsigned n0 = (d0 <= 9U) ? d0 : ((a0 <= 5U) ? (a0 + 10U) : 16U);
        unsigned n1 = (d1 <= 9U) ? d1 : ((a1 <= 5U) ? (a1 + 10U) : 16U);
        if (__builtin_expect((n0 | n1) >= 16U, 0)) return false;
        out[i] = uint8_t((n0 << 4) | n1);
    }
    std::memcpy(&lo, out, 8);
    std::memcpy(&hi, out + 8, 8);
    return true;
#endif
}

static SS_ALWAYS_INLINE bool encode_uhex128(const char* s, size_t len, uint64_t& lo, uint64_t& hi) {
    return encode_hex128<'u'>(s, len, lo, hi);
}

static SS_ALWAYS_INLINE bool encode_ehex128(const char* s, size_t len, uint64_t& lo, uint64_t& hi) {
    return encode_hex128<'e'>(s, len, lo, hi);
}

static SS_COLD_NOINLINE void decode_uhex128(uint64_t lo, uint64_t hi, char out[33]) {
    static constexpr char HEX[16] = {'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'};
    uint8_t b[16];
    std::memcpy(b, &lo, 8);
    std::memcpy(b + 8, &hi, 8);
    out[0] = 'u';
    for (int i = 0; i < 16; ++i) {
        out[1 + 2 * i] = HEX[b[i] >> 4];
        out[2 + 2 * i] = HEX[b[i] & 15U];
    }
}

static SS_ALWAYS_INLINE uint64_t hash_uhex128(uint64_t lo, uint64_t hi) {
    unsigned __int128 r = static_cast<unsigned __int128>(lo ^ 0xa0761d6478bd642fULL) *
                          static_cast<unsigned __int128>(hi ^ 0xe7037ed1a0b428dbULL);
    return uint64_t(r) ^ uint64_t(r >> 64);
}

static SS_COLD_NOINLINE uint64_t encode_key64_slow(const char* s, size_t len, StringArenaPool& arenas, int tid) {
    if (len == 0) return 0;
    if (len == 1) {
        uint8_t b0 = uint8_t(s[0]);
        if (uint8_t(b0 - 1) < 127) return uint64_t(b0) << 56;
    } else if (len <= 9) {
        uint8_t b0 = uint8_t(s[0]);
        if (uint8_t(b0 - 1) < 127) {
            uint8_t tmp[8]{};
            bool ok = true;
            for (size_t i = 1; i < len; ++i) {
                uint8_t c = uint8_t(s[i]);
                if (uint8_t(c - 1) >= 127) { ok = false; break; }
                tmp[i - 1] = c;
            }
            if (ok) {
                uint64_t raw8;
                std::memcpy(&raw8, tmp, 8);
                return (uint64_t(b0) << 56) | pext56(__builtin_bswap64(raw8));
            }
        }
    }
    return arenas.store_fallback(tid, s, len);
}

// Branchless 2..9 byte order-preserving Key64 encoder, or zero-copy mmap KeyRef for long/UTF-8 IDs!
static SS_ALWAYS_INLINE uint64_t encode_keyref_padded(
    const char* s, size_t len, uint32_t seg_id, const char* seg_base_ptr, StringArenaPool& arenas, int tid
) {
    if (__builtin_expect((len - 2) <= 7, 1)) {
        uint8_t b0 = uint8_t(s[0]);
        uint64_t raw8;
        std::memcpy(&raw8, s + 1, 8);
        uint64_t mask = (~0ULL) >> ((9 - len) << 3);
        raw8 &= mask;
        uint64_t check = raw8 | (~mask & 0x0101010101010101ULL);
        uint64_t bad = (check & 0x8080808080808080ULL) |
                       ((check - 0x0101010101010101ULL) & ~check & 0x8080808080808080ULL);
        if (__builtin_expect((bad == 0) & (uint8_t(b0 - 1) < 127), 1)) {
            return (uint64_t(b0) << 56) | pext56(__builtin_bswap64(raw8));
        }
    }
    if (__builtin_expect(len == 1, 0)) {
        uint8_t b0 = uint8_t(s[0]);
        if (uint8_t(b0 - 1) < 127) return uint64_t(b0) << 56;
    }
    if (__builtin_expect(len == 0, 0)) return 0;
    if (__builtin_expect(seg_id < StringArenaPool::ARENA_SEG_BASE, 1)) {
        return arenas.make_mmap_ref(seg_id, size_t(s - seg_base_ptr), s, len, tid);
    }
    return arenas.store_fallback(tid, s, len);
}

// Branchless 2..9 byte order-preserving Key64 encoder (requires >= 8 readable pad bytes after s+1!).
static SS_ALWAYS_INLINE uint64_t encode_key64_padded(const char* s, size_t len, StringArenaPool& arenas, int tid) {
    if (__builtin_expect((len - 2) <= 7, 1)) {
        uint8_t b0 = uint8_t(s[0]);
        uint64_t raw8;
        std::memcpy(&raw8, s + 1, 8);
        uint64_t mask = (~0ULL) >> ((9 - len) << 3);
        raw8 &= mask;
        uint64_t check = raw8 | (~mask & 0x0101010101010101ULL);
        uint64_t bad = (check & 0x8080808080808080ULL) |
                       ((check - 0x0101010101010101ULL) & ~check & 0x8080808080808080ULL);
        if (__builtin_expect((bad == 0) & (uint8_t(b0 - 1) < 127), 1)) {
            return (uint64_t(b0) << 56) | pext56(__builtin_bswap64(raw8));
        }
    }
    return encode_key64_slow(s, len, arenas, tid);
}

// Decode inline Key64 (writes 9 bytes into `out`, returns exact length in 1..9).
static SS_ALWAYS_INLINE size_t decode_inline(uint64_t k, char* out) {
    out[0] = char(k >> 56);
    uint64_t be = pdep56(k);
    uint64_t le = __builtin_bswap64(be);
    std::memcpy(out + 1, &le, 8);
    return be ? (9U - (uint32_t(__builtin_ctzll(be)) >> 3)) : 1U;
}

static SS_ALWAYS_INLINE std::string_view key_to_sv(uint64_t k, char tmp[10], const StringArenaPool& arenas) {
    if (__builtin_expect((k & FALLBACK_BIT) == 0, 1)) {
        size_t n = decode_inline(k, tmp);
        return {tmp, n};
    }
    return arenas.get_fallback(k);
}

static SS_ALWAYS_INLINE uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static SS_ALWAYS_INLINE uint64_t wymix(uint64_t a, uint64_t b) {
    unsigned __int128 r = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
    return uint64_t(r) ^ uint64_t(r >> 64);
}

// Ultra-fast parallel 128-bit-mul hash (3 muls total for ALL 16..40B strings including 33B hex IDs!)
static SS_ALWAYS_INLINE uint64_t hash_bytes(const char* s, size_t len) {
    constexpr uint64_t K0 = 0xa0761d6478bd642fULL;
    constexpr uint64_t K1 = 0xe7037ed1a0b428dbULL;
    constexpr uint64_t K2 = 0x8ebc6af09c88c6e3ULL;
    constexpr uint64_t K3 = 0x589965cc75374cc3ULL;
    if (__builtin_expect(len >= 16 && len <= 40, 1)) {
        uint64_t w0, w1, w2, w3;
        std::memcpy(&w0, s, 8);
        std::memcpy(&w1, s + 8, 8);
        std::memcpy(&w2, s + len - 16, 8);
        std::memcpy(&w3, s + len - 8, 8);
        uint64_t a = wymix(w0 ^ K0, w2 ^ K1);
        uint64_t b = wymix(w1 ^ K2, w3 ^ K3);
        if (__builtin_expect(len > 32, 1)) {
            uint64_t wm;
            std::memcpy(&wm, s + 16, 8);
            a ^= wymix(wm ^ K1, K3);
        }
        return wymix(a ^ len, b ^ K0);
    }
    if (len >= 8) {
        if (len < 16) {
            uint64_t w0, w1;
            std::memcpy(&w0, s, 8);
            std::memcpy(&w1, s + len - 8, 8);
            return wymix(w0 ^ K0, w1 ^ K1 ^ len);
        }
        uint64_t h0 = K0 ^ len, h1 = K1;
        for (size_t i = 0; i + 16 <= len; i += 16) {
            uint64_t w0, w1;
            std::memcpy(&w0, s + i, 8);
            std::memcpy(&w1, s + i + 8, 8);
            h0 = wymix(h0 ^ w0, K2);
            h1 = wymix(h1 ^ w1, K3);
        }
        uint64_t t0, t1;
        std::memcpy(&t0, s + len - 16, 8);
        std::memcpy(&t1, s + len - 8, 8);
        return wymix(h0 ^ t0, h1 ^ t1);
    }
    if (len >= 4) {
        uint32_t w0, w1;
        std::memcpy(&w0, s, 4);
        std::memcpy(&w1, s + len - 4, 4);
        return wymix(uint64_t(w0) ^ K0, (uint64_t(w1) << 32) ^ K1 ^ len);
    }
    if (len > 0) {
        uint64_t w = (uint64_t(uint8_t(s[0])) << 16) | (uint64_t(uint8_t(s[len >> 1])) << 8) | uint64_t(uint8_t(s[len - 1]));
        return wymix(w ^ K0, K1 ^ len);
    }
    return K0;
}

static SS_ALWAYS_INLINE uint64_t hash_key64(uint64_t k, const StringArenaPool& arenas) {
    if (__builtin_expect((k & FALLBACK_BIT) == 0, 1)) {
        return mix64(k);
    }
    auto sv = arenas.get_fallback(k);
    return hash_bytes(sv.data(), sv.size());
}

// Ultra-fast 1-multiply hash for Stage 3 local person hash table (`pht`, 4K-16K slots!)
static SS_ALWAYS_INLINE size_t hash_pkey_local(uint64_t pk, const StringArenaPool& arenas) {
    if (__builtin_expect((pk & FALLBACK_BIT) == 0, 1)) {
        return size_t(((pk ^ (pk >> 9)) * 0x9e3779b97f4a7c15ULL) >> 40);
    }
    auto sv = arenas.get_fallback(pk);
    return size_t(hash_bytes(sv.data(), sv.size()));
}

static SS_ALWAYS_INLINE bool key_eq_raw(uint64_t k, const char* s, size_t len, const StringArenaPool& arenas) {
    if (__builtin_expect((k & FALLBACK_BIT) != 0, 1)) {
        auto sv = arenas.get_fallback(k);
        return sv.size() == len && (len == 0 || std::memcmp(sv.data(), s, len) == 0);
    }
    char tmp[10];
    size_t n = decode_inline(k, tmp);
    return n == len && std::memcmp(tmp, s, n) == 0;
}

static SS_ALWAYS_INLINE bool key_eq(uint64_t a, uint64_t b, const StringArenaPool& arenas) {
    if (__builtin_expect(a == b, 1)) return true;
    if (__builtin_expect(((a & b) & FALLBACK_BIT) == 0, 1)) return false;
    auto sa = arenas.get_fallback(a);
    auto sb = arenas.get_fallback(b);
    return sa.size() == sb.size() && (sa.empty() || std::memcmp(sa.data(), sb.data(), sa.size()) == 0);
}

static SS_NOINLINE bool key_less_fallback(uint64_t a, uint64_t b, const StringArenaPool& arenas) {
    if (a == b) return false;
    char ba[10], bb[10];
    auto sa = key_to_sv(a, ba, arenas);
    auto sb = key_to_sv(b, bb, arenas);
    size_t m = sa.size() < sb.size() ? sa.size() : sb.size();
    int cmp = m ? std::memcmp(sa.data(), sb.data(), m) : 0;
    if (cmp != 0) return cmp < 0;
    return sa.size() < sb.size();
}

static SS_ALWAYS_INLINE bool key_less(uint64_t a, uint64_t b, const StringArenaPool& arenas) {
    if (__builtin_expect(((a | b) & FALLBACK_BIT) == 0, 1)) {
        return a < b;
    }
    return key_less_fallback(a, b, arenas);
}

// ============================================================================
// Calendar + 4 KB L1D FastDay[256] LUT + 64-bit SWAR Timestamp Parsers
// ============================================================================
static inline bool is_leap_year(int64_t y) {
    return (y % 4 == 0) && ((y % 100 != 0) || (y % 400 == 0));
}

static inline int64_t days_from_civil(int64_t y, int64_t m, int64_t d) {
    y -= (m <= 2);
    int64_t era = (y >= 0 ? y : y - 399) / 400;
    int64_t yoe = y - era * 400;
    int64_t doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}

static inline void civil_from_days(int64_t z, int& y, int& m, int& d) {
    z += 719468;
    int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    int64_t doe = z - era * 146097;
    int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    int64_t yr = yoe + era * 400;
    int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    int64_t mp = (5 * doy + 2) / 153;
    d = int(doy - (153 * mp + 2) / 5 + 1);
    m = int(mp < 10 ? mp + 3 : mp - 9);
    y = int(yr + (m <= 2));
}

static constexpr uint8_t DAYS_IN_MONTH[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

static SS_ALWAYS_INLINE bool parse_2d(const char* p, int& out) {
    unsigned d0 = uint8_t(p[0]) - uint8_t('0');
    unsigned d1 = uint8_t(p[1]) - uint8_t('0');
    if (__builtin_expect((d0 > 9) | (d1 > 9), 0)) return false;
    out = int(d0 * 10 + d1);
    return true;
}

static SS_ALWAYS_INLINE bool parse_4d(const char* p, int& out) {
    unsigned d0 = uint8_t(p[0]) - uint8_t('0');
    unsigned d1 = uint8_t(p[1]) - uint8_t('0');
    unsigned d2 = uint8_t(p[2]) - uint8_t('0');
    unsigned d3 = uint8_t(p[3]) - uint8_t('0');
    if (__builtin_expect((d0 > 9) | (d1 > 9) | (d2 > 9) | (d3 > 9), 0)) return false;
    out = int(d0 * 1000 + d1 * 100 + d2 * 10 + d3);
    return true;
}

struct FastDay {
    uint64_t ym;
    uint16_t dd16;
    uint16_t _pad;
    int32_t  rel_sec;
};

struct DateLut {
    static constexpr int NUM_YEARS = 8;
    static constexpr int32_t INVALID = INT32_MIN;
    FastDay days[256]{};
    int64_t win_start_sec = 0;
    int base_year = -1000000;
    int32_t table[NUM_YEARS * 16 * 32]{};

    void init(int64_t click_start_ms, int64_t win_start_ms, int64_t win_end_ms) {
        win_start_sec = win_start_ms / 1000LL;
        int64_t d_start = (click_start_ms / DAY_MS) - 2;
        int64_t d_end = (win_end_ms / DAY_MS) + 10;
        int y1, m1, dy1;
        civil_from_days(d_start, y1, m1, dy1);
        base_year = y1 - 1;
        for (int i = 0; i < NUM_YEARS * 16 * 32; ++i) table[i] = INVALID;
        for (int yo = 0; yo < NUM_YEARS; ++yo) {
            int y = base_year + yo;
            bool leap = is_leap_year(y);
            for (int m = 1; m <= 12; ++m) {
                int maxd = (m == 2 && leap) ? 29 : DAYS_IN_MONTH[m];
                for (int d = 1; d <= maxd; ++d) {
                    table[(yo << 9) | (m << 5) | d] = int32_t(days_from_civil(y, m, d));
                }
            }
        }
        for (auto& e : days) { e.ym = 0; e.dd16 = 0; e._pad = 0; e.rel_sec = INT32_MIN; }
        if (d_end - d_start <= 110) {
            for (int64_t d = d_start; d <= d_end; ++d) {
                int y, m, dy;
                civil_from_days(d, y, m, dy);
                if (y < 0 || y > 9999) continue;
                char b[10];
                b[0] = char('0' + (y / 1000) % 10);
                b[1] = char('0' + (y / 100) % 10);
                b[2] = char('0' + (y / 10) % 10);
                b[3] = char('0' + y % 10);
                b[4] = '-';
                b[5] = char('0' + m / 10);
                b[6] = char('0' + m % 10);
                b[7] = '-';
                b[8] = char('0' + dy / 10);
                b[9] = char('0' + dy % 10);
                uint32_t idx = ((uint8_t(b[6]) & 3U) << 6) | ((uint8_t(b[8]) & 3U) << 4) | (uint8_t(b[9]) & 15U);
                uint64_t ym; uint16_t dd16;
                std::memcpy(&ym, b, 8);
                std::memcpy(&dd16, b + 8, 2);
                int64_t rel = d * 86400LL - win_start_sec;
                if (rel < INT32_MIN + 1 || rel > INT32_MAX) continue;
                if (days[idx].ym == 0) {
                    days[idx] = {ym, dd16, 0, int32_t(rel)};
                } else {
                    days[idx].ym = ~0ULL;
                }
            }
        }
    }

    SS_COLD_NOINLINE bool lookup_days_slow(int y, int m, int d, int64_t& out_days) const {
        unsigned yo = unsigned(y - base_year);
        if (__builtin_expect(yo < NUM_YEARS && unsigned(m) < 16 && unsigned(d) < 32, 1)) {
            int32_t v = table[(yo << 9) | (m << 5) | d];
            if (__builtin_expect(v != INVALID, 1)) { out_days = v; return true; }
            return false;
        }
        if (m < 1 || m > 12) return false;
        int maxd = (m == 2 && is_leap_year(y)) ? 29 : DAYS_IN_MONTH[m];
        if (d < 1 || d > maxd) return false;
        out_days = days_from_civil(y, m, d);
        return true;
    }

    SS_COLD_NOINLINE bool parse_date_rel_slow(const char* s, int64_t& out_day_rel_sec) const {
        if (s[4] != '-' || s[7] != '-') return false;
        int y, mo, dd;
        if (!parse_4d(s, y) || !parse_2d(s + 5, mo) || !parse_2d(s + 8, dd)) return false;
        int64_t d;
        if (!lookup_days_slow(y, mo, dd, d)) return false;
        out_day_rel_sec = d * 86400LL - win_start_sec;
        return true;
    }

    static SS_ALWAYS_INLINE bool parse_hms_swar(const char* s, uint32_t& out_hms) {
        if (__builtin_expect(s[10] != 'T', 0)) return false;
        uint64_t w;
        std::memcpy(&w, s + 11, 8);
        uint64_t hi = w & 0x8080808080808080ULL;
        uint64_t t = (w | 0x8080808080808080ULL) - 0x30303A30303A3030ULL;
        uint64_t d = t & 0x7F7F7F7F7F7F7F7FULL;
        uint64_t u = d + 0x767A7F767A7F767DULL;
        uint64_t bad = hi | ((~t | u) & 0x8080808080808080ULL);
        uint64_t p = d * 10ULL + (d >> 8);
        uint32_t hh = uint32_t(p & 0xFFU);
        if (__builtin_expect((bad != 0) | (hh > 23U), 0)) return false;
        uint32_t mm = uint32_t((p >> 24) & 0xFFU);
        uint32_t ss = uint32_t((p >> 48) & 0xFFU);
        out_hms = hh * 3600U + mm * 60U + ss;
        return true;
    }

    // Trusted F1 parser for ingest_ts (rules.md: "ingest_ts is always valid F1") - zero extra live state!
    SS_ALWAYS_INLINE bool parse_ingest_f1_rel_sec(const char* s, int64_t& out_rel_sec) const {
        uint64_t w;
        std::memcpy(&w, s + 11, 8);
        uint64_t d = w - 0x30303A30303A3030ULL;
        uint64_t p = d * 10ULL + (d >> 8);
        uint32_t hms = uint32_t(p & 0xFFU) * 3600U + uint32_t((p >> 24) & 0xFFU) * 60U + uint32_t((p >> 48) & 0xFFU);
        uint64_t ym; uint16_t dd16;
        std::memcpy(&ym, s, 8);
        std::memcpy(&dd16, s + 8, 2);
        uint32_t idx = (((uint32_t(ym >> 48)) & 3U) << 6) | ((uint32_t(dd16) & 3U) << 4) | ((uint32_t(dd16 >> 8)) & 15U);
        const FastDay& fd = days[idx];
        if (__builtin_expect((fd.ym == ym) & (fd.dd16 == dd16), 1)) {
            out_rel_sec = int64_t(fd.rel_sec) + int64_t(hms);
            return true;
        }
        int64_t day_rel;
        if (!parse_date_rel_slow(s, day_rel)) return false;
        out_rel_sec = day_rel + int64_t(hms);
        return true;
    }

    // Branchless F1 (20B 'Z') + F2 (25B '±HH:MM') ISO parser: zero same_day mispredict & zero noinline F2 call!
    // Caller guarantees >= 25 readable bytes at `s` (mmap guard page / padded ReadBuf).
    SS_ALWAYS_INLINE bool parse_iso_f12_rel_sec(const char* s, size_t len, int64_t& out_rel_sec) const {
        uint32_t hms;
        if (__builtin_expect(!parse_hms_swar(s, hms), 0)) return false;
        uint64_t ym; uint16_t dd16;
        std::memcpy(&ym, s, 8);
        std::memcpy(&dd16, s + 8, 2);
        uint32_t idx = (((uint32_t(ym >> 48)) & 3U) << 6) | ((uint32_t(dd16) & 3U) << 4) | ((uint32_t(dd16 >> 8)) & 15U);
        const FastDay& fd = days[idx];
        int64_t day_rel;
        if (__builtin_expect((fd.ym == ym) & (fd.dd16 == dd16), 1)) {
            day_rel = int64_t(fd.rel_sec);
        } else {
            if (!parse_date_rel_slow(s, day_rel)) return false;
        }
        uint8_t c19 = uint8_t(s[19]);
        unsigned oh0 = uint8_t(s[20]) - uint8_t('0'), oh1 = uint8_t(s[21]) - uint8_t('0');
        unsigned om0 = uint8_t(s[23]) - uint8_t('0'), om1 = uint8_t(s[24]) - uint8_t('0');
        unsigned oh = oh0 * 10U + oh1, om = om0 * 10U + om1;
        bool is_f1 = (len == 20) & (c19 == 'Z');
        bool is_f2 = (len == 25) & ((c19 == '+') | (c19 == '-')) & (s[22] == ':') &
                     (oh0 <= 9U) & (oh1 <= 9U) & (om0 <= 9U) & (om1 <= 9U) & (oh <= 23U) & (om <= 59U);
        if (__builtin_expect(!(is_f1 | is_f2), 0)) return false;
        int64_t off_mag = int64_t(oh * 3600U + om * 60U);
        int64_t off_s = is_f2 ? ((c19 == '+') ? -off_mag : off_mag) : 0LL;
        out_rel_sec = day_rel + int64_t(hms) + off_s;
        return true;
    }

    // Trusted F1 parser for ingest_ts (rules.md: "ingest_ts is always valid F1") - 8 insns!
    SS_ALWAYS_INLINE bool parse_ingest_f1_rel_sec_day(
        const char* s, int64_t& out_rel_sec, uint64_t& out_ym, uint16_t& out_dd16, int64_t& out_day_rel
    ) const {
        uint64_t w;
        std::memcpy(&w, s + 11, 8);
        uint64_t d = w - 0x30303A30303A3030ULL;
        uint64_t p = d * 10ULL + (d >> 8);
        uint32_t hms = uint32_t(p & 0xFFU) * 3600U + uint32_t((p >> 24) & 0xFFU) * 60U + uint32_t((p >> 48) & 0xFFU);
        uint64_t ym; uint16_t dd16;
        std::memcpy(&ym, s, 8);
        std::memcpy(&dd16, s + 8, 2);
        out_ym = ym;
        out_dd16 = dd16;
        uint32_t idx = (((uint32_t(ym >> 48)) & 3U) << 6) | ((uint32_t(dd16) & 3U) << 4) | ((uint32_t(dd16 >> 8)) & 15U);
        const FastDay& fd = days[idx];
        if (__builtin_expect((fd.ym == ym) & (fd.dd16 == dd16), 1)) {
            out_day_rel = int64_t(fd.rel_sec);
            out_rel_sec = out_day_rel + int64_t(hms);
            return true;
        }
        if (!parse_date_rel_slow(s, out_day_rel)) return false;
        out_rel_sec = out_day_rel + int64_t(hms);
        return true;
    }

    SS_ALWAYS_INLINE bool parse_iso19_rel_sec(const char* s, int64_t& out_rel_sec) const {
        uint32_t hms;
        if (__builtin_expect(!parse_hms_swar(s, hms), 0)) return false;
        uint64_t ym; uint16_t dd16;
        std::memcpy(&ym, s, 8);
        std::memcpy(&dd16, s + 8, 2);
        uint32_t idx = (((uint32_t(ym >> 48)) & 3U) << 6) | ((uint32_t(dd16) & 3U) << 4) | ((uint32_t(dd16 >> 8)) & 15U);
        const FastDay& fd = days[idx];
        if (__builtin_expect((fd.ym == ym) & (fd.dd16 == dd16), 1)) {
            out_rel_sec = int64_t(fd.rel_sec) + int64_t(hms);
            return true;
        }
        int64_t day_rel;
        if (!parse_date_rel_slow(s, day_rel)) return false;
        out_rel_sec = day_rel + int64_t(hms);
        return true;
    }

    SS_ALWAYS_INLINE bool parse_iso19_rel_sec_reuse(
        const char* s, uint64_t cached_ym, uint16_t cached_dd16, int64_t cached_day_rel, int64_t& out_rel_sec
    ) const {
        uint32_t hms;
        if (__builtin_expect(!parse_hms_swar(s, hms), 0)) return false;
        uint64_t ym; uint16_t dd16;
        std::memcpy(&ym, s, 8);
        std::memcpy(&dd16, s + 8, 2);
        if (__builtin_expect((ym == cached_ym) & (dd16 == cached_dd16), 1)) {
            out_rel_sec = cached_day_rel + int64_t(hms);
            return true;
        }
        uint32_t idx = (((uint32_t(ym >> 48)) & 3U) << 6) | ((uint32_t(dd16) & 3U) << 4) | ((uint32_t(dd16 >> 8)) & 15U);
        const FastDay& fd = days[idx];
        if (__builtin_expect((fd.ym == ym) & (fd.dd16 == dd16), 1)) {
            out_rel_sec = int64_t(fd.rel_sec) + int64_t(hms);
            return true;
        }
        int64_t day_rel;
        if (!parse_date_rel_slow(s, day_rel)) return false;
        out_rel_sec = day_rel + int64_t(hms);
        return true;
    }

    SS_ALWAYS_INLINE bool parse_f2_rel_sec(const char* s, int64_t& out_rel_sec) const {
        if (__builtin_expect(((s[19] != '+') & (s[19] != '-')) | (s[22] != ':'), 0)) return false;
        int64_t rel_s;
        int oh, om;
        if (__builtin_expect(!parse_iso19_rel_sec(s, rel_s) || !parse_2d(s + 20, oh) || !parse_2d(s + 23, om), 0)) return false;
        if (__builtin_expect((oh > 23) | (om > 59), 0)) return false;
        int64_t off_s = int64_t(oh) * 3600LL + int64_t(om) * 60LL;
        out_rel_sec = rel_s + ((s[19] == '+') ? -off_s : off_s);
        return true;
    }
};

// Zero-division 13-digit SWAR parser returning (sec, ms_rem) directly!
static SS_ALWAYS_INLINE bool parse_13d_sec_ms(const char* s, int64_t& out_sec, uint32_t& out_ms_rem) {
    uint64_t v0, w;
    std::memcpy(&v0, s, 8);
    std::memcpy(&w, s + 5, 8);
    uint64_t bad = ((v0 & 0xF0F0F0F0F0F0F0F0ULL) ^ 0x3030303030303030ULL) |
                   (((v0 + 0x0606060606060606ULL) & 0xF0F0F0F0F0F0F0F0ULL) ^ 0x3030303030303030ULL) |
                   ((w & 0xF0F0F0F0F0F0F0F0ULL) ^ 0x3030303030303030ULL) |
                   (((w + 0x0606060606060606ULL) & 0xF0F0F0F0F0F0F0F0ULL) ^ 0x3030303030303030ULL);
    if (__builtin_expect(bad != 0, 0)) return false;
    v0 &= 0x0F0F0F0F0F0F0F0FULL;
    v0 = (v0 * 10 + (v0 >> 8)) & 0x00FF00FF00FF00FFULL;
    v0 = (v0 * 100 + (v0 >> 16)) & 0x0000FFFF0000FFFFULL;
    uint64_t hi8 = (v0 * 10000 + (v0 >> 32)) & 0xFFFFFFFFULL;
    uint64_t lo = (w >> 24) & 0x0F0F0F0F0FULL;
    out_sec = int64_t(hi8 * 100ULL + uint32_t(lo & 0xFU) * 10U + uint32_t((lo >> 8) & 0xFU));
    out_ms_rem = uint32_t((lo >> 16) & 0xFU) * 100U + uint32_t((lo >> 24) & 0xFU) * 10U + uint32_t(lo >> 32);
    return true;
}

static SS_COLD_NOINLINE bool parse_ts_slow(const char* s, size_t len, int64_t& out_ms, const DateLut& lut) {
    if (len == 20 && s[19] == 'Z') {
        int64_t rel_s;
        if (!lut.parse_iso19_rel_sec(s, rel_s)) return false;
        out_ms = (lut.win_start_sec + rel_s) * 1000LL;
        return true;
    }
    if (len == 25 && (s[19] == '+' || s[19] == '-')) {
        int64_t rel_s;
        if (!lut.parse_f2_rel_sec(s, rel_s)) return false;
        out_ms = (lut.win_start_sec + rel_s) * 1000LL;
        return true;
    }
    if (len == 0) return false;
    size_t i = 0;
    while (i < len && s[i] == '0') ++i;
    if (i == len) { out_ms = 0; return true; }
    if (len - i > 19) return false;
    uint64_t v = 0;
    for (; i < len; ++i) {
        unsigned d = uint8_t(s[i]) - uint8_t('0');
        if (d > 9) return false;
        if (v > (uint64_t(INT64_MAX) - d) / 10) return false;
        v = v * 10 + d;
    }
    out_ms = int64_t(v);
    return true;
}

// 100% branchless F1 + F2 + F3 timestamp parser (eliminates ~14M random 3-way branch mispredicts!).
// Caller guarantees >= 25 readable bytes at `s` (true in guard-paged mmap and padded ReadBuf!).
// Returns true iff handled cleanly as valid in-LUT F1/F2 or valid 13-digit F3; false -> parse_ts_slow!
static SS_ALWAYS_INLINE bool parse_ts_branchless_rel(
    const char* s, size_t len,
    const DateLut& lut,
    uint64_t ing_ym, uint16_t ing_dd16, int64_t ing_day_rel,
    int64_t win_start_sec,
    int64_t& out_rel_sec, uint32_t& out_ms_rem
) {
    // Path A: F3 13-digit epoch ms (zero branches)
    uint64_t v0, w13;
    std::memcpy(&v0, s, 8);
    std::memcpy(&w13, s + 5, 8);
    uint64_t bad13 = ((v0 & 0xF0F0F0F0F0F0F0F0ULL) ^ 0x3030303030303030ULL) |
                     (((v0 + 0x0606060606060606ULL) & 0xF0F0F0F0F0F0F0F0ULL) ^ 0x3030303030303030ULL) |
                     ((w13 & 0xF0F0F0F0F0F0F0F0ULL) ^ 0x3030303030303030ULL) |
                     (((w13 + 0x0606060606060606ULL) & 0xF0F0F0F0F0F0F0F0ULL) ^ 0x3030303030303030ULL);
    v0 &= 0x0F0F0F0F0F0F0F0FULL;
    v0 = (v0 * 10 + (v0 >> 8)) & 0x00FF00FF00FF00FFULL;
    v0 = (v0 * 100 + (v0 >> 16)) & 0x0000FFFF0000FFFFULL;
    uint64_t hi8 = (v0 * 10000 + (v0 >> 32)) & 0xFFFFFFFFULL;
    uint64_t lo5 = (w13 >> 24) & 0x0F0F0F0F0FULL;
    int64_t f3_sec = int64_t(hi8 * 100ULL + uint32_t(lo5 & 0xFU) * 10U + uint32_t((lo5 >> 8) & 0xFU));
    uint32_t f3_msr = uint32_t((lo5 >> 16) & 0xFU) * 100U + uint32_t((lo5 >> 24) & 0xFU) * 10U + uint32_t(lo5 >> 32);
    int64_t f3_rel = f3_sec - win_start_sec;
    bool is_f3 = (len == 13);
    bool f3_ok = is_f3 & (bad13 == 0);

    // Path B: ISO F1 (len==20 'Z') & F2 (len==25 '±HH:MM') merged branchlessly!
    uint64_t wh;
    std::memcpy(&wh, s + 11, 8);
    uint64_t hi = wh & 0x8080808080808080ULL;
    uint64_t t  = (wh | 0x8080808080808080ULL) - 0x30303A30303A3030ULL;
    uint64_t d  = t & 0x7F7F7F7F7F7F7F7FULL;
    uint64_t u  = d + 0x767A7F767A7F767DULL;
    uint64_t bad_hms = hi | ((~t | u) & 0x8080808080808080ULL);
    uint64_t p  = d * 10ULL + (d >> 8);
    uint32_t hh = uint32_t(p & 0xFFU);
    uint32_t mm = uint32_t((p >> 24) & 0xFFU);
    uint32_t ss = uint32_t((p >> 48) & 0xFFU);
    uint32_t hms = hh * 3600U + mm * 60U + ss;
    bool hms_ok = (s[10] == 'T') & (bad_hms == 0) & (hh <= 23U);

    uint64_t ym; uint16_t dd16;
    std::memcpy(&ym, s, 8);
    std::memcpy(&dd16, s + 8, 2);
    uint32_t idx = (((uint32_t(ym >> 48)) & 3U) << 6) | ((uint32_t(dd16) & 3U) << 4) | ((uint32_t(dd16 >> 8)) & 15U);
    const FastDay& fd = lut.days[idx];
    bool same_day = (ym == ing_ym) & (dd16 == ing_dd16);
    bool lut_hit  = (fd.ym == ym) & (fd.dd16 == dd16);
    int64_t day_rel = same_day ? ing_day_rel : int64_t(fd.rel_sec);
    bool day_ok = same_day | lut_hit;

    uint8_t c19 = uint8_t(s[19]);
    uint32_t oh0 = uint8_t(s[20]) - uint8_t('0');
    uint32_t oh1 = uint8_t(s[21]) - uint8_t('0');
    uint32_t om0 = uint8_t(s[23]) - uint8_t('0');
    uint32_t om1 = uint8_t(s[24]) - uint8_t('0');
    uint32_t oh  = oh0 * 10U + oh1;
    uint32_t om  = om0 * 10U + om1;
    bool off_ok = (oh0 <= 9U) & (oh1 <= 9U) & (om0 <= 9U) & (om1 <= 9U) & (oh <= 23U) & (om <= 59U) & (s[22] == ':');
    bool is_f1 = (len == 20) & (c19 == 'Z');
    bool is_f2 = (len == 25) & ((c19 == '+') | (c19 == '-')) & off_ok;
    int64_t off_mag = int64_t(oh * 3600U + om * 60U);
    int64_t off_s = is_f2 ? ((c19 == '+') ? -off_mag : off_mag) : 0LL;
    int64_t iso_rel = day_rel + int64_t(hms) + off_s;
    bool iso_ok = (is_f1 | is_f2) & hms_ok & day_ok;

    out_rel_sec = is_f3 ? f3_rel : iso_rel;
    out_ms_rem  = is_f3 ? f3_msr : 0U;
    return is_f3 ? f3_ok : iso_ok;
}

static SS_ALWAYS_INLINE bool parse_ts_branchless_ms(const char* s, size_t len, int64_t& out_ms, const DateLut& lut) {
    int64_t rel_s = 0;
    uint32_t msr = 0;
    // Pass impossible cached_ym = ~0ULL so same_day is always false (uses lut.days directly!)
    if (__builtin_expect(parse_ts_branchless_rel(s, len, lut, ~0ULL, 0U, 0LL, lut.win_start_sec, rel_s, msr), 1)) {
        out_ms = (lut.win_start_sec + rel_s) * 1000LL + int64_t(msr);
        return true;
    }
    return parse_ts_slow(s, len, out_ms, lut);
}

// Out-of-line F2 (±HH:MM) + slow timestamp helper returning by value (zero stack-escaping out-params!)
struct TsF2SlowRes {
    int64_t rel_sec;
    int64_t ms;
    bool ok;
    bool from_sec;
};

static SS_NOINLINE TsF2SlowRes parse_ts_f2_or_slow_rel(const char* s, size_t len, const DateLut& lut) {
    if (__builtin_expect(len == 25 && (s[19] == '+' || s[19] == '-'), 1)) {
        int64_t rel_sec = 0;
        bool ok = lut.parse_f2_rel_sec(s, rel_sec);
        return {rel_sec, 0LL, ok, true};
    }
    int64_t ms = 0;
    bool ok = parse_ts_slow(s, len, ms, lut);
    return {0LL, ms, ok, false};
}

static SS_NOINLINE bool parse_ts_f2_or_slow_ms(const char* s, size_t len, int64_t& out_ms, const DateLut& lut) {
    if (__builtin_expect(len == 25 && (s[19] == '+' || s[19] == '-'), 1)) {
        int64_t rel_s;
        if (__builtin_expect(!lut.parse_f2_rel_sec(s, rel_s), 0)) return false;
        out_ms = (lut.win_start_sec + rel_s) * 1000LL;
        return true;
    }
    return parse_ts_slow(s, len, out_ms, lut);
}

static SS_ALWAYS_INLINE bool parse_ts(const char* s, size_t len, int64_t& out_ms, const DateLut& lut) {
    if (__builtin_expect(len == 20 && s[19] == 'Z', 1)) {
        int64_t rel_s;
        if (__builtin_expect(!lut.parse_iso19_rel_sec(s, rel_s), 0)) return false;
        out_ms = (lut.win_start_sec + rel_s) * 1000LL;
        return true;
    }
    if (__builtin_expect(len == 13, 1)) {
        int64_t sec; uint32_t msr;
        if (__builtin_expect(!parse_13d_sec_ms(s, sec, msr), 0)) return false;
        out_ms = sec * 1000LL + int64_t(msr);
        return true;
    }
    return parse_ts_f2_or_slow_ms(s, len, out_ms, lut);
}

static SS_ALWAYS_INLINE bool parse_price_cents(const char* s, size_t len, uint32_t& out_price) {
    if (__builtin_expect(len == 0, 0)) return false;
    uint32_t v = 0;
    for (size_t i = 0; i < len; ++i) {
        unsigned d = uint8_t(s[i]) - uint8_t('0');
        if (__builtin_expect(d > 9, 0)) return false;
        if (__builtin_expect(v > 1000000U, 0)) return false;
        v = v * 10U + d;
    }
    if (__builtin_expect(v == 0 || v > 10000000U, 0)) return false;
    out_price = v;
    return true;
}

// ============================================================================
// Multi-ISA 256-bit AVX-512VL / AVX2 / NEON Vec64 + CLMUL Zero-Load CSV Extractors
// ============================================================================
struct Slice {
    const char* ptr;
    size_t len;
};

template <int N>
static SS_COLD_NOINLINE bool parse_csv_line_scalar(const char* line, const char* end, Slice out[N]) {
    if (__builtin_expect(line == end, 0)) return false;
    const char* p = line;
    for (int f = 0; f < N; ++f) {
        const char* fs = p;
        bool has_quote = false;
        if (f < N - 1) {
            while (p < end && *p != ',') {
                has_quote |= (*p == '"');
                ++p;
            }
            if (__builtin_expect(p == end, 0)) return false;
        } else {
            while (p < end) {
                char c = *p;
                if (__builtin_expect(c == ',', 0)) return false;
                has_quote |= (c == '"');
                ++p;
            }
        }
        const char* fe = p;
        if (f < N - 1) ++p;

        if (__builtin_expect(!has_quote, 1)) {
            out[f] = {fs, size_t(fe - fs)};
        } else {
            size_t flen = size_t(fe - fs);
            if (__builtin_expect(flen < 2 || fs[0] != '"' || fe[-1] != '"', 0)) return false;
            const char* is = fs + 1;
            const char* ie = fe - 1;
            for (const char* q = is; q < ie; ++q) {
                if (__builtin_expect(*q == '"', 0)) return false;
            }
            out[f] = {is, size_t(ie - is)};
        }
    }
    return true;
}

static SS_ALWAYS_INLINE uint64_t prefix_xor(uint64_t q) {
#if defined(__PCLMUL__)
    return uint64_t(_mm_cvtsi128_si64(_mm_clmulepi64_si128(_mm_set_epi64x(0, q), _mm_set1_epi8(-1), 0)));
#elif defined(__aarch64__) && defined(__ARM_FEATURE_CRYPTO)
    return uint64_t(vmull_p64(q, ~0ULL));
#else
    q ^= q << 1; q ^= q << 2; q ^= q << 4; q ^= q << 8; q ^= q << 16; q ^= q << 32;
    return q;
#endif
}

// 256-bit AVX-512VL Vec64: uses 1-cycle %k opmask compares with ZERO 512-bit ZMM downclocking on Cascade Lake!
#if defined(__AVX512BW__) && defined(__AVX512VL__)
struct Vec64 {
    __m256i v0, v1;
    static SS_ALWAYS_INLINE Vec64 load(const char* p) {
        return {
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32))
        };
    }
    SS_ALWAYS_INLINE uint64_t eq(char ch) const {
        __m256i vc = _mm256_set1_epi8(ch);
        uint64_t m0 = uint32_t(_mm256_cmpeq_epi8_mask(v0, vc));
        uint64_t m1 = uint32_t(_mm256_cmpeq_epi8_mask(v1, vc));
        return m0 | (m1 << 32);
    }
};
#elif defined(__AVX2__)
struct Vec64 {
    __m256i v0, v1;
    static SS_ALWAYS_INLINE Vec64 load(const char* p) {
        return {
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32))
        };
    }
    SS_ALWAYS_INLINE uint64_t eq(char ch) const {
        __m256i vc = _mm256_set1_epi8(ch);
        uint64_t m0 = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, vc)));
        uint64_t m1 = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, vc)));
        return m0 | (m1 << 32);
    }
};
#elif defined(__aarch64__)
struct Vec64 {
    uint8x16_t v0, v1, v2, v3;
    static SS_ALWAYS_INLINE Vec64 load(const char* p) {
        const uint8_t* u = reinterpret_cast<const uint8_t*>(p);
        return {vld1q_u8(u), vld1q_u8(u + 16), vld1q_u8(u + 32), vld1q_u8(u + 48)};
    }
    SS_ALWAYS_INLINE uint64_t eq(char ch) const {
        const uint8x16_t vc = vdupq_n_u8(uint8_t(ch));
        const uint8x16_t bm = {0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,
                               0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80};
        uint8x16_t t0 = vandq_u8(vceqq_u8(v0, vc), bm);
        uint8x16_t t1 = vandq_u8(vceqq_u8(v1, vc), bm);
        uint8x16_t t2 = vandq_u8(vceqq_u8(v2, vc), bm);
        uint8x16_t t3 = vandq_u8(vceqq_u8(v3, vc), bm);
        uint8x16_t s0 = vpaddq_u8(t0, t1);
        uint8x16_t s1 = vpaddq_u8(t2, t3);
        s0 = vpaddq_u8(s0, s1);
        s0 = vpaddq_u8(s0, s0);
        return vgetq_lane_u64(vreinterpretq_u64_u8(s0), 0);
    }
};
#elif defined(__SSE2__)
struct Vec64 {
    __m128i v0, v1, v2, v3;
    static SS_ALWAYS_INLINE Vec64 load(const char* p) {
        return {
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 32)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 48))
        };
    }
    SS_ALWAYS_INLINE uint64_t eq(char ch) const {
        __m128i vc = _mm_set1_epi8(ch);
        uint64_t m0 = uint32_t(_mm_movemask_epi8(_mm_cmpeq_epi8(v0, vc)));
        uint64_t m1 = uint32_t(_mm_movemask_epi8(_mm_cmpeq_epi8(v1, vc)));
        uint64_t m2 = uint32_t(_mm_movemask_epi8(_mm_cmpeq_epi8(v2, vc)));
        uint64_t m3 = uint32_t(_mm_movemask_epi8(_mm_cmpeq_epi8(v3, vc)));
        return m0 | (m1 << 16) | (m2 << 32) | (m3 << 48);
    }
};
#endif

// CLMUL-validated 6-tzcnt zero-load extractor for 7-field events with L in [65, 127].
// ZERO byte loads for quote checking (100% in registers via PCLMULQDQ + bit shifts)!
static SS_ALWAYS_INLINE bool extract7_hi64(
    const char* line, uint32_t L,
    uint64_t cm0, uint64_t cm1,
    uint64_t qm0, uint64_t qm1,
    Slice out[7]
) {
    uint64_t m1 = (1ULL << (L - 64U)) - 1ULL;
    cm1 &= m1;
    qm1 &= m1;
    uint32_t k0 = uint32_t(__builtin_popcountll(cm0));
    uint32_t k1 = uint32_t(__builtin_popcountll(cm1));
    if (__builtin_expect(k0 + k1 != 6U, 0)) return false;

    uint64_t st0 = 1ULL | (cm0 << 1);
    uint64_t st1 = (cm0 >> 63) | (cm1 << 1);
    uint64_t en0 = (cm0 >> 1) | (cm1 << 63);
    uint64_t en1 = (cm1 >> 1) | (1ULL << (L - 65U));
    uint64_t qbad = (qm0 & ~(st0 | en0)) | (qm1 & ~(st1 | en1)) |
                    (qm0 & st0 & en0) | (qm1 & st1 & en1);
    uint64_t iq0 = prefix_xor(qm0);
    uint64_t iq1 = prefix_xor(qm1) ^ uint64_t(-int64_t(iq0 >> 63));
    qbad |= (iq0 & cm0) | (iq1 & cm1) | (iq1 >> 63);
    if (__builtin_expect(qbad != 0, 0)) return false;

    if (__builtin_expect(k0 < 3U, 0)) return parse_csv_line_scalar<7>(line, line + L, out);

    uint64_t a = cm0;
    uint32_t e0 = uint32_t(__builtin_ctzll(a)); a &= a - 1;
    uint32_t e1 = uint32_t(__builtin_ctzll(a)); a &= a - 1;
    uint32_t e2 = uint32_t(__builtin_ctzll(a)); a &= a - 1;
    uint32_t s3 = e2 + 1U;
    if (__builtin_expect((s3 >= 64U) | (L - s3 >= 64U), 0)) {
        return parse_csv_line_scalar<7>(line, line + L, out);
    }
    if (__builtin_expect(e0 == 0, 0)) return false;

    uint32_t rsh3 = 64U - s3;
    uint64_t crem = (a >> s3) | (cm1 << rsh3);
    uint64_t qrem = (qm0 >> s3) | (qm1 << rsh3);
    uint32_t r3 = uint32_t(__builtin_ctzll(crem)); crem &= crem - 1;
    uint32_t r4 = uint32_t(__builtin_ctzll(crem)); crem &= crem - 1;
    uint32_t r5 = uint32_t(__builtin_ctzll(crem));

    uint32_t qs0 = uint32_t(qm0 & 1ULL);
    uint32_t qs1 = uint32_t((qm0 >> (e0 + 1U)) & 1ULL);
    uint32_t qs2 = uint32_t((qm0 >> (e1 + 1U)) & 1ULL);
    uint32_t qs3 = uint32_t(qrem & 1ULL);
    uint32_t qs4 = uint32_t((qrem >> (r3 + 1U)) & 1ULL);
    uint32_t qs5 = uint32_t((qrem >> (r4 + 1U)) & 1ULL);
    uint32_t qs6 = uint32_t((qrem >> (r5 + 1U)) & 1ULL);

    out[0] = {line + qs0,                size_t(e0 - (qs0 << 1))};
    out[1] = {line + e0 + 1 + qs1,       size_t((e1 - e0 - 1) - (qs1 << 1))};
    out[2] = {line + e1 + 1 + qs2,       size_t((e2 - e1 - 1) - (qs2 << 1))};
    out[3] = {line + s3 + qs3,           size_t(r3 - (qs3 << 1))};
    out[4] = {line + s3 + r3 + 1 + qs4,  size_t((r4 - r3 - 1) - (qs4 << 1))};
    out[5] = {line + s3 + r4 + 1 + qs5,  size_t((r5 - r4 - 1) - (qs5 << 1))};
    out[6] = {line + s3 + r5 + 1 + qs6,  size_t((L - s3 - r5 - 1) - (qs6 << 1))};
    return true;
}

// CLMUL-validated single-word 64-bit extractor for clicks (N=4) and links (N=2) when L in [1, 63]!
template <int N>
static SS_ALWAYS_INLINE bool extract_short64(
    const char* line, uint32_t L,
    uint64_t cm0, uint64_t qm0,
    Slice out[N]
) {
    uint64_t lm = (1ULL << L) - 1ULL;
    cm0 &= lm;
    qm0 &= lm;
    if (__builtin_expect(uint32_t(__builtin_popcountll(cm0)) != uint32_t(N - 1), 0)) return false;
    uint64_t st0 = 1ULL | (cm0 << 1);
    uint64_t en0 = (cm0 >> 1) | (1ULL << (L - 1U));
    uint64_t iq0 = prefix_xor(qm0);
    uint64_t qbad = (qm0 & ~(st0 | en0)) | (qm0 & st0 & en0) | (iq0 & cm0) | (iq0 >> 63);
    if (__builtin_expect(qbad != 0, 0)) return false;

    if constexpr (N == 2) {
        uint32_t e0 = uint32_t(__builtin_ctzll(cm0));
        if (__builtin_expect(e0 == 0 || e0 + 1 == L, 0)) return false;
        uint32_t qs0 = uint32_t(qm0 & 1ULL);
        uint32_t qs1 = uint32_t((qm0 >> (e0 + 1U)) & 1ULL);
        out[0] = {line + qs0,          size_t(e0 - (qs0 << 1))};
        out[1] = {line + e0 + 1 + qs1, size_t((L - e0 - 1) - (qs1 << 1))};
        return true;
    } else if constexpr (N == 4) {
        uint64_t a = cm0;
        uint32_t e0 = uint32_t(__builtin_ctzll(a)); a &= a - 1;
        uint32_t e1 = uint32_t(__builtin_ctzll(a)); a &= a - 1;
        uint32_t e2 = uint32_t(__builtin_ctzll(a));
        if (__builtin_expect(e0 == 0, 0)) return false;
        uint32_t qs0 = uint32_t(qm0 & 1ULL);
        uint32_t qs1 = uint32_t((qm0 >> (e0 + 1U)) & 1ULL);
        uint32_t qs2 = uint32_t((qm0 >> (e1 + 1U)) & 1ULL);
        uint32_t qs3 = uint32_t((qm0 >> (e2 + 1U)) & 1ULL);
        out[0] = {line + qs0,          size_t(e0 - (qs0 << 1))};
        out[1] = {line + e0 + 1 + qs1, size_t((e1 - e0 - 1) - (qs1 << 1))};
        out[2] = {line + e1 + 1 + qs2, size_t((e2 - e1 - 1) - (qs2 << 1))};
        out[3] = {line + e2 + 1 + qs3, size_t((L - e2 - 1) - (qs3 << 1))};
        return true;
    } else {
        return parse_csv_line_scalar<N>(line, line + L, out);
    }
}

// Branchless safe ctz64 returning uint64_t (zero-extended natively by _tzcnt_u64, zero movabs masks!)
static SS_ALWAYS_INLINE uint64_t ctz64_u64(uint64_t x) {
#if defined(__BMI__)
    return _tzcnt_u64(x);
#else
    return uint64_t(__builtin_ctzll(x | (1ULL << 63)));
#endif
}

static SS_ALWAYS_INLINE uint32_t ctz64_safe(uint64_t x) {
    return uint32_t(ctz64_u64(x));
}

static SS_ALWAYS_INLINE uint64_t pext_u64_fast(uint64_t src, uint64_t mask) {
#if defined(__BMI2__)
    return _pext_u64(src, mask);
#else
    uint64_t res = 0, bit = 1;
    while (mask) {
        uint64_t lsb = mask & -mask;
        if (src & lsb) res |= bit;
        mask ^= lsb;
        bit <<= 1;
    }
    return res;
#endif
}

// Zero-quote-load separator-indexed CSV row accessor (1-cycle quote flags from pext((RQ<<1)|rq_carry, S)!)
template <int N>
struct SepsCsvRow {
    const char* bptr;
    const uint16_t* sp;
    uint32_t ls;
    uint32_t qm;

    template <int I>
    SS_ALWAYS_INLINE Slice get() const {
        uint32_t s;
        if constexpr (I == 0) s = ls;
        else s = uint32_t(sp[I - 1]) + 1U;
        uint32_t e = uint32_t(sp[I]);
        uint32_t q = (qm >> I) & 1U;
        return {bptr + s + q, size_t((e - s) - (q << 1))};
    }

    template <int I>
    SS_ALWAYS_INLINE bool nonempty() const {
        uint32_t s;
        if constexpr (I == 0) s = ls;
        else s = uint32_t(sp[I - 1]) + 1U;
        uint32_t e = uint32_t(sp[I]);
        uint32_t q = (qm >> I) & 1U;
        return (e - s) > (q << 1);
    }
};

// Register-only lazy CSV row accessors (ZERO stack spills, ZERO store-forwarding stalls!)
template <int N>
struct PackedCsvRow {
    const char* lp;
    uint64_t pk; // byte i = offset of comma i (for i = 0..N-2)
    uint32_t L;

    template <int I>
    SS_ALWAYS_INLINE Slice get() const {
        uint32_t s;
        if constexpr (I == 0) s = 0U;
        else s = uint32_t((pk >> ((I - 1) * 8)) & 0xFFU) + 1U;
        uint32_t e;
        if constexpr (I == N - 1) e = L;
        else e = uint32_t((pk >> (I * 8)) & 0xFFU);
        uint32_t l = e - s;
        uint32_t q = (l > 0U) & (lp[s] == '"');
        return {lp + s + q, size_t(l - (q << 1))};
    }

    template <int I>
    SS_ALWAYS_INLINE bool nonempty() const {
        uint32_t s;
        if constexpr (I == 0) s = 0U;
        else s = uint32_t((pk >> ((I - 1) * 8)) & 0xFFU) + 1U;
        uint32_t e;
        if constexpr (I == N - 1) e = L;
        else e = uint32_t((pk >> (I * 8)) & 0xFFU);
        uint32_t l = e - s;
        return (l > 2U) | ((l > 0U) & (lp[s] != '"'));
    }
};

struct SliceCsvRow {
    const Slice* f;
    template <int I>
    SS_ALWAYS_INLINE Slice get() const { return f[I]; }
    template <int I>
    SS_ALWAYS_INLINE bool nonempty() const { return f[I].len > 0; }
};

// Pure-register comma-offset packer for block-validated clean 4KB windows!
// Eliminates the 20-cycle overlapping vmovdqu->scalar store-forwarding stall AND all 15 Slice f[7] stack spills!
// Returns: 1 = packed into `out_pk`, 0 = invalid comma count, -1 = long/empty line needing scalar fallback.
template <int N>
static SS_ALWAYS_INLINE int extract_clean_packed(
    size_t ls, uint32_t L, const uint64_t* cms, uint64_t& out_pk
) {
    if constexpr (N == 7) {
        if (__builtin_expect(L == 0U || L >= 128U, 0)) return -1;
        size_t sbi = ls >> 6;
        unsigned sh = unsigned(ls & 63);
        uint64_t c0, c1;
        if (__builtin_expect(sh != 0, 1)) {
            unsigned rsh = 64U - sh;
            uint64_t nc = cms[sbi + 1];
            c0 = (cms[sbi] >> sh) | (nc << rsh);
            c1 = (nc >> sh) | (cms[sbi + 2] << rsh);
        } else {
            c0 = cms[sbi];
            c1 = cms[sbi + 1];
        }
        if (__builtin_expect(L < 64U, 0)) {
            c0 &= (1ULL << L) - 1ULL;
            c1 = 0ULL;
        } else {
            c1 &= (1ULL << (L - 64U)) - 1ULL;
        }
        uint32_t k0 = uint32_t(__builtin_popcountll(c0));
        uint32_t k1 = uint32_t(__builtin_popcountll(c1));
        if (__builtin_expect(k0 + k1 != 6U, 0)) return 0;

        if (__builtin_expect(k0 >= 2U && k0 <= 4U, 1)) {
            // 100% branchless, 100% in-register byte packing! ZERO stack array, ZERO store-forwarding stall!
            uint64_t w0 = uint64_t(ctz64_safe(c0));          c0 &= c0 - 1;
            w0 |= uint64_t(ctz64_safe(c0)) << 8;             c0 &= c0 - 1;
            w0 |= uint64_t(ctz64_safe(c0)) << 16;            c0 &= c0 - 1;
            w0 |= uint64_t(ctz64_safe(c0)) << 24;
            uint64_t w1 = uint64_t(64U + ctz64_safe(c1));    c1 &= c1 - 1;
            w1 |= uint64_t(64U + ctz64_safe(c1)) << 8;       c1 &= c1 - 1;
            w1 |= uint64_t(64U + ctz64_safe(c1)) << 16;      c1 &= c1 - 1;
            w1 |= uint64_t(64U + ctz64_safe(c1)) << 24;
            uint32_t bsh = k0 << 3;
            out_pk = (w0 & ((1ULL << bsh) - 1ULL)) | (w1 << bsh);
            return 1;
        }
        uint64_t pk = 0;
        uint32_t bsh = 0;
        while (c0) { pk |= uint64_t(__builtin_ctzll(c0)) << bsh; bsh += 8; c0 &= c0 - 1; }
        while (c1) { pk |= uint64_t(64U + __builtin_ctzll(c1)) << bsh; bsh += 8; c1 &= c1 - 1; }
        out_pk = pk;
        return 1;
    } else if constexpr (N == 4) {
        if (__builtin_expect(L == 0U || L >= 128U, 0)) return -1;
        size_t sbi = ls >> 6;
        unsigned sh = unsigned(ls & 63);
        uint64_t c0, c1;
        if (__builtin_expect(sh != 0, 1)) {
            unsigned rsh = 64U - sh;
            uint64_t nc = cms[sbi + 1];
            c0 = (cms[sbi] >> sh) | (nc << rsh);
            c1 = (nc >> sh) | (cms[sbi + 2] << rsh);
        } else {
            c0 = cms[sbi];
            c1 = cms[sbi + 1];
        }
        if (__builtin_expect(L < 64U, 1)) {
            c0 &= (1ULL << L) - 1ULL;
            c1 = 0ULL;
        } else {
            c1 &= (1ULL << (L - 64U)) - 1ULL;
        }
        uint32_t k0 = uint32_t(__builtin_popcountll(c0));
        uint32_t k1 = uint32_t(__builtin_popcountll(c1));
        if (__builtin_expect(k0 + k1 != 3U, 0)) return 0;
        // 100% branchless, 100% in-register for ALL k0 in {0, 1, 2, 3}!
        uint64_t w0 = uint64_t(ctz64_safe(c0));          c0 &= c0 - 1;
        w0 |= uint64_t(ctz64_safe(c0)) << 8;             c0 &= c0 - 1;
        w0 |= uint64_t(ctz64_safe(c0)) << 16;
        uint64_t w1 = uint64_t(64U + ctz64_safe(c1));    c1 &= c1 - 1;
        w1 |= uint64_t(64U + ctz64_safe(c1)) << 8;       c1 &= c1 - 1;
        w1 |= uint64_t(64U + ctz64_safe(c1)) << 16;
        uint32_t bsh = k0 << 3;
        out_pk = (w0 & ((1ULL << bsh) - 1ULL)) | (w1 << bsh);
        return 1;
    } else if constexpr (N == 2) {
        if (__builtin_expect(L == 0U || L >= 128U, 0)) return -1;
        size_t sbi = ls >> 6;
        unsigned sh = unsigned(ls & 63);
        uint64_t c0, c1;
        if (__builtin_expect(sh != 0, 1)) {
            unsigned rsh = 64U - sh;
            uint64_t nc = cms[sbi + 1];
            c0 = (cms[sbi] >> sh) | (nc << rsh);
            c1 = (nc >> sh) | (cms[sbi + 2] << rsh);
        } else {
            c0 = cms[sbi];
            c1 = cms[sbi + 1];
        }
        if (__builtin_expect(L < 64U, 1)) {
            c0 &= (1ULL << L) - 1ULL;
            c1 = 0ULL;
        } else {
            c1 &= (1ULL << (L - 64U)) - 1ULL;
        }
        if (__builtin_expect(uint32_t(__builtin_popcountll(c0)) +
                             uint32_t(__builtin_popcountll(c1)) != 1U, 0)) return 0;
        out_pk = c0 ? uint64_t(__builtin_ctzll(c0)) : (64ULL + uint64_t(__builtin_ctzll(c1)));
        return 1;
    } else {
        return -1;
    }
}

// Zero-CLMUL, zero-quote-realign extractor for block-validated clean 4KB windows!
// Returns: 1 = extracted cleanly into `out`, 0 = invalid comma count, -1 = long/empty line needing scalar fallback.
// NEVER passes `out` to any noinline function so GCC/Clang can 100% SRA `Slice f[N]` into registers!
template <int N>
static SS_ALWAYS_INLINE int extract_clean_csv(
    const char* lp, size_t ls, uint32_t L, const uint64_t* cms, Slice out[N]
) {
    uint64_t pk = 0;
    int er = extract_clean_packed<N>(ls, L, cms, pk);
    if (__builtin_expect(er <= 0, 0)) return er;
    PackedCsvRow<N> row{lp, pk, L};
    if constexpr (N == 7) {
        out[0] = row.template get<0>(); out[1] = row.template get<1>(); out[2] = row.template get<2>();
        out[3] = row.template get<3>(); out[4] = row.template get<4>(); out[5] = row.template get<5>();
        out[6] = row.template get<6>();
    } else if constexpr (N == 4) {
        out[0] = row.template get<0>(); out[1] = row.template get<1>();
        out[2] = row.template get<2>(); out[3] = row.template get<3>();
    } else if constexpr (N == 2) {
        out[0] = row.template get<0>(); out[1] = row.template get<1>();
    }
    return 1;
}

static SS_ALWAYS_INLINE const char* find_nl_simd(const char* p, const char* end) {
    while (size_t(end - p) >= 64) {
        uint64_t m = Vec64::load(p).eq('\n');
        if (m) return p + __builtin_ctzll(m);
        p += 64;
    }
    while (p < end) {
        if (*p == '\n') return p;
        ++p;
    }
    return nullptr;
}

static SS_ALWAYS_INLINE int32_t ms_to_day(int64_t ms) {
    int64_t q = ms / DAY_MS;
    int64_t r = ms % DAY_MS;
    if (r < 0) --q;
    return int32_t(q);
}

} // namespace SS_ENGINE_NS
