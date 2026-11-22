#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
        size_t head = size_t(aligned - addr);
        size_t tail = req - head - aligned_bytes;
        if (head > 0) ::munmap(raw, head);
        if (tail > 0) ::munmap(reinterpret_cast<void*>(aligned + aligned_bytes), tail);
        void* p = reinterpret_cast<void*>(aligned);
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
    static constexpr size_t RESERVATION = 2ULL << 30; // 2 GiB VA (0 physical RAM until touched!)
    char* base[MAX_THREADS]{};
    size_t pos[MAX_THREADS]{};

    void init(int nthreads) {
        for (int t = 0; t < nthreads; ++t) {
            base[t] = static_cast<char*>(alloc_mmap(RESERVATION, false));
            pos[t] = 0;
        }
    }

    SS_COLD_NOINLINE uint64_t store_fallback(int tid, const char* s, size_t len) {
        size_t off = pos[tid];
        if (__builtin_expect(len > 0x1FFFFFFULL || off + len > RESERVATION, 0)) {
            std::fprintf(stderr, "StringArena overflow on thread %d (off=%zu len=%zu)\n", tid, off, len);
            std::abort();
        }
        if (len) std::memcpy(base[tid] + off, s, len);
        pos[tid] = off + len;
        return FALLBACK_BIT | (uint64_t(tid) << 57) | (uint64_t(off) << 25) | uint64_t(len);
    }

    SS_ALWAYS_INLINE std::string_view get_fallback(uint64_t k) const {
        uint32_t tid = uint32_t((k >> 57) & 0x3F);
        uint32_t off = uint32_t((k >> 25) & 0xFFFFFFFFULL);
        uint32_t len = uint32_t(k & 0x1FFFFFFULL);
        return {base[tid] + off, len};
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
            out_uid48 = pext48(__builtin_bswap64(x));
            return true;
        }
    }
    return false;
}

// Decode uid48 directly into ASCII bytes ('u' + 1..8 chars), returns exact length in 2..9!
static SS_ALWAYS_INLINE size_t decode_uid48_inline(uint64_t uid48, char* out) {
    out[0] = 'u';
    uint64_t u = pdep48(uid48);
    uint64_t b5 = u & 0x2020202020202020ULL;
    uint64_t be = u | (b5 << 1) | ((~b5) & ((u & 0x1010101010101010ULL) << 1));
    uint64_t le = __builtin_bswap64(be);
    std::memcpy(out + 1, &le, 8);
    return be ? (9U - (uint32_t(__builtin_ctzll(be)) >> 3)) : 1U;
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

static SS_COLD_NOINLINE uint64_t hash_bytes(const char* s, size_t len) {
    uint64_t h = 0x9e3779b97f4a7c15ULL ^ (len * 0x517cc1b727220a95ULL);
    size_t i = 0;
    while (i + 8 <= len) {
        uint64_t v;
        std::memcpy(&v, s + i, 8);
        h = mix64(h ^ v);
        i += 8;
    }
    uint64_t tail = 0;
    if (i < len) std::memcpy(&tail, s + i, len - i);
    return mix64(h ^ tail);
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

static SS_COLD_NOINLINE bool key_eq_fallback(uint64_t a, uint64_t b, const StringArenaPool& arenas) {
    if (((a ^ b) & FALLBACK_BIT) != 0) return false;
    auto sa = arenas.get_fallback(a);
    auto sb = arenas.get_fallback(b);
    return sa.size() == sb.size() && std::memcmp(sa.data(), sb.data(), sa.size()) == 0;
}

static SS_ALWAYS_INLINE bool key_eq(uint64_t a, uint64_t b, const StringArenaPool& arenas) {
    if (__builtin_expect(((a | b) & FALLBACK_BIT) == 0, 1)) {
        return a == b;
    }
    return key_eq_fallback(a, b, arenas);
}

static SS_COLD_NOINLINE bool key_less_fallback(uint64_t a, uint64_t b, const StringArenaPool& arenas) {
    char ba[10], bb[10];
    auto sa = key_to_sv(a, ba, arenas);
    auto sb = key_to_sv(b, bb, arenas);
    size_t m = sa.size() < sb.size() ? sa.size() : sb.size();
    int cmp = std::memcmp(sa.data(), sb.data(), m);
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
    if (__builtin_expect(len == 25 && (s[19] == '+' || s[19] == '-'), 1)) {
        int64_t rel_s;
        if (__builtin_expect(!lut.parse_f2_rel_sec(s, rel_s), 0)) return false;
        out_ms = (lut.win_start_sec + rel_s) * 1000LL;
        return true;
    }
    return parse_ts_slow(s, len, out_ms, lut);
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
