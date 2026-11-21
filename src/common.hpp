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

namespace ss {

static constexpr int64_t SECOND_MS       = 1000LL;
static constexpr int64_t MINUTE_MS       = 60LL * SECOND_MS;
static constexpr int64_t HOUR_MS         = 60LL * MINUTE_MS;
static constexpr int64_t DAY_MS          = 24LL * HOUR_MS;
static constexpr int64_t SESSION_GAP_MS  = 30LL * MINUTE_MS;
static constexpr int64_t CLICK_WINDOW_MS = 7LL * DAY_MS;

static constexpr uint64_t FALLBACK_BIT = 1ULL << 63;

// Fast anonymous mmap helper (uses THP on Linux where available).
static inline void* alloc_mmap(size_t bytes) {
    if (bytes == 0) bytes = 4096;
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        std::fprintf(stderr, "mmap failed for %zu bytes\n", bytes);
        std::abort();
    }
#ifdef MADV_HUGEPAGE
    if (bytes >= (2ULL << 20)) {
        ::madvise(p, bytes, MADV_HUGEPAGE);
    }
#endif
    return p;
}

// Per-thread virtual-reserved bump arena for fallback strings (>10 bytes or non-[0-9A-Za-z_]).
struct StringArenaPool {
    static constexpr int MAX_THREADS = 64;
    static constexpr size_t RESERVATION = 256ULL << 20;
    char* base[MAX_THREADS]{};
    size_t pos[MAX_THREADS]{};

    void init(int nthreads) {
        for (int t = 0; t < nthreads; ++t) {
            base[t] = static_cast<char*>(alloc_mmap(RESERVATION));
            pos[t] = 0;
        }
    }

    inline uint64_t store_fallback(int tid, const char* s, size_t len) {
        size_t off = pos[tid];
        if (__builtin_expect(off + len > RESERVATION, 0)) {
            std::fprintf(stderr, "StringArena overflow on thread %d\n", tid);
            std::abort();
        }
        std::memcpy(base[tid] + off, s, len);
        pos[tid] = off + len;
        return FALLBACK_BIT | (uint64_t(tid) << 57) | (uint64_t(off) << 25) | uint64_t(len);
    }

    inline std::string_view get_fallback(uint64_t k) const {
        uint32_t tid = uint32_t((k >> 57) & 0x3F);
        uint32_t off = uint32_t((k >> 25) & 0xFFFFFFFFULL);
        uint32_t len = uint32_t(k & 0x1FFFFFFULL);
        return {base[tid] + off, len};
    }
};

// 6-bit ASCII-order-preserving encoding for [0-9A-Za-z_] (length 1..10).
// Valid chars map to 1..63; invalid chars map to 0x80 (so single OR & 0x80 checks all chars!).
alignas(64) static constexpr uint8_t ENC6[256] = {
    0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
    0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
    0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
    1,2,3,4,5,6,7,8, 9,10,0x80,0x80,0x80,0x80,0x80,0x80, // '0'..'9'
    0x80,11,12,13,14,15,16,17, 18,19,20,21,22,23,24,25,  // 'A'..'O'
    26,27,28,29,30,31,32,33, 34,35,36,0x80,0x80,0x80,0x80,37, // 'P'..'Z', '_'
    0x80,38,39,40,41,42,43,44, 45,46,47,48,49,50,51,52,  // 'a'..'o'
    53,54,55,56,57,58,59,60, 61,62,63,0x80,0x80,0x80,0x80,0x80, // 'p'..'z'
    0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
    0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
    0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
    0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
    0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
    0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
    0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
    0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
};

alignas(64) static constexpr char DEC6[64] = {
    0,
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    '_',
    'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'
};

// Unrolled branchless fast path for len==9 (93% of generator IDs) and len==8 (7%)!
static inline uint64_t encode_key64(const char* s, size_t len, StringArenaPool& arenas, int tid) {
    if (__builtin_expect(len == 9, 1)) {
        uint8_t c0 = ENC6[uint8_t(s[0])];
        uint8_t c1 = ENC6[uint8_t(s[1])];
        uint8_t c2 = ENC6[uint8_t(s[2])];
        uint8_t c3 = ENC6[uint8_t(s[3])];
        uint8_t c4 = ENC6[uint8_t(s[4])];
        uint8_t c5 = ENC6[uint8_t(s[5])];
        uint8_t c6 = ENC6[uint8_t(s[6])];
        uint8_t c7 = ENC6[uint8_t(s[7])];
        uint8_t c8 = ENC6[uint8_t(s[8])];
        if (__builtin_expect(((c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8) & 0x80) == 0, 1)) {
            return (uint64_t(c0) << 54) | (uint64_t(c1) << 48) | (uint64_t(c2) << 42) |
                   (uint64_t(c3) << 36) | (uint64_t(c4) << 30) | (uint64_t(c5) << 24) |
                   (uint64_t(c6) << 18) | (uint64_t(c7) << 12) | (uint64_t(c8) << 6);
        }
        return arenas.store_fallback(tid, s, len);
    }
    if (__builtin_expect(len == 8, 1)) {
        uint8_t c0 = ENC6[uint8_t(s[0])];
        uint8_t c1 = ENC6[uint8_t(s[1])];
        uint8_t c2 = ENC6[uint8_t(s[2])];
        uint8_t c3 = ENC6[uint8_t(s[3])];
        uint8_t c4 = ENC6[uint8_t(s[4])];
        uint8_t c5 = ENC6[uint8_t(s[5])];
        uint8_t c6 = ENC6[uint8_t(s[6])];
        uint8_t c7 = ENC6[uint8_t(s[7])];
        if (__builtin_expect(((c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7) & 0x80) == 0, 1)) {
            return (uint64_t(c0) << 54) | (uint64_t(c1) << 48) | (uint64_t(c2) << 42) |
                   (uint64_t(c3) << 36) | (uint64_t(c4) << 30) | (uint64_t(c5) << 24) |
                   (uint64_t(c6) << 18) | (uint64_t(c7) << 12);
        }
        return arenas.store_fallback(tid, s, len);
    }
    if (__builtin_expect(len == 0, 0)) return 0;
    if (__builtin_expect(len <= 10, 1)) {
        uint64_t acc = 0;
        int shift = 54;
        uint8_t bad = 0;
        for (size_t i = 0; i < len; ++i) {
            uint8_t c = ENC6[uint8_t(s[i])];
            bad |= c;
            acc |= uint64_t(c) << shift;
            shift -= 6;
        }
        if (__builtin_expect((bad & 0x80) == 0, 1)) return acc;
    }
    return arenas.store_fallback(tid, s, len);
}

static inline size_t decode_inline(uint64_t k, char out[10]) {
    size_t n = 0;
    int shift = 54;
    while (shift >= 0) {
        uint8_t c = uint8_t((k >> shift) & 63);
        if (c == 0) break;
        out[n++] = DEC6[c];
        shift -= 6;
    }
    return n;
}

static inline std::string_view key_to_sv(uint64_t k, char tmp[10], const StringArenaPool& arenas) {
    if (__builtin_expect((k & FALLBACK_BIT) == 0, 1)) {
        size_t n = decode_inline(k, tmp);
        return {tmp, n};
    }
    return arenas.get_fallback(k);
}

static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static inline uint64_t hash_bytes(const char* s, size_t len) {
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

static inline uint64_t hash_key64(uint64_t k, const StringArenaPool& arenas) {
    if (__builtin_expect((k & FALLBACK_BIT) == 0, 1)) {
        return mix64(k);
    }
    auto sv = arenas.get_fallback(k);
    return hash_bytes(sv.data(), sv.size());
}

static inline bool key_eq(uint64_t a, uint64_t b, const StringArenaPool& arenas) {
    if (__builtin_expect(((a | b) & FALLBACK_BIT) == 0, 1)) {
        return a == b;
    }
    if (((a ^ b) & FALLBACK_BIT) != 0) return false;
    auto sa = arenas.get_fallback(a);
    auto sb = arenas.get_fallback(b);
    return sa.size() == sb.size() && std::memcmp(sa.data(), sb.data(), sa.size()) == 0;
}

static inline bool key_less(uint64_t a, uint64_t b, const StringArenaPool& arenas) {
    if (__builtin_expect(((a | b) & FALLBACK_BIT) == 0, 1)) {
        return a < b;
    }
    char ba[10], bb[10];
    auto sa = key_to_sv(a, ba, arenas);
    auto sb = key_to_sv(b, bb, arenas);
    size_t m = sa.size() < sb.size() ? sa.size() : sb.size();
    int cmp = std::memcmp(sa.data(), sb.data(), m);
    if (cmp != 0) return cmp < 0;
    return sa.size() < sb.size();
}

// ============================================================================
// Calendar & FastMonth + SWAR Timestamp Parsing (Strict F1, F2, F3 per rules.md)
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

struct FastMonth {
    uint64_t prefix_u64; // "YYYY-MM-" little-endian u64
    int64_t day0_sec;    // epoch seconds of YYYY-MM-00T00:00:00Z
    uint8_t max_day;
};

struct DateLut {
    static constexpr int NUM_YEARS = 8;
    static constexpr int32_t INVALID = INT32_MIN;
    int base_year = -1000000;
    int32_t table[NUM_YEARS * 16 * 32]{};
    FastMonth fm[4]{};
    int num_fm = 0;

    void init(int64_t click_start_ms, int64_t win_end_ms) {
        int y1, m1, d1, y2, m2, d2;
        civil_from_days((click_start_ms / DAY_MS) - 2, y1, m1, d1);
        civil_from_days((win_end_ms / DAY_MS) + 10, y2, m2, d2);
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
        // Populate up to 4 FastMonth entries covering [y1-m1 .. y2-m2]
        // Put the main window months FIRST so branch predictor hits fm[0]/fm[1]!
        num_fm = 0;
        int wy, wm, wd;
        civil_from_days((win_end_ms / DAY_MS) - 1, wy, wm, wd);
        auto add_ym = [&](int y, int m) {
            if (num_fm >= 4) return;
            char buf[8];
            buf[0] = char('0' + (y / 1000) % 10);
            buf[1] = char('0' + (y / 100) % 10);
            buf[2] = char('0' + (y / 10) % 10);
            buf[3] = char('0' + y % 10);
            buf[4] = '-';
            buf[5] = char('0' + m / 10);
            buf[6] = char('0' + m % 10);
            buf[7] = '-';
            uint64_t u;
            std::memcpy(&u, buf, 8);
            for (int i = 0; i < num_fm; ++i) if (fm[i].prefix_u64 == u) return;
            uint8_t md = (m == 2 && is_leap_year(y)) ? 29 : DAYS_IN_MONTH[m];
            int64_t day0_s = (days_from_civil(y, m, 1) - 1) * 86400LL;
            fm[num_fm++] = {u, day0_s, md};
        };
        add_ym(wy, wm);
        int cy = y1, cm = m1;
        for (int step = 0; step < 4; ++step) {
            add_ym(cy, cm);
            if (cy == y2 && cm == m2) break;
            if (++cm > 12) { cm = 1; ++cy; }
        }
    }

    inline bool lookup(int y, int m, int d, int64_t& out_days) const {
        unsigned yo = unsigned(y - base_year);
        if (__builtin_expect(yo < NUM_YEARS && unsigned(m) < 16 && unsigned(d) < 32, 1)) {
            int32_t v = table[(yo << 9) | (m << 5) | d];
            if (__builtin_expect(v != INVALID, 1)) {
                out_days = v;
                return true;
            }
            return false;
        }
        if (m < 1 || m > 12) return false;
        int maxd = (m == 2 && is_leap_year(y)) ? 29 : DAYS_IN_MONTH[m];
        if (d < 1 || d > maxd) return false;
        out_days = days_from_civil(y, m, d);
        return true;
    }

    inline bool lookup_fast_ym(uint64_t prefix_u64, int& out_idx) const {
        if (__builtin_expect(num_fm > 0 && fm[0].prefix_u64 == prefix_u64, 1)) { out_idx = 0; return true; }
        if (__builtin_expect(num_fm > 1 && fm[1].prefix_u64 == prefix_u64, 1)) { out_idx = 1; return true; }
        if (num_fm > 2 && fm[2].prefix_u64 == prefix_u64) { out_idx = 2; return true; }
        if (num_fm > 3 && fm[3].prefix_u64 == prefix_u64) { out_idx = 3; return true; }
        return false;
    }
};

static inline bool parse_2d(const char* p, int& out) {
    unsigned d0 = uint8_t(p[0]) - uint8_t('0');
    unsigned d1 = uint8_t(p[1]) - uint8_t('0');
    if (__builtin_expect((d0 > 9) | (d1 > 9), 0)) return false;
    out = int(d0 * 10 + d1);
    return true;
}

static inline bool parse_4d(const char* p, int& out) {
    unsigned d0 = uint8_t(p[0]) - uint8_t('0');
    unsigned d1 = uint8_t(p[1]) - uint8_t('0');
    unsigned d2 = uint8_t(p[2]) - uint8_t('0');
    unsigned d3 = uint8_t(p[3]) - uint8_t('0');
    if (__builtin_expect((d0 > 9) | (d1 > 9) | (d2 > 9) | (d3 > 9), 0)) return false;
    out = int(d0 * 1000 + d1 * 100 + d2 * 10 + d3);
    return true;
}

// Parse 8 ASCII digits via SWAR (strictly '0'..'9', zero borrow risk!)
static inline bool parse_8d_swar(const char* p, uint64_t& out) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    uint64_t bad = ((v & 0xF0F0F0F0F0F0F0F0ULL) ^ 0x3030303030303030ULL) |
                   (((v + 0x0606060606060606ULL) & 0xF0F0F0F0F0F0F0F0ULL) ^ 0x3030303030303030ULL);
    if (__builtin_expect(bad != 0, 0)) return false;
    v &= 0x0F0F0F0F0F0F0F0FULL;
    v = (v * 10 + (v >> 8)) & 0x00FF00FF00FF00FFULL;
    v = (v * 100 + (v >> 16)) & 0x0000FFFF0000FFFFULL;
    out = (v * 10000 + (v >> 32)) & 0xFFFFFFFFULL;
    return true;
}

// Fast 19-byte YYYY-MM-DDTHH:MM:SS core -> UTC seconds (before offset/Z).
static inline bool parse_iso19_sec(const char* s, int64_t& out_sec, const DateLut& lut) {
    if (__builtin_expect(s[10] != 'T' || s[13] != ':' || s[16] != ':', 0)) return false;
    int dd, hh, mm, ss;
    if (__builtin_expect(!parse_2d(s + 8, dd) || !parse_2d(s + 11, hh) ||
                         !parse_2d(s + 14, mm) || !parse_2d(s + 17, ss), 0)) return false;
    if (__builtin_expect(hh > 23 || mm > 59 || ss > 59, 0)) return false;

    uint64_t ym_u64;
    std::memcpy(&ym_u64, s, 8);
    int fi;
    if (__builtin_expect(lut.lookup_fast_ym(ym_u64, fi), 1)) {
        const FastMonth& m = lut.fm[fi];
        if (__builtin_expect(dd < 1 || dd > m.max_day, 0)) return false;
        out_sec = m.day0_sec + int64_t(dd) * 86400LL + hh * 3600LL + mm * 60LL + ss;
        return true;
    }

    if (s[4] != '-' || s[7] != '-') return false;
    int y, mo;
    if (!parse_4d(s, y) || !parse_2d(s + 5, mo)) return false;
    int64_t days;
    if (!lut.lookup(y, mo, dd, days)) return false;
    out_sec = days * 86400LL + hh * 3600LL + mm * 60LL + ss;
    return true;
}

// Fast strict F1/F2/F3 timestamp parser -> epoch milliseconds UTC.
static inline bool parse_ts(const char* s, size_t len, int64_t& out_ms, const DateLut& lut) {
    if (__builtin_expect(len == 20 && s[19] == 'Z', 1)) {
        int64_t sec;
        if (__builtin_expect(!parse_iso19_sec(s, sec, lut), 0)) return false;
        out_ms = sec * 1000LL;
        return true;
    }

    if (__builtin_expect(len == 13, 1)) {
        uint64_t hi8;
        if (__builtin_expect(!parse_8d_swar(s, hi8), 0)) return false;
        unsigned d0 = uint8_t(s[8]) - uint8_t('0');
        unsigned d1 = uint8_t(s[9]) - uint8_t('0');
        unsigned d2 = uint8_t(s[10]) - uint8_t('0');
        unsigned d3 = uint8_t(s[11]) - uint8_t('0');
        unsigned d4 = uint8_t(s[12]) - uint8_t('0');
        if (__builtin_expect((d0 > 9) | (d1 > 9) | (d2 > 9) | (d3 > 9) | (d4 > 9), 0)) return false;
        out_ms = int64_t(hi8 * 100000ULL + d0 * 10000U + d1 * 1000U + d2 * 100U + d3 * 10U + d4);
        return true;
    }

    if (len == 25 && (s[19] == '+' || s[19] == '-') && s[22] == ':') {
        int64_t sec;
        int oh, om;
        if (!parse_iso19_sec(s, sec, lut) || !parse_2d(s + 20, oh) || !parse_2d(s + 23, om)) return false;
        if (oh > 23 || om > 59) return false;
        int64_t off_sec = oh * 3600LL + om * 60LL;
        sec += (s[19] == '+') ? -off_sec : off_sec;
        out_ms = sec * 1000LL;
        return true;
    }

    if (__builtin_expect(len == 0, 0)) return false;
    // General F3: ASCII digits only, leading zeros allowed, huge values overflow -> outside window
    size_t i = 0;
    while (i < len && s[i] == '0') ++i;
    if (i == len) {
        out_ms = 0;
        return true;
    }
    size_t digits = len - i;
    if (digits > 19) return false;
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

static inline bool parse_ingest_sec(const char* s, size_t len, int64_t& out_sec, const DateLut& lut) {
    if (__builtin_expect(len == 20 && s[19] == 'Z', 1)) {
        return parse_iso19_sec(s, out_sec, lut);
    }
    int64_t ms;
    if (!parse_ts(s, len, ms, lut)) return false;
    out_sec = ms / 1000LL;
    return true;
}

// Strict purchase price parser: ASCII digits only, value in [1, 10_000_000], leading zeros allowed.
static inline bool parse_price_cents(const char* s, size_t len, uint32_t& out_price) {
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
// Strict CSV Line Parser (Scalar Fallback + Runtime-Dispatched AVX2/SSE2/NEON!)
// ============================================================================
struct Slice {
    const char* ptr;
    size_t len;
};

template <int N>
static inline bool parse_csv_line_scalar(const char* line, const char* end, Slice out[N]) {
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
            if (__builtin_expect(std::memchr(is, '"', size_t(ie - is)) != nullptr, 0)) return false;
            out[f] = {is, size_t(ie - is)};
        }
    }
    return true;
}

// Helper: given exact 128-bit comma & quote bitmasks over a line of length L in [1, 128],
// extract N fields and validate quotes strictly using bit operations!
template <int N>
static inline bool extract_fields_from_masks(
    const char* line,
    size_t L,
    uint64_t cm0, uint64_t cm1,
    uint64_t qm0, uint64_t qm1,
    Slice out[N]
) {
    if (__builtin_expect(__builtin_popcountll(cm0) + __builtin_popcountll(cm1) != (N - 1), 0)) {
        return false;
    }
    const bool any_q = (qm0 | qm1) != 0;
    uint32_t fstart = 0;

    auto finish_field = [&](int f, uint32_t fend) -> bool {
        if (__builtin_expect(!any_q || fstart == fend, 1)) {
            out[f] = {line + fstart, size_t(fend - fstart)};
            return true;
        }
        // Check quote bits strictly in [fstart, fend)
        uint64_t fq0 = 0, fq1 = 0;
        if (fstart < 64) {
            uint64_t hi_mask = (fend >= 64) ? ~0ULL : ((1ULL << fend) - 1ULL);
            uint64_t lo_mask = ~((1ULL << fstart) - 1ULL);
            fq0 = qm0 & hi_mask & lo_mask;
        }
        if (fend > 64) {
            uint64_t hi_mask = (fend >= 128) ? ~0ULL : ((1ULL << (fend - 64)) - 1ULL);
            uint64_t lo_mask = (fstart <= 64) ? ~0ULL : ~((1ULL << (fstart - 64)) - 1ULL);
            fq1 = qm1 & hi_mask & lo_mask;
        }
        if (__builtin_expect((fq0 | fq1) == 0, 1)) {
            out[f] = {line + fstart, size_t(fend - fstart)};
            return true;
        }
        // Must be length >= 2 and quotes ONLY at fstart and fend - 1!
        if (__builtin_expect(fend - fstart < 2, 0)) return false;
        uint32_t last = fend - 1;
        uint64_t exp0 = (fstart < 64 ? (1ULL << fstart) : 0ULL) | (last < 64 ? (1ULL << last) : 0ULL);
        uint64_t exp1 = (fstart >= 64 ? (1ULL << (fstart - 64)) : 0ULL) | (last >= 64 ? (1ULL << (last - 64)) : 0ULL);
        if (__builtin_expect(fq0 != exp0 || fq1 != exp1, 0)) return false;
        out[f] = {line + fstart + 1, size_t(fend - fstart - 2)};
        return true;
    };

    int f = 0;
    while (cm0) {
        uint32_t pos = uint32_t(__builtin_ctzll(cm0));
        cm0 &= cm0 - 1;
        if (!finish_field(f++, pos)) return false;
        fstart = pos + 1;
    }
    while (cm1) {
        uint32_t pos = 64U + uint32_t(__builtin_ctzll(cm1));
        cm1 &= cm1 - 1;
        if (!finish_field(f++, pos)) return false;
        fstart = pos + 1;
    }
    return finish_field(f, uint32_t(L));
}

#if defined(__x86_64__)

static inline bool has_avx2_runtime() {
    static const bool supported = __builtin_cpu_supports("avx2");
    return supported;
}

__attribute__((target("avx2")))
static inline void scan_128_avx2(const char* p, size_t L, uint64_t& cm0, uint64_t& cm1, uint64_t& qm0, uint64_t& qm1) {
    const __m256i vc = _mm256_set1_epi8(',');
    const __m256i vq = _mm256_set1_epi8('"');
    __m256i b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32));
    uint64_t c0 = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(b0, vc)));
    uint64_t c1 = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(b1, vc)));
    uint64_t q0 = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(b0, vq)));
    uint64_t q1 = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(b1, vq)));
    cm0 = c0 | (c1 << 32);
    qm0 = q0 | (q1 << 32);
    if (L <= 64) {
        uint64_t m = (L == 64) ? ~0ULL : ((1ULL << L) - 1ULL);
        cm0 &= m; qm0 &= m;
        cm1 = 0; qm1 = 0;
    } else {
        __m256i b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 64));
        __m256i b3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 96));
        uint64_t c2 = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(b2, vc)));
        uint64_t c3 = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(b3, vc)));
        uint64_t q2 = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(b2, vq)));
        uint64_t q3 = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(b3, vq)));
        uint64_t m = (L == 128) ? ~0ULL : ((1ULL << (L - 64)) - 1ULL);
        cm1 = (c2 | (c3 << 32)) & m;
        qm1 = (q2 | (q3 << 32)) & m;
    }
}

static inline void scan_128_sse2(const char* p, size_t L, uint64_t& cm0, uint64_t& cm1, uint64_t& qm0, uint64_t& qm1) {
    const __m128i vc = _mm_set1_epi8(',');
    const __m128i vq = _mm_set1_epi8('"');
    auto chunk16 = [&](const char* ptr, uint64_t& c, uint64_t& q) {
        __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
        c = uint32_t(_mm_movemask_epi8(_mm_cmpeq_epi8(b, vc)));
        q = uint32_t(_mm_movemask_epi8(_mm_cmpeq_epi8(b, vq)));
    };
    uint64_t c0, q0, c1, q1, c2, q2, c3, q3;
    chunk16(p, c0, q0); chunk16(p + 16, c1, q1);
    chunk16(p + 32, c2, q2); chunk16(p + 48, c3, q3);
    cm0 = c0 | (c1 << 16) | (c2 << 32) | (c3 << 48);
    qm0 = q0 | (q1 << 16) | (q2 << 32) | (q3 << 48);
    if (L <= 64) {
        uint64_t m = (L == 64) ? ~0ULL : ((1ULL << L) - 1ULL);
        cm0 &= m; qm0 &= m;
        cm1 = 0; qm1 = 0;
    } else {
        chunk16(p + 64, c0, q0); chunk16(p + 80, c1, q1);
        chunk16(p + 96, c2, q2); chunk16(p + 112, c3, q3);
        uint64_t m = (L == 128) ? ~0ULL : ((1ULL << (L - 64)) - 1ULL);
        cm1 = (c0 | (c1 << 16) | (c2 << 32) | (c3 << 48)) & m;
        qm1 = (q0 | (q1 << 16) | (q2 << 32) | (q3 << 48)) & m;
    }
}

#elif defined(__aarch64__)

static inline uint64_t neon_movemask64(uint8x16_t a, uint8x16_t b, uint8x16_t c, uint8x16_t d) {
    const uint8x16_t bit_mask = {0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,
                                 0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80};
    uint8x16_t t0 = vandq_u8(a, bit_mask);
    uint8x16_t t1 = vandq_u8(b, bit_mask);
    uint8x16_t t2 = vandq_u8(c, bit_mask);
    uint8x16_t t3 = vandq_u8(d, bit_mask);
    uint8x16_t sum0 = vpaddq_u8(t0, t1);
    uint8x16_t sum1 = vpaddq_u8(t2, t3);
    sum0 = vpaddq_u8(sum0, sum1);
    sum0 = vpaddq_u8(sum0, sum0);
    return vgetq_lane_u64(vreinterpretq_u64_u8(sum0), 0);
}

static inline void scan_128_neon(const char* p, size_t L, uint64_t& cm0, uint64_t& cm1, uint64_t& qm0, uint64_t& qm1) {
    const uint8x16_t vc = vdupq_n_u8(',');
    const uint8x16_t vq = vdupq_n_u8('"');
    const uint8_t* u = reinterpret_cast<const uint8_t*>(p);
    uint8x16_t b0 = vld1q_u8(u);
    uint8x16_t b1 = vld1q_u8(u + 16);
    uint8x16_t b2 = vld1q_u8(u + 32);
    uint8x16_t b3 = vld1q_u8(u + 48);
    cm0 = neon_movemask64(vceqq_u8(b0, vc), vceqq_u8(b1, vc), vceqq_u8(b2, vc), vceqq_u8(b3, vc));
    qm0 = neon_movemask64(vceqq_u8(b0, vq), vceqq_u8(b1, vq), vceqq_u8(b2, vq), vceqq_u8(b3, vq));
    if (L <= 64) {
        uint64_t m = (L == 64) ? ~0ULL : ((1ULL << L) - 1ULL);
        cm0 &= m; qm0 &= m;
        cm1 = 0; qm1 = 0;
    } else {
        uint8x16_t b4 = vld1q_u8(u + 64);
        uint8x16_t b5 = vld1q_u8(u + 80);
        uint8x16_t b6 = vld1q_u8(u + 96);
        uint8x16_t b7 = vld1q_u8(u + 112);
        uint64_t m = (L == 128) ? ~0ULL : ((1ULL << (L - 64)) - 1ULL);
        cm1 = neon_movemask64(vceqq_u8(b4, vc), vceqq_u8(b5, vc), vceqq_u8(b6, vc), vceqq_u8(b7, vc)) & m;
        qm1 = neon_movemask64(vceqq_u8(b4, vq), vceqq_u8(b5, vq), vceqq_u8(b6, vq), vceqq_u8(b7, vq)) & m;
    }
}
#endif

// SIMD-accelerated CSV parser (REQUIRES >= 128 bytes of readable padding past `line` when `padded == true`!)
template <int N>
static inline bool parse_csv_line_padded(const char* line, const char* end, Slice out[N], bool use_avx2) {
    size_t L = size_t(end - line);
    if (__builtin_expect(L == 0, 0)) return false;
    if (__builtin_expect(L <= 128, 1)) {
        uint64_t cm0, cm1, qm0, qm1;
#if defined(__x86_64__)
        if (use_avx2) {
            scan_128_avx2(line, L, cm0, cm1, qm0, qm1);
        } else {
            scan_128_sse2(line, L, cm0, cm1, qm0, qm1);
        }
#elif defined(__aarch64__)
        (void)use_avx2;
        scan_128_neon(line, L, cm0, cm1, qm0, qm1);
#else
        (void)use_avx2;
        return parse_csv_line_scalar<N>(line, end, out);
#endif
        return extract_fields_from_masks<N>(line, L, cm0, cm1, qm0, qm1, out);
    }
    return parse_csv_line_scalar<N>(line, end, out);
}

// Keep non-padded alias for small files (products.csv / campaigns.csv)
template <int N>
static inline bool parse_csv_line(const char* line, const char* end, Slice out[N]) {
    return parse_csv_line_scalar<N>(line, end, out);
}

static inline int32_t ms_to_day(int64_t ms) {
    int64_t q = ms / DAY_MS;
    int64_t r = ms % DAY_MS;
    if (r < 0) --q;
    return int32_t(q);
}

} // namespace ss
