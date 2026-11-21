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

// ============================================================================
// 9-Byte 7-Bit ASCII Order-Preserving Key64 via BMI2 PEXT/PDEP (SWAR fallback)
// - Bit 63 == 0: inline 1..9 byte string, every byte in 0x01..0x7F.
//   Bits 62..56 = s[0], Bits 55..0 = 8x 7-bit packed s[1..8] (0-padded right).
//   Unsigned u64 '<' is 100% identical to exact byte-wise lexicographic order!
//   Encode/Decode: 1 BMI2 instruction (3 cycles) on x86-64-v3/v4, 9 ALU ops on ARM64!
// - Bit 63 == 1: fallback arena (>9 bytes, NUL byte, or non-ASCII UTF-8).
// ============================================================================
static inline uint64_t pext56(uint64_t be) {
#if defined(__BMI2__)
    return _pext_u64(be, 0x7F7F7F7F7F7F7F7FULL);
#else
    uint64_t t = ((be & 0x7F007F007F007F00ULL) >> 1) | (be & 0x007F007F007F007FULL);
    t = ((t & 0x3FFF00003FFF0000ULL) >> 2) | (t & 0x00003FFF00003FFFULL);
    return ((t & 0x0FFFFFFF00000000ULL) >> 4) | (t & 0x000000000FFFFFFFULL);
#endif
}

static inline uint64_t pdep56(uint64_t k) {
#if defined(__BMI2__)
    return _pdep_u64(k, 0x7F7F7F7F7F7F7F7FULL);
#else
    uint64_t t = ((k & 0x00FFFFFFF0000000ULL) << 4) | (k & 0x000000000FFFFFFFULL);
    t = ((t & 0x0FFFC0000FFFC000ULL) << 2) | (t & 0x00003FFF00003FFFULL);
    return ((t & 0x3F803F803F803F80ULL) << 1) | (t & 0x007F007F007F007FULL);
#endif
}

static inline uint64_t encode_key64(const char* s, size_t len, StringArenaPool& arenas, int tid) {
    if (__builtin_expect(len == 9, 1)) {
        uint8_t b0 = uint8_t(s[0]);
        uint64_t raw8;
        std::memcpy(&raw8, s + 1, 8);
        uint64_t bad = (raw8 & 0x8080808080808080ULL) |
                       ((raw8 - 0x0101010101010101ULL) & ~raw8 & 0x8080808080808080ULL);
        if (__builtin_expect(bad == 0 && uint8_t(b0 - 1) < 127, 1)) {
            return (uint64_t(b0) << 56) | pext56(__builtin_bswap64(raw8));
        }
        return arenas.store_fallback(tid, s, len);
    }
    if (__builtin_expect(len == 8, 1)) {
        uint8_t b0 = uint8_t(s[0]);
        uint64_t raw8 = 0;
        std::memcpy(&raw8, s + 1, 7);
        uint64_t check = raw8 | 0x0100000000000000ULL;
        uint64_t bad = (check & 0x8080808080808080ULL) |
                       ((check - 0x0101010101010101ULL) & ~check & 0x8080808080808080ULL);
        if (__builtin_expect(bad == 0 && uint8_t(b0 - 1) < 127, 1)) {
            return (uint64_t(b0) << 56) | pext56(__builtin_bswap64(raw8));
        }
        return arenas.store_fallback(tid, s, len);
    }
    if (__builtin_expect(len == 0, 0)) return 0;
    if (__builtin_expect(len <= 9, 1)) {
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

// Decode inline Key64 (writes up to 9 bytes into `out`, returns exact length in 1..9).
static inline size_t decode_inline(uint64_t k, char* out) {
    out[0] = char(k >> 56);
    uint64_t be = pdep56(k);
    uint64_t le = __builtin_bswap64(be);
    std::memcpy(out + 1, &le, 8);
    return be ? (9U - (uint32_t(__builtin_ctzll(be)) >> 3)) : 1U;
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
    uint64_t prefix_u64;
    int64_t day0_sec;
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
// Compile-Time Multi-ISA Single-Pass 64-Byte SIMD CSV Scanner + 3-Cycle Quotes
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
            for (const char* q = is; q < ie; ++q) {
                if (__builtin_expect(*q == '"', 0)) return false;
            }
            out[f] = {is, size_t(ie - is)};
        }
    }
    return true;
}

template <int N>
static inline bool parse_csv_line(const char* line, const char* end, Slice out[N]) {
    return parse_csv_line_scalar<N>(line, end, out);
}

#if defined(__AVX512BW__)
struct Vec64 {
    __m512i v;
    static inline Vec64 load(const char* p) { return {_mm512_loadu_si512(p)}; }
    inline uint64_t eq(char ch) const {
        return uint64_t(_mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(ch)));
    }
};
#elif defined(__AVX2__)
struct Vec64 {
    __m256i v0, v1;
    static inline Vec64 load(const char* p) {
        return {
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32))
        };
    }
    inline uint64_t eq(char ch) const {
        __m256i vc = _mm256_set1_epi8(ch);
        uint64_t m0 = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, vc)));
        uint64_t m1 = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, vc)));
        return m0 | (m1 << 32);
    }
};
#elif defined(__aarch64__)
struct Vec64 {
    uint8x16_t v0, v1, v2, v3;
    static inline Vec64 load(const char* p) {
        const uint8_t* u = reinterpret_cast<const uint8_t*>(p);
        return {vld1q_u8(u), vld1q_u8(u + 16), vld1q_u8(u + 32), vld1q_u8(u + 48)};
    }
    inline uint64_t eq(char ch) const {
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
    static inline Vec64 load(const char* p) {
        return {
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 32)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 48))
        };
    }
    inline uint64_t eq(char ch) const {
        __m128i vc = _mm_set1_epi8(ch);
        uint64_t m0 = uint32_t(_mm_movemask_epi8(_mm_cmpeq_epi8(v0, vc)));
        uint64_t m1 = uint32_t(_mm_movemask_epi8(_mm_cmpeq_epi8(v1, vc)));
        uint64_t m2 = uint32_t(_mm_movemask_epi8(_mm_cmpeq_epi8(v2, vc)));
        uint64_t m3 = uint32_t(_mm_movemask_epi8(_mm_cmpeq_epi8(v3, vc)));
        return m0 | (m1 << 16) | (m2 << 32) | (m3 << 48);
    }
};
#endif

// Ultra-fast field extractor with 3-cycle line-wide quote position validation!
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
    if (any_q) {
        // Every quote MUST sit at a field start (bit 0 or cm<<1) or field end (cm>>1 or bit L-1)!
        uint64_t st0 = 1ULL | (cm0 << 1);
        uint64_t st1 = (cm0 >> 63) | (cm1 << 1);
        uint64_t en0 = (cm0 >> 1) | (cm1 << 63) | (L <= 64 ? (1ULL << (L - 1)) : 0ULL);
        uint64_t en1 = (cm1 >> 1) | (L > 64 ? (1ULL << (L - 65)) : 0ULL);
        if (__builtin_expect(((qm0 & ~(st0 | en0)) | (qm1 & ~(st1 | en1))) != 0, 0)) {
            return false;
        }
    }

    uint32_t fstart = 0;
    auto finish_field = [&](int f, uint32_t fend) -> bool {
        uint32_t fs = fstart, fe = fend;
        if (any_q && fs < fe) {
            bool qs = (line[fs] == '"');
            bool qe = (line[fe - 1] == '"');
            if (__builtin_expect(qs != qe || (qs && (fe - fs) < 2), 0)) return false;
            fs += uint32_t(qs);
            fe -= uint32_t(qs);
        }
        out[f] = {line + fs, size_t(fe - fs)};
        return true;
    };

    int f = 0;
    while (cm0) {
        uint32_t pos = uint32_t(__builtin_ctzll(cm0));
        cm0 &= cm0 - 1;
        if (__builtin_expect(!finish_field(f++, pos), 0)) return false;
        fstart = pos + 1;
    }
    while (cm1) {
        uint32_t pos = 64U + uint32_t(__builtin_ctzll(cm1));
        cm1 &= cm1 - 1;
        if (__builtin_expect(!finish_field(f++, pos), 0)) return false;
        fstart = pos + 1;
    }
    return finish_field(f, uint32_t(L));
}

static inline const char* find_nl_simd(const char* p, const char* end) {
    while (p + 64 <= end) {
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

static inline int32_t ms_to_day(int64_t ms) {
    int64_t q = ms / DAY_MS;
    int64_t r = ms % DAY_MS;
    if (r < 0) --q;
    return int32_t(q);
}

} // namespace ss
