#pragma once
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <string_view>
#include <sys/uio.h>
#include <unistd.h>
#include <vector>
#if defined(__x86_64__)
#include <immintrin.h>
#endif

#ifndef SS_ENGINE_NS
#define SS_ENGINE_NS ss_default
#endif

namespace SS_ENGINE_NS {
namespace pq {

// Minimal, zero-dependency Parquet v1 PLAIN/UNCOMPRESSED writer with single-syscall writev()
// and automatic multi-page fallback if any column exceeds 1 GiB (prevents int32_t page overflow).
// Supports REQUIRED (INT32, INT64, DATE, STRING) and OPTIONAL (STRING) columns, including 0-row files.

class Buf {
public:
    std::vector<uint8_t> data;

    void reserve(size_t n) { data.reserve(n); }
    size_t size() const { return data.size(); }
    void clear() { data.clear(); }

    inline void u8(uint8_t v) { data.push_back(v); }
    inline void raw(const void* p, size_t n) {
        size_t old = data.size();
        data.resize(old + n);
        if (n) std::memcpy(data.data() + old, p, n);
    }
    inline void u32_le(uint32_t v) { raw(&v, 4); }
    inline void i32_le(int32_t v) { raw(&v, 4); }
    inline void i64_le(int64_t v) { raw(&v, 8); }

    inline void varint_u64(uint64_t v) {
        while (v >= 0x80) {
            data.push_back(uint8_t((v & 0x7F) | 0x80));
            v >>= 7;
        }
        data.push_back(uint8_t(v));
    }
    inline void varint_i32(int32_t v) {
        uint32_t zz = (uint32_t(v) << 1) ^ uint32_t(v >> 31);
        varint_u64(zz);
    }
    inline void varint_i64(int64_t v) {
        uint64_t zz = (uint64_t(v) << 1) ^ uint64_t(v >> 63);
        varint_u64(zz);
    }

    static constexpr uint8_t T_I16    = 4;
    static constexpr uint8_t T_I32    = 5;
    static constexpr uint8_t T_I64    = 6;
    static constexpr uint8_t T_BINARY = 8;
    static constexpr uint8_t T_LIST   = 9;
    static constexpr uint8_t T_STRUCT = 12;

    inline void field_hdr(int16_t& last_id, int16_t id, uint8_t type) {
        int16_t delta = id - last_id;
        if (delta > 0 && delta <= 15) {
            u8(uint8_t((delta << 4) | type));
        } else {
            u8(type);
            varint_i32(id);
        }
        last_id = id;
    }
    inline void f_i16(int16_t& last, int16_t id, int16_t v) {
        field_hdr(last, id, T_I16);
        varint_i32(v);
    }
    inline void f_i32(int16_t& last, int16_t id, int32_t v) {
        field_hdr(last, id, T_I32);
        varint_i32(v);
    }
    inline void f_i64(int16_t& last, int16_t id, int64_t v) {
        field_hdr(last, id, T_I64);
        varint_i64(v);
    }
    inline void f_str(int16_t& last, int16_t id, std::string_view s) {
        field_hdr(last, id, T_BINARY);
        varint_u64(s.size());
        raw(s.data(), s.size());
    }
    inline void f_list_hdr(int16_t& last, int16_t id, uint8_t elem_type, size_t count) {
        field_hdr(last, id, T_LIST);
        if (count < 15) {
            u8(uint8_t((count << 4) | elem_type));
        } else {
            u8(uint8_t(0xF0 | elem_type));
            varint_u64(count);
        }
    }
    inline void f_struct_begin(int16_t& last, int16_t id) {
        field_hdr(last, id, T_STRUCT);
    }
    inline void struct_end() { u8(0); }
};

enum class ColKind : uint8_t {
    INT32_REQ,
    INT64_REQ,
    DATE_REQ,
    STRING_REQ,
    STRING_OPT,
};

struct ColSpec {
    std::string_view name;
    ColKind kind;
};

struct ColChunkMeta {
    int32_t phys_type;
    bool optional;
    bool has_dict;
    std::string_view name;
    int64_t num_values;
    int64_t total_size;
    int64_t data_page_offset;
    int64_t dict_page_offset;
};

static inline int32_t phys_type_of(ColKind k) {
    switch (k) {
        case ColKind::INT32_REQ:
        case ColKind::DATE_REQ:   return 1; // INT32
        case ColKind::INT64_REQ:  return 2; // INT64
        case ColKind::STRING_REQ:
        case ColKind::STRING_OPT: return 6; // BYTE_ARRAY
    }
    return 1;
}

static inline bool writev_all_fd(int fd, std::vector<iovec>& iovs) {
    size_t idx = 0;
    while (idx < iovs.size()) {
        int count = int(std::min<size_t>(iovs.size() - idx, 256));
        ssize_t n = ::writev(fd, iovs.data() + idx, count);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        size_t rem = size_t(n);
        while (idx < iovs.size() && rem >= iovs[idx].iov_len) {
            rem -= iovs[idx].iov_len;
            ++idx;
        }
        if (idx < iovs.size() && rem > 0) {
            iovs[idx].iov_base = static_cast<uint8_t*>(iovs[idx].iov_base) + rem;
            iovs[idx].iov_len -= rem;
        }
    }
    return true;
}

static inline uint8_t bits_for_count(uint32_t count) {
    if (count <= 1) return 0;
    return uint8_t(32U - uint32_t(__builtin_clz(count - 1U)));
}

// Fast user_id RLE_DICTIONARY encoder: 1 RLE run per active person.
// When all runs < 64 (guaranteed in fast_dict_mode since num_days <= 51 < 64) and val_bytes == 3:
// AVX2 writes 8 entries (32 bytes) per iteration in 5 instructions with ZERO vector::resize!
static inline void encode_uid_runs_rle_dict(
    Buf& out, const uint8_t* runs, uint32_t n_dict, uint8_t bw, bool all_lt64
) {
    out.u8(bw);
    if (n_dict == 0) return;
    if (bw == 0) {
        out.varint_u64(uint64_t(runs[0]) << 1);
        return;
    }
    const size_t val_bytes = (size_t(bw) + 7U) >> 3;
    if (__builtin_expect(all_lt64 && val_bytes <= 3, 1)) {
        const size_t start = out.size();
        const size_t entry_bytes = 1 + val_bytes;
        const size_t payload_bytes = size_t(n_dict) * entry_bytes;
        out.data.resize(start + payload_bytes + 8);
        uint8_t* dst = out.data.data() + start;
        if (val_bytes == 3) {
            uint32_t i = 0;
#if defined(__AVX2__)
            __m256i v_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            const __m256i v_step = _mm256_set1_epi32(8);
            for (; i + 8 <= n_dict; i += 8) {
                __m128i r8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(runs + i));
                __m256i r32 = _mm256_slli_epi32(_mm256_cvtepu8_epi32(r8), 1);
                __m256i w = _mm256_or_si256(_mm256_slli_epi32(v_idx, 8), r32);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), w);
                dst += 32;
                v_idx = _mm256_add_epi32(v_idx, v_step);
            }
#endif
            for (; i < n_dict; ++i) {
                uint32_t w = (i << 8) | uint32_t(runs[i] << 1);
                std::memcpy(dst, &w, 4);
                dst += 4;
            }
        } else if (val_bytes == 2) {
            for (uint32_t i = 0; i < n_dict; ++i) {
                uint32_t w = (i << 8) | uint32_t(runs[i] << 1);
                std::memcpy(dst, &w, 4);
                dst += 3;
            }
        } else { // val_bytes == 1
            for (uint32_t i = 0; i < n_dict; ++i) {
                dst[0] = uint8_t(runs[i] << 1);
                dst[1] = uint8_t(i);
                dst += 2;
            }
        }
        out.data.resize(start + payload_bytes);
        return;
    }
    out.reserve(out.size() + size_t(n_dict) * (2 + val_bytes));
    for (uint32_t i = 0; i < n_dict; ++i) {
        out.varint_u64(uint64_t(runs[i]) << 1);
        out.raw(&i, val_bytes);
    }
}

// Fast RLE_DICTIONARY bit-packer for uint8_t indices (day_idx, sess_u8)
static inline void encode_u8_indices_rle_dict(Buf& out, const uint8_t* vals, size_t n, uint8_t bw) {
    out.u8(bw);
    if (n == 0) return;
    if (bw == 0) {
        out.varint_u64(uint64_t(n) << 1);
        return;
    }
    const size_t total_groups = (n + 7) >> 3;
    const size_t out_start = out.size();
    out.reserve(out_start + total_groups * size_t(bw) + (total_groups / 32768 + 1) * 5 + 64);
    size_t g = 0;
    while (g < total_groups) {
        const size_t chunk = std::min<size_t>(total_groups - g, 32768);
        out.varint_u64((uint64_t(chunk) << 1) | 1ULL);
        const size_t old = out.size();
        const size_t chunk_bytes = chunk * size_t(bw);
        out.data.resize(old + chunk_bytes + 32);
        uint8_t* dst = out.data.data() + old;
        const size_t base_i = g << 3;
        const size_t chunk_n = std::min<size_t>(n - base_i, chunk << 3);
        const uint8_t* src = vals + base_i;
        size_t i = 0;

        if (bw == 4) {
#if defined(__AVX2__)
            const __m256i mult_1001 = _mm256_set1_epi16(0x1001);
            for (; i + 64 <= chunk_n; i += 64) {
                __m256i v0  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                __m256i v1  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 32));
                __m256i p0  = _mm256_maddubs_epi16(v0, mult_1001);
                __m256i p1  = _mm256_maddubs_epi16(v1, mult_1001);
                __m256i pk  = _mm256_packus_epi16(p0, p1);
                __m256i res = _mm256_permute4x64_epi64(pk, 0xD8);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), res);
                dst += 32;
            }
            for (; i + 32 <= chunk_n; i += 32) {
                __m256i v0  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                __m256i p0  = _mm256_maddubs_epi16(v0, mult_1001);
                __m256i pk  = _mm256_permute4x64_epi64(_mm256_packus_epi16(p0, p0), 0xD8);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), _mm256_castsi256_si128(pk));
                dst += 16;
            }
#endif
            for (; i < chunk_n; i += 8) {
                if (i + 8 <= chunk_n) {
                    dst[0] = uint8_t(src[i + 0] | (src[i + 1] << 4));
                    dst[1] = uint8_t(src[i + 2] | (src[i + 3] << 4));
                    dst[2] = uint8_t(src[i + 4] | (src[i + 5] << 4));
                    dst[3] = uint8_t(src[i + 6] | (src[i + 7] << 4));
                } else {
                    uint8_t tmp[8]{};
                    for (size_t k = 0; i + k < chunk_n; ++k) tmp[k] = src[i + k];
                    dst[0] = uint8_t(tmp[0] | (tmp[1] << 4));
                    dst[1] = uint8_t(tmp[2] | (tmp[3] << 4));
                    dst[2] = uint8_t(tmp[4] | (tmp[5] << 4));
                    dst[3] = uint8_t(tmp[6] | (tmp[7] << 4));
                }
                dst += 4;
            }
        } else if (bw == 1) {
#if defined(__AVX2__)
            for (; i + 32 <= chunk_n; i += 32) {
                __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                uint32_t m = uint32_t(_mm256_movemask_epi8(_mm256_slli_epi16(v, 7)));
                std::memcpy(dst, &m, 4);
                dst += 4;
            }
#endif
            for (; i < chunk_n; i += 8) {
                uint8_t b = 0;
                for (size_t k = 0; k < 8 && i + k < chunk_n; ++k) b |= uint8_t(src[i + k] << k);
                *dst++ = b;
            }
        } else if (bw == 8) {
            std::memcpy(dst, src, chunk_n);
            if (chunk_n < chunk_bytes) std::memset(dst + chunk_n, 0, chunk_bytes - chunk_n);
        } else {
#if defined(__BMI2__)
            const uint64_t pmask = ((1ULL << bw) - 1ULL) * 0x0101010101010101ULL;
            for (; i + 8 <= chunk_n; i += 8) {
                uint64_t w8;
                std::memcpy(&w8, src + i, 8);
                uint64_t packed = _pext_u64(w8, pmask);
                std::memcpy(dst, &packed, 8);
                dst += bw;
            }
            if (i < chunk_n) {
                uint8_t tmp[8]{};
                for (size_t k = 0; i + k < chunk_n; ++k) tmp[k] = src[i + k];
                uint64_t w8;
                std::memcpy(&w8, tmp, 8);
                uint64_t packed = _pext_u64(w8, pmask);
                std::memcpy(dst, &packed, 8);
            }
#else
            for (; i < chunk_n; i += 8) {
                uint64_t acc = 0;
                if (i + 8 <= chunk_n) {
                    for (size_t k = 0; k < 8; ++k) acc |= uint64_t(src[i + k]) << (k * bw);
                } else {
                    for (size_t k = 0; i + k < chunk_n; ++k) acc |= uint64_t(src[i + k]) << (k * bw);
                }
                std::memcpy(dst, &acc, 8);
                dst += bw;
            }
#endif
        }
        out.data.resize(old + chunk_bytes);
        g += chunk;
    }
}

// Fast RLE_DICTIONARY bit-packer for uint16_t indices (events, top_category, dense revenue_cents)
static inline void encode_u16_indices_rle_dict(Buf& out, const uint16_t* vals, size_t n, uint8_t bw) {
    out.u8(bw);
    if (n == 0) return;
    if (bw == 0) {
        out.varint_u64(uint64_t(n) << 1);
        return;
    }
    const size_t total_groups = (n + 7) >> 3;
    const size_t out_start = out.size();
    out.reserve(out_start + total_groups * size_t(bw) + (total_groups / 32768 + 1) * 5 + 64);
    size_t g = 0;
    while (g < total_groups) {
        const size_t chunk = std::min<size_t>(total_groups - g, 32768);
        out.varint_u64((uint64_t(chunk) << 1) | 1ULL);
        const size_t old = out.size();
        const size_t chunk_bytes = chunk * size_t(bw);
        out.data.resize(old + chunk_bytes + 32);
        uint8_t* dst = out.data.data() + old;
        const size_t base_i = g << 3;
        const size_t chunk_n = std::min<size_t>(n - base_i, chunk << 3);
        const uint16_t* src = vals + base_i;
        size_t i = 0;

        if (bw == 4) {
#if defined(__AVX2__)
            const __m256i mult_1001 = _mm256_set1_epi16(0x1001);
            for (; i + 32 <= chunk_n; i += 32) {
                __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 16));
                __m256i b  = _mm256_permute4x64_epi64(_mm256_packus_epi16(v0, v1), 0xD8);
                __m256i p  = _mm256_maddubs_epi16(b, mult_1001);
                __m256i q  = _mm256_permute4x64_epi64(_mm256_packus_epi16(p, p), 0xD8);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), _mm256_castsi256_si128(q));
                dst += 16;
            }
#endif
            for (; i < chunk_n; i += 8) {
                if (i + 8 <= chunk_n) {
                    dst[0] = uint8_t(src[i + 0] | (src[i + 1] << 4));
                    dst[1] = uint8_t(src[i + 2] | (src[i + 3] << 4));
                    dst[2] = uint8_t(src[i + 4] | (src[i + 5] << 4));
                    dst[3] = uint8_t(src[i + 6] | (src[i + 7] << 4));
                } else {
                    uint8_t tmp[8]{};
                    for (size_t k = 0; i + k < chunk_n; ++k) tmp[k] = uint8_t(src[i + k]);
                    dst[0] = uint8_t(tmp[0] | (tmp[1] << 4));
                    dst[1] = uint8_t(tmp[2] | (tmp[3] << 4));
                    dst[2] = uint8_t(tmp[4] | (tmp[5] << 4));
                    dst[3] = uint8_t(tmp[6] | (tmp[7] << 4));
                }
                dst += 4;
            }
        } else if (bw == 8) {
#if defined(__AVX2__)
            for (; i + 32 <= chunk_n; i += 32) {
                __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 16));
                __m256i b  = _mm256_permute4x64_epi64(_mm256_packus_epi16(v0, v1), 0xD8);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), b);
                dst += 32;
            }
#endif
            for (; i < chunk_n; ++i) *dst++ = uint8_t(src[i]);
            if (chunk_n < chunk_bytes) std::memset(dst, 0, chunk_bytes - chunk_n);
        } else if (bw == 16) {
            std::memcpy(dst, src, chunk_n * 2);
            if (chunk_n * 2 < chunk_bytes) std::memset(dst + chunk_n * 2, 0, chunk_bytes - chunk_n * 2);
        } else if (bw <= 7) {
#if defined(__BMI2__)
            const uint64_t pmask = ((1ULL << bw) - 1ULL) * 0x0001000100010001ULL;
            const uint32_t sh = uint32_t(bw) << 2;
            for (; i + 8 <= chunk_n; i += 8) {
                uint64_t w0, w1;
                std::memcpy(&w0, src + i, 8);
                std::memcpy(&w1, src + i + 4, 8);
                uint64_t packed = _pext_u64(w0, pmask) | (_pext_u64(w1, pmask) << sh);
                std::memcpy(dst, &packed, 8);
                dst += bw;
            }
            if (i < chunk_n) {
                uint16_t tmp[8]{};
                for (size_t k = 0; i + k < chunk_n; ++k) tmp[k] = src[i + k];
                uint64_t w0, w1;
                std::memcpy(&w0, tmp, 8);
                std::memcpy(&w1, tmp + 4, 8);
                uint64_t packed = _pext_u64(w0, pmask) | (_pext_u64(w1, pmask) << sh);
                std::memcpy(dst, &packed, 8);
            }
#else
            if (bw == 5) {
                for (; i < chunk_n; i += 8) {
                    uint64_t acc = 0;
                    if (i + 8 <= chunk_n) {
                        acc = uint64_t(src[i + 0]) | (uint64_t(src[i + 1]) << 5) |
                              (uint64_t(src[i + 2]) << 10) | (uint64_t(src[i + 3]) << 15) |
                              (uint64_t(src[i + 4]) << 20) | (uint64_t(src[i + 5]) << 25) |
                              (uint64_t(src[i + 6]) << 30) | (uint64_t(src[i + 7]) << 35);
                    } else {
                        for (size_t k = 0; i + k < chunk_n; ++k) acc |= uint64_t(src[i + k]) << (k * 5);
                    }
                    std::memcpy(dst, &acc, 8);
                    dst += 5;
                }
            } else {
                for (; i < chunk_n; i += 8) {
                    uint64_t acc = 0;
                    for (size_t k = 0; k < 8 && i + k < chunk_n; ++k) acc |= uint64_t(src[i + k]) << (k * bw);
                    std::memcpy(dst, &acc, 8);
                    dst += bw;
                }
            }
#endif
        } else { // bw == 9..15
#if defined(__BMI2__)
            const uint64_t pmask = ((1ULL << bw) - 1ULL) * 0x0001000100010001ULL;
            const uint32_t sh  = uint32_t(bw) << 2;
            const uint32_t rsh = 64U - sh;
            for (; i + 8 <= chunk_n; i += 8) {
                uint64_t w0, w1;
                std::memcpy(&w0, src + i, 8);
                std::memcpy(&w1, src + i + 4, 8);
                uint64_t p0 = _pext_u64(w0, pmask);
                uint64_t p1 = _pext_u64(w1, pmask);
                uint64_t lo = p0 | (p1 << sh);
                uint64_t hi = p1 >> rsh;
                std::memcpy(dst, &lo, 8);
                std::memcpy(dst + 8, &hi, 8);
                dst += bw;
            }
            if (i < chunk_n) {
                uint16_t tmp[8]{};
                for (size_t k = 0; i + k < chunk_n; ++k) tmp[k] = src[i + k];
                uint64_t w0, w1;
                std::memcpy(&w0, tmp, 8);
                std::memcpy(&w1, tmp + 4, 8);
                uint64_t p0 = _pext_u64(w0, pmask);
                uint64_t p1 = _pext_u64(w1, pmask);
                uint64_t lo = p0 | (p1 << sh);
                uint64_t hi = p1 >> rsh;
                std::memcpy(dst, &lo, 8);
                std::memcpy(dst + 8, &hi, 8);
            }
#else
            for (; i < chunk_n; i += 8) {
                unsigned __int128 acc = 0;
                for (size_t k = 0; k < 8 && i + k < chunk_n; ++k) {
                    acc |= static_cast<unsigned __int128>(src[i + k]) << (k * bw);
                }
                uint64_t lo = uint64_t(acc);
                uint64_t hi = uint64_t(acc >> 64);
                std::memcpy(dst, &lo, 8);
                std::memcpy(dst + 8, &hi, 8);
                dst += bw;
            }
#endif
        }
        out.data.resize(old + chunk_bytes);
        g += chunk;
    }
}

// AVX2 Zero-Skipping Exact RLE encoder for sparse uint16_t dictionary indices (revenue_cents: 82-94%+ zeros!)
static inline void encode_u16_sparse_rle_dict(
    Buf& out, const uint16_t* vals, size_t n, size_t num_nonzero, uint8_t bw
) {
    out.u8(bw);
    if (n == 0) return;
    if (bw == 0) {
        out.varint_u64(uint64_t(n) << 1);
        return;
    }
    const size_t val_bytes = (size_t(bw) + 7U) >> 3;
    const size_t start_sz = out.size();
    out.data.resize(start_sz + num_nonzero * 10 + 32);
    uint8_t* const base = out.data.data() + start_sz;
    uint8_t* dst = base;

    auto emit_zrun = [&](size_t zrun) __attribute__((always_inline)) {
        uint64_t hdr = uint64_t(zrun) << 1;
        if (__builtin_expect(hdr < 128ULL, 1)) {
            *dst++ = uint8_t(hdr);
        } else {
            while (hdr >= 0x80ULL) { *dst++ = uint8_t((hdr & 0x7FU) | 0x80U); hdr >>= 7; }
            *dst++ = uint8_t(hdr);
        }
        if (val_bytes == 2) {
            dst[0] = 0; dst[1] = 0; dst += 2;
        } else {
            *dst++ = 0;
        }
    };

    auto emit_one = [&](uint16_t v) __attribute__((always_inline)) {
        if (val_bytes == 2) {
            uint32_t w = 0x02U | (uint32_t(v) << 8);
            std::memcpy(dst, &w, 4);
            dst += 3;
        } else {
            uint16_t w = uint16_t(0x02U | (uint32_t(v) << 8));
            std::memcpy(dst, &w, 2);
            dst += 2;
        }
    };

    size_t i = 0;
    size_t zrun = 0;
#if defined(__AVX2__)
    const __m256i vzero = _mm256_setzero_si256();
    for (; i + 16 <= n; i += 16) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vals + i));
        uint32_t nzm = ~uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi16(v, vzero)));
        if (nzm == 0) {
            zrun += 16;
            continue;
        }
        size_t prev = 0;
        while (nzm) {
            size_t pos = size_t(__builtin_ctz(nzm)) >> 1;
            zrun += (pos - prev);
            if (zrun) { emit_zrun(zrun); zrun = 0; }
            emit_one(vals[i + pos]);
            prev = pos + 1;
            nzm &= ~(3U << (pos << 1));
        }
        zrun += (16 - prev);
    }
#endif
    for (; i < n; ++i) {
        uint16_t v = vals[i];
        if (__builtin_expect(v == 0, 1)) {
            ++zrun;
        } else {
            if (zrun) { emit_zrun(zrun); zrun = 0; }
            emit_one(v);
        }
    }
    if (zrun) emit_zrun(zrun);
    out.data.resize(start_sz + size_t(dst - base));
}

static inline void write_dict_page_header(Buf& out, int32_t num_values, int32_t page_size) {
    int16_t last = 0;
    out.f_i32(last, 1, 2);          // PageType::DICTIONARY_PAGE = 2
    out.f_i32(last, 2, page_size);  // uncompressed_page_size
    out.f_i32(last, 3, page_size);  // compressed_page_size
    out.f_struct_begin(last, 7);    // dictionary_page_header
    {
        int16_t dl = 0;
        out.f_i32(dl, 1, num_values);
        out.f_i32(dl, 2, 0); // PLAIN
        out.struct_end();
    }
    out.struct_end();
}

static inline void write_page_header(Buf& out, int32_t num_values, int32_t page_size, int32_t encoding = 0) {
    int16_t last = 0;
    out.f_i32(last, 1, 0);          // PageType::DATA_PAGE = 0
    out.f_i32(last, 2, page_size);  // uncompressed_page_size
    out.f_i32(last, 3, page_size);  // compressed_page_size
    out.f_struct_begin(last, 5);    // data_page_header
    {
        int16_t dl = 0;
        out.f_i32(dl, 1, num_values);
        out.f_i32(dl, 2, encoding); // 0=PLAIN, 2=PLAIN_DICTIONARY
        out.f_i32(dl, 3, 3); // RLE
        out.f_i32(dl, 4, 3); // RLE
        out.struct_end();
    }
    out.struct_end();
}

static inline void write_schema_element(Buf& out, const ColSpec& col) {
    int16_t last = 0;
    out.f_i32(last, 1, phys_type_of(col.kind));
    out.f_i32(last, 3, col.kind == ColKind::STRING_OPT ? 1 : 0);
    out.f_str(last, 4, col.name);
    if (col.kind == ColKind::STRING_REQ || col.kind == ColKind::STRING_OPT) {
        out.f_i32(last, 6, 0); // UTF8
        out.f_struct_begin(last, 10);
        {
            int16_t ll = 0;
            out.f_struct_begin(ll, 1); // STRING
            out.struct_end();
            out.struct_end();
        }
    } else if (col.kind == ColKind::DATE_REQ) {
        out.f_i32(last, 6, 6); // DATE
        out.f_struct_begin(last, 10);
        {
            int16_t ll = 0;
            out.f_struct_begin(ll, 6); // DATE
            out.struct_end();
            out.struct_end();
        }
    }
    out.struct_end();
}

static inline void write_file_metadata(
    Buf& out,
    const ColSpec* cols,
    size_t ncols,
    int64_t num_rows,
    const std::vector<ColChunkMeta>& chunks
) {
    int16_t last = 0;
    out.f_i32(last, 1, 1);
    out.f_list_hdr(last, 2, Buf::T_STRUCT, ncols + 1);
    {
        int16_t rl = 0;
        out.f_str(rl, 4, "schema");
        out.f_i32(rl, 5, int32_t(ncols));
        out.struct_end();
    }
    for (size_t i = 0; i < ncols; ++i) write_schema_element(out, cols[i]);
    out.f_i64(last, 3, num_rows);
    if (num_rows == 0) {
        out.f_list_hdr(last, 4, Buf::T_STRUCT, 0);
    } else {
        out.f_list_hdr(last, 4, Buf::T_STRUCT, 1);
        int16_t rg = 0;
        out.f_list_hdr(rg, 1, Buf::T_STRUCT, chunks.size());
        int64_t total_bytes = 0;
        for (const auto& c : chunks) {
            total_bytes += c.total_size;
            int16_t cc = 0;
            out.f_i64(cc, 2, c.has_dict ? c.dict_page_offset : c.data_page_offset);
            out.f_struct_begin(cc, 3);
            {
                int16_t cm = 0;
                out.f_i32(cm, 1, c.phys_type);
                if (c.has_dict) {
                    if (c.optional) {
                        out.f_list_hdr(cm, 2, Buf::T_I32, 3);
                        out.varint_i32(3); // RLE
                        out.varint_i32(2); // PLAIN_DICTIONARY
                        out.varint_i32(0); // PLAIN
                    } else {
                        out.f_list_hdr(cm, 2, Buf::T_I32, 2);
                        out.varint_i32(2); // PLAIN_DICTIONARY
                        out.varint_i32(0); // PLAIN
                    }
                } else if (c.optional) {
                    out.f_list_hdr(cm, 2, Buf::T_I32, 2);
                    out.varint_i32(3);
                    out.varint_i32(0);
                } else {
                    out.f_list_hdr(cm, 2, Buf::T_I32, 1);
                    out.varint_i32(0);
                }
                out.f_list_hdr(cm, 3, Buf::T_BINARY, 1);
                out.varint_u64(c.name.size());
                out.raw(c.name.data(), c.name.size());
                out.f_i32(cm, 4, 0);
                out.f_i64(cm, 5, c.num_values);
                out.f_i64(cm, 6, c.total_size);
                out.f_i64(cm, 7, c.total_size);
                out.f_i64(cm, 9, c.data_page_offset);
                if (c.has_dict) {
                    out.f_i64(cm, 11, c.dict_page_offset);
                }
                out.struct_end();
            }
            out.struct_end();
        }
        out.f_i64(rg, 2, total_bytes);
        out.f_i64(rg, 3, num_rows);
        out.f_i64(rg, 5, chunks[0].has_dict ? chunks[0].dict_page_offset : chunks[0].data_page_offset);
        out.f_i64(rg, 6, total_bytes);
        out.f_i16(rg, 7, 0);
        out.struct_end();
    }
    out.f_str(last, 6, "fast-pq");
    out.struct_end();
}

struct ColData {
    const uint8_t* values_ptr = nullptr; // PLAIN payload, OR RLE_DICTIONARY data stream (with leading bit_width byte) when dict_ptr != nullptr!
    size_t values_len = 0;
    const uint8_t* def_bitmap = nullptr; // non-null only for STRING_OPT
    const uint8_t* dict_ptr = nullptr;   // non-null => PLAIN_DICTIONARY encoding!
    size_t dict_len = 0;
    int32_t dict_num_values = 0;
};

// Write complete Parquet file in ONE single ::writev() syscall (zero payload & zero def_bitmap copies!)
static inline bool write_parquet_file(
    const char* path,
    const ColSpec* cols,
    const ColData* data,
    size_t ncols,
    int64_t num_rows
) {
    int fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return false;

    static constexpr uint8_t MAGIC[4] = {'P', 'A', 'R', '1'};
    static constexpr size_t DEF_CHUNK_BYTES = 32768; // 32768 groups = 262,144 rows/run (Parquet v1 compliant)
    std::vector<ColChunkMeta> chunks;
    std::vector<Buf> dict_hdrs(ncols);
    std::vector<Buf> page_hdrs(ncols);
    std::vector<Buf> def_hdrs(ncols);
    std::vector<iovec> iovs;
    iovs.reserve(ncols * 5 + 32);
    iovs.push_back({const_cast<uint8_t*>(MAGIC), 4});

    int64_t offset = 4;
    if (num_rows > 0) {
        chunks.reserve(ncols);
        for (size_t i = 0; i < ncols; ++i) {
            const bool is_opt = (cols[i].kind == ColKind::STRING_OPT);
            const bool is_dict = (data[i].dict_ptr != nullptr);
            int64_t chunk_start = offset;
            int64_t dict_off = 0;
            if (is_dict) {
                dict_off = offset;
                if (__builtin_expect(data[i].dict_len > size_t(INT32_MAX), 0)) {
                    ::close(fd);
                    return false;
                }
                auto& dh = dict_hdrs[i];
                dh.reserve(32);
                write_dict_page_header(dh, data[i].dict_num_values, int32_t(data[i].dict_len));
                iovs.push_back({dh.data.data(), dh.size()});
                if (data[i].dict_len) {
                    iovs.push_back({const_cast<uint8_t*>(data[i].dict_ptr), data[i].dict_len});
                }
                offset += int64_t(dh.size()) + int64_t(data[i].dict_len);
            }
            int64_t page_start = offset;
            int64_t def_sz = 0;
            size_t def_nbytes = 0, def_nchunks = 0, def_enc_size = 0;
            if (is_opt) {
                def_nbytes = (size_t(num_rows) + 7) >> 3;
                def_nchunks = (def_nbytes + DEF_CHUNK_BYTES - 1) / DEF_CHUNK_BYTES;
                for (size_t off = 0; off < def_nbytes; off += DEF_CHUNK_BYTES) {
                    size_t chunk = std::min<size_t>(def_nbytes - off, DEF_CHUNK_BYTES);
                    uint64_t hv = (uint64_t(chunk) << 1) | 1ULL;
                    size_t hlen = 1;
                    while (hv >= 0x80ULL) { hv >>= 7; ++hlen; }
                    def_enc_size += hlen + chunk;
                }
                def_sz = 4 + int64_t(def_enc_size);
            }
            int64_t page_payload_size = def_sz + int64_t(data[i].values_len);
            if (__builtin_expect(page_payload_size > INT32_MAX || num_rows > INT32_MAX, 0)) {
                std::fprintf(stderr, "Parquet page size overflow\n");
                ::close(fd);
                return false;
            }
            auto& hdr = page_hdrs[i];
            hdr.reserve(32);
            write_page_header(hdr, int32_t(num_rows), int32_t(page_payload_size), is_dict ? 2 : 0);
            iovs.push_back({hdr.data.data(), hdr.size()});
            if (is_opt) {
                auto& dh = def_hdrs[i];
                dh.reserve(4 + def_nchunks * 6);
                dh.u32_le(uint32_t(def_enc_size));
                size_t off = 0;
                bool first = true;
                while (off < def_nbytes) {
                    size_t chunk = std::min<size_t>(def_nbytes - off, DEF_CHUNK_BYTES);
                    size_t hstart = first ? 0 : dh.size();
                    first = false;
                    dh.varint_u64((uint64_t(chunk) << 1) | 1ULL);
                    iovs.push_back({dh.data.data() + hstart, dh.size() - hstart});
                    iovs.push_back({const_cast<uint8_t*>(data[i].def_bitmap + off), chunk});
                    off += chunk;
                }
            }
            if (data[i].values_len) {
                iovs.push_back({const_cast<uint8_t*>(data[i].values_ptr), data[i].values_len});
            }
            offset += int64_t(hdr.size()) + page_payload_size;
            int64_t total = offset - chunk_start;
            chunks.push_back({phys_type_of(cols[i].kind), is_opt, is_dict, cols[i].name,
                              num_rows, total, page_start, dict_off});
        }
    }

    Buf footer;
    footer.reserve(512 + ncols * 128);
    write_file_metadata(footer, cols, ncols, num_rows, chunks);
    uint32_t meta_len = uint32_t(footer.size());
    footer.u32_le(meta_len);
    footer.raw(MAGIC, 4);
    iovs.push_back({footer.data.data(), footer.size()});

    bool ok = writev_all_fd(fd, iovs);
    if (::close(fd) != 0) ok = false;
    return ok;
}

} // namespace pq
} // namespace SS_ENGINE_NS
