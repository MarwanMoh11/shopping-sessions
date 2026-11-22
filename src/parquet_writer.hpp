#pragma once
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
        int count = int(std::min<size_t>(iovs.size() - idx, 64));
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

static inline void encode_def_levels_bitpacked(Buf& out, const uint8_t* bitmap, size_t num_rows) {
    size_t nbytes = (num_rows + 7) >> 3;
    size_t full_chunks = nbytes / 63;
    size_t tail = nbytes % 63;
    size_t enc_size = full_chunks * 64 + (tail ? (tail + 1) : 0);
    out.u32_le(uint32_t(enc_size));
    size_t off = 0;
    for (size_t c = 0; c < full_chunks; ++c) {
        out.u8(uint8_t((63U << 1) | 1U));
        out.raw(bitmap + off, 63);
        off += 63;
    }
    if (tail) {
        out.u8(uint8_t((tail << 1) | 1U));
        out.raw(bitmap + off, tail);
    }
}

static inline uint8_t bits_for_count(uint32_t count) {
    if (count <= 1) return 0;
    return uint8_t(32U - uint32_t(__builtin_clz(count - 1U)));
}

static inline void encode_u16_indices_rle_dict(Buf& out, const uint16_t* vals, size_t n, uint8_t bw) {
    out.u8(bw);
    if (n == 0) return;
    if (bw == 0) {
        out.varint_u64(uint64_t(n) << 1);
        return;
    }
    size_t total_groups = (n + 7) >> 3;
    size_t out_start = out.size();
    out.reserve(out_start + total_groups * size_t(bw) + 64);
    size_t g = 0;
    while (g < total_groups) {
        size_t chunk = std::min<size_t>(total_groups - g, 32768);
        out.varint_u64((chunk << 1) | 1ULL);
        size_t old = out.size();
        out.data.resize(old + chunk * size_t(bw));
        uint8_t* dst = out.data.data() + old;
        size_t base_i = g << 3;
        if (bw == 4) {
            for (size_t c = 0; c < chunk; ++c) {
                size_t i = base_i + (c << 3);
                if (i + 8 <= n) {
                    dst[0] = uint8_t(vals[i + 0] | (vals[i + 1] << 4));
                    dst[1] = uint8_t(vals[i + 2] | (vals[i + 3] << 4));
                    dst[2] = uint8_t(vals[i + 4] | (vals[i + 5] << 4));
                    dst[3] = uint8_t(vals[i + 6] | (vals[i + 7] << 4));
                } else {
                    uint8_t tmp[8]{};
                    for (size_t k = 0; k < 8 && i + k < n; ++k) tmp[k] = uint8_t(vals[i + k]);
                    dst[0] = uint8_t(tmp[0] | (tmp[1] << 4));
                    dst[1] = uint8_t(tmp[2] | (tmp[3] << 4));
                    dst[2] = uint8_t(tmp[4] | (tmp[5] << 4));
                    dst[3] = uint8_t(tmp[6] | (tmp[7] << 4));
                }
                dst += 4;
            }
        } else if (bw == 5) {
            for (size_t c = 0; c < chunk; ++c) {
                size_t i = base_i + (c << 3);
                uint64_t acc = 0;
                if (i + 8 <= n) {
                    acc = uint64_t(vals[i + 0]) |
                          (uint64_t(vals[i + 1]) << 5) |
                          (uint64_t(vals[i + 2]) << 10) |
                          (uint64_t(vals[i + 3]) << 15) |
                          (uint64_t(vals[i + 4]) << 20) |
                          (uint64_t(vals[i + 5]) << 25) |
                          (uint64_t(vals[i + 6]) << 30) |
                          (uint64_t(vals[i + 7]) << 35);
                } else {
                    for (size_t k = 0; k < 8 && i + k < n; ++k) acc |= uint64_t(vals[i + k]) << (k * 5);
                }
                dst[0] = uint8_t(acc);
                dst[1] = uint8_t(acc >> 8);
                dst[2] = uint8_t(acc >> 16);
                dst[3] = uint8_t(acc >> 24);
                dst[4] = uint8_t(acc >> 32);
                dst += 5;
            }
        } else {
            for (size_t c = 0; c < chunk; ++c) {
                size_t i = base_i + (c << 3);
                uint64_t acc = 0;
                int bits = 0;
                for (size_t k = 0; k < 8; ++k) {
                    uint32_t v = (i + k < n) ? uint32_t(vals[i + k]) : 0U;
                    acc |= uint64_t(v) << bits;
                    bits += bw;
                    while (bits >= 8) { *dst++ = uint8_t(acc); acc >>= 8; bits -= 8; }
                }
            }
        }
        g += chunk;
    }
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

// Write complete Parquet file in ONE single ::writev() syscall (zero extra payload copies!)
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
    std::vector<ColChunkMeta> chunks;
    std::vector<Buf> dict_hdrs(ncols);
    std::vector<Buf> page_hdrs(ncols);
    std::vector<Buf> def_bufs(ncols);
    std::vector<iovec> iovs;
    iovs.reserve(ncols * 5 + 4);
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
            if (is_opt) {
                auto& db = def_bufs[i];
                size_t nbytes = (size_t(num_rows) + 7) >> 3;
                db.reserve(nbytes + (nbytes / 63) + 16);
                encode_def_levels_bitpacked(db, data[i].def_bitmap, size_t(num_rows));
                def_sz = int64_t(db.size());
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
                iovs.push_back({def_bufs[i].data.data(), def_bufs[i].size()});
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
