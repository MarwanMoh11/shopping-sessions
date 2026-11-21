#pragma once
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace pq {

// Minimal, zero-dependency Parquet v1 PLAIN/UNCOMPRESSED writer.
// Supports REQUIRED (INT32, INT64, DATE, STRING) and OPTIONAL (STRING) columns,
// including 0-row files with exact schema. Verified against Polars, PyArrow, DuckDB.

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

    // Thrift Compact Protocol helpers
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
    std::string_view name;
    int64_t num_values;
    int64_t total_size;
    int64_t data_page_offset;
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

static inline void write_all_fd(int fd, const uint8_t* ptr, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t n = ::write(fd, ptr + off, len - off);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            break;
        }
        off += size_t(n);
    }
}

// Build RLE/Bit-Packed definition levels payload (with 4-byte LE byte-length prefix)
// from a packed LSB-first bitmap of `num_rows` bits (1 = present, 0 = null).
// Uses 63-byte chunks (504 values -> 1-byte varint header) for universal reader compatibility.
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

static inline void write_page_header(Buf& out, int32_t num_values, int32_t page_size) {
    int16_t last = 0;
    out.f_i32(last, 1, 0);          // PageType::DATA_PAGE = 0
    out.f_i32(last, 2, page_size);  // uncompressed_page_size
    out.f_i32(last, 3, page_size);  // compressed_page_size
    out.f_struct_begin(last, 5);    // data_page_header
    {
        int16_t dl = 0;
        out.f_i32(dl, 1, num_values);
        out.f_i32(dl, 2, 0); // PLAIN
        out.f_i32(dl, 3, 3); // RLE (definition_level_encoding)
        out.f_i32(dl, 4, 3); // RLE (repetition_level_encoding)
        out.struct_end();
    }
    out.struct_end();
}

static inline void write_schema_element(Buf& out, const ColSpec& col) {
    int16_t last = 0;
    out.f_i32(last, 1, phys_type_of(col.kind)); // type
    out.f_i32(last, 3, col.kind == ColKind::STRING_OPT ? 1 : 0); // OPTIONAL=1, REQUIRED=0
    out.f_str(last, 4, col.name);
    if (col.kind == ColKind::STRING_REQ || col.kind == ColKind::STRING_OPT) {
        out.f_i32(last, 6, 0); // ConvertedType::UTF8 = 0
        out.f_struct_begin(last, 10); // LogicalType union
        {
            int16_t ll = 0;
            out.f_struct_begin(ll, 1); // STRING
            out.struct_end();
            out.struct_end();
        }
    } else if (col.kind == ColKind::DATE_REQ) {
        out.f_i32(last, 6, 6); // ConvertedType::DATE = 6
        out.f_struct_begin(last, 10); // LogicalType union
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
    out.f_i32(last, 1, 1); // version = 1
    // field 2: schema (root + ncols)
    out.f_list_hdr(last, 2, Buf::T_STRUCT, ncols + 1);
    {
        // Root SchemaElement
        int16_t rl = 0;
        out.f_str(rl, 4, "schema");
        out.f_i32(rl, 5, int32_t(ncols));
        out.struct_end();
    }
    for (size_t i = 0; i < ncols; ++i) {
        write_schema_element(out, cols[i]);
    }
    out.f_i64(last, 3, num_rows);
    // field 4: row_groups
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
            out.f_i64(cc, 2, c.data_page_offset); // file_offset
            out.f_struct_begin(cc, 3); // meta_data
            {
                int16_t cm = 0;
                out.f_i32(cm, 1, c.phys_type);
                if (c.optional) {
                    out.f_list_hdr(cm, 2, Buf::T_I32, 2);
                    out.varint_i32(3); // RLE
                    out.varint_i32(0); // PLAIN
                } else {
                    out.f_list_hdr(cm, 2, Buf::T_I32, 1);
                    out.varint_i32(0); // PLAIN
                }
                out.f_list_hdr(cm, 3, Buf::T_BINARY, 1);
                out.varint_u64(c.name.size());
                out.raw(c.name.data(), c.name.size());
                out.f_i32(cm, 4, 0); // UNCOMPRESSED
                out.f_i64(cm, 5, c.num_values);
                out.f_i64(cm, 6, c.total_size);
                out.f_i64(cm, 7, c.total_size);
                out.f_i64(cm, 9, c.data_page_offset);
                out.struct_end();
            }
            out.struct_end();
        }
        out.f_i64(rg, 2, total_bytes);
        out.f_i64(rg, 3, num_rows);
        out.f_i64(rg, 5, chunks[0].data_page_offset);
        out.f_i64(rg, 6, total_bytes);
        out.f_i16(rg, 7, 0);
        out.struct_end();
    }
    out.f_str(last, 6, "fast-pq");
    out.struct_end();
}

// Column payload descriptor for direct streaming write (zero extra copies of big column buffers!)
struct ColData {
    const uint8_t* values_ptr;
    size_t values_len;
    const uint8_t* def_bitmap; // non-null only for STRING_OPT
};

// Write a complete Parquet file directly from column buffers with minimal syscalls.
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
    write_all_fd(fd, MAGIC, 4);
    int64_t offset = 4;

    std::vector<ColChunkMeta> chunks;
    Buf tmp;
    if (num_rows > 0) {
        chunks.reserve(ncols);
        for (size_t i = 0; i < ncols; ++i) {
            int64_t page_start = offset;
            if (cols[i].kind == ColKind::STRING_OPT) {
                Buf def_buf;
                def_buf.reserve(((size_t(num_rows) + 7) >> 3) + 64);
                encode_def_levels_bitpacked(def_buf, data[i].def_bitmap, size_t(num_rows));
                int64_t page_payload_size = int64_t(def_buf.size() + data[i].values_len);
                tmp.clear();
                write_page_header(tmp, int32_t(num_rows), int32_t(page_payload_size));
                write_all_fd(fd, tmp.data.data(), tmp.size());
                write_all_fd(fd, def_buf.data.data(), def_buf.size());
                if (data[i].values_len) {
                    write_all_fd(fd, data[i].values_ptr, data[i].values_len);
                }
                int64_t total = int64_t(tmp.size()) + page_payload_size;
                offset += total;
                chunks.push_back({phys_type_of(cols[i].kind), true, cols[i].name, num_rows, total, page_start});
            } else {
                int64_t page_payload_size = int64_t(data[i].values_len);
                tmp.clear();
                write_page_header(tmp, int32_t(num_rows), int32_t(page_payload_size));
                write_all_fd(fd, tmp.data.data(), tmp.size());
                if (data[i].values_len) {
                    write_all_fd(fd, data[i].values_ptr, data[i].values_len);
                }
                int64_t total = int64_t(tmp.size()) + page_payload_size;
                offset += total;
                chunks.push_back({phys_type_of(cols[i].kind), false, cols[i].name, num_rows, total, page_start});
            }
        }
    }

    Buf footer;
    footer.reserve(512 + ncols * 128);
    write_file_metadata(footer, cols, ncols, num_rows, chunks);
    uint32_t meta_len = uint32_t(footer.size());
    footer.u32_le(meta_len);
    footer.raw(MAGIC, 4);
    write_all_fd(fd, footer.data.data(), footer.size());
    ::close(fd);
    return true;
}

} // namespace pq
