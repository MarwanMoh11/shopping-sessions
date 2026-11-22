#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#ifdef SS_PGO_GEN
extern "C" void __gcov_dump(void);
static void record_pgo_isa(const char* isa) {
    if (const char* p = std::getenv("PGO_ISA_OUT")) {
        if (FILE* f = std::fopen(p, "wb")) {
            std::fputs(isa, f);
            std::fclose(f);
        }
    }
}
#endif

extern "C" [[noreturn]] void ss_on_exit(int code) {
#ifdef SS_PGO_GEN
    __gcov_dump();
#endif
    ::_exit(code);
}

namespace ss_base { __attribute__((visibility("default"))) int run_engine(int argc, char** argv); }
#if defined(__x86_64__)
namespace ss_avx2 { __attribute__((visibility("default"))) int run_engine(int argc, char** argv); }
#ifndef SS_NO_AVX512
namespace ss_avx512 { __attribute__((visibility("default"))) int run_engine(int argc, char** argv); }
#endif
#endif

int main(int argc, char** argv) {
#if defined(__x86_64__)
#ifdef SS_PGO_GEN
    if (const char* f = std::getenv("FORCE_ISA")) {
        if (std::strcmp(f, "avx2") == 0) { record_pgo_isa("avx2"); return ss_avx2::run_engine(argc, argv); }
        if (std::strcmp(f, "base") == 0) { record_pgo_isa("base"); return ss_base::run_engine(argc, argv); }
    }
#endif
#ifndef SS_NO_AVX512
    if (__builtin_cpu_supports("x86-64-v4") && __builtin_cpu_supports("pclmul")) {
#ifdef SS_PGO_GEN
        record_pgo_isa("avx512");
#endif
        return ss_avx512::run_engine(argc, argv);
    }
#endif
    if (__builtin_cpu_supports("x86-64-v3") && __builtin_cpu_supports("pclmul")) {
#ifdef SS_PGO_GEN
        record_pgo_isa("avx2");
#endif
        return ss_avx2::run_engine(argc, argv);
    }
#endif
#ifdef SS_PGO_GEN
    record_pgo_isa("base");
#endif
    return ss_base::run_engine(argc, argv);
}
