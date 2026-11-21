namespace ss_base { __attribute__((visibility("default"))) int run_engine(int argc, char** argv); }
#if defined(__x86_64__)
namespace ss_avx2 { __attribute__((visibility("default"))) int run_engine(int argc, char** argv); }
namespace ss_avx512 { __attribute__((visibility("default"))) int run_engine(int argc, char** argv); }
#endif

int main(int argc, char** argv) {
#if defined(__x86_64__)
    if (__builtin_cpu_supports("x86-64-v4")) {
        return ss_avx512::run_engine(argc, argv);
    }
    if (__builtin_cpu_supports("x86-64-v3")) {
        return ss_avx2::run_engine(argc, argv);
    }
#endif
    return ss_base::run_engine(argc, argv);
}
