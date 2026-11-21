namespace ss_base { int run_engine(int argc, char** argv); }
#if defined(__x86_64__)
namespace ss_avx2 { int run_engine(int argc, char** argv); }
namespace ss_avx512 { int run_engine(int argc, char** argv); }
#endif

int main(int argc, char** argv) {
#if defined(__x86_64__)
    if (__builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") &&
        __builtin_cpu_supports("avx512dq")) {
        return ss_avx512::run_engine(argc, argv);
    }
    if (__builtin_cpu_supports("avx2") &&
        __builtin_cpu_supports("bmi2") &&
        __builtin_cpu_supports("popcnt")) {
        return ss_avx2::run_engine(argc, argv);
    }
#endif
    return ss_base::run_engine(argc, argv);
}
