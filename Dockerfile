# Ultra-fast multi-ISA static native C++20 engine for Databricks Shopping Sessions (v16: Self-Contained GCC 14 PGO!).
# Built with GCC 14 as a 100% static FROM scratch binary (zero dynamic linker overhead!)
# with 256-bit AVX-512VL (zero ZMM Cascade Lake turbo downclock!) + BMI2 + PCLMULQDQ + PGO dispatch.
FROM gcc:14-bookworm AS builder

WORKDIR /src
COPY src/ /src/

RUN set -eux; \
    ARCH="$(uname -m)"; \
    FLAGS="-O3 -std=c++20 -DNDEBUG -fno-exceptions -fno-rtti -pthread -fno-plt -fno-pie -fno-stack-protector -fomit-frame-pointer -fno-semantic-interposition -ffunction-sections -fdata-sections"; \
    if [ "$ARCH" = "x86_64" ]; then \
        X86_JCC="-Wa,-mbranches-within-32B-boundaries -falign-functions=32 -falign-jumps=32 -falign-loops=32 -malign-data=cacheline"; \
        PGO_GEN="-fprofile-generate=/tmp/pgo_prof -fprofile-update=prefer-atomic -DSS_PGO_GEN"; \
        PGO_USE="-fprofile-use=/tmp/pgo_prof -fprofile-correction -fprofile-partial-training -Wno-missing-profile"; \
        mkdir -p /tmp/pgo_prof /tmp/pgo_data /tmp/pgo_out1 /tmp/pgo_out2; \
        g++ -O2 -std=c++20 /src/pgo_gen.cpp -o /tmp/pgo_gen && /tmp/pgo_gen /tmp/pgo_data; \
        g++ $FLAGS $PGO_GEN -march=x86-64 -c /src/main.cpp -o /src/main.o & p1=$!; \
        g++ $FLAGS $X86_JCC -fvisibility=hidden -march=x86-64 -c /src/engine_base.cpp -o /src/engine_base.o & p2=$!; \
        g++ $FLAGS $X86_JCC $PGO_GEN -fvisibility=hidden -march=x86-64-v3 -mtune=cascadelake -mpclmul -c /src/engine_avx2.cpp -o /src/engine_avx2.o & p3=$!; \
        g++ $FLAGS $X86_JCC $PGO_GEN -fvisibility=hidden -march=x86-64-v4 -mtune=cascadelake -mpclmul -mprefer-vector-width=256 -c /src/engine_avx512.cpp -o /src/engine_avx512.o & p4=$!; \
        wait $p1 $p2 $p3 $p4; \
        g++ $FLAGS -no-pie $PGO_GEN /src/main.o /src/engine_base.o /src/engine_avx2.o /src/engine_avx512.o -o /tmp/app_pgo; \
        PGO_ISA_OUT=/tmp/pgo_isa NCPU=4 /tmp/app_pgo /tmp/pgo_data /tmp/pgo_out1; \
        RAN_ISA="$(cat /tmp/pgo_isa)"; \
        echo "PGO_TRAINED_ISA=$RAN_ISA"; \
        if [ "$RAN_ISA" = "avx512" ]; then \
            FORCE_ISA=avx2 NCPU=4 /tmp/app_pgo /tmp/pgo_data /tmp/pgo_out2; \
            g++ $FLAGS -march=x86-64 -c /src/main.cpp -o /src/main.o & p1=$!; \
            g++ $FLAGS $X86_JCC $PGO_USE -fvisibility=hidden -march=x86-64-v3 -mtune=cascadelake -mpclmul -c /src/engine_avx2.cpp -o /src/engine_avx2.o & p3=$!; \
            g++ $FLAGS $X86_JCC $PGO_USE -fvisibility=hidden -march=x86-64-v4 -mtune=cascadelake -mpclmul -mprefer-vector-width=256 -c /src/engine_avx512.cpp -o /src/engine_avx512.o & p4=$!; \
            wait $p1 $p3 $p4; \
            g++ $FLAGS -no-pie -static -Wl,--gc-sections \
                /src/main.o /src/engine_base.o /src/engine_avx2.o /src/engine_avx512.o -o /app; \
        else \
            g++ $FLAGS -DSS_NO_AVX512 -march=x86-64 -c /src/main.cpp -o /src/main.o & p1=$!; \
            g++ $FLAGS $X86_JCC $PGO_USE -fvisibility=hidden -march=x86-64-v3 -mtune=cascadelake -mpclmul -c /src/engine_avx2.cpp -o /src/engine_avx2.o & p3=$!; \
            wait $p1 $p3; \
            g++ $FLAGS -no-pie -static -Wl,--gc-sections \
                /src/main.o /src/engine_base.o /src/engine_avx2.o -o /app; \
        fi; \
        rm -rf /tmp/pgo_prof /tmp/pgo_data /tmp/pgo_out1 /tmp/pgo_out2 /tmp/pgo_gen /tmp/app_pgo /tmp/pgo_isa; \
    else \
        g++ $FLAGS -static -Wl,--gc-sections /src/main.cpp /src/engine_base.cpp -o /app; \
    fi; \
    strip -s /app; \
    mkdir -p /rootfs/data /rootfs/output /rootfs/scratch /rootfs/tmp

FROM scratch
COPY --from=builder /rootfs /
COPY --from=builder /app /app

ENTRYPOINT ["/app"]
