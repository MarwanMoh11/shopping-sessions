# Ultra-fast multi-ISA static native C++20 engine for Databricks Shopping Sessions (v7).
# Built with GCC 14 as a 100% static FROM scratch binary (zero dynamic linker overhead!)
# with 256-bit AVX-512VL (zero ZMM Cascade Lake turbo downclock!) + BMI2 + PCLMULQDQ dispatch.
FROM gcc:14-bookworm AS builder

WORKDIR /src
COPY src/ /src/

RUN set -eux; \
    ARCH="$(uname -m)"; \
    FLAGS="-O3 -std=c++20 -DNDEBUG -fno-exceptions -fno-rtti -pthread -fno-plt -fno-stack-protector -fomit-frame-pointer -ffunction-sections -fdata-sections"; \
    if [ "$ARCH" = "x86_64" ]; then \
        g++ $FLAGS -march=x86-64 -c /src/main.cpp -o /src/main.o; \
        g++ $FLAGS -fvisibility=hidden -march=x86-64                                     -c /src/engine_base.cpp   -o /src/engine_base.o; \
        g++ $FLAGS -fvisibility=hidden -march=x86-64-v3 -mpclmul                         -c /src/engine_avx2.cpp   -o /src/engine_avx2.o; \
        g++ $FLAGS -fvisibility=hidden -march=x86-64-v4 -mtune=cascadelake -mpclmul -mprefer-vector-width=256 -c /src/engine_avx512.cpp -o /src/engine_avx512.o; \
        g++ $FLAGS -static -Wl,--gc-sections \
            /src/main.o /src/engine_base.o /src/engine_avx2.o /src/engine_avx512.o -o /app; \
    else \
        g++ $FLAGS -static -Wl,--gc-sections /src/main.cpp /src/engine_base.cpp -o /app; \
    fi; \
    strip -s /app; \
    mkdir -p /rootfs/data /rootfs/output /rootfs/scratch /rootfs/tmp

FROM scratch
COPY --from=builder /rootfs /
COPY --from=builder /app /app

ENTRYPOINT ["/app"]
