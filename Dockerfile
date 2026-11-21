# Ultra-fast multi-ISA native C++20 engine for Databricks Shopping Sessions (v5).
# Built with GCC 14 + glibc packaged in a ~1.6 MB FROM scratch container with exact
# runtime CPUID dispatch across x86-64-v4 (AVX-512), x86-64-v3 (AVX2+BMI2), and x86-64.
FROM gcc:14-bookworm AS builder

WORKDIR /src
COPY src/ /src/

RUN set -eux; \
    ARCH="$(uname -m)"; \
    FLAGS="-O3 -std=c++20 -DNDEBUG -fno-exceptions -fno-rtti -pthread -fno-plt"; \
    if [ "$ARCH" = "x86_64" ]; then \
        g++ $FLAGS -march=x86-64 -c /src/main.cpp -o /src/main.o; \
        g++ $FLAGS -fvisibility=hidden -march=x86-64    -c /src/engine_base.cpp   -o /src/engine_base.o; \
        g++ $FLAGS -fvisibility=hidden -march=x86-64-v3 -c /src/engine_avx2.cpp   -o /src/engine_avx2.o; \
        g++ $FLAGS -fvisibility=hidden -march=x86-64-v4 -c /src/engine_avx512.cpp -o /src/engine_avx512.o; \
        g++ $FLAGS -Wl,--as-needed -static-libstdc++ -static-libgcc \
            /src/main.o /src/engine_base.o /src/engine_avx2.o /src/engine_avx512.o -o /app; \
    else \
        g++ $FLAGS -Wl,--as-needed -static-libstdc++ -static-libgcc /src/main.cpp /src/engine_base.cpp -o /app; \
    fi; \
    strip /app; \
    mkdir -p /rootfs/data /rootfs/output /rootfs/scratch /rootfs/tmp; \
    ldd /app | awk '{for(i=1;i<=NF;i++) if($i ~ /^\//) print $i}' | sort -u | while read -r lib; do \
        mkdir -p "/rootfs$(dirname "$lib")"; \
        cp -L "$lib" "/rootfs$lib"; \
    done

FROM scratch
COPY --from=builder /rootfs /
COPY --from=builder /app /app

ENTRYPOINT ["/app"]
