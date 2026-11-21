# Ultra-minimal native static C++20 engine for Databricks Shopping Sessions.
# Built natively on scorer (linux/amd64) with runtime AVX2/SSE2 CPUID dispatch (no -march=native!).
FROM alpine:3.21 AS builder

RUN apk add --no-cache g++ musl-dev

WORKDIR /src
COPY src/ /src/

# Compile portable x86-64 static binary (AVX2 hot path selected via runtime CPUID dispatch!)
RUN g++ -O3 -std=c++20 -DNDEBUG -fno-exceptions -fno-rtti -pthread -static \
    /src/main.cpp -o /app && strip /app

# Create empty mountpoint dirs for --read-only rootfs compatibility
RUN mkdir -p /rootfs/data /rootfs/output /rootfs/scratch /rootfs/tmp

FROM scratch
COPY --from=builder /rootfs /
COPY --from=builder /app /app

ENTRYPOINT ["/app"]
