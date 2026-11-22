// Realistic v4 PGO training data generator (~700K events, 175K clicks, 32K links, 21 categories, 40 campaigns).
// Disjoint small user clusters (NO giant-component percolation!), 40% 1-session + 55% multi-session + heavy users!
// Compiled and run ONLY inside Dockerfile builder stage to generate GCC 14 .gcda profile data, then deleted!
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

static uint64_t rng_s = 0x9e3779b97f4a7c15ULL;
static inline uint64_t rng() {
    uint64_t x = rng_s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rng_s = x;
    return x * 2685821657736338717ULL;
}

static void mk_short_id(char pfx, uint64_t v, char* out) {
    static constexpr char B36[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    out[0] = pfx;
    for (int i = 8; i >= 1; --i) { out[i] = B36[v % 36]; v /= 36; }
    out[9] = '\0';
}

static void mk_hex_id(char pfx, uint64_t lo, uint64_t hi, char* out) {
    static constexpr char HEX[] = "0123456789abcdef";
    out[0] = pfx;
    for (int i = 0; i < 16; ++i) {
        out[1 + i]  = HEX[(lo >> (i * 4)) & 15];
        out[17 + i] = HEX[(hi >> (i * 4)) & 15];
    }
    out[33] = '\0';
}

static void mk_uid(uint32_t u, char* out) {
    if ((u & 7U) == 0) mk_hex_id('u', uint64_t(u) * 0xbf58476d1ce4e5b9ULL, uint64_t(u) ^ 0x94d049bb133111ebULL, out);
    else mk_short_id('u', uint64_t(u) + 1000000ULL, out);
}

static void fmt_f1(int64_t sec, char* out) {
    int64_t rel = sec - 1772323200LL;
    int day = int(rel / 86400LL) + 1;
    int rem = int(rel % 86400LL);
    if (rem < 0) { rem += 86400; --day; }
    int mo = 3;
    if (day < 1) { mo = 2; day += 28; }
    else if (day > 31) { mo = 4; day -= 31; }
    std::snprintf(out, 32, "2026-%02d-%02dT%02d:%02d:%02dZ", mo, day, rem / 3600, (rem / 60) % 60, rem % 60);
}

static void fmt_f2(int64_t sec, int off_h, char* out) {
    int64_t loc = sec + int64_t(off_h) * 3600LL;
    int64_t rel = loc - 1772323200LL;
    int day = int(rel / 86400LL) + 1;
    int rem = int(rel % 86400LL);
    if (rem < 0) { rem += 86400; --day; }
    int mo = 3;
    if (day < 1) { mo = 2; day += 28; }
    else if (day > 31) { mo = 4; day -= 31; }
    char sgn = off_h >= 0 ? '+' : '-';
    int ah = off_h >= 0 ? off_h : -off_h;
    std::snprintf(out, 32, "2026-%02d-%02dT%02d:%02d:%02d%c%02d:00", mo, day, rem / 3600, (rem / 60) % 60, rem % 60, sgn, ah);
}

static inline void qwrap(FILE* f, const char* s, bool q) {
    if (q) { std::fputc('"', f); std::fputs(s, f); std::fputc('"', f); }
    else std::fputs(s, f);
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    std::string dir = argv[1];
    ::mkdir(dir.c_str(), 0755);
    ::mkdir((dir + "/links").c_str(), 0755);
    ::mkdir((dir + "/clicks").c_str(), 0755);
    ::mkdir((dir + "/events").c_str(), 0755);

    const int64_t W0 = 1772323200LL; // 2026-03-01T00:00:00Z
    const int64_t W_DUR = 14LL * 86400LL;
    {
        FILE* f = std::fopen((dir + "/manifest.json").c_str(), "wb");
        std::fputs("{\"window_start\":\"2026-03-01T00:00:00Z\",\"window_end\":\"2026-03-15T00:00:00Z\"}\n", f);
        std::fclose(f);
    }
    {
        // 21 categories => bw == 5 in Parquet RLE_DICTIONARY (matches real v4!)
        static const char* CATS[21] = {
            "automotive","baby","beauty","books","clothing","electronics","fashion","furniture",
            "games","garden","grocery","health","home","jewelry","music","office",
            "outdoors","pets","sports","toys","travel"
        };
        FILE* f = std::fopen((dir + "/products.csv").c_str(), "wb");
        std::fputs("product_id,category\n", f);
        for (int i = 1; i <= 5000; ++i) std::fprintf(f, "p%d,%s\n", i, CATS[i % 21]);
        std::fclose(f);
    }
    {
        static const char* CHS[4] = {"search","social","email","display"};
        FILE* f = std::fopen((dir + "/campaigns.csv").c_str(), "wb");
        std::fputs("campaign_id,channel\n", f);
        for (int i = 1; i <= 40; ++i) std::fprintf(f, "c%03d,%s\n", i, CHS[i & 3]);
        std::fclose(f);
    }

    const uint32_t N_USERS = 80000;
    // Disjoint small identity clusters (size 2..4 + 60 hubs of size 8): ZERO giant component!
    {
        FILE* f = std::fopen((dir + "/links/part-00000.csv").c_str(), "wb");
        std::fputs("user_id_a,user_id_b\n", f);
        char ua[40], ub[40];
        // 60 hubs of size 8: users [0 .. 479]
        for (uint32_t h = 0; h < 60; ++h) {
            uint32_t base = h * 8;
            for (uint32_t j = 1; j < 8; ++j) {
                mk_uid(base, ua); mk_uid(base + j, ub);
                qwrap(f, ua, (rng() % 5) == 0); std::fputc(',', f);
                qwrap(f, ub, (rng() % 5) == 0); std::fputc('\n', f);
            }
        }
        // Disjoint 2..4 chains across users [480 .. 36480) (~27K edges, realistic small components)
        for (uint32_t base = 480; base + 4 < 36480; base += 4) {
            uint32_t sz = 2 + uint32_t(rng() % 3); // 2, 3, or 4
            for (uint32_t j = 0; j + 1 < sz; ++j) {
                mk_uid(base + j, ua); mk_uid(base + j + 1, ub);
                qwrap(f, ua, (rng() % 5) == 0); std::fputc(',', f);
                qwrap(f, ub, (rng() % 5) == 0); std::fputc('\n', f);
            }
        }
        std::fclose(f);
    }

    // Clicks: 160,000 rows across 4 parts (leaves ~45% of users organic cn==0, ~55% with 1..8 clicks!)
    for (int p = 0; p < 4; ++p) {
        FILE* f = std::fopen((dir + "/clicks/part-0000" + std::to_string(p) + ".csv").c_str(), "wb");
        std::fputs("click_id,user_id,campaign_id,click_ts\n", f);
        char cid[40], uid[40], cmp[16], ts[40];
        for (uint32_t i = 0; i < 40000; ++i) {
            uint64_t r = rng();
            if ((r & 7ULL) == 0) mk_hex_id('k', r, rng(), cid);
            else mk_short_id('k', uint64_t(p) * 50000ULL + i + 1000ULL, cid);
            // Click users: biased to [0..50000) so [50000..80000) train organic cn==0!
            uint32_t u = (r % 25 == 0) ? uint32_t(rng() % 480) : uint32_t(rng() % 50000);
            mk_uid(u, uid);
            std::snprintf(cmp, sizeof(cmp), "c%03u", uint32_t((rng() % 40) + 1));
            int64_t sec = W0 - 7LL * 86400LL + int64_t(rng() % (W_DUR + 7LL * 86400LL));
            uint32_t fmt = uint32_t(rng() % 100);
            if (fmt < 57) fmt_f1(sec, ts);
            else if (fmt < 90) std::snprintf(ts, sizeof(ts), "%lld", (long long)(sec * 1000LL + int64_t(rng() % 1000)));
            else fmt_f2(sec, int(rng() % 13) - 6, ts);
            qwrap(f, cid, (rng() % 5) == 0); std::fputc(',', f);
            qwrap(f, uid, (rng() % 5) == 0); std::fputc(',', f);
            qwrap(f, cmp, (rng() % 5) == 0); std::fputc(',', f);
            qwrap(f, ts,  (rng() % 5) == 0); std::fputc('\n', f);
        }
        std::fclose(f);
    }

    // Events: 640,000 rows across 8 parts:
    // - Users [0..400): heavy (~300 evs/person -> trains sort_msd256 + AVX2 gap scan)
    // - Users [400..48000): multi-session (5..25 evs/person -> trains AVX2 bitonic sort <=16 + MSD64)
    // - Users [48000..80000): 1-session clustered within 15 min (trains 1-session fast path!)
    for (int p = 0; p < 8; ++p) {
        FILE* f = std::fopen((dir + "/events/part-0000" + std::to_string(p) + ".csv").c_str(), "wb");
        std::fputs("event_id,user_id,event_ts,ingest_ts,event_type,product_id,price_cents\n", f);
        char eid[40], uid[40], ets[40], its[40], prod[16], prc[16];
        for (uint32_t i = 0; i < 80000; ++i) {
            uint64_t r = rng();
            uint64_t idn = ((r % 14) == 0 && i > 16)
                ? (uint64_t(p) * 100000ULL + (i - (r % 16) - 1))
                : (uint64_t(p) * 100000ULL + i + 10000ULL);
            if ((idn & 7ULL) == 0) mk_hex_id('e', idn * 0xbf58476d1ce4e5b9ULL, idn ^ 0x94d049bb133111ebULL, eid);
            else mk_short_id('e', idn, eid);

            uint32_t bucket = uint32_t(r % 100);
            uint32_t u;
            int64_t sec;
            if (bucket < 3) {
                u = uint32_t(rng() % 64); // heavy users (~300 events each)
                sec = W0 + int64_t(rng() % W_DUR);
            } else if (bucket < 80) {
                u = 400U + uint32_t(rng() % 47600U); // multi-session ~10 events/user
                sec = W0 + int64_t(rng() % W_DUR);
            } else {
                u = 48000U + uint32_t(rng() % 32000U); // 1-session users (~4 events within 15 min)
                int64_t base_s = W0 + int64_t((uint64_t(u) * 2654435761ULL) % uint64_t(W_DUR - 1800LL));
                sec = base_s + int64_t(rng() % 900);
            }
            mk_uid(u, uid);
            int64_t ing = sec + int64_t(rng() % 3600);
            if ((r % 200) == 0) ing = sec + 90000LL; // 0.5% late invalid
            uint32_t fmt = uint32_t(rng() % 100);
            if (fmt < 57) fmt_f1(sec, ets);
            else if (fmt < 90) std::snprintf(ets, sizeof(ets), "%lld", (long long)(sec * 1000LL + int64_t(rng() % 1000)));
            else fmt_f2(sec, int(rng() % 13) - 6, ets);
            fmt_f1(ing, its);
            uint32_t tp = uint32_t(rng() % 100);
            const char* et = (tp < 77) ? "view" : ((tp < 97) ? "cart" : "purchase");
            std::snprintf(prod, sizeof(prod), "p%u", uint32_t((rng() % 5000) + 1));
            if (tp >= 97 || (rng() % 3) == 0) std::snprintf(prc, sizeof(prc), "%u", uint32_t((rng() % 20000) + 100));
            else prc[0] = '\0';
            qwrap(f, eid,  (rng() % 5) == 0); std::fputc(',', f);
            qwrap(f, uid,  (rng() % 5) == 0); std::fputc(',', f);
            qwrap(f, ets,  (rng() % 5) == 0); std::fputc(',', f);
            qwrap(f, its,  (rng() % 5) == 0); std::fputc(',', f);
            qwrap(f, et,   (rng() % 5) == 0); std::fputc(',', f);
            qwrap(f, prod, (rng() % 5) == 0); std::fputc(',', f);
            qwrap(f, prc,  (rng() % 5) == 0); std::fputc('\n', f);
        }
        std::fclose(f);
    }
    return 0;
}
