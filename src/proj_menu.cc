#include "proj_menu.h"

#include <stdio.h>
#include <stdint.h>

#include "cfu.h"
#include "menu.h"

namespace {

#define MAC_SIGNED(a, b) cfu_op(1, 0, (a), (b))

static constexpr int MAX_TAPS = 32;
static constexpr int MAX_SAMPLES = 256;
static constexpr int SWEEP_ITERS = 500;

static inline uint32_t get_cycles() {
    uint32_t c;
    asm volatile ("rdcycle %0" : "=r"(c));
    return c;
}

static inline uint32_t pack16(int16_t lo, int16_t hi) {
    return ((uint32_t)(uint16_t)hi << 16) | (uint16_t)lo;
}

static int16_t g_coeffs[MAX_TAPS];
static int16_t g_samples[MAX_SAMPLES];
static uint32_t g_coeff_pack[MAX_TAPS / 2];

static void init_buffers(int taps, int num_samples) {
    for (int i = 0; i < taps; i++) {
        g_coeffs[i] = (int16_t)(i + 1);
    }

    for (int i = 0; i < num_samples; i++) {
        g_samples[i] = (int16_t)(i & 0x7FFF);
    }

    for (int p = 0; p < taps / 2; p++) {
        g_coeff_pack[p] = pack16(g_coeffs[2 * p], g_coeffs[2 * p + 1]);
    }
}

static uint32_t measure_loop_overhead(int taps, int num_samples, int iters) {
    volatile int32_t sink = 0;

    uint32_t start = get_cycles();

    for (int it = 0; it < iters; it++) {
        for (int n = taps - 1; n < num_samples; n++) {
            sink += n;
        }
    }

    uint32_t end = get_cycles();
    return end - start;
}

static uint32_t fir_scalar(int taps, int num_samples, int iters,
                           int32_t* checksum_out) {
    volatile int32_t sink = 0;

    uint32_t start = get_cycles();

    for (int it = 0; it < iters; it++) {
        for (int n = taps - 1; n < num_samples; n++) {
            int32_t acc = 0;

            for (int i = 0; i < taps; i++) {
                acc += (int32_t)g_samples[n - i] * (int32_t)g_coeffs[i];
            }

            sink += acc;
        }
    }

    uint32_t end = get_cycles();

    if (checksum_out) {
        *checksum_out = sink;
    }

    return end - start;
}

static uint32_t fir_mac16(int taps, int num_samples, int iters,
                          int32_t* checksum_out) {
    volatile int32_t sink = 0;
    int pairs = taps / 2;

    uint32_t start = get_cycles();

    for (int it = 0; it < iters; it++) {
        for (int n = taps - 1; n < num_samples; n++) {
            int32_t acc = 0;

            for (int p = 0; p < pairs; p++) {
                int idx = n - 2 * p;
                uint32_t xp = pack16(g_samples[idx], g_samples[idx - 1]);
                acc += (int32_t)MAC_SIGNED(xp, g_coeff_pack[p]);
            }

            sink += acc;
        }
    }

    uint32_t end = get_cycles();

    if (checksum_out) {
        *checksum_out = sink;
    }

    return end - start;
}

static uint32_t fir_mac16_unrolled(int taps, int num_samples, int iters,
                                   int32_t* checksum_out) {
    volatile int32_t sink = 0;
    int pairs = taps / 2;

    uint32_t start = get_cycles();

    for (int it = 0; it < iters; it++) {
        for (int n = taps - 1; n < num_samples; n++) {
            int32_t acc0 = 0;
            int32_t acc1 = 0;
            int p = 0;

            for (; p + 1 < pairs; p += 2) {
                int i0 = n - 2 * p;
                int i1 = n - 2 * (p + 1);

                uint32_t xp0 = pack16(g_samples[i0], g_samples[i0 - 1]);
                uint32_t xp1 = pack16(g_samples[i1], g_samples[i1 - 1]);

                acc0 += (int32_t)MAC_SIGNED(xp0, g_coeff_pack[p]);
                acc1 += (int32_t)MAC_SIGNED(xp1, g_coeff_pack[p + 1]);
            }

            for (; p < pairs; p++) {
                int idx = n - 2 * p;
                uint32_t xp = pack16(g_samples[idx], g_samples[idx - 1]);
                acc0 += (int32_t)MAC_SIGNED(xp, g_coeff_pack[p]);
            }

            sink += acc0 + acc1;
        }
    }

    uint32_t end = get_cycles();

    if (checksum_out) {
        *checksum_out = sink;
    }

    return end - start;
}

static void sweep_row(const char* variant,
                      int taps,
                      int N,
                      uint32_t (*fn)(int, int, int, int32_t*),
                      int iters) {
    int32_t cs = 0;

    (void)fn(taps, N, 2, &cs);

    uint32_t cycles = fn(taps, N, iters, &cs);
    uint32_t overhead = measure_loop_overhead(taps, N, iters);
    uint32_t net = (cycles > overhead) ? (cycles - overhead) : 0;

    int outputs = (N - taps + 1) * iters;

    unsigned long per_out_x100 = outputs
        ? (unsigned long)((uint64_t)net * 100 / (uint64_t)outputs)
        : 0;

    printf("CSV,%s,%d,%d,%d,%lu,%lu,%lu,%lu,%ld\r\n",
           variant,
           taps,
           N,
           iters,
           (unsigned long)cycles,
           (unsigned long)overhead,
           (unsigned long)net,
           per_out_x100,
           (long)cs);
}

void do_sweep_full() {
    puts("\r\n=== sweep: scalar vs mac16 vs unroll2 ===\r\n");
    puts("CSV,variant,taps,N,iters,cycles,overhead,net,per_out_x100,checksum\r\n");

    const int taps_list[] = {4, 8, 16, 32};
    const int Ns[] = {32, 64, 128};

    struct Variant {
        const char* name;
        uint32_t (*fn)(int, int, int, int32_t*);
    };

    Variant variants[] = {
        {"scalar", fir_scalar},
        {"mac16", fir_mac16},
        {"unroll2", fir_mac16_unrolled},
    };

    for (unsigned vi = 0; vi < sizeof(variants) / sizeof(variants[0]); vi++) {
        for (unsigned ti = 0; ti < sizeof(taps_list) / sizeof(taps_list[0]); ti++) {
            int taps = taps_list[ti];

            if (taps > MAX_TAPS) {
                continue;
            }

            for (unsigned ni = 0; ni < sizeof(Ns) / sizeof(Ns[0]); ni++) {
                int N = Ns[ni];

                if (N > MAX_SAMPLES) {
                    continue;
                }

                init_buffers(taps, N);
                sweep_row(variants[vi].name,
                          taps,
                          N,
                          variants[vi].fn,
                          SWEEP_ITERS);
            }
        }
    }

    puts("\r\n=== end sweep ===\r\n");
}

void do_verify() {
    puts("\r\n=== correctness check: scalar vs mac16 vs unroll2 ===\r\n");

    int taps = 4;
    int N = 64;

    init_buffers(taps, N);

    int32_t ref = 0;
    int32_t got = 0;

    (void)fir_scalar(taps, N, 1, &ref);

    (void)fir_mac16(taps, N, 1, &got);
    printf("mac16   : %s ref=%ld got=%ld\r\n",
           (got == ref) ? "PASS" : "FAIL",
           (long)ref,
           (long)got);

    (void)fir_mac16_unrolled(taps, N, 1, &got);
    printf("unroll2 : %s ref=%ld got=%ld\r\n",
           (got == ref) ? "PASS" : "FAIL",
           (long)ref,
           (long)got);
}

void do_mac16_spot_check() {
    puts("\r\n=== mac16 spot check ===\r\n");

    uint32_t rs1 = pack16(3, 2);
    uint32_t rs2 = pack16(5, -4);

    int32_t got = (int32_t)MAC_SIGNED(rs1, rs2);
    int32_t exp = 3 * 5 + 2 * -4;

    printf("got=%ld expected=%ld %s\r\n",
           (long)got,
           (long)exp,
           (got == exp) ? "PASS" : "FAIL");
}

struct Menu MENU = {
    "Project Menu",
    "project",
    {
        MENU_ITEM('v', "verify scalar, mac16, unroll2", do_verify),
        MENU_ITEM('m', "mac16 check", do_mac16_spot_check),
        MENU_ITEM('f', "sweep scalar vs mac16 vs unroll2", do_sweep_full),
        MENU_END,
    },
};

}

extern "C" void do_proj_menu() {
    menu_run(&MENU);
}