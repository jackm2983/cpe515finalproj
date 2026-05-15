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
static constexpr int REPEATS = 3;

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

    return get_cycles() - start;
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

    uint32_t cycles = get_cycles() - start;

    if (checksum_out) {
        *checksum_out = sink;
    }

    return cycles;
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

    uint32_t cycles = get_cycles() - start;

    if (checksum_out) {
        *checksum_out = sink;
    }

    return cycles;
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

    uint32_t cycles = get_cycles() - start;

    if (checksum_out) {
        *checksum_out = sink;
    }

    return cycles;
}

struct Result {
    uint32_t net;
    uint32_t per_x100;
    int32_t checksum;
};

static Result run_best(int taps,
                       int N,
                       uint32_t (*fn)(int, int, int, int32_t*)) {
    Result best = {0xffffffffu, 0, 0};

    for (int r = 0; r < REPEATS; r++) {
        int32_t cs = 0;

        (void)fn(taps, N, 3, &cs);

        uint32_t cycles = fn(taps, N, SWEEP_ITERS, &cs);
        uint32_t overhead = measure_loop_overhead(taps, N, SWEEP_ITERS);
        uint32_t net = (cycles > overhead) ? (cycles - overhead) : 0;

        int outputs = (N - taps + 1) * SWEEP_ITERS;
        uint32_t per_x100 = outputs
            ? (uint32_t)(((uint64_t)net * 100) / (uint64_t)outputs)
            : 0;

        if (net < best.net) {
            best.net = net;
            best.per_x100 = per_x100;
            best.checksum = cs;
        }
    }

    return best;
}

static void sweep_case(int taps, int N) {
    init_buffers(taps, N);

    Result scalar = run_best(taps, N, fir_scalar);
    Result mac16 = run_best(taps, N, fir_mac16);
    Result unroll2 = run_best(taps, N, fir_mac16_unrolled);

    uint32_t mac_speed_x100 = mac16.per_x100
        ? (uint32_t)(((uint64_t)scalar.per_x100 * 100) / mac16.per_x100)
        : 0;

    uint32_t unroll_speed_x100 = unroll2.per_x100
        ? (uint32_t)(((uint64_t)scalar.per_x100 * 100) / unroll2.per_x100)
        : 0;


    printf("CSV,scalar,%d,%d,%d,%lu,0,%lu,%lu,%ld\r\n",
       taps,
       N,
       SWEEP_ITERS,
       (unsigned long)scalar.net,
       (unsigned long)scalar.net,
       (unsigned long)scalar.per_x100,
       (long)scalar.checksum);

    printf("CSV,mac16,%d,%d,%d,%lu,0,%lu,%lu,%ld\r\n",
        taps,
        N,
        SWEEP_ITERS,
        (unsigned long)mac16.net,
        (unsigned long)mac16.net,
        (unsigned long)mac16.per_x100,
        (long)mac16.checksum);

    printf("CSV,unroll2,%d,%d,%d,%lu,0,%lu,%lu,%ld\r\n",
        taps,
        N,
        SWEEP_ITERS,
        (unsigned long)unroll2.net,
        (unsigned long)unroll2.net,
        (unsigned long)unroll2.per_x100,
        (long)unroll2.checksum);
}

void do_sweep_full() {
    puts("\r\n=== useful sweep: scalar vs mac16 vs unroll2 ===\r\n");
    puts("CSV,variant,taps,N,cycles_per_out_x100,speedup_x100,checksum\r\n");

    const int taps_list[] = {4, 8, 16, 32};
    const int Ns[] = {64, 128, 256};

    for (unsigned ti = 0; ti < sizeof(taps_list) / sizeof(taps_list[0]); ti++) {
        for (unsigned ni = 0; ni < sizeof(Ns) / sizeof(Ns[0]); ni++) {
            int taps = taps_list[ti];
            int N = Ns[ni];

            if (taps > MAX_TAPS || N > MAX_SAMPLES || N < taps * 2) {
                continue;
            }

            sweep_case(taps, N);
        }
    }

    puts("\r\n=== end sweep ===\r\n");
}

void do_verify() {
    puts("\r\n=== correctness check ===\r\n");
    puts("CSV,variant,taps,N,checksum,pass\r\n");

    const int taps_list[] = {4, 8, 16, 32};
    const int Ns[] = {64, 128, 256};

    for (unsigned ti = 0; ti < sizeof(taps_list) / sizeof(taps_list[0]); ti++) {
        for (unsigned ni = 0; ni < sizeof(Ns) / sizeof(Ns[0]); ni++) {
            int taps = taps_list[ti];
            int N = Ns[ni];

            if (taps > MAX_TAPS || N > MAX_SAMPLES || N < taps * 2) {
                continue;
            }

            init_buffers(taps, N);

            int32_t ref = 0;
            int32_t got = 0;

            (void)fir_scalar(taps, N, 1, &ref);

            (void)fir_mac16(taps, N, 1, &got);
            printf("CSV,mac16,%d,%d,%ld,%s\r\n",
                   taps,
                   N,
                   (long)got,
                   (got == ref) ? "PASS" : "FAIL");

            (void)fir_mac16_unrolled(taps, N, 1, &got);
            printf("CSV,unroll2,%d,%d,%ld,%s\r\n",
                   taps,
                   N,
                   (long)got,
                   (got == ref) ? "PASS" : "FAIL");
        }
    }
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
        MENU_ITEM('f', "useful sweep", do_sweep_full),
        MENU_END,
    },
};

}

extern "C" void do_proj_menu() {
    menu_run(&MENU);
}