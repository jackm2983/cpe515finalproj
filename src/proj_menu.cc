#include "proj_menu.h"

#include <stdio.h>
#include <stdint.h>

#include "cfu.h"
#include "menu.h"

namespace {

// generic CFU call: cfu_op(funct3, funct7, rs1, rs2)
// funct3=0 -> add. funct3=1 -> mac/fir family, funct7 selects sub-op.
#define MAC_SIGNED(a,b)    cfu_op(1, 0, (a), (b))
#define MAC_UNSIGNED(a,b)  cfu_op(1, 1, (a), (b))
#define MAC_ACC(a,b)       cfu_op(1, 2, (a), (b))
#define ACC_READ()         cfu_op(1, 3, 0, 0)
#define LOAD_TAPS(lo,hi)   cfu_op(1, 4, (lo), (hi))
#define MAC_LOADED(a,b)    cfu_op(1, 5, (a), (b))
#define SWIN_PUSH(s)       cfu_op(1, 6, (s), 0)
#define SWIN_RESET()       cfu_op(1, 7, 0, 0)

// ---------------------------------------------------------------
// configuration
// ---------------------------------------------------------------
static constexpr int MAX_TAPS      = 32;
static constexpr int MAX_SAMPLES   = 256;
static constexpr int DEFAULT_ITERS = 1000;
static constexpr int SWEEP_ITERS   = 500;

// ---------------------------------------------------------------
// helpers
// ---------------------------------------------------------------

static inline uint32_t get_cycles() {
    uint32_t c;
    asm volatile ("rdcycle %0" : "=r"(c));
    return c;
}

static inline uint32_t pack16(int16_t lo, int16_t hi) {
    return ((uint32_t)(uint16_t)hi << 16) | (uint16_t)lo;
}

// ---------------------------------------------------------------
// buffers
// ---------------------------------------------------------------

static int16_t  g_coeffs[MAX_TAPS];
static int16_t  g_samples[MAX_SAMPLES];
static uint32_t g_coeff_pack[MAX_TAPS / 2];

static void init_buffers(int taps, int num_samples) {
    for (int i = 0; i < taps; i++) {
        g_coeffs[i] = (int16_t)(i + 1);
    }
    for (int i = 0; i < num_samples; i++) {
        g_samples[i] = (int16_t)(i & 0x7FFF);
    }
    for (int p = 0; p < taps / 2; p++) {
        g_coeff_pack[p] = pack16(g_coeffs[2*p], g_coeffs[2*p + 1]);
    }
}

// ---------------------------------------------------------------
// overhead loop. mirrors the iter+output loop structure with no work.
// ---------------------------------------------------------------

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

// ---------------------------------------------------------------
// variant 1: scalar rv32im baseline
// ---------------------------------------------------------------
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
    if (checksum_out) *checksum_out = sink;
    return end - start;
}

// ---------------------------------------------------------------
// variant 2: mac16 packed simd (basic)
// ---------------------------------------------------------------
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
    if (checksum_out) *checksum_out = sink;
    return end - start;
}

// ---------------------------------------------------------------
// variant 3: mac16 + 2x loop unrolling
// ---------------------------------------------------------------
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
            sink += (acc0 + acc1);
        }
    }
    uint32_t end = get_cycles();
    if (checksum_out) *checksum_out = sink;
    return end - start;
}

// ---------------------------------------------------------------
// variant 4: mac16 + accumulator register
// ---------------------------------------------------------------
static uint32_t fir_mac16_acc(int taps, int num_samples, int iters,
                              int32_t* checksum_out) {
    volatile int32_t sink = 0;
    int pairs = taps / 2;
    uint32_t start = get_cycles();
    for (int it = 0; it < iters; it++) {
        for (int n = taps - 1; n < num_samples; n++) {
            (void)ACC_READ();  // clear acc
            for (int p = 0; p < pairs; p++) {
                int idx = n - 2 * p;
                uint32_t xp = pack16(g_samples[idx], g_samples[idx - 1]);
                (void)MAC_ACC(xp, g_coeff_pack[p]);
            }
            int32_t acc = (int32_t)ACC_READ();
            sink += acc;
        }
    }
    uint32_t end = get_cycles();
    if (checksum_out) *checksum_out = sink;
    return end - start;
}

// ---------------------------------------------------------------
// variant 5: sliding window hardware (4-tap only)
// ---------------------------------------------------------------
static uint32_t fir_mac16_swin(int taps, int num_samples, int iters,
                               int32_t* checksum_out) {
    volatile int32_t sink = 0;
    if (taps != 4) {
        if (checksum_out) *checksum_out = 0;
        return 0;
    }
    uint32_t start = get_cycles();
    for (int it = 0; it < iters; it++) {
        (void)SWIN_RESET();
        (void)LOAD_TAPS(g_coeff_pack[0], g_coeff_pack[1]);
        for (int i = 0; i < taps - 1; i++) {
            (void)SWIN_PUSH((uint32_t)(uint16_t)g_samples[i]);
        }
        for (int n = taps - 1; n < num_samples; n++) {
            int32_t acc = (int32_t)SWIN_PUSH(
                (uint32_t)(uint16_t)g_samples[n]);
            sink += acc;
        }
    }
    uint32_t end = get_cycles();
    if (checksum_out) *checksum_out = sink;
    return end - start;
}

// ---------------------------------------------------------------
// variant 6: circular addressing (software emulation, power-of-2 N)
// ---------------------------------------------------------------
static uint32_t fir_mac16_circ(int taps, int num_samples, int iters,
                               int32_t* checksum_out) {
    volatile int32_t sink = 0;
    int pairs = taps / 2;
    int mask = num_samples - 1;
    if ((num_samples & mask) != 0) {
        if (checksum_out) *checksum_out = 0;
        return 0;
    }
    uint32_t start = get_cycles();
    for (int it = 0; it < iters; it++) {
        for (int n = taps - 1; n < num_samples; n++) {
            int32_t acc = 0;
            for (int p = 0; p < pairs; p++) {
                int i0 = (n - 2 * p) & mask;
                int i1 = (n - 2 * p - 1) & mask;
                uint32_t xp = pack16(g_samples[i0], g_samples[i1]);
                acc += (int32_t)MAC_SIGNED(xp, g_coeff_pack[p]);
            }
            sink += acc;
        }
    }
    uint32_t end = get_cycles();
    if (checksum_out) *checksum_out = sink;
    return end - start;
}

// ---------------------------------------------------------------
// variant 7: mac_loaded (4-tap fir in one instruction, taps preloaded)
// ---------------------------------------------------------------
static uint32_t fir_mac_loaded(int taps, int num_samples, int iters,
                               int32_t* checksum_out) {
    volatile int32_t sink = 0;
    if (taps != 4) {
        if (checksum_out) *checksum_out = 0;
        return 0;
    }
    uint32_t start = get_cycles();
    for (int it = 0; it < iters; it++) {
        (void)LOAD_TAPS(g_coeff_pack[0], g_coeff_pack[1]);
        for (int n = taps - 1; n < num_samples; n++) {
            uint32_t xp0 = pack16(g_samples[n],     g_samples[n - 1]);
            uint32_t xp1 = pack16(g_samples[n - 2], g_samples[n - 3]);
            int32_t acc = (int32_t)MAC_LOADED(xp0, xp1);
            sink += acc;
        }
    }
    uint32_t end = get_cycles();
    if (checksum_out) *checksum_out = sink;
    return end - start;
}

// ---------------------------------------------------------------
// runners
// ---------------------------------------------------------------

static void run_single(const char* name,
                       uint32_t (*fn)(int, int, int, int32_t*),
                       int taps, int num_samples, int iters) {
    int32_t cs = 0;
    (void)fn(taps, num_samples, 1, &cs);  // warmup
    uint32_t cycles = fn(taps, num_samples, iters, &cs);
    uint32_t overhead = measure_loop_overhead(taps, num_samples, iters);
    uint32_t net = (cycles > overhead) ? (cycles - overhead) : 0;
    int outputs = (num_samples - taps + 1) * iters;
    printf("%-24s  taps=%2d N=%3d iters=%5d cycles=%9lu "
           "overhead=%7lu net=%9lu per_out=%6lu  cs=%ld\r\n",
           name, taps, num_samples, iters,
           (unsigned long)cycles,
           (unsigned long)overhead,
           (unsigned long)net,
           outputs ? (unsigned long)(net / outputs) : 0UL,
           (long)cs);
}

void do_fir_scalar() {
    init_buffers(4, 64);
    run_single("scalar baseline", fir_scalar, 4, 64, DEFAULT_ITERS);
}

void do_fir_mac16() {
    init_buffers(4, 64);
    run_single("mac16 basic", fir_mac16, 4, 64, DEFAULT_ITERS);
}

void do_fir_mac16_unrolled() {
    init_buffers(4, 64);
    run_single("mac16 unrolled 2x", fir_mac16_unrolled, 4, 64, DEFAULT_ITERS);
}

void do_fir_mac16_acc() {
    init_buffers(4, 64);
    run_single("mac16 + acc reg", fir_mac16_acc, 4, 64, DEFAULT_ITERS);
}

void do_fir_swin() {
    init_buffers(4, 64);
    run_single("sliding window hw", fir_mac16_swin, 4, 64, DEFAULT_ITERS);
}

void do_fir_circ() {
    init_buffers(4, 64);
    run_single("circular addressing", fir_mac16_circ, 4, 64, DEFAULT_ITERS);
}

void do_fir_loaded() {
    init_buffers(4, 64);
    run_single("mac_loaded (4tap/instr)", fir_mac_loaded, 4, 64, DEFAULT_ITERS);
}

// ---------------------------------------------------------------
// sweeps - emit CSV-prefixed rows
// ---------------------------------------------------------------

// run a single variant across many taps/N combos with the variant
// staying hot in the icache. each variant gets its own warmup pass
// before measurement. emits one CSV row per (variant, taps, N) cell.
//
// per_out_x100 column is per_output * 100, so 7.49 cycles/output
// shows as 749. avoids losing precision to integer floor.
static void sweep_row_isolated(const char* variant,
                               int taps, int N,
                               uint32_t (*fn)(int, int, int, int32_t*),
                               int iters) {
    int32_t cs = 0;
    // multiple warmup passes to settle icache and any branch predictor
    (void)fn(taps, N, 2, &cs);
    uint32_t cy  = fn(taps, N, iters, &cs);
    uint32_t ov  = measure_loop_overhead(taps, N, iters);
    uint32_t net = (cy > ov) ? (cy - ov) : 0;
    int outs = (N - taps + 1) * iters;
    // scaled per_output: cycles/output * 100, rounded
    unsigned long per_x100 = outs
        ? (unsigned long)((uint64_t)net * 100 / (uint64_t)outs)
        : 0;
    printf("CSV,%s,%d,%d,%d,%lu,%lu,%lu,%lu,%ld\r\n",
           variant, taps, N, iters,
           (unsigned long)cy, (unsigned long)ov,
           (unsigned long)net, per_x100, (long)cs);
}

void do_sweep_taps() {
    puts("\r\n=== sweep: tap count at N=64 (isolated) ===\r\n");
    puts("CSV,variant,taps,N,iters,cycles,overhead,net,per_out_x100,checksum\r\n");
    const int taps_list[] = {4, 8, 16, 32};
    const int n_taps = (int)(sizeof(taps_list)/sizeof(int));
    int N = 64;
    int iters = SWEEP_ITERS;

    // outer loop: variant. inner loop: taps. each variant runs
    // contiguously so the icache stays warm across tap counts.
    struct V {
        const char* name;
        uint32_t (*fn)(int, int, int, int32_t*);
        bool t4_only;
    } variants[] = {
        {"scalar",  fir_scalar,         false},
        {"mac16",   fir_mac16,          false},
        {"unroll2", fir_mac16_unrolled, false},
        {"acc_reg", fir_mac16_acc,      false},
        {"circ",    fir_mac16_circ,     false},
        {"swin",    fir_mac16_swin,     true},
        {"loaded",  fir_mac_loaded,     true},
    };
    const int n_var = (int)(sizeof(variants)/sizeof(variants[0]));

    for (int vi = 0; vi < n_var; vi++) {
        for (int ti = 0; ti < n_taps; ti++) {
            int taps = taps_list[ti];
            if (taps > MAX_TAPS) continue;
            if (variants[vi].t4_only && taps != 4) continue;
            init_buffers(taps, N);
            sweep_row_isolated(variants[vi].name, taps, N,
                               variants[vi].fn, iters);
        }
    }
    puts("\r\n=== end sweep ===\r\n");
}

void do_sweep_n() {
    puts("\r\n=== sweep: N at taps=4 (isolated) ===\r\n");
    puts("CSV,variant,taps,N,iters,cycles,overhead,net,per_out_x100,checksum\r\n");
    const int Ns[] = {16, 32, 64, 128, 256};
    const int n_N = (int)(sizeof(Ns)/sizeof(int));
    int taps = 4;
    int iters = SWEEP_ITERS;

    struct V {
        const char* name;
        uint32_t (*fn)(int, int, int, int32_t*);
    } variants[] = {
        {"scalar",  fir_scalar},
        {"mac16",   fir_mac16},
        {"unroll2", fir_mac16_unrolled},
        {"acc_reg", fir_mac16_acc},
        {"circ",    fir_mac16_circ},
        {"swin",    fir_mac16_swin},
        {"loaded",  fir_mac_loaded},
    };
    const int n_var = (int)(sizeof(variants)/sizeof(variants[0]));

    for (int vi = 0; vi < n_var; vi++) {
        for (int ni = 0; ni < n_N; ni++) {
            int N = Ns[ni];
            if (N > MAX_SAMPLES) continue;
            init_buffers(taps, N);
            sweep_row_isolated(variants[vi].name, taps, N,
                               variants[vi].fn, iters);
        }
    }
    puts("\r\n=== end sweep ===\r\n");
}

// big combined sweep: every variant × every (taps, N) combo that makes
// sense. takes a while in sim. emits one csv block, useful for paper.
void do_sweep_full() {
    puts("\r\n=== full sweep: variants x taps x N (isolated) ===\r\n");
    puts("CSV,variant,taps,N,iters,cycles,overhead,net,per_out_x100,checksum\r\n");
    const int taps_list[] = {4, 8, 16, 32};
    const int Ns[]        = {32, 64, 128};
    const int n_taps = (int)(sizeof(taps_list)/sizeof(int));
    const int n_N    = (int)(sizeof(Ns)/sizeof(int));
    int iters = SWEEP_ITERS;

    struct V {
        const char* name;
        uint32_t (*fn)(int, int, int, int32_t*);
        bool t4_only;
    } variants[] = {
        {"scalar",  fir_scalar,         false},
        {"mac16",   fir_mac16,          false},
        {"unroll2", fir_mac16_unrolled, false},
        {"acc_reg", fir_mac16_acc,      false},
        {"circ",    fir_mac16_circ,     false},
        {"swin",    fir_mac16_swin,     true},
        {"loaded",  fir_mac_loaded,     true},
    };
    const int n_var = (int)(sizeof(variants)/sizeof(variants[0]));

    // variant-outermost so each variant's code stays hot across all
    // (taps, N) measurements before we move to the next
    for (int vi = 0; vi < n_var; vi++) {
        for (int ti = 0; ti < n_taps; ti++) {
            int taps = taps_list[ti];
            if (taps > MAX_TAPS) continue;
            if (variants[vi].t4_only && taps != 4) continue;
            for (int ni = 0; ni < n_N; ni++) {
                int N = Ns[ni];
                if (N > MAX_SAMPLES) continue;
                init_buffers(taps, N);
                sweep_row_isolated(variants[vi].name, taps, N,
                                   variants[vi].fn, iters);
            }
        }
    }
    puts("\r\n=== end full sweep ===\r\n");
}

// ---------------------------------------------------------------
// correctness check
// ---------------------------------------------------------------

void do_verify() {
    puts("\r\n=== correctness check (taps=4 N=64 iters=1) ===\r\n");
    int taps = 4, N = 64;
    init_buffers(taps, N);

    int32_t ref = 0;
    (void)fir_scalar(taps, N, 1, &ref);

    int32_t cs;
    (void)fir_mac16(taps, N, 1, &cs);
    printf("  mac16    : %s (ref=%ld got=%ld)\r\n",
           (cs == ref) ? "PASS" : "FAIL", (long)ref, (long)cs);
    (void)fir_mac16_unrolled(taps, N, 1, &cs);
    printf("  unroll2  : %s (ref=%ld got=%ld)\r\n",
           (cs == ref) ? "PASS" : "FAIL", (long)ref, (long)cs);
    (void)fir_mac16_acc(taps, N, 1, &cs);
    printf("  acc_reg  : %s (ref=%ld got=%ld)\r\n",
           (cs == ref) ? "PASS" : "FAIL", (long)ref, (long)cs);
    (void)fir_mac16_circ(taps, N, 1, &cs);
    printf("  circ     : %s (ref=%ld got=%ld)\r\n",
           (cs == ref) ? "PASS" : "FAIL", (long)ref, (long)cs);
    (void)fir_mac16_swin(taps, N, 1, &cs);
    printf("  swin     : %s (ref=%ld got=%ld)\r\n",
           (cs == ref) ? "PASS" : "FAIL", (long)ref, (long)cs);
    (void)fir_mac_loaded(taps, N, 1, &cs);
    printf("  loaded   : %s (ref=%ld got=%ld)\r\n",
           (cs == ref) ? "PASS" : "FAIL", (long)ref, (long)cs);
}

// ---------------------------------------------------------------
// legacy spot checks
// ---------------------------------------------------------------

void do_exercise_cfu_op0() {
    puts("\r\nExercise CFU Op0 (ADD)\r\n");
    unsigned pass = 0, fail = 0, count = 0;
    for (unsigned a = 0x00004567; a < 0xF8000000; a += 0x00212345) {
        for (unsigned b = 0x0000ba98; b < 0xFF000000; b += 0x00770077) {
            unsigned r = cfu_op(0, 0, a, b);
            if (r != a + b) {
                printf("[%4u] a=%08x b=%08x exp=%08x got=%08x FAIL\r\n",
                       count, a, b, a + b, r);
                fail++;
            } else {
                pass++;
            }
            count++;
        }
    }
    printf("\r\n%u total, %u pass, %u fail\r\n", count, pass, fail);
}

void do_exercise_cfu_op1() {
    puts("\r\nExercise CFU mac16 signed\r\n");
    uint32_t rs1 = pack16(3, 2);
    uint32_t rs2 = pack16(5, -4);
    int32_t r = (int32_t)MAC_SIGNED(rs1, rs2);
    printf("test 1: got %ld expect 7   %s\r\n",
           (long)r, (r == 7) ? "PASS" : "FAIL");

    rs1 = pack16(-100, -200);
    rs2 = pack16(-3, -4);
    r = (int32_t)MAC_SIGNED(rs1, rs2);
    int32_t exp = (-100)*(-3) + (-200)*(-4);
    printf("test 2: got %ld expect %ld %s\r\n",
           (long)r, (long)exp, (r == exp) ? "PASS" : "FAIL");

    rs1 = pack16(1000, -1000);
    rs2 = pack16(32, 32);
    r = (int32_t)MAC_SIGNED(rs1, rs2);
    exp = 1000*32 + (-1000)*32;
    printf("test 3: got %ld expect %ld %s\r\n",
           (long)r, (long)exp, (r == exp) ? "PASS" : "FAIL");
}

void do_hello_world() {
    puts("Hello, World!!!\n");
}

struct Menu MENU = {
    "Project Menu",
    "project",
    {
        MENU_ITEM('0', "cfu op0 exercise (add)",        do_exercise_cfu_op0),
        MENU_ITEM('1', "cfu mac16 exercise",            do_exercise_cfu_op1),
        MENU_ITEM('v', "verify all variants",           do_verify),
        MENU_ITEM('s', "fir: scalar baseline",          do_fir_scalar),
        MENU_ITEM('m', "fir: mac16 basic",              do_fir_mac16),
        MENU_ITEM('u', "fir: mac16 unrolled 2x",        do_fir_mac16_unrolled),
        MENU_ITEM('a', "fir: mac16 + acc reg",          do_fir_mac16_acc),
        MENU_ITEM('w', "fir: sliding window",           do_fir_swin),
        MENU_ITEM('c', "fir: circular addressing",      do_fir_circ),
        MENU_ITEM('l', "fir: mac_loaded (4tap/instr)",  do_fir_loaded),
        MENU_ITEM('t', "sweep: vary taps at N=64",      do_sweep_taps),
        MENU_ITEM('n', "sweep: vary N at taps=4",       do_sweep_n),
        MENU_ITEM('f', "sweep: full (variants*taps*N)", do_sweep_full),
        MENU_ITEM('h', "say Hello",                     do_hello_world),
        MENU_END,
    },
};

}  // namespace

extern "C" void do_proj_menu() {
    menu_run(&MENU);
}