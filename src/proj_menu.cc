#include "proj_menu.h"

#include <stdio.h>
#include <stdint.h>

#include "cfu.h"
#include "menu.h"

namespace {

// cycle counter
static inline uint32_t get_cycles() {
  uint32_t cycles;
  asm volatile ("rdcycle %0" : "=r"(cycles));
  return cycles;
}

// pack two signed 16-bit values into one 32-bit word
// low half = a, high half = b
static inline uint32_t pack16(int16_t a, int16_t b) {
  return ((uint32_t)(uint16_t)b << 16) | (uint16_t)a;
}

void do_hello_world(void) {
  puts("Hello, World!!!\n");
}

// ------------------------------------------------------------
// op0 sanity check (unchanged)
// ------------------------------------------------------------
void do_exercise_cfu_op0(void) {
  puts("\r\nExercise CFU Op0 aka ADD\r\n");
  unsigned int a = 0;
  unsigned int b = 0;
  unsigned int cfu = 0;
  unsigned int count = 0;
  unsigned int pass_count = 0;
  unsigned int fail_count = 0;
  for (a = 0x00004567; a < 0xF8000000; a += 0x00212345) {
    for (b = 0x0000ba98; b < 0xFF000000; b += 0x00770077) {
      cfu = cfu_op0(0, a, b);
      if (cfu != a + b) {
        printf("[%4d] a:%08x b:%08x a+b=%08x cfu=%08x FAIL\r\n",
               count, a, b, a + b, cfu);
        fail_count++;
      } else {
        pass_count++;
      }
      count++;
    }
  }
  printf("\r\nPerformed %d comparisons, %d pass, %d fail\r\n",
         count, pass_count, fail_count);
}

// ------------------------------------------------------------
// mac16 spot check
// verify cfu_op1 matches software dot-product on a known case
// ------------------------------------------------------------
void do_exercise_cfu_op1(void) {
  puts("\r\nExercise CFU Op1 aka MAC16\r\n");

  // rs1 = [a1=2, a0=3]   rs2 = [b1=-4, b0=5]
  // expect 3*5 + 2*-4 = 7
  uint32_t rs1 = pack16(3, 2);
  uint32_t rs2 = pack16(5, -4);
  int32_t r = (int32_t)cfu_op1(0, rs1, rs2);
  printf("test 1: got %ld expect 7   %s\r\n",
         (long)r, (r == 7) ? "PASS" : "FAIL");

  // negative-only case
  rs1 = pack16(-100, -200);
  rs2 = pack16(-3, -4);
  r = (int32_t)cfu_op1(0, rs1, rs2);
  int32_t exp = (-100)*(-3) + (-200)*(-4);
  printf("test 2: got %ld expect %ld %s\r\n",
         (long)r, (long)exp, (r == exp) ? "PASS" : "FAIL");

  // mixed sign, larger magnitude
  rs1 = pack16(1000, -1000);
  rs2 = pack16(32, 32);
  r = (int32_t)cfu_op1(0, rs1, rs2);
  exp = 1000*32 + (-1000)*32;
  printf("test 3: got %ld expect %ld %s\r\n",
         (long)r, (long)exp, (r == exp) ? "PASS" : "FAIL");
}

// ------------------------------------------------------------
// scalar fir baseline
// 4 taps, 64 samples, 1000 iterations
// uses int16 storage to match the cfu version exactly
// ------------------------------------------------------------
void do_fir_scalar(void) {
  puts("\r\nScalar FIR (baseline)\r\n");

  const int TAPS = 4;
  const int NUM_SAMPLES = 64;
  const int ITERS = 1000;

  int16_t coeffs[TAPS] = {1, 2, 3, 4};
  int16_t samples[NUM_SAMPLES];
  for (int i = 0; i < NUM_SAMPLES; i++) {
    samples[i] = (int16_t)i;
  }

  volatile int32_t sink = 0;

  uint32_t start = get_cycles();
  for (int iter = 0; iter < ITERS; iter++) {
    // valid output indices only. no shifting, no boundary.
    for (int n = TAPS - 1; n < NUM_SAMPLES; n++) {
      int32_t acc = 0;
      for (int i = 0; i < TAPS; i++) {
        acc += (int32_t)samples[n - i] * (int32_t)coeffs[i];
      }
      sink += acc;
    }
  }
  uint32_t end = get_cycles();

  printf("Cycles: %lu\r\n", (unsigned long)(end - start));
  printf("Sink (ignore): %ld\r\n", (long)sink);
}

// ------------------------------------------------------------
// cfu-accelerated fir
// same filter, same loop count. inner mac replaced with cfu_op1.
// 4 taps -> 2 packed words per output sample.
// ------------------------------------------------------------
void do_fir_cfu(void) {
  puts("\r\nCFU FIR (mac16)\r\n");

  const int TAPS = 4;
  const int NUM_SAMPLES = 64;
  const int ITERS = 1000;

  int16_t coeffs[TAPS] = {1, 2, 3, 4};
  int16_t samples[NUM_SAMPLES];
  for (int i = 0; i < NUM_SAMPLES; i++) {
    samples[i] = (int16_t)i;
  }

  // pre-pack coefficients (reversed order to match samples[n-i] access)
  // for n, the window is samples[n], samples[n-1], samples[n-2], samples[n-3]
  // packed pair 0: [samples[n],   samples[n-1]] * [coeffs[0], coeffs[1]]
  // packed pair 1: [samples[n-2], samples[n-3]] * [coeffs[2], coeffs[3]]
  uint32_t hpack0 = pack16(coeffs[0], coeffs[1]);
  uint32_t hpack1 = pack16(coeffs[2], coeffs[3]);

  volatile int32_t sink = 0;

  uint32_t start = get_cycles();
  for (int iter = 0; iter < ITERS; iter++) {
    for (int n = TAPS - 1; n < NUM_SAMPLES; n++) {
      uint32_t xpack0 = pack16(samples[n],     samples[n - 1]);
      uint32_t xpack1 = pack16(samples[n - 2], samples[n - 3]);

      int32_t acc = (int32_t)cfu_op1(0, xpack0, hpack0)
                  + (int32_t)cfu_op1(0, xpack1, hpack1);
      sink += acc;
    }
  }
  uint32_t end = get_cycles();

  printf("Cycles: %lu\r\n", (unsigned long)(end - start));
  printf("Sink (ignore): %ld\r\n", (long)sink);
}

// ------------------------------------------------------------
// run both, print speedup
// ------------------------------------------------------------
void do_fir_compare(void) {
  puts("\r\n=== FIR comparison ===\r\n");
  do_fir_scalar();
  do_fir_cfu();
  puts("\r\ncompare the two cycle counts above\r\n");
}

struct Menu MENU = {
    "Project Menu",
    "project",
    {
        MENU_ITEM('0', "exercise cfu op0 (add)",     do_exercise_cfu_op0),
        MENU_ITEM('1', "exercise cfu op1 (mac16)",   do_exercise_cfu_op1),
        MENU_ITEM('2', "scalar FIR baseline",        do_fir_scalar),
        MENU_ITEM('3', "CFU FIR (mac16)",            do_fir_cfu),
        MENU_ITEM('4', "run both, compare cycles",   do_fir_compare),
        MENU_ITEM('h', "say Hello",                  do_hello_world),
        MENU_END,
    },
};

};  // namespace

extern "C" void do_proj_menu() {
  menu_run(&MENU);
}
