#include <stdint.h>
#include "software_cfu.h"

// dispatch:
//   funct3=0           -> add (rs1+rs2)
//   funct3=1, funct7=0 -> mac16 signed (stateless)
//   funct3=1, funct7=1 -> mac16 unsigned
//   funct3=1, funct7=2 -> mac16_acc (signed, adds to acc)
//   funct3=1, funct7=3 -> acc_read (returns acc, clears it)
//   funct3=1, funct7=4 -> load_taps (rs1->tap_lo, rs2->tap_hi)
//   funct3=1, funct7=5 -> mac_loaded (4-tap fir, taps preloaded)
//   funct3=1, funct7=6 -> swin_push_mac
//   funct3=1, funct7=7 -> swin_reset
//
// packed layout: rs[15:0] = lane0, rs[31:16] = lane1
// signed result = a0*b0 + a1*b1 (32-bit, wraps on overflow)

static int32_t  g_acc      = 0;
static uint32_t g_tap_lo   = 0;
static uint32_t g_tap_hi   = 0;
static uint32_t g_swin_lo  = 0;
static uint32_t g_swin_hi  = 0;

static inline int32_t mac16_signed(uint32_t rs1, uint32_t rs2)
{
    int16_t a0 = (int16_t)(rs1 & 0xFFFF);
    int16_t a1 = (int16_t)(rs1 >> 16);
    int16_t b0 = (int16_t)(rs2 & 0xFFFF);
    int16_t b1 = (int16_t)(rs2 >> 16);
    int32_t p0 = (int32_t)a0 * (int32_t)b0;
    int32_t p1 = (int32_t)a1 * (int32_t)b1;
    return p0 + p1;
}

static inline uint32_t mac16_unsigned(uint32_t rs1, uint32_t rs2)
{
    uint16_t a0 = (uint16_t)(rs1 & 0xFFFF);
    uint16_t a1 = (uint16_t)(rs1 >> 16);
    uint16_t b0 = (uint16_t)(rs2 & 0xFFFF);
    uint16_t b1 = (uint16_t)(rs2 >> 16);
    return (uint32_t)a0 * (uint32_t)b0
         + (uint32_t)a1 * (uint32_t)b1;
}

uint32_t software_cfu(int funct3, int funct7, uint32_t rs1, uint32_t rs2)
{
    if (funct3 == 0) {
        return rs1 + rs2;
    }
    if (funct3 != 1) {
        return rs1;
    }

    switch (funct7) {
    case 0:
        return (uint32_t)mac16_signed(rs1, rs2);

    case 1:
        return mac16_unsigned(rs1, rs2);

    case 2: {
        int32_t out = g_acc + mac16_signed(rs1, rs2);
        g_acc = out;
        return (uint32_t)out;
    }

    case 3: {
        uint32_t out = (uint32_t)g_acc;
        g_acc = 0;
        return out;
    }

    case 4:
        g_tap_lo = rs1;
        g_tap_hi = rs2;
        return 0;

    case 5: {
        int32_t d0 = mac16_signed(rs1, g_tap_lo);
        int32_t d1 = mac16_signed(rs2, g_tap_hi);
        return (uint32_t)(d0 + d1);
    }

    case 6: {
        uint16_t ns = (uint16_t)(rs1 & 0xFFFF);
        uint32_t new_lo = ((uint32_t)(g_swin_lo & 0xFFFF) << 16) | ns;
        uint32_t new_hi = ((uint32_t)(g_swin_hi & 0xFFFF) << 16)
                          | (g_swin_lo >> 16);
        g_swin_lo = new_lo;
        g_swin_hi = new_hi;
        int32_t d0 = mac16_signed(new_lo, g_tap_lo);
        int32_t d1 = mac16_signed(new_hi, g_tap_hi);
        return (uint32_t)(d0 + d1);
    }

    case 7:
        g_swin_lo = 0;
        g_swin_hi = 0;
        return 0;

    default:
        return rs1;
    }
}