
#include <stdint.h>
#include "software_cfu.h"

// funct3 dispatch
//   0  passthrough add (legacy sanity check)
//   1  mac16  signed packed 16x16 -> 32 dot product
//   2  mac16u unsigned packed 16x16 -> 32 dot product
//   3+ reserved for shuffle, sliding window, etc.
//
// operand layout for mac16 variants
//   rs1[15:0]  = a0   rs1[31:16] = a1
//   rs2[15:0]  = b0   rs2[31:16] = b1
//   result     = a0*b0 + a1*b1  (32-bit, wraps on overflow)
//
// note funct7 is unused for now. reserved for future variants
// like rounding mode or saturation.

static inline uint32_t mac16_signed(uint32_t rs1, uint32_t rs2)
{
    int16_t a0 = (int16_t)(rs1 & 0xFFFF);
    int16_t a1 = (int16_t)(rs1 >> 16);
    int16_t b0 = (int16_t)(rs2 & 0xFFFF);
    int16_t b1 = (int16_t)(rs2 >> 16);

    // sign-extend each lane to 32-bit, multiply, sum
    int32_t p0 = (int32_t)a0 * (int32_t)b0;
    int32_t p1 = (int32_t)a1 * (int32_t)b1;
    return (uint32_t)(p0 + p1);
}

static inline uint32_t mac16_unsigned(uint32_t rs1, uint32_t rs2)
{
    uint16_t a0 = (uint16_t)(rs1 & 0xFFFF);
    uint16_t a1 = (uint16_t)(rs1 >> 16);
    uint16_t b0 = (uint16_t)(rs2 & 0xFFFF);
    uint16_t b1 = (uint16_t)(rs2 >> 16);

    uint32_t p0 = (uint32_t)a0 * (uint32_t)b0;
    uint32_t p1 = (uint32_t)a1 * (uint32_t)b1;
    return p0 + p1;
}

uint32_t software_cfu(int funct3, int funct7, uint32_t rs1, uint32_t rs2)
{
	(void)funct7;

    switch (funct3)
    {
    case 0:
        return rs1 + rs2;
    case 1:
        return mac16_signed(rs1, rs2);
    case 2:
        return mac16_unsigned(rs1, rs2);
    default:
        return rs1;
    }
}
