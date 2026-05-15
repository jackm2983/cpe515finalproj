#include <stdint.h>

extern "C" uint32_t software_cfu(
    int funct3,
    int funct7,
    uint32_t rs1,
    uint32_t rs2)
{
    return 0;
}