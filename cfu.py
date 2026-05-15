#!/bin/env python

from amaranth import *
from amaranth_cfu import InstructionBase, InstructionTestBase, simple_cfu, CfuTestBase
import unittest


class AddInstruction(InstructionBase):
    def elab(self, m):
        with m.If(self.start):
            m.d.sync += [
                self.output.eq(self.in0 + self.in1),
                self.done.eq(1),
            ]
        with m.Else():
            m.d.sync += self.done.eq(0)


class Mac16Instruction(InstructionBase):
    def elab(self, m):
        a0 = Signal(signed(16))
        a1 = Signal(signed(16))
        b0 = Signal(signed(16))
        b1 = Signal(signed(16))

        p0 = Signal(signed(32))
        p1 = Signal(signed(32))
        result = Signal(signed(32))

        m.d.comb += [
            a0.eq(self.in0[0:16]),
            a1.eq(self.in0[16:32]),
            b0.eq(self.in1[0:16]),
            b1.eq(self.in1[16:32]),
            p0.eq(a0 * b0),
            p1.eq(a1 * b1),
            result.eq(p0 + p1),
        ]

        with m.If(self.start):
            m.d.sync += [
                self.output.eq(result),
                self.done.eq(1),
            ]
        with m.Else():
            m.d.sync += self.done.eq(0)


def make_cfu():
    return simple_cfu({
        0: AddInstruction(),
        1: Mac16Instruction(),
    })


class CfuTest(CfuTestBase):
    def create_dut(self):
        return make_cfu()

    def test(self):
        DATA = [
            ((0, 22, 22), 44),

            # op1 mac16
            # rs1 = pack16(3, 2) = 0x00020003
            # rs2 = pack16(5, -4) = 0xfffc0005
            # 3*5 + 2*(-4) = 7
            ((1, 0x00020003, 0xfffc0005), 7),

            # -100*-3 + -200*-4 = 1100
            ((1, 0xff38ff9c, 0xfffcfffd), 1100),

            # 1000*32 + -1000*32 = 0
            ((1, 0xfc1803e8, 0x00200020), 0),
        ]
        return self.run_ops(DATA)


if __name__ == '__main__':
    unittest.main()
