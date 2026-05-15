#!/bin/env python

# cpe515 fir study cfu
#
# top-level dispatch (funct3):
#   0  add               passthrough sanity check
#   1  mac/fir family    sub-dispatched on funct7
#
# mac/fir family (funct3=1, varying funct7):
#   funct7=0  mac16 signed     stateless packed 16x16 -> 32 dot product
#   funct7=1  mac16 unsigned   unsigned packed variant
#   funct7=2  mac16_acc        signed mac16, adds product to internal acc
#   funct7=3  acc_read         returns acc, clears it
#   funct7=4  load_taps        rs1 -> tap_lo, rs2 -> tap_hi
#   funct7=5  mac_loaded       4-tap fir in one call, taps preloaded.
#                              rs1 = packed [s[n], s[n-1]]
#                              rs2 = packed [s[n-2], s[n-3]]
#                              result = dot(rs1, tap_lo) + dot(rs2, tap_hi)
#   funct7=6  swin_push_mac    sliding window 4-tap fir. push new sample
#                              (rs1[15:0]), return dot of new window vs taps.
#   funct7=7  swin_reset       zero the sliding window
#
# packed layout: rs[15:0] = lane0, rs[31:16] = lane1
# all results 32-bit, wrap on overflow.

from amaranth import *
from amaranth_cfu import InstructionBase, simple_cfu
import unittest


def packed_dot_signed(m, a, b):
    a0 = Signal(signed(16))
    a1 = Signal(signed(16))
    b0 = Signal(signed(16))
    b1 = Signal(signed(16))
    p0 = Signal(signed(32))
    p1 = Signal(signed(32))
    out = Signal(signed(32))
    m.d.comb += [
        a0.eq(a[0:16]),
        a1.eq(a[16:32]),
        b0.eq(b[0:16]),
        b1.eq(b[16:32]),
        p0.eq(a0 * b0),
        p1.eq(a1 * b1),
        out.eq(p0 + p1),
    ]
    return out


def packed_dot_unsigned(m, a, b):
    a0 = Signal(unsigned(16))
    a1 = Signal(unsigned(16))
    b0 = Signal(unsigned(16))
    b1 = Signal(unsigned(16))
    p0 = Signal(unsigned(32))
    p1 = Signal(unsigned(32))
    out = Signal(unsigned(32))
    m.d.comb += [
        a0.eq(a[0:16]),
        a1.eq(a[16:32]),
        b0.eq(b[0:16]),
        b1.eq(b[16:32]),
        p0.eq(a0 * b0),
        p1.eq(a1 * b1),
        out.eq(p0 + p1),
    ]
    return out


class AddInstruction(InstructionBase):
    """funct3=0: simple add for sanity checks."""
    def elab(self, m):
        with m.If(self.start):
            m.d.sync += [
                self.output.eq(self.in0 + self.in1),
                self.done.eq(1),
            ]
        with m.Else():
            m.d.sync += self.done.eq(0)


class MacFirInstruction(InstructionBase):
    """funct3=1: mac/fir family, sub-dispatched on funct7."""

    def elab(self, m):
        # internal state. owned entirely by this instruction.
        acc     = Signal(signed(32))
        tap_lo  = Signal(32)
        tap_hi  = Signal(32)
        swin_lo = Signal(32)
        swin_hi = Signal(32)

        # combinational results for each sub-op
        signed_dot   = packed_dot_signed(m, self.in0, self.in1)
        unsigned_dot = packed_dot_unsigned(m, self.in0, self.in1)

        acc_plus_dot = Signal(signed(32))
        m.d.comb += acc_plus_dot.eq(acc + signed_dot)

        d_lo = packed_dot_signed(m, self.in0, tap_lo)
        d_hi = packed_dot_signed(m, self.in1, tap_hi)
        mac_loaded_out = Signal(signed(32))
        m.d.comb += mac_loaded_out.eq(d_lo + d_hi)

        new_sample = self.in0[0:16]
        new_swin_lo = Signal(32)
        new_swin_hi = Signal(32)
        m.d.comb += [
            new_swin_lo.eq(Cat(new_sample, swin_lo[0:16])),
            new_swin_hi.eq(Cat(swin_lo[16:32], swin_hi[0:16])),
        ]
        d_lo_new = packed_dot_signed(m, new_swin_lo, tap_lo)
        d_hi_new = packed_dot_signed(m, new_swin_hi, tap_hi)
        swin_out = Signal(signed(32))
        m.d.comb += swin_out.eq(d_lo_new + d_hi_new)

        # default deassert done; start triggers the work
        m.d.sync += self.done.eq(0)
        with m.If(self.start):
            m.d.sync += self.done.eq(1)
            with m.Switch(self.funct7):
                with m.Case(0):
                    m.d.sync += self.output.eq(signed_dot)
                with m.Case(1):
                    m.d.sync += self.output.eq(unsigned_dot)
                with m.Case(2):
                    m.d.sync += [
                        self.output.eq(acc_plus_dot),
                        acc.eq(acc_plus_dot),
                    ]
                with m.Case(3):
                    m.d.sync += [
                        self.output.eq(acc),
                        acc.eq(0),
                    ]
                with m.Case(4):
                    m.d.sync += [
                        self.output.eq(0),
                        tap_lo.eq(self.in0),
                        tap_hi.eq(self.in1),
                    ]
                with m.Case(5):
                    m.d.sync += self.output.eq(mac_loaded_out)
                with m.Case(6):
                    m.d.sync += [
                        self.output.eq(swin_out),
                        swin_lo.eq(new_swin_lo),
                        swin_hi.eq(new_swin_hi),
                    ]
                with m.Case(7):
                    m.d.sync += [
                        self.output.eq(0),
                        swin_lo.eq(0),
                        swin_hi.eq(0),
                    ]
                with m.Default():
                    m.d.sync += self.output.eq(0)


def make_cfu():
    return simple_cfu({
        0: AddInstruction(),
        1: MacFirInstruction(),
    })


if __name__ == '__main__':
    unittest.main()