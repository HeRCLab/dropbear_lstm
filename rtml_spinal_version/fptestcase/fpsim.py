import sys
from FixedPoint import FXfamily, FXnum
from collections import deque

# Same as bit allocation on the fixed-point pipeline modules.
#  8 bits for the integer part (including sign) [-128..127]
# 24 bits for the fractional part
fixed_point_format = FXfamily(24, 8)


def mkfxnum(n):
    return FXnum(n, fixed_point_format)


# deque -> FXnum -> FXnum -> FXnum -> deque
def step(pipeline, a, b, c):
    pass


if __name__ == '__main__':
    a = [16/(x+1) for x in range(0, 1024)]
    b = [y for y in range(0, 8) for x in range(0, 128)]
    c = []
    print([int(mkfxnum(x).toBinaryString().replace('.', ''), 2) for x in reversed(a)])
    print([int(mkfxnum(x).toBinaryString().replace('.', ''), 2) for x in reversed(b)])
    #print([int(mkfxnum(x)) for x in reversed(a)])
    #print([int(mkfxnum(x)) for x in reversed(b)])
    # stages:   (F) (M M M) (WB)
    num_cycles = 1 + 3 + 1
    p = deque([None] * num_cycles, num_cycles)
    accum = 0
    for i in range(0, 1024 + num_cycles + 1):
        fA = None
        fB = None
        fC = accum
        c.append(int(mkfxnum(accum).toBinaryString().replace('.', ''), 2))
        if len(a) > 0:
            fA = mkfxnum(a.pop())
            fB = mkfxnum(b.pop())
            fC = mkfxnum(accum)
        result = p.pop()
        if fA is not None:
            p.appendleft((fA * fB) + fC)
            accum = (fA * fB) + fC
        else:
            p.appendleft(None)
        print("cycle: {}, a: {}, b: {}, c: {}, result: {}".format(i, fA, fB, fC, result))
        sys.stdout.flush()
print(c)