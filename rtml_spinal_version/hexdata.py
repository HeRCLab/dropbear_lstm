# Generates hex files for use with Spinal HDL.
import random
import sys


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage:\n  python3 hexdata.py BIT_WIDTH DEPTH")
        exit(1)

    bit_width = int(sys.argv[1])
    depth = int(sys.argv[2])
    fmt_str = "{:0" + str(bit_width//4) + "X}"

    for i in range(0, depth):
        print(fmt_str.format(random.randint(0, 2 ** bit_width - 1)))
