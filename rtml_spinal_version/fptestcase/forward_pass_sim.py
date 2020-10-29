# Forward-Pass Test Case generator/tester
# This tester is designed to be cycle-accurate to what the fixed-point pipeline will produce.

from FixedPoint import FXfamily, FXnum
from collections import deque

# Same as bit allocation on the fixed-point pipeline modules.
#  8 bits for the integer part (including sign) [-128..127]
# 24 bits for the fractional part
fixed_point_format = FXfamily(24, 8)

# S-AXI streams are fed in as dictionaries with the keys {"ready", "valid", "payload"}
#                                                        {bool,    bool,    FXnum    }
class fp_add32(object):
    def __init__(self):
        self.sfixA = FXnum(0, fixed_point_format)
        self.sfixB = FXnum(0, fixed_point_format)
        self.result_payload = deque([None, None], maxlen=2)
        self.ready_valid_tracker = deque([None, None, None], maxlen=3)
        self.a_ready = False
        self.b_ready = False

    def step(self, a, b, result):
        # Do steps in reverse-of-logical order, to ensure correct data dependencies.
        self.result_payload.appendleft(self.sfixA + self.sfixB)
        self.ready_valid_tracker.appendleft(all([result["ready"], self.a_ready, self.b_ready]))
        self.a_ready = result["ready"]
        self.b_ready = result["ready"]
        self.sfixA = a["payload"]
        self.sfixB = b["payload"]

    def output(self):
        return {"valid": self.ready_valid_tracker[2], "payload": self.result_payload[1]}


class fp_mul32(object):
    def __init__(self):
        self.sfixA = FXnum(0, fixed_point_format)
        self.sfixB = FXnum(0, fixed_point_format)
        self.intern_result = FXnum(0, fixed_point_format)
        self.result_payload = FXnum(0, fixed_point_format)
        self.ready_valid_tracker = deque([0, 0, 0], maxlen=3)
        self.a_ready = False
        self.b_ready = False

    # Returns (ready, valid, result_payload)
    def step(self, a, b, result):
        # Do steps in reverse-of-logical order, to ensure correct data dependencies.
        self.result_payload = self.intern_result
        self.intern_result = self.sfixA * self.sfixB
        self.ready_valid_tracker.appendleft(all([result["ready"], self.a_ready, self.b_ready]))
        self.a_ready = result["ready"]
        self.b_ready = result["ready"]
        self.sfixA = a["payload"]
        self.sfixB = b["payload"]

    def output(self):
        return {"valid": self.ready_valid_tracker[2], "payload": self.intern_result}


# Advances the core's state by 1 cycle.
class forward_pass(object):
    def __init__(self):
        empty_bram = [0] * 1024
        self.brams = {i: list(empty_bram) for i in range(0, 8)}
        self.multiplier_core = fp_mul32()
        self.adder_core = fp_add32()
        self.output_fifo_payload = 
        self.output_fifo_valid = False

    # Control param is a ready-valid-payload dict: {"ready", "valid", "payload"}
    def step(self, bram0, bram1, bram2, bram3, bram4, bram5, bram6, bram7, control):
        pass


def step_core():
    pass


def run_test():
    # Always:
    #   When state machine in Idle state, and control parameter available, start state machine.
    #   Wait until inputs exhausted and pipeline empty.
    #   Move to final state.
    # Always:
    #   If state machine in Working state:
    #     Foreach input, fetch one value, and store in an input register.
    #     Foreach input register, feed input pair into MAC pipeline.
    #     Feed input to second stage of pipeline.
    #     Feed input to feedback register or FIFO.
    #   If FIFO filled:
    #     State machine moves to Done state.
    #   If state machine in Done state:
    #     Wait until FIFO empty.
    #     State machine moves to Idle state.
    pass


if __name__ == '__main__':
    run_test()
