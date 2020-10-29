# Schedule predictor for MAC core pipelining.
# Copyright (c) Philip Conrad, 2020. All rights reserved.

import sys
from pprint import pprint

usage = """Usage:
    python3 scheduler.py MULT_UNITS ADD_UNITS INPUT_LEN LAYERS

Notes:
    LAYERS is expected to be comma-separated integers.
Ex:
    scheduler.py 4 6 20 5,10,1
"""

class WorkUnit(object):
    def __init__(self, input_idx, neuron_idx):
        self.input_idx = input_idx
        self.neuron_idx = neuron_idx
        self.state = "-"

    def advance_state(self):
        if self.state == "-":
            self.state = "F"
        elif self.state == "F":
            self.state = "M"
        elif self.state == "M":
            self.state = "A"
        elif self.state == "A":
            self.state = "O"
        else:
            self.state = ""

    def label(self):
        return "n{}-".format(self.input_idx, self.neuron_idx)

    def __str__(self):
        return self.state + " {}-{}".format(self.input_idx, self.neuron_idx)

    def __repr__(self):
        return self.__str__()


def dump_schedule(mult_unit_states, adder_unit_states, outputs):
    pass

def run_sched(inputs, weights, neurons, fr_inputs, fr_weights, mult_units=1, add_units=1):
    mult_in_use = [False for x in range(0, mult_units)]
    add_in_use = [False for x in range(0, mult_units)]
    outputs = []
    neuron_ids = [x for x in range(0, neurons)]
    fetched_inputs = []
    fetched_weights = []
    work_units = [WorkUnit(x, y) for (x,y) in [(a,b) for a in inputs for b in neuron_ids]]
    pprint(work_units)

    while len(outputs) < neurons:
        # Do fetches.
        fetched_inputs  += inputs[:fr_inputs]
        fetched_weights += inputs[:fr_weights]
        inputs  = inputs[:fr_inputs]
        weights = weights[:fr_weights]
        # React to anything that we can move to the first pipeline stage.
        for i in range(0, len(fetched_weights))
        outputs.append(None)


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print(usage)
        exit(1)
    multiplier_units = int(sys.argv[1])
    adder_units = int(sys.argv[2])
    input_length = int(sys.argv[3])
    # Only need to schedule first layer pair on weights for now.
    layers = [int(x) for x in sys.argv[4].split(",")]
    weights = [x for x in range(0, input_length * layers[0])]
    inputs = [x for x in range(0, input_length)]

    fetch_rate_inputs = 1
    fetch_rate_weights = 7

    run_sched(inputs, weights, layers[0], fetch_rate_inputs, fetch_rate_weights, multiplier_units, adder_units)
