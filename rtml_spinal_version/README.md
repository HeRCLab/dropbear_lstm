Spinal HDL RTML implementation
------------------------------

Currently, this project builds an application called `mlpx2rtl`, which will generate a Verilog or VHDL file implementing a neural network described by an MLPX file.


## Usage

Given a single-layer neural network in a file called `example.mlpx`, you would run `mlpx2rtl` as follows:

    mlpx2rtl example.mlpx

This will generate a file named after the MLPX file: `example.v`


## Planned Features

 - Verilog/VHDL generation.
 - Estimated hardware usage statistics.

Forward-propagation:

 - Multiple selectable lowering strategies.
   - Pipelined vector-mult layers.
   - Pipelined, shared MAC core.
 - Multiple storage options for weights/inputs.

Back-propagation:

 - Support is planned for this, but development is on-hold for now.


