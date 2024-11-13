# PJRT plugin based on Apple CoreML

The hope is to at some point to get Jax to execute on the Apple Neural Engine (ANE) by going through CoreML.

The ANE is a piece of dedicated hardware withing M-series processors allowing fast and memory-efficient calculations on `fp16` types.

## How to use
In order to run there examples, you need to:

1. Set up a new Python 3.10 or 3.11 (coremltools requirement) environment
2. Install the `stablehlo-coreml-experimental` python package
3. Use the Rust nightly toolchain, and build the dylib using `cargo build`
4. Run the examples, fx `python tests/test_simple.py`

To gain additional error and debug information, you can run the program as:
```sh
RUST_BACKTRACE=1 RUST_LOG=DEBUG python tests/test_simple.py
```

# Missing functionality

* Highly experimental, and not well tested
* Jax PRNG's rely on respectively the `uint32` datatype, as well as the ability to perform logical `and`, `or`, `xor` etc. on `uint32`'s. Unfortunately this is not currently supported through CoreML.
* 

# Benchmarks

I need to perform some proper benchmarks...

