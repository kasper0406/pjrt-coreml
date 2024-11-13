# PJRT plugin based on Apple CoreML

The hope is to at some point to get Jax to execute on the Apple Neural Engine (ANE) by going through CoreML.

The ANE is a piece of dedicated hardware withing M-series processors allowing fast and memory-efficient calculations on `fp16` types.

# Missing functionality

* Highly experimental, and not well tested
* Jax PRNG's rely on respectively the `uint32` datatype, as well as the ability to perform logical `and`, `or`, `xor` etc. on `uint32`'s. Unfortunately this is not currently supported through CoreML.
* 

# Benchmarks

I need to perform some proper benchmarks...

