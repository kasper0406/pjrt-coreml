
import os
import jax
import jax._src.xla_bridge as xb
import jax.numpy as jnp

import faulthandler
faulthandler.enable()


def initialize():
    path = os.path.join(os.path.dirname(__file__), 'target/debug/libpjrt_coreml.dylib')
    xb.register_plugin('pjrt-coreml', priority=500, library_path=path, options=None)

initialize()
jax.config.update("jax_platforms", "pjrt-coreml")

jnp.add(1, 2)

# foo = jnp.arange(10)
a = jnp.array([1.0, 2.0, 10.0])
result = a + a

print(result)


@jax.jit
def calculate_matrix_product(a, b):
    foo = ((0.02 * a) @ (0.01 * b)) @ (0.03 * a)
    return jnp.max(foo)


size = 1024 * 8
a = jnp.ones((size, size), dtype=jnp.float32)
b = jnp.ones((size, size), dtype=jnp.float32)

for i in range(100):
    print(f"Calculating prodct {i}")
    print(calculate_matrix_product(a, b))
