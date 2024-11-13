
import jax
import jax.numpy as jnp
from setup_jax import initialize_jax_coreml

initialize_jax_coreml()

a = jnp.array([1.0, 2.0, 10.0])
result = a + a + 4 * a
print(result)


@jax.jit
def calculate_matrix_product(a, b):
    mul = ((0.02 * a) @ (0.01 * b)) @ (0.03 * a)
    result = jnp.max(mul)
    return result

size = 1024 * 8
a = jnp.ones((size, size), dtype=jnp.float32)
b = jnp.ones((size, size), dtype=jnp.float32)

for i in range(10):
    print(f"Calculating prodct {i}")
    print(calculate_matrix_product(a, b))
