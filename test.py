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

# jnp.add(1, 2)

# foo = jnp.arange(10)
a = jnp.array([1.0, 2.0, 10.0])
result = a + a

print(result)
