import os
import jax
import jax._src.xla_bridge as xb
import jax.numpy as jnp


def initialize():
    path = os.path.join(os.path.dirname(__file__), 'target/debug/libpjrt_coreml.dylib')
    xb.register_plugin('pjrt-coreml', priority=500, library_path=path, options=None)


initialize()
jax.config.update("jax_platforms", "pjrt-coreml")

# jnp.add(1, 2)

jnp.arange(10)
