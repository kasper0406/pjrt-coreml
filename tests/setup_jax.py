import jax
import jax._src.xla_bridge as xb
import os

import faulthandler
faulthandler.enable()


def initialize_jax_coreml():
    path = os.path.join(os.path.dirname(__file__), '../target/debug/libpjrt_coreml.dylib')
    xb.register_plugin('pjrt-coreml', priority=500, library_path=path, options=None)
    jax.config.update("jax_platforms", "pjrt-coreml")
