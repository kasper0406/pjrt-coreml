import jax
import jax.numpy as jnp
from setup_jax import initialize_jax_coreml
from benchmarking import benchmark, Benchmarking
import numpy as np

# initialize_jax_coreml()

benchmarking = Benchmarking()

def benchmark_matrix_mul():
    @jax.jit
    def calculate_matrix_product(a, b):
        mul = (a @ b) @ a
        result = jnp.max(mul)
        return result

    size = 1024 * 8
    a = np.random.standard_normal((size, size))
    b = np.random.standard_normal((size, size))

    def test_function():
        for i in range(10):
            calculate_matrix_product(a, b)

    result = benchmark("matrix_mul", benchmarking, test_function)
    print(result)

# This currently fails due to it using PRNGs
# def benchmark_res_net():
#     import torch
#     import torchvision
#     import torch_xla2

#     env = torch_xla2.default_env()
#     with env:
#         # inputs = torch.randn(4, 3, 224, 224)
#         inputs = np.random.standard_normal((4, 3, 224, 224))
#         model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
#         model.eval()

#         def test_function():
#             for i in range(10):
#                 outputs = model(inputs)
#                 print(outputs)

#         result = benchmark("resnet18", benchmarking, test_function)
#         print(result)


benchmark_matrix_mul()
# benchmark_res_net()

benchmarking.done_benchmarking()
