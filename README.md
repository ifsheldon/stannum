
<div align="center">
  <img width="300px" src="https://github.com/ifsheldon/stannum/raw/main/logo.PNG"/>
</div>


# Stannum

![Gradient Tests](https://github.com/ifsheldon/stannum/actions/workflows/run_tests.yaml/badge.svg)

Fusing Taichi into PyTorch

**PRs are always welcomed, please see TODOs and issues.**

## Why Stannum?

In differentiable rendering including neural rendering, rendering algorithms are transferred to the field of computer vision, but some rendering operations (e.g., ray tracing and direct volume rendering) are not easy to be expressed in tensor operations but in kernels. Differentiable kernels of Taichi enables fast, efficient and differentiable implementation of rendering algorithms while tensor operators provides math expressiveness. 

Stannum bridges Taichi and PyTorch to have advantage of both kernel-based and operator-based parallelism.

## Documentation and Usage

Please see [documentation](https://fengliang.io/stannum/).

Code sample of `Tube`:

```python
import taichi as ti
import torch

@ti.kernel
def mul(arr: ti.template(), out: ti.template()):
    for i in arr:
        out[i] = arr[i] * 2.0


if __name__ == "__main__":
    ti.init(ti.cpu)
    a = torch.ones(10, requires_grad=True)
    tube = Tube() \
        .register_input_tensor((10,), torch.float32, "arr") \
        .register_output_tensor((10,), torch.float32, "out", True) \
        .register_kernel(mul, ["arr", "out"]) \
        .finish()
    out = tube(a)
    loss = out.sum()
    loss.backward()
    assert torch.allclose(out, torch.ones_like(out) * 2)
    assert torch.allclose(a.grad, torch.ones_like(a) * 2)
```



## Installation & Dependencies

Install `stannum` with `pip` by

`python -m pip install stannum`

Make sure you have the following installed:

* PyTorch
* latest Taichi
    * For performance concerns, we strongly recommend to use Taichi >= 1.1.3 (see Issue #9 for more information)

## Bugs & Issues

Please feel free to file issues. If a runtime error occurs from the dependencies of `stannum`, you may also want to check the [upstream breaking change tracker](https://github.com/ifsheldon/stannum/issues/11).
