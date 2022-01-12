import torch
import taichi as ti

from src.stannum.tube import Tube


@ti.kernel
def mul(arr: ti.template(), out: ti.template()):
    for i in arr:
        out[i] = arr[i] * 2.0


def test_simple_backward():
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


def test_loop_backward():
    ti.init(ti.cpu)

    for i in range(10):
        print(i)
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
        assert torch.allclose(a.grad, torch.ones_like(a) * 2), f"{a.grad}"
