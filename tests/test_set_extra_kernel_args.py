import taichi as ti
import torch

from src.stannum import EmptyTin
from src.stannum.tube import Tube


def test_set_extra_args_tin():
    ti.init(ti.cpu)
    arr = ti.field(ti.f32, shape=2, needs_grad=True)
    out = ti.field(ti.f32, shape=2, needs_grad=True)

    @ti.kernel
    def mull(multiplier: float):
        for i in arr:
            out[i] = arr[i] * multiplier

    device = torch.device("cpu")
    tin_layer = EmptyTin(device, True) \
        .register_kernel(mull, 2.0) \
        .register_input_field(arr) \
        .register_output_field(out) \
        .finish()

    a = torch.ones(2, requires_grad=True)
    b = tin_layer(a)
    l = b.sum()
    tin_layer.set_kernel_args(mull, 3.)
    l.backward()

    assert torch.allclose(b, torch.ones(2) * 2.)
    assert torch.allclose(a.grad, torch.ones(2) * 2.)

    a = torch.ones(2, requires_grad=True)
    b = tin_layer(a)
    l = b.sum()
    l.backward()
    assert torch.allclose(b, torch.ones(2) * 3.)
    assert torch.allclose(a.grad, torch.ones(2) * 3.)


@ti.kernel
def mul(arr: ti.template(), out: ti.template(), multiplier: float):
    for i in arr:
        out[i] = arr[i] * multiplier


def test_set_extra_args_tube():
    ti.init(ti.cpu)
    cpu = torch.device("cpu")
    a = torch.ones(10, requires_grad=True)
    tube = Tube(cpu) \
        .register_input_tensor((10,), torch.float32, "arr", True) \
        .register_output_tensor((10,), torch.float32, "out", True) \
        .register_kernel(mul, ["arr", "out"], 2.0) \
        .finish()
    out = tube(a)
    assert torch.allclose(out, torch.full_like(out, 2.))
    l = out.sum()
    tube.set_kernel_extra_args(mul, 3.0)
    l.backward()
    assert torch.allclose(a.grad, torch.full_like(out, 2.))

    a = torch.ones(10, requires_grad=True)
    out = tube(a)
    l = out.sum()
    l.backward()
    assert torch.allclose(out, torch.full_like(out, 3.))
    assert torch.allclose(a.grad, torch.full_like(out, 3.))
