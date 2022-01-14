import taichi as ti
import torch
from torch import allclose
from src.stannum.tube import Tube


@ti.kernel
def int_add(a: ti.template(), b: ti.template(), out: ti.template()):
    out[None] = a[None] + b[None]


def test_batch_backward_scalar():
    ti.init(ti.cpu)
    b = torch.tensor(1., requires_grad=True)
    batched_a = torch.ones(10, requires_grad=True)
    tube = Tube() \
        .register_input_tensor((None,), torch.float32, "a") \
        .register_input_tensor((), torch.float32, "b") \
        .register_output_tensor((None,), torch.float32, "out", True) \
        .register_kernel(int_add, ["a", "b", "out"]) \
        .finish()
    out = tube(batched_a, b)
    loss = out.sum()
    loss.backward()
    assert allclose(torch.ones_like(batched_a) + 1, out)
    assert b.grad == 10.
    assert allclose(torch.ones_like(batched_a), batched_a.grad)


def test_batch():
    ti.init(ti.cpu)
    batched_a = torch.ones(10, requires_grad=True)
    batched_b = torch.ones(10, requires_grad=True)
    tube = Tube() \
        .register_input_tensor((None,), torch.float32, "a") \
        .register_input_tensor((None,), torch.float32, "b") \
        .register_output_tensor((None,), torch.float32, "out", True) \
        .register_kernel(int_add, ["a", "b", "out"]) \
        .finish()
    out = tube(batched_a, batched_b)
    loss = out.sum()
    loss.backward()
    one = torch.ones(10)
    assert allclose(one + 1, out)
    assert allclose(one, batched_b.grad)
    assert allclose(one, batched_a.grad)
