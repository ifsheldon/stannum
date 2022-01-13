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


@ti.kernel
def mul_complex(arr0: ti.template(),
                arr1: ti.template(),
                arr2: ti.template(),
                arr3: ti.template(),
                arr4: ti.template(),
                arr5: ti.template(),
                arr6: ti.template(),
                out: ti.template()):
    for i in arr0:
        out[i] = (arr0[i] + arr1[i] + arr2[i] + arr3[i] + arr4[i] + arr5[i] + arr6[i]) * 2.0


def test_loop_backward():
    ti.init(ti.cpu)

    for i in range(100):
        print(i)
        a0 = torch.ones(10, requires_grad=True)
        a1 = torch.ones(10, requires_grad=True)
        a2 = torch.ones(10, requires_grad=True)
        a3 = torch.ones(10, requires_grad=True)
        a4 = torch.ones(10, requires_grad=True)
        a5 = torch.ones(10, requires_grad=True)
        a6 = torch.ones(10, requires_grad=True)
        tube = Tube() \
            .register_input_tensor((10,), torch.float32, "arr0") \
            .register_input_tensor((10,), torch.float32, "arr1") \
            .register_input_tensor((10,), torch.float32, "arr2") \
            .register_input_tensor((10,), torch.float32, "arr3") \
            .register_input_tensor((10,), torch.float32, "arr4") \
            .register_input_tensor((10,), torch.float32, "arr5") \
            .register_input_tensor((10,), torch.float32, "arr6") \
            .register_output_tensor((10,), torch.float32, "out", True) \
            .register_kernel(mul_complex, ["arr0", "arr1", "arr2", "arr3", "arr4", "arr5", "arr6", "out"]) \
            .finish()
        out = tube(a0, a1, a2, a3, a4, a5, a6)
        loss = out.sum()
        loss.backward()
        two = torch.ones(10) * 2
        assert torch.allclose(a0.grad, two), f"{a0.grad}"
        assert torch.allclose(a1.grad, two), f"{a1.grad}"
        assert torch.allclose(a2.grad, two), f"{a2.grad}"
        assert torch.allclose(a3.grad, two), f"{a3.grad}"
        assert torch.allclose(a4.grad, two), f"{a4.grad}"
        assert torch.allclose(a5.grad, two), f"{a5.grad}"
        assert torch.allclose(a6.grad, two), f"{a6.grad}"
