import taichi as ti
import torch
from src.stannum.tube import Tube
import pytest


@ti.kernel
def ti_add(arr_a: ti.template(), arr_b: ti.template(), output_arr: ti.template()):
    for i in arr_a:
        output_arr[i] = arr_a[i] + arr_b[i]


@ti.kernel
def ti_print(arr: ti.template(), out: ti.template()):
    for i in ti.grouped(arr):
        x = i[0]
        y = i[1]
        print(f"{x} {y} {arr[i]}")
        out[i] = arr[i]


@ti.kernel
def ti_print_complex(arr: ti.template(), out: ti.template()):
    for i in arr:
        real = arr[i][0]
        imag = arr[i][1]
        print(f"{real} + {imag}i")
        out[i] = arr[i]


def test_view():
    ti.init(ti.cpu)
    a = torch.tensor([[0., 1.], [2., 3.]])
    a_t = a.T
    assert a_t._is_view()
    cpu = torch.device("cpu")
    tube = Tube(cpu) \
        .register_input_tensor((2, 2), torch.float, "arr", False) \
        .register_output_tensor((2, 2), torch.float, "out", False) \
        .register_kernel(ti_print, ["arr", "out"]) \
        .finish()
    out = tube(a_t)
    assert torch.allclose(out, a_t)


def test_complex_values():
    ti.init(ti.cpu)
    a = torch.ones(10, dtype=torch.cfloat)
    cpu = torch.device("cpu")
    tube = Tube(cpu) \
        .register_input_tensor((10,), torch.cfloat, "arr", False) \
        .register_output_tensor((10,), torch.cfloat, "out", False) \
        .register_kernel(ti_print_complex, ["arr", "out"]) \
        .finish()
    out = tube(a)
    assert torch.allclose(out, a)


def test_simple_add():
    ti.init(ti.cpu)
    cpu = torch.device("cpu")
    a = torch.ones(10)
    b = torch.ones(10)
    tube = Tube(cpu) \
        .register_input_tensor((10,), torch.float32, "arr_a", False) \
        .register_input_tensor((10,), torch.float32, "arr_b", False) \
        .register_output_tensor((10,), torch.float32, "output_arr", False) \
        .register_kernel(ti_add, ["arr_a", "arr_b", "output_arr"]) \
        .finish()
    out = tube(a, b)
    assert torch.allclose(out, torch.full_like(out, 2.))


def test_any_dims_match():
    ti.init(ti.cpu)
    cpu = torch.device("cpu")
    tube = Tube(cpu) \
        .register_input_tensor((-2,), torch.float32, "arr_a", False) \
        .register_input_tensor((-2,), torch.float32, "arr_b", False) \
        .register_output_tensor((-2,), torch.float32, "output_arr", False) \
        .register_kernel(ti_add, ["arr_a", "arr_b", "output_arr"]) \
        .finish()
    dim = 10
    a = torch.ones(dim)
    b = torch.ones(dim)
    out = tube(a, b)
    assert torch.allclose(out, torch.full((dim,), 2.))
    dim = 100
    a = torch.ones(dim)
    b = torch.ones(dim)
    out = tube(a, b)
    assert torch.allclose(out, torch.full((dim,), 2.))


def test_any_dims_unregister_error():
    ti.init(ti.cpu)
    cpu = torch.device("cpu")
    with pytest.raises(Exception):
        tube = Tube(cpu) \
            .register_input_tensor((-2,), torch.float32, "arr_a", False) \
            .register_input_tensor((-2,), torch.float32, "arr_b", False) \
            .register_output_tensor((-3,), torch.float32, "output_arr", False) \
            .register_kernel(ti_add, ["arr_a", "arr_b", "output_arr"]) \
            .finish()


def test_any_dims_error():
    ti.init(ti.cpu)
    cpu = torch.device("cpu")
    with pytest.raises(Exception):
        tube = Tube(cpu) \
            .register_input_tensor((-1,), torch.float32, "arr_a", False) \
            .register_input_tensor((-1,), torch.float32, "arr_b", False) \
            .register_output_tensor((-1,), torch.float32, "output_arr", False) \
            .register_kernel(ti_add, ["arr_a", "arr_b", "output_arr"]) \
            .finish()


def test_any_dims():
    ti.init(ti.cpu)
    cpu = torch.device("cpu")

    tube = Tube(cpu) \
        .register_input_tensor((-1,), torch.float32, "arr_a", False) \
        .register_input_tensor((-1,), torch.float32, "arr_b", False) \
        .register_output_tensor((13,), torch.float32, "output_arr", False) \
        .register_kernel(ti_add, ["arr_a", "arr_b", "output_arr"]) \
        .finish()

    a = torch.ones(10)
    b = torch.ones(15)
    out = tube(a, b)
    assert torch.allclose(out, torch.tensor([2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 0., 0., 0.]))


@ti.kernel
def add_scalar(arr: ti.template(), scalar: ti.template(), out: ti.template()):
    for i in arr:
        out[i] = arr[i] + scalar[None]


def test_scalar():
    ti.init(ti.cpu)
    cpu = torch.device("cpu")
    tube = Tube(cpu) \
        .register_input_tensor((10,), torch.float32, "arr") \
        .register_input_tensor((), torch.float32, "scalar") \
        .register_output_tensor((10,), torch.float32, "out", True) \
        .register_kernel(add_scalar, ["arr", "scalar", "out"]) \
        .finish()
    a = torch.ones(10, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    out = tube(a, b)
    out.sum().backward()
    assert torch.allclose(out, torch.full_like(out, 2.0))
    assert b.grad == 10.

# def test_any_dims_out_of_bound():
#     ti.init(ti.cpu)
#     cpu = torch.device("cpu")
#
#     tube = Tube(cpu) \
#         .register_input_tensor((-1,), torch.float32, "arr_a", False) \
#         .register_input_tensor((-1,), torch.float32, "arr_b", False) \
#         .register_output_tensor((5,), torch.float32, "output_arr", False) \
#         .register_kernel(ti_add, ["arr_a", "arr_b", "output_arr"]) \
#         .finish()
#
#     a = torch.ones(10)
#     b = torch.ones(2)
#     out = tube(a, b)
#     print(out)
