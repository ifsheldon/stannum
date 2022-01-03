import taichi as ti
import torch
from src.stannum.tube import Tube
import pytest


@ti.kernel
def ti_add(arr_a: ti.template(), arr_b: ti.template(), output_arr: ti.template()):
    for i in arr_a:
        print(i)
        output_arr[i] = arr_a[i] + arr_b[i]


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