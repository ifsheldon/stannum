from typing import Dict, Tuple

import taichi as ti
import torch
from src.stannum import Tube, AnyDim, DimensionCalculator, DimOption


class D1ConvDC(DimensionCalculator):

    def calc_dimension(self,
                       field_name: str,
                       input_dimensions: Dict[str, Tuple[DimOption, ...]],
                       input_tensor_shapes: Dict[str, Tuple[int, ...]]) -> Tuple[DimOption, ...]:
        # 1D conv with stride = 1
        assert field_name == "out"
        conv_shape = input_tensor_shapes["conv"]  # (conv_dim, )
        arr_shape = input_tensor_shapes["arr"]  # (dim, )
        assert conv_shape[0] % 2 == 1
        shrink = conv_shape[0] - 1
        output_dim = (arr_shape[0] - shrink,)
        return output_dim


@ti.kernel
def d1conv(arr: ti.template(), conv: ti.template(), out: ti.template()):
    for out_idx in range(out.shape[0]):
        for conv_idx in range(conv.shape[0]):
            out[out_idx] += arr[out_idx + conv_idx] * conv[conv_idx]


def test_1D_conv():
    ti.init(ti.cpu)
    tube = Tube() \
        .register_input_tensor((AnyDim,), torch.float32, "arr", False) \
        .register_input_tensor((AnyDim,), torch.float32, "conv", False) \
        .register_output_tensor(D1ConvDC(), torch.float32, "out", False) \
        .register_kernel(d1conv, ["arr", "conv", "out"]) \
        .finish()
    arr = torch.arange(5, dtype=torch.float32)
    conv = torch.ones(3, dtype=torch.float32)
    out = tube(arr, conv)
    assert out.shape[0] == 3
    assert torch.allclose(out, torch.tensor([3., 6., 9.]))

    arr = torch.arange(7, dtype=torch.float32)
    conv = torch.ones(3, dtype=torch.float32)
    out = tube(arr, conv)
    assert out.shape[0] == 5
    assert torch.allclose(out, torch.tensor([3., 6., 9., 12., 15.]))

    arr = torch.arange(7, dtype=torch.float32)
    conv = torch.ones(5, dtype=torch.float32)
    out = tube(arr, conv)
    assert out.shape[0] == 3
    assert torch.allclose(out, torch.tensor([10., 15., 20.]))


def test_1D_conv_backward():
    ti.init(ti.cpu)
    tube = Tube() \
        .register_input_tensor((AnyDim,), torch.float32, "arr", True) \
        .register_input_tensor((AnyDim,), torch.float32, "conv", True) \
        .register_output_tensor(D1ConvDC(), torch.float32, "out", True) \
        .register_kernel(d1conv, ["arr", "conv", "out"]) \
        .finish()
    arr = torch.arange(5, dtype=torch.float32, requires_grad=True)
    conv = torch.ones(3, dtype=torch.float32, requires_grad=True)
    out = tube(arr, conv)
    assert out.shape[0] == 3
    assert torch.allclose(out, torch.tensor([3., 6., 9.]))
    l = out.sum()
    l.backward()

    darr = arr.detach()
    dconv = conv.detach()
    expected_conv_grad = torch.stack([darr[:3].sum(), darr[1:4].sum(), darr[2:5].sum()])
    expected_arr_grad = torch.stack([dconv[0:1],
                                     torch.sum(dconv[0:2], (0,), keepdim=True),
                                     torch.sum(dconv[0:3], (0,), keepdim=True),
                                     torch.sum(dconv[1:3], (0,), keepdim=True),
                                     dconv[2:3]]).flatten()
    assert torch.allclose(arr.grad, expected_arr_grad)
    assert torch.allclose(conv.grad, expected_conv_grad)
