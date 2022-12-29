import taichi as ti
import torch

from src.stannum import Tin


@ti.data_oriented
class Multiplier0:
    def __init__(self, multiplier):
        self.multiplier = ti.field(ti.f32, (), needs_grad=True)
        self.input_field = ti.field(ti.f32, shape=(2, 2), needs_grad=True)
        self.output_field = ti.field(ti.f32, shape=(2, 2), needs_grad=True)
        self.multiplier[None] = multiplier

    @ti.kernel
    def forward_kernel(self, num: float):
        for i in range(self.input_field.shape[0]):
            self.output_field[i, 0] = self.multiplier[None] * self.input_field[i, 0] + num
            self.output_field[i, 1] = self.multiplier[None] * self.input_field[i, 1] + num


def test_complex_scalar_field():
    ti.init(ti.cpu)
    data_oriented_scalar_field = Multiplier0(2)
    device = torch.device("cpu")
    tin_layer = Tin(data_oriented_scalar_field, device, True) \
        .register_kernel(data_oriented_scalar_field.forward_kernel, 1.0) \
        .register_input_field(data_oriented_scalar_field.input_field, complex_dtype=True) \
        .register_output_field(data_oriented_scalar_field.output_field, complex_dtype=True) \
        .register_internal_field(data_oriented_scalar_field.multiplier) \
        .finish()
    data_tensor = torch.tensor([1 + 1j, 2 + 2j], requires_grad=True, device=device)
    w1 = torch.ones(2).requires_grad_(True).to(device)
    w2 = torch.tensor([2.0, 2.0]).requires_grad_(True).to(device)
    output1 = w1 * data_tensor
    output2 = tin_layer(output1)
    output3 = output2 * w2
    loss = torch.view_as_real(output3).sum()
    loss.backward()

    data_tensor_ref = torch.tensor([1 + 1j, 2 + 2j], requires_grad=True, device=device)
    w1_ref = torch.ones(2).requires_grad_(True).to(device)
    w2_ref = torch.tensor([2.0, 2.0]).requires_grad_(True).to(device)
    o1 = w1_ref * data_tensor_ref
    o2 = torch.view_as_complex(torch.view_as_real(o1) * 2 + 1.0)
    o3 = o2 * w2_ref
    l = torch.view_as_real(o3).sum()
    l.backward()
    assert torch.allclose(data_tensor_ref.grad, data_tensor.grad)
    assert torch.allclose(w1_ref.grad, w1.grad)
    assert torch.allclose(w2_ref.grad, w2.grad)


@ti.data_oriented
class Multiplier1:
    def __init__(self, multiplier):
        self.multiplier = ti.field(ti.f32, (), needs_grad=True)
        self.input_field = ti.Vector.field(n=2, dtype=ti.f32, shape=(2,), needs_grad=True)
        self.output_field = ti.Vector.field(n=2, dtype=ti.f32, shape=(2,), needs_grad=True)
        self.multiplier[None] = multiplier

    @ti.kernel
    def forward_kernel(self, num: float):
        for i in range(self.input_field.shape[0]):
            self.output_field[i] = self.multiplier[None] * self.input_field[i] + num


def test_complex_vector_field():
    ti.init(ti.cpu)
    data_oriented_vector_field = Multiplier1(2)
    device = torch.device("cpu")
    tin_layer = Tin(data_oriented_vector_field, device, True) \
        .register_kernel(data_oriented_vector_field.forward_kernel, 1.0) \
        .register_input_field(data_oriented_vector_field.input_field, complex_dtype=True) \
        .register_output_field(data_oriented_vector_field.output_field, complex_dtype=True) \
        .register_internal_field(data_oriented_vector_field.multiplier) \
        .finish()
    data_tensor = torch.tensor([1 + 1j, 2 + 2j], requires_grad=True, device=device)
    w1 = torch.ones(2, requires_grad=True, device=device)
    w2 = torch.tensor([2.0, 2.0], requires_grad=True, device=device)
    output1 = w1 * data_tensor
    output2 = tin_layer(output1)
    output3 = output2 * w2
    loss = torch.view_as_real(output3).sum()
    loss.backward()

    data_tensor_ref = torch.tensor([1 + 1j, 2 + 2j], requires_grad=True, device=device)
    w1_ref = torch.ones(2, requires_grad=True, device=device)
    w2_ref = torch.tensor([2.0, 2.0], requires_grad=True, device=device)
    o1 = w1_ref * data_tensor_ref
    o2 = torch.view_as_complex(torch.view_as_real(o1) * 2 + 1.0)
    o3 = o2 * w2_ref
    l = torch.view_as_real(o3).sum()
    l.backward()
    assert torch.allclose(data_tensor_ref.grad, data_tensor.grad)
    assert torch.allclose(w1_ref.grad, w1.grad)
    assert torch.allclose(w2_ref.grad, w2.grad)
