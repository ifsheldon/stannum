import taichi as ti
import torch

from src.stannum import Tin


@ti.data_oriented
class Multiplier:
    def __init__(self, multiplier):
        self.multiplier = ti.field(ti.f32, (), needs_grad=True)
        self.input_field = ti.field(ti.f32, shape=2, needs_grad=True)
        self.output_field = ti.field(ti.f32, shape=2, needs_grad=True)
        self.multiplier[None] = multiplier

    @ti.kernel
    def forward_kernel(self, num: float):
        for i in self.input_field:
            self.output_field[i] = self.multiplier[None] * self.input_field[i] + num


def test_grad_with_data_oriented_class():
    ti.init(ti.cpu, default_fp=ti.f32)
    data_oriented = Multiplier(2.0)
    device = torch.device("cpu")
    tin_layer = Tin(data_oriented, device, True) \
        .register_kernel(data_oriented.forward_kernel, 1.0, kernel_name="forward") \
        .register_input_field(data_oriented.input_field) \
        .register_output_field(data_oriented.output_field) \
        .register_internal_field(data_oriented.multiplier, name="multiplier num") \
        .finish()
    data_tensor = torch.tensor([0.5, 0.5]).requires_grad_(True).to(device)
    w1 = torch.ones(2).requires_grad_(True).to(device)
    w2 = torch.tensor([2.0, 2.0]).requires_grad_(True).to(device)
    output1 = w1 * data_tensor
    output2 = tin_layer(output1)
    output3 = output2 * w2
    loss = output3.sum()
    loss.backward()
    assert torch.allclose(w1.grad, torch.tensor(2.0))
    assert torch.allclose(w2.grad, torch.tensor(2.0))
    assert torch.allclose(data_tensor.grad, torch.tensor(4.0))
    assert torch.allclose(data_oriented.multiplier.grad.to_torch(device), torch.tensor(2.0))


@ti.data_oriented
class TiAdder:
    def __init__(self):
        self.arr0 = ti.field(ti.f32, shape=(2,), needs_grad=True)
        self.arr1 = ti.field(ti.f32, shape=(2,), needs_grad=False)
        self.arr_sum = ti.field(ti.f32, shape=(2,), needs_grad=True)

    @ti.kernel
    def add(self):
        for i in self.arr0:
            self.arr_sum[i] = self.arr0[i] + self.arr1[i]


def test_grad_with_some_nongrad_field():
    ti.init(ti.cpu)
    data_oriented = TiAdder()
    device = torch.device("cpu")
    tin_layer = Tin(data_oriented, device, True) \
        .register_kernel(data_oriented.add) \
        .register_input_field(data_oriented.arr0) \
        .register_input_field(data_oriented.arr1) \
        .register_output_field(data_oriented.arr_sum) \
        .finish()

    a = torch.ones(2, requires_grad=True)
    b = torch.ones(2)
    l = tin_layer(a, b).sum()
    l.backward()
    assert torch.allclose(a.grad, torch.ones(2))
