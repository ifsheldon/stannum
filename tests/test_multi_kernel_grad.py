import taichi as ti
import torch

from stannum import Tin


@ti.data_oriented
class Multiplier:
    def __init__(self, multiplier):
        self.multiplier = ti.field(ti.f32, (), needs_grad=True)
        self.input_field = ti.field(ti.f32, shape=2, needs_grad=True)
        self.intermediate_field = ti.field(ti.f32, shape=2, needs_grad=True)
        self.output_field = ti.field(ti.f32, shape=2, needs_grad=True)
        self.multiplier[None] = multiplier

    @ti.kernel
    def forward_kernel1(self, num: float):
        for i in self.input_field:
            self.intermediate_field[i] = self.multiplier[None] * self.input_field[i] + num

    @ti.kernel
    def forward_kernel2(self):
        for i in self.intermediate_field:
            self.output_field[i] = 2.0 * self.intermediate_field[i]


def test_grad_with_data_oriented_class_multikernel():
    ti.init(ti.cpu, default_fp=ti.f32)
    data_oriented = Multiplier(2.0)
    device = torch.device("cpu")
    tin_layer = Tin(data_oriented, device=device) \
        .register_kernel(data_oriented.forward_kernel1, 1.0, kernel_name="forward1") \
        .register_kernel(data_oriented.forward_kernel2, kernel_name="forward2") \
        .register_input_field(data_oriented.input_field) \
        .register_output_field(data_oriented.output_field) \
        .register_internal_field(data_oriented.multiplier, name="multiplier num") \
        .register_internal_field(data_oriented.intermediate_field, name="intermediate results") \
        .finish()
    data_tensor = torch.tensor([0.5, 0.5]).requires_grad_(True).to(device)
    w1 = torch.ones(2).requires_grad_(True).to(device)
    w2 = torch.tensor([2.0, 2.0]).requires_grad_(True).to(device)
    output1 = w1 * data_tensor
    output2 = tin_layer(output1)
    output3 = output2 * w2
    loss = output3.sum()
    loss.backward()
    assert torch.allclose(w1.grad, torch.tensor(4.0))
    assert torch.allclose(w2.grad, torch.tensor(4.0))
    assert torch.allclose(data_tensor.grad, torch.tensor(8.0))
    assert torch.allclose(data_oriented.multiplier.grad.to_torch(device), torch.tensor(4.0))

if __name__ == "__main__":
    test_grad_with_data_oriented_class_multikernel()
