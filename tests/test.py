import taichi as ti
import torch

from pytin import Tin


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


if __name__ == "__main__":
    ti.init(ti.cpu, default_fp=ti.f32)
    data_oriented = Multiplier(2.0)
    device = torch.device("cpu")
    tin_layer = Tin(data_oriented, device=device) \
        .register_kernel(data_oriented.forward_kernel) \
        .register_input_field(data_oriented.input_field, True) \
        .register_output_field(data_oriented.output_field, True) \
        .register_weight_field(data_oriented.multiplier, True, name="multiplier num") \
        .finish()
    tin_layer.set_kernel_args(1.0)
    data_tensor = torch.tensor([0.5, 0.5]).requires_grad_(True).to(device)
    print(f"data = {data_tensor}")
    w1 = torch.ones(2).requires_grad_(True).to(device)
    w2 = torch.tensor([2.0, 2.0]).requires_grad_(True).to(device)
    output1 = w1 * data_tensor
    print(f"output1 = {output1}")
    output2 = tin_layer(output1)
    output2.retain_grad()
    print(f"output2 = {output2}")
    output3 = output2 * w2
    print(f"output3 = {output3}")
    loss = output3.sum()
    loss.backward()
    print(f"w1 = {w1}\nw2 = {w2}")
    print(f"gradients:\n"
          f"w1 grad = {w1.grad}\n"
          f"w2 grad = {w2.grad}\n"
          f"output2 grad = {output2.grad}\n"
          f"data tensor grad = {data_tensor.grad}\n"
          f"multiplier grad = {data_oriented.multiplier.grad}"
          )
