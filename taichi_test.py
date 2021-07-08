from typing import Any

import taichi as ti
import torch.random

import tin


class FakeFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _input):
        ctx._input = _input
        return torch.ones(2).requires_grad_(True)

    @staticmethod
    def backward(ctx, *grad_outputs):
        print("????")
        return torch.ones(2)


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
    multiplier = Multiplier(2.0)
    multiplier_field = tin.TaichiField(multiplier.multiplier, True, True)
    input_field = tin.TaichiField(multiplier.input_field, True, True)
    output_field = tin.TaichiField(multiplier.output_field, False, True)
    device = torch.device("cpu")
    taichi_function = tin.TinFunc()
    tin_configs = tin.TinConfigs(multiplier,
                                 [multiplier_field, input_field],
                                 [output_field],
                                 device,
                                 1.0
                                 )
    fake_func = FakeFunc()
    data_tensor = torch.tensor([0.5, 0.5]).requires_grad_(True).to(device)
    print(f"data = {data_tensor}")
    w1 = torch.ones(2).requires_grad_(True).to(device)
    w2 = torch.tensor([2.0, 2.0]).requires_grad_(True).to(device)
    output1 = w1 * data_tensor
    print(f"output1 = {output1}")
    # output2 = fake_func.apply(output1)
    output2 = taichi_function.apply(tin_configs, multiplier.multiplier.to_torch(device=device), output1)
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
          f"multiplier grad = {multiplier.multiplier.grad}"
          )
