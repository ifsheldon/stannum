import taichi as ti
import torch

from stannum import EmptyTin

ti.init(ti.cpu)
multiplier = ti.field(ti.f32, (), needs_grad=True)
input_field = ti.field(ti.f32, shape=2, needs_grad=True)
output_field = ti.field(ti.f32, shape=2, needs_grad=True)
multiplier[None] = 2.0


@ti.kernel
def forward_kernel(num: float):
    for i in input_field:
        output_field[i] = multiplier[None] * input_field[i] + num


if __name__ == "__main__":
    device = torch.device("cpu")
    tin_layer = EmptyTin(device=device) \
        .register_kernel(forward_kernel, 1.0) \
        .register_input_field(input_field, True) \
        .register_output_field(output_field, True) \
        .register_weight_field(multiplier, True, name="multiplier num") \
        .finish()
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
          f"multiplier grad = {multiplier.grad}"
          )
