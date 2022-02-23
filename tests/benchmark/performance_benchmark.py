import taichi as ti
import torch
from src.stannum.tube import Tube
from tqdm import tqdm
from time import time


@ti.kernel
def mul(arr: ti.template(), out: ti.template()):
    for i in arr:
        out[i] = arr[i] * 2.0


def persistent_vs_eager_mode():
    ti.init(ti.cpu)
    eager_tube = Tube(persistent_field=False) \
        .register_input_tensor((2,), torch.float32, "arr") \
        .register_output_tensor((2,), torch.float32, "out", True) \
        .register_kernel(mul, ["arr", "out"]) \
        .finish()
    repeats = 100
    start = time()
    a = torch.ones(2, requires_grad=True)
    losses = []
    for _ in tqdm(range(repeats)):
        out = eager_tube(a)
        l = out.sum()
        losses.append(l)
    loss = torch.stack(losses).sum()
    loss.backward()
    end = time()
    assert torch.allclose(a.grad, torch.ones_like(a) * 2 * repeats)
    print(f"{repeats=}, took {end - start} seconds, avg = {(end - start) / repeats}")

    ti.reset()
    ti.init(ti.cpu)
    persistent_tube = Tube(persistent_field=True) \
        .register_input_tensor((2,), torch.float32, "arr") \
        .register_output_tensor((2,), torch.float32, "out", True) \
        .register_kernel(mul, ["arr", "out"]) \
        .finish()

    start = time()
    a = torch.ones(2, requires_grad=True)
    losses = []
    for _ in tqdm(range(repeats)):
        out = persistent_tube(a)
        l = out.sum()
        losses.append(l)
    loss = torch.stack(losses).sum()
    loss.backward()
    end = time()
    assert torch.allclose(a.grad, torch.ones_like(a) * 2 * repeats)
    print(f"{repeats=}, took {end - start} seconds, avg = {(end - start) / repeats}")


if __name__ == "__main__":
    persistent_vs_eager_mode()
