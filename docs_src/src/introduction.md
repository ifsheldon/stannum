# Stannum

<div align="center">
  <img width="300px" src="https://github.com/ifsheldon/stannum/raw/main/logo.PNG"/>
</div>

**Fusing Taichi into PyTorch**

## Why Stannum?

In differentiable rendering including neural rendering, rendering algorithms are transferred to the field of computer vision, but some rendering operations (e.g., ray tracing and direct volume rendering) are not easy to be expressed in tensor operations but in kernels. Differentiable kernels of Taichi enables fast, efficient and differentiable implementation of rendering algorithms while tensor operators provides math expressiveness. 

Stannum bridges Taichi and PyTorch to have advantage of both kernel-based and operator-based parallelism.

## Installation

Install `stannum` with `pip` by

`python -m pip install stannum`

Make sure you have the following installed:

* PyTorch
* **latest** Taichi
    * For performance concerns, we strongly recommend to use Taichi >= 1.1.3 (see Issue #9 for more information)

## Differentiability

Stannum does **NOT** check the differentiability of your kernels, so you may not get correct gradients if your kernel is not differentiable. Please refer to [Differentiable Programming of Taichi](https://docs.taichi-lang.org/docs/differentiable_programming) for more information.

## `Tin` or `Tube`?

`stannum` mainly has two high-level APIs, `Tin` and `Tube`. `Tin` aims to be the thinnest bridge layer with the least overhead while `Tube` has more functionalities and convenience with some more overhead.

See the comparison below:

|                                 |         `Tin`/`EmptyTin`         |                      `Tube`                       |
| :-----------------------------: | :------------------------------: | :-----------------------------------------------: |
|          Overhead[^1] [^2]           |               Low❤️               | A bit more overhead due to auto memory management |
|        Field Management         | Users must manage Taichi fields⚠️ |                 Auto management♻️                  |
|      Forward Pass Bridging      |                ✅                 |                         ✅                         |
| Backward Pass Gradient Bridging |                ✅                 |                         ✅                         |
|            Batching             |                ❌                 |                         ✅                         |
|     Variable Tensor Shapes      |                ❌                 |                         ✅                         |

[^1]: (Performance Tip) A lot of assertions in `stannum` make sure you do the right thing or get a right error when you do it wrong, which is helpful in debugging but incurs a bit overhead. To get rid of assertion overhead, pass `-O` to Python as suggested in [the Python doc about assertions](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement).

[^2]: See Issue #9 for more information about the performance if you want to use `Tube` with legacy Taichi < `1.1.3`.

## Bugs & Issues

Please feel free to file issues on [Github](https://github.com/ifsheldon/stannum). If a runtime error occurs from the dependencies of `stannum`, you may also want to check the [upstream breaking change tracker](https://github.com/ifsheldon/stannum/issues/11).
