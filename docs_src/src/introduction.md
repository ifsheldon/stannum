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
    * For performance concerns, we strongly recommend to use Taichi>=0.9.1
    * If possible always use latest stable Taichi until the tracking issue #9 is fully resolved by Taichi developers.

## Differentiability

Stannum does **NOT** check the differentiability of your kernels, so you may not get correct gradients if your kernel is not differentiable. Please refer to [Differentiable Programming of Taichi](https://docs.taichi-lang.org/docs/differentiable_programming) for more information.

## `Tin` or `Tube`?

`stannum` mainly has two high-level APIs, `Tin` and `Tube`. `Tin` aims to be the thinnest bridge layer with the least overhead while `Tube` has more functionalities and convenience with some more overhead.

See the comparison below:

|                                 |         `Tin`/`EmptyTin`         |                            `Tube`                            |
| :-----------------------------: | :------------------------------: | :----------------------------------------------------------: |
|            Overhead             |               Low❤️               | Too many invocations in one forward pass will incur perf loss (see [issue #9](https://github.com/ifsheldon/stannum/issues/9))⚠️ |
|        Field Management         | Users must manage Taichi fields⚠️ |                       Auto management♻️                       |
|      Forward Pass Bridging      |                ✅                 |                              ✅                               |
| Backward Pass Gradient Bridging |                ✅                 |                              ✅                               |
|            Batching             |                ❌                 |                              ✅                               |
|     Variable Tensor Shapes      |                ❌                 |                              ✅                               |

## Bugs & Issues

Please feel free to file issues on [Github](https://github.com/ifsheldon/stannum). If a runtime error occurs from the dependencies of `stannum`, you may also want to check the [upstream breaking change tracker](https://github.com/ifsheldon/stannum/issues/11).
