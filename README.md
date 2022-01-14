# Stannum

![Gradient Tests](https://github.com/ifsheldon/stannum/actions/workflows/run_tests.yaml/badge.svg)

Fusing Taichi into PyTorch

**PRs are welcomed, please see TODOs and issues.**

## Usage

### `Tin` and `EmptyTin`

```python
from stannum import Tin
import torch

data_oriented = TiClass()  # some Taichi data-oriented class 
device = torch.device("cpu")
kernel_args = (1.0,)
tin_layer = Tin(data_oriented, device=device)
.register_kernel(data_oriented.forward_kernel, *kernel_args, kernel_name="forward")  # on old Taichi
# .register_kernel(data_oriented.forward_kernel, *kernel_args)  # on new Taichi
.register_input_field(data_oriented.input_field)
.register_output_field(data_oriented.output_field)
.register_internal_field(data_oriented.weight_field, name="field name")
.finish()  # finish() is required to finish construction
output = tin_layer(input_tensor)
```

It is **NOT** necessary to have a `@ti.data_oriented` class as long as you correctly register the fields that your
kernel needs for forward and backward calculation.

Please use `EmptyTin` in this case. Example:

```python
from stannum import EmptyTin
import torch
import taichi as ti

input_field = ti.field(ti.f32)
output_field = ti.field(ti.f32)
internal_field = ti.field(ti.f32)


@ti.kernel
def some_kernel():
    output_field[None] = input_field[None] + internal_field[None]


device = torch.device("cpu")
kernel_args = (1.0,)
tin_layer = EmptyTin(device)\
    .register_kernel(some_kernel, *kernel_args)\
    .register_input_field(input_field)\
    .register_output_field(output_field)\
    .register_internal_field(internal_field, name="field name")\
    .finish()  # finish() is required to finish construction
output = tin_layer(input_tensor)
```

For input and output:

* We can register multiple `input_field`, `output_field`, `weight_field`.
* At least one `input_field` and one `output_field` should be registered.
* The order of input tensors must match the registration order of `input_field`s.
* The output order will align with the registration order of `output_field`s.
* Kernel args must be acceptable by Taichi kernels and they will not get gradients.

### `Tube`
`Tube` is more flexible than `Tin` and slower in that it helps you create necessary fields and do automatic batching. 

#### Registrations
All you need to do is to register:
* Input/intermediate/output **tensor shapes** instead of fields
* At least one kernel that takes the following as arguments
  * Taichi fields: correspond to tensors (may or may not require gradients)
  * (Optional) Extra arguments: will NOT receive gradients

Acceptable dimensions of tensors to be registered:
* `None`: means the flexible batch dimension, must be the first dimension e.g. `(None, 2, 3, 4)`
* Positive integers: fixed dimensions with the indicated dimensionality
* Negative integers:
  * `-1`: means any number `[1, +inf)`, only usable in the registration of input tensors.
  * Negative integers < -1: indices of some dimensions that must be of the same dimensionality
    * Restriction: negative indices must be "declared" in the registration of input tensors first, then used in the registration of intermediate and output tensors. 
    * Example 1: tensor `a` and `b` of shapes `a: (2, -2, 3)` and `b: (-2, 5, 6)` mean the dimensions of `-2` must match.
    * Example 2: tensor `a` and `b` of shapes `a: (-1, 2, 3)` and `b: (-1, 5, 6)` mean no restrictions on the first dimensions.

Registration order:
Input tensors/intermediate fields/output tensors must be registered first, and then kernel.
```python
@ti.kernel
def ti_add(arr_a: ti.template(), arr_b: ti.template(), output_arr: ti.template()):
    for i in arr_a:
        output_arr[i] = arr_a[i] + arr_b[i]

ti.init(ti.cpu)
cpu = torch.device("cpu")
a = torch.ones(10)
b = torch.ones(10)
tube = Tube(cpu) \
    .register_input_tensor((10,), torch.float32, "arr_a", False) \
    .register_input_tensor((10,), torch.float32, "arr_b", False) \
    .register_output_tensor((10,), torch.float32, "output_arr", False) \
    .register_kernel(ti_add, ["arr_a", "arr_b", "output_arr"]) \
    .finish()
out = tube(a, b)
```
When registering a kernel, a list of field/tensor names is required, for example, the above `["arr_a", "arr_b", "output_arr"]`.
This list should correspond to the fields in the arguments of a kernel (e.g. above `ti_add()`).

The order of input tensors should match the input fields of a kernel.

#### Automatic batching
Automatic batching is done simply by running kernels `batch` times. The batch number is determined by the leading dimension of tensors of registered shape `(None, ...)`.

It's required that if any input tensors or intermediate fields are batched (which means they have registered the first dimension to be `None`), all output tensors must be registered as batched.

#### Examples
Simple one without negative indices or batch dimension:
```python
@ti.kernel
def ti_add(arr_a: ti.template(), arr_b: ti.template(), output_arr: ti.template()):
    for i in arr_a:
        output_arr[i] = arr_a[i] + arr_b[i]

ti.init(ti.cpu)
cpu = torch.device("cpu")
a = torch.ones(10)
b = torch.ones(10)
tube = Tube(cpu) \
    .register_input_tensor((10,), torch.float32, "arr_a", False) \
    .register_input_tensor((10,), torch.float32, "arr_b", False) \
    .register_output_tensor((10,), torch.float32, "output_arr", False) \
    .register_kernel(ti_add, ["arr_a", "arr_b", "output_arr"]) \
    .finish()
out = tube(a, b)
```

With negative dimension index:

```python
ti.init(ti.cpu)
cpu = torch.device("cpu")
tube = Tube(cpu) \
    .register_input_tensor((-2,), torch.float32, "arr_a", False) \
    .register_input_tensor((-2,), torch.float32, "arr_b", False) \
    .register_output_tensor((-2,), torch.float32, "output_arr", False) \
    .register_kernel(ti_add, ["arr_a", "arr_b", "output_arr"]) \
    .finish()
dim = 10
a = torch.ones(dim)
b = torch.ones(dim)
out = tube(a, b)
assert torch.allclose(out, torch.full((dim,), 2.))
dim = 100
a = torch.ones(dim)
b = torch.ones(dim)
out = tube(a, b)
assert torch.allclose(out, torch.full((dim,), 2.))
```

With batch dimension:
```python
@ti.kernel
def int_add(a: ti.template(), b: ti.template(), out: ti.template()):
    out[None] = a[None] + b[None]

ti.init(ti.cpu)
b = torch.tensor(1., requires_grad=True)
batched_a = torch.ones(10, requires_grad=True)
tube = Tube() \
    .register_input_tensor((None,), torch.float32, "a") \
    .register_input_tensor((), torch.float32, "b") \
    .register_output_tensor((None,), torch.float32, "out", True) \
    .register_kernel(int_add, ["a", "b", "out"]) \
    .finish()
out = tube(batched_a, b)
loss = out.sum()
loss.backward()
assert torch.allclose(torch.ones_like(batched_a) + 1, out)
assert b.grad == 10.
assert torch.allclose(torch.ones_like(batched_a), batched_a.grad)
```

For more invalid use examples, please see tests in `tests/test_tube`.

#### Advanced field construction with `FieldManager`
There is a way to tweak how fields are constructed in order to gain performance improvement in kernel calculations. 

By supplying a customized `FieldManager` when registering a field, you can construct a field however you want.

Please refer to the code `FieldManger` in `src/stannum/auxiliary.py` for more information.

If you don't know why constructing fields differently can improve performance, don't use this feature.

If you don't know how to construct fields differently, please refer to [Taichi field documentation](https://docs.taichi.graphics/lang/articles/advanced/layout).

### Complex Tensor Support

When registering input fields and output fields, you can pass `complex_dtype=True` to enable simple complex tensor input
and output support. For instance, `Tin(..).register_input_field(input_field, complex_dtype=True)`.

Now the complex tensor support is limited in that the representation of complex numbers is a barebone 2D vector, since
Taichi has no official support on complex numbers.

This means although `stannum` provides some facilities to deal with complex tensor input and output, you have to define
and do the operations on the proxy 2D vectors yourself.

In practice, we now have these limitations:

* The registered field with `complex_dtype=True` must be an appropriate `VectorField` or `ScalarField`
    * If it's `VectorField`, `n` should be `2`, like `v_field = ti.Vector.field(n=2, dtype=ti.f32, shape=(2, 3, 4, 5))`
    * If it's a `ScalarField`, the last dimension of it should be `2`,
      like `field = ti.field(ti.f32, shape=(2,3,4,5,2))`
    * The above examples accept tensors of `dtype=torch.cfloat, shape=(2,3,4,5)`
* The semantic of complex numbers is not preserved in kernels, so you are manipulating regular fields, and as a
  consequence, you need to implement complex number operators yourself
    * Example:
  ```python
  @ti.kernel
  def element_wise_complex_mul(self):
    for i in self.complex_array0:
        # this is not complex number multiplication, but only a 2D vector element-wise multiplication
        self.complex_output_array[i] = self.complex_array0[i] * self.complex_array1[i] 
  ```

## Installation & Dependencies

Install `stannum` with `pip` by

`python -m pip install stannum`

Make sure you have the following installed:

* PyTorch
* latest Taichi

## TODOs

### Documentation

* Documentation for users

### Features

* PyTorch-related:
    * PyTorch checkpoint and save model
    * Proxy `torch.nn.parameter.Parameter` for weight fields for optimizers
* Taichi related:
    * Wait for Taichi to have native PyTorch tensor view to optimize performance(i.e., no need to copy data back and
      forth)
    * Automatic Batching for `Tin` - waiting for upstream Taichi improvement
        * workaround for now: do static manual batching, that is to extend fields with one more dimension for batching

### Misc

* A nice logo
