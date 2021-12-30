# Stannum
![Gradient Tests](https://github.com/ifsheldon/stannum/actions/workflows/run_tests.yaml/badge.svg)

PyTorch wrapper for Taichi data-oriented class

**PRs are welcomed, please see TODOs.**

## Usage

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
    .finish() # finish() is required to finish construction
output = tin_layer(input_tensor)
```
### Complex Tensor Support
When registering input fields and output fields, you can pass `complex_dtype=True` to enable simple complex tensor input and output support. For instance, `Tin(..).register_input_field(input_field, complex_dtype=True)`.

Now the complex tensor support is limited in that the representation of complex numbers is a barebone 2D vector, since Taichi has no official support on complex numbers.

This means although `stannum` provides some facilities to deal with complex tensor input and output, you have to define and do the operations on the proxy 2D vectors yourself.

In practice, we now have these limitations:
* The registered field with `complex_dtype=True` must be an appropriate `VectorField` or `ScalarField`
  * If it's `VectorField`, `n` should be `2`, like `v_field = ti.Vector.field(n=2, dtype=ti.f32, shape=(2, 3, 4, 5))`
  * If it's a `ScalarField`, the last dimension of it should be `2`, like `field = ti.field(ti.f32, shape=(2,3,4,5,2))`
  * The above examples accept tensors of `dtype=torch.cfloat, shape=(2,3,4,5)`
* The semantic of complex numbers is not preserved in kernels, so you are manipulating regular fields, and as a consequence, you need to implement complex number operators yourself
  * Example:
  ```python
  @ti.kernel
  def element_wise_complex_mul(self):
    for i in self.complex_array0:
        # this is not complex number multiplication, but only a 2D vector element-wise multiplication
        self.complex_output_array[i] = self.complex_array0[i] * self.complex_array1[i] 
  ```
### Note: 

It is **NOT** necessary to have a `@ti.data_oriented` class as long as you correctly register the fields that your kernel needs for forward and backward calculation. Please use `EmptyTin` in this case.

For input and output:

* We can register multiple `input_field`, `output_field`, `weight_field`.
* At least one `input_field` and one `output_field` should be registered.
* The order of input tensors must match the registration order of `input_field`s.
* The output order will align with the registration order of `output_field`s.

## Installation & Dependencies
Install `stannum` with `pip` by 

`python -m pip install stannum`

Make sure you have the following installed:
* PyTorch
* Taichi

## TODOs

### Documentation

* Documentation for users

### Features
* PyTorch-related:
  * PyTorch checkpoint and save model
  * Proxy `torch.nn.parameter.Parameter` for weight fields for optimizers
* Taichi related:
  * Wait for Taichi to have native PyTorch tensor view to optimize performance(i.e., no need to copy data back and forth)
  * Automatic Batching - waiting for upstream Taichi improvement
    * workaround for now: do static manual batching, that is to extend fields with one more dimension for batching

### Misc

* A nice logo
