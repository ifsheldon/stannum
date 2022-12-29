# Tin

`Tin` and `EmptyTin` are thinest layers that bridge Taichi and PyTorch. 

## Usage

To use them, you will need:

*  a Taichi kernel or a Taichi data-oriented-class instance
* Taichi fields
* and registration as shown below

```python
from stannum import Tin
import torch

data_oriented = TiClass()  # some Taichi data-oriented class 
device = torch.device("cpu")
kernel_args = (1.0,)
tin_layer = Tin(data_oriented, device=device, auto_clear_grad=True)
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

Please use `EmptyTin` in this case, for example:

```python
from stannum import EmptyTin
import torch
import taichi as ti

input_field = ti.field(ti.f32)
output_field = ti.field(ti.f32)
internal_field = ti.field(ti.f32)


@ti.kernel
def some_kernel(bias: float):
    output_field[None] = input_field[None] + internal_field[None] + bias


device = torch.device("cpu")
kernel_args = (1.0,)
tin_layer = EmptyTin(device, True)\
    .register_kernel(some_kernel, *kernel_args)\
    .register_input_field(input_field)\
    .register_output_field(output_field)\
    .register_internal_field(internal_field, name="field name")\
    .finish()  # finish() is required to finish construction
output = tin_layer(input_tensor)
```

## Restrictions & Warning

For input and output, the restrictions are:

* We can register multiple `input_field`, `output_field`, `internal_field`.
* At least one `input_field` and one `output_field` should be registered.
* The order of input tensors must match the registration order of `input_field`s.
* The output order will align with the registration order of `output_field`s.
* Kernel args must be acceptable by Taichi kernels and they will not get gradients.

Be warned that it is **YOUR** responsibility to create and manage fields and if you don’t manage fields properly, memory leaks and read-after-free of fields can happen.

## APIs

### Constructors

`Tin`:

```python
def __init__(self,
             data_oriented: Any,
             device: torch.device,
             auto_clear_grad: bool,
             _auto_clear: bool = need_auto_clearing_fields):
    """
    Init a Tin instance
    @param data_oriented: @ti.data_oriented class instance
    @param device: torch.device instance
    @param auto_clear_grad: auto clear gradients in fields before backward computation
    @param _auto_clear: clear fields before use
    """
```

`EmptyTin`:

```python
def __init__(self,
             device: torch.device,
             auto_clear_grad: bool,
             _auto_clear: bool = need_auto_clearing_fields):
    """
    Init an EmptyTin instance

    @param device: torch.device instance
    @param auto_clear_grad: auto clear gradients in fields before backward computation
    @param _auto_clear: clear fields before use, for legacy Taichi
    """
```

If `_auto_clear` is `True`, then all the registered fields will be cleared before running the kernel(s), which prevents some undefined behaviors due to un-initialized memory of fields before Taichi `0.9.1`. After Taichi `0.9.1`, the memory of fields is automatically cleared after creation, so `auto_clear` is not necessary anymore. But it is still configurable if desired.

### Registrations

#### Register Kernels

`EmptyTin`:

```python
def register_kernel(self, kernel: Callable, *kernel_args: Any, kernel_name: Optional[str] = None):
    """
        Register a kernel for forward calculation

        @param kernel: Taichi kernel
        @param kernel_args: arguments for the kernel
        @param kernel_name: kernel name, optional for new Taichi, compulsory for old Taichi
        @return: self
        """
```

`Tin`:

```python
def register_kernel(self, kernel: Union[Callable, str], *kernel_args: Any, kernel_name: Optional[str] = None):
    """
        Register a kernel for forward calculation
        @param kernel: kernel function or kernel name
        @param kernel_args: args for the kernel, optional
        @param kernel_name: kernel name, optional for new Taichi, compulsory for old Taichi
        @return: self
        """
```

### Register Fields

Register input fields:

```python
def register_input_field(self, field: Union[ScalarField, MatrixField],
                         name: Optional[str] = None,
                         needs_grad: Optional[bool] = None,
                         complex_dtype: bool = False):
    """
        Register an input field which requires a tensor input in the forward calculation

        @param field: Taichi field
        @param name: name of this field, default: "input_field_ith"
        @param needs_grad: whether the field needs grad, `None` for automatic configuration
        @param complex_dtype: whether the input tensor that is going to be filled into this field is complex numbers
        @return: self
        """
```

Register internal fields that are used to store intermediate values if multiple kernels are used:

```python
def register_internal_field(self, field: Union[ScalarField, MatrixField],
                            needs_grad: Optional[bool] = None,
                            name: Optional[str] = None,
                            value: Optional[torch.Tensor] = None,
                            complex_dtype: bool = False):
    """
        Register a field that serves as weights internally and whose values are required by the kernel function

        @param field: Taichi field
        @param needs_grad: whether the field needs grad, `None` for automatic configuration
        @param name: name for the field, facilitating later value setting, `None` for default number naming
        @param value: optional initial values from a tensor
        @param complex_dtype: whether the input tensor that is going to be filled into this field is complex numbers
        @return: self
        """
```

Register output fields:

```python
def register_output_field(self, field: Union[ScalarField, MatrixField],
                          name: Optional[str] = None,
                          needs_grad: Optional[bool] = None,
                          complex_dtype: bool = False):
    """
        Register an output field that backs an output tensor in the forward calculation

        @param field: Taichi field
        @param name: name of this field, default: "output_field_ith"
        @param needs_grad: whether the field needs grad, `None` for automatic configuration
        @param complex_dtype: whether the input tensor that is going to be filled into this field is complex numbers
        @return: self
        """
```



### Setters

#### Internal fields

The values of internal fields can be set by:

```python
def set_internal_field(self, field_name: Union[str, int], tensor: torch.Tensor):
    """
        Sets the value of an internal field from a tensor

        @param field_name: integer(when using default number naming) or string name
        @param tensor: values for the field
        @return: None
        """
```

#### Kernel Arguments

Kernels may need arguments that do not need gradients, then you can set extra arguments with `Tin/EmptyTin.set_kernelargs()` or set extra arguments in `Tin/EmptyTin.register_kernel()`

```python
def set_kernel_args(self, kernel: Union[Callable, str], *kernel_args: Any):
    """
        Set args for a kernel
        @param kernel: kernel function or its name
        @param kernel_args: kernel arguments
        """
```

One example is shown below. Note that the kernel has already contains references to fields, which differs from the case in `Tube`.

```python
input_field = ti.field(ti.f32)
output_field = ti.field(ti.f32)
internal_field = ti.field(ti.f32)


@ti.kernel
def some_kernel(adder: float):
    output_field[None] = input_field[None] + internal_field[None] + adder
```

