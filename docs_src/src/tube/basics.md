# Tube

`Tube`, compared to `Tin`, helps you:

* create necessary fields
* manage fields
* (optionally) do automatic batching
* (optionally) create fields and tensors of which shapes are dynamically calculated. For example, the case in convolution.

So, `Tube` is more flexible and convenient, but it also introduces some overhead.

## Usage

All you need to do is to register:

* Input/intermediate/output **tensor shapes** instead of fields
* At least one kernel that takes the following as arguments
    * Taichi fields: correspond to tensors (may or may not require gradients)
    * (Optional) Extra arguments: will NOT receive gradients
        * see below [Set Kernel Arguments](basics.md#Set-Kernel-Extra-Arguments)
* (Optional) Tweak field creation, providing a `FieldManager`, see [Advanced Field Construction](advanced_field_construction.md)
* (Optional) Dynamically calculate the shape of fields and tensors, providing a `DimensionCalculator`, see [Dynamic Dimension Calculation](dynamic_dimensions.md)

## Requirements 

Registration order: Input tensors/intermediate fields/output tensors must be registered first, and then kernel.

When registering a kernel, a list of field/tensor names is required, for example, the above `["arr_a", "arr_b", "output_arr"]`.

This list should correspond to the fields in the arguments of a kernel (e.g., below `ti_add()`).

The order of input tensors should match the input fields of a kernel.

A valid example is shown below:

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

Acceptable dimensions of tensors to be registered are:

* `stannum.BatchDim`: means the flexible batch dimension, must be the first dimension e.g. `(BatchDim, 2, 3, 4)`
* Positive integers: fixed dimensions with the indicated dimensionality
* `stannum.AnyDim`: means any number `[1, +inf)`, **only usable** in the registration of input tensors.
* `stannum.MatchDim(dim_id: str | int)`: means some dimensions with the same `dim_id` must be of the same dimensionality
    * Restriction: `MatchDim` must be "declared" in the registration of input tensors first, then used in the registration of intermediate and output tensors. 
    * Example: tensor `a` and `b` of shapes `a: (2, MatchDim("some_dim"), 3)` and `b: (MatchDim("some_dim"), 5, 6)` mean the dimensions of `some_dim` must match, that is, the second dimension of `a` must match the first dimension of `b`.


## Automatic Batching

Automatic batching is done simply by running kernels `batch` times. The batch number is determined by the leading dimension of tensors of registered shape `(None, ...)`.

It's required that if any input tensors are batched (which means they have registered the first dimension to be `None`), all intermediate fields and output tensors must be registered as batched.

## More Examples

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

With dimension matching:

```python
ti.init(ti.cpu)
cpu = torch.device("cpu")
tube = Tube(cpu) \
.register_input_tensor((MatchDim(0),), torch.float32, "arr_a", False) \
.register_input_tensor((MatchDim(0),), torch.float32, "arr_b", False) \
.register_output_tensor((MatchDim(0),), torch.float32, "output_arr", False) \
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
.register_input_tensor((BatchDim,), torch.float32, "a") \
.register_input_tensor((), torch.float32, "b") \
.register_output_tensor((BatchDim,), torch.float32, "out", True) \
.register_kernel(int_add, ["a", "b", "out"]) \
.finish()
out = tube(batched_a, b)
loss = out.sum()
loss.backward()
assert allclose(torch.ones_like(batched_a) + 1, out)
assert b.grad == 10.
assert allclose(torch.ones_like(batched_a), batched_a.grad)
```

For more valid and invalid use examples, please see [test files](../../../tests/test_tube) in the test folder.

## APIs

### Constructor

```python
def __init__(self,
             device: Optional[torch.device] = None,
             persistent_field: bool = True,
             enable_backward: bool = True):
    """
        Init a tube

        @param device: Optional, torch.device tensors are on, if it's None, the device is determined by input tensors
        @param persistent_field: whether or not to save fields during forward pass.
        If True, created fields will not be destroyed until compute graph is cleaned,
        otherwise they will be destroyed right after forward pass is done and re-created in backward pass.
        Having two modes is due to Taichi's performance issue, see https://github.com/taichi-dev/taichi/pull/4356
        @param enable_backward: whether or not to enable backward gradient computation, disable it will have performance
        improvement in forward pass, but attempting to do backward computation will cause runtime error.
        """
```

### Registrations

Register input tensor shapes:

```python
def register_input_tensor(self,
                          dims: Tuple[DimOption] | List[DimOption],
                          dtype: torch.dtype,
                          name: str,
                          requires_grad: Optional[bool] = None,
                          field_manager: Optional[FieldManager] = None):
    """
        Register an input tensor

        @param dims: dims can contain `None`, positive and negative numbers,
        for restrictions and requirements, see README
        @param dtype: torch data type
        @param name: name of the tensor and corresponding field
        @param requires_grad: optional, if it's None, it will be determined by input tensor
        @param field_manager: customized field manager, if it's None, a DefaultFieldManger will be used
        """

```

Register intermediate field shapes:

```python
def register_intermediate_field(self,
                                dims_or_calc: Tuple[DimOption] | List[DimOption] | Callable | DimensionCalculator,
                                ti_dtype: TiDataType,
                                name: str,
                                needs_grad: bool,
                                field_manager: Optional[FieldManager] = None):
    """
        Register an intermediate field,
        which can be useful if multiple kernels are used and intermediate results between kernels are stored

        @param dims_or_calc: dims can contain `None`, positive and negative numbers,
        for restrictions and requirements, see README; Or DimensionCalculator instance or a function can be passed
        to dynamically calculate dimensions
        @param ti_dtype: taichi data type
        @param name: name of the field
        @param needs_grad: if the field needs gradients.
        @param dims: dims can contain `None`, positive and negative numbers,
        for restrictions and requirements, see README
        @param dim_calc: DimensionCalculator instance or a function
        @param field_manager: customized field manager, if it's None, a DefaultFieldManger will be used
        """
```

Register output tensor shapes:

```python
def register_output_tensor(self,
                           dims_or_calc: Tuple[DimOption] | List[DimOption] | Callable | DimensionCalculator,
                           dtype: torch.dtype,
                           name: str,
                           requires_grad: bool,
                           field_manager: Optional[FieldManager] = None):
    """
        Register an output tensor

        @param dims_or_calc: dims can contain `None`, positive and negative numbers,
        for restrictions and requirements, see README; Or DimensionCalculator instance or a function can be passed
        to dynamically calculate dimensions
        @param dtype: torch data type
        @param name: name of the tensor and corresponding field
        @param dims: dims can contain `None`, positive and negative numbers,
        for restrictions and requirements, see README
        @param dim_calc: DimensionCalculator instance or a function
        @param requires_grad: if the output requires gradients
        @param field_manager: customized field manager, if it's None, a DefaultFieldManger will be used
        """
```

Register kernels:

```python
def register_kernel(self, kernel: Callable, tensor_names: List[str], *extra_args: Any, name: Optional[str] = None):
    """
        Register a Taichi kernel

        @param kernel: Taichi kernel. For requirements, see README
        @param tensor_names: the names of registered tensors that are to be used in this kernel
        @param extra_args: any extra arguments passed to the kernel
        @param name: name of this kernel, if it's None, it will be kernel.__name__
        """
```

### Set Kernel Extra Arguments

Kernels may need extra arguments that do not need gradients, then you can set extra arguments with `Tube.set_kernel_extra_args()` or set extra arguments in `Tube.register_kernel()`

```python
def set_kernel_extra_args(self, kernel: Callable | str, *extra_args: Any):
    """
        Set args for a kernel
        @param kernel: kernel function or its name
        @param extra_args: extra kernel arguments
        """
```

One example kernel is shown below, in which `multiplier` is an extra kernel argument.

```python
@ti.kernel
def mul(arr: ti.template(), out: ti.template(), multiplier: float):
    for i in arr:
        out[i] = arr[i] * multiplier
```
