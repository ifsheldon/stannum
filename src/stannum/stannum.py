import torch
from .utils import check_field_needs_grad, autofill_kernel_name_available, is_kernel, eprint
from typing import Optional, List, Dict, Union, Callable, Tuple, Any
from taichi.lang.matrix import MatrixField
from taichi.lang.field import ScalarField


class TaichiKernelBundle:
    def __init__(self, kernel: Callable, kernel_name: str, *args):
        self.kernel: Callable = kernel
        self.name: str = kernel.__name__ if kernel_name is None else kernel_name
        self.args: Tuple[Any, ...] = args

    def forward(self):
        self.kernel(*self.args)

    def backward(self):
        self.kernel.grad(*self.args)


class TaichiField:
    """An extensive wrapper around Taichi field"""

    def __init__(self, field: Union[ScalarField, MatrixField],
                 needs_grad: bool,
                 name: str,
                 complex_dtype: bool = False):
        if isinstance(field, ScalarField):
            if complex_dtype:
                assert field.shape[-1] == 2, \
                    f"Field {name}: ScalarField needs to have its last dimension to be 2 to hold complex values"
                self.acceptable_tensor_shape = tuple(field.shape[:-1])
            else:
                self.acceptable_tensor_shape = tuple(field.shape)
        elif isinstance(field, MatrixField):
            if complex_dtype:
                assert field.n == 2 and field.m == 1, \
                    f"Field {name}: MatrixField needs to have its matrix dimension to be (2, 1) to hold complex values, " \
                    f"got {(field.n, field.m)}"
                self.acceptable_tensor_shape = tuple(field.shape)
            else:
                if field.m == 1:
                    self.acceptable_tensor_shape = tuple(field.shape) + (field.n,)
                else:
                    self.acceptable_tensor_shape = tuple(field.shape) + (field.n, field.m)
        else:
            raise Exception(f"Field {name}: Only accept ti ScalarField or MatrixField, got {type(field)}")
        self.field: Union[ScalarField, MatrixField] = field
        self.grad: Union[ScalarField, MatrixField] = field.grad
        self.needs_grad: bool = needs_grad
        # TODO: wait for upstream support on complex numbers
        self.complex_dtype: Optional[bool] = complex_dtype
        self.name: str = name

    def check_tensor_acceptable(self, tensor: torch.Tensor):
        tensor_shape = tuple(tensor.shape)
        if self.acceptable_tensor_shape != tensor_shape:
            if self.complex_dtype:
                raise Exception(
                    f"Field {self.name}: Expecting a complex tensor of shape {self.acceptable_tensor_shape}, "
                    f"got {tensor_shape}")
            else:
                raise Exception(f"Field {self.name}: Expecting a real tensor of shape {self.acceptable_tensor_shape}, "
                                f"got {tensor_shape}")
        if self.complex_dtype and tensor.dtype != torch.cfloat and tensor.dtype != torch.cdouble:
            raise Exception(f"Field {self.name}: Expecting a complex tensor, got dtype = {tensor.dtype}")

    def from_torch(self, tensor: torch.Tensor):
        self.check_tensor_acceptable(tensor)
        if self.complex_dtype:
            tensor = torch.view_as_real(tensor)
        self.field.from_torch(tensor)

    def grad_from_torch(self, tensor: torch.Tensor):
        self.check_tensor_acceptable(tensor)
        if self.complex_dtype:
            tensor = torch.view_as_real(tensor)
        self.grad.from_torch(tensor)

    def to_torch(self, device: Optional[torch.device] = None):
        if device is not None:
            tensor = self.field.to_torch(device)
        else:
            tensor = self.field.to_torch()
        if self.complex_dtype:
            tensor = torch.view_as_complex(tensor)
        return tensor

    def grad_to_torch(self, device: Optional[torch.device] = None):
        if device is not None:
            tensor = self.grad.to_torch(device)
        else:
            tensor = self.grad.to_torch()
        if self.complex_dtype:
            tensor = torch.view_as_complex(tensor)
        return tensor


class TinConfigs:
    """
    A "struct" for storing objects needed in TinFunc
    """

    def __init__(self,
                 ti_kernel_bundles: List[TaichiKernelBundle],
                 input_fields: List[TaichiField],
                 weight_fields: List[TaichiField],
                 output_fields: List[TaichiField],
                 device: torch.device):
        self.kernel_bundles: List[TaichiKernelBundle] = ti_kernel_bundles
        self.input_fields: List[TaichiField] = input_fields
        self.internal_fields: List[TaichiField] = weight_fields
        self.output_fields: List[TaichiField] = output_fields
        self.device: torch.device = device


class TinFunc(torch.autograd.Function):
    """Customized autograd function used in Tin layers"""

    @staticmethod
    def forward(ctx, tin_configs: TinConfigs, *input_tensors: torch.Tensor):
        ctx.tin_configs = tin_configs
        assert len(input_tensors) == len(tin_configs.input_fields)
        for input_tensor, field in zip(input_tensors, tin_configs.input_fields):
            field.from_torch(input_tensor)
        for kernel_bundle in tin_configs.kernel_bundles:
            kernel_bundle.forward()
        output_tensors = []
        non_grad_tensors = []
        for output_field in tin_configs.output_fields:
            output_tensor = output_field.to_torch(device=tin_configs.device).requires_grad_(output_field.needs_grad)
            if not output_field.needs_grad:
                non_grad_tensors.append(output_tensor)
            output_tensors.append(output_tensor)

        ctx.mark_non_differentiable(*non_grad_tensors)

        if len(output_tensors) > 1:
            return tuple(output_tensors)
        else:
            return output_tensors[0]

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        tin_configs = ctx.tin_configs
        for grad_output, output_field in zip(grad_outputs, tin_configs.output_fields):
            if output_field.needs_grad:
                output_field.grad_from_torch(grad_output)
        for kernel_bundle in reversed(tin_configs.kernel_bundles):
            kernel_bundle.backward()
        gradient_tensors = [None]
        for input_field in tin_configs.input_fields:
            if input_field.needs_grad:
                gradient_tensors.append(input_field.grad_to_torch(device=tin_configs.device))
            else:
                gradient_tensors.append(None)

        if any(map(lambda x: x.needs_grad, tin_configs.internal_fields)):
            eprint("Some internal fields require gradients.\n"
                   "Although they got gradients during back propagation in the grad field,\n"
                   "values of them will NOT be updated automatically")
        return tuple(gradient_tensors)


class EmptyTin(torch.nn.Module):
    """A Taichi field wrapper that requires no @ti.data_oriented class"""

    def __init__(self, device: torch.device):
        """
        Init an EmptyTin instance
        :param device: torch.device instance
        """
        super().__init__()
        self.input_fields: List[TaichiField] = []
        self.internal_fields: Dict[str, TaichiField] = {}
        self.output_fields: List[TaichiField] = []
        assert isinstance(device, torch.device), "device must be an instance of torch.device"
        self.device: torch.device = device
        self.tin_configs: TinConfigs = None
        self.kernel_bundles: List[TaichiKernelBundle] = []
        self.kernel_bundle_dict: Dict[str, TaichiKernelBundle] = {}
        self.finished: bool = False

    def register_input_field(self, field: Union[ScalarField, MatrixField],
                             name: Optional[str] = None,
                             needs_grad: Optional[bool] = None,
                             complex_dtype: bool = False):
        """
        Register an input field which requires a tensor input in the forward calculation

        :param field: Taichi field
        :param name: name of this field, default: "input_field_ith"
        :param needs_grad: whether the field needs grad, `None` for automatic configuration
        :param complex_dtype: whether the input tensor that is going to be filled into this field is complex numbers
        :return: self
        """
        assert not self.finished, "Registration after .finish()"
        needs_grad = check_field_needs_grad(field, needs_grad)
        field_name = name if name is not None else "input_field_" + str(len(self.output_fields))
        self.input_fields.append(TaichiField(field, needs_grad, field_name, complex_dtype))
        return self

    def register_output_field(self, field: Union[ScalarField, MatrixField],
                              name: Optional[str] = None,
                              needs_grad: Optional[bool] = None,
                              complex_dtype: bool = False):
        """
        Register an output field that backs an output tensor in the forward calculation
        :param field: Taichi field
        :param name: name of this field, default: "output_field_ith"
        :param needs_grad: whether the field needs grad, `None` for automatic configuration
        :param complex_dtype: whether the input tensor that is going to be filled into this field is complex numbers
        :return: self
        """
        assert not self.finished, "Registration after .finish()"
        needs_grad = check_field_needs_grad(field, needs_grad)
        field_name = name if name is not None else "output_field_" + str(len(self.output_fields))
        self.output_fields.append(TaichiField(field, needs_grad, field_name, complex_dtype))
        return self

    def register_internal_field(self, field: Union[ScalarField, MatrixField],
                                needs_grad: Optional[bool] = None,
                                name: Optional[str] = None,
                                value: Optional[torch.Tensor] = None,
                                complex_dtype: bool = False):
        """
        Register a field that serves as weights internally and whose values are required by the kernel function
        :param field: Taichi field
        :param needs_grad: whether the field needs grad, `None` for automatic configuration
        :param name: name for the field, facilitating later value setting, `None` for default number naming
        :param value: optional initial values from a tensor
        :param complex_dtype: whether the input tensor that is going to be filled into this field is complex numbers
        :return: self
        """
        assert not self.finished, "Registration after .finish()"
        field_name = name if name is not None else str(len(self.internal_fields))
        needs_grad = check_field_needs_grad(field, needs_grad)
        if value is not None:
            field.from_torch(value)
        self.internal_fields[field_name] = TaichiField(field, needs_grad, field_name, complex_dtype)
        return self

    def register_kernel(self, kernel: Callable, *kernel_args: Any, kernel_name: Optional[str] = None):
        """
        Register a kernel for forward calculation
        :param kernel: Taichi kernel
        :param kernel_args: arguments for the kernel
        :param kernel_name: kernel name, optional for new Taichi, compulsory for old Taichi
        :return: self
        """
        assert not self.finished, "Registration after .finish()"
        assert kernel is not None, "Kernel must not be None"
        assert autofill_kernel_name_available(kernel) or kernel_name is not None, \
            "kernel has no __name__, please update your Taichi or specify its name"
        assert not isinstance(kernel, str), "Please pass the kernel function, not its name"
        assert is_kernel(kernel), "Passed function is not a Taichi kernel"
        kernel_bundle = TaichiKernelBundle(kernel, kernel_name, *kernel_args)
        assert kernel_bundle.name not in self.kernel_bundle_dict, f"Kernel name {kernel_bundle.name} already registered"
        self.kernel_bundles.append(kernel_bundle)
        self.kernel_bundle_dict[kernel_bundle.name] = kernel_bundle
        return self

    def set_internal_field(self, field_name: Union[str, int], tensor: torch.Tensor):
        """
        Sets the value of an internal field from a tensor
        :param field_name: integer(when using default number naming) or string name
        :param tensor: values for the field
        :return: None
        """
        assert self.finished, "Fields for weights can only be set after finishing registrations"
        if isinstance(field_name, int):
            field_name = str(field_name)
        assert field_name in self.internal_fields, f"field with the name {field_name} not registered"
        self.internal_fields[field_name].from_torch(tensor)

    def set_kernel_args(self, kernel: Union[Callable, str], *kernel_args: Any):
        """
        Set args for a kernel
        @param kernel: kernel function or its name
        @param kernel_args: kernel arguments
        """
        if isinstance(kernel, str):
            kernel_name = kernel
        else:
            kernel_name = kernel.__name__
        assert kernel_name in self.kernel_bundle_dict, "Kernel not found, please register it first"
        self.kernel_bundle_dict[kernel_name].args = kernel_args

    def finish(self):
        """
        Finish all configurations and initializations
        :return: self
        """
        assert len(self.input_fields) > 0, "Must register at least 1 input field"
        assert len(self.output_fields) > 0, "Must register at least 1 output field"
        assert len(self.kernel_bundles) > 0, "Must register at least 1 kernel"
        self.tin_configs = TinConfigs(self.kernel_bundles,
                                      self.input_fields,
                                      list(self.internal_fields.values()),
                                      self.output_fields,
                                      self.device)
        self.finished = True
        return self

    def forward(self, *input_tensors: torch.Tensor):
        assert self.finished, "Please finish registrations by calling .finish() before using this layer"
        return TinFunc.apply(self.tin_configs, *input_tensors)


class Tin(EmptyTin):
    """A Taichi field wrapper that requires a @ti.data_oriented class for registering a kernel by name"""

    def __init__(self, data_oriented: Any, device: torch.device):
        """
        Init a Tin instance
        :param data_oriented: @ti.data_oriented class instance
        :param device: torch.device instance
        """
        super(Tin, self).__init__(device=device)
        if not hasattr(data_oriented, "_data_oriented"):
            raise Exception("Requires a Taichi data-oriented instance")
        self.data_oriented = data_oriented

    def register_kernel(self, kernel: Union[Callable, str], *kernel_args: Any, kernel_name: Optional[str] = None):
        """
        Register a kernel for forward calculation
        @param kernel: kernel function or kernel name
        @param kernel_args: args for the kernel, optional
        @param kernel_name: kernel name, optional for new Taichi, compulsory for old Taichi
        @return: self
        """
        assert kernel is not None, "Kernel must not be None"
        if isinstance(kernel, str):
            kernel_name = kernel
            kernel = getattr(self.data_oriented, kernel)
            assert kernel is not None, f"Cannot find the kernel with the name {kernel_name}"
        assert autofill_kernel_name_available(kernel) or kernel_name is not None
        super(Tin, self).register_kernel(kernel, *kernel_args, kernel_name=kernel_name)
        return self
