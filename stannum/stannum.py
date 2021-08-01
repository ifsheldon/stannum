import torch
from enum import Enum
from .utils import check_field_needs_grad, autofill_kernel_name_available


class FieldType(Enum):
    INPUT = 0
    OUTPUT = 1
    WEIGHTS = 2


class TinConfigs:
    """
    A "struct" for storing objects needed in TinFunc
    """

    def __init__(self,
                 ti_kernel_bundles,
                 input_fields,
                 weight_fields,
                 output_fields,
                 device):
        self.kernel_bundles = ti_kernel_bundles
        self.input_fields: [TaichiField] = input_fields
        self.weight_fields: [TaichiField] = weight_fields
        self.output_fields: [TaichiField] = output_fields
        self.device: torch.device = device


class TaichiKernelBundle:
    def __init__(self, kernel, kernel_name, *args):
        self.kernel = kernel
        self.name = kernel.__name__ if kernel_name is None else kernel_name
        self.args = args

    def forward(self):
        self.kernel(*self.args)

    def backward(self):
        self.kernel.grad(*self.args)


class TaichiField:
    """An extensive wrapper around Taichi field"""

    def __init__(self, field, field_type: FieldType, needs_grad: bool):
        self.field = field
        self.grad = field.grad
        self.field_type = field_type
        self.needs_grad = needs_grad

    def from_torch(self, tensor):
        return self.field.from_torch(tensor)

    def to_torch(self, device=None):
        if device is not None:
            return self.field.to_torch(device)
        else:
            return self.field.to_torch()


class TinFunc(torch.autograd.Function):
    """Customized autograd function used in Tin layers"""

    @staticmethod
    def forward(ctx, tin_configs, *input_tensors):
        ctx.tin_configs = tin_configs
        all_input_fields = tin_configs.input_fields + tin_configs.weight_fields
        assert len(input_tensors) == len(all_input_fields)
        for input_tensor, field in zip(input_tensors, all_input_fields):
            field.from_torch(input_tensor)
        for kernel_bundle in tin_configs.kernel_bundles:
            kernel_bundle.forward()
        output_tensors = []
        for output_field in tin_configs.output_fields:
            output_tensor = output_field.to_torch(device=tin_configs.device).requires_grad_(True)
            output_tensors.append(output_tensor)

        if len(output_tensors) > 1:
            return tuple(output_tensors)
        else:
            return output_tensors[0]

    @staticmethod
    def backward(ctx, *grad_outputs):
        tin_configs = ctx.tin_configs
        for grad_output, output_field in zip(grad_outputs, tin_configs.output_fields):
            if output_field.needs_grad:
                output_field.grad.from_torch(grad_output)
        for kernel_bundle in reversed(tin_configs.kernel_bundles):
            kernel_bundle.backward()
        gradient_tensors = [None]
        for input_field in tin_configs.input_fields:
            if input_field.needs_grad:
                gradient_tensors.append(input_field.grad.to_torch(device=tin_configs.device))
        for weight_field in tin_configs.weight_fields:
            if weight_field.needs_grad:
                gradient_tensors.append(weight_field.grad.to_torch(device=tin_configs.device))
        return tuple(gradient_tensors)


class EmptyTin(torch.nn.Module):
    """A Taichi field wrapper that requires no @ti.data_oriented class"""

    def __init__(self, device: torch.device):
        """
        Init an EmptyTin instance
        :param device: torch.device instance
        """
        super().__init__()
        self.input_fields = []
        self.weight_fields = {}
        self.output_fields = []
        assert isinstance(device, torch.device), "device must be an instance of torch.device"
        self.device = device
        self.tin_configs = None
        self.kernel_bundles = []
        self.kernel_bundle_dict = {}
        self.kernel_args = None
        self.finished = False

    def register_input_field(self, field, needs_grad=None):
        """
        Register an input field which requires a tensor input in the forward calculation
        :param field: Taichi field
        :param needs_grad: whether the field needs grad, `None` for automatic configuration
        :return: self
        """
        assert not self.finished, "Registration after .finish()"
        needs_grad = check_field_needs_grad(field, needs_grad)
        self.input_fields.append(TaichiField(field, FieldType.INPUT, needs_grad))
        return self

    def register_output_field(self, field, needs_grad=None):
        """
        Register an output field that backs an output tensor in the forward calculation
        :param field: Taichi field
        :param needs_grad: whether the field needs grad, `None` for automatic configuration
        :return: self
        """
        assert not self.finished, "Registration after .finish()"
        needs_grad = check_field_needs_grad(field, needs_grad)
        self.output_fields.append(TaichiField(field, FieldType.OUTPUT, needs_grad))
        return self

    def register_weight_field(self, field, needs_grad=None, name=None, value=None):
        """
        Register a field that serves as weights internally and whose values are required by the kernel function
        :param field: Taichi field
        :param needs_grad: whether the field needs grad, `None` for automatic configuration
        :param name: name for the field, facilitating later value setting, `None` for default number naming
        :param value: optional initial values from a tensor
        :return: self
        """
        assert not self.finished, "Registration after .finish()"
        field_name = name if name is not None else str(len(self.weight_fields))
        needs_grad = check_field_needs_grad(field, needs_grad)
        if value is not None:
            field.from_torch(value)
        self.weight_fields[field_name] = TaichiField(field, FieldType.WEIGHTS, needs_grad)
        return self

    def register_kernel(self, kernel, *kernel_args, kernel_name=None):
        """
        Register a kernel for forward calculation
        :param kernel: Taichi kernel
        :param kernel_args: arguments for the kernel
        :param kernel_name: kernel name, optional for new Taichi, compulsory for old Taichi
        :return: self
        """
        assert not self.finished, "Registration after .finish()"
        assert kernel is not None, "Kernel must not be None"
        assert autofill_kernel_name_available(
            kernel) or kernel_name is not None, "kernel has no __name__, please update your Taichi or specify its name"
        assert not isinstance(kernel, str), "Please pass the kernel function, not its name"
        kernel_bundle = TaichiKernelBundle(kernel, kernel_name, *kernel_args)
        assert kernel_bundle.name not in self.kernel_bundle_dict, "Kernel name not found"
        self.kernel_bundles.append(kernel_bundle)
        self.kernel_bundle_dict[kernel_bundle.name] = kernel_bundle
        return self

    def set_weight_field(self, field_name, tensor):
        """
        Sets the value of a weight field from a tensor
        :param field_name: integer(when using default number naming) or string name
        :param tensor: values for the field
        :return: None
        """
        assert self.finished, "Fields for weights can only be set after finishing registrations"
        if isinstance(field_name, int):
            field_name = str(field_name)
        assert field_name in self.weight_fields
        self.weight_fields[field_name].from_torch(tensor)

    def set_kernel_args(self, kernel, *kernel_args):
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
                                      list(self.weight_fields.values()),
                                      self.output_fields,
                                      self.device)
        self.finished = True
        return self

    def forward(self, *input_tensors):
        assert self.finished, "Please finish registrations by calling .finish() before using this layer"
        weight_tensors = tuple(field.to_torch(device=self.device) for field in self.weight_fields.values())
        return TinFunc.apply(self.tin_configs, *(input_tensors + weight_tensors))


class Tin(EmptyTin):
    """A Taichi field wrapper that requires a @ti.data_oriented class for registering a kernel by name"""

    def __init__(self, data_oriented, device: torch.device):
        """
        Init a Tin instance
        :param data_oriented: @ti.data_oriented class instance
        :param device: torch.device instance
        """
        super(Tin, self).__init__(device=device)
        if not hasattr(data_oriented, "_data_oriented"):
            raise Exception("Requires a Taichi data-oriented instance")
        self.data_oriented = data_oriented

    def register_kernel(self, kernel, *kernel_args, kernel_name=None):
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
