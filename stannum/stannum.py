import torch
from enum import Enum
from .utils import check_field_needs_grad


class FieldType(Enum):
    INPUT = 0
    OUTPUT = 1
    WEIGHTS = 2


class TinConfigs:
    def __init__(self,
                 ti_kernel,
                 input_fields,
                 weight_fields,
                 output_fields,
                 device,
                 *kernel_args):
        self.ti_kernel = ti_kernel
        self.input_fields: [TaichiField] = input_fields
        self.weight_fields: [TaichiField] = weight_fields
        self.output_fields: [TaichiField] = output_fields
        self.kernel_args = kernel_args
        self.device: torch.device = device


class TaichiField:
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
    @staticmethod
    def forward(ctx, tin_configs, *input_tensors):
        ctx.tin_configs = tin_configs
        all_input_fields = tin_configs.input_fields + tin_configs.weight_fields
        assert len(input_tensors) == len(all_input_fields)
        for input_tensor, field in zip(input_tensors, all_input_fields):
            field.from_torch(input_tensor)
        tin_configs.ti_kernel(*tin_configs.kernel_args)
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
        tin_configs.ti_kernel.grad(*tin_configs.kernel_args)
        gradient_tensors = [None]
        for input_field in tin_configs.input_fields:
            if input_field.needs_grad:
                gradient_tensors.append(input_field.grad.to_torch(device=tin_configs.device))
        for weight_field in tin_configs.weight_fields:
            if weight_field.needs_grad:
                gradient_tensors.append(weight_field.grad.to_torch(device=tin_configs.device))
        return tuple(gradient_tensors)


class EmptyTin(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.input_fields = []
        self.weight_fields = {}
        self.output_fields = []
        assert device is not None
        self.device = device
        self.tin_configs = None
        self.kernel = None
        self.kernel_args = None
        self.finished = False

    def register_input_field(self, field, needs_grad=None):
        assert not self.finished
        needs_grad = check_field_needs_grad(field, needs_grad)
        self.input_fields.append(TaichiField(field, FieldType.INPUT, needs_grad))
        return self

    def register_output_field(self, field, needs_grad=None):
        assert not self.finished
        needs_grad = check_field_needs_grad(field, needs_grad)
        self.output_fields.append(TaichiField(field, FieldType.OUTPUT, needs_grad))
        return self

    def register_weight_field(self, field, needs_grad=None, name=None, value=None):
        assert not self.finished
        field_name = name if name is not None else str(len(self.weight_fields))
        needs_grad = check_field_needs_grad(field, needs_grad)
        if value is not None:
            field.from_torch(value)
        self.weight_fields[field_name] = TaichiField(field, FieldType.WEIGHTS, needs_grad)
        return self

    def register_kernel(self, kernel):
        assert not self.finished
        assert kernel is not None
        assert not isinstance(kernel, str)
        self.kernel = kernel
        return self

    def set_weight_field(self, field_name, tensor):
        assert self.finished
        if isinstance(field_name, int):
            field_name = str(field_name)
        assert field_name in self.weight_fields
        self.weight_fields[field_name].from_torch(tensor)

    def set_kernel_args(self, *kernel_args):
        self.kernel_args = kernel_args
        if self.finished:
            self.tin_configs.kernel_args = kernel_args

    def finish(self):
        assert len(self.input_fields) > 0
        assert len(self.output_fields) > 0
        assert self.kernel is not None
        self.tin_configs = TinConfigs(self.kernel,
                                      self.input_fields,
                                      list(self.weight_fields.values()),
                                      self.output_fields,
                                      self.device,
                                      self.kernel_args)
        self.finished = True
        return self

    def forward(self, *input_tensors):
        assert self.finished
        weight_tensors = tuple(field.to_torch(device=self.device) for field in self.weight_fields.values())
        return TinFunc.apply(self.tin_configs, *(input_tensors + weight_tensors))


class Tin(EmptyTin):
    def __init__(self, data_oriented, device: torch.device):
        super(Tin, self).__init__(device=device)
        if not hasattr(data_oriented, "_data_oriented"):
            raise Exception("Requires a Taichi data-oriented instance")
        self.data_oriented = data_oriented

    def register_kernel(self, kernel):
        if isinstance(kernel, str):
            kernel = getattr(self.data_oriented, kernel)
        assert kernel is not None
        super(Tin, self).register_kernel(kernel)
        return self
