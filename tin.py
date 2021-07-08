import taichi as ti
import torch


class TinConfigs:
    def __init__(self,
                 data_oriented,
                 input_fields,
                 output_fields,
                 device,
                 *kernel_args,
                 **kernel_kwargs):
        self.data_oriented = data_oriented
        self.input_fields: [TaichiField] = input_fields
        self.output_fields: [TaichiField] = output_fields
        self.kernel_args = kernel_args
        self.kernel_kwargs = kernel_kwargs
        self.device: torch.device = device


class TaichiField:
    def __init__(self, field, is_input_field, needs_grad):
        self.field = field
        self.grad = field.grad
        self.is_input_field = is_input_field
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
        for input_tensor, input_field in zip(input_tensors, tin_configs.input_fields):
            input_field.from_torch(input_tensor)
        if len(tin_configs.kernel_args) == 0:
            tin_configs.data_oriented.forward_kernel()
        else:
            tin_configs.data_oriented.forward_kernel(*tin_configs.kernel_args)
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
        for grad_output, output_field in zip(grad_outputs, ctx.tin_configs.output_fields):
            if output_field.needs_grad:
                output_field.grad.from_torch(grad_output)
        if len(ctx.tin_configs.kernel_args) == 0:
            ctx.tin_configs.data_oriented.forward_kernel.grad()
        else:
            ctx.tin_configs.data_oriented.forward_kernel.grad(*ctx.tin_configs.kernel_args)
        gradient_tensors = [None]
        for input_field in ctx.tin_configs.input_fields:
            if input_field.needs_grad:
                gradient_tensors.append(input_field.field.grad.to_torch(device=ctx.tin_configs.device))
        return tuple(gradient_tensors)


class Tin(torch.nn.Module):
    def __init__(self, data_oriented, device: torch.device):
        super().__init__()
        if not hasattr(data_oriented, "_data_oriented"):
            raise Exception("Requires a Taichi data-oriented instance")
        self.data_oriented = data_oriented
        self.input_fields = []
        self.output_fields = []
        assert device is not None
        self.device = device
        self.tin_func = TinFunc()
        self.tin_configs = None
        self.kernel_args = None
        self.kernel_kwargs = None
        self.finished = False

    def register_input_field(self, field, needs_grad):
        assert not self.finished
        self.input_fields.append(TaichiField(field, True, needs_grad))
        return self

    def register_output_field(self, field, needs_grad):
        assert not self.finished
        self.output_fields.append(TaichiField(field, False, needs_grad))
        return self

    def set_kernel_args(self, *kernel_args):
        self.kernel_args = kernel_args
        if self.finished:
            self.tin_configs.kernel_args = kernel_args

    def set_kernel_kwargs(self, **kernel_kwargs):
        self.kernel_kwargs = kernel_kwargs
        if self.finished:
            self.tin_configs.kernel_kwargs = kernel_kwargs

    def finish(self):
        assert len(self.input_fields) > 0
        self.tin_configs = TinConfigs(self.data_oriented,
                                      self.input_fields,
                                      self.output_fields,
                                      self.device,
                                      self.kernel_args,
                                      self.kernel_kwargs)
        self.finished = True
        return self

    def forward(self, *input_tensors):
        assert self.finished
        # TODO
        pass
