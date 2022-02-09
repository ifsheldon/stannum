import taichi as ti
import torch
from taichi.lang.impl import axes
from typing import Optional, Callable, Union, Tuple, List, Iterable, Dict, Any
from taichi.lang.util import to_taichi_type
from taichi._lib.core.taichi_core import DataType as TiDataType
from taichi.lang.field import ScalarField
from taichi.lang.matrix import MatrixField
from functools import partial
from torch.autograd.function import once_differentiable

from .utils import is_kernel, autofill_kernel_name_available
from .auxiliary import FieldManager, SNode


class DefaultFieldManager(FieldManager):
    """
    Default field manager which layouts data in tensors by constructing fields
    with the ordinary multidimensional array layout
    """

    def __init__(self,
                 dtype: TiDataType,
                 complex_dtype: bool,
                 device: torch.device):
        self.dtype: TiDataType = dtype
        self.complex_dtype: bool = complex_dtype
        self.device: torch.device = device

    def construct_field(self,
                        fields_builder: ti.FieldsBuilder,
                        concrete_tensor_shape: Tuple[int, ...],
                        needs_grad: bool) -> Union[ScalarField, MatrixField]:
        assert not fields_builder.finalized
        if self.complex_dtype:
            field = ti.Vector.field(2, dtype=self.dtype, needs_grad=needs_grad)
        else:
            field = ti.field(self.dtype, needs_grad=needs_grad)

        if needs_grad:
            fields_builder \
                .dense(axes(*range(len(concrete_tensor_shape))), concrete_tensor_shape) \
                .place(field, field.grad)
        else:
            fields_builder.dense(axes(*range(len(concrete_tensor_shape))), concrete_tensor_shape).place(field)
        return field

    def to_tensor(self, field: Union[ScalarField, MatrixField]) -> torch.Tensor:
        tensor = field.to_torch(device=self.device)
        if self.complex_dtype:
            tensor = torch.view_as_complex(tensor)
        return tensor

    def grad_to_tensor(self, grad_field: Union[ScalarField, MatrixField]) -> torch.Tensor:
        tensor = grad_field.to_torch(device=self.device)
        if self.complex_dtype:
            tensor = torch.view_as_complex(tensor)
        return tensor

    def from_tensor(self, field: Union[ScalarField, MatrixField], tensor: torch.Tensor):
        if self.complex_dtype:
            tensor = torch.view_as_real(tensor).clone()
        field.from_torch(tensor)

    def grad_from_tensor(self, grad_field: Union[ScalarField, MatrixField], tensor: torch.Tensor):
        if self.complex_dtype:
            tensor = torch.view_as_real(tensor).clone()
        grad_field.from_torch(tensor)


class ConcreteField:
    """
    An extension of Taichi fields with auto deconstruction
    """

    def __init__(self,
                 dtype: TiDataType,
                 concrete_tensor_shape: Tuple[int, ...],
                 field_manager: FieldManager,
                 fields_builder: ti.FieldsBuilder,
                 complex_dtype: bool,
                 requires_grad: bool,
                 device: torch.device,
                 name: str):
        assert all(map(lambda x: isinstance(x, int), concrete_tensor_shape))
        if field_manager is None:
            field_manager = DefaultFieldManager(dtype, complex_dtype, device)
        field = field_manager.construct_field(fields_builder, concrete_tensor_shape, requires_grad)
        self.fb = fields_builder
        self.complex_dtype: bool = complex_dtype
        self.field: Union[ScalarField, MatrixField] = field
        self.device: torch.device = device
        self.name: str = name
        self.field_manager: FieldManager = field_manager
        self.requires_grad: bool = requires_grad

    def clear_grad(self):
        assert self.fb.finalized
        if self.requires_grad:
            self.field.grad.fill(0)

    def clear_field(self):
        assert self.fb.finalized
        self.field.fill(0)

    def to_tensor(self) -> torch.Tensor:
        return self.field_manager.to_tensor(self.field)

    def grad_to_tensor(self) -> torch.Tensor:
        return self.field_manager.grad_to_tensor(self.field.grad)

    def from_tensor(self, tensor):
        self.field_manager.from_tensor(self.field, tensor)

    def grad_from_tensor(self, tensor):
        self.field_manager.grad_from_tensor(self.field.grad, tensor)


class Seal:

    def __init__(self, dtype: Union[TiDataType, torch.dtype],
                 *dims: int,
                 field_manager: Optional[FieldManager] = None,
                 requires_grad: Optional[bool] = None,
                 name: Optional[str] = None):
        assert dtype is not None, "dtype must not be None"
        # validate dims
        if len(dims) > 0:  # not scalar
            if dims[0] is None:
                for i in range(1, len(dims)):
                    assert dims[i] is not None, "Only the leading dimension can be None (i.e. the batch dimension)"
            else:
                for i in range(len(dims)):
                    assert dims[i] is not None, "Only the leading dimension can be None (i.e. the batch dimension)"

        for i in dims:
            assert i != 0, f"Dimension cannot be 0, got {dims}"

        self.complex_dtype = dtype == torch.cfloat or dtype == torch.cdouble
        if self.complex_dtype:
            dtype = ti.f32 if dtype == torch.cfloat else ti.f64
        self.dtype: TiDataType = to_taichi_type(dtype) if dtype is not None else dtype
        self.field_manager: FieldManager = field_manager
        self.dims: Tuple[int, ...] = dims
        self.batched: bool = len(dims) > 0 and dims[0] is None
        self.name: str = name
        self.requires_grad: bool = requires_grad

    def concretize(self, concrete_shape: Tuple[int, ...],
                   fields_builder: ti.FieldsBuilder,
                   device: torch.device,
                   needs_grad: bool) -> ConcreteField:
        return ConcreteField(self.dtype, concrete_shape,
                             self.field_manager, fields_builder,
                             self.complex_dtype,
                             needs_grad if self.requires_grad is None else self.requires_grad,
                             device, self.name)


class TubeKernelBundle:
    """
    Extension of Taichi kernel
    """

    def __init__(self, kernel: Callable, name: Optional[str], seals: List[Seal], extra_args: Tuple[Any, ...]):
        self.kernel: Callable = kernel
        self.name: str = kernel.__name__ if name is None else name
        self.seals: List[Seal] = seals
        self.seal_names: List[str] = [s.name for s in seals]
        self.extra_args: Tuple[Any, ...] = extra_args

    def forward(self, seal_name_to_concrete_field: Dict[str, ConcreteField]):
        concrete_fields = map(lambda seal_name: seal_name_to_concrete_field[seal_name], self.seal_names)
        ti_fields = tuple(map(lambda x: x.field, concrete_fields))
        self.kernel(*(ti_fields + self.extra_args))

    def backward(self, seal_name_to_concrete_field: Dict[str, ConcreteField]):
        concrete_fields = map(lambda seal_name: seal_name_to_concrete_field[seal_name], self.seal_names)
        ti_fields = tuple(map(lambda x: x.field, concrete_fields))
        self.kernel.grad(*(ti_fields + self.extra_args))


class Tube(torch.nn.Module):
    """
    Self-managed Taichi-PyTorch adapter
    """

    def __init__(self,
                 device: Optional[torch.device] = None):
        """
        Init a tube

        @param device: Optional, torch.device tensors are on, if it's None, the device is determined by input tensors
        """
        super().__init__()
        self.input_placeholders: List[Seal] = []
        self.output_placeholders: List[Seal] = []
        self.intermediate_field_placeholders: List[Seal] = []
        self.seals: Dict[str, Seal] = {}
        self.kernel_bundles: List[TubeKernelBundle] = []
        self.device: Optional[torch.device] = device
        self._finished: bool = False
        self.batched: bool = False
        self.kernel_bundle_dict: Dict[str, TubeKernelBundle] = {}

    def register_input_tensor(self,
                              dims: Iterable[Union[int, None]],
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
        assert not self._finished, "Try to register input tensor after .finish()"
        assert dtype is not None, "dtype cannot be None"
        assert isinstance(dtype, torch.dtype)
        assert name is not None, "name cannot be None"
        assert name not in self.seals, "name registered"
        seal = Seal(dtype, *dims,
                    field_manager=field_manager,
                    requires_grad=requires_grad,
                    name=name)
        if seal.batched:
            self.batched = True
        self.input_placeholders.append(seal)
        self.seals[name] = seal
        return self

    def register_output_tensor(self,
                               dims: Iterable[Union[int, None]],
                               dtype: torch.dtype,
                               name: str,
                               requires_grad: bool,
                               field_manager: Optional[FieldManager] = None):
        """
        Register an output tensor
        @param dims: dims can contain `None`, positive and negative numbers,
        for restrictions and requirements, see README
        @param dtype: torch data type
        @param name: name of the tensor and corresponding field
        @param requires_grad: if the output requires gradients
        @param field_manager: customized field manager, if it's None, a DefaultFieldManger will be used
        """
        assert not self._finished, "Try to register output tensor after .finish()"
        assert dtype is not None, "dtype cannot be None"
        assert isinstance(dtype, torch.dtype)
        assert name is not None, "name cannot be None"
        assert name not in self.seals, "name registered"
        assert requires_grad is not None, "requires_grad cannot be None when registering an output tensor"
        assert not any(map(lambda d: d == -1, dims)), \
            "Dim = -1 is not allowed when registering output tensors but only registering input tensors"
        seal = Seal(dtype, *dims,
                    field_manager=field_manager,
                    requires_grad=requires_grad,
                    name=name)
        if self.batched:
            assert seal.batched, \
                "Already registered batched inputs, so outputs should also be batched, which means dims[0] must be None"
        self.output_placeholders.append(seal)
        self.seals[name] = seal
        return self

    def register_intermediate_field(self,
                                    dims: Iterable[Union[int, None]],
                                    ti_dtype: TiDataType,
                                    name: str,
                                    needs_grad: bool,
                                    field_manager: Optional[FieldManager] = None):
        """
        Register an intermediate field,
        which can be useful if multiple kernels are used and intermediate results between kernels are stored

        @param dims: dims can contain `None`, positive and negative numbers,
        for restrictions and requirements, see README
        @param ti_dtype: taichi data type
        @param name: name of the field
        @param needs_grad: if the field needs gradients.
        @param field_manager: customized field manager, if it's None, a DefaultFieldManger will be used
        """
        assert not self._finished, "Try to register intermediate field after .finish()"
        assert ti_dtype is not None, "dtype cannot be None"
        assert isinstance(ti_dtype, TiDataType)
        assert name is not None, "name cannot be None"
        assert name not in self.seals, "name registered"
        assert needs_grad is not None, "requires_grad cannot be None when registering an intermediate field"
        assert not any(map(lambda d: d == -1, dims)), \
            "Dim = -1 is not allowed when registering intermediate fields but only registering input tensors"
        seal = Seal(ti_dtype, *dims,
                    field_manager=field_manager,
                    requires_grad=needs_grad,
                    name=name)
        if seal.batched:
            self.batched = True
        self.intermediate_field_placeholders.append(seal)
        self.seals[name] = seal
        return self

    def register_kernel(self, kernel: Callable, tensor_names: List[str], *extra_args: Any, name: Optional[str] = None):
        """
        Register a Taichi kernel

        @param kernel: Taichi kernel. For requirements, see README
        @param tensor_names: the names of registered tensors that are to be used in this kernel
        @param extra_args: any extra arguments passed to the kernel
        @param name: name of this kernel, if it's None, it will be kernel.__name__
        """
        assert not self._finished, "Try to register kernel after .finish()"
        assert is_kernel(kernel), "Passed function is not a Taichi kernel"
        assert autofill_kernel_name_available(kernel) or name is not None, \
            "kernel has no __name__, please update your Taichi or specify its name"
        assert all(map(lambda x: isinstance(x, str), tensor_names)), "arg_names must be strings"
        not_registered_names = list(filter(lambda x: x not in self.seals, tensor_names))
        assert len(not_registered_names) == 0, f"Some names are not registered: {not_registered_names}"
        seals = list(map(lambda x: self.seals[x], tensor_names))
        kernel_bundle = TubeKernelBundle(kernel, name, seals, extra_args)
        assert kernel_bundle.name not in self.kernel_bundle_dict, \
            f"Kernel with name {kernel_bundle.name} already registered"
        self.kernel_bundles.append(kernel_bundle)
        self.kernel_bundle_dict[kernel_bundle.name] = kernel_bundle
        return self

    def set_kernel_extra_args(self, kernel: Union[Callable, str], *extra_args: Any):
        """
        Set args for a kernel
        @param kernel: kernel function or its name
        @param extra_args: extra kernel arguments
        """
        if isinstance(kernel, str):
            kernel_name = kernel
        else:
            kernel_name = kernel.__name__
        assert kernel_name in self.kernel_bundle_dict, \
            f"Kernel with name {kernel_name} not found, please register it first"
        self.kernel_bundle_dict[kernel_name].extra_args = extra_args

    def finish(self):
        """
        Finish all registrations
        """
        if self._finished:
            return self
        assert len(self.input_placeholders) > 0, "Must register at least 1 input field"
        assert len(self.output_placeholders) > 0, "Must register at least 1 output field"
        assert len(self.kernel_bundles) > 0, "Must register at least 1 kernel"
        if self.batched:
            for output_seal in self.output_placeholders:
                assert output_seal.batched, \
                    "Already registered batched inputs, so outputs should also be batched, " \
                    "which means dims[0] must be None"
        # neg dim check
        neg_dims = {-1}
        for ip in self.input_placeholders:
            for d in ip.dims:
                if d is not None and d < 0:
                    neg_dims.add(d)

        for placeholder in self.intermediate_field_placeholders + self.output_placeholders:
            for d in placeholder.dims:
                if d is not None and d < 0 and d not in neg_dims:
                    raise Exception(f"Dimension={d} in {placeholder.name} is not registered in any input tensors")
        self._finished = True
        return self

    def forward(self, *input_tensors: torch.Tensor):
        input_tensors = map(lambda x: x.clone() if x._is_view() else x, input_tensors)
        return TubeFunc.apply(self, *input_tensors)


def unify_and_concretize_shapes(tensor_shapes: List[Tuple[int, ...]],
                                input_placeholders: List[Seal],
                                intermediate_fields: List[Seal],
                                output_placeholders: List[Seal]) \
        -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]], List[Tuple[int, ...]], Optional[int]]:
    """
    Try to find out concrete numbers in dimension placeholders (like `None`, `-1`, `-2`)
    """
    input_dims = list(map(lambda x: x.dims, input_placeholders))

    # check dimensionality and batch nums
    batch_num = None
    for i, (tensor_shape, input_dim) in enumerate(zip(tensor_shapes, input_dims)):
        assert len(tensor_shape) == len(input_dim), \
            f"Dimensionality check failed, expecting the {i}th tensor to be {len(input_dim)}D, got {len(tensor_shape)}D"
        if len(input_dim) == 0:  # scalar, shape = ()
            continue
        elif input_dim[0] is None:  # shape = (None, ...)
            if batch_num is None:
                batch_num = tensor_shape[0]
            else:
                assert tensor_shape[0] == batch_num, f"Batch num of {i}th tensor not match, " \
                                                     f"expect: {batch_num}, got {tensor_shape[0]}"
        else:
            assert all(map(lambda x, y: y < 0 or x == y, tensor_shape, input_dim)), \
                f"{i}th tensor dimensions not match, expect: {input_dim}, got {tensor_shape}"

    # fill in <0 dimensions and batch dimension
    concrete_input_dims = list(map(list, input_dims))
    neg_dims = {}
    for idx, input_dim in enumerate(input_dims):
        for i, d in enumerate(input_dim):
            if d is None:
                concrete_input_dims[idx][i] = batch_num
            elif d == -1:
                concrete_input_dims[idx][i] = tensor_shapes[idx][i]
            elif d < -1:
                concrete_dim = tensor_shapes[idx][i]
                if d in neg_dims:
                    assert neg_dims[d] == concrete_dim, f"Dim = {d} not match"
                else:
                    neg_dims[d] = concrete_dim
                concrete_input_dims[idx][i] = concrete_dim

    output_dims = list(map(lambda x: x.dims, output_placeholders))
    intermediate_dims = list(map(lambda x: x.dims, intermediate_fields))
    concrete_output_dims = list(map(list, output_dims))
    concrete_intermediate_dims = list(map(list, intermediate_dims))
    for idx, output_dim in enumerate(output_dims):
        for i, d in enumerate(output_dim):
            if d is None:
                concrete_output_dims[idx][i] = batch_num
            elif d < -1:
                concrete_output_dims[idx][i] = neg_dims[d]
            else:  # d > 0, no d == -1
                pass

    for idx, inter_dim in enumerate(intermediate_dims):
        for i, d in enumerate(inter_dim):
            if d is None:
                concrete_intermediate_dims[idx][i] = batch_num
            elif d < -1:
                concrete_intermediate_dims[idx][i] = neg_dims[d]
            else:  # d > 0, no d == -1
                pass

    concrete_input_dims = list(map(tuple, concrete_input_dims))
    concrete_output_dims = list(map(tuple, concrete_output_dims))
    concrete_intermediate_dims = list(map(tuple, concrete_intermediate_dims))
    return concrete_input_dims, concrete_intermediate_dims, concrete_output_dims, batch_num


class TubeFunc(torch.autograd.Function):

    @staticmethod
    def select_concrete_field(seals: List[Seal],
                              concrete_fields: List[Union[ConcreteField, List[ConcreteField]]],
                              batch_idx: int) -> List[ConcreteField]:
        assert len(seals) == len(concrete_fields)
        selected_concrete_fields = []
        for seal, concrete_field in zip(seals, concrete_fields):
            if seal.batched:
                selected_concrete_fields.append(concrete_field[batch_idx])
            else:
                selected_concrete_fields.append(concrete_field)
        return selected_concrete_fields

    @staticmethod
    def select_tensor(seals: List[Seal], tensors: Tuple[torch.Tensor], batch_idx: int) -> List[torch.Tensor]:
        assert len(seals) == len(tensors)
        selected_tensors = []
        for seal, tensor in zip(seals, tensors):
            if seal.batched:
                selected_tensors.append(tensor[batch_idx])
            else:
                selected_tensors.append(tensor)
        return selected_tensors

    @staticmethod
    def concretize(device: torch.device,
                   fields_builder: ti.FieldsBuilder,
                   needs_grad: bool,
                   concrete_shape: Tuple[int, ...],
                   seal: Seal) -> ConcreteField:
        concrete_field = seal.concretize(concrete_shape, fields_builder, device, needs_grad)
        return concrete_field

    @staticmethod
    def forward(ctx, tube: Tube, *input_tensors: torch.Tensor):
        assert len(input_tensors) == len(tube.input_placeholders)
        input_seals = tube.input_placeholders
        output_seals = tube.output_placeholders
        intermediate_seals = tube.intermediate_field_placeholders
        if tube.device is None:
            device = input_tensors[0].device
            for t in input_tensors:
                assert t.device == device, f"Tensors not on the same device {device}"
        else:
            device = tube.device
            for t in input_tensors:
                assert t.device == device, f"Tensors not on the device {device}"

        fb = ti.FieldsBuilder()
        input_tensor_shapes = list(map(lambda x: x.shape, input_tensors))
        concrete_input_shapes, concrete_intermediate_shapes, concrete_output_shapes, batch_num = unify_and_concretize_shapes(
            input_tensor_shapes,
            input_seals, intermediate_seals, output_seals)
        input_field_concretizer = partial(TubeFunc.concretize, device, fb)
        other_field_concretizer = partial(TubeFunc.concretize, device, fb, None)
        if batch_num is None:
            input_concrete_fields: List[Union[ConcreteField, List[ConcreteField]]] = list(
                map(input_field_concretizer,
                    map(lambda x: x.requires_grad, input_tensors),
                    concrete_input_shapes,
                    input_seals))
            intermediate_concrete_fields: List[Union[ConcreteField, List[ConcreteField]]] = list(
                map(other_field_concretizer, concrete_intermediate_shapes, intermediate_seals))
            output_concrete_fields: List[Union[ConcreteField, List[ConcreteField]]] = list(
                map(other_field_concretizer, concrete_output_shapes, output_seals))
            seal_name_to_concrete_fields = {
                seal.name: concrete_field
                for seal, concrete_field in
                zip(input_seals + intermediate_seals + output_seals,
                    input_concrete_fields + intermediate_concrete_fields + output_concrete_fields)
            }
            snode = fb.finalize()
            for field in intermediate_concrete_fields + output_concrete_fields:
                field.clear_field()

            for tensor, concrete_input_field in zip(input_tensors, input_concrete_fields):
                concrete_input_field.from_tensor(tensor)

            for kernel_bundle in tube.kernel_bundles:
                kernel_bundle.forward(seal_name_to_concrete_fields)

            output_tensors = tuple(ocf.to_tensor().requires_grad_(s.requires_grad) for s, ocf in
                                   zip(output_seals, output_concrete_fields))
        else:
            input_concrete_fields: List[Union[ConcreteField, List[ConcreteField]]] = []
            for tensor_shape, seal, tensor in zip(concrete_input_shapes, input_seals, input_tensors):
                requires_grad = tensor.requires_grad
                if seal.batched:
                    tensor_shape = tensor_shape[1:]
                    concrete_fields = [input_field_concretizer(requires_grad, tensor_shape, seal)
                                       for _ in range(batch_num)]
                else:
                    concrete_fields = input_field_concretizer(requires_grad, tensor_shape, seal)
                input_concrete_fields.append(concrete_fields)

            intermediate_concrete_fields: List[Union[ConcreteField, List[ConcreteField]]] = []
            for tensor_shape, seal in zip(concrete_intermediate_shapes, intermediate_seals):
                if seal.batched:
                    tensor_shape = tensor_shape[1:]
                    concrete_fields = [other_field_concretizer(tensor_shape, seal) for _ in range(batch_num)]
                else:
                    concrete_fields = other_field_concretizer(tensor_shape, seal)
                intermediate_concrete_fields.append(concrete_fields)

            output_concrete_fields: List[Union[ConcreteField, List[ConcreteField]]] = []
            for tensor_shape, seal in zip(concrete_output_shapes, output_seals):
                if seal.batched:
                    tensor_shape = tensor_shape[1:]
                    concrete_fields = [other_field_concretizer(tensor_shape, seal) for _ in range(batch_num)]
                else:
                    concrete_fields = other_field_concretizer(tensor_shape, seal)
                output_concrete_fields.append(concrete_fields)

            snode = fb.finalize()
            scf = TubeFunc.select_concrete_field
            output_tensor_batches = []
            for batch_idx in range(batch_num):
                concrete_input_field_batch = scf(input_seals, input_concrete_fields, batch_idx)
                concrete_intermediate_field_batch = scf(intermediate_seals, intermediate_concrete_fields, batch_idx)
                concrete_output_field_batch = scf(output_seals, output_concrete_fields, batch_idx)
                seal_name_to_concrete_fields = {
                    seal.name: concrete_field
                    for seal, concrete_field in
                    zip(input_seals + intermediate_seals + output_seals,
                        concrete_input_field_batch + concrete_intermediate_field_batch + concrete_output_field_batch)
                }
                for field in concrete_intermediate_field_batch + concrete_output_field_batch:
                    field.clear_field()
                input_tensor_batch = TubeFunc.select_tensor(input_seals, input_tensors, batch_idx)
                for tensor, concrete_input_field in zip(input_tensor_batch, concrete_input_field_batch):
                    concrete_input_field.from_tensor(tensor)
                for kernel_bundle in tube.kernel_bundles:
                    kernel_bundle.forward(seal_name_to_concrete_fields)
                output_tensors = [ocf.to_tensor() for ocf in concrete_output_field_batch]
                output_tensor_batches.append(output_tensors)

            output_tensors = []
            for output_idx, output_seal in enumerate(output_seals):
                tensors = [output_tensor_batches[batch_idx][output_idx]
                           for batch_idx in range(batch_num)]
                output_tensors.append(torch.stack(tensors, dim=0).requires_grad_(output_seal.requires_grad))

            output_tensors = tuple(output_tensors)

        ctx.input_concrete_fields = input_concrete_fields
        ctx.intermediate_concrete_fields = intermediate_concrete_fields
        ctx.output_concrete_fields = output_concrete_fields
        ctx.batch_num = batch_num
        ctx.tube = tube
        ctx.snode = SNode(snode)
        ctx.mark_non_differentiable(*filter(lambda x: not x.requires_grad, output_tensors))
        if len(output_tensors) == 1:
            return output_tensors[0]
        else:
            return output_tensors

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> Any:
        tube: Tube = ctx.tube
        batch_num = ctx.batch_num
        input_seals = tube.input_placeholders
        output_seals = tube.output_placeholders
        intermediate_seals = tube.intermediate_field_placeholders
        input_concrete_fields = ctx.input_concrete_fields
        intermediate_concrete_fields = ctx.intermediate_concrete_fields
        output_concrete_fields = ctx.output_concrete_fields
        if batch_num is None:
            for grad_tensor, output_concrete_field in zip(grad_outputs, output_concrete_fields):
                if output_concrete_field.requires_grad:
                    output_concrete_field.grad_from_tensor(grad_tensor)
            seal_name_to_concrete_fields = {
                seal.name: concrete_field
                for seal, concrete_field in
                zip(input_seals + intermediate_seals + output_seals,
                    input_concrete_fields + intermediate_concrete_fields + output_concrete_fields)
            }
            for field in intermediate_concrete_fields + input_concrete_fields:
                field.clear_grad()
            for kernel_bundle in reversed(tube.kernel_bundles):
                kernel_bundle.backward(seal_name_to_concrete_fields)

            gradient_tensors = [None]
            for input_concrete_field in input_concrete_fields:
                if input_concrete_field.requires_grad:
                    gradient_tensors.append(input_concrete_field.grad_to_tensor())
                else:
                    gradient_tensors.append(None)
            ctx.snode.destroy()
            return tuple(gradient_tensors)
        else:
            scf = TubeFunc.select_concrete_field
            gradients = []
            for batch_idx in range(batch_num):
                grad_output_batch = TubeFunc.select_tensor(output_seals, grad_outputs, batch_idx)
                output_concrete_field_batch = scf(output_seals, output_concrete_fields, batch_idx)
                for grad_tensor, output_concrete_field in zip(grad_output_batch, output_concrete_field_batch):
                    if output_concrete_field.requires_grad:
                        output_concrete_field.grad_from_tensor(grad_tensor)
                input_concrete_field_batch = scf(input_seals, input_concrete_fields, batch_idx)
                intermediate_concrete_field_batch = scf(intermediate_seals, intermediate_concrete_fields, batch_idx)
                seal_name_to_concrete_fields = {
                    seal.name: concrete_field
                    for seal, concrete_field in
                    zip(input_seals + intermediate_seals + output_seals,
                        input_concrete_field_batch + intermediate_concrete_field_batch + output_concrete_field_batch)
                }
                for field in intermediate_concrete_field_batch + input_concrete_field_batch:
                    field.clear_grad()
                for kernel_bundle in reversed(tube.kernel_bundles):
                    kernel_bundle.backward(seal_name_to_concrete_fields)
                grad_tensor_batch = []
                for input_concrete_field in input_concrete_field_batch:
                    if input_concrete_field.requires_grad:
                        grad_tensor_batch.append(input_concrete_field.grad_to_tensor())
                    else:
                        grad_tensor_batch.append(None)
                gradients.append(grad_tensor_batch)

            input_grads = [None]
            for input_idx, input_seal in enumerate(input_seals):
                grad_per_input = [gradients[batch_idx][input_idx] for batch_idx in range(batch_num)]
                if any(map(lambda x: x is None, grad_per_input)):
                    input_grads.append(None)
                else:
                    if input_seal.batched:
                        input_grads.append(torch.stack(grad_per_input, dim=0))
                    else:
                        input_grads.append(torch.stack(grad_per_input, dim=0).sum(dim=0))
            ctx.snode.destroy()
            return tuple(input_grads)
