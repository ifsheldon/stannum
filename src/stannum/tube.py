import taichi as ti
import torch
from taichi.lang.impl import axes
from typing import Optional, Callable, Union, Tuple, List, Iterable, Dict, Any
from taichi import to_taichi_type
from taichi._lib.core.taichi_core import DataType as TiDataType
from taichi.lang.field import ScalarField
from taichi.lang.matrix import MatrixField
from taichi.snode.snode_tree import SNodeTree
from abc import ABC, abstractmethod
from functools import partial
from torch.autograd.function import once_differentiable

from .utils import is_kernel, autofill_kernel_name_available


class FieldManager(ABC):
    """
    FieldManagers enable potential flexible field constructions and manipulations.

    For example, instead of ordinarily layout-ting a multidimensional field,
    you can do hierarchical placements for fields, which may gives dramatic performance improvements
    based on applications. Since hierarchical fields may not have the same shape of input tensor,
    it's YOUR responsibility to write a FieldManager that can correctly transform field values into/from tensors
    """

    @abstractmethod
    def construct_field(self,
                        fields_builder: ti.FieldsBuilder,
                        concrete_tensor_shape: Tuple[int, ...],
                        needs_grad: bool) -> Union[ScalarField, MatrixField]:
        pass

    @abstractmethod
    def to_tensor(self, field: Union[ScalarField, MatrixField]) -> torch.Tensor:
        pass

    @abstractmethod
    def grad_to_tensor(self, grad_field: Union[ScalarField, MatrixField]) -> torch.Tensor:
        pass

    @abstractmethod
    def from_tensor(self, field: Union[ScalarField, MatrixField], tensor: torch.Tensor):
        pass

    @abstractmethod
    def grad_from_tensor(self, grad_field: Union[ScalarField, MatrixField], tensor: torch.Tensor):
        pass


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
        self.snode: SNodeTree = None

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
            tensor = torch.view_as_real(tensor)
        field.from_torch(tensor)

    def grad_from_tensor(self, grad_field: Union[ScalarField, MatrixField], tensor: torch.Tensor):
        if self.complex_dtype:
            tensor = torch.view_as_real(tensor)
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
    def __init__(self, kernel: Callable, name: Optional[str], seals: List[Seal]):
        self.kernel: Callable = kernel
        self.name: str = kernel.__name__ if name is None else name
        self.seals: List[Seal] = seals
        self.seal_names: List[str] = [s.name for s in seals]

    def forward(self, seal_name_to_concrete_field: Dict[str, ConcreteField]):
        concrete_fields = map(lambda seal_name: seal_name_to_concrete_field[seal_name], self.seal_names)
        ti_fields = map(lambda x: x.field, concrete_fields)
        self.kernel(*ti_fields)

    def backward(self, seal_name_to_concrete_field: Dict[str, ConcreteField]):
        concrete_fields = map(lambda seal_name: seal_name_to_concrete_field[seal_name], self.seal_names)
        ti_fields = map(lambda x: x.field, concrete_fields)
        self.kernel.grad(*ti_fields)


class Tube(torch.nn.Module):

    def __init__(self,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.input_placeholders: List[Seal] = []
        self.output_placeholders: List[Seal] = []
        self.intermediate_field_placeholders: List[Seal] = []
        self.seals: Dict[str, Seal] = {}
        self.kernel_bundles: List[TubeKernelBundle] = []
        self.device: Optional[torch.device] = device
        self._finished: bool = False

    def finish(self):
        if self._finished:
            return self
        assert len(self.input_placeholders) > 0, "Must register at least 1 input field"
        assert len(self.output_placeholders) > 0, "Must register at least 1 output field"
        assert len(self.kernel_bundles) > 0, "Must register at least 1 kernel"
        # neg dim check
        neg_dims = {-1}
        for ip in self.input_placeholders:
            for d in ip.dims:
                if d is not None and d < 0:
                    neg_dims.add(d)

        for placeholder in self.intermediate_field_placeholders + self.output_placeholders:
            for d in placeholder.dims:
                if d < 0 and d not in neg_dims:
                    raise Exception(f"Dimension={d} in {placeholder.name} is not registered in any input tensors")
        self._finished = True
        return self

    def register_input_tensor(self,
                              dims: Iterable[int],
                              dtype: torch.dtype,
                              name: str,
                              requires_grad: Optional[bool] = None,
                              field_manager: Optional[FieldManager] = None):
        assert not self._finished, "Try to register input tensor after .finish()"
        assert dtype is not None, "dtype cannot be None"
        assert isinstance(dtype, torch.dtype)
        assert name is not None, "name cannot be None"
        assert name not in self.seals, "name registered"
        seal = Seal(dtype, *dims,
                    field_manager=field_manager,
                    requires_grad=requires_grad,
                    name=name)
        self.input_placeholders.append(seal)
        self.seals[name] = seal
        return self

    def register_output_tensor(self,
                               dims: Iterable[int],
                               dtype: torch.dtype,
                               name: str,
                               requires_grad: bool,
                               field_manager: Optional[FieldManager] = None):
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
        self.output_placeholders.append(seal)
        self.seals[name] = seal
        return self

    def register_intermediate_field(self,
                                    dims: Iterable[int],
                                    ti_dtype: TiDataType,
                                    name: str,
                                    requires_grad: bool,
                                    field_manager: Optional[FieldManager] = None):
        assert not self._finished, "Try to register intermediate field after .finish()"
        assert ti_dtype is not None, "dtype cannot be None"
        assert isinstance(ti_dtype, TiDataType)
        assert name is not None, "name cannot be None"
        assert name not in self.seals, "name registered"
        assert requires_grad is not None, "requires_grad cannot be None when registering an intermediate field"
        assert not any(map(lambda d: d == -1, dims)), \
            "Dim = -1 is not allowed when registering intermediate fields but only registering input tensors"
        seal = Seal(ti_dtype, *dims,
                    field_manager=field_manager,
                    requires_grad=requires_grad,
                    name=name)
        self.intermediate_field_placeholders.append(seal)
        self.seals[name] = seal
        return self

    def register_kernel(self, kernel: Callable, arg_names: List[str], name: Optional[str] = None):
        assert not self._finished, "Try to register kernel after .finish()"
        assert is_kernel(kernel), "Passed function is not a Taichi kernel"
        assert autofill_kernel_name_available(kernel) or name is not None, \
            "kernel has no __name__, please update your Taichi or specify its name"
        assert all(map(lambda x: isinstance(x, str), arg_names)), "arg_names must be strings"
        not_registered_names = list(filter(lambda x: x not in self.seals, arg_names))
        assert len(not_registered_names) == 0, f"Some names are not registered: {not_registered_names}"
        seals = list(map(lambda x: self.seals[x], arg_names))
        kernel_bundle = TubeKernelBundle(kernel, name, seals)
        self.kernel_bundles.append(kernel_bundle)
        return self

    def forward(self, *input_tensors: torch.Tensor):
        return TubeFunc.apply(self, *input_tensors)


def unify_and_concretize_shapes(tensor_shapes: List[Tuple[int, ...]],
                                input_placeholders: List[Seal],
                                intermediate_fields: List[Seal],
                                output_placeholders: List[Seal]) \
        -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]], List[Tuple[int, ...]]]:
    # TODO: test
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
    return concrete_input_dims, concrete_intermediate_dims, concrete_output_dims


class TubeFunc(torch.autograd.Function):

    @staticmethod
    def concretize(device: torch.device,
                   fields_builder: ti.FieldsBuilder,
                   needs_grad: bool,
                   concrete_shape: Tuple[int, ...],
                   seal: Seal) -> Tuple[Seal, ConcreteField]:
        concrete_field = seal.concretize(concrete_shape, fields_builder, device, needs_grad)
        return seal, concrete_field

    @staticmethod
    def forward(ctx, tube: Tube, *input_tensors: torch.Tensor):
        assert len(input_tensors) == len(tube.input_placeholders)
        device = input_tensors[0].device
        for t in input_tensors:
            assert t.device == device, f"Tensors not on the same device"

        fb = ti.FieldsBuilder()
        input_tensor_shapes = list(map(lambda x: x.shape, input_tensors))
        concrete_input_shapes, concrete_intermediate_shapes, concrete_output_shapes = unify_and_concretize_shapes(
            input_tensor_shapes, tube.input_placeholders,
            tube.intermediate_field_placeholders,
            tube.output_placeholders)
        input_field_concretizer = partial(TubeFunc.concretize, device, fb)
        input_concrete_fields: List[Tuple[Seal, ConcreteField]] = list(map(input_field_concretizer,
                                                                           map(lambda x: x.requires_grad,
                                                                               input_tensors),
                                                                           concrete_input_shapes,
                                                                           tube.input_placeholders))
        other_field_concretize = partial(TubeFunc.concretize, device, fb, None)
        intermediate_concrete_fields: List[Tuple[Seal, ConcreteField]] = list(
            map(other_field_concretize, concrete_intermediate_shapes, tube.intermediate_field_placeholders))
        output_concrete_fields: List[Tuple[Seal, ConcreteField]] = list(
            map(other_field_concretize, concrete_output_shapes, tube.output_placeholders))
        seal_name_to_concrete_fields = {
            seal.name: concrete_field
            for seal, concrete_field in input_concrete_fields + intermediate_concrete_fields + output_concrete_fields
        }

        snode = fb.finalize()
        for _, field in intermediate_concrete_fields + output_concrete_fields:
            field.clear_field()

        for tensor, (_, concrete_input_field) in zip(input_tensors, input_concrete_fields):
            concrete_input_field.from_tensor(tensor)

        for kernel_bundle in tube.kernel_bundles:
            kernel_bundle.forward(seal_name_to_concrete_fields)

        output_tensors = tuple(ocf.to_tensor().requires_grad_(s.requires_grad) for s, ocf in output_concrete_fields)

        ctx.input_concrete_fields = input_concrete_fields
        ctx.intermediate_concrete_fields = intermediate_concrete_fields
        ctx.output_concrete_fields = output_concrete_fields
        ctx.seal_name_to_concrete_fields = seal_name_to_concrete_fields
        ctx.kernel_bundles = tube.kernel_bundles
        ctx.snode = snode
        ctx.mark_non_differentiable(*filter(lambda x: not x.requires_grad, output_tensors))
        if len(output_tensors) == 1:
            return output_tensors[0]
        else:
            return output_tensors

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> Any:
        for grad_tensor, (_, output_concrete_field) in zip(grad_outputs, ctx.output_concrete_fields):
            if output_concrete_field.requires_grad:
                output_concrete_field.grad_from_tensor(grad_tensor)

        for _, field in ctx.intermediate_concrete_fields + ctx.input_concrete_fields:
            field.clear_grad()

        for kernel_bundle in reversed(ctx.kernel_bundles):
            kernel_bundle.backward(ctx.seal_name_to_concrete_fields)

        gradient_tensors = [None]
        for _, input_concrete_field in ctx.input_concrete_fields:
            if input_concrete_field.requires_grad:
                gradient_tensors.append(input_concrete_field.grad_to_tensor())
            else:
                gradient_tensors.append(None)
        ctx.snode.destroy()
        return tuple(gradient_tensors)
