import taichi as ti
import torch
from taichi.lang.impl import axes
from typing import Optional, Callable, Union, Tuple, List, Iterable, Dict, Any
from .utils import to_taichi_type, need_auto_clearing_fields, TiDataType
from taichi.lang.field import ScalarField
from taichi.lang.matrix import MatrixField
from functools import partial

from .utils import is_kernel, autofill_kernel_name_available
from .auxiliary import FieldManager, SNode, DimensionCalculator, DimEnum, DimOption, DimID


class Tube(torch.nn.Module):
    """
    Self-managed Taichi-PyTorch adapter
    """

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
        super().__init__()
        self.input_placeholders: List[Seal] = []
        self.output_placeholders: List[Seal] = []
        self.intermediate_field_placeholders: List[Seal] = []
        self.seals: Dict[str, Seal] = {}
        self.device: Optional[torch.device] = device
        self._finished: bool = False
        self.batched: bool = False
        self.kernel_bundle_dict: Dict[str, TubeKernelBundle] = {}
        if not enable_backward:
            func = EagerTubeFunc
        else:
            if persistent_field:
                func = PersistentTubeFunc
            else:
                func = EagerTubeFunc
        self.func: torch.autograd.Function = func
        self.enable_backward: bool = enable_backward

    def register_input_tensor(self,
                              dims: Union[Tuple[DimOption], List[DimOption]],
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
        assert all(map(is_dim_option, dims)), f"dims must be all DimOptions, got {dims}"
        assert all(map(lambda x: x > 0,
                       filter(lambda x: isinstance(x, int), dims))), \
            f"all integer dims must be positive, got {dims}"
        seal = Seal(dtype, *dims,
                    field_manager=field_manager,
                    requires_grad=requires_grad,
                    name=name)
        seal.batched = len(dims) > 0 and is_batch_dim(dims[0])
        self.batched = self.batched or seal.batched
        self.input_placeholders.append(seal)
        self.seals[name] = seal
        return self

    def register_output_tensor(self,
                               dims_or_calc: Union[Tuple[DimOption], List[DimOption], Callable, DimensionCalculator],
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
        assert not self._finished, "Try to register output tensor after .finish()"
        assert dtype is not None, "dtype cannot be None"
        assert isinstance(dtype, torch.dtype)
        assert name is not None, "name cannot be None"
        assert name not in self.seals, "name registered"
        assert requires_grad is not None, "requires_grad cannot be None when registering an output tensor"
        if isinstance(dims_or_calc, (Callable, DimensionCalculator)):  # calculator
            seal = Seal(dtype,
                        shape_calc=dims_or_calc,
                        field_manager=field_manager,
                        requires_grad=requires_grad,
                        name=name)
        elif isinstance(dims_or_calc, (Tuple, List)):  # explicit dims
            assert all(map(is_dim_option, dims_or_calc)), f"dims must be all DimOptions, got {dims_or_calc}"
            assert not any(map(is_any_dim, dims_or_calc)), \
                "Dim = Any is not allowed when registering output tensors but only registering input tensors"
            assert all(map(lambda x: x > 0,
                           filter(lambda x: isinstance(x, int), dims_or_calc))), \
                f"all integer dims must be positive, got {dims_or_calc}"
            seal = Seal(dtype, *dims_or_calc,
                        field_manager=field_manager,
                        requires_grad=requires_grad,
                        name=name)
            seal.batched = len(dims_or_calc) > 0 and is_batch_dim(dims_or_calc[0])
        else:
            raise Exception(f"Invalid dims_or_calc of type {type(dims_or_calc)}")

        self.output_placeholders.append(seal)
        self.seals[name] = seal
        return self

    def register_intermediate_field(self,
                                    dims_or_calc: Union[Tuple[DimOption], List[DimOption], Callable, DimensionCalculator],
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
        assert not self._finished, "Try to register intermediate field after .finish()"
        assert ti_dtype is not None, "dtype cannot be None"
        assert isinstance(ti_dtype, TiDataType)
        assert name is not None, "name cannot be None"
        assert name not in self.seals, "name registered"
        assert needs_grad is not None, "requires_grad cannot be None when registering an intermediate field"
        if isinstance(dims_or_calc, (Callable, DimensionCalculator)):  # calculator
            seal = Seal(ti_dtype,
                        shape_calc=dims_or_calc,
                        field_manager=field_manager,
                        requires_grad=needs_grad,
                        name=name)
        elif isinstance(dims_or_calc, (Tuple, List)):  # explicit dims
            assert all(map(is_dim_option, dims_or_calc)), f"dims must be all DimOptions, got {dims_or_calc}"
            assert not any(map(is_any_dim, dims_or_calc)), \
                "Dim = Any is not allowed when registering intermediate fields but only registering input tensors"
            assert all(map(lambda x: x > 0,
                           filter(lambda x: isinstance(x, int), dims_or_calc))), \
                f"all integer dims must be positive, got {dims_or_calc}"
            seal = Seal(ti_dtype, *dims_or_calc,
                        field_manager=field_manager,
                        requires_grad=needs_grad,
                        name=name)
            seal.batched = len(dims_or_calc) > 0 and is_batch_dim(dims_or_calc[0])
        else:
            raise Exception(f"Invalid dims_or_calc of type {type(dims_or_calc)}")
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
        old_kernel_bundle = self.kernel_bundle_dict[kernel_name]
        self.kernel_bundle_dict[kernel_name] = TubeKernelBundle(old_kernel_bundle.kernel,
                                                                old_kernel_bundle.name,
                                                                old_kernel_bundle.seals,
                                                                extra_args)

    def finish(self):
        """
        Finish all registrations
        """
        if self._finished:
            return self
        assert len(self.input_placeholders) > 0, "Must register at least 1 input field"
        assert len(self.output_placeholders) > 0, "Must register at least 1 output field"
        assert len(self.kernel_bundle_dict) > 0, "Must register at least 1 kernel"
        # MatchDim check
        match_dims = set()
        for ip in self.input_placeholders:
            for d in ip.dims:
                if is_match_dim(d):
                    match_dims.add(d.dim_id)

        for placeholder in self.intermediate_field_placeholders + self.output_placeholders:
            if placeholder.dims_calc is not None:  # dims are calculated dynamically
                continue
            for d in placeholder.dims:
                if is_match_dim(d) and d.dim_id not in match_dims:
                    raise Exception(f"Dimension={d} in {placeholder.name} is not registered in any input tensors")
        self._finished = True
        return self

    def forward(self, *input_tensors: torch.Tensor):
        return self.func.apply(self, list(self.kernel_bundle_dict.values()), *input_tensors)

    from taichi import __version__ as ti_version
    if ti_version < (1, 1, 3):
        import warnings
        warnings.warn(f"You are using Taichi = {ti_version[0]}.{ti_version[1]}.{ti_version[2]} "
                      f"older than the recommended Taichi = 1.1.3. "
                      f"Using Tube with old Taichi may suffer from performance downgrade. "
                      f"See Stannum issue #9 for more information.",
                      stacklevel=2)


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
    """
    Encapsulation of the information needed to construct a concrete Taichi field
    """

    def __init__(self, dtype: Union[TiDataType, torch.dtype],
                 *dims: DimOption,
                 shape_calc: Optional[Union[Callable, DimensionCalculator]] = None,
                 field_manager: Optional[FieldManager] = None,
                 requires_grad: Optional[bool] = None,
                 name: Optional[str] = None):
        assert dtype is not None, "dtype must not be None"
        if shape_calc is None:  # explicit dims
            # validate dims
            if len(dims) > 0:  # if it is not a scalar(dims is an empty tuple or list), further check
                if is_batch_dim(dims[0]):
                    for i in range(1, len(dims)):
                        assert not is_batch_dim(dims[i]), \
                            "Only the leading dimension can be None (i.e. the batch dimension)"
                else:
                    for d in dims:
                        assert not is_batch_dim(d), \
                            "Only the leading dimension can be None (i.e. the batch dimension)"

            for i in dims:
                assert i != 0, f"Dimension cannot be 0, got {dims}"

            self.dims: Optional[Tuple[DimOption, ...]] = dims
            self.dims_calc: Optional[Union[Callable, DimensionCalculator]] = None
        else:  # dim calculator
            self.dims: Optional[Tuple[DimOption, ...]] = None
            self.dims_calc: Optional[Union[Callable, DimensionCalculator]] = shape_calc

        self.complex_dtype = dtype == torch.cfloat or dtype == torch.cdouble
        if self.complex_dtype:
            dtype = ti.f32 if dtype == torch.cfloat else ti.f64
        self.dtype: TiDataType = to_taichi_type(dtype) if dtype is not None else dtype
        self.field_manager: FieldManager = field_manager
        self.batched: Optional[bool] = None
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

    def calc_dimensions(self,
                        input_dimensions: Dict[str, Tuple[DimOption, ...]],
                        input_tensor_shapes: Dict[str, Tuple[int, ...]]) -> Tuple[int, ...]:
        assert self.dims_calc is not None
        if isinstance(self.dims_calc, DimensionCalculator):
            dimensions = self.dims_calc.calc_dimension(self.name, input_dimensions, input_tensor_shapes)
        else:
            dimensions = self.dims_calc(self.name, input_dimensions, input_tensor_shapes)

        self.batched = len(dimensions) > 0 and dimensions[0] is None
        return dimensions


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


def is_dim_option(dim_opt: Any):
    return isinstance(dim_opt, (int, DimEnum))  # legacy python cannot deal with isinstance(d, DimOption)


def is_batch_dim(dim_opt: DimOption):
    return isinstance(dim_opt, DimEnum) and dim_opt.id == DimEnum.BATCH_ID


def is_match_dim(dim_opt: DimOption):
    return isinstance(dim_opt, DimEnum) and dim_opt.id == DimEnum.MATCH_ID


def is_any_dim(dim_opt: DimOption):
    return isinstance(dim_opt, DimEnum) and dim_opt.id == DimEnum.ANY_ID


def concretize(device: torch.device,
               fields_builder: ti.FieldsBuilder,
               needs_grad: bool,
               concrete_shape: Tuple[int, ...],
               seal: Seal) -> ConcreteField:
    concrete_field = seal.concretize(concrete_shape, fields_builder, device, needs_grad)
    return concrete_field


def select_tensor(seals: List[Seal], tensors: Tuple[torch.Tensor], batch_idx: int) -> List[torch.Tensor]:
    assert len(seals) == len(tensors)
    selected_tensors = []
    for seal, tensor in zip(seals, tensors):
        if seal.batched:
            selected_tensors.append(tensor[batch_idx])
        else:
            selected_tensors.append(tensor)
    return selected_tensors


def select_concrete_field(seals: List[Seal],
                          concrete_fields: List[Union[ConcreteField, List[ConcreteField]]],
                          batch_idx: int) -> List[ConcreteField]:
    assert len(seals) == len(concrete_fields)
    selected_concrete_fields = [
        concrete_field[batch_idx] if seal.batched else concrete_field
        for seal, concrete_field in zip(seals, concrete_fields)
    ]
    return selected_concrete_fields


def concretize_dimensions_and_unify(input_tensor_shapes: List[Tuple[int, ...]],
                                    input_placeholders: List[Seal],
                                    intermediate_fields: List[Seal],
                                    output_placeholders: List[Seal]) \
        -> Tuple[List[List[int]], List[List[int]], List[List[int]], Union[int, None]]:
    """
    Try to find out concrete numbers in dimension placeholders (like AnyDim, MatchDim, BatchDim)
    """
    input_dims = list(map(lambda x: x.dims, input_placeholders))

    # check dimensionality and batch nums
    batch_num = None
    for i, (tensor_shape, input_dim) in enumerate(zip(input_tensor_shapes, input_dims)):
        assert len(tensor_shape) == len(input_dim), \
            f"Dimensionality check failed, expecting the {i}th tensor to be {len(input_dim)}D, got {len(tensor_shape)}D"
        if len(input_dim) == 0:  # scalar, shape = ()
            continue
        elif is_batch_dim(input_dim[0]):  # shape = (Batch, ...)
            if batch_num is None:
                batch_num = tensor_shape[0]
            else:
                assert tensor_shape[0] == batch_num, f"Batch num of {i}th tensor not match, " \
                                                     f"expect: {batch_num}, got {tensor_shape[0]}"
        else:
            assert all(map(lambda shape, dim: (is_match_dim(dim) or is_any_dim(dim)) or shape == dim,
                           tensor_shape, input_dim)), \
                f"{i}th tensor dimensions not match, expect: {input_dim}, got {tensor_shape}"

    # fill in {Any, Match} dimensions and batch dimension
    concrete_input_dims: List[List[int]] = list(map(list, input_dims))
    dim_id_to_shape: Dict[DimID, int] = {}
    for idx, input_dim in enumerate(input_dims):
        for i, d in enumerate(input_dim):
            if is_batch_dim(d):
                concrete_input_dims[idx][i] = batch_num
            elif is_any_dim(d):
                concrete_input_dims[idx][i] = input_tensor_shapes[idx][i]
            elif is_match_dim(d):
                concrete_shape = input_tensor_shapes[idx][i]
                if d.dim_id in dim_id_to_shape:
                    assert dim_id_to_shape[d.dim_id] == concrete_shape, f"Dim = {d} not match"
                else:
                    dim_id_to_shape[d.dim_id] = concrete_shape
                concrete_input_dims[idx][i] = concrete_shape

    input_tensor_dimensions_dict = {}
    input_tensor_shapes_dict = {}
    for input_placeholder, input_tensor_shape in zip(input_placeholders, input_tensor_shapes):
        name = input_placeholder.name
        input_tensor_dimensions_dict[name] = input_placeholder.dims
        input_tensor_shapes_dict[name] = input_tensor_shape

    get_dims = lambda x: x.dims if x.dims is not None \
        else x.calc_dimensions(input_tensor_dimensions_dict, input_tensor_shapes_dict)

    output_dims: Union[List[Tuple[DimOption, ...], List[DimOption]]] = list(map(get_dims, output_placeholders))
    intermediate_dims: Union[List[Tuple[DimOption, ...], List[DimOption]]] = list(map(get_dims, intermediate_fields))
    # batching check
    if batch_num is not None:  # batching
        for seal in output_placeholders + intermediate_fields:
            assert seal.batched, \
                "The dimension 0 of registered output tensors or intermediate fields is not batched (not None) " \
                "but the input is batched. " \
                "If the input is batched (auto-batching is on), " \
                "the registered or calculated dimensions of registered output tensors or intermediate fields MUST be None" \
                "\nName of error intermediate field or output tensor is {}".format(seal.name)

    # calculate output and intermediate dims
    concrete_output_dims: List[List[int]] = list(map(list, output_dims))
    concrete_intermediate_dims: List[List[int]] = list(map(list, intermediate_dims))
    for idx, output_dim in enumerate(output_dims):
        for i, d in enumerate(output_dim):
            if is_batch_dim(d):
                concrete_output_dims[idx][i] = batch_num
            elif is_match_dim(d):
                concrete_output_dims[idx][i] = dim_id_to_shape[d.dim_id]
            elif is_any_dim(d):
                raise Exception("Dim = Any is not allowed when constructing output fields "
                                "but only registering input tensors. "
                                "If you are using a DimensionCalculator, check it. "
                                "Otherwise, this should not happen, then file a bug report")

    for idx, inter_dim in enumerate(intermediate_dims):
        for i, d in enumerate(inter_dim):
            if is_batch_dim(d):
                concrete_intermediate_dims[idx][i] = batch_num
            elif is_match_dim(d):
                concrete_intermediate_dims[idx][i] = dim_id_to_shape[d.dim_id]
            elif is_any_dim(d):
                raise Exception("Dim = Any is not allowed when constructing intermediate fields "
                                "but only registering input tensors. "
                                "If you are using a DimensionCalculator, check it. "
                                "Otherwise, this should not happen, then file a bug report")

    return concrete_input_dims, concrete_intermediate_dims, concrete_output_dims, batch_num


class EagerTubeFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tube: Tube, kernel_bundles: List[TubeKernelBundle], *input_tensors: torch.Tensor):
        assert len(input_tensors) == len(tube.input_placeholders)
        ctx.kernel_bundles = kernel_bundles
        input_seals = tube.input_placeholders
        output_seals = tube.output_placeholders
        intermediate_seals = tube.intermediate_field_placeholders
        if not tube.enable_backward:
            for i, t in enumerate(input_tensors):
                assert not t.requires_grad, f"enable_backward = False, " \
                                            f"so the {i}th of input tensors cannot require grad, detach it first"
        # basic checking
        if tube.device is None:
            device = input_tensors[0].device
            for t in input_tensors:
                assert t.device == device, f"Tensors not on the same device {device}"
        else:
            device = tube.device
            for t in input_tensors:
                assert t.device == device, f"Tensors not on the device {device}"

        input_tensor_shapes = [x.shape for x in input_tensors]
        concrete_input_shapes, concrete_intermediate_shapes, concrete_output_shapes, batch_num = concretize_dimensions_and_unify(
            input_tensor_shapes,
            input_seals, intermediate_seals, output_seals)

        fb = ti.FieldsBuilder()
        input_field_concretizer = partial(concretize, device, fb)
        other_field_concretizer = partial(concretize, device, fb, None)
        if batch_num is None:
            # concretize fields
            input_concrete_fields: List[Union[ConcreteField, List[ConcreteField]]] = list(
                map(input_field_concretizer,
                    [x.requires_grad for x in input_tensors],
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
            if need_auto_clearing_fields:
                # clear fields due to uninitialized memory in old Taichi
                for field in intermediate_concrete_fields + output_concrete_fields:
                    field.clear_field()

            # load tensor to field
            for tensor, concrete_input_field in zip(input_tensors, input_concrete_fields):
                concrete_input_field.from_tensor(tensor)

            # forward pass
            for kernel_bundle in kernel_bundles:
                kernel_bundle.forward(seal_name_to_concrete_fields)

            output_tensors = tuple(ocf.to_tensor().requires_grad_(s.requires_grad) for s, ocf in
                                   zip(output_seals, output_concrete_fields))
            saved_tensors = list(input_tensors)
            saved_tensors += [field.to_tensor() for field in intermediate_concrete_fields]
            saved_tensors += list(output_tensors)
        else:
            # concretize fields
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
            scf = select_concrete_field
            intermediate_tensor_batches = []
            output_tensor_batches = []
            for batch_idx in range(batch_num):
                # select fields given a batch_idx
                concrete_input_field_batch = scf(input_seals, input_concrete_fields, batch_idx)
                concrete_intermediate_field_batch = scf(intermediate_seals, intermediate_concrete_fields, batch_idx)
                concrete_output_field_batch = scf(output_seals, output_concrete_fields, batch_idx)
                seal_name_to_concrete_fields = {
                    seal.name: concrete_field
                    for seal, concrete_field in
                    zip(input_seals + intermediate_seals + output_seals,
                        concrete_input_field_batch + concrete_intermediate_field_batch + concrete_output_field_batch)
                }
                if need_auto_clearing_fields:
                    # clear fields due to uninitialized memory in old Taichi
                    for field in concrete_intermediate_field_batch + concrete_output_field_batch:
                        field.clear_field()
                input_tensor_batch = select_tensor(input_seals, input_tensors, batch_idx)
                # load tensor to fields
                for tensor, concrete_input_field in zip(input_tensor_batch, concrete_input_field_batch):
                    concrete_input_field.from_tensor(tensor)
                # forward pass
                for kernel_bundle in kernel_bundles:
                    kernel_bundle.forward(seal_name_to_concrete_fields)
                output_tensors = [ocf.to_tensor() for ocf in concrete_output_field_batch]
                output_tensor_batches.append(output_tensors)
                intermediate_tensors = [icf.to_tensor() for icf in concrete_intermediate_field_batch]
                intermediate_tensor_batches.append(intermediate_tensors)

            # stack tensors calculated per slice of a batch
            output_tensors = []
            for output_idx, output_seal in enumerate(output_seals):
                tensors = [output_tensor_batches[batch_idx][output_idx]
                           for batch_idx in range(batch_num)]
                output_tensors.append(torch.stack(tensors, dim=0).requires_grad_(output_seal.requires_grad))

            intermediate_tensors = []
            for intermediate_idx, intermediate_seal in enumerate(intermediate_seals):
                tensors = [output_tensor_batches[batch_idx][intermediate_idx]
                           for batch_idx in range(batch_num)]
                intermediate_tensors.append(torch.stack(tensors, dim=0).requires_grad_(intermediate_seal.requires_grad))

            output_tensors = tuple(output_tensors)
            saved_tensors = list(input_tensors)
            saved_tensors += intermediate_tensors
            saved_tensors += list(output_tensors)

        snode.destroy()
        ctx.batch_num = batch_num
        ctx.input_tensor_num = len(input_tensors)
        ctx.concrete_input_shapes = concrete_input_shapes
        ctx.concrete_intermediate_shapes = concrete_intermediate_shapes
        ctx.concrete_output_shapes = concrete_output_shapes
        ctx.save_for_backward(*saved_tensors)
        ctx.tube = tube
        ctx.mark_non_differentiable(*filter(lambda x: not x.requires_grad, output_tensors))
        if len(output_tensors) == 1:
            return output_tensors[0]
        else:
            return output_tensors

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> Any:
        tube = ctx.tube
        kernel_bundles = ctx.kernel_bundles
        assert tube.enable_backward, "Attempting to run backward computation when enable_backward = False"
        input_tensor_num = ctx.input_tensor_num
        all_tensors = ctx.saved_tensors
        input_tensors = all_tensors[:input_tensor_num]
        batch_num = ctx.batch_num
        concrete_input_shapes = ctx.concrete_input_shapes
        concrete_intermediate_shapes = ctx.concrete_intermediate_shapes
        concrete_output_shapes = ctx.concrete_output_shapes
        input_seals = tube.input_placeholders
        output_seals = tube.output_placeholders
        intermediate_seals = tube.intermediate_field_placeholders

        device = grad_outputs[0].device
        fb = ti.FieldsBuilder()
        input_field_concretizer = partial(concretize, device, fb)
        other_field_concretizer = partial(concretize, device, fb, None)
        if batch_num is None:
            # concretize fields
            input_concrete_fields: List[Union[ConcreteField, List[ConcreteField]]] = list(
                map(input_field_concretizer,
                    [x.requires_grad for x in input_tensors],
                    concrete_input_shapes,
                    input_seals))
            intermediate_concrete_fields: List[Union[ConcreteField, List[ConcreteField]]] = list(
                map(other_field_concretizer, concrete_intermediate_shapes, intermediate_seals))
            output_concrete_fields: List[Union[ConcreteField, List[ConcreteField]]] = list(
                map(other_field_concretizer, concrete_output_shapes, output_seals))
            snode = fb.finalize()

            # load tensor values to field
            for tensor, field in zip(all_tensors,
                                     input_concrete_fields + intermediate_concrete_fields + output_concrete_fields):
                field.from_tensor(tensor)

            # load grad values to field
            for grad_tensor, output_concrete_field in zip(grad_outputs, output_concrete_fields):
                if output_concrete_field.requires_grad:
                    output_concrete_field.grad_from_tensor(grad_tensor)

            seal_name_to_concrete_fields = {
                seal.name: concrete_field
                for seal, concrete_field in
                zip(input_seals + intermediate_seals + output_seals,
                    input_concrete_fields + intermediate_concrete_fields + output_concrete_fields)
            }
            if need_auto_clearing_fields:
                # clear grad field due to uninitialized memory in old Taichi
                for field in intermediate_concrete_fields + input_concrete_fields:
                    field.clear_grad()
            # backward pass
            for kernel_bundle in reversed(kernel_bundles):
                kernel_bundle.backward(seal_name_to_concrete_fields)

            gradient_tensors = [None, None]
            for input_concrete_field in input_concrete_fields:
                if input_concrete_field.requires_grad:
                    gradient_tensors.append(input_concrete_field.grad_to_tensor())
                else:
                    gradient_tensors.append(None)

            snode.destroy()
            return tuple(gradient_tensors)
        else:
            # concretize fields
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

            scf = select_concrete_field
            gradients = []
            for batch_idx in range(batch_num):
                # select fields given a batch_idx
                concrete_input_field_batch = scf(input_seals, input_concrete_fields, batch_idx)
                concrete_intermediate_field_batch = scf(intermediate_seals, intermediate_concrete_fields, batch_idx)
                concrete_output_field_batch = scf(output_seals, output_concrete_fields, batch_idx)
                tensor_batch = select_tensor(input_seals + intermediate_seals + output_seals, all_tensors, batch_idx)
                # load fields with tensor values
                for tensor, concrete_field in zip(tensor_batch,
                                                  concrete_input_field_batch + concrete_intermediate_field_batch + concrete_output_field_batch):
                    concrete_field.from_tensor(tensor)

                # load grad fields with grad values
                grad_output_batch = select_tensor(output_seals, grad_outputs, batch_idx)
                for grad_tensor, output_concrete_field in zip(grad_output_batch, concrete_output_field_batch):
                    if output_concrete_field.requires_grad:
                        output_concrete_field.grad_from_tensor(grad_tensor)

                seal_name_to_concrete_fields = {
                    seal.name: concrete_field
                    for seal, concrete_field in
                    zip(input_seals + intermediate_seals + output_seals,
                        concrete_input_field_batch + concrete_intermediate_field_batch + concrete_output_field_batch)
                }
                if need_auto_clearing_fields:
                    # clear grad fields due to uninitialized memory on Taichi
                    for field in concrete_intermediate_field_batch + concrete_input_field_batch:
                        field.clear_grad()
                else:
                    # clear grad fields due to mixing batched and non-batched inputs
                    for seal, field in zip(input_seals + intermediate_seals,
                                           concrete_input_field_batch + concrete_intermediate_field_batch):
                        if not seal.batched:
                            field.clear_grad()
                # backward pass
                for kernel_bundle in reversed(kernel_bundles):
                    kernel_bundle.backward(seal_name_to_concrete_fields)

                grad_tensor_batch = []
                for input_concrete_field in concrete_input_field_batch:
                    if input_concrete_field.requires_grad:
                        grad_tensor_batch.append(input_concrete_field.grad_to_tensor())
                    else:
                        grad_tensor_batch.append(None)
                gradients.append(grad_tensor_batch)

            input_grads = [None, None]
            for input_idx, input_seal in enumerate(input_seals):
                grad_per_input = [gradients[batch_idx][input_idx] for batch_idx in range(batch_num)]
                if any(map(lambda x: x is None, grad_per_input)):
                    input_grads.append(None)
                else:
                    if input_seal.batched:
                        input_grads.append(torch.stack(grad_per_input, dim=0))
                    else:
                        input_grads.append(torch.stack(grad_per_input, dim=0).sum(dim=0))

            snode.destroy()
            return tuple(input_grads)


class PersistentTubeFunc(torch.autograd.Function):

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
    def forward(ctx, tube: Tube, kernel_bundles: List[TubeKernelBundle], *input_tensors: torch.Tensor):
        assert len(input_tensors) == len(tube.input_placeholders)
        ctx.kernel_bundles = kernel_bundles
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

        input_tensor_shapes = [x.shape for x in input_tensors]
        concrete_input_shapes, concrete_intermediate_shapes, concrete_output_shapes, batch_num = concretize_dimensions_and_unify(
            input_tensor_shapes,
            input_seals, intermediate_seals, output_seals)
        fb = ti.FieldsBuilder()
        input_field_concretizer = partial(concretize, device, fb)
        other_field_concretizer = partial(concretize, device, fb, None)
        if batch_num is None:
            input_concrete_fields: List[Union[ConcreteField, List[ConcreteField]]] = list(
                map(input_field_concretizer,
                    [x.requires_grad for x in input_tensors],
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
            if need_auto_clearing_fields:
                # clear grad field due to uninitialized memory in old Taichi
                for field in intermediate_concrete_fields + output_concrete_fields:
                    field.clear_field()

            for tensor, concrete_input_field in zip(input_tensors, input_concrete_fields):
                concrete_input_field.from_tensor(tensor)

            for kernel_bundle in kernel_bundles:
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
            scf = select_concrete_field
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
                if need_auto_clearing_fields:
                    # clear grad field due to uninitialized memory in old Taichi
                    for field in concrete_intermediate_field_batch + concrete_output_field_batch:
                        field.clear_field()
                input_tensor_batch = select_tensor(input_seals, input_tensors, batch_idx)
                for tensor, concrete_input_field in zip(input_tensor_batch, concrete_input_field_batch):
                    concrete_input_field.from_tensor(tensor)
                for kernel_bundle in kernel_bundles:
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
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> Any:
        tube: Tube = ctx.tube
        kernel_bundles = ctx.kernel_bundles
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
            if need_auto_clearing_fields:
                # clear grad field due to uninitialized memory in old Taichi
                for field in intermediate_concrete_fields + input_concrete_fields:
                    field.clear_grad()
            for kernel_bundle in reversed(kernel_bundles):
                kernel_bundle.backward(seal_name_to_concrete_fields)

            gradient_tensors = [None, None]
            for input_concrete_field in input_concrete_fields:
                if input_concrete_field.requires_grad:
                    gradient_tensors.append(input_concrete_field.grad_to_tensor())
                else:
                    gradient_tensors.append(None)
            return tuple(gradient_tensors)
        else:
            scf = select_concrete_field
            gradients = []
            for batch_idx in range(batch_num):
                grad_output_batch = select_tensor(output_seals, grad_outputs, batch_idx)
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

                if need_auto_clearing_fields:
                    # clear grad fields due to uninitialized memory on Taichi
                    for field in intermediate_concrete_field_batch + input_concrete_field_batch:
                        field.clear_grad()
                else:
                    # clear grad fields due to mixing batched and non-batched inputs
                    for seal, field in zip(input_seals + intermediate_seals,
                                           input_concrete_field_batch + intermediate_concrete_field_batch):
                        if not seal.batched:
                            field.clear_grad()
                for kernel_bundle in reversed(kernel_bundles):
                    kernel_bundle.backward(seal_name_to_concrete_fields)
                grad_tensor_batch = []
                for input_concrete_field in input_concrete_field_batch:
                    if input_concrete_field.requires_grad:
                        grad_tensor_batch.append(input_concrete_field.grad_to_tensor())
                    else:
                        grad_tensor_batch.append(None)
                gradients.append(grad_tensor_batch)

            input_grads = [None, None]
            for input_idx, input_seal in enumerate(input_seals):
                grad_per_input = [gradients[batch_idx][input_idx] for batch_idx in range(batch_num)]
                if any(map(lambda x: x is None, grad_per_input)):
                    input_grads.append(None)
                else:
                    if input_seal.batched:
                        input_grads.append(torch.stack(grad_per_input, dim=0))
                    else:
                        input_grads.append(torch.stack(grad_per_input, dim=0).sum(dim=0))
            return tuple(input_grads)
