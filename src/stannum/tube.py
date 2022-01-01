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

from .utils import is_kernel, autofill_kernel_name_available


class FieldManager(ABC):

    @abstractmethod
    def construct_field(self, concrete_tensor_shape: Tuple[int, ...], needs_grad: bool) \
            -> Tuple[SNodeTree, Union[ScalarField, MatrixField]]:
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

    def __init__(self, dtype: TiDataType, complex_dtype: bool, device: torch.device):
        self.dtype = dtype
        self.complex_dtype = complex_dtype
        self.device = device

    def construct_field(self, concrete_tensor_shape: Tuple[int, ...], needs_grad: bool) -> Tuple[
        SNodeTree, Union[ScalarField, MatrixField]]:
        if self.complex_dtype:
            field = ti.Vector.field(2, dtype=self.dtype, needs_grad=needs_grad)
        else:
            field = ti.field(self.dtype, needs_grad=needs_grad)

        fb = ti.FieldsBuilder()
        fb.dense(axes(range(len(concrete_tensor_shape))), concrete_tensor_shape).place(field)
        snode_handle = fb.finalize()
        return snode_handle, field

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


class Seal:
    class ConcreteField:

        def __init__(self,
                     dtype: TiDataType,
                     concrete_tensor_shape: Tuple[int, ...],
                     field_manager: FieldManager,
                     complex_dtype: bool,
                     requires_grad: bool,
                     device: torch.device,
                     name: str):
            assert all(map(lambda x: isinstance(x, int), concrete_tensor_shape))
            if field_manager is None:
                field_manager = DefaultFieldManager(dtype, complex_dtype, device)
            snode_handle, field = field_manager.construct_field(concrete_tensor_shape, requires_grad)
            assert field is not None and snode_handle is not None
            self.complex_dtype = complex_dtype
            self.field = field
            self.snode_handle = snode_handle
            self.device = device
            self.name = name
            self.field_manager = field_manager

        def to_tensor(self) -> torch.Tensor:
            return self.field_manager.to_tensor(self.field)

        def grad_to_tensor(self) -> torch.Tensor:
            return self.field_manager.grad_to_tensor(self.field.grad)

        def from_tensor(self, tensor):
            self.field_manager.from_tensor(self.field, tensor)

        def grad_from_tensor(self, tensor):
            self.field_manager.grad_from_tensor(self.field, tensor)

        def __del__(self):
            self.snode_handle.destroy()

    def __init__(self, dtype: Union[TiDataType, torch.dtype],
                 *dims: int,
                 field_manager: Optional[FieldManager] = None,
                 requires_grad: Optional[bool] = None,
                 name: Optional[str] = None):
        assert dtype is not None, "dtype must not be None"
        # validate dims
        if dims[0] is None:
            assert len(dims) >= 2, "Dimensions must have one that's not batch dimension"
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
        self.dtype = to_taichi_type(dtype) if dtype is not None else dtype
        self.field_manager = field_manager
        self.dims = dims
        self.name = name
        self.requires_grad = requires_grad
        self.expect_tensor_shape = dims
        self.concrete_field = None
        self.snode_handle = None

    def concretize(self, concrete_shape: Tuple[int, ...], device: torch.device, needs_grad: bool):
        return Seal.ConcreteField(self.dtype, concrete_shape, self.field_manager, self.complex_dtype,
                                  needs_grad if self.requires_grad is None else self.requires_grad,
                                  device, self.name)


class TubeKernelBundle:
    def __init__(self, kernel: Callable, name: Optional[str], seals: List[Seal]):
        self.kernel = kernel
        self.name: str = kernel.__name__ if name is None else name
        self.seals = seals

    def try_concretize(self, *tensors: torch.Tensor) -> List[Seal.ConcreteField]:
        # TODO: method to unify shapes and do batching
        pass


class TubeConfigs:
    def __init__(self):
        # TODO
        pass


class TubeFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        # TODO
        pass

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        # TODO
        pass


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
        self.tube_configs: TubeConfigs = None
        self._finished: bool = False

    def finish(self):
        if self._finished:
            return self
        assert len(self.input_placeholders) > 0, "Must register at least 1 input field"
        assert len(self.output_placeholders) > 0, "Must register at least 1 output field"
        assert len(self.kernel_bundles) > 0, "Must register at least 1 kernel"
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
        assert len(dims) > 0, "Input tensor must have at least 1D"
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
        assert len(dims) > 0, "Output tensor must have at least 1D"
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
        assert len(dims) > 0, "Intermediate field must have at least 1D"
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
        # TODO: nn.Module forward
        pass
