from abc import ABC, abstractmethod
from typing import Tuple, Union, Dict, Optional
import taichi as ti
from taichi.lang.field import ScalarField
from taichi.lang.matrix import MatrixField
import torch

if hasattr(ti, "snode"):
    from taichi.snode.snode_tree import SNodeTree
elif hasattr(ti, "_snode"):
    from taichi._snode.snode_tree import SNodeTree
else:
    from .utils import __ti_version
    raise Exception(f"Unable to import SNodeTree, Taichi version = {__ti_version}")


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


class SNode:
    """
    Pythonic wrapper around SNodeTree that can auto recycle memory
    """

    def __init__(self, snode: SNodeTree):
        self.snode = snode
        self.destroyed = False

    def destroy(self):
        if not self.destroyed:
            self.snode.destroy()
            self.destroyed = True

    def __del__(self):
        self.destroy()


DimID = Union[str, int]


class DimEnum:
    """
    Enum with payload(i.e., dim_id)
    """
    ANY_ID = 0
    BATCH_ID = 1
    MATCH_ID = 2

    def __init__(self, id: int, dim_id: Optional[DimID] = None):
        self.id = id
        self.dim_id = dim_id

    def __eq__(self, other):
        if not isinstance(other, DimEnum):
            return False
        if self.id != other.id:
            return False
        else:
            if self.id == DimEnum.MATCH_ID:
                return self.dim_id == other.dim_id
            else:
                return True

    def __hash__(self):
        if self.id != DimEnum.MATCH_ID:
            return hash((self.id, str(self)))
        else:
            return hash((self.id, self.dim_id))

    def __str__(self):
        if self.id == DimEnum.ANY_ID:
            return "DimEnum.Any"
        elif self.id == DimEnum.BATCH_ID:
            return "DimEnum.Batch"
        else:
            return f"DimEnum.Match(dim_id = {self.dim_id})"


AnyDim = DimEnum(DimEnum.ANY_ID, None)
BatchDim = DimEnum(DimEnum.BATCH_ID, None)


def MatchDim(dim_id: DimID):
    # legacy Python cannot handle isinstance(dim_id, DimID)
    assert isinstance(dim_id, (str, int)), f"Unsupported dim_id type of {type(dim_id)}"
    return DimEnum(DimEnum.MATCH_ID, dim_id)


DimOption = Union[int, DimEnum]


class DimensionCalculator(ABC):
    """
    An interface for implementing a calculator that hints Tube how to construct fields
    """

    @abstractmethod
    def calc_dimension(self,
                       field_name: str,
                       input_dimensions: Dict[str, Tuple[DimOption, ...]],
                       input_tensor_shapes: Dict[str, Tuple[int, ...]]) -> Tuple[DimOption, ...]:
        """
        Calculate dimensions for a output/intermediate field

        @param field_name: the name of the field for which the dimensions are calculated
        @param input_dimensions: the dict mapping names of input fields to input fields
        @param input_tensor_shapes: the dict mapping names of input fields to
        shapes of input tensors that correspond to input fields
        """
        pass
