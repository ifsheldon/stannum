from abc import ABC, abstractmethod
from typing import Tuple, Union
import taichi as ti
from taichi import ScalarField, MatrixField
import torch
from taichi.snode.snode_tree import SNodeTree


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
