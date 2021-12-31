import taichi as ti
import torch
from taichi.lang.impl import axes
from typing import Optional, Callable, Union, Tuple
from taichi import to_taichi_type
from taichi._lib.core.taichi_core import DataType as TiDataType


class Seal:
    class ConcreteField:
        def __init__(self,
                     dtype: TiDataType,
                     n: int,
                     m: int,
                     concrete_shape: Tuple[int, ...],
                     constructor: Callable):
            assert all(map(lambda x: isinstance(x, int), concrete_shape))
            if constructor is not None:
                field, snode_handle = constructor(n, m, concrete_shape)
            else:
                fb = ti.FieldsBuilder()
                if n != 0:
                    if m != 0:
                        field = ti.Matrix.field(n=n, m=m, shape=concrete_shape, dtype=dtype)
                    else:
                        field = ti.Vector.field(n=n, shape=concrete_shape, dtype=dtype)
                else:
                    assert m == 0
                    field = ti.field(dtype=dtype)
                fb.dense(axes(range(len(concrete_shape))), concrete_shape).place(field)
                snode_handle = fb.finalize()
            assert field is not None and snode_handle is not None
            self.field = field
            self.snode_handle = snode_handle

        def __del__(self):
            self.snode_handle.destroy()

    def __init__(self, *dims: int,
                 n: int = 0, m: int = 0,
                 dtype: Union[TiDataType, torch.dtype] = None,
                 constructor: Optional[Callable] = None):
        # basic checking
        assert n is not None and m is not None
        if n == 0:
            assert m == 0, f"Invalid (n,m) combination = ({n}, {m})"

        # validate dims
        batch_dim_started = True
        for i in dims:
            if i == 0:
                raise Exception(f"Dimension cannot be 0, got {dims}")
            if not batch_dim_started and i is None:
                raise Exception("You can only specify batch dimensions in the leading dimensions")
            if batch_dim_started and i is not None:
                batch_dim_started = False

        self.dtype = to_taichi_type(dtype) if dtype is not None else dtype
        self.constructor = constructor
        self.n = n
        self.m = m
        self.dims = dims
        expect_tensor_shape = dims
        if n != 0:
            expect_tensor_shape += (n,)
        if m != 0:
            expect_tensor_shape += (m,)
        self.expect_tensor_shape = dims
        self.concrete_field = None
        self.snode_handle = None

    def concretize(self, concrete_shape: Tuple[int, ...], n: int, m: int, dtype: TiDataType):
        return Seal.ConcreteField(dtype, n, m, concrete_shape, self.constructor)


class Tube:
    # TODO
    pass
