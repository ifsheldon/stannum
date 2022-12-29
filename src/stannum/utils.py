import sys
from typing import Union, Callable

from taichi import __version__ as ti_version
from taichi.lang.kernel_impl import Kernel
from taichi.lang.matrix import MatrixField
from taichi.lang.field import ScalarField

if ti_version >= (1, 1, 0):
    from taichi._lib.core.taichi_python import DataType as TiDataType
else:
    from taichi._lib.core.taichi_core import DataType as TiDataType
import torch
import taichi as ti

# Before Taichi 0.9.1, memory of fields is not initialized after creation, causing some undefined behaviors; fixed in 0.9.1
need_auto_clearing_fields = ti_version < (0, 9, 1)


def to_taichi_type(dt):
    """Convert numpy or torch data type to its counterpart in taichi.

    Args:
        dt (DataType): The desired data type to convert.

    Returns:
        DataType: The counterpart data type in taichi.

    """
    if type(dt) == TiDataType:
        return dt

    if dt == torch.float32:
        return ti.f32
    if dt == torch.float64:
        return ti.f64
    if dt == torch.int32:
        return ti.i32
    if dt == torch.int64:
        return ti.i64
    if dt == torch.int8:
        return ti.i8
    if dt == torch.int16:
        return ti.i16
    if dt == torch.uint8:
        return ti.u8
    if dt == torch.float16:
        return ti.f16

    raise AssertionError(f"Not support type {dt}")


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def check_field_needs_grad(field: Union[MatrixField, ScalarField], needs_grad: Union[None, bool]):
    """
    Checks if the Taichi in use is new enough to support automatic grad configuration based on field.snode.needs_grad
    :param field: a Taichi field
    :param needs_grad: boolean or None
    :return: boolean, whether a field needs gradients
    """
    is_legacy_taichi = ti_version < (0, 7, 26)
    if needs_grad is None:
        if is_legacy_taichi:
            raise Exception(
                f"You are using legacy Taichi (v{ti_version[0]}.{ti_version[1]}.{ti_version[2]} < v0.7.26), "
                f"you need to specify needs_grad yourself when registering a field")
        else:
            snode_ptr = field.snode.ptr
            if hasattr(snode_ptr, "has_grad"):
                return snode_ptr.has_grad()
            elif hasattr(snode_ptr, "has_adjoint"):
                return snode_ptr.has_adjoint()
            else:
                raise Exception("Oops! Seems Taichi APIs changed again, "
                                "check [Upstream breaking change tracker](github.com/ifsheldon/stannum/issues/11). "
                                "Or, file an issue please.")
    return needs_grad


def autofill_kernel_name_available(kernel: Callable):
    """
    check if the taichi implementation have func.__name__ in a @ti.kernel method of a @ti.data_oriented class
    @param kernel: a @ti.kernel function/method
    @return: if support autofill kernel name
    """
    is_legacy_taichi = ti_version <= (0, 7, 26)
    return not is_legacy_taichi or hasattr(kernel, "__name__")


def is_kernel(kernel):
    return hasattr(kernel, "_adjoint") and isinstance(kernel._adjoint, Kernel)
