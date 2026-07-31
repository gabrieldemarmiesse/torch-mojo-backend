"""Compile-time operation/dtype gates and the specialized-module ABI."""

from std.python.bindings import PythonModuleBuilder
from std.python._cpython import (
    PyCFunction,
    PyCFunctionFast,
    PyCFunctionWithKeywords,
)
from std.sys.defines import get_defined_string

comptime _OP = get_defined_string["OP", ""]()
comptime _DTYPE_ARG_0 = get_defined_string["DTYPE_ARG_0", ""]()
comptime _DTYPE_ARG_1 = get_defined_string["DTYPE_ARG_1", ""]()
comptime _DTYPE_ARG_2 = get_defined_string["DTYPE_ARG_2", ""]()
comptime _DTYPE_ARG_3 = get_defined_string["DTYPE_ARG_3", ""]()
comptime _DTYPE_ARG_4 = get_defined_string["DTYPE_ARG_4", ""]()
comptime _DTYPE_ARG_5 = get_defined_string["DTYPE_ARG_5", ""]()
comptime _DTYPE_ARG_6 = get_defined_string["DTYPE_ARG_6", ""]()
comptime _DTYPE_ARG_7 = get_defined_string["DTYPE_ARG_7", ""]()
comptime _DTYPE_ARG_8 = get_defined_string["DTYPE_ARG_8", ""]()
comptime _DTYPE_ARG_9 = get_defined_string["DTYPE_ARG_9", ""]()
comptime _DTYPE_ARG_10 = get_defined_string["DTYPE_ARG_10", ""]()
comptime _DTYPE_ARG_11 = get_defined_string["DTYPE_ARG_11", ""]()
comptime _DTYPE_ARG_12 = get_defined_string["DTYPE_ARG_12", ""]()
comptime _DTYPE_ARG_13 = get_defined_string["DTYPE_ARG_13", ""]()
comptime _DTYPE_ARG_14 = get_defined_string["DTYPE_ARG_14", ""]()
comptime _DTYPE_ARG_15 = get_defined_string["DTYPE_ARG_15", ""]()
comptime _DTYPE_OUT = get_defined_string["DTYPE_OUT", ""]()
comptime _DTYPE_OUT_0 = get_defined_string["DTYPE_OUT_0", ""]()
comptime _DTYPE_OUT_1 = get_defined_string["DTYPE_OUT_1", ""]()
comptime _DTYPE_OUT_2 = get_defined_string["DTYPE_OUT_2", ""]()
comptime _DTYPE_OUT_3 = get_defined_string["DTYPE_OUT_3", ""]()


@always_inline
def _op_on[name: StaticString]() -> Bool:
    """Whether ``name`` is the sole implementation selected for this build."""
    comptime if _OP == name:
        return True
    else:
        return False


@always_inline
def _dtype_arg_on[index: Int, dt: DType]() -> Bool:
    """Whether ``dt`` is the exact dtype of tensor argument ``index``."""
    comptime name = String(dt)
    comptime if index == 0:
        return _DTYPE_ARG_0 == name
    elif index == 1:
        return _DTYPE_ARG_1 == name
    elif index == 2:
        return _DTYPE_ARG_2 == name
    elif index == 3:
        return _DTYPE_ARG_3 == name
    elif index == 4:
        return _DTYPE_ARG_4 == name
    elif index == 5:
        return _DTYPE_ARG_5 == name
    elif index == 6:
        return _DTYPE_ARG_6 == name
    elif index == 7:
        return _DTYPE_ARG_7 == name
    elif index == 8:
        return _DTYPE_ARG_8 == name
    elif index == 9:
        return _DTYPE_ARG_9 == name
    elif index == 10:
        return _DTYPE_ARG_10 == name
    elif index == 11:
        return _DTYPE_ARG_11 == name
    elif index == 12:
        return _DTYPE_ARG_12 == name
    elif index == 13:
        return _DTYPE_ARG_13 == name
    elif index == 14:
        return _DTYPE_ARG_14 == name
    elif index == 15:
        return _DTYPE_ARG_15 == name
    else:
        return False


@always_inline
def _dtype_out_on[index: Int, dt: DType]() -> Bool:
    """Whether ``dt`` is the exact dtype of output tensor ``index``."""
    comptime name = String(dt)
    comptime if index == 0:
        # DTYPE_OUT is retained as the canonical spelling for one-output ops.
        return _DTYPE_OUT == name or _DTYPE_OUT_0 == name
    elif index == 1:
        return _DTYPE_OUT_1 == name
    elif index == 2:
        return _DTYPE_OUT_2 == name
    elif index == 3:
        return _DTYPE_OUT_3 == name
    else:
        return False


@always_inline
def _dtype_arg_abi_on[index: Int, dt: DType]() -> Bool:
    """Exact argument gate with bool payloads represented as uint8 storage."""
    comptime if _dtype_arg_on[index, dt]():
        return True
    elif dt == DType.uint8 and _dtype_arg_on[index, DType.bool]():
        return True
    else:
        return False


@always_inline
def _dtype_arg_width_on[index: Int, bits: Int]() -> Bool:
    """Whether an argument's logical dtype has the selected storage width."""
    comptime if bits == 8:
        return (
            _dtype_arg_on[index, DType.bool]()
            or _dtype_arg_on[index, DType.uint8]()
            or _dtype_arg_on[index, DType.int8]()
        )
    elif bits == 16:
        return (
            _dtype_arg_on[index, DType.float16]()
            or _dtype_arg_on[index, DType.bfloat16]()
            or _dtype_arg_on[index, DType.uint16]()
            or _dtype_arg_on[index, DType.int16]()
        )
    elif bits == 32:
        return (
            _dtype_arg_on[index, DType.float32]()
            or _dtype_arg_on[index, DType.uint32]()
            or _dtype_arg_on[index, DType.int32]()
        )
    elif bits == 64:
        return (
            _dtype_arg_on[index, DType.float64]()
            or _dtype_arg_on[index, DType.uint64]()
            or _dtype_arg_on[index, DType.int64]()
        )
    else:
        return False


def _register_call(
    mut builder: PythonModuleBuilder,
    function: PyCFunction,
    _operation_name: StaticString,
    docstring: StaticString = "",
):
    """Register the one callable exposed by a specialized extension module."""
    builder.def_py_c_function(function, "call", docstring)


def _register_call(
    mut builder: PythonModuleBuilder,
    function: PyCFunctionWithKeywords,
    _operation_name: StaticString,
    docstring: StaticString = "",
):
    """Register the one callable exposed by a specialized extension module."""
    builder.def_py_c_function(function, "call", docstring)


def _register_call(
    mut builder: PythonModuleBuilder,
    function: PyCFunctionFast,
    _operation_name: StaticString,
    docstring: StaticString = "",
):
    """Register the one callable exposed by a specialized extension module."""
    builder.def_py_c_function(function, "call", docstring)
