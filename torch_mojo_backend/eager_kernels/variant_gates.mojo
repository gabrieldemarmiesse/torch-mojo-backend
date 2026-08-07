"""Compile-time operation/dtype gates and the specialized-module ABI."""

from std.python.bindings import PythonModuleBuilder
from std.python._cpython import PyCFunctionFast
from std.sys.defines import get_defined_string

comptime _OP = get_defined_string["OP", ""]()


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
    # The define name is built at compile time, so there is no index ceiling:
    # an argument whose define was never passed reads back as "" and gates off.
    comptime defined = get_defined_string["DTYPE_ARG_" + String(index), ""]()
    return defined == String(dt)


@always_inline
def _dtype_out_on[index: Int, dt: DType]() -> Bool:
    """Whether ``dt`` is the exact dtype of output tensor ``index``."""
    comptime name = String(dt)
    comptime if index == 0:
        # DTYPE_OUT is retained as the canonical spelling for one-output ops.
        return (
            get_defined_string["DTYPE_OUT", ""]() == name
            or get_defined_string["DTYPE_OUT_0", ""]() == name
        )
    else:
        return get_defined_string["DTYPE_OUT_" + String(index), ""]() == name


@always_inline
def _dtype_supported[dtypes: List[DType], index: Int = 0](d: DType) -> Bool:
    """Whether ``d`` is in ``dtypes`` AND the compiled-in dtype of tensor
    argument ``index`` — the shared form of every bridge's support-check
    ladder (``var supported = False; comptime for dt ...``).

    Lives in this file on purpose: the define-name scanner
    (``eager_kernels._scan_define_names``) skips the gate library and instead
    recognizes ``_dtype_supported[...]`` call sites as reads of
    ``DTYPE_ARG_<index>``, so a bridge whose only argument gate is this check
    still keeps that define in its cache key and compile line.
    """
    comptime for dt in dtypes:
        comptime if _dtype_arg_on[index, dt]():
            if d == dt:
                return True
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
    """Whether an argument's logical dtype has the selected storage width.

    Only the width selects generated code here, so a caller that specializes
    exclusively through this gate may pass the canonical unsigned
    representative of the width (uint8/uint16/uint32/uint64) as
    ``DTYPE_ARG_<index>`` and share one build across every dtype of that
    width, instead of compiling a byte-identical .so per dtype name.
    """
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
    function: PyCFunctionFast,
    docstring: StaticString = "",
):
    """Register the one callable exposed by a specialized extension module.

    Every dispatcher in this package is a `METH_FASTCALL` entry point, and the
    operation it serves is already named by the enclosing
    `comptime if _op_on["..."]()` gate, so neither a second name argument nor
    the `PyCFunction`/`PyCFunctionWithKeywords` shapes are needed here.
    """
    builder.def_py_c_function(function, "call", docstring)
