from std.os import abort
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr, Py_ssize_t
from std.python.bindings import PythonModuleBuilder
from std.utils.coord import Coord
from std.sys.info import has_apple_gpu_accelerator, size_of
from std.utils import IndexList

from op_utils import TensorHolder, TensorSpec


def ping() -> PythonObject:
    return PythonObject(42)


from std.memory import Pointer
from op_utils import _raw_int, _raw_ret_none, _spec_unsupported


def _probe_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _ = _raw_int(args[0])
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


@export
def PyInit_th_probe() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("th_probe")
        _ = (
            m.add_type[TensorHolder]("TensorHolder")
            .def_method[TensorHolder.data_ptr]("data_ptr")
            .def_method[TensorHolder.get_nbytes]("get_nbytes")
        )
        _ = m.add_type[TensorSpec]("TensorSpec")
        m.def_function[ping]("ping")
        m.def_py_c_function(_probe_dispatcher, "Probe", docstring="probe")
        return m.finalize()
    except e:
        abort(t"failed to create th_probe python module: {e}")
