from typing import Protocol, runtime_checkable

from max.experimental.tensor import Tensor as MaxEagerTensor
from max.graph import Dim, TensorValue

MaxTensor = TensorValue | MaxEagerTensor
Scalar = int | float | Dim
SymIntType = int | Dim


@runtime_checkable
class CountedCallable(Protocol):
    """A function wrapped with the test-only call-count instrumentation.

    `map_to` (aten_functions.py) wraps every graph-backend op with
    `functools.wraps` plus a `call_count` attribute so tests can assert an
    op actually ran (see `testing.CallChecker`; the native mojo device is
    counted by the C++ shim instead). `functools.wraps` preserves
    `__name__` but its stub doesn't know about the extra attribute, so the
    wrapped callable needs this structural type instead of plain `Callable`.
    """

    __name__: str
    call_count: int

    def __call__(self, *args: object, **kwargs: object) -> object: ...
