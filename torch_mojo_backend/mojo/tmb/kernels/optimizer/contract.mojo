"""Runtime metadata contract for the eager fused-optimizer kernels."""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder


# Descriptors per launch: bounds ONE by-value launch argument (8 Ints each),
# not the list -- a longer TensorList becomes several launches.
comptime FUSED_OPT_DESC_CAP = 32
comptime FUSED_OPT_THREADS = 256


struct FusedOptDesc(
    DevicePassable,
    ImplicitlyCopyable,
    TrivialRegisterPassable,
):
    """One parameter of the list: the addresses of its param, grad, up to
    three optimizer states and its step scalar, its numel, and the running
    prefix sum of chunk counts. Unused state slots stay zero and are never
    dereferenced (which states an algorithm reads is a compile-time fact)."""

    comptime device_type: AnyType = Self

    var param_addr: Int
    var grad_addr: Int
    var state0_addr: Int
    var state1_addr: Int
    var state2_addr: Int
    var step_addr: Int
    var numel: Int
    var chunk_end: Int

    def __init__(
        out self,
        param_addr: Int,
        grad_addr: Int,
        state0_addr: Int,
        state1_addr: Int,
        state2_addr: Int,
        step_addr: Int,
        numel: Int,
        chunk_end: Int,
    ):
        self.param_addr = param_addr
        self.grad_addr = grad_addr
        self.state0_addr = state0_addr
        self.state1_addr = state1_addr
        self.state2_addr = state2_addr
        self.step_addr = step_addr
        self.numel = numel
        self.chunk_end = chunk_end

    def _to_device_type(
        self,
        mut encoder: Some[DeviceTypeEncoder],
        target: MutOpaquePointer[_],
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "FusedOptDesc"


@always_inline
def empty_fused_opt_desc() -> FusedOptDesc:
    return FusedOptDesc(0, 0, 0, 0, 0, 0, 0, 0)
