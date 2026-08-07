"""Runtime metadata contract for eager multi-tensor optimizer kernels."""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder


comptime ADAMW_DESC_CAP = 32
comptime ADAMW_CHUNK_ELEMENTS = 65_536
comptime ADAMW_THREADS = 256


struct AdamWDesc(
    DevicePassable,
    ImplicitlyCopyable,
    TrivialRegisterPassable,
):
    comptime device_type: AnyType = Self

    var param_addr: Int
    var grad_addr: Int
    var exp_avg_addr: Int
    var exp_avg_sq_addr: Int
    var max_exp_avg_sq_addr: Int
    var step_addr: Int
    var numel: Int
    var chunk_end: Int

    def __init__(
        out self,
        param_addr: Int,
        grad_addr: Int,
        exp_avg_addr: Int,
        exp_avg_sq_addr: Int,
        max_exp_avg_sq_addr: Int,
        step_addr: Int,
        numel: Int,
        chunk_end: Int,
    ):
        self.param_addr = param_addr
        self.grad_addr = grad_addr
        self.exp_avg_addr = exp_avg_addr
        self.exp_avg_sq_addr = exp_avg_sq_addr
        self.max_exp_avg_sq_addr = max_exp_avg_sq_addr
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
        return "AdamWDesc"


@always_inline
def empty_adamw_desc() -> AdamWDesc:
    return AdamWDesc(0, 0, 0, 0, 0, 0, 0, 0)
