"""Compile-time operation and dtype gates for eager-kernel variants."""

from std.sys.defines import get_defined_string

comptime _TMB_OPS = get_defined_string["TMB_OPS", "all"]()
comptime _TMB_DTYPES = get_defined_string["TMB_DTYPES", "all"]()


@always_inline
def _tmb_csv_has(csv: StaticString, name: String) -> Bool:
    """Comptime-foldable membership test in a comma-separated define."""
    if csv == "all":
        return True
    var padded = String(",") + String(csv) + ","
    return padded.find(String(",") + name + ",") != -1


@always_inline
def _op_on[name: StaticString]() -> Bool:
    """Whether the named Python entry point belongs in this variant."""
    comptime if _tmb_csv_has(_TMB_OPS, String(name)):
        return True
    else:
        return False


@always_inline
def _dt_on[dt: DType]() -> Bool:
    """Whether the dtype should be instantiated in this variant."""
    comptime if _tmb_csv_has(_TMB_DTYPES, String(dt)):
        return True
    else:
        return False
