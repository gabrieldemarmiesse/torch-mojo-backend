"""Every OpInfo in PyTorch's `op_db`, executed on the mojo device and compared
against the same call on the CPU.

What comes from OpInfo rather than from us:

* the sample inputs (`sample_inputs_func`) and, with `--reference-inputs`, the
  larger `reference_inputs_func` corpus of awkward shapes, non-contiguous
  layouts, broadcasts and scalar overloads;
* which of a sample's tensors belong on the device and which ATen requires on
  the CPU, read back from where the sample generator itself puts them when it
  targets a device (`_opinfo_placements`);
* the dtypes each operator is expected to support, per `op.supported_dtypes()`
  — which for an out-of-tree backend resolves to `op.dtypes`, the CPU set,
  since `dtypesIf` is keyed by backend name and no OpInfo declares one for
  "mojo";
* the accuracy bar: `op.precisionOverride` when the operator declares one,
  otherwise `torch.testing.assert_close`'s dtype defaults, applied through
  `TestCase.assertEqual`.  We never invent a tolerance;
* the corner cases: `error_inputs_func`, i.e. the inputs PyTorch asserts must
  RAISE, and the exception type and message it requires.

An operator we have not implemented raises NotImplementedError from the
dispatch layer, and that is a FAILURE here like any wrong answer, unless the
case is listed in `known_unsupported.py`.  Absence is declared there, node by
node, never read off the exception: an implementation that could excuse itself
by the exception it raised would erase the difference between "absent" and
"broken", which is the whole point of this suite.  A declared case still runs
and must still fail (reported xfail); when it starts passing the suite fails
and says which entry to delete.
"""

from __future__ import annotations

import functools
import unittest
from collections.abc import Callable

import known_unsupported
import pytest
import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.opinfo.core import OpInfo, SampleInput

from torch_mojo_backend.testing import assert_close_fp64_anchored

# The dtypes worth exercising on an accelerator backend.  Deliberately not the
# full OpInfo set: float64 is absent on some GPUs we target and complex is not
# implemented at all, so including them would report a backend-wide gap once
# per operator instead of once.
_DTYPES = (torch.float32, torch.bfloat16, torch.float16, torch.int64, torch.bool)


def _to_cpu(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list | tuple):
        return type(value)(_to_cpu(v) for v in value)
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    return value


def _run_declared_unsupported(reason: str, run: Callable[[], None]):
    """Run a case `known_unsupported.py` declares cannot pass, and hold it to it.

    xfail-strict semantics, and the strictness is the entire reason a
    declaration is safe.  A list that goes on muting a case after the case
    starts working would be worse than the exception-sniffing it replaces: a
    permanent mute nobody is ever told to remove.  So the case runs exactly as
    an undeclared one does, any failure is reported as an expected one, and a
    PASS is turned into a loud failure naming the entry to delete.

    A skip is not a failure and is not converted into one: a case the harness
    itself declined to compare (no sample inputs for this dtype, or OpInfo
    saying the output may legitimately differ across devices) never ran, so it
    is evidence neither for nor against the declaration.
    """
    try:
        run()
    except unittest.SkipTest:
        raise
    except Exception:  # noqa: BLE001 - any failure is the declared outcome
        pytest.xfail(reason)
    raise AssertionError(
        f"This case is {reason}, and it now PASSES. Delete that entry: "
        "support is the ABSENCE of an entry, and a declaration that outlives "
        "the gap it describes silently mutes a working case."
    )


def _declining_rather_than_rejecting(
    exc: BaseException, required: type[BaseException] | tuple[type[BaseException], ...]
) -> bool:
    """Whether `exc` is us declining the operator, not us rejecting the input.

    `NotImplementedError` IS a `RuntimeError` subclass, and `RuntimeError` is
    what most `error_inputs_func` corpora declare (`Exception`, which matches
    anything, is the second most common).  So an operator we never implemented
    satisfies `assertRaises` by not existing, and `test_errors_match` would
    credit "correctly rejects this malformed call" to an operator that rejects
    every call there is -- the false pass this suite is least able to afford,
    since the entire point of the test is that silently accepting a bad call is
    worse than not running it.  Declining is therefore never the required
    rejection, unless NotImplementedError is what OpInfo asked for (nothing in
    op_db does today, but the rule should not depend on that).
    """
    if not isinstance(exc, NotImplementedError):
        return False
    demanded = required if isinstance(required, tuple) else (required,)
    return not any(issubclass(kind, NotImplementedError) for kind in demanded)


def _tensor_specs(
    sample: SampleInput,
) -> list[tuple[torch.Size, torch.dtype, torch.device]]:
    """Shape, dtype and device of every tensor in `sample`, in the order
    `SampleInput.transform` visits them.

    `transform` is also what moves the sample to the device below, so walking
    with it is what makes two walks line up position by position.
    """
    specs: list[tuple[torch.Size, torch.dtype, torch.device]] = []

    def record(value: object) -> object:
        if isinstance(value, torch.Tensor):
            specs.append((value.shape, value.dtype, value.device))
        return value

    sample.transform(record)
    return specs


def _opinfo_placements(
    op: OpInfo, dtype: torch.dtype
) -> list[list[tuple[torch.Size, torch.dtype, torch.device]]] | None:
    """Where OpInfo itself puts each sample-input tensor when it builds this
    operator's samples for a device -- one entry per tensor, per sample.

    Not every tensor in a sample is data.  Some are metadata that ATen
    requires on the CPU whatever device the operand lives on:
    `tensor_split`'s `tensor_indices_or_sections` is checked outright
    (`aten/src/ATen/native/TensorShape.cpp`, `tensor_split`), and `narrow`'s
    `start`, `logsumexp`'s `dim` and advanced indexing's index tensors are
    conventionally left there too.  `sample_inputs_func` already encodes this,
    by building such arguments with a plain `torch.tensor(...)` while the
    operand goes through `make_tensor(device=device)`.  So asking a generator
    to target a device and reading back where each tensor landed IS the
    per-argument device rule, straight from OpInfo -- no table of ours to
    maintain and, crucially, nothing to learn from an exception after the
    fact.  Moving every tensor found in a sample and then recognising ATen's
    complaint by its wording was the previous answer here; it reported a
    harness artifact as an operator skip, and silently became a spurious
    failure the moment upstream reworded the message.

    The device asked for is `meta`: it allocates nothing and runs no kernel,
    so it cannot fail for the reason samples are built on the CPU in the first
    place (a backend missing an op that `make_tensor` reaches), and it does
    not need the backend under test to exist at all.  Only `.device` is read
    from the result.

    None means the placement is not observable and the caller should move
    everything, as this harness always did: the generator raised on meta
    (data-dependent construction -- `nonzero`, `bincount`, `searchsorted` and
    ~100 others), or yielded a different number of samples there (`sort`,
    `argsort` and `to` branch on the device), so position no longer identifies
    an argument.
    """
    rng_state = torch.get_rng_state()
    try:
        samples = list(op.sample_inputs("meta", dtype, requires_grad=False))
    except Exception:  # noqa: BLE001 - any failure here just means "unobservable"
        return None
    finally:
        # Meta construction fills nothing, so today no generator draws from the
        # default RNG here (checked across op_db: sample values are byte-identical
        # with and without this probe).  Rewind anyway: one generator hardcoding a
        # `torch.randn(...)` would otherwise shift every later sample's values and
        # quietly move tolerance-borderline comparisons suite-wide, for a probe
        # whose only output is a device.
        torch.set_rng_state(rng_state)
    return [_tensor_specs(sample) for sample in samples]


def _to_device(
    sample: SampleInput,
    device: str,
    placement: list[tuple[torch.Size, torch.dtype, torch.device]] | None,
) -> SampleInput:
    """`sample`, built on the CPU, with every tensor moved to `device` except
    the ones `placement` says OpInfo keeps on the CPU.

    `placement` is only trusted when its tensors line up with this sample's
    one for one, same shapes and same dtypes; otherwise the two builds
    diverged and position does not identify an argument, so everything moves.
    """
    if placement is not None:
        specs = _tensor_specs(sample)
        if [s[:2] for s in specs] != [p[:2] for p in placement]:
            placement = None
    keep_on_cpu = (
        iter([p[2].type == "cpu" for p in placement]) if placement is not None else None
    )

    def move(value: object) -> object:
        # transform() also visits torch.dtype values, which have no device.
        if not isinstance(value, torch.Tensor):
            return value
        if keep_on_cpu is not None and next(keep_on_cpu):
            return value
        return value.to(device)

    return sample.transform(move)


def _cross_device_comparison_skip_reason(op: OpInfo, dtype: torch.dtype) -> str | None:
    """None, or why `test_matches_cpu` should not compare this op's output.

    OpInfo already flags operators whose output is legitimately allowed to
    differ across devices by skipping `TestCommon.test_compare_cpu`,
    upstream's own CPU-vs-other-device consistency test: uninitialized
    memory for the `empty` family, a per-device RNG stream for `dropout`
    and `multi_head_attention_forward`. `native_batch_norm` is a sharper
    case: with `training=False` ATen's CPU kernel returns
    `save_mean`/`save_invstd` as empty `(0,)` tensors
    (`aten/src/ATen/native/Normalization.cpp`, `batch_norm_cpu`), while its
    CUDA kernel always returns them populated with `running_mean` and
    `rsqrt(running_var + eps)` (`aten/src/ATen/native/cuda/Normalization.cu`,
    `batch_norm_cuda_out`) -- required by `native_batch_norm_backward` so a
    frozen (eval-mode) BatchNorm still produces gradients. Upstream has
    known about this CPU/CUDA split since 2022 (pytorch/pytorch#85960) and,
    rather than pick a side, encodes it as an unconditional
    `DecorateInfo(expectedFailure, 'TestCommon', 'test_compare_cpu')` on the
    op's own OpInfo entry: CPU and an accelerator are not required to agree
    here. `test_matches_cpu` is the same comparison upstream's
    `test_compare_cpu` makes, just against a different accelerator, so it
    honors the same verdict instead of hand-maintaining a duplicate list of
    "known-divergent" ops that would drift out of sync with upstream's own.

    Not every `test_compare_cpu` skip upstream carries is this principled --
    e.g. `nn.functional.conv3d` skips it unconditionally with only "break
    slow tests" as a comment, not a documented semantic split. Honoring
    those too trades a small amount of value-comparison coverage for the
    same reason: a hand-picked subset would itself be a drift-prone list,
    and re-deriving "principled vs. merely convenient" from a skip that
    carries no machine-readable reason is not something this function can
    do reliably.
    """
    for skip in op.skips:
        # op.skips is not guaranteed to be homogeneous: some OpInfo entries
        # (e.g. linalg.solve_triangular) mix in a bare decorator function
        # (skipCPUIfNoLapack) alongside proper DecorateInfo instances.
        if getattr(skip, "test_name", None) != "test_compare_cpu":
            continue
        if skip.cls_name not in (None, "TestCommon"):
            continue
        if skip.device_type not in (None, "privateuse1"):
            continue
        if skip.dtypes is not None and dtype not in skip.dtypes:
            continue
        if callable(skip.active_if):
            # DecorateInfo.is_active() would call this with the concrete
            # test's param_kwargs, which we don't have here; treating a
            # callable as unconditionally active (its truthiness, not its
            # result) would be wrong, so decline it rather than guess.
            continue
        if not skip.active_if:
            continue
        return (
            "OpInfo skips TestCommon.test_compare_cpu for this op "
            "(output not expected to match across devices)"
        )
    return None


# Nodes compared through `assert_close_fp64_anchored` instead of the default
# bar: fp32 reductions whose result depends on summation order, so two correct
# kernels differ by more than rtol 1.3e-6 / atol 1e-5 on a few elements, and
# WHICH elements depends on the CPU's SIMD width (this node passed or failed
# on GitHub's runners depending on the machine drawn). torch's own fp32
# conv2d lands up to 4.6e-5 from the float64 answer on these very samples.
# Kept to the nodes that have shown it; not a general policy.
_FP64_ANCHORED: frozenset[tuple[str, torch.dtype]] = frozenset(
    {
        ("nn_functional_conv2d", torch.float32),
        # rsub with alpha on float16: CUDA's (and our) sub kernel computes
        # other - alpha * self as one fma in float; CPU torch rounds
        # alpha * self to float16 first, one or two ulps farther from the
        # float64 answer (sample alpha=-3.125: CPU 2.4453, CUDA 2.4414,
        # exact 2.4421).
        ("rsub", torch.float16),
    }
)

# Per-accelerator extensions to `_FP64_ANCHORED`, keyed like
# `known_unsupported._ACCELERATOR_DELTAS` (see `known_unsupported.accelerator_key()`).
# The base set above holds one node measured on GitHub's CPU runners; an
# accelerator that produces its own last-ulp differences records them here
# instead of widening the base set for everyone.
#
# gfx942 (MI300A, ROCm 6.4.3), measured 2026-09-14. The regeneration run
# (14 failed / 1112 passed / 236 skipped / 1424 xfailed, 0 operators absent)
# flagged 14 (op, dtype) nodes as last-ulp mismatches; running them anchored
# confirmed 6 as exactly that -- ordinary rounding/reduction-order
# differences, no farther from float64 than torch's own result:
#   bmm f32: 1/250 elements, abs 1.44e-5 vs 1e-5 allowed, rel 1.19e-5 vs 1.3e-6
#     (sample (10,5,10)@(10,10,5)).
#   addr f16: 1/50 elements, abs 3.9e-3, rel 1.045e-3 vs 1e-3.
#   instance_norm bf16, f16 and conv2d bf16, f16: 1-7 elements, one bf16/f16
#     ulp each.
# The other 8 were anchored once `assert_close_fp64_anchored` anchored the
# finite elements of a sample that also holds a masked -inf or a NaN and
# survived an empty sample (see the entries below); `pow` float32 was a real
# precision gap until float32 pow went through float64 (logic_ops.mojo).
_FP64_ANCHORED_BY_ACCELERATOR: dict[str, frozenset[tuple[str, torch.dtype]]] = {
    # M4: validate these CPU vector/scalar rounding differences against the
    # same float64 oracle used for the corresponding CUDA/HIP cases.
    "4-metal4": frozenset(
        {("addr", torch.float16), ("sub", torch.float16)}
        | {
            (op, dtype)
            for op in (
                "log_softmax",
                "masked_log_softmax",
                "nn_functional_batch_norm",
                "nn_functional_conv2d",
                "nn_functional_instance_norm",
            )
            for dtype in (torch.bfloat16, torch.float16)
        }
    ),
    # sm_90a (H100), measured 2026-09-14 after the base tables were regenerated
    # for the native backend: the same last-ulp class as gfx942's below (a
    # reduction-order or one-ulp difference against CPU torch, no farther from
    # float64 than torch's own result), which the absence-based tables cannot
    # express and CPU-only CI never sees.
    "sm_90a": frozenset(
        {
            ("__rpow__", torch.float32),
            ("addr", torch.bfloat16),
            ("addr", torch.float16),
            ("bmm", torch.float32),
            ("log_softmax", torch.bfloat16),
            ("log_softmax", torch.float16),
            ("masked_log_softmax", torch.bfloat16),
            ("masked_log_softmax", torch.float16),
            ("nn_functional_batch_norm", torch.bfloat16),
            ("nn_functional_batch_norm", torch.float16),
            ("nn_functional_conv2d", torch.bfloat16),
            ("nn_functional_conv2d", torch.float16),
            ("nn_functional_instance_norm", torch.bfloat16),
            ("nn_functional_instance_norm", torch.float16),
            ("pow", torch.float32),
        }
    ),
    "gfx942": frozenset(
        {
            ("bmm", torch.float32),
            ("addr", torch.float16),
            ("nn_functional_instance_norm", torch.bfloat16),
            ("nn_functional_instance_norm", torch.float16),
            ("nn_functional_conv2d", torch.bfloat16),
            ("nn_functional_conv2d", torch.float16),
            # The same one-ulp / summation-order class, anchorable only once
            # `assert_close_fp64_anchored` anchored the finite elements of a
            # sample that also holds a masked -inf or a NaN, and survived an
            # empty sample: __rpow__ rel 1.4e-6 on one element (abs 38 on
            # 1.4e7); masked_log_softmax and log_softmax one bf16/f16 ulp
            # (3.8e-3 / 4.3e-4) where the CPU result is exactly 0; batch_norm
            # one ulp on 1 to 7 of 125 elements.
            ("__rpow__", torch.float32),
            ("log_softmax", torch.bfloat16),
            ("log_softmax", torch.float16),
            ("masked_log_softmax", torch.bfloat16),
            ("masked_log_softmax", torch.float16),
            ("nn_functional_batch_norm", torch.bfloat16),
            ("nn_functional_batch_norm", torch.float16),
        }
    ),
}

# Merged once at collection time: the base set plus this accelerator's own
# extension, if it has one.
_FP64_ANCHORED = _FP64_ANCHORED | _FP64_ANCHORED_BY_ACCELERATOR.get(
    known_unsupported.accelerator_key(), frozenset()
)


def _to_float64(sample: SampleInput) -> SampleInput:
    return sample.transform(lambda t: t.double() if t.is_floating_point() else t)


class TestOpInfoConformance(TestCase):
    """One test per (operator, dtype), driven entirely by OpInfo metadata."""

    @ops(op_db, allowed_dtypes=_DTYPES)
    def test_matches_cpu(self, device: str, dtype: torch.dtype, op: OpInfo):
        """Same operator, same inputs, mojo vs CPU, at OpInfo's own tolerance.

        Samples are built on the CPU and moved, never built on the device.
        `sample_inputs(device=...)` constructs its tensors THERE, so a backend
        missing any op that `make_tensor` reaches fails during input
        construction and reports as if the operator under test were broken --
        which is a property of the harness, not of the operator.  Moving also
        makes both legs read bit-identical inputs.

        Which tensors to move is OpInfo's call, not ours: `_opinfo_placements`
        reads back where the operator's own sample generator puts each tensor
        when it targets a device, so arguments ATen requires on the CPU stay
        there instead of being moved and rejected.

        Whether this operator and dtype are expected to work at all is read
        from `known_unsupported.py` BEFORE anything runs, so the operator never
        gets the chance to launder its own failure into a skip.
        """
        skip_reason = _cross_device_comparison_skip_reason(op, dtype)
        if skip_reason is not None:
            self.skipTest(skip_reason)
        run = functools.partial(self._compare_with_cpu, device, dtype, op)
        reason = known_unsupported.declared_unsupported(
            "test_matches_cpu", op.formatted_name, dtype
        )
        if reason is None:
            run()
        else:
            _run_declared_unsupported(reason, run)

    def _compare_with_cpu(self, device: str, dtype: torch.dtype, op: OpInfo):
        """One `test_matches_cpu` case: every sample, device leg vs CPU leg."""
        placements = _opinfo_placements(op, dtype)
        checked = 0
        for index, sample in enumerate(
            op.sample_inputs("cpu", dtype, requires_grad=False)
        ):
            placement = None
            if placements is not None and index < len(placements):
                placement = placements[index]
            moved = _to_device(sample, device, placement)
            actual = op(moved.input, *moved.args, **moved.kwargs)
            expected = op(sample.input, *sample.args, **sample.kwargs)
            if (
                (op.formatted_name, dtype) in _FP64_ANCHORED
                and isinstance(actual, torch.Tensor)
                and isinstance(expected, torch.Tensor)
            ):
                exact = _to_float64(sample)
                reference = op(exact.input, *exact.args, **exact.kwargs)
                assert isinstance(reference, torch.Tensor)
                cpu_actual = _to_cpu(actual)
                assert isinstance(cpu_actual, torch.Tensor)
                assert_close_fp64_anchored(cpu_actual, expected, reference)
            else:
                # assertEqual carries the OpInfo precisionOverride for this
                # dtype when the operator declares one; otherwise assert_close
                # defaults.
                self.assertEqual(_to_cpu(actual), expected, exact_dtype=True)
            checked += 1
        if checked == 0:
            self.skipTest("OpInfo produced no sample inputs for this dtype")

    @ops(
        [op for op in op_db if op.error_inputs_func is not None],
        allowed_dtypes=(torch.float32,),
    )
    def test_errors_match(self, device: str, dtype: torch.dtype, op: OpInfo):
        """The inputs PyTorch says must raise, must raise here too.

        A backend that silently accepts a malformed call is a worse failure
        than one that cannot run it at all, and only OpInfo knows which calls
        those are per operator.

        An operator that cannot even be reached -- because we do not implement
        it, or because the error input is built out of a dtype the device does
        not have -- is declared in `known_unsupported.py` under this test's own
        name, and read before anything runs.  The two tests are listed
        separately because they ask different questions: an operator can be
        absent from one and present in the other, and one dtype's error corpus
        can be unbuildable while its samples run.
        """
        run = functools.partial(self._check_error_inputs, device, op)
        reason = known_unsupported.declared_unsupported(
            "test_errors_match", op.formatted_name, dtype
        )
        if reason is None:
            run()
        else:
            _run_declared_unsupported(reason, run)

    def _check_error_inputs(self, device: str, op: OpInfo):
        """One `test_errors_match` case: every error input OpInfo declares."""
        checked = 0
        for error_input in op.error_inputs(device):
            sample = error_input.sample_input
            # Nothing that raises `unittest.SkipTest` may run inside a
            # `with self.assertRaises(error_input.error_type):` block: a
            # sufficiently broad error_type (plain `Exception`, which every
            # `unittest.SkipTest` also is) would catch it as if it were the
            # expected exception, silently turning a real skip into a false
            # pass. So classify the raised/returned outcome first, and only
            # reach for assertRaises afterwards, scoped to the one check
            # that needs its swallow-on-match behavior (see below).
            try:
                out = op(sample.input, *sample.args, **sample.kwargs)
            except Exception as exc:  # noqa: BLE001 - triaging is the point
                if _declining_rather_than_rejecting(exc, error_input.error_type):
                    raise
                if not isinstance(exc, error_input.error_type):
                    raise
            else:
                # Some dunder ops (e.g. __rmod__) called directly, rather
                # than through operator syntax, correctly decline by
                # returning the `NotImplemented` sentinel instead of raising.
                # Upstream's own test_ops.py::test_errors accepts exactly
                # that outcome, via this identical assertFalse nested inside
                # `assertRaises(error_type)`: nesting it only here (not
                # around the skipTest above, which the original bug did)
                # still fails for real on a mismatch or on a call that
                # returned an ordinary value without raising anything.
                with self.assertRaises(error_input.error_type):
                    self.assertFalse(isinstance(out, type(NotImplemented)))
            checked += 1
        if checked == 0:
            self.skipTest("OpInfo declared no error inputs for this operator")


instantiate_device_type_tests(
    TestOpInfoConformance, globals(), only_for=("privateuse1",)
)


if __name__ == "__main__":
    run_tests()
