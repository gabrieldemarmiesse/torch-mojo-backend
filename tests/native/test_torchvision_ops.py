"""Torchvision detection kernels through public APIs on the mojo GPU."""

from __future__ import annotations

import atexit
import functools
import io
import math
import os
import select
import shutil
import struct
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from tests.native.conftest import skip_if_metal

vision = pytest.importorskip("torchvision")
DTYPES = (torch.float32, torch.float16, torch.float64)


def _dtype_supported(device: str, dtype: torch.dtype):
    if dtype == torch.float64:
        skip_if_metal(device, "Metal does not support float64")


def _boxes(count: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(731)
    starts = torch.rand(count, 2, generator=generator) * 20
    sizes = torch.rand(count, 2, generator=generator) * 30 + 1
    return torch.cat((starts, starts + sizes), dim=1).to(dtype)


def _rois(scale: float = 1.0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    result = torch.tensor(
        [
            [0, 1.25, 2.5, 18.75, 20.25],
            [1, -4, -3, 15, 17],
            [2, 45, 29, 58, 42],
            [0, -20, -15, -8, -4],
            [2, 60, 45, 71, 56],
            [1, 8, 8, 8, 8],
        ],
        dtype=dtype,
    )
    result[:, 1:] /= scale
    return result


def _roi_op(
    kind: str,
    x: torch.Tensor,
    rois: torch.Tensor,
    output: tuple[int, int] = (7, 5),
    scale: float = 1.0,
    sampling: int = 2,
    aligned: bool = False,
) -> torch.Tensor:
    if kind == "align":
        return vision.ops.roi_align(x, rois, output, scale, sampling, aligned=aligned)
    return vision.ops.roi_pool(x, rois, output, scale)


def _assert_close(got: torch.Tensor, want: torch.Tensor, dtype: torch.dtype):
    tolerance = {torch.float16: 2e-2, torch.float32: 3e-5, torch.float64: 2e-12}[dtype]
    torch.testing.assert_close(
        got.cpu(), want.to(dtype), rtol=tolerance, atol=tolerance
    )


def _reference_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONHOME", "PYTHONEXECUTABLE"}
    }


def _cuda_reference_candidates() -> tuple[str, ...]:
    candidates = [os.environ.get("TORCHVISION_CUDA_REFERENCE_PYTHON"), sys.executable]
    candidates.extend(shutil.which(name) for name in ("python", "python3"))
    for root in (Path.cwd(), Path.cwd().parent):
        for pattern in (".venv-cuda/bin/python", "torch_cu*/bin/python"):
            candidates.extend(str(path) for path in sorted(root.glob(pattern)))
    return tuple(dict.fromkeys(path for path in candidates if path))


@functools.cache
def _find_cuda_reference(candidates: tuple[str, ...]) -> tuple[str | None, str]:
    script = textwrap.dedent("""
        import sys
        try:
            import torch
            import torchvision
            if not torch.cuda.is_available():
                raise RuntimeError("torch.cuda.is_available() is false")
            for op in ("nms", "roi_align", "_roi_align_backward", "roi_pool",
                       "_roi_pool_backward", "ps_roi_align", "_ps_roi_align_backward",
                       "ps_roi_pool", "_ps_roi_pool_backward", "deform_conv2d",
                       "_deform_conv2d_backward"):
                if not torch._C._dispatch_has_kernel_for_dispatch_key(
                        "torchvision::" + op, "CUDA"):
                    raise RuntimeError("missing CUDA kernel: torchvision::" + op)
            torchvision.ops.nms(torch.zeros(1, 4, device="cuda"),
                                torch.ones(1, device="cuda"), 0.5)
            torch.cuda.synchronize()
        except Exception as error:
            print(type(error).__name__ + ": " + str(error))
            sys.exit(77)
    """)
    failures = []
    for interpreter in candidates:
        try:
            result = subprocess.run(
                [interpreter, "-c", script],
                capture_output=True,
                env=_reference_env(),
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            failures.append(f"{interpreter}: {error}")
            continue
        if result.returncode == 0:
            return interpreter, ""
        reason = (result.stdout + result.stderr).decode(errors="replace").strip()
        failures.append(f"{interpreter}: {reason or f'exit {result.returncode}'}")
    return None, "stock CUDA torchvision unavailable; tried " + "; ".join(failures)


_REFERENCE_WORKER = textwrap.dedent("""
    import io
    import struct
    import sys
    import traceback
    import torch
    import torchvision
    replies = sys.stdout.buffer
    sys.stdout = sys.stderr  # nothing but replies may reach the pipe
    requests = sys.stdin.buffer
    while True:
        head = requests.read(8)
        if len(head) < 8:
            break
        payload = requests.read(struct.unpack("<Q", head)[0])
        try:
            op, tensors, kwargs, grad, autocast = torch.load(
                io.BytesIO(payload), weights_only=True)
            inputs = tuple(
                t.detach().cuda().requires_grad_(t.requires_grad) for t in tensors)
            with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
                if op == "deform_conv2d" and len(inputs) == 5:
                    output = torchvision.ops.deform_conv2d(
                        *inputs[:4], mask=inputs[4], **kwargs)
                else:
                    output = getattr(torchvision.ops, op)(*inputs, **kwargs)
            if grad is not None:
                output.backward(grad.cuda())
            result = (output.detach().cpu(),) + tuple(
                t.grad.cpu() for t in inputs if t.requires_grad)
            stream = io.BytesIO()
            torch.save(result, stream)
            ok, body = 1, stream.getvalue()
        except BaseException:
            ok, body = 0, traceback.format_exc().encode()
        replies.write(struct.pack("<BQ", ok, len(body)) + body)
        replies.flush()
""")


class _ReferenceWorker:
    """One stock-CUDA interpreter answering every reference of the session.

    A process per reference paid torch's import and CUDA's initialization each
    time, about 2.5 s: 345 s of this file's 349 s on a warm CI run.
    """

    def __init__(self, interpreter: str):
        self.process = subprocess.Popen(
            [interpreter, "-c", _REFERENCE_WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=_reference_env(),
            bufsize=0,
        )
        atexit.register(self.close)

    def close(self):
        if self.process.poll() is None:
            assert self.process.stdin is not None
            self.process.stdin.close()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def _read(self, count: int, deadline: float) -> bytes:
        assert self.process.stdout is not None
        fd = self.process.stdout.fileno()
        chunks = []
        while count:
            ready, _, _ = select.select(
                [fd], [], [], max(0.0, deadline - time.monotonic())
            )
            assert ready, "the CUDA reference worker timed out"
            chunk = os.read(fd, min(count, 1 << 20))
            assert chunk, f"the CUDA reference worker died (exit {self.process.poll()})"
            chunks.append(chunk)
            count -= len(chunk)
        return b"".join(chunks)

    def call(self, payload: bytes, timeout: float = 180) -> bytes:
        assert self.process.stdin is not None
        self.process.stdin.write(struct.pack("<Q", len(payload)) + payload)
        deadline = time.monotonic() + timeout
        ok, size = struct.unpack("<BQ", self._read(9, deadline))
        body = self._read(size, deadline)
        assert ok, body.decode(errors="replace")
        return body


@functools.cache
def _reference_worker(interpreter: str) -> _ReferenceWorker:
    return _ReferenceWorker(interpreter)


def _cuda_reference(
    op: str,
    tensors: tuple[torch.Tensor, ...],
    kwargs: dict[str, int | float | bool | tuple[int, int]],
    grad: torch.Tensor | None = None,
    autocast: bool = False,
) -> tuple[torch.Tensor, ...]:
    interpreter, reason = _find_cuda_reference(_cuda_reference_candidates())
    if interpreter is None:
        pytest.skip(reason)
    payload = io.BytesIO()
    torch.save((op, tensors, kwargs, grad, autocast), payload)
    worker = _reference_worker(str(interpreter))
    try:
        body = worker.call(payload.getvalue())
    except BaseException:
        # A failed reference may have left a sticky CUDA error behind.
        worker.close()
        _reference_worker.cache_clear()
        raise
    return torch.load(io.BytesIO(body), weights_only=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["roi_pool", "ps_roi_pool"])
def test_roi_pool_rounding_neighbors(mojo_gpu: str, dtype: torch.dtype, kind: str):
    _dtype_supported(mojo_gpu, dtype)
    # Include the review's exact float32 predecessor of 0.5 and both signs.
    centers = torch.arange(-4, 5, dtype=dtype) + 0.5
    values = torch.stack(
        (
            torch.nextafter(centers, torch.full_like(centers, -math.inf)),
            centers,
            torch.nextafter(centers, torch.full_like(centers, math.inf)),
        )
    ).flatten()
    data = torch.tensor([[[[10, 1], [10, 1]]]], dtype=dtype)
    rois = torch.zeros(values.numel(), 5, dtype=dtype)
    rois[:, 1] = values
    rois[:, 3:] = 1
    reference = data.requires_grad_()
    grad = torch.ones(values.numel(), 1, 1, 1, dtype=dtype)
    expected, expected_grad = _cuda_reference(
        kind, (reference, rois), {"output_size": (1, 1)}, grad
    )
    ours = data.detach().to(mojo_gpu).requires_grad_()
    result = getattr(vision.ops, kind)(ours, rois.to(mojo_gpu), (1, 1))
    result.backward(grad.to(mojo_gpu))
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    assert ours.grad is not None
    torch.testing.assert_close(ours.grad.cpu(), expected_grad, rtol=0, atol=0)


def test_nms_half_intersection_rounding(mojo_gpu: str):
    boxes = torch.tensor(
        [[0, 0, 1, 1], [0.2498779296875, 0, 1.5, 1]], dtype=torch.float16
    )
    scores = torch.tensor([2, 1], dtype=torch.float16)
    (expected,) = _cuda_reference("nms", (boxes, scores), {"iou_threshold": 0.5})
    assert expected.tolist() == [0, 1]
    result = vision.ops.nms(boxes.to(mojo_gpu), scores.to(mojo_gpu), 0.5)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("autocast", [False, True])
def test_nms_threshold_sweep(mojo_gpu: str, dtype: torch.dtype, autocast: bool):
    _dtype_supported(mojo_gpu, dtype)
    boxes = torch.tensor([[0, 0, 2, 1], [1, 0, 3, 1]], dtype=dtype)
    scores = torch.tensor([2, 1], dtype=dtype)
    center = torch.tensor(1 / 3, dtype=torch.float32)
    thresholds = [
        math.nextafter(1 / 3, -math.inf),
        1 / 3,
        math.nextafter(1 / 3, math.inf),
        torch.nextafter(center, torch.tensor(-math.inf)).item(),
        center.item(),
        torch.nextafter(center, torch.tensor(math.inf)).item(),
    ]
    for threshold in thresholds:
        (expected,) = _cuda_reference(
            "nms", (boxes, scores), {"iou_threshold": threshold}, autocast=autocast
        )
        with torch.autocast("mojo", dtype=torch.float16, enabled=autocast):
            result = vision.ops.nms(boxes.to(mojo_gpu), scores.to(mojo_gpu), threshold)
        torch.testing.assert_close(
            result.cpu(), expected, rtol=0, atol=0, msg=f"threshold={threshold!r}"
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_roi_pool_integer_batch_bound(mojo_gpu: str, dtype: torch.dtype):
    _dtype_supported(mojo_gpu, dtype)
    data = torch.zeros(16777217, 1, 1, 1, dtype=dtype)
    data[-1] = 7
    data.requires_grad_()
    rois = torch.tensor([[16777216, 0, 0, 0, 0]], dtype=dtype)
    grad = torch.ones(1, 1, 1, 1, dtype=dtype)
    expected, expected_grad = _cuda_reference(
        "roi_pool", (data, rois), {"output_size": (1, 1)}, grad
    )
    ours = data.detach().to(mojo_gpu).requires_grad_()
    result = vision.ops.roi_pool(ours, rois.to(mojo_gpu), (1, 1))
    result.backward(grad.to(mojo_gpu))
    assert expected.item() == 7 and expected_grad[-1].item() == 1
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    assert ours.grad is not None
    torch.testing.assert_close(ours.grad.cpu(), expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
@pytest.mark.parametrize(
    "kind", ["roi_align", "roi_pool", "ps_roi_align", "ps_roi_pool"]
)
def test_roi_cuda_dtype_arithmetic(mojo_gpu: str, dtype: torch.dtype, kind: str):
    _dtype_supported(mojo_gpu, dtype)
    data = (
        (torch.arange(100, dtype=torch.float64).reshape(1, 4, 5, 5) / 37)
        .to(dtype)
        .requires_grad_()
    )
    rois = torch.tensor([[0, 0.4999, 0.7501, 3.3, 3.7]], dtype=dtype)
    kwargs = {"output_size": (2, 2), "spatial_scale": 0.73}
    if "align" in kind:
        kwargs["sampling_ratio"] = 1
    channels = 1 if kind.startswith("ps") else 4
    grad = torch.zeros(1, channels, 2, 2, dtype=dtype)
    grad[0, 0, 0, 0] = 0.3
    expected, expected_grad = _cuda_reference(kind, (data, rois), kwargs, grad)
    ours = data.detach().to(mojo_gpu).requires_grad_()
    result = getattr(vision.ops, kind)(ours, rois.to(mojo_gpu), **kwargs)
    result.backward(grad.to(mojo_gpu))
    tolerance = 0 if dtype == torch.float16 else 2e-15
    torch.testing.assert_close(result.cpu(), expected, rtol=tolerance, atol=tolerance)
    assert ours.grad is not None
    torch.testing.assert_close(
        ours.grad.cpu(), expected_grad, rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("count", [0, 1, 47, 3073])
@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_nms(mojo_gpu: str, dtype: torch.dtype, count: int, threshold: float):
    _dtype_supported(mojo_gpu, dtype)
    boxes = _boxes(count, dtype)
    scores = torch.rand(count, generator=torch.Generator().manual_seed(171)).to(dtype)
    if dtype == torch.float16:
        (want,) = _cuda_reference("nms", (boxes, scores), {"iou_threshold": threshold})
    else:
        want = vision.ops.nms(boxes, scores, threshold)
    got = vision.ops.nms(boxes.to(mojo_gpu), scores.to(mojo_gpu), threshold)
    assert got.dtype == torch.int64
    assert got.device.type == "mojo"
    torch.testing.assert_close(got.cpu(), want)


def test_cuda_reference_environment_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    interpreter = tmp_path / "stock-python"
    interpreter.touch()
    monkeypatch.setenv("TORCHVISION_CUDA_REFERENCE_PYTHON", str(interpreter))
    stream = io.BytesIO()
    expected = (torch.tensor([0]),)
    torch.save(expected, stream)

    run = Mock(return_value=subprocess.CompletedProcess([], 0, b"", b""))
    monkeypatch.setattr(subprocess, "run", run)
    worker = Mock()
    worker.return_value.call.return_value = stream.getvalue()
    monkeypatch.setattr(sys.modules[__name__], "_reference_worker", worker)
    result = _cuda_reference("nms", (), {})
    assert run.call_args_list
    assert all(call.args[0][0] == str(interpreter) for call in run.call_args_list)
    worker.assert_called_once_with(str(interpreter))
    torch.testing.assert_close(result[0], expected[0])


def test_cuda_reference_skip_reason(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    interpreter = tmp_path / "cpu-python"
    interpreter.touch()
    monkeypatch.setenv("TORCHVISION_CUDA_REFERENCE_PYTHON", str(interpreter))

    run = Mock(
        return_value=subprocess.CompletedProcess(
            [], 77, b"RuntimeError: torch.cuda.is_available() is false\n", b""
        )
    )
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(pytest.skip.Exception) as skipped:
        _cuda_reference("nms", (), {})
    reason = str(skipped.value)
    assert str(interpreter) in reason
    assert "torch.cuda.is_available() is false" in reason
    assert "tried" in reason


@pytest.mark.parametrize(
    "kind", ["roi_align", "roi_pool", "ps_roi_align", "ps_roi_pool"]
)
def test_roi_half_scale_midpoint_neighbors(mojo_gpu: str, kind: str):
    data = torch.tensor(
        [[[[10, 1, 3, 7], [2, 9, 4, 6], [5, 3, 8, 1], [7, 4, 2, 9]]]],
        dtype=torch.float16,
        requires_grad=True,
    )
    rois = torch.tensor([[0, 1, 0, 2, 2]], dtype=torch.float16)
    grad = torch.ones(1, 1, 1, 1, dtype=torch.float16)
    for lower in (0.499755859375, 0.5, 0.99951171875, 1.0):
        lo = torch.tensor(lower, dtype=torch.float16)
        hi = torch.nextafter(lo, torch.full_like(lo, math.inf)).item()
        midpoint = (lower + hi) / 2
        for scale in (
            math.nextafter(midpoint, -math.inf),
            midpoint,
            math.nextafter(midpoint, math.inf),
        ):
            kwargs = {"output_size": (1, 1), "spatial_scale": scale}
            if "align" in kind:
                kwargs["sampling_ratio"] = 1
            expected, expected_grad = _cuda_reference(kind, (data, rois), kwargs, grad)
            ours = data.detach().to(mojo_gpu).requires_grad_()
            result = getattr(vision.ops, kind)(ours, rois.to(mojo_gpu), **kwargs)
            result.backward(grad.to(mojo_gpu))
            torch.testing.assert_close(
                result.cpu(), expected, rtol=0, atol=0, msg=f"scale={scale!r}"
            )
            assert ours.grad is not None
            torch.testing.assert_close(
                ours.grad.cpu(), expected_grad, rtol=0, atol=0, msg=f"scale={scale!r}"
            )


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_nms_ties_and_degenerate_boxes(mojo_gpu: str, threshold: float):
    boxes = torch.cat((_boxes(127), torch.zeros(3, 4), torch.ones(4, 4)))
    scores = torch.ones(boxes.shape[0])
    want = vision.ops.nms(boxes, scores, threshold)
    got = vision.ops.nms(boxes.to(mojo_gpu), scores.to(mojo_gpu), threshold)
    # Equal scores use the CPU's stable input order, including zero-area boxes.
    torch.testing.assert_close(got.cpu(), want)


def test_batched_nms(mojo_gpu: str):
    boxes = _boxes(301)
    scores = torch.linspace(0.001, 1.0, boxes.shape[0])
    labels = torch.arange(boxes.shape[0]) % 4
    want = vision.ops.batched_nms(boxes, scores, labels, 0.5)
    got = vision.ops.batched_nms(
        boxes.to(mojo_gpu), scores.to(mojo_gpu), labels.to(mojo_gpu), 0.5
    )
    torch.testing.assert_close(got.cpu(), want)


@pytest.mark.parametrize("aligned", [False, True])
@pytest.mark.parametrize("sampling", [-1, 0, 2])
@pytest.mark.parametrize("scale", [1.0, 0.25, 1 / 16])
def test_roi_align_forward_backward(
    mojo_gpu: str, aligned: bool, sampling: int, scale: float
):
    _check_roi(mojo_gpu, "align", torch.float32, scale, sampling, aligned)


@pytest.mark.parametrize("kind", ["align", "pool"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_roi_dtypes(mojo_gpu: str, kind: str, dtype: torch.dtype):
    _dtype_supported(mojo_gpu, dtype)
    _check_roi(mojo_gpu, kind, dtype, 1.0, 2, True)


@pytest.mark.parametrize("scale", [1.0, 0.25, 1 / 16])
def test_roi_pool_forward_backward(mojo_gpu: str, scale: float):
    _check_roi(mojo_gpu, "pool", torch.float32, scale, 2, False)


def _check_roi(
    device: str,
    kind: str,
    dtype: torch.dtype,
    scale: float,
    sampling: int,
    aligned: bool,
):
    data = torch.randn(3, 7, 37, 53, generator=torch.Generator().manual_seed(172)).to(
        dtype
    )
    reference_dtype = dtype
    x = data.to(reference_dtype).detach().requires_grad_()
    ours = data.to(device).detach().requires_grad_()
    rois = _rois(scale, dtype)
    want = _roi_op(
        kind,
        x,
        rois.to(reference_dtype),
        scale=scale,
        sampling=sampling,
        aligned=aligned,
    )
    got = _roi_op(
        kind, ours, rois.to(device), scale=scale, sampling=sampling, aligned=aligned
    )
    assert got.dtype == dtype
    _assert_close(got, want, dtype)
    grad = torch.randn(got.shape, generator=torch.Generator().manual_seed(173)).to(
        dtype
    )
    want.backward(grad.to(reference_dtype))
    got.backward(grad.to(device))
    assert ours.grad is not None and x.grad is not None
    _assert_close(ours.grad, x.grad, dtype)
    first_grad = ours.grad.cpu()
    ours.grad = None
    _roi_op(
        kind, ours, rois.to(device), scale=scale, sampling=sampling, aligned=aligned
    ).backward(grad.to(device))
    assert ours.grad is not None
    _assert_close(ours.grad, first_grad, dtype)


@pytest.mark.parametrize("kind", ["align", "pool"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("count", [19, 255, 257, 1025])
@pytest.mark.parametrize("full_image", [False, True])
def test_roi_backward_overlapping(
    mojo_gpu: str, kind: str, dtype: torch.dtype, count: int, full_image: bool
):
    _dtype_supported(mojo_gpu, dtype)
    data = ((torch.arange(2 * 3 * 17 * 19) * 71) % 997).reshape(2, 3, 17, 19)
    data = (data.to(torch.float64) / 1024).to(dtype)
    rois = torch.tensor(
        [[0, 0, 0, 18, 16] if full_image else [0, -2, -3, 10.5, 12.25]], dtype=dtype
    ).repeat(count, 1)
    rois[:, 0] = torch.arange(count) % 2
    reference_dtype = dtype
    reference = data.to(reference_dtype).requires_grad_()
    ours = data.detach().to(mojo_gpu).requires_grad_()
    expected = _roi_op(kind, reference, rois.to(reference_dtype), (3, 5))
    result = _roi_op(kind, ours, rois.to(mojo_gpu), (3, 5))
    grad = ((torch.arange(result.numel()) * 13) % 31 - 15).reshape(result.shape)
    grad = (grad.to(torch.float64) / 16).to(dtype)
    expected.backward(grad.to(reference_dtype))
    result.backward(grad.to(mojo_gpu))
    _assert_close(result, expected, dtype)
    assert ours.grad is not None and reference.grad is not None
    if dtype == torch.float16:
        # The boxes scatter into shared cells with atomic adds in fp16, in
        # thread order. Over 60 runs on an H100 the 1025-box roi_align cases
        # reached 1.01x and 0.68x of the usual 2e-2 (1 run in 60 failed); every
        # other case stayed under 0.4x.
        torch.testing.assert_close(
            ours.grad.cpu(), reference.grad, rtol=2e-2, atol=6e-2
        )
    else:
        _assert_close(ours.grad, reference.grad, dtype)


@pytest.mark.parametrize("kind", ["align", "pool"])
@pytest.mark.parametrize("warn_only", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_roi_backward_deterministic_algorithms(
    mojo_gpu: str, kind: str, warn_only: bool, empty: bool
):
    data = torch.arange(35, dtype=torch.float32).reshape(1, 1, 5, 7)
    rois = torch.tensor([[0.0, 0.0, 0.0, 6.0, 4.0]])[: 0 if empty else 1]
    ours = data.to(mojo_gpu).requires_grad_()
    result = _roi_op(kind, ours, rois.to(mojo_gpu), (3, 2))
    grad = torch.ones(result.shape).to(mojo_gpu)
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        if empty:
            result.backward(grad)
            assert ours.grad is not None
            torch.testing.assert_close(ours.grad.cpu(), torch.zeros_like(data))
        elif warn_only:
            with pytest.warns(UserWarning, match=f"roi_{kind}_backward_kernel"):
                result.backward(grad)
            assert ours.grad is not None
        else:
            with pytest.raises(RuntimeError, match="does not have a deterministic"):
                result.backward(grad)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


@pytest.mark.parametrize(
    "batches,width,count", [(1, 1, 33), (1, 2, 257), (2049, 1, 33)]
)
def test_roi_pool_backward_half_subnormal(
    mojo_gpu: str, batches: int, width: int, count: int
):
    data = torch.ones(batches, 1, 1, width, dtype=torch.float16)
    rois = torch.zeros(count, 5, dtype=data.dtype)
    rois[:, 0] = batches - 1
    rois[:, 1] = torch.arange(count) % width
    rois[:, 3] = rois[:, 1]
    ours = data.to(mojo_gpu).requires_grad_()
    result = vision.ops.roi_pool(ours, rois.to(mojo_gpu), (1, 1))
    grad = torch.full(result.shape, 2**-24, dtype=data.dtype)
    result.backward(grad.to(mojo_gpu))
    expected = torch.zeros_like(data)
    for pixel in range(width):
        expected[-1, 0, 0, pixel] = ((count + width - 1 - pixel) // width) * 2**-24
    assert ours.grad is not None
    torch.testing.assert_close(ours.grad.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("batches,count", [(1, 1025), (2, 257)])
@pytest.mark.parametrize("positive", [False, True])
def test_roi_pool_backward_half_odd_extent(
    mojo_gpu: str, batches: int, count: int, positive: bool
):
    index = torch.arange(batches * 3 * 17 * 19, dtype=torch.int64)
    data = ((index * 1103515245 + 12345) % 65521 % 17 - 8).float() / 16
    data = data.half().reshape(batches, 3, 17, 19)
    rois = torch.tensor([[0, 0, 0, 18, 16]], dtype=data.dtype).repeat(count, 1)
    rois[:, 0] = torch.arange(count) % batches
    reference = data.detach().requires_grad_()
    ours = data.to(mojo_gpu).requires_grad_()
    expected = vision.ops.roi_pool(reference, rois, (3, 5))
    result = vision.ops.roi_pool(ours, rois.to(mojo_gpu), (3, 5))
    index = torch.arange(result.numel(), dtype=torch.int64)
    grad = ((index * 1103515245 + 12345) % 65521 % 17 - 8).float() / 16
    grad = grad.half().reshape(result.shape)
    if positive:
        grad.fill_(0.0625)
    expected.backward(grad)
    result.backward(grad.to(mojo_gpu))
    assert ours.grad is not None and reference.grad is not None
    _assert_close(ours.grad, reference.grad, data.dtype)


def test_roi_pool_backward_half_batch_bound(mojo_gpu: str):
    data = torch.ones(2049, 1, 1, 1, dtype=torch.float16)
    rois = torch.tensor([[2048, 0, 0, 0, 0]], dtype=torch.float16)
    ours = data.to(mojo_gpu).requires_grad_()
    result = vision.ops.roi_pool(ours, rois.to(mojo_gpu), (1, 1))
    result.backward(torch.ones(result.shape, dtype=data.dtype).to(mojo_gpu))
    expected = torch.zeros_like(data)
    expected[2048] = 1
    assert ours.grad is not None
    torch.testing.assert_close(ours.grad.cpu(), expected)


def test_roi_pool_backward_rounded_bin_edge(mojo_gpu: str):
    data = torch.zeros(1, 1, 3, 59)
    data[0, 0, 1, 57] = 100
    rois = torch.tensor([[0.0, 0.0, 0.0, 56.0, 2.0]])
    reference = data.requires_grad_()
    ours = data.detach().to(mojo_gpu).requires_grad_()
    expected = vision.ops.roi_pool(reference, rois, (1, 7))
    result = vision.ops.roi_pool(ours, rois.to(mojo_gpu), (1, 7))
    # ceil(7 * float32(57 / 7)) is 58: saved argmax defines the footprint.
    expected.backward(torch.ones_like(expected))
    result.backward(torch.ones(result.shape).to(mojo_gpu))
    assert reference.grad is not None and ours.grad is not None
    assert reference.grad[0, 0, 1, 57] == 1
    torch.testing.assert_close(result.cpu(), expected)
    torch.testing.assert_close(ours.grad.cpu(), reference.grad)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", [31, 32, 33])
def test_roi_pool_geometry_regimes(mojo_gpu: str, dtype: torch.dtype, channels: int):
    _dtype_supported(mojo_gpu, dtype)
    data = torch.randn(
        1, channels, 15, 15, generator=torch.Generator().manual_seed(781)
    ).to(dtype)
    rois = torch.tensor([[0, 0, 0, 13, 13], [0, -1, -1, 12, 12]], dtype=dtype).repeat(
        600, 1
    )
    expected, expected_indices = torch.ops.torchvision.roi_pool(data, rois, 1.0, 7, 7)
    result, indices = torch.ops.torchvision.roi_pool(
        data.to(mojo_gpu), rois.to(mojo_gpu), 1.0, 7, 7
    )
    _assert_close(result, expected, dtype)
    torch.testing.assert_close(indices.cpu(), expected_indices)


@pytest.mark.parametrize("kind", ["align", "pool"])
def test_roi_empty_forward_backward(mojo_gpu: str, kind: str):
    x = torch.randn(2, 3, 13, 17, requires_grad=True)
    ours = x.detach().to(mojo_gpu).requires_grad_()
    rois = torch.empty(0, 5)
    want = _roi_op(kind, x, rois)
    got = _roi_op(kind, ours, rois.to(mojo_gpu))
    torch.testing.assert_close(got.cpu(), want)
    want.sum().backward()
    got.sum().backward()
    assert ours.grad is not None and x.grad is not None
    torch.testing.assert_close(ours.grad.cpu(), x.grad)


@pytest.mark.parametrize("kind", ["align", "pool"])
def test_roi_noncontiguous(mojo_gpu: str, kind: str):
    data = torch.randn(3, 7, 53, 37)
    rois = _rois()
    want = _roi_op(kind, data.transpose(2, 3), rois)
    try:
        got = _roi_op(kind, data.to(mojo_gpu).transpose(2, 3), rois.to(mojo_gpu))
    except NotImplementedError as exc:
        assert "contigu" in str(exc).lower()
    else:
        _assert_close(got, want, torch.float32)


def test_roi_pool_argmax(mojo_gpu: str):
    x = torch.arange(2 * 3 * 9 * 11, dtype=torch.float32).reshape(2, 3, 9, 11)
    rois = torch.tensor([[0, -3, -2, 8, 7], [1, 1, 2, 9, 8], [0, -9, -8, -6, -4]])
    want, want_argmax = torch.ops.torchvision.roi_pool(x, rois.float(), 1.0, 3, 4)
    got, got_argmax = torch.ops.torchvision.roi_pool(
        x.to(mojo_gpu), rois.float().to(mojo_gpu), 1.0, 3, 4
    )
    torch.testing.assert_close(got.cpu(), want)
    torch.testing.assert_close(got_argmax.cpu(), want_argmax)


def _ps_roi_op(
    kind: str,
    x: torch.Tensor,
    rois: torch.Tensor,
    output: tuple[int, int] = (3, 5),
    scale: float = 1.0,
    sampling: int = 2,
) -> torch.Tensor:
    if kind == "align":
        return vision.ops.ps_roi_align(x, rois, output, scale, sampling)
    return vision.ops.ps_roi_pool(x, rois, output, scale)


def _check_ps_roi(
    device: str,
    kind: str,
    dtype: torch.dtype,
    scale: float = 1.0,
    sampling: int = 2,
    empty: bool = False,
    noncontiguous: bool = False,
):
    _dtype_supported(device, dtype)
    generator = torch.Generator().manual_seed(201)
    data = torch.randn(3, 45, 37, 53, generator=generator).to(dtype)
    rois = _rois(scale, dtype)
    rois[-1, 3:] += 1 / scale
    if empty:
        rois = rois[:0]
    reference_dtype = dtype
    reference = data.to(reference_dtype).detach().requires_grad_()
    ours = data.to(device).detach().requires_grad_()
    reference_input, device_input = reference, ours
    device_rois = rois.to(device)
    if noncontiguous:
        reference_input = reference.transpose(2, 3)
        device_input = ours.transpose(2, 3)
        rois = rois.t().contiguous().t()
        device_rois = rois.t().contiguous().to(device).t()
    expected = _ps_roi_op(
        kind, reference_input, rois.to(reference_dtype), scale=scale, sampling=sampling
    )
    result = _ps_roi_op(kind, device_input, device_rois, scale=scale, sampling=sampling)
    assert result.shape == (rois.shape[0], 3, 3, 5)
    assert result.dtype == dtype
    _assert_close(result, expected, dtype)
    grad = torch.randn(result.shape, generator=generator).to(dtype)
    if noncontiguous:
        grad = grad.transpose(2, 3).contiguous().transpose(2, 3)
    expected.backward(grad.to(reference_dtype))
    device_grad = grad.transpose(2, 3).contiguous().to(device).transpose(2, 3)
    result.backward(device_grad)
    assert reference.grad is not None and ours.grad is not None
    _assert_close(ours.grad, reference.grad, dtype)


@pytest.mark.parametrize("kind", ["align", "pool"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_ps_roi_forward_backward(
    mojo_gpu: str, kind: str, dtype: torch.dtype, noncontiguous: bool
):
    _check_ps_roi(mojo_gpu, kind, dtype, noncontiguous=noncontiguous)


@pytest.mark.parametrize("sampling", [-1, 0, 1, 3])
@pytest.mark.parametrize("scale", [1.0, 0.25, 1 / 16])
def test_ps_roi_align_sampling(mojo_gpu: str, sampling: int, scale: float):
    _check_ps_roi(mojo_gpu, "align", torch.float32, scale, sampling)


@pytest.mark.parametrize("scale", [0.25, 1 / 16])
def test_ps_roi_pool_scale(mojo_gpu: str, scale: float):
    _check_ps_roi(mojo_gpu, "pool", torch.float32, scale)


@pytest.mark.parametrize("kind", ["align", "pool"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_ps_roi_empty(mojo_gpu: str, kind: str, dtype: torch.dtype):
    _check_ps_roi(mojo_gpu, kind, dtype, empty=True)


@pytest.mark.parametrize("sampling", [-1, 0, 2])
def test_ps_roi_align_zero_area(mojo_gpu: str, sampling: int):
    data = torch.arange(4 * 7 * 9, dtype=torch.float32).reshape(1, 4, 7, 9)
    rois = torch.tensor([[0.0, 2.0, 3.0, 2.0, 3.0]])
    reference = data.detach().requires_grad_()
    ours = data.to(mojo_gpu).requires_grad_()
    expected = _ps_roi_op("align", reference, rois, (2, 2), sampling=sampling)
    result = _ps_roi_op("align", ours, rois.to(mojo_gpu), (2, 2), sampling=sampling)
    torch.testing.assert_close(result.cpu(), expected, equal_nan=True)
    expected.backward(torch.ones_like(expected))
    result.backward(torch.ones(result.shape).to(mojo_gpu))
    assert reference.grad is not None and ours.grad is not None
    torch.testing.assert_close(ours.grad.cpu(), reference.grad, equal_nan=True)


@pytest.mark.parametrize("kind", ["align", "pool"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_ps_roi_channel_mapping(mojo_gpu: str, kind: str, dtype: torch.dtype):
    _dtype_supported(mojo_gpu, dtype)
    data = torch.arange(30, dtype=dtype).reshape(1, 30, 1, 1).expand(1, 30, 9, 11)
    rois = torch.tensor([[0, 1, 1, 8, 7], [0, -9, -8, -6, -4]], dtype=dtype)
    op = getattr(torch.ops.torchvision, f"ps_roi_{kind}")
    args = (1.0, 3, 5, 2) if kind == "align" else (1.0, 3, 5)
    reference_dtype = dtype
    expected, mapping = op(data.to(reference_dtype), rois.to(reference_dtype), *args)
    result, actual_mapping = op(
        data.contiguous().to(mojo_gpu), rois.to(mojo_gpu), *args
    )
    _assert_close(result, expected, dtype)
    assert actual_mapping.dtype == torch.int32
    assert not actual_mapping.requires_grad
    torch.testing.assert_close(actual_mapping.cpu(), mapping)
    torch.testing.assert_close(
        mapping[0].flatten(), torch.arange(30, dtype=torch.int32)
    )


@pytest.mark.parametrize("kind", ["align", "pool"])
def test_ps_roi_module(mojo_gpu: str, kind: str):
    module = (
        vision.ops.PSRoIAlign((3, 5), 0.25, 2)
        if kind == "align"
        else vision.ops.PSRoIPool((3, 5), 0.25)
    )
    data = torch.randn(3, 45, 37, 53, generator=torch.Generator().manual_seed(202))
    rois = _rois(0.25)
    _assert_close(
        module(data.to(mojo_gpu), rois.to(mojo_gpu)), module(data, rois), data.dtype
    )


@pytest.mark.parametrize("kind", ["align", "pool"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_ps_roi_autocast(mojo_gpu: str, kind: str, dtype: torch.dtype):
    data = torch.randn(3, 30, 37, 53, generator=torch.Generator().manual_seed(203)).to(
        dtype
    )
    rois = _rois(dtype=dtype)
    reference = data.float().detach().requires_grad_()
    ours = data.to(mojo_gpu).detach().requires_grad_()
    expected = _ps_roi_op(kind, reference, rois.float())
    op = getattr(torch.ops.torchvision, f"ps_roi_{kind}")
    args = (1.0, 3, 5, 2) if kind == "align" else (1.0, 3, 5)
    _, mapping = op(reference.detach(), rois.float(), *args)
    with torch.autocast("mojo", dtype=torch.float16):
        result = _ps_roi_op(kind, ours, rois.to(mojo_gpu))
        tuple_result, actual_mapping = op(ours, rois.to(mojo_gpu), *args)
    assert result.dtype == dtype and actual_mapping.dtype == dtype
    assert not actual_mapping.requires_grad
    _assert_close(result, expected, dtype)
    _assert_close(tuple_result, expected, dtype)
    torch.testing.assert_close(actual_mapping.cpu(), mapping.to(dtype))
    grad = torch.randn(result.shape, generator=torch.Generator().manual_seed(204)).to(
        dtype
    )
    expected.backward(grad.float())
    result.backward(grad.to(mojo_gpu))
    assert reference.grad is not None and ours.grad is not None
    _assert_close(ours.grad, reference.grad, dtype)


@pytest.mark.parametrize("kind", ["align", "pool"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("warn_only", [False, True])
def test_ps_roi_deterministic_algorithms(
    mojo_gpu: str, kind: str, empty: bool, warn_only: bool
):
    data = torch.ones(1, 6, 5, 7)
    ours = data.to(mojo_gpu).requires_grad_()
    rois = torch.tensor([[0.0, 0.0, 0.0, 6.0, 4.0]])[: 0 if empty else 1]
    result = _ps_roi_op(kind, ours, rois.to(mojo_gpu), (3, 2))
    grad = torch.ones(result.shape).to(mojo_gpu)
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        if empty:
            result.backward(grad)
            assert ours.grad is not None
            torch.testing.assert_close(ours.grad.cpu(), torch.zeros_like(data))
        elif warn_only:
            with pytest.warns(UserWarning, match=f"ps_roi_{kind}_backward_kernel"):
                result.backward(grad)
        else:
            with pytest.raises(RuntimeError, match=f"ps_roi_{kind}_backward_kernel"):
                result.backward(grad)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


def _deform_data(
    dtype: torch.dtype,
    groups: int = 1,
    offset_groups: int = 1,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
    batch: int = 2,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(205)
    height, width = 9, 13
    kernel_h, kernel_w = 3, 2
    out_h = (height + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[
        0
    ] + 1
    out_w = (width + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1
    data = torch.randn(batch, 6, height, width, generator=generator) * 0.2
    weight = torch.randn(6, 6 // groups, kernel_h, kernel_w, generator=generator) * 0.2
    offset = (
        torch.randn(
            batch,
            2 * offset_groups * kernel_h * kernel_w,
            out_h,
            out_w,
            generator=generator,
        )
        * 0.7
    )
    mask = torch.rand(
        batch, offset_groups * kernel_h * kernel_w, out_h, out_w, generator=generator
    )
    bias = torch.randn(6, generator=generator) * 0.2
    return tuple(tensor.to(dtype) for tensor in (data, offset, weight, bias, mask))


def _deform_op(
    tensors: tuple[torch.Tensor, ...],
    use_mask: bool,
    use_bias: bool,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
) -> torch.Tensor:
    data, offset, weight, bias, mask = tensors
    return vision.ops.deform_conv2d(
        data,
        offset,
        weight,
        bias if use_bias else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        mask=mask if use_mask else None,
    )


def _check_deform(
    device: str,
    dtype: torch.dtype,
    groups: int = 1,
    offset_groups: int = 1,
    use_mask: bool = True,
    use_bias: bool = True,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
    batch: int = 2,
    noncontiguous: bool = False,
    autocast: bool = False,
):
    _dtype_supported(device, dtype)
    tensors = _deform_data(
        dtype, groups, offset_groups, stride, padding, dilation, batch
    )
    reference_dtype = torch.float32 if autocast and dtype == torch.float16 else dtype
    reference = tuple(t.to(reference_dtype).detach().requires_grad_() for t in tensors)
    ours = tuple(t.to(device).detach().requires_grad_() for t in tensors)
    if noncontiguous:
        reference = tuple(
            t.t().contiguous().t().detach().requires_grad_()
            if t.ndim == 2
            else t.transpose(0, -1)
            .contiguous()
            .transpose(0, -1)
            .detach()
            .requires_grad_()
            for t in reference
        )
        ours = tuple(
            t.transpose(0, -1)
            .contiguous()
            .to(device)
            .transpose(0, -1)
            .detach()
            .requires_grad_()
            for t in tensors
        )
    expected = _deform_op(reference, use_mask, use_bias, stride, padding, dilation)
    with torch.autocast("mojo", dtype=torch.float16, enabled=autocast):
        result = _deform_op(ours, use_mask, use_bias, stride, padding, dilation)
    assert result.dtype == dtype
    _assert_close(result, expected, dtype)
    grad = (
        torch.randn(result.shape, generator=torch.Generator().manual_seed(206)) * 0.2
    ).to(dtype)
    expected.backward(grad.to(reference_dtype))
    result.backward(grad.to(device))
    for index, (actual, wanted) in enumerate(zip(ours, reference, strict=True)):
        if (index == 3 and not use_bias) or (index == 4 and not use_mask):
            assert actual.grad is None and wanted.grad is None
        else:
            assert actual.grad is not None and wanted.grad is not None
            _assert_close(actual.grad, wanted.grad, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "use_mask,use_bias", [(False, False), (False, True), (True, False), (True, True)]
)
def test_deform_conv2d_forward_backward(
    mojo_gpu: str, dtype: torch.dtype, use_mask: bool, use_bias: bool
):
    _check_deform(mojo_gpu, dtype, use_mask=use_mask, use_bias=use_bias)


@pytest.mark.parametrize(
    "groups,offset_groups", [(1, 3), (2, 1), (2, 3), (3, 2), (6, 6)]
)
@pytest.mark.parametrize(
    "stride,padding,dilation", [((2, 1), (0, 2), (1, 2)), ((1, 2), (2, 1), (2, 1))]
)
def test_deform_conv2d_groups_geometry(
    mojo_gpu: str,
    groups: int,
    offset_groups: int,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
):
    _check_deform(
        mojo_gpu,
        torch.float32,
        groups,
        offset_groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


@pytest.mark.parametrize("batch", [33, 34])
@pytest.mark.parametrize("groups", [1, 2])
def test_deform_conv2d_batch_chunks(mojo_gpu: str, batch: int, groups: int):
    _check_deform(mojo_gpu, torch.float32, groups=groups, offset_groups=3, batch=batch)


@pytest.mark.parametrize("dtype", DTYPES)
def test_deform_conv2d_noncontiguous(mojo_gpu: str, dtype: torch.dtype):
    _check_deform(mojo_gpu, dtype, groups=2, offset_groups=3, noncontiguous=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("use_mask", [False, True])
def test_deform_conv2d_empty(mojo_gpu: str, dtype: torch.dtype, use_mask: bool):
    _check_deform(mojo_gpu, dtype, use_mask=use_mask, batch=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("use_mask,use_bias", [(False, False), (True, True)])
def test_deform_conv2d_autocast(
    mojo_gpu: str, dtype: torch.dtype, use_mask: bool, use_bias: bool
):
    _check_deform(mojo_gpu, dtype, use_mask=use_mask, use_bias=use_bias, autocast=True)


def test_deform_conv2d_module(mojo_gpu: str):
    data, offset, weight, bias, mask = _deform_data(
        torch.float32, groups=2, offset_groups=3
    )
    module = vision.ops.DeformConv2d(6, 6, (3, 2), padding=(1, 1), groups=2)
    with torch.no_grad():
        module.weight.copy_(weight)
        assert module.bias is not None
        module.bias.copy_(bias)
    reference = data.detach().requires_grad_()
    expected = module(reference, offset, mask)
    expected.backward(torch.ones_like(expected))
    expected_weight = module.weight.grad
    expected_bias = module.bias.grad
    module.zero_grad(set_to_none=True)
    module.to(mojo_gpu)
    ours = data.to(mojo_gpu).requires_grad_()
    result = module(ours, offset.to(mojo_gpu), mask.to(mojo_gpu))
    result.backward(torch.ones(result.shape).to(mojo_gpu))
    _assert_close(result, expected, data.dtype)
    assert ours.grad is not None and reference.grad is not None
    _assert_close(ours.grad, reference.grad, data.dtype)
    assert module.weight.grad is not None and expected_weight is not None
    assert module.bias.grad is not None and expected_bias is not None
    _assert_close(module.weight.grad, expected_weight, data.dtype)
    _assert_close(module.bias.grad, expected_bias, data.dtype)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("warn_only", [False, True])
def test_deform_conv2d_deterministic_algorithms(
    mojo_gpu: str, empty: bool, warn_only: bool
):
    tensors = _deform_data(torch.float32, batch=0 if empty else 1)
    ours = tuple(t.to(mojo_gpu).requires_grad_() for t in tensors)
    result = _deform_op(ours, True, True)
    grad = torch.ones(result.shape).to(mojo_gpu)
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        if empty:
            result.backward(grad)
            for tensor in ours:
                assert tensor.grad is not None
                torch.testing.assert_close(
                    tensor.grad.cpu(), torch.zeros_like(tensor.cpu())
                )
        elif warn_only:
            with pytest.warns(UserWarning, match="compute_grad_input"):
                result.backward(grad)
        else:
            with pytest.raises(RuntimeError, match="compute_grad_input"):
                result.backward(grad)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


@pytest.mark.parametrize("kind", ["align", "pool"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_ps_roi_overlapping_backward(mojo_gpu: str, kind: str, dtype: torch.dtype):
    _dtype_supported(mojo_gpu, dtype)
    data = (torch.arange(30 * 11 * 13).reshape(1, 30, 11, 13) % 17).to(dtype) / 16
    rois = torch.tensor([[0, -2, -1, 10, 9]], dtype=dtype).repeat(257, 1)
    reference_dtype = dtype
    reference = data.to(reference_dtype).detach().requires_grad_()
    ours = data.to(mojo_gpu).detach().requires_grad_()
    expected = _ps_roi_op(kind, reference, rois.to(reference_dtype))
    result = _ps_roi_op(kind, ours, rois.to(mojo_gpu))
    grad = (
        (torch.arange(result.numel()) % 9 - 4).reshape(result.shape).float() / 16
    ).to(dtype)
    expected.backward(grad.to(reference_dtype))
    result.backward(grad.to(mojo_gpu))
    _assert_close(result, expected, dtype)
    assert ours.grad is not None and reference.grad is not None
    if dtype == torch.float16:
        # 257 boxes scatter into the same cells with atomic adds in fp16, in
        # whatever order the threads land. Over 40 runs on an H100 the gradient
        # moved by up to 0.12 run to run, and CPU's own fp16 sum sits 0.055 off
        # the fp64 one; the usual 2e-2 is inside that noise.
        torch.testing.assert_close(
            ours.grad.cpu(), reference.grad, rtol=2e-2, atol=0.25
        )
    else:
        _assert_close(ours.grad, reference.grad, dtype)


@pytest.mark.parametrize("use_mask", [False, True])
def test_deform_conv2d_zero_offsets(mojo_gpu: str, use_mask: bool):
    data, offset, weight, bias, mask = _deform_data(
        torch.float32, groups=2, offset_groups=3
    )
    offset.zero_()
    mask.fill_(1)
    expected = torch.nn.functional.conv2d(data, weight, bias, padding=(1, 1), groups=2)
    ours = tuple(t.to(mojo_gpu) for t in (data, offset, weight, bias, mask))
    result = _deform_op(ours, use_mask, True)
    _assert_close(result, expected, data.dtype)


@pytest.mark.parametrize("offset_value", [-100.0, -1.0, 0.0, 0.5, 100.0])
def test_deform_conv2d_offset_boundaries(mojo_gpu: str, offset_value: float):
    tensors = _deform_data(torch.float32, batch=1)
    tensors[1].fill_(offset_value)
    reference = tuple(t.detach().requires_grad_() for t in tensors)
    ours = tuple(t.detach().to(mojo_gpu).requires_grad_() for t in tensors)
    expected = _deform_op(reference, True, True)
    result = _deform_op(ours, True, True)
    grad = torch.full(expected.shape, 0.125)
    expected.backward(grad)
    result.backward(grad.to(mojo_gpu))
    _assert_close(result, expected, torch.float32)
    for actual, wanted in zip(ours, reference, strict=True):
        assert actual.grad is not None and wanted.grad is not None
        _assert_close(actual.grad, wanted.grad, torch.float32)


@pytest.mark.parametrize(
    ("channels", "out_channels", "height", "width", "offset_groups"),
    [(64, 128, 65, 67, 2), (65, 132, 67, 69, 5)],
)
def test_deform_conv2d_large_matrix(
    mojo_gpu: str,
    channels: int,
    out_channels: int,
    height: int,
    width: int,
    offset_groups: int,
):
    data = (torch.arange(2 * channels * height * width) % 17 - 8).float() / 32
    data = data.reshape(2, channels, height, width)
    weight = (torch.arange(out_channels * channels * 9) % 13 - 6).float() / 256
    weight = weight.reshape(out_channels, channels, 3, 3)
    offset = torch.full((2, 18 * offset_groups, height, width), 0.25)
    offset[:, 1::2] = -0.25
    mask = torch.full((2, 9 * offset_groups, height, width), 0.5)
    bias = (torch.arange(out_channels) % 5 - 2).float() / 16
    reference = tuple(
        t.detach().requires_grad_() for t in (data, offset, weight, bias, mask)
    )
    ours = tuple(t.detach().to(mojo_gpu).requires_grad_() for t in reference)
    expected = vision.ops.deform_conv2d(
        *reference[:3], bias=reference[3], mask=reference[4], padding=1
    )
    result = vision.ops.deform_conv2d(*ours[:3], bias=ours[3], mask=ours[4], padding=1)
    grad = ((torch.arange(expected.numel()) % 7 - 3).float() / 256).reshape(
        expected.shape
    )
    expected.backward(grad)
    result.backward(grad.to(mojo_gpu))
    _assert_close(result, expected, torch.float32)
    for actual, wanted in zip(ours, reference, strict=True):
        assert actual.grad is not None and wanted.grad is not None
        _assert_close(actual.grad, wanted.grad, torch.float32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_deform_conv2d_large_spatial_sampling(mojo_gpu: str, dtype: torch.dtype):
    height, width = 513, 517
    data = (
        ((torch.arange(4 * height * width) % 17 - 8).float() / 16)
        .reshape(1, 4, height, width)
        .to(dtype)
    )
    weight = (
        ((torch.arange(2 * 4 * 2 * 2) % 9 - 4).float() / 32)
        .reshape(2, 4, 2, 2)
        .to(dtype)
    )
    offset = torch.full((1, 8, height - 1, width - 1), 0.25, dtype=dtype)
    reference = tuple(t.detach().requires_grad_() for t in (data, offset, weight))
    ours = tuple(
        t.to(mojo_gpu).detach().requires_grad_() for t in (data, offset, weight)
    )
    expected = vision.ops.deform_conv2d(*reference)
    result = vision.ops.deform_conv2d(*ours)
    grad = (
        ((torch.arange(result.numel()) % 7 - 3).float() / 16)
        .reshape(result.shape)
        .to(dtype)
    )
    expected.backward(grad)
    result.backward(grad.to(mojo_gpu))
    _assert_close(result, expected, dtype)
    for actual, wanted in zip(ours, reference, strict=True):
        assert actual.grad is not None and wanted.grad is not None
        _assert_close(actual.grad, wanted.grad, dtype)


@pytest.mark.parametrize("batch", [0, 2])
@pytest.mark.parametrize("dtype", DTYPES)
def test_deform_conv2d_ignored_mask_gradient(
    mojo_gpu: str, batch: int, dtype: torch.dtype
):
    _dtype_supported(mojo_gpu, dtype)
    data = torch.ones(batch, 2, 3, 5, dtype=dtype)
    weight = torch.full((3, 2, 1, 1), 0.25, dtype=dtype)
    offset = torch.zeros(batch, 2, 3, 5, dtype=dtype)
    mask = torch.full((37, 41), 13.0, dtype=dtype).t()
    bias = torch.zeros(3, dtype=dtype)
    grad = torch.full((batch, 3, 3, 5), 0.125, dtype=dtype)
    tensors = (grad, data, weight, offset, mask, bias)
    reference_dtype = dtype
    reference = tuple(t.to(reference_dtype) for t in tensors)
    ours = tuple(t.to(mojo_gpu) for t in tensors)
    parameters = (1, 1, 0, 0, 1, 1, 1, 1, False)
    expected = torch.ops.torchvision._deform_conv2d_backward(*reference, *parameters)
    result = torch.ops.torchvision._deform_conv2d_backward(*ours, *parameters)
    for actual, wanted in zip(result, expected, strict=True):
        _assert_close(actual, wanted, dtype)
    assert result[3].shape == mask.shape
    torch.testing.assert_close(result[3].cpu(), torch.zeros_like(mask))


@pytest.mark.parametrize("axis", [0, 1])
def test_deform_conv2d_half_offset_knot(mojo_gpu: str, axis: int):
    height, width = (3, 1) if axis == 0 else (1, 3)
    data = torch.tensor([0, 1, 5], dtype=torch.float16).reshape(1, 1, height, width)
    offset = torch.zeros(1, 2, height, width, dtype=torch.float16)
    offset[0, axis, height // 2, width // 2] = -(2**-12)
    weight = torch.ones(1, 1, 1, 1, dtype=torch.float16)
    inputs = tuple(t.requires_grad_() for t in (data, offset, weight))
    grad = torch.zeros_like(data)
    grad[0, 0, height // 2, width // 2] = 1
    expected = _cuda_reference("deform_conv2d", inputs, {}, grad)
    ours = tuple(t.detach().to(mojo_gpu).requires_grad_() for t in inputs)
    result = vision.ops.deform_conv2d(*ours)
    result.backward(grad.to(mojo_gpu))
    assert expected[2][0, axis, height // 2, width // 2] == 4
    assert all(t.grad is not None for t in ours)
    actual = (result.cpu(),) + tuple(t.grad.cpu() for t in ours if t.grad is not None)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_deform_conv2d_cuda_dtype_arithmetic(mojo_gpu: str, dtype: torch.dtype):
    _dtype_supported(mojo_gpu, dtype)
    data = (torch.arange(24, dtype=torch.float64) / 13).to(dtype).reshape(1, 2, 3, 4)
    offset = torch.full((1, 2, 3, 4), 0.23, dtype=dtype)
    weight = torch.tensor([0.3, -0.7], dtype=dtype).reshape(1, 2, 1, 1)
    bias = torch.tensor([0.13], dtype=dtype)
    mask = torch.full((1, 1, 3, 4), 0.73, dtype=dtype)
    tensors = tuple(t.requires_grad_() for t in (data, offset, weight, bias, mask))
    grad = torch.zeros(1, 1, 3, 4, dtype=dtype)
    grad[0, 0, 1, 1] = 0.37
    # A single contributing output makes all five gradients order-independent.
    script_tensors = tensors[:4]
    expected = _cuda_reference("deform_conv2d", script_tensors + (mask,), {}, grad)
    ours = tuple(t.detach().to(mojo_gpu).requires_grad_() for t in tensors)
    result = vision.ops.deform_conv2d(*ours[:4], mask=ours[4])
    result.backward(grad.to(mojo_gpu))
    assert all(t.grad is not None for t in ours)
    actual = (result.cpu(),) + tuple(t.grad.cpu() for t in ours if t.grad is not None)
    tolerance = 0 if dtype == torch.float16 else 2e-15
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        torch.testing.assert_close(
            got, want, rtol=tolerance, atol=tolerance, msg=f"output/gradient {index}"
        )


@pytest.mark.parametrize("axis", [0, 1])
def test_deform_conv2d_half_integer_neighbors(mojo_gpu: str, axis: int):
    height, width = (2053, 1) if axis == 0 else (1, 2053)
    data = torch.zeros(1, 1, height, width, dtype=torch.float16)
    data[0, 0, -1, -1] = 1
    data.requires_grad_()
    offset = torch.zeros(1, 2, height, width, dtype=data.dtype)
    weight = torch.ones(1, 1, 1, 1, dtype=data.dtype)
    grad = torch.zeros_like(data)
    grad[0, 0, -1, -1] = 1
    expected, expected_grad = _cuda_reference(
        "deform_conv2d", (data, offset, weight), {}, grad
    )
    ours = data.detach().to(mojo_gpu).requires_grad_()
    result = vision.ops.deform_conv2d(ours, offset.to(mojo_gpu), weight.to(mojo_gpu))
    result.backward(grad.to(mojo_gpu))
    assert expected_grad.flatten()[-2:].tolist() == [1, 1]
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    assert ours.grad is not None
    torch.testing.assert_close(ours.grad.cpu(), expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("axis", [0, 1])
def test_deform_conv2d_half_spatial_bound(mojo_gpu: str, axis: int):
    height, width = (2049, 1) if axis == 0 else (1, 2049)
    data = torch.zeros(1, 1, height, width, dtype=torch.float16)
    data[0, 0, -1, -1] = 2
    offset = torch.zeros(1, 2, height, width, dtype=torch.float16)
    weight = torch.full((1, 1, 1, 1), 3.0, dtype=torch.float16)
    inputs = tuple(t.requires_grad_() for t in (data, offset, weight))
    grad = torch.zeros_like(data)
    grad[0, 0, -1, -1] = 1
    expected = _cuda_reference("deform_conv2d", inputs, {}, grad)
    ours = tuple(t.detach().to(mojo_gpu).requires_grad_() for t in inputs)
    result = vision.ops.deform_conv2d(*ours)
    result.backward(grad.to(mojo_gpu))
    assert expected[0][0, 0, -1, -1] == 6
    assert all(t.grad is not None for t in ours)
    actual = (result.cpu(),) + tuple(t.grad.cpu() for t in ours if t.grad is not None)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("axis", [0, 1])
def test_deform_conv2d_half_last_neighbor(mojo_gpu: str, axis: int):
    height, width = (2052, 1) if axis == 0 else (1, 2052)
    data = torch.zeros(1, 1, height, width, dtype=torch.float16, requires_grad=True)
    offset = torch.zeros(1, 2, height, width, dtype=torch.float16)
    weight = torch.ones(1, 1, 1, 1, dtype=torch.float16)
    grad = torch.zeros_like(data)
    grad[0, 0, -1, -1] = 1
    expected, expected_grad = _cuda_reference(
        "deform_conv2d", (data, offset, weight), {}, grad
    )
    assert expected_grad[0, 0, -1, -1].item() == 1
    ours = data.detach().to(mojo_gpu).requires_grad_()
    result = vision.ops.deform_conv2d(ours, offset.to(mojo_gpu), weight.to(mojo_gpu))
    result.backward(grad.to(mojo_gpu))
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    assert ours.grad is not None
    torch.testing.assert_close(ours.grad.cpu(), expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("axis", [0, 1])
def test_deform_conv2d_offset_boundary_derivative(mojo_gpu: str, axis: int):
    data = torch.tensor([[[[2.0]]]])
    weight = torch.tensor([[[[3.0]]]])
    offset = torch.zeros(1, 2, 1, 1)
    offset[0, axis, 0, 0] = -1
    reference_offset = offset.detach().requires_grad_()
    device_offset = offset.to(mojo_gpu).requires_grad_()
    expected = vision.ops.deform_conv2d(data, reference_offset, weight)
    result = vision.ops.deform_conv2d(
        data.to(mojo_gpu), device_offset, weight.to(mojo_gpu)
    )
    expected.backward(torch.ones_like(expected))
    result.backward(torch.ones(result.shape).to(mojo_gpu))
    assert reference_offset.grad is not None and device_offset.grad is not None
    assert reference_offset.grad[0, axis, 0, 0] == 6
    torch.testing.assert_close(result.cpu(), expected)
    torch.testing.assert_close(device_offset.grad.cpu(), reference_offset.grad)


def test_deform_conv2d_numerical_gradient(mojo_gpu: str):
    _dtype_supported(mojo_gpu, torch.float64)
    generator = torch.Generator().manual_seed(207)
    tensors = (
        torch.randn(1, 1, 4, 3, generator=generator, dtype=torch.float64),
        torch.full((1, 8, 3, 2), 0.23, dtype=torch.float64),
        torch.randn(1, 1, 2, 2, generator=generator, dtype=torch.float64),
        torch.randn(1, generator=generator, dtype=torch.float64),
        torch.rand(1, 4, 3, 2, generator=generator, dtype=torch.float64),
    )
    ours = tuple(t.to(mojo_gpu).requires_grad_() for t in tensors)
    result = _deform_op(ours, True, True, padding=(0, 0))
    result.backward(torch.ones(result.shape, dtype=torch.float64).to(mojo_gpu))
    epsilon = 1e-5
    indices = ((0, 5, 11), (0, 7, 13, 47), (0, 1, 2, 3), (0,), (0, 11, 23))
    for tensor_index, sample_indices in enumerate(indices):
        gradient = ours[tensor_index].grad
        assert gradient is not None
        gradient = gradient.cpu().flatten()
        for element_index in sample_indices:
            plus = tuple(t.clone() for t in tensors)
            minus = tuple(t.clone() for t in tensors)
            plus[tensor_index].flatten()[element_index] += epsilon
            minus[tensor_index].flatten()[element_index] -= epsilon
            numerical = (
                _deform_op(plus, True, True, padding=(0, 0)).sum()
                - _deform_op(minus, True, True, padding=(0, 0)).sum()
            ) / (2 * epsilon)
            torch.testing.assert_close(
                gradient[element_index], numerical, atol=1e-8, rtol=1e-8
            )


@pytest.mark.parametrize("kind", ["align", "pool"])
def test_roi_numerical_gradient(mojo_gpu: str, kind: str):
    _dtype_supported(mojo_gpu, torch.float64)
    data = torch.randn(
        1, 1, 4, 5, generator=torch.Generator().manual_seed(74), dtype=torch.float64
    )
    rois = torch.tensor([[0, 0.2, 0.3, 3.5, 3.2]], dtype=torch.float64)
    ours = data.to(mojo_gpu).requires_grad_()
    output = _roi_op(kind, ours, rois.to(mojo_gpu), output=(2, 2))
    output.backward(torch.ones(output.shape, dtype=output.dtype).to(mojo_gpu))
    assert ours.grad is not None
    gradient = ours.grad.cpu().flatten()
    eps = 1e-5
    for index in range(data.numel()):
        plus, minus = data.clone(), data.clone()
        plus.flatten()[index] += eps
        minus.flatten()[index] -= eps
        numerical = (
            _roi_op(kind, plus, rois, output=(2, 2)).sum()
            - _roi_op(kind, minus, rois, output=(2, 2)).sum()
        ) / (2 * eps)
        torch.testing.assert_close(gradient[index], numerical, atol=1e-8, rtol=1e-8)


def test_roi_align_module(mojo_gpu: str):
    module = vision.ops.RoIAlign((7, 5), 0.25, 2, aligned=True)
    x = torch.randn(3, 7, 37, 53)
    rois = _rois(0.25)
    torch.testing.assert_close(
        module(x.to(mojo_gpu), rois.to(mojo_gpu)).cpu(),
        module(x, rois),
        rtol=3e-5,
        atol=3e-5,
    )


def test_detection_roi_heads(mojo_gpu: str):
    pool = vision.ops.MultiScaleRoIAlign(["0", "1"], output_size=3, sampling_ratio=2)
    heads = vision.models.detection.roi_heads.RoIHeads(
        pool,
        torch.nn.Identity(),
        torch.nn.Identity(),
        0.5,
        0.5,
        32,
        0.25,
        None,
        0.05,
        0.5,
        20,
    )
    features = {"0": torch.randn(2, 3, 32, 40), "1": torch.randn(2, 3, 16, 20)}
    proposals = [_boxes(13) * 2, _boxes(11) * 2]
    for boxes in proposals:
        boxes[0] = torch.tensor([0, 0, 150, 120])
    shapes = [(128, 160), (128, 160)]
    ours_features = {name: value.to(mojo_gpu) for name, value in features.items()}
    ours_proposals = [boxes.to(mojo_gpu) for boxes in proposals]
    want = heads.box_roi_pool(features, proposals, shapes)
    got = heads.box_roi_pool(ours_features, ours_proposals, shapes)
    torch.testing.assert_close(got.cpu(), want, atol=3e-5, rtol=3e-5)
    logits = torch.randn(24, 3, generator=torch.Generator().manual_seed(32))
    regression = torch.zeros(24, 12)
    want_result = heads.postprocess_detections(logits, regression, proposals, shapes)
    got_result = heads.postprocess_detections(
        logits.to(mojo_gpu), regression.to(mojo_gpu), ours_proposals, shapes
    )
    for got_list, want_list in zip(got_result, want_result, strict=True):
        for result, reference in zip(got_list, want_list, strict=True):
            torch.testing.assert_close(result.cpu(), reference, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_autocast(mojo_gpu: str, dtype: torch.dtype):
    x = torch.randn(3, 7, 37, 53).to(dtype)
    rois = _rois(dtype=dtype)
    boxes = _boxes(71, dtype)
    scores = torch.linspace(0.01, 1, 71).to(dtype)
    with torch.autocast("mojo", dtype=torch.float16):
        got = vision.ops.roi_align(
            x.to(mojo_gpu), rois.to(mojo_gpu), (7, 5), sampling_ratio=2
        )
        keep = vision.ops.nms(boxes.to(mojo_gpu), scores.to(mojo_gpu), 0.5)
    # Torchvision 0.26 executes ROIAlign in fp32, then restores the input dtype.
    want = vision.ops.roi_align(x.float(), rois.float(), (7, 5), sampling_ratio=2).to(
        dtype
    )
    assert got.dtype == dtype
    _assert_close(got, want, dtype)
    assert keep.dtype == torch.int64
    torch.testing.assert_close(
        keep.cpu(), vision.ops.nms(boxes.float(), scores.float(), 0.5)
    )


@pytest.mark.parametrize("vision_first", [False, True])
def test_import_order(mojo_gpu: str, vision_first: bool):
    script = textwrap.dedent(f"""
        import torch
        if {vision_first!r}:
            import torchvision
        from torch_mojo_backend import register_mojo_devices
        register_mojo_devices()
        import torchvision
        boxes = torch.tensor([[0., 0., 4., 4.], [1., 1., 3., 3.]])
        scores = torch.tensor([0.9, 0.8])
        expected = torchvision.ops.nms(boxes, scores, 0.5)
        actual = torchvision.ops.nms(boxes.to({mojo_gpu!r}), scores.to({mojo_gpu!r}), 0.5)
        torch.testing.assert_close(actual.cpu(), expected)
        x = torch.randn(1, 2, 8, 8)
        rois = torch.tensor([[0., 1., 1., 6., 6.]])
        expected = torchvision.ops.roi_align(x, rois, (3, 3), sampling_ratio=2)
        actual = torchvision.ops.roi_align(x.to({mojo_gpu!r}), rois.to({mojo_gpu!r}), (3, 3), sampling_ratio=2)
        torch.testing.assert_close(actual.cpu(), expected)
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["align", "pool", "nms"])
def test_unsupported_dtype(mojo_gpu: str, kind: str):
    if kind == "nms":
        with pytest.raises(NotImplementedError, match="dtype|float"):
            vision.ops.nms(
                _boxes(4).to(torch.bfloat16).to(mojo_gpu),
                torch.ones(4, dtype=torch.bfloat16, device=mojo_gpu),
                0.5,
            )
    else:
        with pytest.raises(NotImplementedError, match="dtype|float"):
            _roi_op(
                kind,
                torch.ones(3, 2, 37, 53, dtype=torch.bfloat16, device=mojo_gpu),
                _rois(dtype=torch.bfloat16).to(mojo_gpu),
            )


@pytest.mark.parametrize("kind", ["align", "pool", "nms"])
def test_wrong_device(mojo_gpu: str, kind: str):
    with pytest.raises(NotImplementedError, match="same mojo|same.*device"):
        if kind == "nms":
            vision.ops.nms(_boxes(4).to(mojo_gpu), torch.ones(4), 0.5)
        else:
            _roi_op(kind, torch.ones(3, 2, 37, 53, device=mojo_gpu), _rois())


@pytest.mark.parametrize("kind", ["align", "pool", "nms"])
def test_malformed_shape(mojo_gpu: str, kind: str):
    with pytest.raises(NotImplementedError, match="NCHW|expected boxes|shape"):
        if kind == "nms":
            vision.ops.nms(
                torch.ones(4, 5, device=mojo_gpu), torch.ones(4, device=mojo_gpu), 0.5
            )
        else:
            _roi_op(kind, torch.ones(3, 37, 53, device=mojo_gpu), _rois().to(mojo_gpu))


@pytest.mark.parametrize("operand", ["boxes", "scores"])
def test_nms_noncontiguous(mojo_gpu: str, operand: str):
    boxes = _boxes(24).to(mojo_gpu)
    scores = torch.linspace(0.01, 1, 24).to(mojo_gpu)
    if operand == "boxes":
        boxes = boxes[::2]
        scores = scores[:12]
    else:
        boxes = boxes[:12]
        scores = scores[::2]
    with pytest.raises(NotImplementedError, match="contigu"):
        vision.ops.nms(boxes, scores, 0.5)


def test_second_gpu(mojo_gpu: str):
    if getattr(torch, "mojo").device_count() < 3:
        pytest.skip("requires two GPUs (the final mojo device is CPU)")
    test_nms("mojo:1", torch.float32, 47, 0.5)
    _check_roi("mojo:1", "align", torch.float32, 1.0, 2, True)
    _check_roi("mojo:1", "pool", torch.float32, 1.0, 2, False)
    _check_ps_roi("mojo:1", "align", torch.float32)
    _check_ps_roi("mojo:1", "pool", torch.float32)
    _check_deform("mojo:1", torch.float32, groups=2, offset_groups=3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_roi_pool_autocast_forward_backward(mojo_gpu: str, dtype: torch.dtype):
    data = torch.randn(2, 3, 9, 11, generator=torch.Generator().manual_seed(91)).to(
        dtype
    )
    rois = torch.tensor([[0, -2, -1, 7, 6], [1, 2, 1, 10, 8]], dtype=dtype)
    reference = data.float().detach().requires_grad_()
    ours = data.to(mojo_gpu).detach().requires_grad_()
    device_rois = rois.to(mojo_gpu)
    want = vision.ops.roi_pool(reference, rois.float(), (3, 4))
    _, expected_argmax = torch.ops.torchvision.roi_pool(
        reference.detach(), rois.float(), 1.0, 3, 4
    )
    with torch.autocast("mojo", dtype=torch.float16):
        got = vision.ops.roi_pool(ours, device_rois, (3, 4))
        tuple_output, argmax = torch.ops.torchvision.roi_pool(
            ours, device_rois, 1.0, 3, 4
        )
    assert got.dtype == dtype
    assert argmax.dtype == dtype
    assert not argmax.requires_grad
    _assert_close(got, want, dtype)
    _assert_close(tuple_output, want, dtype)
    torch.testing.assert_close(argmax.cpu(), expected_argmax.to(dtype))
    grad = torch.randn(got.shape, generator=torch.Generator().manual_seed(92)).to(dtype)
    want.backward(grad.float())
    got.backward(grad.to(mojo_gpu))
    assert ours.grad is not None and reference.grad is not None
    _assert_close(ours.grad, reference.grad, dtype)


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_nms_identical_boxes(
    mojo_gpu: str, threshold: float, tied: bool, dtype: torch.dtype
):
    _dtype_supported(mojo_gpu, dtype)
    boxes = torch.tensor([[0.125, 0.3125, 13.234, 31.445]], dtype=dtype).repeat(9, 1)
    scores = (
        torch.ones(9, dtype=dtype) if tied else torch.linspace(0.1, 0.9, 9).to(dtype)
    )
    if dtype == torch.float16:
        (want,) = _cuda_reference("nms", (boxes, scores), {"iou_threshold": threshold})
    else:
        want = vision.ops.nms(boxes, scores, threshold)
    got = vision.ops.nms(boxes.to(mojo_gpu), scores.to(mojo_gpu), threshold)
    assert want.numel() == (9 if threshold == 1.0 else 1)
    torch.testing.assert_close(got.cpu(), want)


@pytest.mark.parametrize("sampling", [-1, 0, 2])
@pytest.mark.parametrize("aligned", [False, True])
def test_roi_align_zero_spatial_scale(mojo_gpu: str, sampling: int, aligned: bool):
    data = torch.randn(3, 2, 5, 7, generator=torch.Generator().manual_seed(93))
    reference = data.detach().requires_grad_()
    ours = data.to(mojo_gpu).requires_grad_()
    rois = _rois()
    want = vision.ops.roi_align(reference, rois, (3, 2), 0.0, sampling, aligned)
    got = vision.ops.roi_align(ours, rois.to(mojo_gpu), (3, 2), 0.0, sampling, aligned)
    _assert_close(got, want, torch.float32)
    grad = torch.randn(want.shape, generator=torch.Generator().manual_seed(94))
    want.backward(grad)
    got.backward(grad.to(mojo_gpu))
    assert ours.grad is not None and reference.grad is not None
    _assert_close(ours.grad, reference.grad, torch.float32)


def test_multiscale_roi_align_backward(mojo_gpu: str):
    pool = vision.ops.MultiScaleRoIAlign(["0", "1"], output_size=3, sampling_ratio=2)
    features = {
        "0": torch.randn(2, 3, 32, 40, requires_grad=True),
        "1": torch.randn(2, 3, 16, 20, requires_grad=True),
    }
    proposals = [_boxes(13) * 2, _boxes(11) * 2]
    for boxes in proposals:
        boxes[0] = torch.tensor([0, 0, 150, 120])
    shapes = [(128, 160), (128, 160)]
    ours = {
        name: tensor.detach().to(mojo_gpu).requires_grad_()
        for name, tensor in features.items()
    }
    expected = pool(features, proposals, shapes)
    result = pool(ours, [boxes.to(mojo_gpu) for boxes in proposals], shapes)
    torch.testing.assert_close(result.cpu(), expected, atol=3e-5, rtol=3e-5)
    grad = torch.randn(expected.shape, generator=torch.Generator().manual_seed(95))
    expected.backward(grad)
    result.backward(grad.to(mojo_gpu))
    for name, reference in features.items():
        actual_grad = ours[name].grad
        assert reference.grad is not None and actual_grad is not None
        torch.testing.assert_close(
            actual_grad.cpu(), reference.grad, atol=3e-5, rtol=3e-5
        )


@pytest.mark.parametrize("kind", ["align", "pool"])
def test_roi_backward_grid_stride(mojo_gpu: str, kind: str):
    generator = torch.Generator().manual_seed(96)
    data = torch.randn(2, 5, 241, 317, generator=generator)
    starts = torch.rand(17, 2, generator=generator) * torch.tensor([220, 170])
    sizes = torch.rand(17, 2, generator=generator) * torch.tensor([80, 60]) + 1
    batches = (torch.arange(17) % 2).float().unsqueeze(1)
    rois = torch.cat((batches, starts, starts + sizes), dim=1)
    reference = data.detach().requires_grad_()
    ours = data.to(mojo_gpu).detach().requires_grad_()
    expected = _roi_op(kind, reference, rois, aligned=True)
    result = _roi_op(kind, ours, rois.to(mojo_gpu), aligned=True)
    _assert_close(result, expected, torch.float32)
    grad = torch.randn(expected.shape, generator=torch.Generator().manual_seed(97))
    expected.backward(grad)
    result.backward(grad.to(mojo_gpu))
    assert reference.grad is not None and ours.grad is not None
    _assert_close(ours.grad, reference.grad, torch.float32)
