"""CI partitions must cover every unit test once, before pytest-split shards it."""

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


@pytest.fixture
def selection_suite(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> pytest.Pytester:
    # Exercise the real hooks in a fresh, GPU-free pytest process. Explicitly
    # load pytest-split so unrelated installed plugins cannot affect ordering.
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    root = Path(__file__).resolve().parents[1]
    pytester.makeconftest((root / "conftest.py").read_text())
    pytester.makeini("""
        [pytest]
        markers =
            gpu: needs an accelerator
            cuda: CUDA graph test
            cpu_torch: needs a CPU torch wheel on a GPU machine
    """)
    pytester.makepyfile(
        test_sample="""
        import pytest

        @pytest.fixture
        def mojo_gpu():
            pytest.skip("no GPU on this machine")

        @pytest.fixture
        def wrapped_device(mojo_gpu):
            return mojo_gpu

        @pytest.fixture
        def cuda_available():
            return False

        @pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
        def device(request, cuda_available):
            return request.param

        class Conf:
            def __init__(self, device):
                self.device = device

        @pytest.fixture(params=[Conf("mojo:gpu")])
        def conf(request):
            return request.param

        @pytest.mark.parametrize("case", range(12))
        def test_cpu(case):
            pass

        def test_conf_default(conf):
            pass

        @pytest.mark.parametrize(
            "conf", [Conf("cpu"), Conf("mojo:gpu")], ids=["cpu", "mojo"], indirect=True
        )
        def test_conf(conf):
            pass

        def test_device(device):
            pass

        def test_transitive_gpu(wrapped_device):
            pass

        @pytest.mark.gpu
        def test_explicit_gpu():
            pass

        @pytest.mark.cpu_torch
        def test_inductor(wrapped_device):
            pass

        @pytest.mark.cpu_torch
        def test_inductor_cpu_validation():
            pass
    """
    )
    return pytester


def _collect(pytester: pytest.Pytester, *args: str) -> list[str]:
    result = pytester.runpytest_subprocess(
        "-p", "pytest_split.plugin", "--strict-markers", "--collect-only", "-q", *args
    )
    assert result.ret == pytest.ExitCode.OK
    return [line for line in result.outlines if line.startswith("test_sample.py::")]


def test_ci_selections_are_disjoint_and_exhaustive(selection_suite: pytest.Pytester):
    all_tests = set(_collect(selection_suite))
    cpu = set(_collect(selection_suite, "-m", "not gpu"))
    cuda = set(_collect(selection_suite, "-m", "gpu and cuda"))
    mojo = set(_collect(selection_suite, "-m", "gpu and not cuda and not cpu_torch"))
    inductor = set(_collect(selection_suite, "-m", "gpu and cpu_torch"))

    assert cuda == {"test_sample.py::test_device[cuda]"}
    assert mojo == {
        "test_sample.py::test_transitive_gpu",
        "test_sample.py::test_explicit_gpu",
        "test_sample.py::test_conf_default[conf0]",
        "test_sample.py::test_conf[mojo]",
    }
    assert inductor == {"test_sample.py::test_inductor"}
    assert "test_sample.py::test_device[cpu]" in cpu
    assert "test_sample.py::test_inductor_cpu_validation" in cpu
    # torch.compile on CPU tensors needs no accelerator.
    assert "test_sample.py::test_conf[cpu]" in cpu
    groups = [cpu, cuda, mojo, inductor]
    assert set.union(*groups) == all_tests
    assert sum(map(len, groups)) == len(all_tests)


def test_cpu_shards_cover_the_filtered_selection_once(selection_suite: pytest.Pytester):
    cpu = _collect(selection_suite, "-m", "not gpu")
    shards = [
        _collect(selection_suite, "-m", "not gpu", "--splits", "3", "--group", str(n))
        for n in range(1, 4)
    ]
    assert set().union(*shards) == set(cpu)
    assert sum(map(len, shards)) == len(cpu)
    for shard in shards:
        # Shard membership is shuffled; execution order is still collection order.
        assert shard == [node for node in cpu if node in shard]
