"""Repo-root pytest hooks.

CI selects CPU tests with `-m 'not gpu'` and GPU tests with `-m gpu`.
GPU requirements are inferred from the shared device fixtures, including
transitive dependencies; tests managing their own devices use an explicit
`pytest.mark.gpu`. Parametrized CPU/CUDA fixtures mark only the CUDA case.

CI shards the CPU selection 18 ways with `pytest-split` (`--splits 18 --group N`).
pytest-split has no durations file, so it falls back to cutting the *collection
order* into equal-count chunks -- and collection order is the worst possible
order to cut, because cost is clustered in it: a file's tests, and above all the
parametrizations of one test function, sit next to each other and cost the same.
A chunk is therefore a slice of one or two files rather than a sample of the
suite, and the shards came out between 41 s and 1416 s.

So we deal the tests out pseudo-randomly before pytest-split chunks them. The
seed is fixed, so every shard of a run agrees on the deal and every test lands
in exactly one shard; each chunk becomes a uniform sample of the whole suite and
the shards converge on the mean. This needs no durations file, which matters
here because durations would have to be *measured* on CI to be right: this suite
skips a different set of tests on a GPU box than on the CPU-only runners, so
timings recorded anywhere else would balance for the wrong machine.

The deal decides shard membership and nothing else. `pytest_collection_finish`
puts the surviving items back in collection order before anything runs, so
execution order within a shard is what it always was -- same neighbours, same
module grouping, same module-scoped fixture setups.
"""

from __future__ import annotations

import random

import pytest

# Any fixed value works; this one is the date the deal was introduced. Changing
# it reshuffles every shard, which is harmless but throws away the per-shard
# native build caches CI keys on the shard index.
SHARD_DEAL_SEED = 20260915

_COLLECTION_INDEX = pytest.StashKey[int]()

# These fixtures require an accelerator, even when they skip because none
# was found. Never classify by hardware availability: both CI pools must
# collect the same partition on machines with and without a GPU.
_GPU_FIXTURES = frozenset(
    {
        "mojo_gpu",
        "mojo_device",
        "mojo_gpu_available",
        "mojo_pair",
        "two_gpus",
        "mojo_h100",
        "cuda_device",
    }
)


def _sharding(config: pytest.Config) -> bool:
    """Is this run one shard of a pytest-split run?"""
    return getattr(config.option, "splits", None) is not None


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    """Mark device requirements before `-m` filtering, then deal the shards.

    pytest-split's own hook is `trylast`, so this one runs first and it chunks
    the dealt order. Un-sharded runs keep their original order.
    """
    for item in items:
        if isinstance(item, pytest.Function) and _GPU_FIXTURES.intersection(
            item.fixturenames
        ):
            item.add_marker(pytest.mark.gpu)
        # CUDA graph tests need a process without PrivateUse1 registration;
        # CPU-torch integration tests need a different torch wheel, but still
        # execute on the GPU. Both belong to the GPU partition.
        if item.get_closest_marker("cuda") is not None:
            item.add_marker(pytest.mark.gpu)

    if not _sharding(config):
        return
    for index, item in enumerate(items):
        item.stash[_COLLECTION_INDEX] = index
    random.Random(SHARD_DEAL_SEED).shuffle(items)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_finish(session: pytest.Session):
    """Restore collection order for the items this shard kept.

    Runs after every `pytest_collection_modifyitems`, on the very list pytest is
    about to execute, so the deal above never reaches execution order.
    `tryfirst` only puts this ahead of the terminal reporter's own
    `pytest_collection_finish`, so `--collect-only` prints a shard in the order
    it will actually run rather than in dealt order.
    """
    if not _sharding(session.config):
        return
    session.items.sort(key=lambda item: item.stash[_COLLECTION_INDEX])
