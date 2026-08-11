"""Shared machinery for the performance-regression benchmark suite.

The suite lives in `benchmarks/` and is plain pytest: every (op, shape,
layout, dtype) combination is one test node, and the node id is the key
under which its ratio against stock PyTorch is stored in
`benchmarks/baselines.json`.  See `benchmarks/conftest.py` for the
pass/fail and update rules, and AGENTS.md for how to run it.
"""
