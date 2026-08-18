# FA4 source provenance

The Mojo FA4 implementation in this directory was copied from
`gabrieldemarmiesse/flash-attn-mojo` at commit
`da241b436ede069192db1ef1973101f07e33d38e`. The repository owner explicitly
authorized reuse in this backend on 2026-07-18.

Original SHA-256 hashes:

- `_tma4.mojo`: `102713b4b43ff971c272561da6f1e6e6d7db5cea933a8309f80195b382bea519`
- `_wgmma_f16.mojo`: `b3cd920de8f9c9e99fc9b193d879c7de14685571ae586e9623d5588507b00be5`
- `fwd_fa4/common.mojo`: `1891cc40fb181accaa93967e07ed1752724f4a4d1ef5d18d7d780274394d0c34`
- `fwd_fa4/kernel.mojo`: `79364fb346106eeaa28956d12e5f6e34cf4ec4fbbdcd1a82b8d4f7ba9b59364a`
- `fwd_fa4/launch.mojo`: `54634e1b524ef26ae9a9cbda69ab1c0957b2fe47636409731eb3468551bbbd73`
- `bwd_fa4/common.mojo`: `d83c8e815876819e7f3239ee2566977e4543018cc882ebb50d585dd890e94db9`
- `bwd_fa4/kernel.mojo`: `6ad8a2880b3af71b0f45e6747680a26c31f4f1117c8c5d3be8b46452ebde2f3e`
- `bwd_fa4/launch.mojo`: `4f2b5a1f13b7148c2068d7aca45eb1737a97074d77af9dc3928f0911bb31d75e`

The arithmetic kernel bodies carry one deliberate deviation from upstream: the
online-softmax `P = exp2(...)` in the two forward kernels and in the backward
P-recompute rounds the scaled score on its own before subtracting the rowmax
(`s.fma[FastMathFlag.NONE](scale_log2, 0) - m`) instead of fusing the two into a single
`fma(s, scale_log2, -m)`. The fused form leaves the max element's exponent
slightly non-zero by an amount that grows with logit magnitude; see the comment
at the P loop in `fa4_fwd_kernel.mojo` and
`tests/test_eager_kernels.py::test_fa4_forward_saturated_extreme_logit_accuracy`.
Nothing else in the kernel bodies differs. Files were flattened and imports
were prefixed so the official Mojo Python importer can resolve them without
custom include flags. The launcher copies use the backend's existing MAX
`DeviceContext` and omit the upstream default-context `synchronize()` calls;
same-context FIFO ordering preserves dependencies. `fa4_ops.mojo` is a new
raw-pointer CPython bridge combining the three backward launches.
