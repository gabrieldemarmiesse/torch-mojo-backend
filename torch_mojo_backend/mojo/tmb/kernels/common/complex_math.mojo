"""Complex values as (re, im) pairs of a real dtype (float32 / float64).

Mojo has no complex DType, so a complex64 / complex128 tensor is, to the
kernels, its interleaved float32 / float64 storage (torch's layout), and
kernel code takes and returns the two components as scalars.
"""


@fieldwise_init
struct Cx[dt: DType](Copyable, Movable):
    """One complex value as its two components."""

    var re: Scalar[Self.dt]
    var im: Scalar[Self.dt]
