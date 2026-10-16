"""Half-type arithmetic that keeps the roundings c10 operators perform:
input-dtype products, and opmath results narrowed only once rounded."""

from std.sys import inlined_assembly, llvm_intrinsic, is_nvidia_gpu, is_amd_gpu


@always_inline
def _product[dt: DType](a: Scalar[dt], b: Scalar[dt]) -> Scalar[dt]:
    comptime if dt == DType.float16 and is_nvidia_gpu():
        # Explicit .rn also prevents ptxas from contracting c10::Half operators.
        return inlined_assembly[
            "mul.rn.f16 $0, $1, $2;",
            Scalar[dt],
            constraints="=h,h,h",
            has_side_effect=False,
        ](a, b)
    elif dt == DType.float16 and is_amd_gpu():
        return llvm_intrinsic[
            "llvm.arithmetic.fence", Scalar[dt], has_side_effect=False
        ](a * b)
    elif dt == DType.float16:
        var value = a * b
        var ptr = Pointer(to=value)
        ptr.unsafe_store[volatile=True](0, value)
        return ptr.unsafe_load[volatile=True](0)
    else:
        return a * b


@always_inline
def amd_opaque[w: DType, n: Int](v: SIMD[w, n]) -> SIMD[w, n]:
    """`v` (float32) through a `v_mov_b32` LLVM cannot see into, on AMD
    GPUs; `v` itself elsewhere. Nothing folds across it: no contraction
    of the product it holds into the next add, no narrowing of the float
    operation consuming it to half, no mixed-precision fusion.

    Not `llvm.arithmetic.fence`, which costs no instruction: AMDGPU
    instruction selection fails on the fence of a wave-uniform value that
    lives in a VGPR ("illegal VGPR to SGPR copy", e.g. a loss kernel's
    broadcast scalar operand, LLVM 19 and Mojo 1.1 alike).
    """
    comptime if is_amd_gpu() and w == DType.float32:
        var r = SIMD[w, n]()
        comptime for i in range(n):
            r[i] = inlined_assembly[
                "v_mov_b32 $0, $1",
                Scalar[w],
                constraints="=v,v",
                has_side_effect=False,
            ](v[i])
        return r
    else:
        return v


@always_inline
def cast_rounded[
    dtype: DType, w: DType, n: Int
](v: SIMD[w, n]) -> SIMD[dtype, n]:
    """`v.cast[dtype]()` for an opmath product or fma `v`, rounded to `w`
    first: `static_cast<c10::Half>(float_a * float_b)` on CUDA and CPU.

    AMDGPU selects `fptrunc(fmul / fma float)` (no fast-math flag needed)
    as one `v_fma_mixlo_f16`, which rounds the exact product to half once
    and skips the float rounding: another half wherever the float product
    lands on a half tie (`x * 0.9f`: 12 of 515 elements of an arange, on
    an MI300A). A vector body keeps `v_pk_mul_f32` + `v_cvt_f16_f32`, so
    the result even moved with the size and alignment. Stock ROCm torch
    (hipcc) rounds once there too; CUDA and CPU torch round twice, and so
    does this on every GPU.
    """
    comptime if dtype != w:
        return amd_opaque(v).cast[dtype]()
    else:
        return v.cast[dtype]()
