import extensibility as compiler
from max.gpu.host import DeviceContext
from extensibility import InputTensor, OutputTensor, foreach
from std.utils.index import IndexList
from std.utils.coord import Coord


@compiler.register("grayscale")
struct Grayscale:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        img_out: OutputTensor[dtype=DType.float32, rank=2, ...],
        img_in: InputTensor[dtype=DType.uint8, rank=3, ...],
        ctx: DeviceContext,
    ) raises:
        @parameter
        @always_inline
        def color_to_grayscale[
            simd_width: Int
        ](idx: Coord) -> SIMD[DType.float32, simd_width]:
            @parameter
            def load(
                idx: IndexList[img_in.rank],
            ) -> SIMD[DType.float32, simd_width]:
                return img_in.load[simd_width](idx).cast[DType.float32]()

            row = Int(idx[0].value())
            col = Int(idx[1].value())

            # Load RGB values
            r = load(IndexList[3](row, col, 0))
            g = load(IndexList[3](row, col, 1))
            b = load(IndexList[3](row, col, 2))

            # Apply standard grayscale conversion formula
            gray = 0.21 * r + 0.71 * g + 0.07 * b
            return min(gray, 255)

        foreach[color_to_grayscale, target=target, simd_width=1](img_out, ctx)


# Multiple inputs and outputs example
@compiler.register("grayscale_multi")
struct GrayscaleMulti:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        img_out: OutputTensor[dtype=DType.float32, rank=2, ...],
        red_out: OutputTensor[dtype=DType.float32, rank=2, ...],
        img_in: InputTensor[dtype=DType.uint8, rank=3, ...],
        noise_in: InputTensor[dtype=DType.uint8, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        @parameter
        @always_inline
        def color_to_grayscale[
            simd_width: Int
        ](idx: Coord) -> SIMD[DType.float32, simd_width]:
            row = Int(idx[0].value())
            col = Int(idx[1].value())

            noise = noise_in.load[simd_width](IndexList[2](row, col)).cast[
                DType.float32
            ]()

            # Load RGB values
            r = (
                img_in.load[simd_width](IndexList[3](row, col, 0)).cast[
                    DType.float32
                ]()
                + noise
            )
            g = (
                img_in.load[simd_width](IndexList[3](row, col, 1)).cast[
                    DType.float32
                ]()
                + noise
            )
            b = (
                img_in.load[simd_width](IndexList[3](row, col, 2)).cast[
                    DType.float32
                ]()
                + noise
            )

            # Apply standard grayscale conversion formula
            gray = 0.21 * r + 0.71 * g + 0.07 * b
            red_out.store(IndexList[2](row, col), r)
            return min(gray, 255)

        foreach[color_to_grayscale, target=target, simd_width=1](img_out, ctx)
