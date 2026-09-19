"""Tile ownership shared by batched copy kernels."""
from std.collections import InlineArray


trait CopyTileSegment(ImplicitlyCopyable, TrivialRegisterPassable):
    def tile_limit(self) -> Int:
        ...


@always_inline
def copy_segment_index[
    Segment: CopyTileSegment, capacity: Int
](segs: InlineArray[Segment, capacity], count: Int, tile: Int) -> Int:
    var lo = 0
    var hi = count - 1
    while lo < hi:
        var mid = (lo + hi) // 2
        if segs[mid].tile_limit() <= tile:
            lo = mid + 1
        else:
            hi = mid
    return lo
