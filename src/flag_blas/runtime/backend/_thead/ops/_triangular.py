import triton
import triton.language as tl


_MAX_CORE_DIM = 65535


@triton.jit
def triangular_tile_ids(tile_id, UPLO: tl.constexpr):
    high = ((tl.sqrt(8.0 * tile_id + 1.0) - 1.0) * 0.5).to(tl.int32)
    low = tile_id - high * (high + 1) // 2
    if UPLO == 0:
        return high, low
    return low, high


def triangular_grid(n):
    def grid(meta):
        tiles = triton.cdiv(n, meta["BLOCK_SIZE"])
        tile_count = tiles * (tiles + 1) // 2
        return (min(tile_count, _MAX_CORE_DIM),)

    return grid
