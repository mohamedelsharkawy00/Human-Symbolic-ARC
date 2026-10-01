# The "ARC-AGI-3" CUDA Bitmasking Summary

**The Problem:** Standard 2D arrays in CUDA cause warp divergence and waste VRAM. ARC grids (max 30x30) are sparse and "ragged," making standard matrix kernels inefficient for symbolic search.

**The Solution:** Encode the grid into a compact bitmask representation that fits entirely within GPU registers and allows for SIMD-within-a-register operations.

## 1. Data Structure: The 128-bit "Warp-Grid"

- **Encoding:** Each 30x30 grid is represented by 10 bitmasks (one for each ARC color, 0–9).
- **Dimensions:** Since 30×30 = 900 bits, use a custom struct of four `uint256` or fifteen `uint64` integers per color.
- **Memory Alignment:** Align to 128-bit boundaries to ensure coalesced loads from Global Memory to Shared Memory.

## 2. Object Topology Representation

Instead of a whole grid, mask out individual Objects (contiguous blobs).

- **Object Mask:** A 64-bit or 128-bit signature representing a bounding box or relative shape.
- **Unary Operators as Bit-Shifts:**
    - **Translation:** `mask << shift_amount` or `mask >> shift_amount`.
    - **Flipping:** Use `__brevll()` (reverse bits) for horizontal flips.
    - **Rotation:** Pre-computed bit-swizzle patterns stored in `__constant__` memory.

## 3. The Beam Search Optimization

- **Warp-Level Parallelism:** Map 1 Warp (32 threads) to 1 "Path" in the beam.
- **Collision Detection:** Use bitwise AND (`maskA & maskB`) to check for object overlaps in 1 clock cycle.
- **Goal State Check:** XOR the current state with the target example. If the result is 0, the solution is found.
- **Pruning:** Calculate Hamming Distance (using `__popcll`) to find the "nearest" state to the goal for the beam's heuristic.

## 4. Technical Constraints for the Kernel

- **Avoid Global Memory:** Keep the active "Beam Front" in Shared Memory (`__shared__`) or Registers.
- **Warp Shuffle:** Use `__shfl_sync()` to broadcast the winning transformation from one thread to the whole warp without hitting memory.
- **Grid Flattening:** Treat the 30x30 as a 1D bitstream. Offset = y × 30 + x.

## The Prompt for your next LLM Session

> I am building a high-performance ARC-AGI-3 solver in raw CUDA. I need to implement a topological beam search using bitmasked grids.
>
> Specs:
> 1. Represent 30x30 grids as a series of `uint64_t` bitmasks per color.
> 2. Implement Unary Operators (Rotate, Flip, Translate) as bitwise operations.
> 3. Optimize for Warp-Level parallelism (1 warp per search branch).
> 4. Focus on minimizing warp divergence by flattening 2D coordinates into a 1D bitstream.
> 5. Use `__popcll` for heuristic scoring.
>
> Please provide the CUDA kernel boilerplate for the bitwise translation and the struct definition for the bitmasked grid.
