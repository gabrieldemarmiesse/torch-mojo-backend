# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/graph/topo.cc

from std.ffi import external_call

from tmb.ccl.misc.utils import alloc_bytes, c_string, read_c_string


# ===-------------------------------------------------------------------=== #
# Picking the NIC nearest this rank's GPU
# ===-------------------------------------------------------------------=== #


def realpath(path: String) -> String:
    """realpath(3): the ABSOLUTE /sys/devices path behind a sysfs symlink.

    `readlink` would return the stored relative target (`../../..0000:18:00.0`),
    and two of those share no comparable prefix -- the whole point here is to
    compare where a GPU and a NIC sit in one PCI tree.
    """
    var buf = alloc_bytes(4096)
    var rc = Int(external_call["realpath", Int64](c_string(String(path)), buf))
    if rc == 0:
        return String("")
    return read_c_string(Int(buf), 4095)


def common_prefix(a: String, b: String) -> Int:
    var ab = a.as_bytes()
    var bb = b.as_bytes()
    var n = min(len(ab), len(bb))
    var i = 0
    while i < n and ab[i] == bb[i]:
        i += 1
    return i


def pci_pick(
    device_paths: List[String], gpu_bdf: String, local_rank: Int
) raises -> Int:
    """Index of the NIC this rank should use, by PCI proximity to its GPU.

    Both the GPU and the NIC are PCI devices, so the longer the shared prefix
    of their /sys/devices paths, the fewer switch hops between them -- the
    cheap version of what NCCL's topology detection measures. Ranks that tie
    (the common case: one PCI switch serving a pair of GPUs) fall back to
    `local_rank % n` over the tied set, which on these nodes already hands
    every rank its own NIC. `device_paths[i]` is the sysfs `device` symlink
    of NIC i (`/sys/class/infiniband/<hca>/device`, `/sys/class/cxi/<cxiN>/device`).
    """
    if len(device_paths) == 0:
        raise Error("mojoccl: no network device to choose from")
    if len(device_paths) == 1 or gpu_bdf.byte_length() == 0:
        return local_rank % len(device_paths)
    var gpu_path = realpath("/sys/bus/pci/devices/" + gpu_bdf)
    if gpu_path.byte_length() == 0:
        return local_rank % len(device_paths)
    var best_score = -1
    var ties = 0
    for i in range(len(device_paths)):
        var score = common_prefix(gpu_path, realpath(device_paths[i]))
        if score > best_score:
            best_score = score
            ties = 1
        elif score == best_score:
            ties += 1
    var chosen = List[Int]()
    for i in range(len(device_paths)):
        if common_prefix(gpu_path, realpath(device_paths[i])) == best_score:
            chosen.append(i)
    if len(chosen) == 0:
        return local_rank % len(device_paths)
    return chosen[local_rank % len(chosen)]
