# mojoccl transport self-tests that need no GPU

Standalone Mojo programs that exercise
`torch_mojo_backend/distributed/mojoccl/{bootstrap,ibverbs,libfabric,netutil,internode,vmm}.mojo`
without a GPU or an allocation of one. They caught six real bugs
(bootstrap/QP/immediate wiring, resource leaks on a failed `ib_setup`, a
silently-misread port LID) before any GPU time was spent chasing them.

`ib_bringup` and `ib_pipeline` need a NIC, and run against **either**
transport backend:

* `MOJOCCL_NET=verbs` — a host with an ACTIVE InfiniBand port. On a cluster
  that has one this is the login node, which is what makes them cheap enough
  to run on every change.
* `MOJOCCL_NET=fabric` — a host with a libfabric RMA provider. On Adastra
  that is a **compute node**: the login node's only rdma device is a RoCE
  bond (`link_layer: Ethernet`, which `list_ib_ports` correctly skips) and
  the compute nodes have four Slingshot NICs, `/dev/cxi[0-3]`, and no
  InfiniBand at all. `FI_LOCAL_COMM` means several processes on one node
  talk to each other over the NIC, so one node is enough.

`geometry_test`, `fd_exchange`, `sock_deadline` and `fabric_abi` need no NIC
either, and the last two need no peer processes. `fabric_hmem` is the
exception to the whole file: it needs a GPU.

Build (from a checkout of this repo, no accelerator needed except where
noted):

    uv run --no-sync mojo build tests/multinode/selftest/bs_test.mojo \
        -I torch_mojo_backend/mojo -o /tmp/bs_test
    uv run --no-sync mojo build tests/multinode/selftest/ib_bringup.mojo \
        -I torch_mojo_backend/mojo -o /tmp/ib_bringup
    uv run --no-sync mojo build tests/multinode/selftest/ib_pipeline.mojo \
        -I torch_mojo_backend/mojo -o /tmp/ib_pipeline
    uv run --no-sync mojo build tests/multinode/selftest/geometry_test.mojo \
        -I torch_mojo_backend/mojo -o /tmp/geometry_test
    uv run --no-sync mojo build tests/multinode/selftest/fd_exchange.mojo \
        -I torch_mojo_backend/mojo -o /tmp/fd_exchange
    uv run --no-sync mojo build tests/multinode/selftest/sock_deadline.mojo \
        -I torch_mojo_backend/mojo -o /tmp/sock_deadline
    uv run --no-sync mojo build tests/multinode/selftest/fabric_abi.mojo \
        -I torch_mojo_backend/mojo -o /tmp/fabric_abi_check
    # fabric_hmem picks libamdhip64 vs libcuda from the BUILD host's
    # accelerator (driver.mojo, comptime): build it on a GPU node, or name
    # the target.
    uv run --no-sync mojo build tests/multinode/selftest/fabric_hmem.mojo \
        --target-accelerator amdgpu:gfx942 \
        -I torch_mojo_backend/mojo -o /tmp/fabric_hmem

## `bs_test.mojo` — the TCP bootstrap

`bs_test <rank> <nranks> <uid-file> <fake-host-index>`. Rank 0 writes the
128-byte unique id to `<uid-file>`; the rest poll for it. Each rank adds
`<fake-host-index>` to its host hash, so one box can pretend to be several
nodes and the derived node/local_rank table can be checked against a known
answer. Runs both blob rounds (16-byte and 256-byte, payload verified) and
the closing barrier.

    for r in $(seq 0 15); do /tmp/bs_test $r 16 /tmp/uid.txt $((r/8)) & done; wait

Exercised: 8 ranks/1 node, 16/2, 6/3.

## `ib_bringup.mojo` — the RDMA transport

`ib_bringup <rank> <nranks> <uid-file>`. Every rank is its own "node"
(local_world 1), so `nranks-1` queue pairs are created per rank. Registers
host memory as the region, runs the bootstrap, moves every QP to RTS, then
runs 400 exchanges through `internode.ib_exchange_now` — the inline
equivalent of the stream callback — alternating inbox halves and verifying
every peer's slot against a rank/sequence-derived pattern.

    for r in 0 1 2 3; do
        /tmp/ib_bringup $r 4 /tmp/uid4.txt &
    done; wait

On Slingshot, the same thing with the other backend (on a compute node --
`srun --overlap --ntasks=1 --cpus-per-task=8`, no GPU needed):

    for r in 0 1 2 3; do
        MOJOCCL_NET=fabric /tmp/ib_bringup $r 4 /tmp/uid4.txt &
    done; wait

Exercised on `cxi` at 2, 4 and 6 processes, and at 4 with
`MOJOCCL_FABRIC_DOMAIN=cxi$((r % 4))` so each process drives its own NIC.
If a whole batch fails with `fi_enable failed, rc=-12`, that is the
unresolved flake described in `agents_docs/distributed.md` ("One unresolved flake on
Slingshot") -- a node condition, not this code. Retry; that has always
worked.

These synchronous probes explicitly pass `_synchronous_test=True` to `ib_setup`.
They use libc as the driver and host memory as the region, so a GPU-visible
proxy mailbox cannot be allocated. This private argument is only for these
tests; production always enables its proxy. No environment setting is needed.
The GPU-backed `fabric_hmem` probe uses the same private argument to avoid
racing its synchronous exchange loop with a concurrent proxy.

400 exchanges is deliberately past `RECV_DEPTH` (64): a receive WR that is
consumed and not reposted shows up as a hang rather than as wrong data.

## `ib_pipeline.mojo` — several exchanges in flight, and the credits

`ib_pipeline <rank> <nranks> <uid-file> [nslots] [depth] [nexchanges]`
(defaults 5, 4, 200 — the shipped `INBOX_SLOTS` and `PIPE_ARENAS`).
Reproduces the GPU schedule of `mojoccl._do_allreduce` exactly — submit
chunk k, consume chunk k-(depth-1) — with the calling thread standing in
for the stream, so `credit_upto` takes the values it takes in production.

    for r in 0 1 2 3; do
        /tmp/ib_pipeline $r 4 /tmp/uidp.txt 5 4 200 &
    done; wait

`MOJOCCL_NET=fabric` runs the same matrix on Slingshot; the credit protocol
is transport-independent (the immediate's bit 31 means the same thing under
both) and this is what proves it.

It catches the two failures the credit protocol can have, and they look
different: a slot group rewritten before its consumer read it is wrong bytes
(the payload pattern carries the sequence number), while a credit never sent
or never counted is a **hang**, which is why the run goes well past
`nslots` exchanges. Exercised at `nslots depth` of `5 4` (shipped), `5 5`
and `2 2` (the tight window, where a rank blocks until a peer's credit
arrives), `1 1` (fully serial) and `8 4`, at 4 and 6 ranks.

## `geometry_test.mojo` — the region carving, GPU-free and peer-free

`geometry_test`, no arguments. Sweeps `mojoccl`'s own layout arithmetic
(`region_layout`, `inbox_group_bytes`, `max_chunk_bytes`,
`pipeline_chunk_bytes` — the communicator calls these through one-line
wrappers, so this checks the shipped code and not a copy) over region sizes
1 MiB–1 GiB, `local_world` 1–8, 2–16 nodes, every allreduce dtype width and
message sizes from one element to four chunk caps, and asserts that every
address a collective forms stays inside the area it belongs to: inbox slots
inside their group, the shard inside stage_out, the compacted push slots
inside stage_in, chunk offsets 16-byte aligned, the staging total not
growing, and a single-node region byte-identical to the pre-pipeline one.
~99k cases, under a second.

## `fd_exchange.mojo` — the SCM_RIGHTS fd transport of the NVLS bring-up

`fd_exchange <local_rank> <local_world> <dir> <magic>`. `vmm.mojo` hands the
multicast object and every rank's own VMM handle to its node-mates as file
descriptors over an AF_UNIX `SOCK_DGRAM` socket, with the msghdr / cmsghdr /
sockaddr_un structs laid out by hand over `UInt64` words because `std.ffi`
has no C-struct ABI. A wrong offset there does not fail loudly — `sendmsg`
succeeds and the control message is silently dropped, or a descriptor arrives
that belongs to something else — so this runs the shipped functions
(`socket_path`, `scm_bind`, `scm_send`, `scm_recv`, `scm_exchange_fds`,
`scm_unbind`) over ordinary file descriptors and checks that what the receiver reads *through*
the descriptor is what the sender wrote.

    rm -rf /tmp/fdx && mkdir -p /tmp/fdx
    for r in $(seq 0 7); do /tmp/fd_exchange $r 8 /tmp/fdx 987654321 & done; wait

The same `(kind, tag)` dispatch as production, in three rounds: rank 0's
"multicast" descriptor one-to-all, then every rank's own descriptor
all-to-all sending everything before receiving anything, then that same
all-to-all through `scm_exchange_fds`, which interleaves the two halves and
is what `nvls_bind_and_map` calls. Datagrams from several senders arrive in
an arbitrary order, so the tag is the only thing that says whose region one
is. No GPU, no InfiniBand, no multicast hardware. Exercised at 2 and 8 ranks.

## `sock_deadline.mojo` — the deadlines, with the peers deliberately absent

`sock_deadline`, no arguments, no peers, no IB, ~8 s. Five calls that must
FAIL, each near its 1.5 s deadline: the bootstrap root with a rank that never
connects, a connect to a port nothing listens on, a connect to an unroutable
address (TEST-NET-3), `scm_send` into an AF_UNIX datagram queue filled to
`net.unix.max_dgram_qlen`, and `scm_exchange_fds` with a peer that never
binds. Coming back too early fails the case (the deadline is not being
honoured) and so does taking much longer — the second is the bug it exists
for: against the code before it, the unroutable connect returned after
**129.5 s** on a 1.5 s deadline, because `connect` blocked in the kernel and
the deadline was only read after it came back. It is 1.50 s now.

    /tmp/sock_deadline

## `fabric_abi.mojo` + `fabric_abi.c` — the libfabric struct offsets

`libfabric.mojo` reaches libfabric's `static inline` data path the way
`ibverbs.mojo` reaches libibverbs': by loading a function pointer out of an
ops table at a hand-written byte offset (`ep->rma->writemsg`,
`cq->ops->read`, `domain->mr->regattr`, `fid->ops->bind`). `std.ffi` has no
C-struct ABI (MOCO-3692), so those offsets are numbers in the Mojo source,
and a wrong one neither fails to compile nor reliably fails to run -- it
hands the NIC a garbage pointer. `fabric_abi.c` prints every size, offset
and constant that file believes, straight out of the installed headers, and
`fabric_abi.mojo` compares the two lists:

    gcc -O0 -I /opt/cray/libfabric/2.2.0rc1/include \
        -o /tmp/fabric_abi tests/multinode/selftest/fabric_abi.c
    /tmp/fabric_abi > /tmp/fabric_abi.txt
    /tmp/fabric_abi_check /tmp/fabric_abi.txt        # prints PASS

135 constants, no NIC, no peers, instant. Run it against any libfabric
install before trusting the transport on it.

## `fabric_hmem.mojo` — the one part host memory cannot test

`fabric_hmem <rank> <nranks> <uid-file> [nexchanges]`, on a GPU node, under
the GPU lock. `ib_bringup` and `ib_pipeline` register a plain malloc'd
region, which on cxi means `iface = FI_HMEM_SYSTEM`. Production hands
`ib_setup` a `driver.alloc_region` allocation
(`hipExtMallocWithFlags(hipDeviceMallocUncached)` / `cuMemAlloc_v2`), which
has to register as FI_HMEM_ROCR instead -- a different provider path, a
different kernel driver, and the piece most likely to be missing from a
libfabric build.

    for r in 0 1; do
        MOJOCCL_NET=fabric MOJOCCL_IB_TRACE=1 \
            /tmp/fabric_hmem $r 2 /tmp/uidh.txt 50 &
    done; wait

It checks that every exchange RETIRES -- the payload write landed, every
peer's notification arrived, this rank's own writes completed, the flush read
came back, `ib_error` still zero. It does NOT check the bytes: the region is
device memory and this test has no stream to copy it back with.

Covered by these and NOT by anything that needs a GPU: interface selection,
the unique-id encoding, the two-round rendezvous, topology derivation, NIC
and port/domain selection, memory registration, connection setup (QP
INIT/RTR/RTS with NCCL's attribute values, or address-vector insertion), the
ops-table dispatch for the data path of both libraries, the immediate's
sequence and credit tagging (an InfiniBand immediate, or 64 bits of
libfabric remote CQ data), receive reposting, the GPUDirect flush read, the
credit-based flow control with several exchanges in flight, the region
geometry, the `SCM_RIGHTS` fd transport (both its shapes), the socket
deadlines, the libfabric struct ABI, and `ib_setup`'s unwind of
partially-created resources on a failure path. NOT covered: the progress
thread and its two spin kernels (they need pinned host memory and a stream),
byte-level verification of an RMA write into device memory, and everything
in `mojoccl.mojo` above the transport.

## `host_fault_test.mojo` — independent host/device fault records

Calls the production host publisher against a stack-allocated status page.
Checks device-first, host-first, partially published device details, and a
second host fault: host writes never change device fields, and the first
observed source and host details remain latched. No GPU code runs.

```bash
PYTHONPATH=$PWD uv run --no-sync mojo build tests/multinode/selftest/host_fault_test.mojo \
    -I torch_mojo_backend/mojo --target-accelerator sm_90a \
  -o /tmp/mojoccl_host_fault_test
PYTHONPATH=$PWD uv run --no-sync /tmp/mojoccl_host_fault_test
```

## `defaults.mojo` — host dispatch constants across architectures

This probe checks production default readers without opening a GPU context.
Pass expected values independently of the implementation's architecture gate:

```bash
uv run --no-sync mojo build tests/multinode/selftest/defaults.mojo \
    -I torch_mojo_backend/mojo --target-accelerator gfx942 \
    --Werror -o /tmp/mojoccl_defaults
uv run --no-sync /tmp/mojoccl_defaults 8 16 64
```

For `sm_90a` and `sm_80`, rebuild with that target and pass `16 64 256`.
The arguments are the two block caps and staging MiB. The probe also checks
the shared thread counts, thresholds, unrolls, pipeline constant and idle sleep.
The compiled programs run on the build host even without the target GPU.

## `comm_state_probe.mojo` — abort release assertion

Read-only helper for `stream_order_probe.py`, built against the same source
as the tested library. Checks the release flag and cleared IB, status-page,
and completion-event handles after abort.

```bash
PYTHONPATH=$PWD uv run --no-sync mojo build --emit shared-lib \
    tests/multinode/selftest/comm_state_probe.mojo \
    -I torch_mojo_backend/mojo -o /tmp/comm_state_probe.so
export MOJOCCL_STATE_PROBE_LIBRARY=/tmp/comm_state_probe.so
```
