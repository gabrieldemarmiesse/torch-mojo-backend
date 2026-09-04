# Torch Mojo Backend

This project provides a backend for PyTorch in Mojo. The goal is to make it easier to support new devices and accelerators in PyTorch.

## How it works

You only need Torch CPU and a mojo compiler, and if your accelerator is supported by Mojo, then Torch should run on it.
No need to match compiler versions, torch versions, cuda versions, multiple channels, etc... Just pip install and you're ready to go.

Concretely, the backend provides two things:
- It uses the PrivateUse1 device registration method purely in Python, meaning you can just use the `mojo` device with `my_model.to("mojo")` to use your accelerator in eager mode.
- This project also provides a backend for doing `@torch.compile(backend=mojo_backend)`, and it will use mojo (MAX graph) instead of triton to compile your model.

## Warning:

- This project is experimental and should not be used for any serious work. It is currently a proof of concept and its goal is to show what's possible.
- We currently only support a limited set of operations and this was mostly tested on H100, MI300X and Apple M4.
- Due to the high number of operations to implement, the repository make heavy use of AI agents, and it can be seen in the code. While the kernels are very high performance,
  you might find them quite verbose.

You can see our benchmarks for the supported ops [here](https://html-preview.github.io/?url=https://github.com/gabrieldemarmiesse/torch-mojo-backend/raw/refs/heads/main/benchmarks/baselines.html). It can give you an idea of where we're fast and where we're not. We recently tried running nanogpt eager mode on H100, MI300X and Apple M4 on 2.5B tokens, with torch autocast, and we got the same loss curve as stock PyTorch, while being ~2% faster.

Distributed training with DDP works on NVIDIA and AMD GPUs — single node
and multi-node under torchrun, over NCCL or RCCL — see
[docs/distributed.md](docs/distributed.md). On AMD machines install the CPU
wheel of torch (`--index-url https://download.pytorch.org/whl/cpu`): the
CUDA wheel makes every first-use kernel load on HIP about 10x slower, and a
ROCm wheel brings a second HIP runtime the collectives cannot serve.

We don't support yet:
* Using `torch.compile` with the mojo device, only the cuda device is supported for now. 
* DDP on Apple (the process group needs an NCCL-API library: NCCL or RCCL).
* Many GPUs (prefer H100, MI300X and Apple M4)
* Many ops
* Other mojo versions than 1.0

## Installation

```bash
pip install torch-mojo-backend

# or, with uv:
uv add torch-mojo-backend
```

## Quick Start


### Eager mode

The mojo device behaves like any other device in PyTorch.

```python
import torch 
import torch_mojo_backend
torch_mojo_backend.register_mojo_devices()

a = torch.tensor([1, 2, 3], device="mojo:0")
b = torch.tensor([10, 2, 3]).to("mojo:0") # this works too
c = torch.tensor([100, 2, 10]).to("mojo:0")
d = (a + b - c) * 8 / 16
print(d.cpu())
```

You can also write generic code by using `torch.accelerator`. Then your code
will also work on a generic cuda install of Pytorch.

```python
import torch
import torch_mojo_backend
torch_mojo_backend.register_mojo_devices()

device = torch.accelerator.current_accelerator()
a = torch.tensor([1, 2, 3], device=device)
b = torch.tensor([10, 2, 3]).to(device) # this works too
c = torch.tensor([100, 2, 10]).to(device)
d = (a + b - c) * 8 / 16
print(d.cpu())
```

Ops are compiled on the fly, and compilation doesn't stop your code, as long as you don't request a host-device sync.
Torch-mojo-backend uses this optimization to compile multiple ops in the background at the same time. E.g. all the 
ops needed to perform `(a + b - c) * 8 / 16` will be sent to a compiler process pool to be compiled in parallel.
A cache is on disk to make sure we don't recompile when the user restarts the process.
You can look at [this animation](https://html-preview.github.io/?url=https://github.com/gabrieldemarmiesse/torch-mojo-backend/raw/refs/heads/main/docs/kernel_call_queue_animation.html) to understand better how it works.

We guarantee that changing the shapes or the values of the tensors will not trigger a recompilation.

### Torch compile

```python
from torch_mojo_backend import mojo_backend
import torch

model = YourModel().to("cuda")
compiled_model = torch.compile(model, backend=mojo_backend)

output = compiled_model(input_tensor)
```

### Simple Function Example

```python
import torch
from torch_mojo_backend import mojo_backend

@torch.compile(backend=mojo_backend)
def simple_math(x, y):
    return x + y * 2

# Usage
a = torch.tensor([1.0, 2.0, 3.0]).to("cuda")
b = torch.tensor([4.0, 5.0, 6.0]).to("cuda")
print(simple_math(a, b))
```

See [this animation](https://html-preview.github.io/?url=https://github.com/gabrieldemarmiesse/torch-mojo-backend/raw/refs/heads/main/docs/torch_compile_backend_animation.html)
which shows what happens under the hood to convert a dynamo graph to a MAX graph.

### Training

Training works as expected both in eager mode and with `torch.compile`. Here's a simple example of training a model using the mojo backend:

```python
from torch_mojo_backend import mojo_backend
import torch
import torch.nn
import torch.optim
import torch.nn.functional as F

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.linear(x)

device = "cuda"
model = MyModel().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

@torch.compile(backend=mojo_backend)
def train_step(x, y):
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = F.mse_loss(output, y)
    loss.backward()
    optimizer.step()
    return loss

a = torch.randn(5, 3).to(device)
b = torch.randn(5, 2).to(device)

print(train_step(a, b).cpu().detach().numpy())
```

### Compilation Strategy
- Use `fullgraph=True` when possible for better optimization. You'll get an error message if 
  pytorch has to trigger a graph break, making it easy to fix.


### Contributing

We do not currently accept contributions, because we're very early early in the development of this project. 
Feel free to read the code though and steal some good kernels :) 
