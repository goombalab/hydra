import copy

import torch
from torch.utils.benchmark import Timer

from hydra.modules import Hydra


def maxdiff(a, b):
    return (b - a).abs().max().item()


def test_and_benchmark_meff():
    device = torch.device("cuda")
    x = torch.randn(1, 2**17, 512, device=device)

    hydra_meff = Hydra(512, learnable_init_states=True, bias=True, use_mem_eff_path=True)
    hydra_meff.to(device)
    y_meff = hydra_meff(x)
    y_meff.mean().backward()

    hydra_auto = copy.deepcopy(hydra_meff)
    hydra_auto.to(device)
    hydra_auto.use_mem_eff_path = False
    y_auto = hydra_auto(x)
    y_auto.mean().backward()

    print(f"Fwd  {' '* 15} {maxdiff(y_meff, y_auto):.10f}")
    for (n, p1), (_, p2) in zip(hydra_meff.named_parameters(), hydra_auto.named_parameters()):
        print(f"Grad {n:15} {maxdiff(p1.grad, p2.grad):.10f}")
        assert p1.grad.stride() == p2.grad.stride()
    print("=" * 100)

    hydra_meff.zero_grad()
    hydra_auto.zero_grad()

    y_meff = hydra_meff(x)
    loss = y_meff.mean()
    timer = Timer(stmt="loss.backward(retain_graph=True)", globals={"loss": loss})
    print("Bwd meff    ", timer.timeit(10).mean)

    y_auto = hydra_auto(x)
    loss = y_auto.mean()
    timer = Timer(stmt="loss.backward(retain_graph=True)", globals={"loss": loss})
    print("Bwd autograd", timer.timeit(10).mean)


if __name__ == "__main__":
    test_and_benchmark_meff()
