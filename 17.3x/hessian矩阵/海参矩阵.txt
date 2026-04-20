import torch
from typing import Callable

def target_function(x: torch.Tensor) -> torch.Tensor:
    return x[0]**3+(x[1]**2)*x[2]+torch.sin(x[0]*x[2])
    #TODO


def compute_hessian(func: Callable[[torch.Tensor], torch.Tensor],
                    x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.functional.hessian(func,x)
    #TODO


if __name__ == "__main__":
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    H = compute_hessian(target_function, x)
    print(H)