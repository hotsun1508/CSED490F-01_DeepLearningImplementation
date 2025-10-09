import torch
from torch import Tensor
from typing import Tuple

# Step 1: Define custom operators using torch.library API
@torch.library.custom_op("my_ops::batchnorm_forward", mutates_args=("running_mean", "running_var"))
def batchnorm_forward(
    input: Tensor,           # [N, C, H, W]
    gamma: Tensor,           # [C]
    beta: Tensor,            # [C]
    running_mean: Tensor,    # [C]
    running_var: Tensor,     # [C]
    training: bool,
    momentum: float,
    eps: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """forward pass of BatchNorm for 4D input [N, C, H, W]."""

    dims = [0, 2, 3] 
    C = input.shape[1]

    if training:
        mean = torch.mean(input, dim=dims)
        var = torch.var(input, dim=dims, unbiased=False)

        running_mean.mul_((1 - momentum)).add_(momentum * mean.detach())
        running_var.mul_((1 - momentum)).add_(momentum * var.detach())

        save_mean = mean
        save_invstd = 1.0 / torch.sqrt(var + eps)
    else:
        # 추론 모드일 때
        mean = running_mean
        var = running_var
        
        # <--- 수정된 부분 ---
        # 입력 텐서(running_mean)를 직접 반환하지 않고, 복사본(.clone())을 반환하도록 수정합니다.
        save_mean = running_mean.clone()
        save_invstd = 1.0 / torch.sqrt(var + eps)

    mean_reshaped = mean.view(1, C, 1, 1)
    invstd_reshaped = save_invstd.view(1, C, 1, 1)
    gamma_reshaped = gamma.view(1, C, 1, 1)
    beta_reshaped = beta.view(1, C, 1, 1)

    normalized_input = (input - mean_reshaped) * invstd_reshaped
    output = normalized_input * gamma_reshaped + beta_reshaped
    
    return output, save_mean, save_invstd


@torch.library.custom_op("my_ops::batchnorm_backward", mutates_args=())
def batchnorm_backward(
    grad_output: Tensor,     # [N, C, H, W]
    input: Tensor,           # [N, C, H, W]
    gamma: Tensor,           # [C]
    save_mean: Tensor,       # [C]
    save_invstd: Tensor      # [C]
) -> Tuple[Tensor, Tensor, Tensor]:
    """backward pass of BatchNorm for 4D input."""

    N, C, H, W = input.shape
    dims = [0, 2, 3] 
    m = N * H * W 

    mean_reshaped = save_mean.view(1, C, 1, 1)
    invstd_reshaped = save_invstd.view(1, C, 1, 1)
    gamma_reshaped = gamma.view(1, C, 1, 1)

    normalized_input = (input - mean_reshaped) * invstd_reshaped

    # --- Calculate Gradients ---
    grad_beta = torch.sum(grad_output, dim=dims)
    grad_gamma = torch.sum(grad_output * normalized_input, dim=dims)

    grad_normalized = grad_output * gamma_reshaped
    
    sum_grad_normalized = torch.sum(grad_normalized, dim=dims)
    sum_grad_normalized_x_hat = torch.sum(grad_normalized * normalized_input, dim=dims)

    sum_grad_normalized = sum_grad_normalized.view(1, C, 1, 1)
    sum_grad_normalized_x_hat = sum_grad_normalized_x_hat.view(1, C, 1, 1)
    
    grad_input = (1.0 / m) * invstd_reshaped * (
        m * grad_normalized - sum_grad_normalized - normalized_input * sum_grad_normalized_x_hat
    )

    return grad_input, grad_gamma, grad_beta


# Step 2: Connect forward and backward with autograd
class BatchNormCustom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma, beta, running_mean, running_var, training, momentum, eps):
        output, save_mean, save_invstd = torch.ops.my_ops.batchnorm_forward(
            input, gamma, beta, running_mean, running_var, training, momentum, eps
        )
        ctx.save_for_backward(input, gamma, save_mean, save_invstd)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, gamma, save_mean, save_invstd = ctx.saved_tensors
        grad_input, grad_gamma, grad_beta = torch.ops.my_ops.batchnorm_backward(
            grad_output, input, gamma, save_mean, save_invstd
        )
        return grad_input, grad_gamma, grad_beta, None, None, None, None, None