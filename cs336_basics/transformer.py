import math
import torch
from torch import nn
from einops import einsum

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None) -> None:
        super().__init__()
        weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(weight, mean=0, std=std, a=-3*std, b=3*std)
        self.weight = weight
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(weight, mean=0, std=1, a=-3, b=3)
        self.weight = weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids, :]

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) / self.d_model + self.eps)
        result = x / rms * self.weight
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    """
    FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) * W3x)
    """
    def __init__(
        self, 
        d_model: int, 
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = max(math.ceil(d_model * 8 / 3 / 64), 1) * 64
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    @staticmethod
    def _silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self._silu(self.w1(x)) * self.w3(x))
