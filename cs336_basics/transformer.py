import math
import torch
from torch import nn
from einops import einsum, rearrange

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

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ) -> None:
        super().__init__()
        radian = torch.zeros(max_seq_len, int(d_k/2), device=device)
        for i in range(max_seq_len):
            for k in range(1, int(d_k/2) + 1):
                radian[i,k-1] = i / theta ** ((2*k - 2) / d_k)
        cos = torch.cos(radian)
        sin = torch.sin(radian)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len)
        """
        if token_positions.max() >= self.max_seq_len:
            raise ValueError(f"Token position ({token_positions.max()}) exceeds maximum {self.max_seq_len}")
        if token_positions.min() < 0:
            raise ValueError("Token positions cannot be negative")
        if x.shape[-2] != token_positions.shape[-1]:
            raise ValueError(f"Sequence lengths of x ({x.shape[-2]}) and token_positions ({token_positions.shape[-1]}) don't match")
        ret = x.clone()
        f1 = x[..., ::2]
        f2 = x[..., 1::2]
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        ret[..., ::2] = f1 * cos - f2 * sin
        ret[..., 1::2] = f1 * sin + f2 * cos
        
        # old loop version
        # for i in token_positions:
        #     for k in range(1, int(x.shape[-1]/2) + 1):
        #         f1 = x[..., i, 2*k-1 - 1]
        #         f2 = x[..., i, 2*k - 1]
        #         cos = self.cos[i, k-1]
        #         sin = self.sin[i, k-1]
        #         ret[..., i, 2*k-1 - 1] = f1 * cos - f2 * sin
        #         ret[..., i, 2*k - 1] = f1 * sin + f2 * cos
        return ret

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    shifted = x - x.amax(dim=dim, keepdim=True)
    exp = shifted.exp()
    return exp / exp.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    queries: (batch_size, ..., seq_len, d_k)
    keys: (batch_size, ..., seq_len, d_k)
    values: (batch_size, ..., seq_len, d_v)
    mask: (seq_len, seq_len), torch.bool. Attention probability set to 0 if mask value is False
    """
    d_k = keys.shape[-1]
    attention_scores = einsum(queries, keys, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k") / math.sqrt(d_k)
    if mask is not None:
        attention_scores.masked_fill_(~mask, -math.inf)
    attentoin_weights = softmax(attention_scores, dim=-1)
    ret = einsum(attentoin_weights, values, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")
    return ret

class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None
    ) -> None:
        super().__init__()
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)
        d_k = d_model // num_heads
        if d_model % num_heads != 0:
            raise ValueError(f"model dimension ({d_model}) is not divisible by number of heads ({num_heads})")
        if theta is not None:
            if max_seq_len is None:
                raise ValueError("need max_seq_len to perform RoPE")
            if d_k % 2 != 0:
                raise ValueError(f"d_k ({d_k}) needs to be an even number to perform RoPE")
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
        else:
            self.rope = None
        self.d_model = d_model
        self.num_heads = num_heads

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x: (..., seq_len, d_model)
        return: (..., seq_len, d_model)
        with causal masking and RoPE. max_seq_len, theta, token_positions are for RoPE.
        """
        seq_len = x.shape[-2]
        d_k = self.d_model // self.num_heads
        if x.shape[-1] != self.d_model:
            raise ValueError(f"input's last dim ({x.shape[-1]}) is different from MHA's ({self.d_model})")
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        Q = rearrange(self.q_proj(x), "... T (h d) -> ... h T d", h=self.num_heads, d=d_k)
        K = rearrange(self.k_proj(x), "... T (h d) -> ... h T d", h=self.num_heads, d=d_k)
        V = rearrange(self.v_proj(x), "... T (h d) -> ... h T d", h=self.num_heads, d=d_k)

        if self.rope is not None:
            token_positions = token_positions.unsqueeze(-2) # adds the head dimension: B T -> B h T
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        mha = scaled_dot_product_attention(Q, K, V, mask)
        attn = rearrange(mha, "... h T d -> ... T (h d)", h=self.num_heads, d=d_k)
        out = self.o_proj(attn)
        return out
