import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import math
from dataclasses import dataclass

from cvmm import cvmm, cvmm_prepare_sel2, CVMMSel
from moeut_code import (
    AttentionMask, MoEUTOutput, MultilayerKVCache, KVCache,
    entropy_reg, RotaryPosEncoding, SigmaMoE
)

@dataclass
class AdaptiveMoEUTOutput:
    outputs: torch.Tensor
    reg_loss: torch.Tensor
    cache: MultilayerKVCache
    tokens_halted_at: torch.Tensor
    halt_logits: list[torch.Tensor]


class SwitchHeadCore(torch.nn.Module):
    def __init__(self, state_size: int, n_heads: int, n_experts: int, dropout: float = 0.0,
                 projection_size: Optional[int] = None, expert_dropout: float = 0.0, moe_k: int = 2):
        super().__init__()
        self.input_size = state_size
        self.output_size = state_size
        self.pe_size = self.input_size
        self.expert_dropout = expert_dropout
        self.moe_k = moe_k
        self.attention_to_visualize = []
        self.selections_to_visualize = {}
        self.n_experts = n_experts

        self.sel_hist = []

        self.n_heads = n_heads
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else lambda x: x
        self.projection_size = projection_size or (state_size // n_heads)

        self.q = torch.nn.Linear(self.input_size, self.projection_size * self.n_heads, bias=False)
        self.k = torch.nn.Linear(self.input_size, self.projection_size * self.n_heads, bias=False)

        if self.n_experts > 1:
            self.v = torch.nn.Parameter(torch.empty(self.n_heads * self.n_experts, self.input_size, self.projection_size))
            self.o = torch.nn.Parameter(torch.empty(self.n_heads * self.n_experts, self.projection_size, self.output_size))
            self.sel_v = torch.nn.Parameter(torch.empty(self.n_heads * self.n_experts, self.input_size))
        else:
            self.v = torch.nn.Parameter(torch.empty(self.n_heads * self.projection_size, self.input_size))
            self.o = torch.nn.Parameter(torch.empty(self.output_size, self.n_heads * self.projection_size))

        self.sel_o = torch.nn.Parameter(torch.empty(self.n_heads * self.n_experts, self.input_size))
        self.register_buffer("scale", torch.full([1], 1.0 / math.sqrt(self.projection_size)), persistent=False)

    @torch.no_grad
    def reset_parameters(self, std_scale: float):
        if self.n_experts > 1:
            torch.nn.init.normal_(self.sel_v, 0, std_scale / math.sqrt(self.input_size))
            self.renorm_rows(self.sel_v)

        torch.nn.init.normal_(self.sel_o, 0, std_scale / math.sqrt(self.input_size))
        self.renorm_rows(self.sel_o)

        torch.nn.init.normal_(self.k.weight, 0, std_scale / math.sqrt(self.input_size))
        torch.nn.init.normal_(self.q.weight, 0, std_scale / math.sqrt(self.input_size))
        torch.nn.init.normal_(self.v, 0, std_scale / math.sqrt(self.input_size))
        torch.nn.init.normal_(self.o, 0, std_scale / math.sqrt(self.n_heads * self.projection_size))

    def renorm_rows(self, x: torch.Tensor):
        with torch.no_grad():
            std_t = x.std(dim=-1, keepdim=True)
            x.div_(x.norm(dim=-1, keepdim=True))
            x.mul_(std_t / x.std())

    def project_to_torch_order(self, x: torch.Tensor):
        return x.view(*x.shape[:-1], self.n_heads, -1).transpose(-2, -3)

    def get_mask_tensor(self, src_len: int, mask: Optional[AttentionMask]) -> Optional[torch.Tensor]:
        if mask is None or (mask.position_mask is None and mask.src_length_mask is None):
            return None

        if mask.position_mask is not None:
            n_pad = src_len - mask.position_mask.shape[-1]
            if n_pad > 0:
                pm = F.pad(mask.position_mask, (n_pad, 0), 'constant', value=False)
            else:
                pm = mask.position_mask

        if mask.position_mask is None:
            m = mask.src_length_mask.unsqueeze(-2).unsqueeze(-2)
        elif mask.src_length_mask is None:
            m = pm
        else:
            m = mask.src_length_mask.unsqueeze(-2).unsqueeze(-2) | pm
        return m

    def get_sel(self, t: torch.Tensor, w: torch.Tensor) -> Tuple[CVMMSel, torch.Tensor]:
        sel = F.linear(t, w).float()
        sel = sel_raw = sel.view(*sel.shape[:-1], self.n_heads, -1)
        sel = sel.sigmoid()

        with torch.no_grad():
            if self.expert_dropout > 0 and self.training:
                mask = torch.rand_like(sel) < self.expert_dropout
                sel2 = sel.masked_fill(mask, float('-inf'))
            else:
                sel2 = sel
            _, sel_index = sel2.topk(self.moe_k, dim=-1, sorted=False)
        sel_val = torch.gather(sel, -1, sel_index)

        sel_index_shifted = (torch.arange(self.n_heads, device=sel_index.device, dtype=sel_index.dtype) * self.n_experts).unsqueeze(-1) + sel_index
        return cvmm_prepare_sel2(sel_index_shifted.flatten(-2,-1), sel_val), sel_raw

    def attend(self, pos_offset: int, v: torch.Tensor, k: torch.Tensor, q: torch.Tensor,
               mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def get_reg_loss(self) -> torch.Tensor:
        loss = 0
        if self.sel_hist:
            for i in range(len(self.sel_hist[0])):
                all_sels = torch.cat([l[i] for l in self.sel_hist], dim=0)
                loss = loss + entropy_reg(all_sels.flatten(-2, -1), -2)
        self.sel_hist = []
        return loss

    def forward(self, q_src: torch.Tensor, k_src: torch.Tensor, v_src: torch.Tensor, mask: Optional[AttentionMask],
                active_mask: torch.Tensor | None = None, kv_cache: KVCache = None, loop_idx: int = 0) -> Tuple[torch.Tensor, KVCache]:
        scale = self.scale.sqrt()

        B, T, _ = q_src.shape
        out_dim = self.projection_size * self.n_heads
        
        q = torch.zeros(B, T, out_dim, device=q_src.device, dtype=q_src.dtype)
        if active_mask is not None and active_mask.any():
            q_active = self.q(q_src[active_mask])
            q[active_mask] = q_active
        q = q * scale.type_as(q)

        k = torch.zeros(B, T, out_dim, device=k_src.device, dtype=k_src.dtype)
        if active_mask is not None and active_mask.any():
            k[active_mask] = self.k(k_src[active_mask])
        k = k * scale.type_as(k)

        if self.n_experts > 1:
            v_sel, v_sel_r = self.get_sel(k_src, self.sel_v)
            
            if active_mask is not None and active_mask.any():
                q_src_active = q_src[active_mask]
                o_sel_active, o_sel_r_active = self.get_sel(q_src_active, self.sel_o) 
            else:
                o_sel_active, o_sel_r_active = self.get_sel(q_src, self.sel_o)
            
            if self.training:
                self.sel_hist.append((o_sel_r_active, v_sel_r.view(-1, *v_sel_r.shape[2:])))

            v = cvmm(v_src, v_sel, self.v).transpose(-2, -3)
        else:
            o_gate = F.sigmoid(F.linear(q_src, self.sel_o))
            v = self.project_to_torch_order(F.linear(v_src, self.v))

        q = self.project_to_torch_order(q)
        k = self.project_to_torch_order(k)

        if kv_cache is not None:
            if loop_idx == 0:
                if "k" not in kv_cache:
                    kv_cache["k"] = k
                    kv_cache["v"] = v
                    kv_cache["T_curr"] = k.shape[-2]
                else:
                    kv_cache["k"] = torch.cat([kv_cache["k"], k], dim=-2)
                    kv_cache["v"] = torch.cat([kv_cache["v"], v], dim=-2)
                    kv_cache["T_curr"] = k.shape[-2]
            else:
                T_curr = kv_cache.get("T_curr", k.shape[-2])
                
                if active_mask is not None and active_mask.any():
                    b_mask = active_mask.view(B, 1, T_curr, 1)
                    
                    curr_k = kv_cache["k"][..., -T_curr:, :]
                    curr_v = kv_cache["v"][..., -T_curr:, :]

                    kv_cache["k"][..., -T_curr:, :] = torch.where(b_mask, k, curr_k)
                    kv_cache["v"][..., -T_curr:, :] = torch.where(b_mask, v, curr_v)

            k_att = kv_cache["k"]
            v_att = kv_cache["v"]
        else:
            k_att = k
            v_att = v

        q = self.dropout(q)
        
        pos_offset = k_att.shape[-2] - q.shape[-2]
        res = self.attend(pos_offset, v_att, k_att, q, self.get_mask_tensor(v_att.shape[-2], mask))
        res = res.transpose(-2, -3)

        if self.n_experts > 1:
            if active_mask is not None and active_mask.any():
                res_active = res[active_mask]

                o_sel_active.sel_index = o_sel_active.out_index // o_sel_active.reduction_weight.shape[-1]
                o_sel_active.reduction_weight = o_sel_active.reduction_weight.flatten(-2)
                
                out_active = cvmm(res_active, o_sel_active, self.o)
                
                B, T = res.shape[0], res.shape[1]
                out = torch.zeros(B, T, self.output_size, device=res.device, dtype=res.dtype)
                out[active_mask] = out_active
            else:
                o_sel_active.sel_index = o_sel_active.out_index // o_sel_active.reduction_weight.shape[-1]
                o_sel_active.reduction_weight = o_sel_active.reduction_weight.flatten(-2)
                out = cvmm(res, o_sel_active, self.o)
        else:
            res = res * o_gate[..., None]
            out = F.linear(res.flatten(-2), self.o)

        return out, kv_cache

class SwitchHeadRope(SwitchHeadCore):
    def __init__(self, state_size: int, n_heads: int, n_experts: int, dropout: float = 0.0,
                 projection_size: Optional[int] = None, expert_dropout: float = 0.0, moe_k: int = 2,
                 rotate_fraction: float = 0.5, rope_base: float = 10000):

        super().__init__(
            state_size, n_heads, n_experts, dropout, projection_size, expert_dropout, moe_k)

        self.n_rotate = int(rotate_fraction * self.projection_size)
        if self.n_rotate > 0:
            self.pe = RotaryPosEncoding(self.n_rotate, seq_dim=-2, base=rope_base)

    def rotate(self, q: torch.Tensor, k: torch.Tensor, offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n_rotate < self.projection_size:
            r_k = k[..., :self.n_rotate]
            nr_k = k[..., self.n_rotate:]
            r_q = q[..., :self.n_rotate]
            nr_q = q[..., self.n_rotate:]

            r_q, r_k = self.pe(r_q, r_k, offset)
            return torch.cat([r_q, nr_q], dim=-1), torch.cat([r_k, nr_k], dim=-1)
        else:
            return self.pe(q, k, offset)

    def attend(self, pos_offset: int, v: torch.Tensor, k: torch.Tensor, q: torch.Tensor,
               mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n_rotate > 0:
            q, k = self.rotate(q, k, pos_offset or 0)
        return F.scaled_dot_product_attention(q, k, v, ~mask, scale=1.0)

class HaltingHead(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(d_model, 1, bias=True),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

    @torch.no_grad
    def reset_parameters(self, init_bias: float) -> None:
        torch.nn.init.zeros_(self.layer[0].weight)
        torch.nn.init.constant_(self.layer[0].bias, init_bias)


class AdaptiveMoEUTLayer(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_expert_size: int, ff_n_experts: int,
                 att_n_experts: int, d_head: Optional[int] = None, att_k: int = 2,
                 ff_k: int = 8, ff_expert_dropout: float = 0.0, att_expert_dropout: float = 0.0,
                 dropout: float = 0.0, attention = SwitchHeadRope):
        super().__init__()
        self.attention = attention(
            d_model, n_heads, att_n_experts, projection_size=d_head, moe_k=att_k,
            expert_dropout=att_expert_dropout)
        self.ffn = SigmaMoE(d_model, ff_n_experts, ff_expert_size, k=ff_k, expert_dropout=ff_expert_dropout)
        self.ln1 = torch.nn.LayerNorm(d_model)      # LayerNorm for x - Q in attention
        self.ln_s = torch.nn.LayerNorm(d_model)     # LayerNorm for s - K in attention
        self.ln2 = torch.nn.LayerNorm(d_model)      # LayerNorm for x - FF selection
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, s: torch.Tensor, active_mask: torch.Tensor, 
        mask: Optional[AttentionMask] = None, kv_cache: KVCache = None, loop_idx: int = 0) -> Tuple[torch.Tensor, KVCache]:
        
        xnorm = torch.zeros_like(x)
        if active_mask.any():
            xnorm[active_mask] = self.ln1(x[active_mask])
        
        snorm = self.ln_s(s)

        att, kv_cache = self.attention(q_src=xnorm, k_src=snorm, v_src=s, mask=mask, 
                                       active_mask=active_mask, kv_cache=kv_cache, loop_idx=loop_idx)
        
        x = x + self.drop(att)
        
        if active_mask.any():
            x_active = x[active_mask]
            x_active_normed = self.ln2(x_active)
            upd_active = self.ffn(x_active, x_active_normed)
            x[active_mask] = x_active + upd_active
        
        return x, kv_cache


class AdaptiveMoEUT(torch.nn.Module):
    def __init__(self, d_model: int, max_loops: int, n_heads: int, ff_expert_size: int, ff_n_experts: int,
                 att_n_experts: int, d_head: Optional[int] = None, att_k: int = 2,
                 ff_k: int = 8, ff_expert_dropout: float = 0.0, att_expert_dropout: float = 0.0,
                 dropout: float = 0.0, entropy_reg: float = 0.01, att_entropy_reg: float = 0.001,
                 attention = SwitchHeadRope, group_size: int = 2, init_halt_bias: float = -3, default_halt_thresh: float = 0.999):
        super().__init__()

        self.entropy_reg = entropy_reg
        self.att_entropy_reg = att_entropy_reg

        self.max_loops = max_loops
        self.layers = torch.nn.ModuleList([
            AdaptiveMoEUTLayer(d_model, n_heads, ff_expert_size, ff_n_experts, att_n_experts, d_head, att_k, ff_k,
                       ff_expert_dropout, att_expert_dropout, dropout, attention)
            for _ in range(group_size)
        ])

        self.halt_head = HaltingHead(d_model)
        self.init_halt_bias = init_halt_bias
        self.default_halt_thresh = default_halt_thresh

        self.reset_parameters()

    def forward(self, x: torch.Tensor, halt_thresh: float = None, mask: Optional[AttentionMask] = None,
                kv_cache: MultilayerKVCache = None) -> MoEUTOutput:
        halt_thresh = halt_thresh if halt_thresh is not None else self.default_halt_thresh
        
        B, T, _ = x.shape
        token_batch_shape = (B, T, 1)

        expected_loops = torch.zeros((B, T), device=x.device, dtype=x.dtype)
        tokens_halted_at = torch.zeros(self.max_loops, device=x.device, dtype=torch.int)
        halt_logits = []
        
        accum_alpha = torch.zeros(token_batch_shape, device=x.device, dtype=x.dtype)
        weighted_prev_x = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        
        s = x.clone()
        old_active_mask = torch.zeros((B, T), device=x.device, dtype=torch.bool)
        active_mask = torch.ones((B, T), device=x.device, dtype=torch.bool)
        new_cache = kv_cache.copy() if kv_cache is not None else {}

        for loop in range(self.max_loops):
            for layer_idx,  layer in enumerate(self.layers):
                cache_in = new_cache.get(layer_idx, {})
                x, cache_out = layer(x, s, active_mask, mask, cache_in, loop_idx=loop)
                new_cache[layer_idx] = cache_out
            
            alpha_hat = torch.zeros(token_batch_shape, device=x.device, dtype=x.dtype)
            active_logits = self.halt_head(x[active_mask])
            alpha_hat[active_mask] = active_logits
            halt_logits.append(active_logits.detach().float())

            alpha = torch.zeros(token_batch_shape, device=x.device, dtype=x.dtype)
            alpha[active_mask] = alpha_hat[active_mask] * (1 - accum_alpha[active_mask])

            s[active_mask] = (1 - accum_alpha[active_mask]) * x[active_mask] + weighted_prev_x[active_mask]

            accum_alpha[active_mask] += alpha[active_mask]
            weighted_prev_x[active_mask] += alpha[active_mask] * x[active_mask]
            
            expected_loops[active_mask] += alpha[active_mask].squeeze(-1) * (loop + 1)

            old_active_mask = active_mask.clone()
            active_mask = (accum_alpha < halt_thresh).squeeze(-1)
            
            just_halted = old_active_mask & ~active_mask
            tokens_halted_at[loop] = just_halted.sum()

            if not active_mask.any():
                break

        remainder = 1.0 - accum_alpha.squeeze(-1)
        expected_loops += remainder * self.max_loops
        
        if active_mask.any():
            tokens_halted_at[self.max_loops - 1] += active_mask.sum()

        reg_loss = expected_loops.mean()
        for layer in self.modules():
            if isinstance(layer, SigmaMoE):
                reg_loss = reg_loss + self.entropy_reg * layer.get_reg_loss()
            elif isinstance(layer, SwitchHeadCore):
                reg_loss = reg_loss + self.att_entropy_reg * layer.get_reg_loss()

        return AdaptiveMoEUTOutput(s, reg_loss, new_cache, tokens_halted_at, halt_logits)

    @torch.no_grad
    def reset_parameters(self):
        scale = math.sqrt(2 / (self.max_loops * len(self.layers)))
        for layer in self.modules():
            if isinstance(layer, (SwitchHeadCore, SigmaMoE)):
                layer.reset_parameters(scale)
            elif isinstance(layer, HaltingHead):
                layer.reset_parameters(self.init_halt_bias)
            elif isinstance(layer, torch.nn.LayerNorm):
                torch.nn.init.ones_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
            
class AdaptiveMoEUTLM(torch.nn.Module):
    def __init__(self, n_tokens: int, d_model: int, max_loops: int, n_heads: int,
                 ff_n_experts: int, att_n_experts: int, d_head: Optional[int] = None,
                 group_size: int = 2, ff_k: int = 8,  att_k: int = 2, ff_expert_dropout: float = 0.0,
                 att_expert_dropout: float = 0.0, ff_expert_size: int = 128, dropout: float = 0.0, 
                 entropy_reg: float = 0.01, att_entropy_reg: float = 0.001, attention = SwitchHeadRope,
                 init_halt_bias: float = -3, default_halt_thresh: float = 0.999):
        super().__init__()
        self.transformer = AdaptiveMoEUT(d_model, max_loops, n_heads, ff_expert_size, ff_n_experts, att_n_experts,
                                 d_head, att_k, ff_k, ff_expert_dropout, att_expert_dropout, dropout,
                                 entropy_reg, att_entropy_reg, attention, group_size, init_halt_bias, default_halt_thresh)

        self.max_loops = max_loops
        self.embedding = torch.nn.Embedding(n_tokens, d_model)
        self.lm_head = torch.nn.Linear(d_model, n_tokens)
        self.out_norm = torch.nn.LayerNorm(d_model)
        self.reset_parameters()

    @torch.no_grad
    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.embedding.weight, mode="fan_in", nonlinearity="linear")
        self.transformer.reset_parameters()

    def forward(self, x: torch.Tensor, mask: Optional[AttentionMask] = None,
                kv_cache: MultilayerKVCache = None) -> AdaptiveMoEUTOutput:

        if mask is None:
            mask = AttentionMask(None, self.generate_causal_attention_mask(x.shape[-1]))

        x = self.embedding(x)
        out = self.transformer(x=x, mask=mask, kv_cache=kv_cache)
        out.outputs = self.lm_head(self.out_norm(out.outputs))
        return out

    def generate_causal_attention_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=self.lm_head.weight.device), diagonal=1)