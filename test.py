import torch
import inspect
from models.source.adaptive_moeut import AdaptiveMoEUTLM, AdaptiveMoEUTLayer

# ==========================================
# 1. NON-DESTRUCTIVE PATCHING & LOGGING
# ==========================================

# Fallback: Ensure AdaptiveMoEUTLayer accepts loop_idx and passes it to attention
sig = inspect.signature(AdaptiveMoEUTLayer.forward)
if 'loop_idx' not in sig.parameters:
    print("[System] Patching AdaptiveMoEUTLayer.forward to support loop_idx...\n")
    def patched_layer_fwd(self, x, s, active_mask, mask=None, kv_cache=None, loop_idx=0):
        xnorm = torch.zeros_like(x)
        if active_mask.any():
            xnorm[active_mask] = self.ln1(x[active_mask])
        snorm = self.ln_s(s)
        
        # Pass loop_idx down to SwitchHeadCore
        att, kv_cache = self.attention(
            q_src=xnorm, k_src=snorm, v_src=s, mask=mask, 
            active_mask=active_mask, kv_cache=kv_cache, loop_idx=loop_idx
        )
        x = x + self.drop(att)
        
        if active_mask.any():
            x_active = x[active_mask]
            upd_active = self.ffn(x_active, self.ln2(x_active))
            x[active_mask] = x_active + upd_active
        return x, kv_cache
    AdaptiveMoEUTLayer.forward = patched_layer_fwd

def inject_logging_hooks(model):
    """Wraps model methods to print internal loop variables without changing source code."""
    
    # 1. Wrap the Halting Head to get alpha_hat (the raw prediction)
    original_halt_head = model.transformer.halt_head.forward
    def logging_halt_head(x_active):
        alpha_hat = original_halt_head(x_active)
        probs = [round(p, 4) for p in alpha_hat.squeeze(-1).detach().cpu().tolist()]
        print(f"    [Halt Head] New alpha_hat for active tokens: {probs}")
        return alpha_hat
    model.transformer.halt_head.forward = logging_halt_head

    # 2. Wrap the first layer to peek into the loop's local variables
    original_layer_0 = model.transformer.layers[0].forward
    def logging_layer_0(x, s, active_mask, mask=None, kv_cache=None, loop_idx=0):
        print(f"\n  --- Loop {loop_idx} ---")
        
        frame = inspect.currentframe()
        while frame:
            if 'accum_alpha' in frame.f_locals and frame.f_locals['accum_alpha'] is not None:
                # We found the AdaptiveMoEUT.forward frame!
                accum_alpha = frame.f_locals['accum_alpha']
                expected_loops = frame.f_locals['expected_loops']
                weighted_prev_x = frame.f_locals['weighted_prev_x']
                
                # Format for clean console output (taking batch 0)
                accum = [round(v, 4) for v in accum_alpha[0].squeeze(-1).detach().cpu().tolist()]
                loops = [round(v, 4) for v in expected_loops[0].detach().cpu().tolist()]
                
                # Calculate L2 norm of the weighted state vector to prove it is accumulating
                w_norms = [round(v, 4) for v in weighted_prev_x[0].norm(dim=-1).detach().cpu().tolist()]
                
                print(f"    [ACT] accum_alpha:      {accum}")
                print(f"    [ACT] expected_loops:   {loops}")
                print(f"    [ACT] prev_x (L2 Norm): {w_norms}")
                break
            frame = frame.f_back
        
        mask_list = active_mask.detach().cpu().int().tolist()
        if len(mask_list) > 0 and isinstance(mask_list[0], list):
            mask_list = mask_list[0] # Flatten batch dim for reading
        print(f"    [Active Mask] {mask_list}")
        
        out_x, out_cache = original_layer_0(x, s, active_mask, mask, kv_cache, loop_idx)
        
        if out_cache is not None and "k" in out_cache:
            print(f"    [KV Cache] 'k' shape: {list(out_cache['k'].shape)}")
            
        return out_x, out_cache
    model.transformer.layers[0].forward = logging_layer_0

    # 3. Hook the main transformer forward to print the final Expected Loops & Reg Loss
    original_transformer_fwd = model.transformer.forward
    def logging_transformer_fwd(*args, **kwargs):
        out = original_transformer_fwd(*args, **kwargs)
        print(f"\n  >>> [Final ACT Output] reg_loss (mean loops + entropy): {out.reg_loss.item():.4f}")
        return out
    model.transformer.forward = logging_transformer_fwd


# ==========================================
# 2. INITIALIZATION & SETUP
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Initializing AdaptiveMoEUTLM...")
model = AdaptiveMoEUTLM(
    n_tokens=10, 
    d_model=32, 
    max_loops=3,     
    n_heads=2, 
    ff_n_experts=2, 
    att_n_experts=2,
    ff_k=1,
    att_k=1,
    d_head=16,
    group_size=1
)

model.reset_parameters()
inject_logging_hooks(model)

model = model.to(device)
model.eval()

# ==========================================
# 3. GENERATION TEST
# ==========================================

prompt = torch.randint(0, 10, (1, 4), device=device)
print(f"\n[Initial Prompt on GPU]: {prompt.tolist()[0]}")

kv_cache = {}
generated = prompt.clone()

with torch.no_grad():
    print("\n==================================")
    print("=== PREFILL PHASE (4 Tokens) ===")
    print("==================================")
    out = model(prompt, kv_cache=kv_cache)
    
    kv_cache = out.cache

    # Get next token prediction
    next_token = out.outputs[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)
    
    print("\n==================================")
    print("=== AUTOREGRESSIVE GENERATION ===")
    print("==================================")
    for step in range(1, 5):
        print(f"\n>>> Generation Step {step} (Feeding 1 token) <<<")
        current_token = generated[:, -1:] 
        
        out = model(current_token, kv_cache=kv_cache)
        
        next_token = out.outputs[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        
print(f"\n[Final Generated Sequence]: {generated.tolist()[0]}")