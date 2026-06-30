import torch

class ACTStatsTracker:
    def __init__(self, max_loops: int, device: torch.device):
        self.max_loops = max_loops
        self.device = device
        self.reset()

    def reset(self):
        self.logit_sums = torch.zeros(self.max_loops, device=self.device, dtype=torch.float32)
        self.logit_sq_sums = torch.zeros(self.max_loops, device=self.device, dtype=torch.float32)
        self.logit_counts = torch.zeros(self.max_loops, device=self.device, dtype=torch.long)
        self.histogram = torch.zeros(self.max_loops, device=self.device, dtype=torch.long)

    @torch.no_grad()
    def update(self, output):
        self.histogram += output.tokens_halted_at

        for loop_idx, loop_logits in enumerate(output.halt_logits):
            if loop_logits.numel() > 0:
                self.logit_sums[loop_idx] += loop_logits.sum()
                self.logit_sq_sums[loop_idx] += (loop_logits ** 2).sum()
                self.logit_counts[loop_idx] += loop_logits.numel()

    @torch.no_grad()
    def log_to_tensorboard(self, writer, step: int):
        values = torch.repeat_interleave(
            torch.arange(self.max_loops, dtype=torch.float32),
            self.histogram.cpu()
        )

        writer.add_histogram(
            "act_depth_distribution",
            values,
            global_step=step
        )
        
        for i in range(self.max_loops):
            count = self.logit_counts[i].item()

            if count > 0:
                mean = self.logit_sums[i] / count
                variance = (self.logit_sq_sums[i] / count) - (mean ** 2)
                std = torch.sqrt(torch.clamp(variance, min=0.0))

                writer.add_scalar(f"act_logits_mean/loop_{i}", mean.item(), step)
                writer.add_scalar(f"act_logits_std/loop_{i}", std.item(), step)

        self.reset()