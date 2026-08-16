"""
Combined restoration loss: robust pixel loss (Charbonnier) + SSIM loss,
with optional LPIPS perceptual term. The eval metrics (SSIM, pSNR,
LPIPS) are exactly what the hackathon scores you on, so training with a
loss that reflects SSIM (and optionally LPIPS) directly targets the
leaderboard metric rather than only pixel error.

LPIPS is optional and off by default: it needs a pretrained VGG/AlexNet
backbone forward pass every step, which is the single biggest VRAM/time
cost you can add on a 6GB laptop GPU. Turn it on with --use_lpips once
your base model trains cleanly; it noticeably improves perceptual
quality but slows each step down.

CHANGE vs. original: CombinedLoss.forward() used to call
self.pixel_loss(...) and self.ssim_loss(...) twice each -- once to
build `loss` and again to populate the `logs` dict -- doubling the
compute of both terms (SSIM's 6 conv2d passes in particular) on every
single training step for no benefit. Fixed to compute each term once
and reuse the tensor for both the loss and the log.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Smooth, robust L1 variant — less sensitive to outlier pixels
    (helpful here since NoisyLR has speckle spikes outside [0,1])."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


def _gaussian_window(window_size, sigma, channels, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = g.t() @ g
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


class SSIMLoss(nn.Module):
    """1 - SSIM, computed with a Gaussian window. Self-contained (no
    external dependency needed beyond torch)."""

    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self._window_cache = {}

    def _get_window(self, channels, device):
        key = (channels, device)
        if key not in self._window_cache:
            self._window_cache[key] = _gaussian_window(self.window_size, self.sigma, channels, device)
        return self._window_cache[key]

    def forward(self, pred, target, data_range=1.0):
        # SSIM's variance terms (E[x^2] - E[x]^2) involve subtracting nearly-equal
        # numbers — numerically sensitive. Force fp32 here regardless of the
        # surrounding autocast context (autocast would otherwise still downcast
        # these convs to bf16/fp16 even if inputs are already float32).
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred = pred.float()
            target = target.float()
            channels = pred.shape[1]
            window = self._get_window(channels, pred.device)
            pad = self.window_size // 2

            mu_p = F.conv2d(pred, window, padding=pad, groups=channels)
            mu_t = F.conv2d(target, window, padding=pad, groups=channels)
            mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t

            sigma_p2 = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_p2
            sigma_t2 = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_t2
            sigma_pt = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_pt

            c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
            ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / \
                       ((mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2))
            return 1.0 - ssim_map.mean()


class CombinedLoss(nn.Module):
    def __init__(self, w_pixel=1.0, w_ssim=0.2, w_lpips=0.0, in_nc=1):
        super().__init__()
        self.pixel_loss = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.w_pixel, self.w_ssim, self.w_lpips = w_pixel, w_ssim, w_lpips
        self.in_nc = in_nc
        self.lpips_fn = None
        if w_lpips > 0:
            import lpips  # pip install lpips
            self.lpips_fn = lpips.LPIPS(net="alex")
            for p in self.lpips_fn.parameters():
                p.requires_grad_(False)

    def forward(self, pred, target):
        pixel = self.pixel_loss(pred, target)      # computed once
        ssim_l = self.ssim_loss(pred, target)       # computed once
        loss = self.w_pixel * pixel + self.w_ssim * ssim_l
        logs = {"pixel": pixel.item(), "ssim_loss": ssim_l.item()}

        if self.lpips_fn is not None:
            p, t = pred, target
            if self.in_nc == 1:  # LPIPS expects 3-channel input
                p, t = p.repeat(1, 3, 1, 1), t.repeat(1, 3, 1, 1)
            lp = self.lpips_fn(p * 2 - 1, t * 2 - 1).mean()  # LPIPS wants [-1,1]
            loss = loss + self.w_lpips * lp
            logs["lpips"] = lp.item()
        return loss, logs
