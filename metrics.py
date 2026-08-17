"""Metrics matching exactly what the hackathon scores: SSIM, pSNR, LPIPS."""
import torch
import torch.nn.functional as F
from losses import SSIMLoss

_ssim_module = SSIMLoss()
_lpips_fn = None


def psnr(pred, target, data_range=1.0):
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    return (10 * torch.log10(data_range ** 2 / mse)).mean().item()


def ssim(pred, target, data_range=1.0):
    return 1.0 - _ssim_module(pred, target, data_range=data_range).item()


def lpips_score(pred, target, in_nc=1):
    global _lpips_fn
    if _lpips_fn is None:
        import lpips
        _lpips_fn = lpips.LPIPS(net="alex")
        if pred.is_cuda:
            _lpips_fn = _lpips_fn.cuda()
    p, t = pred, target
    if in_nc == 1:
        p, t = p.repeat(1, 3, 1, 1), t.repeat(1, 3, 1, 1)
    with torch.no_grad():
        return _lpips_fn(p * 2 - 1, t * 2 - 1).mean().item()


@torch.no_grad()
def evaluate(model, loader, device, in_nc=1, use_lpips=True):
    model.eval()
    tot_psnr = tot_ssim = tot_lpips = n = 0.0
    for lr, gt in loader:
        lr, gt = lr.to(device), gt.to(device)
        pred = model(lr).clamp(0, 1)
        b = lr.size(0)
        tot_psnr += psnr(pred, gt) * b
        tot_ssim += ssim(pred, gt) * b
        if use_lpips:
            tot_lpips += lpips_score(pred, gt, in_nc=in_nc) * b
        n += b
    model.train()
    out = {"psnr": tot_psnr / n, "ssim": tot_ssim / n}
    if use_lpips:
        out["lpips"] = tot_lpips / n
    return out
