"""
Lightweight Residual-in-Residual Dense Block (RRDB) network for joint
denoising + 2x super-resolution (128x128 -> 256x256, or 256x256 -> 512x512;
both are 2x per the problem statement).

Sized deliberately small (nf=48, nb=6 by default) so training fits
comfortably in 6GB of VRAM on a laptop GPU (RTX 3050). Scale it up
(nf=64, nb=12+) later if you move to a bigger GPU / cloud instance —
the architecture is the same family used in ESRGAN / Real-ESRGAN, just
trimmed down.

CHANGES vs. original:
  1. Global residual (bicubic) skip connection: the network output is
     now `bicubic_upsample(x) + learned_residual` instead of a mapping
     learned entirely from scratch. This is the standard EDSR/ESPCN-
     style trick — it gives the network an easy "identity + correction"
     starting point instead of having to reconstruct the whole image,
     which speeds convergence and typically improves PSNR/SSIM at a
     given parameter budget.
  2. Sigmoid output activation removed. It was saturating gradients
     near 0/1 (where most pixel mass sits) and fighting the skip
     connection above. Downstream code already does `.clamp(0, 1)` on
     the output everywhere it matters (metrics.py, evaluate_submission.py,
     make_results_report.py), so nothing else needs to change.
  3. ICNR initialization on the pre-PixelShuffle convs, which avoids
     the checkerboard-artifact pattern that random init can produce in
     sub-pixel upsampling layers (Aitken et al., 2017).

Note: nn.Sigmoid has no parameters, so removing it does not change any
state_dict key — old checkpoints still *load* into this architecture
without error. The forward computation is different though (skip
connection + no squashing), so a checkpoint trained with the old
architecture should be retrained/fine-tuned rather than treated as a
drop-in replacement for scoring.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def icnr_init(conv, upscale_factor=2, initializer=nn.init.kaiming_normal_):
    """Initialize a pre-PixelShuffle conv so that, at init, every output
    sub-pixel channel starts as a copy of the same base filter -- this is
    what a "do nothing extra, just nearest-neighbor upsample" init looks
    like, and it removes the random per-subpixel-position noise that
    causes checkerboard artifacts (https://arxiv.org/abs/1707.02937)."""
    out_ch, in_ch, h, w = conv.weight.shape
    sub_ch = out_ch // (upscale_factor ** 2)
    kernel = torch.zeros([sub_ch, in_ch, h, w])
    initializer(kernel)
    kernel = kernel.repeat_interleave(upscale_factor ** 2, dim=0)
    conv.weight.data.copy_(kernel)


class ResidualDenseBlock(nn.Module):
    """5-conv dense block with residual scaling (beta), as in ESRGAN."""

    def __init__(self, nf=48, gc=24, beta=0.2):
        super().__init__()
        self.beta = beta
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + x5 * self.beta


class RRDB(nn.Module):
    """3 stacked RDBs with an outer residual connection."""

    def __init__(self, nf=48, gc=24, beta=0.2):
        super().__init__()
        self.beta = beta
        self.rdb1 = ResidualDenseBlock(nf, gc, beta)
        self.rdb2 = ResidualDenseBlock(nf, gc, beta)
        self.rdb3 = ResidualDenseBlock(nf, gc, beta)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * self.beta


class RestorationNet(nn.Module):
    """
    in_nc/out_nc: number of channels (1 for grayscale SEM-style images,
        3 for RGB). Auto-detected from the data by train.py.
    scale: upsampling factor. 2 for your 128->256 or 256->512 case. Use 1
        for pure denoising (no resolution change) if you ever train on
        same-resolution pairs.
    nf: base channel width. nb: number of RRDB blocks.
    """

    def __init__(self, in_nc=1, out_nc=1, nf=48, nb=6, gc=24, scale=2):
        super().__init__()
        assert scale in (1, 2, 4), "scale must be 1, 2, or 4"
        self.scale = scale
        self.out_nc = out_nc

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)

        up_layers = []
        n_up = {1: 0, 2: 1, 4: 2}[scale]
        self._upsample_convs = []
        for _ in range(n_up):
            up_conv = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self._upsample_convs.append(up_conv)
            up_layers += [
                up_conv,
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.upsample = nn.Sequential(*up_layers)

        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        for up_conv in self._upsample_convs:
            icnr_init(up_conv, upscale_factor=2)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.upsample(feat) if self.scale != 1 else feat
        residual = self.conv_last(self.lrelu(self.conv_hr(feat)))

        if self.scale != 1:
            # Global skip: bicubic-upsample the (possibly out-of-[0,1],
            # speckle-noisy) input as a base image, then add the learned
            # residual on top. Network only has to learn the correction,
            # not the whole image from scratch.
            base = F.interpolate(x, scale_factor=self.scale, mode="bicubic",
                                  align_corners=False)
        else:
            base = x

        if base.shape[1] != self.out_nc:
            base = base.repeat(1, self.out_nc, 1, 1) if base.shape[1] == 1 \
                else base.mean(dim=1, keepdim=True)

        # No Sigmoid: output is left unclamped for training gradients (the
        # Charbonnier/SSIM losses handle mild out-of-range values fine);
        # every downstream consumer (metrics.py, evaluate_submission.py,
        # make_results_report.py) already clamps to [0,1] before use.
        return base + residual


def build_model(in_nc=1, out_nc=1, scale=2, size="small"):
    """size: 'tiny' (fits any GPU, fastest), 'small' (default, ~2.5M
    params, good fit for 6GB), 'base' (bigger, use if you have more VRAM
    or move training to the cloud)."""
    cfg = {
        "tiny": dict(nf=32, nb=4, gc=16),
        "small": dict(nf=48, nb=6, gc=24),
        "base": dict(nf=64, nb=12, gc=32),
    }[size]
    return RestorationNet(in_nc=in_nc, out_nc=out_nc, scale=scale, **cfg)


if __name__ == "__main__":
    m = build_model(in_nc=1, out_nc=1, scale=2, size="small")
    n_params = sum(p.numel() for p in m.parameters())
    print(f"Params: {n_params/1e6:.2f}M")
    x = torch.randn(2, 1, 128, 128)
    y = m(x)
    print("Input:", x.shape, "Output:", y.shape)
