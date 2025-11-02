import argparse
import math
import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx
from typing import Literal

# =========================================================
# FFT frontend: 1232-point window -> 617-bin features
# =========================================================
class FFT617(nn.Module):
    """
    Takes (B, T) waveform (real). Applies a 1232-point windowed FFT (Hann).
    Returns (B, 617) real features (magnitude, power, or log-power).
    """
    def __init__(self, n_fft: int = 1232, feat_mode: Literal["mag","pow","logpow"]="mag"):
        super().__init__()
        assert n_fft % 2 == 0, "n_fft must be even so rfft gives N/2+1 bins"
        self.n_fft = n_fft
        self.feat_mode = feat_mode

        # Hann window (registered as buffer so it moves to CUDA with .to(device))
        hann = torch.hann_window(n_fft, periodic=True)
        self.register_buffer("window", hann)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) real waveform; if T > n_fft we use the first n_fft samples.
        # If T < n_fft you could pad (not needed here unless you want).
        B, T = x.shape
        assert T >= self.n_fft, f"Need at least {self.n_fft} samples (got {T})."
        nvtx.range_push("FFT617/window")
        x_frame = x[:, :self.n_fft] * self.window  # (B, n_fft)
        nvtx.range_pop()

        nvtx.range_push("FFT617/rfft")
        X = torch.fft.rfft(x_frame, n=self.n_fft)  # (B, n_fft/2+1) = (B, 617)
        nvtx.range_pop()

        nvtx.range_push("FFT617/feature")
        mag = torch.abs(X)  # (B, 617)
        if self.feat_mode == "mag":
            feat = mag
        elif self.feat_mode == "pow":
            feat = mag * mag
        else:  # "logpow"
            feat = torch.log(mag * mag + 1e-12)
        nvtx.range_pop()
        return feat  # (B, 617)

# =========================================================
# HDC encoder + associative memory head
# =========================================================
class HDCHead(nn.Module):
    """
    617 -> 512 random ±1 projection, optional binarization to ±1,
    then associative memory (512 -> num_classes).
    score_mode: dot or cosine (cosine uses unit vectors for comparison).
    """
    def __init__(self, in_dim=617, hv_dim=512, num_classes=26,
                 score_mode: Literal["dot","cosine"]="dot",
                 binarize: bool = True, seed: int = 0):
        super().__init__()
        self.score_mode = score_mode
        self.binarize = binarize

        g = torch.Generator().manual_seed(seed)
        proj = torch.randint(0, 2, (in_dim, hv_dim), generator=g).float().mul_(2).add_(-1)   # ±1
        class_hv = torch.randint(0, 2, (hv_dim, num_classes), generator=g).float().mul_(2).add_(-1)

        self.register_buffer("proj", proj)         # (617, 512)
        self.register_buffer("class_hv", class_hv) # (512, 26)

        if score_mode == "cosine":
            chv_norm = torch.linalg.vector_norm(self.class_hv, dim=0, keepdim=True).clamp_min(1e-8)
            self.register_buffer("class_hv_unit", self.class_hv / chv_norm)
        else:
            self.register_buffer("class_hv_unit", torch.empty(0))

    @staticmethod
    def _binarize_pm1(x: torch.Tensor) -> torch.Tensor:
        # >=0 -> +1, <0 -> -1 (avoid zeros turning to 0)
        return torch.where(x >= 0, torch.tensor(1.0, device=x.device, dtype=x.dtype),
                              torch.tensor(-1.0, device=x.device, dtype=x.dtype))

    def forward(self, feat617: torch.Tensor):
        # feat617: (B, 617)
        nvtx.range_push("HDC/projection")
        hv = feat617 @ self.proj   # (B, 512)
        nvtx.range_pop()

        if self.binarize:
            nvtx.range_push("HDC/binarize")
            hv_bin = self._binarize_pm1(hv)  # (B, 512)
            nvtx.range_pop()
        else:
            hv_bin = hv

        if self.score_mode == "dot":
            nvtx.range_push("HDC/assoc_mem_dot")
            scores = hv_bin @ self.class_hv  # (B, 26)
            nvtx.range_pop()
        else:
            nvtx.range_push("HDC/assoc_mem_cos")
            hv_norm = torch.linalg.vector_norm(hv_bin, dim=1, keepdim=True).clamp_min(1e-8)
            hv_unit = hv_bin / hv_norm
            scores = hv_unit @ self.class_hv_unit
            nvtx.range_pop()

        return scores, hv_bin

# =========================================================
# Refinement MLP: 26 -> 50 -> 26
# =========================================================
class RefinementMLP(nn.Module):
    def __init__(self, in_dim=26, hidden=50, out_dim=26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=True),
        )

    def forward(self, scores26: torch.Tensor):
        nvtx.range_push("MLP/refine")
        out = self.net(scores26)  # (B, 26)
        nvtx.range_pop()
        return out

# =========================================================
# Full pipeline: waveform -> FFT617 -> HDC -> MLP
# =========================================================
class VoiceHDCHybrid(nn.Module):
    def __init__(self, n_fft=1232, feat_mode="mag",
                 hdc_binarize=True, score_mode="dot", seed=0):
        super().__init__()
        self.fft = FFT617(n_fft=n_fft, feat_mode=feat_mode)
        self.hdc = HDCHead(in_dim=n_fft//2+1, hv_dim=512, num_classes=26,
                           score_mode=score_mode, binarize=hdc_binarize, seed=seed)
        self.mlp = RefinementMLP(in_dim=26, hidden=50, out_dim=26)

    def forward(self, wav: torch.Tensor):
        # wav: (B, T), T >= 1232
        nvtx.range_push("FFT617")
        feat617 = self.fft(wav)                 # (B, 617)
        nvtx.range_pop()

        nvtx.range_push("HDC")
        scores26_raw, hv_bin = self.hdc(feat617)  # (B,26), (B,512)
        nvtx.range_pop()

        nvtx.range_push("MLP")
        scores26_ref = self.mlp(scores26_raw)     # (B,26)
        nvtx.range_pop()
        return scores26_raw, scores26_ref, hv_bin, feat617

# =========================================================
# Driver
# =========================================================
def main():
    ap = argparse.ArgumentParser("Voice -> FFT617 -> HDC -> MLP (inference-only)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--length", type=int, default=4096, help="waveform length T (>=1232)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--feat", type=str, default="mag", choices=["mag","pow","logpow"])
    ap.add_argument("--score", type=str, default="dot", choices=["dot","cosine"])
    ap.add_argument("--no-binarize", action="store_true", help="disable HDC binarization")
    ap.add_argument("--create-on-gpu", action="store_true", help="create waveforms directly on GPU")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    assert args.length >= 1232, "Waveform length must be >= 1232."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VoiceHDCHybrid(n_fft=1232, feat_mode=args.feat,
                           hdc_binarize=(not args.no_binarize),
                           score_mode=args.score, seed=args.seed).to(device).eval()

    # Synthetic "voice-like" wave: sum of a few sinusoids + noise
    def synth_signal(B, T, on_gpu=False):
        t = torch.linspace(0, T-1, T, device=device if on_gpu else "cpu") / 16000.0
        sig = (0.6*torch.sin(2*math.pi*220*t) +
               0.3*torch.sin(2*math.pi*700*t) +
               0.2*torch.sin(2*math.pi*1400*t))
        sig = sig.unsqueeze(0).repeat(B, 1)  # (B,T)
        sig += 0.01*torch.randn(B, T, device=sig.device)
        return sig

    # Warmup
    with torch.inference_mode():
        for _ in range(5):
            wav = synth_signal(args.batch, args.length, on_gpu=args.create_on_gpu)
            if wav.device.type != device.type:
                wav = wav.to(device, non_blocking=True)
            _ = model(wav)
        torch.cuda.synchronize() if device.type == "cuda" else None

    # Profiled loop
    with torch.inference_mode():
        nvtx.range_push("inference_run")
        for _ in range(args.iters):
            wav = synth_signal(args.batch, args.length, on_gpu=args.create_on_gpu)
            if wav.device.type != device.type:
                nvtx.range_push("HtoD")
                wav = wav.to(device, non_blocking=True)
                nvtx.range_pop()
            scores_raw, scores_ref, hv_bin, feat617 = model(wav)
        torch.cuda.synchronize() if device.type == "cuda" else None
        nvtx.range_pop()

    # Sanity print
    print("feat617:", tuple(feat617.shape), "hv_bin:", tuple(hv_bin.shape))
    print("scores_raw:", tuple(scores_raw.shape), "scores_ref:", tuple(scores_ref.shape))
    pred_raw = scores_raw.argmax(dim=1)
    pred_ref = scores_ref.argmax(dim=1)
    print("pred_raw[0]:", pred_raw[0].item(), "pred_ref[0]:", pred_ref[0].item())

if __name__ == "__main__":
    main()

