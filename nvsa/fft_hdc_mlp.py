import argparse
import math
import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx
from typing import Literal

# =========================================================
# FFT frontend: configurable window -> n_bins features
# Default: 1022-point window -> 512-bin features
# =========================================================
class FFTFrontend(nn.Module):
    """
    Takes (B, T) waveform (real). Applies a windowed FFT (Hann).
    Returns (B, n_bins) real features (magnitude, power, or log-power).

    Args:
        n_bins   : Number of output frequency bins (default 512).
                   Drives n_fft = 2*(n_bins-1) unless n_fft is set explicitly.
        n_fft    : FFT window length. If None, derived from n_bins.
                   Must satisfy n_fft // 2 + 1 == n_bins.
        feat_mode: One of "mag", "pow", "logpow".
    """
    def __init__(
        self,
        n_bins: int = 512,
        n_fft: int | None = None,
        feat_mode: Literal["mag", "pow", "logpow"] = "mag",
    ):
        super().__init__()
        # Derive n_fft from n_bins if not supplied
        if n_fft is None:
            n_fft = 2 * (n_bins - 1)          # e.g. 512 bins -> 1022-pt FFT
        assert n_fft % 2 == 0, "n_fft must be even so rfft gives n_fft//2+1 bins"
        assert n_fft // 2 + 1 == n_bins, (
            f"n_fft={n_fft} gives {n_fft//2+1} bins, but n_bins={n_bins} was requested. "
            "Set n_fft = 2*(n_bins-1) or leave n_fft=None to auto-derive."
        )
        self.n_fft     = n_fft
        self.n_bins    = n_bins
        self.feat_mode = feat_mode

        # Hann window (moved to device with the model via register_buffer)
        self.register_buffer("window", torch.hann_window(n_fft, periodic=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T) real waveform, T >= n_fft
        returns: (B, n_bins)
        """
        B, T = x.shape
        assert T >= self.n_fft, (
            f"FFTFrontend needs at least {self.n_fft} samples, got {T}."
        )

        nvtx.range_push("FFT/window")
        x_frame = x[:, : self.n_fft] * self.window   # (B, n_fft)
        nvtx.range_pop()

        nvtx.range_push("FFT/rfft")
        X = torch.fft.rfft(x_frame, n=self.n_fft)    # (B, n_bins)
        nvtx.range_pop()

        nvtx.range_push("FFT/feature")
        mag = torch.abs(X)
        if self.feat_mode == "mag":
            feat = mag
        elif self.feat_mode == "pow":
            feat = mag * mag
        else:  # "logpow"
            feat = torch.log(mag * mag + 1e-12)
        nvtx.range_pop()

        return feat   # (B, n_bins)


# =========================================================
# HDC encoder + associative memory head
# =========================================================
class HDCHead(nn.Module):
    """
    Binarize (B, in_dim) to ±1, then multiply by a (in_dim, num_classes)
    random ±1 class-HV matrix -> (B, num_classes) scores.

    Args:
        in_dim      : Feature dimension coming in (default 512).
        num_classes : Number of class hypervectors / output scores (default 32).
        score_mode  : "dot" or "cosine".
        binarize    : If True, binarize the input to ±1 before scoring.
        seed        : RNG seed for reproducible class-HV init.
    """
    def __init__(
        self,
        in_dim: int = 512,
        num_classes: int = 32,
        score_mode: Literal["dot", "cosine"] = "dot",
        binarize: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.score_mode  = score_mode
        self.binarize    = binarize
        self.in_dim      = in_dim
        self.num_classes = num_classes

        g = torch.Generator().manual_seed(seed)
        # Class HV matrix: (in_dim, num_classes) of ±1
        class_hv = (
            torch.randint(0, 2, (in_dim, num_classes), generator=g)
            .float()
            .mul_(2)
            .add_(-1)
        )
        self.register_buffer("class_hv", class_hv)   # (512, 32)

        if score_mode == "cosine":
            chv_norm = torch.linalg.vector_norm(
                class_hv, dim=0, keepdim=True
            ).clamp_min(1e-8)
            self.register_buffer("class_hv_unit", class_hv / chv_norm)
        else:
            self.register_buffer("class_hv_unit", torch.empty(0))

    @staticmethod
    def _binarize_pm1(x: torch.Tensor) -> torch.Tensor:
        """Map values: >=0 -> +1, <0 -> -1."""
        return torch.where(
            x >= 0,
            torch.tensor(1.0, device=x.device, dtype=x.dtype),
            torch.tensor(-1.0, device=x.device, dtype=x.dtype),
        )

    def forward(self, feat: torch.Tensor):
        """
        feat : (B, in_dim)
        returns: scores (B, num_classes), hv_bin (B, in_dim)
        """
        if self.binarize:
            nvtx.range_push("HDC/binarize")
            hv_bin = self._binarize_pm1(feat)   # (B, in_dim) -> ±1
            nvtx.range_pop()
        else:
            hv_bin = feat

        if self.score_mode == "dot":
            nvtx.range_push("HDC/assoc_mem_dot")
            scores = hv_bin @ self.class_hv     # (B, num_classes)
            nvtx.range_pop()
        else:
            nvtx.range_push("HDC/assoc_mem_cos")
            hv_norm = torch.linalg.vector_norm(
                hv_bin, dim=1, keepdim=True
            ).clamp_min(1e-8)
            hv_unit = hv_bin / hv_norm
            scores  = hv_unit @ self.class_hv_unit   # (B, num_classes)
            nvtx.range_pop()

        return scores, hv_bin   # (B, num_classes), (B, in_dim)


# =========================================================
# Refinement MLP: (B, in_dim) -> hidden -> (B, out_dim)
# Default: (B, 32) -> 128 -> (B, 32)
# =========================================================
class RefinementMLP(nn.Module):
    """
    Single-hidden-layer MLP for score refinement.

    Args:
        in_dim  : Input feature dimension (default 32).
        hidden  : Hidden layer width (default 128).
        out_dim : Output dimension (default 32).
    """
    def __init__(self, in_dim: int = 32, hidden: int = 128, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=True),
        )

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        scores : (B, in_dim)
        returns: (B, out_dim)
        """
        nvtx.range_push("MLP/refine")
        out = self.net(scores)
        nvtx.range_pop()
        return out


# =========================================================
# Full pipeline: waveform -> FFTFrontend -> HDCHead -> MLP
# =========================================================
class VoiceHDCHybrid(nn.Module):
    """
    End-to-end neuro-symbolic pipeline.

    Data flow:
        wav  (B, T)
          -> FFTFrontend -> feat    (B, n_bins)       default: (B, 512)
          -> HDCHead     -> scores  (B, num_classes)  default: (B, 32)
          -> RefineMLP   -> refined (B, mlp_out)      default: (B, 32)

    All size parameters are forwarded to the respective sub-modules and
    can be overridden here for easy single-point configuration.

    Args:
        n_bins      : FFT output bins (default 512).
        n_fft       : FFT window length; None -> auto-derived from n_bins.
        feat_mode   : FFT feature type: "mag" | "pow" | "logpow".
        hdc_in_dim  : HDC input dim; should match n_bins (default 512).
        num_classes : HDC / MLP class count (default 32).
        score_mode  : HDC scoring: "dot" | "cosine".
        hdc_binarize: Whether to binarize HDC input (default True).
        mlp_hidden  : Hidden width of refinement MLP (default 128).
        mlp_out_dim : Output width of refinement MLP (default 32).
        seed        : RNG seed for HDC class HVs.
    """
    def __init__(
        self,
        n_bins: int = 512,
        n_fft: int | None = None,
        feat_mode: str = "mag",
        hdc_in_dim: int = 512,
        num_classes: int = 32,
        score_mode: str = "dot",
        hdc_binarize: bool = True,
        mlp_hidden: int = 128,
        mlp_out_dim: int = 32,
        seed: int = 0,
    ):
        super().__init__()
        self.fft = FFTFrontend(
            n_bins=n_bins,
            n_fft=n_fft,
            feat_mode=feat_mode,
        )
        self.hdc = HDCHead(
            in_dim=hdc_in_dim,
            num_classes=num_classes,
            score_mode=score_mode,
            binarize=hdc_binarize,
            seed=seed,
        )
        self.mlp = RefinementMLP(
            in_dim=num_classes,
            hidden=mlp_hidden,
            out_dim=mlp_out_dim,
        )

    def forward(self, wav: torch.Tensor):
        """
        wav : (B, T),  T >= n_fft
        returns:
            scores_raw  (B, num_classes)
            scores_ref  (B, mlp_out_dim)
            hv_bin      (B, hdc_in_dim)
            feat        (B, n_bins)
        """
        nvtx.range_push("FFTFrontend")
        feat = self.fft(wav)                          # (B, 512)
        nvtx.range_pop()

        nvtx.range_push("HDC")
        scores_raw, hv_bin = self.hdc(feat)           # (B, 32), (B, 512)
        nvtx.range_pop()

        nvtx.range_push("MLP")
        scores_ref = self.mlp(scores_raw)             # (B, 32)
        nvtx.range_pop()

        return scores_raw, scores_ref, hv_bin, feat


# =========================================================
# Driver
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        "Voice -> FFTFrontend(512-bin) -> HDC(512x32) -> MLP(128-hidden) [inference-only]"
    )
    ap.add_argument("--batch",        type=int,   default=32)
    ap.add_argument("--length",       type=int,   default=4096,
                    help="Waveform length T (>= n_fft, default 1022)")
    ap.add_argument("--iters",        type=int,   default=50)
    # FFT
    ap.add_argument("--n-bins",       type=int,   default=512,
                    help="Number of FFT output bins")
    ap.add_argument("--n-fft",        type=int,   default=None,
                    help="FFT window length (auto-derived from n_bins if omitted)")
    ap.add_argument("--feat",         type=str,   default="mag",
                    choices=["mag", "pow", "logpow"])
    # HDC
    ap.add_argument("--num-classes",  type=int,   default=32,
                    help="Number of class hypervectors / HDC output dim")
    ap.add_argument("--score",        type=str,   default="dot",
                    choices=["dot", "cosine"])
    ap.add_argument("--no-binarize",  action="store_true",
                    help="Disable HDC binarization")
    # MLP
    ap.add_argument("--mlp-hidden",   type=int,   default=128,
                    help="Hidden layer width of refinement MLP")
    ap.add_argument("--mlp-out-dim",  type=int,   default=32,
                    help="Output dimension of refinement MLP")
    # Misc
    ap.add_argument("--create-on-gpu", action="store_true",
                    help="Create waveforms directly on GPU")
    ap.add_argument("--seed",         type=int,   default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # Derive n_fft to validate length argument
    n_fft_eff = args.n_fft if args.n_fft is not None else 2 * (args.n_bins - 1)
    assert args.length >= n_fft_eff, (
        f"--length ({args.length}) must be >= n_fft ({n_fft_eff})."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VoiceHDCHybrid(
        n_bins       = args.n_bins,
        n_fft        = args.n_fft,
        feat_mode    = args.feat,
        hdc_in_dim   = args.n_bins,          # must match FFT output
        num_classes  = args.num_classes,
        score_mode   = args.score,
        hdc_binarize = not args.no_binarize,
        mlp_hidden   = args.mlp_hidden,
        mlp_out_dim  = args.mlp_out_dim,
        seed         = args.seed,
    ).to(device).eval()

    # Print architecture summary
    print("=" * 55)
    print(f"  FFT  : n_fft={n_fft_eff}, n_bins={args.n_bins}, mode={args.feat}")
    print(f"  HDC  : ({args.n_bins} -> {args.num_classes}), "
          f"binarize={not args.no_binarize}, score={args.score}")
    print(f"  MLP  : ({args.num_classes} -> {args.mlp_hidden} -> {args.mlp_out_dim})")
    print(f"  Batch: {args.batch}  |  Length: {args.length}  |  Device: {device}")
    print("=" * 55)

    # Synthetic "voice-like" wave: sum of sinusoids + noise
    def synth_signal(B: int, T: int, on_gpu: bool = False) -> torch.Tensor:
        dev = device if on_gpu else torch.device("cpu")
        t   = torch.linspace(0, T - 1, T, device=dev) / 16000.0
        sig = (
            0.6 * torch.sin(2 * math.pi * 220  * t) +
            0.3 * torch.sin(2 * math.pi * 700  * t) +
            0.2 * torch.sin(2 * math.pi * 1400 * t)
        )
        sig  = sig.unsqueeze(0).repeat(B, 1)          # (B, T)
        sig += 0.01 * torch.randn(B, T, device=sig.device)
        return sig

    # ---------- Warmup ----------
    with torch.inference_mode():
        for _ in range(5):
            wav = synth_signal(args.batch, args.length, on_gpu=args.create_on_gpu)
            if wav.device != device:
                wav = wav.to(device, non_blocking=True)
            _ = model(wav)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # ---------- Profiled loop ----------
    with torch.inference_mode():
        nvtx.range_push("inference_run")
        for _ in range(args.iters):
            wav = synth_signal(args.batch, args.length, on_gpu=args.create_on_gpu)
            if wav.device != device:
                nvtx.range_push("HtoD")
                wav = wav.to(device, non_blocking=True)
                nvtx.range_pop()
            scores_raw, scores_ref, hv_bin, feat = model(wav)
        if device.type == "cuda":
            torch.cuda.synchronize()
        nvtx.range_pop()

    # ---------- Sanity print ----------
    print("\nOutput shapes:")
    print(f"  feat       : {tuple(feat.shape)}")
    print(f"  hv_bin     : {tuple(hv_bin.shape)}")
    print(f"  scores_raw : {tuple(scores_raw.shape)}")
    print(f"  scores_ref : {tuple(scores_ref.shape)}")
    print(f"\n  pred_raw[0]: {scores_raw.argmax(dim=1)[0].item()}")
    print(f"  pred_ref[0]: {scores_ref.argmax(dim=1)[0].item()}")


if __name__ == "__main__":
    main()