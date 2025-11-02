import argparse
import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx
from typing import Literal

# =========================================================
# 1)  BasicBlock + ResNet-8 backbone
# =========================================================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, 0, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.relu(out + identity)
        return out

class ResNet8Backbone(nn.Module):
    """Outputs 64-D feature vector per image."""
    def __init__(self):
        super().__init__()
        self.in_planes = 16
        self.stem = nn.Sequential(
            nn.Conv2d(3,16,3,1,1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(16,1,1)
        self.layer2 = self._make_layer(32,1,2)
        self.layer3 = self._make_layer(64,1,2)
        self.pool   = nn.AdaptiveAvgPool2d((1,1))

    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return torch.flatten(x,1)  # (B,64)

# =========================================================
# 2)  HDC / VSA head
# =========================================================
class HDCHead(nn.Module):
    def __init__(self, in_dim=64, hv_dim=512, num_classes=100,
                 score_mode: Literal["dot","cosine"]="dot", seed=0):
        super().__init__()
        self.score_mode = score_mode
        g = torch.Generator().manual_seed(seed)
        proj = torch.randint(0,2,(in_dim,hv_dim),generator=g).float().mul_(2).add_(-1)
        class_hv = torch.randint(0,2,(hv_dim,num_classes),generator=g).float().mul_(2).add_(-1)
        self.register_buffer("proj",proj)
        self.register_buffer("class_hv",class_hv)
        if score_mode=="cosine":
            norm = torch.linalg.vector_norm(class_hv,dim=0,keepdim=True).clamp_min(1e-8)
            self.register_buffer("class_hv_unit",class_hv/norm)
        else:
            self.register_buffer("class_hv_unit",torch.empty(0))

    @staticmethod
    def _binarize_pm1(x):
        return torch.where(x>=0, torch.tensor(1.0,device=x.device), torch.tensor(-1.0,device=x.device))

    def forward(self,x):
        nvtx.range_push("HDC/projection")
        hv = x @ self.proj                     # (B,512)
        nvtx.range_pop()

        nvtx.range_push("HDC/binarize")
        hv_bin = self._binarize_pm1(hv)        # ±1
        nvtx.range_pop()

        if self.score_mode=="dot":
            nvtx.range_push("HDC/assoc_mem_dot")
            scores = hv_bin @ self.class_hv    # (B,100)
            nvtx.range_pop()
        else:
            nvtx.range_push("HDC/assoc_mem_cos")
            hv_norm = torch.linalg.vector_norm(hv_bin,dim=1,keepdim=True).clamp_min(1e-8)
            hv_unit = hv_bin / hv_norm
            scores = hv_unit @ self.class_hv_unit
            nvtx.range_pop()
        return scores, hv_bin

# =========================================================
# 3)  Full model wrapper
# =========================================================
class ResNet8_HDC(nn.Module):
    def __init__(self,hv_dim=512,num_classes=100,score_mode="dot",seed=0):
        super().__init__()
        self.backbone = ResNet8Backbone()
        self.hdc_head = HDCHead(64,hv_dim,num_classes,score_mode,seed)
    def forward(self,x):
        nvtx.range_push("backbone")
        feat = self.backbone(x)
        nvtx.range_pop()
        return self.hdc_head(feat)

# =========================================================
# 4)  Main driver
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="ResNet8-HDC inference profiler")
    parser.add_argument("--batch",type=int,default=1,help="Batch size")
    parser.add_argument("--score",type=str,default="dot",choices=["dot","cosine"])
    parser.add_argument("--iters",type=int,default=50,help="Profiling iterations")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet8_HDC(score_mode=args.score).to(device).eval()

    print(f"Running batch={args.batch}, score={args.score}")
    x = torch.randn(args.batch,3,32,32,device=device)

    # Warmup
    with torch.inference_mode():
        for _ in range(5): model(x)
    torch.cuda.synchronize()

    with torch.inference_mode():
        nvtx.range_push("inference_run")
        for _ in range(args.iters):
            scores, hv_bin = model(x)
        torch.cuda.synchronize()
        nvtx.range_pop()

    print("scores:", scores.shape, "hv_bin:", hv_bin.shape)

if __name__ == "__main__":
    main()

