"""Self-distillation with no labels (DINO) for domain adaptation.

A student and an exponential-moving-average teacher see two augmented views of
the same unlabelled bone marrow tile; the student is trained to match the
centred, sharpened teacher distribution of the *other* view. No response labels
are used.
"""
from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from ..data.datasets import IMAGENET_MEAN, IMAGENET_STD


class DINOAugmentation:
    def __init__(self, size: int = 224):
        jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        shared = [T.RandomResizedCrop(size, scale=(0.4, 1.0),
                                      interpolation=T.InterpolationMode.BICUBIC),
                  T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(90),
                  T.RandomApply([jitter], p=0.8)]
        tail = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        self.v1 = T.Compose(shared + [T.RandomApply([T.GaussianBlur(23)], p=0.5)] + tail)
        self.v2 = T.Compose(shared + [T.RandomApply([T.GaussianBlur(23)], p=0.2)] + tail)

    def __call__(self, img):
        img = img.convert("RGB")
        return self.v1(img), self.v2(img)


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, bottleneck_dim))
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        nn.init.normal_(self.last_layer.weight, std=0.01)

    def forward(self, x):
        return self.last_layer(F.normalize(self.mlp(x), dim=-1))


class DINOLoss(nn.Module):
    def __init__(self, out_dim, epochs, warmup_teacher_temp=0.04, teacher_temp=0.07,
                 warmup_epochs=20, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        warm = min(warmup_epochs, epochs)
        self.schedule = (torch.linspace(warmup_teacher_temp, teacher_temp, warm).tolist()
                         + [teacher_temp] * (epochs - warm))

    def forward(self, student_out, teacher_out, epoch):
        t = self.schedule[min(epoch, len(self.schedule) - 1)]
        teacher = F.softmax((teacher_out - self.center) / t, dim=-1).detach()
        loss = -(teacher * F.log_softmax(student_out / self.student_temp, dim=-1)).sum(-1).mean()
        self.update_center(teacher_out)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_out):
        c = teacher_out.mean(0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(c, op=dist.ReduceOp.AVG)
        self.center = self.center * self.center_momentum + c * (1 - self.center_momentum)


def ema_momentum(epoch, total, base=0.996, final=1.0):
    return final - (final - base) * (math.cos(math.pi * epoch / total) + 1) / 2


@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, m: float):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=1 - m)
