#!/usr/bin/env python
"""Step 4 - self-supervised DINO domain adaptation of CONCH / UNI.

Only the last two transformer blocks (CONCH ViT-B/16: blocks 10-11; UNI ViT-L:
blocks 22-23, configurable) and the final normalisation layer are trained;
all other weights stay frozen. Training uses every QC-passed bone marrow tile
of the development cohort and no response labels. The EMA teacher is saved as
the adapted encoder.

Single GPU:
    python scripts/04_dino_adapt.py --config cfg.yaml --encoder conch
Multi GPU (DistributedDataParallel):
    torchrun --nproc_per_node=4 scripts/04_dino_adapt.py --config cfg.yaml --encoder uni
"""
import argparse
import copy
import os

import _common  # noqa: F401
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src.config import add_config_args, config_from_args, dump_config, work_path
from src.models.dino import DINOAugmentation, DINOHead, DINOLoss, ema_momentum, ema_update
from src.models.foundation import freeze_except, load_encoder
from src.pipeline import load_patch_index, patches_dir
from src.utils import get_logger, seed_everything

log = get_logger()


class UnlabelledTiles(Dataset):
    def __init__(self, paths, aug):
        self.paths, self.aug = paths, aug

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.aug(Image.open(self.paths[i]))


def main():
    ap = add_config_args(argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--encoder", choices=["conch", "uni"], required=True)
    ap.add_argument("--resume", default=None, help="periodic checkpoint to resume from")
    ap.add_argument("--max_tiles", type=int, default=None, help="debug: limit #tiles")
    args = ap.parse_args()
    cfg = config_from_args(args)
    dc = cfg["dino"]

    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, local_rank = dist.get_rank(), int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(0 + rank)
    out = work_path(cfg, "dino", args.encoder)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        dump_config(cfg, out / "config_used.yaml")

    root = patches_dir(cfg)
    paths = [str(root / p) for p in load_patch_index(cfg)["patch_path"]]
    if args.max_tiles:
        paths = paths[: args.max_tiles]
    ds = UnlabelledTiles(paths, DINOAugmentation(cfg["foundation"]["tile_size"]))
    sampler = DistributedSampler(ds, shuffle=True) if distributed else None
    bs = dc["batch_size"][args.encoder]
    loader = DataLoader(ds, batch_size=min(bs, len(ds)), sampler=sampler,
                        shuffle=sampler is None, num_workers=dc["num_workers"],
                        drop_last=True, pin_memory=True)
    if rank == 0:
        log.info(f"DINO[{args.encoder}]: {len(ds)} tiles, {len(loader)} batches/epoch/GPU")

    student = load_encoder(args.encoder, cfg, device=device)
    blocks = cfg["foundation"]["encoders"][args.encoder]["trainable_blocks"]
    tr, tot = freeze_except(student, blocks)
    if rank == 0:
        log.info(f"trainable parameters: {tr:,}/{tot:,} (blocks {blocks} + final norm)")
    student.train()
    with torch.no_grad():
        feat_dim = student.encode(torch.zeros(1, 3, 224, 224, device=device)).shape[-1]
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False
    s_head = DINOHead(feat_dim, dc["out_dim"], dc["hidden_dim"], dc["bottleneck_dim"]).to(device)
    t_head = copy.deepcopy(s_head)
    for p in t_head.parameters():
        p.requires_grad = False
    s_enc = DDP(student, device_ids=[device.index]) if distributed else student
    s_hd = DDP(s_head, device_ids=[device.index]) if distributed else s_head

    def enc(m, x):
        return (m.module if isinstance(m, DDP) else m).encode(x)

    loss_fn = DINOLoss(dc["out_dim"], dc["epochs"], dc["warmup_teacher_temp"], dc["teacher_temp"],
                       dc["warmup_teacher_temp_epochs"], dc["student_temp"],
                       dc["center_momentum"]).to(device)
    params = [p for p in student.parameters() if p.requires_grad] + list(s_head.parameters())
    opt = torch.optim.AdamW(params, lr=dc["lr"], weight_decay=dc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=dc["epochs"], eta_min=dc["min_lr"])

    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        student.load_state_dict(ck["student_encoder"]); teacher.load_state_dict(ck["teacher_encoder"])
        s_head.load_state_dict(ck["student_head"]); t_head.load_state_dict(ck["teacher_head"])
        opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        start = ck["epoch"] + 1

    for epoch in range(start, dc["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
        m = ema_momentum(epoch, dc["epochs"], dc["ema_momentum_base"])
        total = 0.0
        for v1, v2 in loader:
            v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
            s1, s2 = s_hd(enc(s_enc, v1)), s_hd(enc(s_enc, v2))
            with torch.no_grad():
                t1, t2 = t_head(teacher.encode(v1)), t_head(teacher.encode(v2))
            loss = (loss_fn(s1, t2, epoch) + loss_fn(s2, t1, epoch)) / 2
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, dc["grad_clip"])
            opt.step()
            ema_update(student, teacher, m)
            ema_update(s_head, t_head, m)
            total += loss.item()
        sched.step()
        if rank == 0:
            log.info(f"epoch {epoch + 1}/{dc['epochs']} loss={total / max(1, len(loader)):.4f} ema={m:.4f}")
            if (epoch + 1) % dc["save_every"] == 0 or epoch + 1 == dc["epochs"]:
                torch.save({"epoch": epoch, "student_encoder": student.state_dict(),
                            "teacher_encoder": teacher.state_dict(),
                            "student_head": s_head.state_dict(), "teacher_head": t_head.state_dict(),
                            "optimizer": opt.state_dict(), "scheduler": sched.state_dict()},
                           out / f"checkpoint_epoch{epoch + 1:03d}.pt")
    if rank == 0:
        torch.save({"encoder_state_dict": teacher.state_dict(), "epochs": dc["epochs"],
                    "encoder": args.encoder, "trainable_blocks": blocks},
                   out / "adapted_encoder_final.pt")
        log.info(f"adapted encoder saved to {out / 'adapted_encoder_final.pt'}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
