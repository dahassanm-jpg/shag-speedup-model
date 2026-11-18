"""Data-parallel training benchmark for S-HAG component calibration (ResNet-50 / BERT-large-like).

Per optimizer step it measures, on every rank:
  t_step      full DDP step (forward + backward with gradient all-reduce + optimizer)
  t_compute   same forward + backward with DDP communication disabled (no_sync)
  t_opt       optimizer update only (replicated on every rank -> serial part s0 in strong scaling)
  t_allreduce explicit all-reduce of a buffer the size of the model gradients (application-sized
              communication micro-benchmark, Section IV-E step 4)
and writes one JSON line with mean/sd/max over steps and over ranks (max over ranks = barrier time).

Strong scaling: --global-batch fixed, per-rank batch = global/world.
Weak scaling:   --per-rank-batch fixed.
Synthetic inputs are used so that data loading does not distort the timings.

Launch:  torchrun --nnodes=$N --nproc-per-node=$G ddp_bench.py --model resnet50 --mode weak ...
Test on CPU: torchrun --nproc-per-node=2 ddp_bench.py --model tiny-resnet --backend gloo --device cpu
"""
import argparse, json, os, statistics as st, time
import torch
import torch.distributed as dist
import torch.nn as nn


def build(name, seq):
    if name in ("resnet50", "tiny-resnet"):
        import torchvision
        m = torchvision.models.resnet50() if name == "resnet50" else torchvision.models.resnet18(num_classes=10)
        shape = (3, 224, 224) if name == "resnet50" else (3, 32, 32)
        ncls = 1000 if name == "resnet50" else 10
        def batch(b, dev):
            return torch.randn(b, *shape, device=dev), torch.randint(0, ncls, (b,), device=dev)
        loss = nn.CrossEntropyLoss()
        return m, batch, lambda out, y: loss(out, y)
    # Transformer encoder (BERT-large: 24 layers, d=1024, 16 heads; tiny: 2 layers, d=128)
    L, d, h = (24, 1024, 16) if name == "bert-large" else (2, 128, 4)
    vocab = 30522 if name == "bert-large" else 1000
    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, d)
            layer = nn.TransformerEncoderLayer(d, h, 4 * d, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, L)
            self.head = nn.Linear(d, vocab)
        def forward(self, x):
            return self.head(self.enc(self.emb(x)))
    def batch(b, dev):
        x = torch.randint(0, vocab, (b, seq), device=dev)
        return x, x
    ce = nn.CrossEntropyLoss()
    return Enc(), batch, lambda out, y: ce(out.reshape(-1, out.shape[-1]), y.reshape(-1))


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50",
                    choices=["resnet50", "bert-large", "tiny-resnet", "tiny-transformer"])
    ap.add_argument("--mode", default="weak", choices=["strong", "weak"])
    ap.add_argument("--global-batch", type=int, default=1024)
    ap.add_argument("--per-rank-batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--backend", default="nccl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default="ddp_results.jsonl")
    a = ap.parse_args()

    dist.init_process_group(a.backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    dev = torch.device(f"cuda:{local}") if a.device == "cuda" else torch.device("cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    torch.manual_seed(1234 + rank)

    model, make_batch, lossf = build(a.model, a.seq_len)
    model = model.to(dev)
    ddp = nn.parallel.DistributedDataParallel(model, device_ids=[local] if dev.type == "cuda" else None)
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-4)
    b = a.per_rank_batch if a.mode == "weak" else max(1, a.global_batch // world)
    x, y = make_batch(b, dev)
    nparam = sum(p.numel() for p in model.parameters())
    grad_buf = torch.zeros(nparam, device=dev, dtype=torch.float16 if a.amp else torch.float32)
    ac = torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=a.amp)

    def fwd_bwd(no_sync):
        ctx = ddp.no_sync() if no_sync else torch.enable_grad()
        with ctx:
            with ac:
                loss = lossf(ddp(x), y)
            loss.backward()

    rec = {k: [] for k in ("t_step", "t_compute", "t_opt", "t_allreduce")}
    for it in range(a.warmup + a.steps):
        # (1) full step
        dist.barrier(); sync(dev); t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True); fwd_bwd(False); opt.step()
        sync(dev); t_step = time.perf_counter() - t0
        # (2) compute only
        dist.barrier(); sync(dev); t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True); fwd_bwd(True)
        sync(dev); t_comp = time.perf_counter() - t0
        # (3) optimizer only
        sync(dev); t0 = time.perf_counter(); opt.step(); sync(dev); t_opt = time.perf_counter() - t0
        # (4) gradient-sized all-reduce
        dist.barrier(); sync(dev); t0 = time.perf_counter()
        dist.all_reduce(grad_buf); sync(dev); t_ar = time.perf_counter() - t0
        if it >= a.warmup:
            for k, v in zip(rec, (t_step, t_comp, t_opt, t_ar)):
                rec[k].append(v)

    local_stats = torch.tensor([st.mean(rec[k]) for k in rec] + [st.stdev(rec[k]) for k in rec],
                               dtype=torch.float64, device=dev)
    gathered = [torch.zeros_like(local_stats) for _ in range(world)]
    dist.all_gather(gathered, local_stats)
    if rank == 0:
        G = torch.stack(gathered).cpu()
        n = len(rec)
        out = dict(model=a.model, mode=a.mode, world=world, nodes=int(os.environ.get("SLURM_JOB_NUM_NODES", 1)),
                   per_rank_batch=b, global_batch=b * world, seq_len=a.seq_len, amp=a.amp,
                   n_params=nparam, steps=a.steps, device=str(dev.type), backend=a.backend)
        for i, k in enumerate(rec):
            out[f"{k}_mean"] = float(G[:, i].mean()); out[f"{k}_max_over_ranks"] = float(G[:, i].max())
            out[f"{k}_sd_over_steps"] = float(G[:, n + i].mean())
        out["samples_per_s"] = b * world / out["t_step_mean"]
        with open(a.out, "a") as f:
            f.write(json.dumps(out) + "\n")
        print(json.dumps(out, indent=1))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
