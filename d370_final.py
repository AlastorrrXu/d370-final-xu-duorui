"""DATA 37000 final project: the code of d370_final_results.ipynb as one script, every code cell in order.
Run it from the project folder with `python d370_final.py`. Figures are written to figures/ instead of being shown."""
import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore", message=".*non-interactive.*")
from IPython.display import display, Markdown


# %% code cell 1
import os, json, time, hashlib, random, platform, sys, zipfile, warnings
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
from datetime import datetime, timezone
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torchvision, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import OxfordIIITPet
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
from scipy.stats import binomtest

SEED = 42
BREEDS = ["Abyssinian", "Bengal", "Persian", "Beagle", "Pug", "Samoyed"]
VAL_FRAC = 0.2
WARMUP_EPOCHS = 1
MIN_EPOCHS, PATIENCE, MAX_EPOCHS = 100, 2, 200
BATCH, EVAL_BATCH = 16, 32
HEAD_LR, L4_LR = 1e-3, 1e-4
NUM_WORKERS = 0
DEVICE = None
OUT = Path.cwd()       # the project folder: open this notebook (or run d370_final.py) from the folder that holds it
assert (OUT / "d370_final_results.ipynb").exists() or (OUT / "d370_final.py").exists(), f"start from the project folder, not {OUT}"
DATA_ROOT = os.environ.get("D370_DATA") or str(OUT.parent / "data" / "bigdata")   # Oxford-IIIT Pet is kept (or downloaded) here
TAG = os.environ.get("D370_TAG", "")   # "" = the reported run's folders; any other tag starts a fresh run in results_<tag>/ etc.
assert MIN_EPOCHS >= 5

sfx = f"_{TAG}" if TAG else ""
RES, FIG, CKPT = OUT / f"results{sfx}", OUT / f"figures{sfx}", OUT / f"checkpoints{sfx}"
for d in (RES, FIG, CKPT):
    d.mkdir(parents=True, exist_ok=True)
os.makedirs(DATA_ROOT, exist_ok=True)
dump = lambda obj, p: Path(p).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False

if DEVICE: device = torch.device(DEVICE)
elif torch.cuda.is_available(): device = torch.device("cuda")
elif torch.backends.mps.is_available(): device = torch.device("mps")
else: device = torch.device("cpu")

env = dict(python=sys.version.split()[0], platform=platform.platform(), torch=torch.__version__,
           torchvision=torchvision.__version__, device=str(device))
if device.type == "cuda":
    p = torch.cuda.get_device_properties(device)
    env.update(gpu=torch.cuda.get_device_name(device), gpu_mem_gb=round(p.total_memory / 2**30, 1),
               cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
else:
    env.update(cpu=platform.processor() or platform.machine(), torch_threads=torch.get_num_threads())
settings = dict(seed=SEED, breeds=BREEDS, val_frac=VAL_FRAC, warmup_epochs=WARMUP_EPOCHS, min_epochs=MIN_EPOCHS,
                patience=PATIENCE, max_epochs=MAX_EPOCHS, batch=BATCH, eval_batch=EVAL_BATCH, head_lr=HEAD_LR,
                layer4_lr=L4_LR, num_workers=NUM_WORKERS, data_root=str(Path(DATA_ROOT).resolve()),
                tf32=False, cudnn_deterministic=True)
dump({"env": env, "settings": settings}, RES / "run_config.json")
env


# %% code cell 2
development = OxfordIIITPet(root=DATA_ROOT, split="trainval", target_types="category", download=True)
assert all(b in development.class_to_idx for b in BREEDS), [b for b in BREEDS if b not in development.class_to_idx]
orig = {b: development.class_to_idx[b] for b in BREEDS}
o2n = {orig[b]: i for i, b in enumerate(BREEDS)}
labels = np.array(development._labels)
ids = [Path(p).stem for p in development._images]

rng = np.random.default_rng(SEED)
tr_idx, va_idx = [], []
for b in BREEDS:
    perm = rng.permutation(np.flatnonzero(labels == orig[b]))
    nv = int(round(VAL_FRAC * len(perm)))
    va_idx += perm[:nv].tolist()
    tr_idx += perm[nv:].tolist()
tr_idx, va_idx = sorted(tr_idx), sorted(va_idx)
assert not set(tr_idx) & set(va_idx)

split = dict(seed=SEED, val_frac=VAL_FRAC, original_label=orig, new_label={b: i for i, b in enumerate(BREEDS)},
             train_idx=tr_idx, val_idx=va_idx, train_ids=[ids[j] for j in tr_idx], val_ids=[ids[j] for j in va_idx])
if (RES / "split.json").exists():
    old = json.loads((RES / "split.json").read_text(encoding="utf-8"))
    assert old["train_idx"] == tr_idx and old["val_idx"] == va_idx, "split changed?"
dump(split, RES / "split.json")
dump({"breeds": BREEDS, "original_index": orig, "new_index": split["new_label"]}, RES / "class_mapping.json")
pd.DataFrame([dict(split=s, dataset_index=j, image_id=ids[j], original_label=int(labels[j]),
                   breed=BREEDS[o2n[int(labels[j])]]) for s, idx in [("train", tr_idx), ("val", va_idx)] for j in idx]
             ).to_csv(RES / "split_manifest.csv", index=False)
counts = pd.DataFrame([dict(breed=b, original_label=orig[b], new_label=i, trainval=int((labels == orig[b]).sum()),
                            train=int(sum(labels[j] == orig[b] for j in tr_idx)),
                            val=int(sum(labels[j] == orig[b] for j in va_idx))) for i, b in enumerate(BREEDS)])
counts.to_csv(RES / "split_counts.csv", index=False)
print(len(development), "trainval images over all 37 breeds |", len(tr_idx), "train,", len(va_idx), "val")
counts


# %% code cell 3
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
tf_train = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(), transforms.Normalize(mean, std)])
tf_eval = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                              transforms.Normalize(mean, std)])
remap = o2n.__getitem__
dev_train = OxfordIIITPet(DATA_ROOT, split="trainval", target_types="category", transform=tf_train, target_transform=remap)
dev_eval = OxfordIIITPet(DATA_ROOT, split="trainval", target_types="category", transform=tf_eval, target_transform=remap)
train_ds, val_ds = Subset(dev_train, tr_idx), Subset(dev_eval, va_idx)
print(len(train_ds), "train /", len(val_ds), "val")


# %% code cell 4
CONDS = ["frozen", "finetune"]

def build():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.requires_grad_(False)
    torch.manual_seed(SEED)
    m.fc = nn.Linear(512, len(BREEDS))
    return m

def setup(m, cond):
    m.requires_grad_(False)
    m.fc.requires_grad_(True)
    if cond == "finetune":
        m.layer4.requires_grad_(True)
    return m

def make_opt(m, cond):
    groups = [{"params": list(m.fc.parameters()), "lr": HEAD_LR}]
    if cond == "finetune":
        groups.insert(0, {"params": list(m.layer4.parameters()), "lr": L4_LR})
    opt = torch.optim.Adam(groups)
    assert {id(p) for g in opt.param_groups for p in g["params"]} == {id(p) for p in m.parameters() if p.requires_grad}
    return opt

def train_mode(m):
    m.train()
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d) and not any(p.requires_grad for p in mod.parameters()):
            mod.eval()

rows = []
for c in CONDS:
    m = setup(build(), c)
    train_mode(m)
    rows.append(dict(condition=c, total=sum(p.numel() for p in m.parameters()),
                     trainable=sum(p.numel() for p in m.parameters() if p.requires_grad),
                     bn_layers=sum(isinstance(x, nn.BatchNorm2d) for x in m.modules()),
                     bn_using_batch_stats=sum(isinstance(x, nn.BatchNorm2d) and x.training for x in m.modules())))
del m
pd.DataFrame(rows)


# %% code cell 5
def sync():
    if device.type == "cuda": torch.cuda.synchronize()
    elif device.type == "mps": torch.mps.synchronize()

def run_epoch(m, opt, epoch):
    train_mode(m)
    g = torch.Generator()
    g.manual_seed(SEED * 1000 + epoch)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, generator=g)
    torch.manual_seed(SEED * 1000 + epoch)
    h = hashlib.sha256()
    tot, correct, n, t_data, t_comp = 0.0, 0, 0, 0.0, 0.0
    sync()
    if device.type == "cuda":
        mem0 = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0 = t_last = time.perf_counter()
    for x, y in loader:
        h.update(y.numpy().astype(np.int64).tobytes())
        h.update(np.float64(x.double().sum().item()).tobytes())
        t1 = time.perf_counter()
        t_data += t1 - t_last
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        out = m(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        opt.step()
        tot += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
        t_last = time.perf_counter()
        t_comp += t_last - t1
    sync()
    r = dict(train_loss=tot / n, train_acc=correct / n, train_time=time.perf_counter() - t0, data_time=t_data,
             compute_time=t_comp, batches_hash=h.hexdigest()[:16], gpu_peak_inc_mb=None)
    if device.type == "cuda":
        r["gpu_peak_inc_mb"] = round((torch.cuda.max_memory_allocated(device) - mem0) / 2**20, 1)
    return r

@torch.no_grad()
def predict(m, ds):
    m.eval()
    outs, ys = [], []
    for x, y in DataLoader(ds, batch_size=EVAL_BATCH, shuffle=False, num_workers=NUM_WORKERS):
        outs.append(m(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(outs), torch.cat(ys)

def macro_f1(y, p):
    k = len(BREEDS)
    cm = np.bincount(k * np.asarray(y) + np.asarray(p), minlength=k * k).reshape(k, k)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    d = 2 * tp + fp + fn
    return float(np.mean(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)))

def evaluate(m, ds):
    sync(); t0 = time.perf_counter()
    logits, y = predict(m, ds)
    sync()
    pred = logits.argmax(1)
    nc = int((pred == y).sum())
    return dict(loss=F.cross_entropy(logits, y).item(), correct=nc, n=len(y), acc=nc / len(y),
                macro_f1=macro_f1(y.numpy(), pred.numpy()), time=time.perf_counter() - t0)

def better(r, b):
    return b is None or r["val_correct"] > b["val_correct"] or (r["val_correct"] == b["val_correct"] and r["val_loss"] < b["val_loss"])

def since_best(hist):
    best, be = -1, 0
    for r in hist:
        if r["val_correct"] > best:
            best, be = r["val_correct"], r["epoch"]
    return hist[-1]["epoch"] - be

def warm_device():
    w = models.resnet18(weights=None)
    w.fc = nn.Linear(512, len(BREEDS))
    w = setup(w, "finetune").to(device)
    o = torch.optim.Adam([p for p in w.parameters() if p.requires_grad], lr=1e-4)
    x = torch.randn(BATCH, 3, 224, 224, device=device)
    y = torch.randint(0, len(BREEDS), (BATCH,), device=device)
    for _ in range(2):
        train_mode(w); o.zero_grad(set_to_none=True); F.cross_entropy(w(x), y).backward(); o.step()
    setup(w, "frozen"); train_mode(w); F.cross_entropy(w(x), y).backward()
    with torch.no_grad():
        w.eval(); w(torch.randn(EVAL_BATCH, 3, 224, 224, device=device))
    sync()
    if device.type == "cuda": torch.cuda.empty_cache()


# %% code cell 6
warm_device()
m = setup(build(), "warmup").to(device)
opt = make_opt(m, "warmup")
before = evaluate(m, val_ds)
w_tr = [run_epoch(m, opt, -e) for e in range(WARMUP_EPOCHS)]
w_val = evaluate(m, val_ds)
torch.save({k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, CKPT / "warmup_head.pt")
dump(dict(train=w_tr, val_before=before, val_after=w_val, train_time=sum(r["train_time"] for r in w_tr)), RES / "warmup.json")
print(f"warm-up: train loss {w_tr[-1]['train_loss']:.4f} ({sum(r['train_time'] for r in w_tr):.1f}s) | "
      f"val acc {before['acc']:.3f} with the random head -> {w_val['acc']:.4f} ({w_val['correct']}/{w_val['n']}), "
      f"val loss {w_val['loss']:.4f}")


# %% code cell 7
warm_device()
(RES / "train_log.txt").write_text("")
w_state = torch.load(CKPT / "warmup_head.pt", map_location="cpu")
runs = {}
for c in CONDS:
    m = build()
    m.load_state_dict(w_state)
    m = setup(m, c).to(device)
    s = evaluate(m, val_ds)
    if s["correct"] != w_val["correct"] or abs(s["loss"] - w_val["loss"]) > 1e-4:
        if abs(s["loss"] - w_val["loss"]) > 1e-2:
            raise RuntimeError(f"{c}: reloaded warm-up gives {s['correct']} / {s['loss']:.5f}")
        warnings.warn(f"{c}: reloaded warm-up {s['correct']} / {s['loss']:.6f} vs {w_val['correct']} / {w_val['loss']:.6f}")
    runs[c] = dict(m=m, opt=make_opt(m, c), hist=[], best=None, best_state=None, plateau=None)

t_start = time.perf_counter()
stop = None
for ep in range(1, MAX_EPOCHS + 1):
    msg = []
    for c, R in runs.items():
        tr = run_epoch(R["m"], R["opt"], ep)
        va = evaluate(R["m"], val_ds)
        rec = dict(condition=c, epoch=ep, **tr, val_loss=va["loss"], val_acc=va["acc"], val_correct=va["correct"],
                   val_n=va["n"], val_macro_f1=va["macro_f1"], val_time=va["time"])
        R["hist"].append(rec)
        if better(rec, R["best"]):
            R["best"] = rec
            R["best_state"] = {k: v.detach().cpu().clone() for k, v in R["m"].state_dict().items()}
        if R["plateau"] is None and ep >= MIN_EPOCHS and since_best(R["hist"]) >= PATIENCE:
            R["plateau"] = ep
        line = (f"{c:8s} ep {ep:3d} | train loss {tr['train_loss']:.4f} acc {tr['train_acc']:.3f} | val acc {va['acc']:.4f} "
                f"({va['correct']}/{va['n']}) loss {va['loss']:.4f} F1 {va['macro_f1']:.4f} | {tr['train_time']:.1f}s "
                f"(compute {tr['compute_time']:.1f}s) | {tr['batches_hash']}")
        with open(RES / "train_log.txt", "a") as f:
            f.write(line + "\n")
        msg.append(f"{c} tr {tr['train_loss']:.3f} val {va['correct']}/{va['n']}{'*' if R['best']['epoch'] == ep else ' '} "
                   f"{va['loss']:.3f} {tr['train_time']:.1f}s")
    print(f"ep {ep:3d} | " + " | ".join(msg))
    if all(R["plateau"] for R in runs.values()):
        stop = ep
        break
if stop is None:
    stop = MAX_EPOCHS
    print("hit MAX_EPOCHS before both runs plateaued")
print(f"\n{time.perf_counter() - t_start:.1f}s for both runs incl. validation")

assert [r["batches_hash"] for r in runs["frozen"]["hist"]] == [r["batches_hash"] for r in runs["finetune"]["hist"]]

lit = {c: next((e for e in range(5, len(R["hist"]) + 1) if since_best(R["hist"][:e]) >= PATIENCE), None) for c, R in runs.items()}
under5 = None
if all(lit.values()):
    e5 = max(lit.values())
    sel5 = {}
    for c, R in runs.items():
        bb = None
        for r in R["hist"][:e5]:
            if better(r, bb):
                bb = r
        sel5[c] = bb["epoch"]
    under5 = dict(plateau=lit, E=e5, selected=sel5)

summary = dict(E=stop, plateau={c: R["plateau"] for c, R in runs.items()}, min_epochs=MIN_EPOCHS, patience=PATIENCE,
               under_floor_5=under5, device=str(device), cond={})
for c, R in runs.items():
    h, b = R["hist"], R["best"]
    torch.save(R["best_state"], CKPT / f"{c}_selected.pt")
    torch.save({k: v.detach().cpu().clone() for k, v in R["m"].state_dict().items()}, CKPT / f"{c}_last.pt")
    ntr = sum(p.numel() for p in R["m"].parameters() if p.requires_grad)
    peaks = [r["gpu_peak_inc_mb"] for r in h[1:] if r["gpu_peak_inc_mb"] is not None]
    summary["cond"][c] = dict(
        selected_epoch=b["epoch"], selected=b, trainable=ntr,
        lr={"layer4": L4_LR, "head": HEAD_LR} if c == "finetune" else {"head": HEAD_LR},
        train_time_total=sum(r["train_time"] for r in h), compute_time_total=sum(r["compute_time"] for r in h),
        data_time_total=sum(r["data_time"] for r in h), val_time_total=sum(r["val_time"] for r in h),
        train_time_to_selected=sum(r["train_time"] for r in h if r["epoch"] <= b["epoch"]),
        train_s_per_epoch=float(np.mean([r["train_time"] for r in h])),
        compute_s_per_epoch=float(np.mean([r["compute_time"] for r in h])),
        data_s_per_epoch=float(np.mean([r["data_time"] for r in h])),
        grad_adam_mib=12 * ntr / 2**20, gpu_peak_inc_mib=max(peaks) if peaks else None,
        sha256=hashlib.sha256((CKPT / f"{c}_selected.pt").read_bytes()).hexdigest())
dump(summary, RES / "training_summary.json")
hist = pd.DataFrame([r for R in runs.values() for r in R["hist"]])
hist.to_csv(RES / "history.csv", index=False)
S = summary["cond"]
for c in CONDS:
    print(f"{c:8s} selected epoch {S[c]['selected_epoch']}: val acc {S[c]['selected']['val_acc']:.4f}, "
          f"val loss {S[c]['selected']['val_loss']:.4f}, val F1 {S[c]['selected']['val_macro_f1']:.4f}")
print("first plateau:", summary["plateau"], "-> both trained E =", stop, "epochs")


# %% code cell 8
cost = pd.DataFrame({c: {"epochs": stop, "selected epoch": S[c]["selected_epoch"],
                         "train s/epoch": round(S[c]["train_s_per_epoch"], 2),
                         "compute s/epoch": round(S[c]["compute_s_per_epoch"], 2),
                         "data loading s/epoch": round(S[c]["data_s_per_epoch"], 2),
                         "train s total": round(S[c]["train_time_total"], 1),
                         "train s to selected": round(S[c]["train_time_to_selected"], 1),
                         "grad + Adam state MiB": round(S[c]["grad_adam_mib"], 2),
                         "GPU peak increment MiB": S[c]["gpu_peak_inc_mib"]} for c in CONDS})
print("device:", env.get("gpu", env["device"]))
print(f"fine-tune / frozen per epoch: wall-clock {S['finetune']['train_s_per_epoch'] / S['frozen']['train_s_per_epoch']:.2f}x, "
      f"compute only {S['finetune']['compute_s_per_epoch'] / S['frozen']['compute_s_per_epoch']:.2f}x")
print("validation-only check with the literal floor of 5:", summary["under_floor_5"])
cost


# %% code cell 9
w_state = torch.load(CKPT / "warmup_head.pt", map_location="cpu")
ver = {}
for c in CONDS:
    for which in ["selected", "last"]:
        sd = torch.load(CKPT / f"{c}_{which}.pt", map_location="cpu")
        changed = [k for k in sd if not torch.equal(sd[k], w_state[k])]
        bad = [k for k in changed if not k.startswith("fc.") and not (c == "finetune" and k.startswith("layer4."))]
        assert not bad, bad
        ver[f"{c}_{which}"] = {"fc changed": any(k.startswith("fc.") for k in changed),
                               "layer4 tensors changed": sum(k.startswith("layer4.") for k in changed),
                               "layer4 BN running stats changed": sum(k.startswith("layer4.") and "running" in k for k in changed),
                               "frozen tensors changed": len(bad)}
dump(ver, RES / "verification.json")
same = (hist[hist.condition == "frozen"].batches_hash.values == hist[hist.condition == "finetune"].batches_hash.values).all()
print("identical mini-batches in every epoch:", same)
pd.DataFrame(ver).T


# %% code cell 10
col = {"frozen": "tab:blue", "finetune": "tab:orange"}
lab = {"frozen": "frozen (head only)", "finetune": "fine-tune (layer4 + head)"}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for c in CONDS:
    h = hist[hist.condition == c]
    ep = [0] + h.epoch.tolist()
    ax[0].plot(ep, [100 * w_val["acc"]] + (100 * h.val_acc).tolist(), "o-", ms=2, lw=1.2, color=col[c], label=lab[c])
    ax[1].plot(ep, [w_val["loss"]] + h.val_loss.tolist(), "o-", ms=2, lw=1.2, color=col[c], label=lab[c])
    ax[2].plot(ep, [w_tr[-1]["train_loss"]] + h.train_loss.tolist(), "o-", ms=2, lw=1.2, color=col[c], label=lab[c])
    se = S[c]["selected_epoch"]
    ax[0].plot(se, 100 * h.val_acc.iloc[se - 1], "*", ms=16, color=col[c], mec="k", label=f"selected ({c}): epoch {se}")
    ax[1].plot(se, h.val_loss.iloc[se - 1], "*", ms=16, color=col[c], mec="k")
for a, t in zip(ax, [f"validation accuracy (%), n = {len(va_idx)}", "validation loss", "training loss (augmented)"]):
    a.axvline(MIN_EPOCHS, color="tab:green", ls=":", lw=1)
    a.axvline(stop, color="gray", ls="--", lw=1)
    a.set_title(t); a.set_xlabel("epoch (0 = shared warm-up)"); a.grid(alpha=0.3)
ax[0].legend(fontsize=8, loc="lower right"); ax[2].legend(fontsize=8)
fig.suptitle(f"plateau checks start at epoch {MIN_EPOCHS} (green); first plateau: frozen {summary['plateau']['frozen']}, "
             f"fine-tune {summary['plateau']['finetune']}; both runs trained E = {stop} epochs (grey); stars = selected", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "fig1_learning_curves.png", dpi=150)
plt.show()


# %% code cell 11
f_, t_ = S["frozen"]["selected"], S["finetune"]["selected"]
if t_["val_correct"] != f_["val_correct"]:
    overall = "finetune" if t_["val_correct"] > f_["val_correct"] else "frozen"
elif t_["val_loss"] != f_["val_loss"]:
    overall = "finetune" if t_["val_loss"] < f_["val_loss"] else "frozen"
else:
    overall = "frozen"
plan1 = ["test accuracy (Wilson 95% CI), macro-F1, per-class precision/recall/F1, confusion matrices",
         "same test images: 2x2 table, exact McNemar, paired bootstrap (10000, seed 42) for acc and macro-F1 differences",
         "error analysis: 3 most confident test errors of the overall model; fallback validation errors, then least confident correct",
         "Grad-CAM at layer4, target = true breed; image A = disagreement with largest change in p(true), image B = another error, prefer both wrong"]
selected1 = {c: dict(epoch=S[c]["selected_epoch"], val_correct=S[c]["selected"]["val_correct"],
                     val_loss=S[c]["selected"]["val_loss"], sha256=S[c]["sha256"]) for c in CONDS}
lock_path1 = RES / "locked_choices.json"
if lock_path1.exists():
    # an existing lock is never rewritten: the retrained checkpoints have to reproduce it bit for bit
    lock = json.loads(lock_path1.read_text(encoding="utf-8"))
    same = (all(lock["selected"][c]["epoch"] == selected1[c]["epoch"] and lock["selected"][c]["sha256"] == selected1[c]["sha256"]
                for c in CONDS) and lock["overall"] == overall and lock["E"] == stop and lock["plan"] == plan1)
    assert same, ("the retrained checkpoints or the plan differ from the locked ones; to rerun from scratch without touching "
                  "the reported run, set the environment variable D370_TAG (see the first code cell)")
    print(f"already locked at {lock['locked_at']}; both retrained checkpoints reproduce the locked SHA-256 hashes bit for bit")
else:
    lock = dict(locked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), selected=selected1, overall=overall,
                E=stop, plateau=summary["plateau"], settings=settings, plan=plan1)
    dump(lock, lock_path1)
    print("locked at", lock["locked_at"])
print(json.dumps({k: lock[k] for k in ["locked_at", "selected", "overall", "E"]}, indent=1))


# %% code cell 12
lock = json.loads((RES / "locked_choices.json").read_text(encoding="utf-8"))
for c in CONDS:
    assert hashlib.sha256((CKPT / f"{c}_selected.pt").read_bytes()).hexdigest() == lock["selected"][c]["sha256"]
test_full = OxfordIIITPet(DATA_ROOT, split="test", target_types="category", transform=tf_eval, target_transform=remap, download=True)
te_idx = [i for i, l in enumerate(test_full._labels) if l in o2n]
te_ids = [Path(test_full._images[i]).stem for i in te_idx]
assert not set(te_ids) & set(split["train_ids"] + split["val_ids"])
test_ds = Subset(test_full, te_idx)

def load_sel(c):
    m = build()
    m.load_state_dict(torch.load(CKPT / f"{c}_selected.pt", map_location="cpu"))
    return setup(m, c).to(device).eval()

def wilson(k, n, z=1.959963984540054):
    ph, den = k / n, 1 + z * z / n
    mid, half = (ph + z * z / (2 * n)) / den, z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [float(mid - half), float(mid + half)]

K = len(BREEDS)
test, P = {"n": len(test_ds), "cond": {}}, {}
for c in CONDS:
    m = load_sel(c)
    t0 = time.perf_counter()
    logits, yt = predict(m, test_ds)
    ti = time.perf_counter() - t0
    y = yt.numpy()
    P[c] = logits.softmax(1).numpy()
    pred = P[c].argmax(1)
    mf = macro_f1(y, pred)
    assert abs(mf - f1_score(y, pred, average="macro", labels=list(range(K)), zero_division=0)) < 1e-12
    pr, rc, f1s, sup = precision_recall_fscore_support(y, pred, labels=list(range(K)), zero_division=0)
    nc = int((pred == y).sum())
    test["cond"][c] = dict(acc=nc / len(y), correct=nc, acc_ci95=wilson(nc, len(y)), macro_f1=mf,
                           loss=F.cross_entropy(logits, yt).item(), infer_time=ti,
                           cm=confusion_matrix(y, pred, labels=list(range(K))).tolist(),
                           per_class={BREEDS[i]: dict(precision=float(pr[i]), recall=float(rc[i]), f1=float(f1s[i]), n=int(sup[i]))
                                      for i in range(K)})
test["counts"] = {BREEDS[i]: int((y == i).sum()) for i in range(K)}

pf, pt = P["frozen"].argmax(1), P["finetune"].argmax(1)
cf, ct = pf == y, pt == y
b10, b01 = int((cf & ~ct).sum()), int((~cf & ct).sum())
B = 10000
rng = np.random.default_rng(SEED)
da, dm = np.empty(B), np.empty(B)
for b in range(B):
    i = rng.integers(0, len(y), len(y))
    da[b] = ct[i].mean() - cf[i].mean()
    dm[b] = macro_f1(y[i], pt[i]) - macro_f1(y[i], pf[i])
test["paired"] = dict(both_correct=int((cf & ct).sum()), only_frozen=b10, only_finetune=b01, both_wrong=int((~cf & ~ct).sum()),
                      mcnemar_p=float(binomtest(min(b10, b01), b10 + b01, 0.5).pvalue) if b10 + b01 else 1.0,
                      d_acc=float(ct.mean() - cf.mean()), d_acc_ci=[float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
                      d_f1=macro_f1(y, pt) - macro_f1(y, pf), d_f1_ci=[float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))],
                      bootstrap=B)
dump(test, RES / "test_metrics.json")

pred_df = pd.DataFrame(dict(position=np.arange(len(y)), dataset_index=te_idx, image_id=te_ids,
                            true=[BREEDS[i] for i in y], label=y))
for c in CONDS:
    pred_df[f"pred_{c}"] = [BREEDS[i] for i in P[c].argmax(1)]
    pred_df[f"conf_{c}"] = P[c].max(1).round(6)
    pred_df[f"p_true_{c}"] = P[c][np.arange(len(y)), y].round(6)
pred_df.to_csv(RES / "test_predictions.csv", index=False)
np.savez(RES / "test_probs.npz", y=y, frozen=P["frozen"], finetune=P["finetune"])
for c in CONDS:
    assert accuracy_score(pred_df.true, pred_df[f"pred_{c}"]) == test["cond"][c]["acc"]
    t = test["cond"][c]
    print(f"{c:8s} test acc {t['acc']:.4f} ({t['correct']}/{test['n']}, 95% CI {t['acc_ci95'][0]:.3f}-{t['acc_ci95'][1]:.3f}), "
          f"macro-F1 {t['macro_f1']:.4f}, loss {t['loss']:.4f}")
print("test images per breed:", test["counts"])


# %% code cell 13
Pv = {}
for c in CONDS:
    lg, yv = predict(load_sel(c), val_ds)
    Pv[c] = lg.softmax(1).numpy()
yv = yv.numpy()
val_df = pd.DataFrame(dict(position=np.arange(len(yv)), dataset_index=va_idx, image_id=split["val_ids"],
                           true=[BREEDS[i] for i in yv], label=yv))
for c in CONDS:
    val_df[f"pred_{c}"] = [BREEDS[i] for i in Pv[c].argmax(1)]
    val_df[f"conf_{c}"] = Pv[c].max(1).round(6)
    val_df[f"p_true_{c}"] = Pv[c][np.arange(len(yv)), yv].round(6)
val_df.to_csv(RES / "val_predictions.csv", index=False)

T = test["cond"]
main = pd.DataFrame({c: {"trainable params": f"{S[c]['trainable']:,}", "selected epoch": S[c]["selected_epoch"],
                         "val acc": f"{100 * S[c]['selected']['val_acc']:.2f}% ({S[c]['selected']['val_correct']}/{len(va_idx)})",
                         "val macro-F1": round(S[c]["selected"]["val_macro_f1"], 4),
                         "test acc": f"{100 * T[c]['acc']:.2f}% ({T[c]['correct']}/{test['n']})",
                         "test acc 95% CI": f"{100 * T[c]['acc_ci95'][0]:.1f}-{100 * T[c]['acc_ci95'][1]:.1f}",
                         "test macro-F1": round(T[c]["macro_f1"], 4), "test loss": round(T[c]["loss"], 4),
                         "epochs": stop, "train s total": round(S[c]["train_time_total"], 1),
                         "train s/epoch": round(S[c]["train_s_per_epoch"], 2), "compute s/epoch": round(S[c]["compute_s_per_epoch"], 2),
                         "train s to selected": round(S[c]["train_time_to_selected"], 1),
                         "grad + Adam MiB": round(S[c]["grad_adam_mib"], 1), "GPU peak inc MiB": S[c]["gpu_peak_inc_mib"]}
                     for c in CONDS})
main.to_csv(RES / "table_main.csv")
per_class = pd.DataFrame({(c, q): {b: round(T[c]["per_class"][b][q], 3) for b in BREEDS} for c in CONDS for q in ["precision", "recall", "f1"]})
per_class.to_csv(RES / "table_per_class.csv")
display(main)
per_class


# %% code cell 14
fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
for a, c in zip(ax, [overall] + [x for x in CONDS if x != overall]):
    cm = np.array(T[c]["cm"])
    a.imshow(cm, cmap="Blues")
    for i in range(K):
        for j in range(K):
            a.text(j, i, cm[i, j], ha="center", va="center", fontsize=10, color="white" if cm[i, j] > cm.max() / 2 else "black")
    a.set_xticks(range(K)); a.set_xticklabels(BREEDS, rotation=35, ha="right")
    a.set_yticks(range(K)); a.set_yticklabels(BREEDS)
    a.set_xlabel("predicted"); a.set_ylabel("true")
    a.set_title(f"{'SELECTED: ' if c == overall else ''}{lab[c]}\ntest acc {100 * T[c]['acc']:.2f}%, macro-F1 {T[c]['macro_f1']:.4f} (n = {test['n']})", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "fig2_confusion_matrices.png", dpi=150)
plt.show()


# %% code cell 15
pc = test["paired"]
display(pd.DataFrame({"fine-tune correct": [pc["both_correct"], pc["only_finetune"]],
                      "fine-tune wrong": [pc["only_frozen"], pc["both_wrong"]]}, index=["frozen correct", "frozen wrong"]))
print(f"exact McNemar p = {pc['mcnemar_p']:.4f}")
print(f"accuracy difference (fine-tune - frozen) {100 * pc['d_acc']:+.2f} pp, 95% CI [{100 * pc['d_acc_ci'][0]:+.2f}, {100 * pc['d_acc_ci'][1]:+.2f}]")
print(f"macro-F1 difference {pc['d_f1']:+.4f}, 95% CI [{pc['d_f1_ci'][0]:+.4f}, {pc['d_f1_ci'][1]:+.4f}]")
print(f"extra cost of fine-tuning: {S['finetune']['trainable'] - S['frozen']['trainable']:,} more trainable params, "
      f"{S['finetune']['train_time_total'] - S['frozen']['train_time_total']:.1f}s more training over {stop} epochs, "
      f"{S['finetune']['grad_adam_mib'] - S['frozen']['grad_adam_mib']:.1f} MiB more grad + Adam state; "
      f"test inference {T['frozen']['infer_time']:.1f}s vs {T['finetune']['infer_time']:.1f}s")
dis = pred_df[pred_df.pred_frozen != pred_df.pred_finetune]
print(len(dis), "test images where the two models disagree")
dis[["image_id", "true", "pred_frozen", "p_true_frozen", "pred_finetune", "p_true_finetune"]]


# %% code cell 16
other = [c for c in CONDS if c != overall][0]
errs = pred_df[pred_df[f"pred_{overall}"] != pred_df.true].sort_values(f"conf_{overall}", ascending=False, kind="stable")
cases = [dict(r, source="test", kind="test error") for r in errs.head(3).to_dict("records")]
if len(cases) < 3:
    ve = val_df[val_df[f"pred_{overall}"] != val_df.true].sort_values(f"conf_{overall}", ascending=False, kind="stable")
    cases += [dict(r, source="val", kind="validation error") for r in ve.head(3 - len(cases)).to_dict("records")]
if len(cases) < 3:
    lc = pred_df[pred_df[f"pred_{overall}"] == pred_df.true].sort_values(f"conf_{overall}", kind="stable")
    cases += [dict(r, source="test", kind="low-confidence correct") for r in lc.head(3 - len(cases)).to_dict("records")]

def show(t):
    return (t.cpu() * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]).clamp(0, 1).permute(1, 2, 0).numpy()

def get_img(cs):
    ds = test_full if cs["source"] == "test" else dev_eval
    x, yy = ds[int(cs["dataset_index"])]
    return x, yy, Path(ds._images[int(cs["dataset_index"])]).stem

mdl = {c: load_sel(c) for c in CONDS}
fig, ax = plt.subplots(2, len(cases), figsize=(5 * len(cases), 8.6), gridspec_kw={"height_ratios": [3, 1.4]}, squeeze=False)
for j, cs in enumerate(cases):
    x, yy, iid = get_img(cs)
    assert iid == cs["image_id"] and BREEDS[yy] == cs["true"]
    with torch.no_grad():
        pr = {c: mdl[c](x[None].to(device)).softmax(1)[0].cpu().numpy() for c in CONDS}
    for c in CONDS:
        if abs(pr[c].max() - cs[f"conf_{c}"]) > 1e-3:
            warnings.warn(f"{iid} {c}: {pr[c].max():.5f} vs saved {cs[f'conf_{c}']:.5f}")
    ax[0, j].imshow(show(x)); ax[0, j].axis("off")
    ax[0, j].set_title(f"{iid} ({cs['kind']})\ntrue: {cs['true']}\n{overall} (selected): {cs[f'pred_{overall}']} "
                       f"({cs[f'conf_{overall}']:.2f}) | {other}: {cs[f'pred_{other}']} ({cs[f'conf_{other}']:.2f})", fontsize=9)
    xs = np.arange(K)
    ax[1, j].bar(xs - 0.19, pr[overall], 0.38, color=col[overall], label=f"{overall} (selected)")
    ax[1, j].bar(xs + 0.19, pr[other], 0.38, color=col[other], label=other)
    ax[1, j].set_xticks(xs); ax[1, j].set_xticklabels(BREEDS, rotation=35, ha="right", fontsize=8)
    ax[1, j].set_ylim(0, 1); ax[1, j].set_ylabel("softmax prob."); ax[1, j].legend(fontsize=7)
fig.suptitle("most confident errors of the selected model (image = the 224x224 centre crop the model sees)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(FIG / "fig3_error_cases.png", dpi=150)
plt.show()
pd.DataFrame(cases)[["kind", "image_id", "true", "pred_frozen", "conf_frozen", "pred_finetune", "conf_finetune"]]


# %% code cell 17
for c in CONDS:
    e = pred_df[pred_df[f"pred_{c}"] != pred_df.true]
    print(f"\n{c}: {len(e)} test errors")
    print(e.groupby(["true", f"pred_{c}"]).size().rename("n").to_string())


# %% code cell 18
def gradcam(m, x, target):
    m.eval()
    store = {}
    hk = m.layer4.register_forward_hook(lambda mod, i, o: store.update(A=o))
    try:
        with torch.enable_grad():
            xi = x.detach().clone().requires_grad_(True)
            out = m(xi)
            A = store["A"]
            g = torch.autograd.grad(out[0, target], A)[0]
    finally:
        hk.remove()
    w = g.mean((2, 3), keepdim=True)
    lin = (w * A).sum(1)[0].detach()
    raw = F.relu(lin)
    cam = (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)
    up = F.interpolate(cam[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    return dict(cam=cam.cpu().numpy(), up=up.cpu().numpy(), raw=raw.cpu().numpy(), lin=lin.cpu().numpy(),
                A=A.detach().cpu(), prob=out.detach().softmax(1)[0].cpu().numpy(), logit=out[0, target].item())

def cam_check(m, r, target):
    wc, bc = m.fc.weight[target].detach().cpu(), m.fc.bias[target].item()
    manual = F.relu((wc[:, None, None] * r["A"][0]).sum(0) / (r["A"].shape[2] * r["A"].shape[3]))
    return max((manual - torch.as_tensor(r["raw"])).abs().max().item(), abs(float(r["lin"].sum()) + bc - r["logit"]))

overlay = lambda img, cam: np.clip(0.5 * img + 0.5 * plt.cm.jet(cam)[..., :3], 0, 1)

d = pred_df.copy()
d["wf"], d["wt"] = d.pred_frozen != d.true, d.pred_finetune != d.true
chosen = []
dd = d[d.pred_frozen != d.pred_finetune]
if len(dd):
    chosen.append(dict(dd.loc[(dd.p_true_finetune - dd.p_true_frozen).abs().idxmax()], why="the two models disagree (largest change in p(true))"))
rest = d[(d.wf | d.wt) & ~d.image_id.isin([c["image_id"] for c in chosen])]
if (rest.wf & rest.wt).any():
    bw = rest[rest.wf & rest.wt]
    chosen.append(dict(bw.loc[(2 - bw.p_true_frozen - bw.p_true_finetune).idxmax()], why="both models wrong"))
elif len(rest):
    chosen.append(dict(rest.loc[np.maximum(1 - rest.p_true_frozen, 1 - rest.p_true_finetune).idxmax()], why="misclassified by one model"))
for r in d.assign(mn=np.minimum(d.p_true_frozen, d.p_true_finetune)).sort_values("mn", kind="stable").to_dict("records"):
    if len(chosen) >= 2:
        break
    if r["image_id"] not in [c["image_id"] for c in chosen]:
        chosen.append(dict(r, why="low-confidence correct (no more errors)"))
chosen = [dict(c, source="test") for c in chosen[:2]]

gc = []
for cs in chosen:
    x, yy, iid = get_img(cs)
    assert iid == cs["image_id"]
    t = int(cs["label"])
    r = {c: gradcam(mdl[c], x[None].to(device), t) for c in CONDS}
    for c in CONDS:
        r[c]["dev"] = cam_check(mdl[c], r[c], t)
        if abs(r[c]["prob"].max() - cs[f"conf_{c}"]) > 1e-3:
            warnings.warn(f"{iid} {c}: Grad-CAM forward {r[c]['prob'].max():.5f} vs saved {cs[f'conf_{c}']:.5f}")
    gc.append((cs, x, iid, t, r))

fig, ax = plt.subplots(len(gc), 3, figsize=(13.5, 5.4 * len(gc)), squeeze=False)
for i, (cs, x, iid, t, r) in enumerate(gc):
    img = show(x)
    ax[i, 0].imshow(img)
    ax[i, 0].set_title(f"{iid} (test), input crop\ntrue: {cs['true']}\nfrozen: {cs['pred_frozen']} ({cs['conf_frozen']:.2f}) | "
                       f"fine-tune: {cs['pred_finetune']} ({cs['conf_finetune']:.2f})", fontsize=9)
    for j, c in enumerate(CONDS, 1):
        ax[i, j].imshow(overlay(img, r[c]["up"]))
        ax[i, j].set_title(f"{lab[c]}\nGrad-CAM at layer4, target = {BREEDS[t]}\np(target) = {r[c]['prob'][t]:.3f}, "
                           f"raw CAM max = {r[c]['raw'].max():.3f}", fontsize=9)
    for a in ax[i]:
        a.axis("off")
fig.suptitle("same image, same target class (true breed), same layer, same display (per-map min-max, jet, alpha 0.5)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.5)
fig.savefig(FIG / "fig4_gradcam_pairs.png", dpi=150)
plt.show()

fig, ax = plt.subplots(len(gc), 2, figsize=(9.2, 5.0 * len(gc)), squeeze=False)
for i, (cs, x, iid, t, r) in enumerate(gc):
    img, vmax = show(x), max(r[c]["raw"].max() for c in CONDS) + 1e-6
    for j, c in enumerate(CONDS):
        up = F.interpolate(torch.as_tensor(r[c]["raw"])[None, None] / vmax, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()
        ax[i, j].imshow(overlay(img, up)); ax[i, j].axis("off")
        ax[i, j].set_title(f"{iid}, {lab[c]}\ntarget {BREEDS[t]}, one shared scale (max of both = {vmax:.3f})", fontsize=9)
fig.tight_layout(h_pad=2.0)
fig.savefig(FIG / "figS1_gradcam_shared_scale.png", dpi=130)
plt.show()

gsum = []
for cs, x, iid, t, r in gc:
    for c in CONDS:
        pk = np.unravel_index(np.argmax(r[c]["cam"]), r[c]["cam"].shape)
        gsum.append(dict(image_id=iid, why=cs["why"], true=BREEDS[t], model=c, pred=cs[f"pred_{c}"],
                         p_target=round(float(r[c]["prob"][t]), 4), target_logit=round(r[c]["logit"], 4),
                         raw_cam_max=round(float(r[c]["raw"].max()), 4), raw_cam_mean=round(float(r[c]["raw"].mean()), 4),
                         cells_above_half=int((r[c]["cam"] > 0.5).sum()), peak_cell=f"{pk[0]},{pk[1]}",
                         gradcam_vs_cam_dev=f"{r[c]['dev']:.1e}"))
def map_corr(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return round(float(np.corrcoef(a, b)[0, 1]), 3) if a.std() > 0 and b.std() > 0 else None   # undefined for an empty map

corr = {iid: map_corr(r["frozen"]["cam"], r["finetune"]["cam"]) for cs, x, iid, t, r in gc}
pd.DataFrame(gsum).to_csv(RES / "gradcam_summary.csv", index=False)
dump({"map_correlation": corr}, RES / "gradcam_correlation.json")
print("correlation between the two 7x7 maps:", corr)
pd.DataFrame(gsum)


# %% code cell 19
# Supplement, added after the locked results were known (descriptive, not part of the lock): the same two photos with the
# target set to the breed the selected model predicted, again the same target, layer and display in both models
sup = []
for cs, x, iid, t, r0 in gc:
    tp = BREEDS.index(cs[f"pred_{overall}"])
    if tp == t:
        continue
    rr = {c: gradcam(mdl[c], x[None].to(device), tp) for c in CONDS}
    for c in CONDS:
        rr[c]["dev"] = cam_check(mdl[c], rr[c], tp)
    sup.append((cs, x, iid, tp, rr))
fig, ax = plt.subplots(len(sup), 3, figsize=(13.5, 5.4 * len(sup)), squeeze=False)
for i, (cs, x, iid, tp, rr) in enumerate(sup):
    img = show(x)
    ax[i, 0].imshow(img)
    ax[i, 0].set_title(f"{iid} (test), input crop\ntrue: {cs['true']}\nfrozen: {cs['pred_frozen']} ({cs['conf_frozen']:.2f}) | "
                       f"fine-tune: {cs['pred_finetune']} ({cs['conf_finetune']:.2f})", fontsize=9)
    for j, c in enumerate(CONDS, 1):
        ax[i, j].imshow(overlay(img, rr[c]["up"]))
        ax[i, j].set_title(f"{lab[c]}\nGrad-CAM at layer4, target = {BREEDS[tp]}\n(the breed the selected model predicted)\n"
                           f"p(target) = {rr[c]['prob'][tp]:.3f}, raw CAM max = {rr[c]['raw'].max():.3f}", fontsize=9)
    for a in ax[i]:
        a.axis("off")
fig.suptitle("supplement: target = the breed the selected model predicted; same layer and display as fig4", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.5)
fig.savefig(FIG / "figS2_gradcam_predicted_breed.png", dpi=150)
plt.show()
sup_tab = pd.DataFrame([dict(image_id=iid, true=cs["true"], target=BREEDS[tp], model=c, pred=cs[f"pred_{c}"],
                             p_target=round(float(rr[c]["prob"][tp]), 4), target_logit=round(rr[c]["logit"], 4),
                             raw_cam_max=round(float(rr[c]["raw"].max()), 4), cells_above_half=int((rr[c]["cam"] > 0.5).sum()),
                             peak_cell="{},{}".format(*np.unravel_index(np.argmax(rr[c]["cam"]), rr[c]["cam"].shape)),
                             gradcam_vs_cam_dev=f"{rr[c]['dev']:.1e}")
                        for cs, x, iid, tp, rr in sup for c in CONDS])
sup_tab.to_csv(RES / "gradcam_supplement.csv", index=False)
print("correlation between the two 7x7 maps:", {iid: map_corr(rr["frozen"]["cam"], rr["finetune"]["cam"]) for cs, x, iid, tp, rr in sup})
sup_tab


# %% code cell 20
def lit_check():
    ts_ = json.loads((RES / "training_summary.json").read_text(encoding="utf-8"))
    u5 = ts_["under_floor_5"]
    if u5 is None:
        print("under the literal rule the plateau never came within the logged epochs, so there is nothing to rebuild")
        return pd.DataFrame()
    sel = {c: int(u5["selected"][c]) for c in CONDS}
    h_ = pd.read_csv(RES / "history.csv", dtype={"batches_hash": str})
    w_ = torch.load(CKPT / "warmup_head.pt", map_location="cpu")
    pz = np.load(RES / "test_probs.npz")
    yv = pz["y"]
    P_ = {}
    for c in CONDS:
        m_ = build()
        m_.load_state_dict(w_)
        m_ = setup(m_, c).to(device)
        o_ = make_opt(m_, c)
        for ep in range(1, sel[c] + 1):
            tr_ = run_epoch(m_, o_, ep)
            va_ = evaluate(m_, val_ds)
            row = h_[(h_.condition == c) & (h_.epoch == ep)].iloc[0]
            assert tr_["batches_hash"] == row.batches_hash, f"{c} epoch {ep}: not the logged batches"
            assert va_["correct"] == row.val_correct and abs(va_["loss"] - row.val_loss) < 1e-5, f"{c} epoch {ep}: not reproduced"
        lg_, yt_ = predict(m_, test_ds)
        assert np.array_equal(yt_.numpy(), yv)
        P_[c] = lg_.softmax(1).numpy()
        if sel[c] == ts_["cond"][c]["selected_epoch"]:
            assert np.abs(P_[c] - pz[c]).max() < 1e-5, f"{c}: the rebuilt checkpoint differs from the locked one"
        del m_
    np.savez(RES / "literal_rule_test_probs.npz", y=yv, **P_)
    tm_ = json.loads((RES / "test_metrics.json").read_text(encoding="utf-8"))

    def paired(pf, pt):
        cf, ct = pf == yv, pt == yv
        b10, b01 = int((cf & ~ct).sum()), int((~cf & ct).sum())
        rng = np.random.default_rng(SEED)
        da = np.empty(10000)
        for b in range(10000):
            i = rng.integers(0, len(yv), len(yv))
            da[b] = ct[i].mean() - cf[i].mean()
        return dict(frozen=int(cf.sum()), finetune=int(ct.sum()), f1_frozen=macro_f1(yv, pf), f1_finetune=macro_f1(yv, pt),
                    d_pp=100 * (ct.mean() - cf.mean()), ci=[100 * np.percentile(da, 2.5), 100 * np.percentile(da, 97.5)],
                    p=float(binomtest(min(b10, b01), b10 + b01, 0.5).pvalue) if b10 + b01 else 1.0)

    pz_locked = np.load(RES / "test_probs.npz")
    rows = []
    for name, E_, s_, P2 in (("locked protocol (plateau checks from epoch 100)", ts_["E"],
                               {c: ts_["cond"][c]["selected_epoch"] for c in CONDS}, {c: pz_locked[c] for c in CONDS}),
                              ("the assignment's literal rule (floor of 5 epochs)", u5["E"], sel, P_)):
        r = paired(P2["frozen"].argmax(1), P2["finetune"].argmax(1))
        rows.append({"protocol": name, "epochs": E_, "selected epoch, frozen": s_["frozen"], "selected epoch, fine-tune": s_["finetune"],
                     "frozen correct": f"{r['frozen']}/{len(yv)}", "fine-tune correct": f"{r['finetune']}/{len(yv)}",
                     "macro-F1, frozen": round(r["f1_frozen"], 4), "macro-F1, fine-tune": round(r["f1_finetune"], 4),
                     "fine-tune minus frozen (pp)": round(r["d_pp"], 2), "95% CI": f"[{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]",
                     "exact McNemar p": round(r["p"], 4)})
    assert rows[0]["frozen correct"] == f"{tm_['cond']['frozen']['correct']}/{tm_['n']}"
    dump(dict(rows=rows, note="literal-rule checkpoints rebuilt from the warm-up and checked against the logged run"),
         RES / "literal_rule_check.json")
    return pd.DataFrame(rows).set_index("protocol")


lit_tab = lit_check()
lit_tab


# %% code cell 21
bundle = OUT / f"d370_results_bundle{sfx}.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    for d in (RES, FIG):
        for f in sorted(d.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(OUT))
print(f"{bundle.name}: {bundle.stat().st_size / 2**20:.1f} MiB")


# %% code cell 22
missing = [n for n in ("CONDS", "build", "setup", "make_opt", "train_mode", "evaluate", "predict", "better", "since_best",
                       "warm_device", "macro_f1", "tf_train", "tf_eval", "dev_train", "dev_eval") if n not in globals()]
assert not missing, f"run the Setup, Data and Model cells first (missing: {missing})"
from PIL import Image

SEEDS = [42, 1, 2, 3, 4]        # 42 is the required run: its two color models are reused, not retrained
MODES = ["color", "gray"]
B_EXT = 10000                   # bootstrap resamples, as in the required part
EXT, EXT_CK = RES / "ext_gray", CKPT / "ext_gray"
for d in (EXT / "runs", EXT / "test", EXT_CK):
    d.mkdir(parents=True, exist_ok=True)
assert SEED in SEEDS, "the seed of the required run must be one of the seeds"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

req_lock = json.loads((RES / "locked_choices.json").read_text(encoding="utf-8"))
for c in CONDS:
    assert sha(CKPT / f"{c}_selected.pt") == req_lock["selected"][c]["sha256"], f"{c}: checkpoint does not match the required lock"
sp = json.loads((RES / "split.json").read_text(encoding="utf-8"))
lab_map = {sp["original_label"][b]: sp["new_label"][b] for b in BREEDS}
base_dev = OxfordIIITPet(DATA_ROOT, split="trainval", target_types="category")
base_test = OxfordIIITPet(DATA_ROOT, split="test", target_types="category")
idx = {"train": sp["train_idx"], "val": sp["val_idx"], "test": [i for i, l in enumerate(base_test._labels) if l in lab_map]}
req_pred = pd.read_csv(RES / "test_predictions.csv")
assert [Path(base_dev._images[j]).stem for j in idx["train"]] == sp["train_ids"]
assert [Path(base_dev._images[j]).stem for j in idx["val"]] == sp["val_ids"]
assert [Path(base_test._images[j]).stem for j in idx["test"]] == req_pred.image_id.tolist()

# every image is decoded and resized once; Resize(256) is deterministic, so applying the rest of the
# transform at each access gives exactly the tensors of the pipeline above (checked in the next cell)
t0 = time.perf_counter()
cache = {part: ([transforms.Resize(256)(Image.open(base._images[j]).convert("RGB")) for j in idx[part]],
                [lab_map[base._labels[j]] for j in idx[part]])
         for part, base in [("train", base_dev), ("val", base_dev), ("test", base_test)]}

def tail(part, mode):
    t = [transforms.RandomCrop(224), transforms.RandomHorizontalFlip()] if part == "train" else [transforms.CenterCrop(224)]
    if mode == "gray":
        t.append(transforms.Grayscale(num_output_channels=3))
    return transforms.Compose(t + [transforms.ToTensor(), transforms.Normalize(mean, std)])

class View(torch.utils.data.Dataset):
    def __init__(self, part, mode):
        self.imgs, self.ys = cache[part]
        self.tf = tail(part, mode)
    def __len__(self):
        return len(self.ys)
    def __getitem__(self, i):
        return self.tf(self.imgs[i]), self.ys[i]

class LabelsOnly(torch.utils.data.Dataset):
    def __init__(self, ys):
        self.ys = ys
    def __len__(self):
        return len(self.ys)
    def __getitem__(self, i):
        return 0, self.ys[i]

DS = {(part, mode): View(part, mode) for part in cache for mode in MODES}
dump(dict(seeds=SEEDS, modes=MODES, bootstrap=B_EXT, gray="transforms.Grayscale(num_output_channels=3) after the crop and flip, "
          "before ToTensor and Normalize(ImageNet mean, std); ITU-R 601 luma", settings=settings, env=env), EXT / "ext_config.json")
print(f"{sum(len(v[1]) for v in cache.values())} images cached in {time.perf_counter() - t0:.1f}s:",
      {p: len(v[1]) for p, v in cache.items()})


# %% code cell 23
def make_loader(ds, seed, epoch):
    g = torch.Generator()
    g.manual_seed(seed * 1000 + epoch)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, generator=g)
    torch.manual_seed(seed * 1000 + epoch)
    return loader

def stream_hash(ds, seed, epoch, images=True):
    h = hashlib.sha256()
    for x, y in make_loader(ds, seed, epoch):
        h.update(y.numpy().astype(np.int64).tobytes())
        if images:
            h.update(np.float64(x.double().sum().item()).tobytes())
    return h.hexdigest()[:16]

labels_hash = lambda seed, epoch: stream_hash(LabelsOnly(cache["train"][1]), seed, epoch, images=False)
mt, st = torch.tensor(mean)[:, None, None], torch.tensor(std)[:, None, None]

# 1) the cache reproduces the original pipeline image by image (same RNG state -> identical tensor)
test_ref = OxfordIIITPet(DATA_ROOT, split="test", target_types="category", transform=tf_eval, target_transform=lab_map.__getitem__)
ref = {"train": Subset(dev_train, idx["train"]), "val": Subset(dev_eval, idx["val"]), "test": Subset(test_ref, idx["test"])}
for part, r in ref.items():
    for pos in (0, 1, len(r) - 1):
        torch.manual_seed(7 + pos); a, ya = r[pos]
        torch.manual_seed(7 + pos); b, yb = DS[part, "color"][pos]
        assert ya == yb and torch.equal(a, b), (part, pos)

# 2) the gray image is the luma of the very same crop, identical in all three channels
for part in cache:
    for pos in (0, 5):
        torch.manual_seed(11 + pos); xc = DS[part, "color"][pos][0] * st + mt
        torch.manual_seed(11 + pos); xg = DS[part, "gray"][pos][0] * st + mt
        assert (xg - xg[:1]).abs().max() < 1e-5, (part, pos)
        assert (0.299 * xc[0] + 0.587 * xc[1] + 0.114 * xc[2] - xg[0]).abs().max() <= 1 / 255 + 1e-5, (part, pos)

# 3) replaying the required run's data stream from the cache gives the hashes it logged
hist_req = pd.read_csv(RES / "history.csv", dtype={"batches_hash": str})
logged = {0: json.loads((RES / "warmup.json").read_text(encoding="utf-8"))["train"][0]["batches_hash"]}
logged.update({e: hist_req.loc[(hist_req.condition == "frozen") & (hist_req.epoch == e), "batches_hash"].item() for e in (1, 2)})
for e, h in logged.items():
    assert stream_hash(DS["train", "color"], SEED, e) == h, f"epoch {e}: the cached data stream differs from the logged one"

# 4) the reused seed-42 color models give the locked validation results and the saved test probabilities
req_probs = np.load(RES / "test_probs.npz")
for c in CONDS:
    m = build()
    m.load_state_dict(torch.load(CKPT / f"{c}_selected.pt", map_location="cpu"))
    m = setup(m, c).to(device).eval()
    v = evaluate(m, DS["val", "color"])
    assert v["correct"] == req_lock["selected"][c]["val_correct"] and abs(v["loss"] - req_lock["selected"][c]["val_loss"]) < 1e-5, (c, v)
    lg, yy = predict(m, DS["test", "color"])
    assert np.array_equal(yy.numpy(), req_probs["y"]) and np.abs(lg.softmax(1).numpy() - req_probs[c]).max() < 1e-5, c
del m
print("checks passed: the cached pipeline gives the original tensors; gray = luma of the same crop in 3 equal channels;",
      f"the logged data stream of epochs {sorted(logged)} is reproduced; the seed-42 color models reproduce their",
      "validation and test outputs")


# %% code cell 24
def build_seed(seed):
    """build() from above, with the seed of the new head as an argument"""
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.requires_grad_(False)
    torch.manual_seed(seed)
    m.fc = nn.Linear(512, len(BREEDS))
    return m

assert all(torch.equal(a, b) for a, b in zip(build_seed(SEED).state_dict().values(), build().state_dict().values()))

def ext_epoch(m, opt, epoch, ds, seed):
    """run_epoch from above, with the data set and the seed as arguments"""
    train_mode(m)
    loader = make_loader(ds, seed, epoch)
    h, hl = hashlib.sha256(), hashlib.sha256()
    tot, correct, n = 0.0, 0, 0
    sync()
    t0 = time.perf_counter()
    for x, y in loader:
        yb = y.numpy().astype(np.int64).tobytes()
        h.update(yb)
        h.update(np.float64(x.double().sum().item()).tobytes())
        hl.update(yb)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        out = m(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        opt.step()
        tot += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    sync()
    return dict(train_loss=tot / n, train_acc=correct / n, train_time=time.perf_counter() - t0,
                batches_hash=h.hexdigest()[:16], labels_hash=hl.hexdigest()[:16])

def ckpt_path(seed, mode, c):
    return CKPT / f"{c}_selected.pt" if (seed, mode) == (SEED, "color") else EXT_CK / f"s{seed}_{mode}_{c}.pt"

def required_pair():
    """seed 42 in color is the required run itself, read back from its files"""
    ts = json.loads((RES / "training_summary.json").read_text(encoding="utf-8"))
    for c in CONDS:
        assert ts["cond"][c]["sha256"] == req_lock["selected"][c]["sha256"]
    return dict(seed=SEED, mode="color", reused=True, E=ts["E"], plateau=ts["plateau"],
                cond={c: dict(selected_epoch=ts["cond"][c]["selected_epoch"], val_correct=ts["cond"][c]["selected"]["val_correct"],
                              val_loss=ts["cond"][c]["selected"]["val_loss"], val_macro_f1=ts["cond"][c]["selected"]["val_macro_f1"],
                              sha256=ts["cond"][c]["sha256"], train_time_total=ts["cond"][c]["train_time_total"]) for c in CONDS})

def train_pair(seed, mode):
    """the required protocol (warm-up, two runs in lockstep, epoch rule, selection) for one seed and one input mode"""
    rd = EXT / "runs" / f"s{seed}_{mode}"
    rd.mkdir(parents=True, exist_ok=True)
    if (rd / "summary.json").exists():
        s = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
        if all(ckpt_path(seed, mode, c).exists() and sha(ckpt_path(seed, mode, c)) == s["cond"][c]["sha256"] for c in CONDS):
            print(f"seed {seed} {mode}: trained earlier in {s['minutes']:.1f} min (E = {s['E']}; selected "
                  + ", ".join(f"{c} epoch {s['cond'][c]['selected_epoch']} ({s['cond'][c]['val_correct']}/{len(DS['val', mode])})"
                              for c in CONDS) + "); checkpoints match their hashes, reloaded")
            return s
    tr_ds, va_ds = DS["train", mode], DS["val", mode]
    warm_device()
    t_start = time.perf_counter()
    m = setup(build_seed(seed), "warmup").to(device)
    opt = make_opt(m, "warmup")
    before = evaluate(m, va_ds)
    w_tr = []
    for e in range(WARMUP_EPOCHS):
        w_tr.append(ext_epoch(m, opt, -e, tr_ds, seed))
        assert w_tr[-1]["labels_hash"] == labels_hash(seed, -e)
    w_val = evaluate(m, va_ds)
    w_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    runs = {}
    for c in CONDS:
        m = build_seed(seed)
        m.load_state_dict(w_state)
        m = setup(m, c).to(device)
        s0 = evaluate(m, va_ds)
        if abs(s0["loss"] - w_val["loss"]) > 1e-2:
            raise RuntimeError(f"{c}: reloaded warm-up gives {s0['correct']} / {s0['loss']:.5f}")
        if s0["correct"] != w_val["correct"] or abs(s0["loss"] - w_val["loss"]) > 1e-4:
            warnings.warn(f"{c}: reloaded warm-up {s0['correct']} / {s0['loss']:.6f} vs {w_val['correct']} / {w_val['loss']:.6f}")
        runs[c] = dict(m=m, opt=make_opt(m, c), hist=[], best=None, best_state=None, plateau=None)
    stop = None
    with open(rd / "train_log.txt", "w") as log:
        for ep in range(1, MAX_EPOCHS + 1):
            want = labels_hash(seed, ep)
            for c, R in runs.items():
                tr = ext_epoch(R["m"], R["opt"], ep, tr_ds, seed)
                assert tr["labels_hash"] == want
                va = evaluate(R["m"], va_ds)
                rec = dict(condition=c, epoch=ep, **tr, val_loss=va["loss"], val_acc=va["acc"], val_correct=va["correct"],
                           val_n=va["n"], val_macro_f1=va["macro_f1"])
                R["hist"].append(rec)
                if better(rec, R["best"]):
                    R["best"] = rec
                    R["best_state"] = {k: v.detach().cpu().clone() for k, v in R["m"].state_dict().items()}
                if R["plateau"] is None and ep >= MIN_EPOCHS and since_best(R["hist"]) >= PATIENCE:
                    R["plateau"] = ep
                log.write(f"{c:8s} ep {ep:3d} | train loss {tr['train_loss']:.4f} acc {tr['train_acc']:.3f} | "
                          f"val {va['correct']}/{va['n']} loss {va['loss']:.4f} | {tr['train_time']:.1f}s | {tr['batches_hash']}\n")
            done = all(R["plateau"] for R in runs.values())
            if ep % 25 == 0 or done:
                print(f"  seed {seed} {mode} ep {ep:3d} | " + " | ".join(
                    f"{c} val {R['hist'][-1]['val_correct']}/{len(va_ds)} (best {R['best']['val_correct']} at {R['best']['epoch']})"
                    for c, R in runs.items()))
            if done:
                stop = ep
                break
    if stop is None:
        stop = MAX_EPOCHS
        print(f"  seed {seed} {mode}: hit MAX_EPOCHS before both runs plateaued")
    assert [r["batches_hash"] for r in runs["frozen"]["hist"]] == [r["batches_hash"] for r in runs["finetune"]["hist"]]
    summary = dict(seed=seed, mode=mode, reused=False, E=stop, plateau={c: R["plateau"] for c, R in runs.items()},
                   warmup=dict(val_correct_before=before["correct"], val_correct=w_val["correct"], val_loss=w_val["loss"],
                               batches_hash=w_tr[-1]["batches_hash"]),
                   minutes=(time.perf_counter() - t_start) / 60, cond={})
    for c, R in runs.items():
        sd = R["best_state"]
        changed = [k for k in sd if not torch.equal(sd[k], w_state[k])]
        bad = [k for k in changed if not k.startswith("fc.") and not (c == "finetune" and k.startswith("layer4."))]
        assert not bad, bad
        torch.save(sd, ckpt_path(seed, mode, c))
        b = R["best"]
        summary["cond"][c] = dict(selected_epoch=b["epoch"], val_correct=b["val_correct"], val_loss=b["val_loss"],
                                  val_macro_f1=b["val_macro_f1"], sha256=sha(ckpt_path(seed, mode, c)),
                                  layer4_tensors_changed=sum(k.startswith("layer4.") for k in changed),
                                  train_time_total=sum(r["train_time"] for r in R["hist"]))
    pd.DataFrame([r for R in runs.values() for r in R["hist"]]).to_csv(rd / "history.csv", index=False)
    dump(summary, rd / "summary.json")
    print(f"seed {seed} {mode}: E = {stop}, selected " + ", ".join(
        f"{c} epoch {summary['cond'][c]['selected_epoch']} ({summary['cond'][c]['val_correct']}/{len(va_ds)})" for c in CONDS)
        + f", {summary['minutes']:.1f} min")
    del runs, w_state, m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


# %% code cell 25
todo = [(s, md) for s in SEEDS for md in MODES if (s, md) != (SEED, "color")]
print(f"{len(todo)} pairs besides the required run (finished ones are reloaded)")
summ = {(SEED, "color"): required_pair()}
t_all = time.perf_counter()
for s, md in todo:
    summ[s, md] = train_pair(s, md)
print(f"\nthis cell took {(time.perf_counter() - t_all) / 60:.1f} min (pairs trained earlier are reloaded, not retrained)")
val_tab = pd.DataFrame([dict(seed=s, trained_on=md, model=c, epochs=summ[s, md]["E"],
                             selected_epoch=summ[s, md]["cond"][c]["selected_epoch"],
                             val_correct=summ[s, md]["cond"][c]["val_correct"], val_loss=round(summ[s, md]["cond"][c]["val_loss"], 4))
                        for s in SEEDS for md in MODES for c in CONDS])
val_tab.to_csv(EXT / "validation_summary.csv", index=False)
print(f"validation images correct (out of {len(idx['val'])}) at the selected epoch:")
val_tab.pivot(index="seed", columns=["trained_on", "model"], values="val_correct").loc[SEEDS]


# %% code cell 26
tm = json.loads((RES / "test_metrics.json").read_text(encoding="utf-8"))
models_ = {}
for s in SEEDS:
    for md in MODES:
        for c in CONDS:
            info, p = summ[s, md]["cond"][c], ckpt_path(s, md, c)
            assert sha(p) == info["sha256"], p
            models_[f"s{s}_{md}_{c}"] = dict(seed=s, trained_on=md, condition=c, checkpoint=p.relative_to(OUT).as_posix(),
                                             sha256=info["sha256"], selected_epoch=info["selected_epoch"],
                                             val_correct=info["val_correct"], val_loss=info["val_loss"])
plan = [
    "test: every model below scored once on the color and once on the gray version of the same test images, in the same order",
    "primary, seed 42: test accuracy of each model trained and tested in gray minus the same model trained and tested in color; "
    "exact McNemar on the same images, paired bootstrap (10000, seed 42, the required part's resamples) for the accuracy and "
    "macro-F1 differences",
    "secondary, seed 42: fine-tune minus frozen in gray, tested the same way; difference in differences (fine-tune gain in gray "
    "minus gain in color), paired bootstrap",
    f"replication over seeds {SEEDS}: the same quantities per seed, their mean and SD, and a bootstrap over seeds and test images "
    "(10000, seed 43) for the means",
    "descriptive: color models scored on gray test images; gray-trained models on color test images; recall per breed; confusions "
    "of the gray models; the test images whose p(true) drops most without color (seed 42, mean of the two models)",
]
expect = [
    "if color matters anywhere, it is among the three cat breeds, where all of the color models' errors already are",
    "a color-trained model scored on gray test images does worse than the same kind of model trained on gray",
    "least certain: fine-tuning gains more without color, since layer4 can adapt to the new input and a frozen backbone cannot",
]
known = (f"the required part's color test results for seed 42 (frozen {tm['cond']['frozen']['correct']}/{tm['n']}, "
         f"fine-tune {tm['cond']['finetune']['correct']}/{tm['n']}); no gray test result and no test result for any other seed")
lock_path = EXT / "locked_extension.json"
if lock_path.exists():
    ext_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert ext_lock["models"] == models_ and ext_lock["plan"] == plan, (
        "the checkpoints or the plan differ from the locked ones; re-lock (delete the lock and test/) only if no test result was seen")
    print("already locked at", ext_lock["locked_at"])
    same1 = all(ext_lock["models"][f"s{SEED}_color_{c}"]["sha256"] == req_lock["selected"][c]["sha256"] for c in CONDS)
    print(f"Part 1's two checkpoint hashes recorded in this lock match locked_choices.json: {same1}")
    print(f"locked_choices.json hashed to {ext_lock['required_lock_sha256'][:12]}... when this lock was written and hashes to "
          f"{sha(RES / 'locked_choices.json')[:12]}... now (its locked_at: {req_lock['locked_at']})")
else:
    ext_lock = dict(locked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    required_lock_sha256=sha(RES / "locked_choices.json"), seeds=SEEDS, modes=MODES, models=models_,
                    plan=plan, expectations=expect, known_before_lock=known)
    dump(ext_lock, lock_path)
    print("locked at", ext_lock["locked_at"])
print(f"{len(models_)} models, lock sha256 {sha(lock_path)[:16]}...")
print("\n".join(f"- {p}" for p in plan))


# %% code cell 27
lock_sha = sha(lock_path)
done_path = EXT / "test" / "test_done.json"
probs = {}
if done_path.exists():
    done = json.loads(done_path.read_text(encoding="utf-8"))
    assert done["lock_sha256"] == lock_sha, "the saved test results belong to a different lock"
    for key in ext_lock["models"]:
        z = np.load(EXT / "test" / f"{key}.npz")
        assert np.array_equal(z["y"], req_probs["y"])
        probs[key] = {tv: z[tv] for tv in MODES}
    print("test results reloaded (scored " + done["time"] + ")")
else:
    t0 = time.perf_counter()
    for key, info in ext_lock["models"].items():
        p = OUT / info["checkpoint"]
        assert sha(p) == info["sha256"], key
        m = build_seed(info["seed"])
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m = setup(m, info["condition"]).to(device).eval()
        out = {}
        for tv in MODES:
            lg, yy = predict(m, DS["test", tv])
            assert np.array_equal(yy.numpy(), req_probs["y"])
            out[tv] = lg.softmax(1).numpy()
        np.savez(EXT / "test" / f"{key}.npz", y=yy.numpy(), **out)
        probs[key] = out
    del m
    dump(dict(lock_sha256=lock_sha, time=datetime.now(timezone.utc).isoformat(timespec="seconds"), n=len(req_probs["y"]),
              models=len(probs)), done_path)
    print(f"{len(probs)} models x {len(MODES)} test versions scored in {time.perf_counter() - t0:.0f}s")
for c in CONDS:
    assert np.abs(probs[f"s{SEED}_color_{c}"]["color"] - req_probs[c]).max() < 1e-5
print("seed-42 color models on the color test images: same probabilities as in the required part")


# %% code cell 28
tm = json.loads((RES / "test_metrics.json").read_text(encoding="utf-8"))
y_t = req_probs["y"]
n_t, KB = len(y_t), len(BREEDS)
PR = lambda s, md, c, tv: probs[f"s{s}_{md}_{c}"][tv]
OK = lambda s, md, c, tv: PR(s, md, c, tv).argmax(1) == y_t
lab_e = {"frozen": "frozen (head only)", "finetune": "fine-tune (layer4 + head)"}
col_e = {"frozen": "tab:blue", "finetune": "tab:orange"}

def wilson_ci(k, n, z=1.959963984540054):
    ph, den = k / n, 1 + z * z / n
    mid, half = (ph + z * z / (2 * n)) / den, z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return float(mid - half), float(mid + half)

rows = []
for c in CONDS:
    for md in MODES:
        r = {"model": lab_e[c], "trained on": md}
        for tv in MODES:
            k = int(OK(SEED, md, c, tv).sum())
            lo, hi = wilson_ci(k, n_t)
            r[f"{tv} test: accuracy"] = f"{100 * k / n_t:.2f}% ({k}/{n_t})"
            r[f"{tv} test: 95% CI"] = f"{100 * lo:.1f}-{100 * hi:.1f}"
            r[f"{tv} test: macro-F1"] = round(macro_f1(y_t, PR(SEED, md, c, tv).argmax(1)), 4)
        rows.append(r)
tab2 = pd.DataFrame(rows)
tab2.to_csv(EXT / "table_seed42.csv", index=False)
display(tab2)

rng = np.random.default_rng(SEED)
IDX = np.stack([rng.integers(0, n_t, n_t) for _ in range(B_EXT)])   # drawn exactly like the required bootstrap
bdiff = lambda a, b: b[IDX].mean(1) - a[IDX].mean(1)
bf1 = lambda pa, pb: np.array([macro_f1(y_t[i], pb[i]) - macro_f1(y_t[i], pa[i]) for i in IDX])
spec = {
    "P1 frozen: gray - color": ((SEED, "color", "frozen", "color"), (SEED, "gray", "frozen", "gray")),
    "P2 fine-tune: gray - color": ((SEED, "color", "finetune", "color"), (SEED, "gray", "finetune", "gray")),
    "S1 gray: fine-tune - frozen": ((SEED, "gray", "frozen", "gray"), (SEED, "gray", "finetune", "gray")),
    "D1 frozen color model: gray test - color test": ((SEED, "color", "frozen", "color"), (SEED, "color", "frozen", "gray")),
    "D2 fine-tune color model: gray test - color test": ((SEED, "color", "finetune", "color"), (SEED, "color", "finetune", "gray")),
    "D3 frozen, gray test: trained on gray - on color": ((SEED, "color", "frozen", "gray"), (SEED, "gray", "frozen", "gray")),
    "D4 fine-tune, gray test: trained on gray - on color": ((SEED, "color", "finetune", "gray"), (SEED, "gray", "finetune", "gray")),
}
comp = []
for name, (A, Bq) in spec.items():
    a, b = OK(*A), OK(*Bq)
    b10, b01 = int((a & ~b).sum()), int((~a & b).sum())
    d = bdiff(a, b)
    r = dict(comparison=name, A=f"{int(a.sum())}/{n_t}", B=f"{int(b.sum())}/{n_t}", diff_pp=100 * (b.mean() - a.mean()),
             ci95_pp=[100 * np.percentile(d, 2.5), 100 * np.percentile(d, 97.5)], only_A_right=b10, only_B_right=b01,
             mcnemar_p=float(binomtest(min(b10, b01), b10 + b01, 0.5).pvalue) if b10 + b01 else 1.0)
    if name[0] in "PS":
        pa, pb = PR(*A).argmax(1), PR(*Bq).argmax(1)
        df1 = bf1(pa, pb)
        r.update(macro_f1_diff=macro_f1(y_t, pb) - macro_f1(y_t, pa),
                 macro_f1_ci95=[float(np.percentile(df1, 2.5)), float(np.percentile(df1, 97.5))])
    comp.append(r)

fr_c, ft_c, fr_g, ft_g = (OK(SEED, md, c, md) for md, c in [("color", "frozen"), ("color", "finetune"), ("gray", "frozen"), ("gray", "finetune")])
d_col = bdiff(fr_c, ft_c)
assert np.allclose([np.percentile(d_col, 2.5), np.percentile(d_col, 97.5)], tm["paired"]["d_acc_ci"]), "resamples differ from the required part"
did_b = 100 * (bdiff(fr_g, ft_g) - d_col)
comp.append(dict(comparison="S2 (fine-tune - frozen in gray) - (fine-tune - frozen in color)",
                 diff_pp=100 * ((ft_g.mean() - fr_g.mean()) - (ft_c.mean() - fr_c.mean())),
                 ci95_pp=[float(np.percentile(did_b, 2.5)), float(np.percentile(did_b, 97.5))]))
dump(comp, EXT / "comparisons_seed42.json")
show_c = pd.DataFrame(comp).set_index("comparison")
show_c["diff_pp"] = show_c.diff_pp.map(lambda v: f"{v:+.2f}")
show_c["ci95_pp"] = show_c.ci95_pp.map(lambda v: f"[{v[0]:+.2f}, {v[1]:+.2f}]")
show_c["mcnemar_p"] = show_c.mcnemar_p.map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
show_c["macro_f1_diff"] = show_c.macro_f1_diff.map(lambda v: "" if pd.isna(v) else f"{v:+.4f}")
show_c["macro_f1_ci95"] = show_c.macro_f1_ci95.map(lambda v: f"[{v[0]:+.4f}, {v[1]:+.4f}]" if isinstance(v, list) else "")
for col_ in ("only_A_right", "only_B_right"):
    show_c[col_] = show_c[col_].map(lambda v: "" if pd.isna(v) else str(int(v)))
display(show_c.astype(object).where(show_c.notna(), ""))
for c in CONDS:
    ac, ag = OK(SEED, "color", c, "color").mean(), OK(SEED, "gray", c, "gray").mean()
    print(f"{lab_e[c]}: {100 * ac:.2f}% in color -> {100 * ag:.2f}% in gray; errors {round(n_t * (1 - ac))} -> "
          f"{round(n_t * (1 - ag))}; keeps {100 * (ag - 1 / KB) / (ac - 1 / KB):.1f}% of its accuracy above chance (1/{KB})")


# %% code cell 29
pc = []
for k, b in enumerate(BREEDS):
    sel = y_t == k
    r = {"breed": b, "n": int(sel.sum())}
    for c in CONDS:
        r[f"{c}: color"] = 100 * OK(SEED, "color", c, "color")[sel].mean()
        r[f"{c}: gray"] = 100 * OK(SEED, "gray", c, "gray")[sel].mean()
        r[f"{c}: change"] = r[f"{c}: gray"] - r[f"{c}: color"]
        r[f"{c}: color model, gray test"] = 100 * OK(SEED, "color", c, "gray")[sel].mean()
    pc.append(r)
pc = pd.DataFrame(pc).set_index("breed")
pc.to_csv(EXT / "per_breed_recall_seed42.csv")
print(f"test recall per breed (%), seed {SEED}")
display(pc.round(1))

for c in CONDS:
    pred_g = PR(SEED, "gray", c, "gray").argmax(1)
    e = pd.DataFrame(dict(true=[BREEDS[i] for i in y_t], pred=[BREEDS[i] for i in pred_g]))
    e = e[e.true != e.pred]
    lost = OK(SEED, "color", c, "color") & ~OK(SEED, "gray", c, "gray")
    gained = ~OK(SEED, "color", c, "color") & OK(SEED, "gray", c, "gray")
    print(f"\n{lab_e[c]}, trained and tested in gray: {len(e)} errors; {int(lost.sum())} images right in color but wrong "
          f"in gray, {int(gained.sum())} the other way round")
    if len(e):
        print(e.groupby(["true", "pred"]).size().rename("n").to_string())

fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
for a_, c in zip(ax, CONDS):
    cm = confusion_matrix(y_t, PR(SEED, "gray", c, "gray").argmax(1), labels=list(range(KB)))
    a_.imshow(cm, cmap="Greys")
    for i in range(KB):
        for j in range(KB):
            a_.text(j, i, cm[i, j], ha="center", va="center", fontsize=10, color="white" if cm[i, j] > cm.max() / 2 else "black")
    a_.set_xticks(range(KB)); a_.set_xticklabels(BREEDS, rotation=35, ha="right")
    a_.set_yticks(range(KB)); a_.set_yticklabels(BREEDS)
    a_.set_xlabel("predicted"); a_.set_ylabel("true")
    a_.set_title(f"{lab_e[c]}, trained and tested in gray (seed {SEED})\n"
                 f"test acc {100 * OK(SEED, 'gray', c, 'gray').mean():.2f}% (n = {n_t})", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "figE3_confusion_gray.png", dpi=150)
plt.show()


# %% code cell 30
ar = np.arange(n_t)
drop = np.mean([PR(SEED, "color", c, "color")[ar, y_t] - PR(SEED, "gray", c, "gray")[ar, y_t] for c in CONDS], axis=0)
top = np.argsort(-drop, kind="stable")[:4]
to_img = lambda x: (x * st + mt).clamp(0, 1).permute(1, 2, 0).numpy()
fig, ax = plt.subplots(2, len(top), figsize=(4.2 * len(top), 10.2), squeeze=False)
for j, i in enumerate(top):
    for row, md in enumerate(MODES):
        ax[row, j].imshow(to_img(DS["test", md][i][0]))
        ax[row, j].axis("off")
        guess = ", ".join(f"{c} {BREEDS[PR(SEED, md, c, md)[i].argmax()]} ({PR(SEED, md, c, md)[i].max():.2f})" for c in CONDS)
        head = f"{req_pred.image_id[i]} (true: {BREEDS[y_t[i]]})\n" if row == 0 else ""
        ax[row, j].set_title(f"{head}{md} models: {guess}".replace(": ", ":\n", 1) if row else f"{head}{md} models:\n{guess}", fontsize=8.5)
fig.suptitle(f"test images that need color the most (largest drop in p(true), mean of the two models, seed {SEED});\n"
             "top row: what the color models see, bottom row: what the gray models see", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=3.0)
fig.savefig(FIG / "figE4_color_dependent_images.png", dpi=130)
plt.show()
pd.DataFrame(dict(image_id=req_pred.image_id.values[top], true=[BREEDS[y_t[i]] for i in top], drop_in_p_true=drop[top].round(4)))


# %% code cell 31
from matplotlib.lines import Line2D
cells4 = [("color", "frozen"), ("color", "finetune"), ("gray", "frozen"), ("gray", "finetune")]
OKS = np.stack([np.stack([OK(s, md, c, md) for md, c in cells4]) for s in SEEDS]).astype(float)   # seeds x 4 x images

def derived(acc):
    return {"frozen: gray - color": acc[..., 2] - acc[..., 0], "fine-tune: gray - color": acc[..., 3] - acc[..., 1],
            "fine-tune - frozen, color": acc[..., 1] - acc[..., 0], "fine-tune - frozen, gray": acc[..., 3] - acc[..., 2],
            "difference in differences": (acc[..., 3] - acc[..., 2]) - (acc[..., 1] - acc[..., 0])}

acc_s = OKS.mean(2)
seed_tab = pd.DataFrame({"seed": SEEDS, **{f"{md}-trained {c} (%)": 100 * acc_s[:, j] for j, (md, c) in enumerate(cells4)},
                         **{f"{k} (pp)": 100 * v for k, v in derived(acc_s).items()}}).set_index("seed")
rng_h = np.random.default_rng(SEED + 1)
S_ = len(SEEDS)
hb = {k: np.empty(B_EXT) for k in derived(acc_s)}
for b in range(B_EXT):
    si, ii = rng_h.integers(0, S_, S_), rng_h.integers(0, n_t, n_t)
    for k, v in derived(OKS[si][:, :, ii].mean(2)).items():
        hb[k][b] = v.mean()
ci = {f"{k} (pp)": [100 * np.percentile(v, 2.5), 100 * np.percentile(v, 97.5)] for k, v in hb.items()}
seed_tab.to_csv(EXT / "seeds.csv")
dump(dict(per_seed=seed_tab.reset_index().to_dict("records"), mean=seed_tab.mean().to_dict(), sd=seed_tab.std().to_dict(),
          ci95_over_seeds_and_images=ci), EXT / "seeds_summary.json")
display(pd.concat([seed_tab, pd.DataFrame({"mean": seed_tab.mean(), "SD": seed_tab.std()}).T]).round(2))
print("95% intervals for the means over seeds (bootstrap over seeds and test images):")
for k, v in ci.items():
    print(f"  {k}: mean {seed_tab[k].mean():+.2f}, [{v[0]:+.2f}, {v[1]:+.2f}]")

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))
pts = [("trained and tested in color", "color", "color", "o", True, -0.2), ("trained and tested in gray", "gray", "gray", "s", True, 0.0),
       ("color model, gray test images", "color", "gray", "^", False, 0.2)]
for xi, c in enumerate(CONDS):
    for name, md, tv, mk, filled, dx in pts:
        k = int(OK(SEED, md, c, tv).sum())
        lo, hi = wilson_ci(k, n_t)
        a = 100 * k / n_t
        ax[0].errorbar(xi + dx, a, yerr=[[max(a - 100 * lo, 0)], [max(100 * hi - a, 0)]], fmt=mk, ms=9, color=col_e[c],
                       mfc=col_e[c] if filled else "white", mew=1.5, capsize=3)
        ax[0].annotate(f"{a:.1f}", (xi + dx, a), textcoords="offset points", xytext=(7, -4), fontsize=8)
ax[0].legend(handles=[Line2D([], [], marker=mk, ls="", color="k", mfc="k" if filled else "white", ms=8, label=name)
                      for name, md, tv, mk, filled, dx in pts], fontsize=8, loc="lower left")
ax[0].set_xticks(range(len(CONDS))); ax[0].set_xticklabels([lab_e[c] for c in CONDS]); ax[0].set_xlim(-0.5, len(CONDS) - 0.5)
ax[0].set_ylabel("test accuracy (%)"); ax[0].grid(alpha=0.3)
ax[0].set_title(f"seed {SEED}: n = {n_t} test images, Wilson 95% CI", fontsize=10)
for c, dx in zip(CONDS, (-0.04, 0.04)):
    for s in SEEDS:
        ax[1].plot([dx, 1 + dx], [100 * OK(s, md, c, md).mean() for md in MODES], "-o", color=col_e[c], alpha=0.35, lw=1, ms=4)
    ax[1].plot([dx, 1 + dx], [100 * np.mean([OK(s, md, c, md).mean() for s in SEEDS]) for md in MODES], "-o", color=col_e[c],
               lw=3, ms=9, label=f"{lab_e[c]}, mean of {len(SEEDS)} seeds")
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["trained and tested in color", "trained and tested in gray"])
ax[1].set_xlim(-0.3, 1.3); ax[1].set_ylabel("test accuracy (%)"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
ax[1].set_title(f"seeds {SEEDS}; thin lines are single seeds", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "figE1_color_vs_gray.png", dpi=150)
plt.show()


# %% code cell 32
rec = {(c, md): np.array([[OK(s, md, c, md)[y_t == k].mean() for k in range(KB)] for s in SEEDS]) for c in CONDS for md in MODES}
pcs = pd.DataFrame({f"{c}: {md}": 100 * rec[c, md].mean(0) for c in CONDS for md in MODES}, index=BREEDS)
for c in CONDS:
    pcs[f"{c}: change"] = pcs[f"{c}: gray"] - pcs[f"{c}: color"]
pcs.index.name = "breed"
pcs.to_csv(EXT / "per_breed_recall_mean_over_seeds.csv")
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True, sharex=True)
for a_, c in zip(ax, CONDS):
    for k, b in enumerate(BREEDS):
        mc, mg = pcs.loc[b, f"{c}: color"], pcs.loc[b, f"{c}: gray"]
        a_.plot([100 * rec[c, "gray"][:, k].min(), 100 * rec[c, "gray"][:, k].max()], [k + 0.15] * 2, "-", color=col_e[c], alpha=0.5, lw=1)
        a_.plot([mc, mg], [k, k], "-", color="0.6", lw=1.5)
        a_.plot(mc, k, "o", color=col_e[c], ms=8)
        a_.plot(mg, k, "s", color=col_e[c], mfc="white", mew=1.5, ms=8)
        a_.annotate(f"{mg - mc:+.1f}", ((mc + mg) / 2, k), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
    a_.set_yticks(range(KB)); a_.set_yticklabels(BREEDS); a_.set_xlabel("test recall (%)"); a_.grid(alpha=0.3)
    a_.set_title(f"{lab_e[c]}: filled = color, open = gray, mean of {len(SEEDS)} seeds\n(thin line: range of the gray seeds)", fontsize=9)
ax[0].invert_yaxis()
fig.tight_layout()
fig.savefig(FIG / "figE2_per_breed_recall.png", dpi=150)
plt.show()
pcs.round(1)


# %% code cell 33
import math
from collections import Counter
PB_SIZES = [1, 5, 20]                       # photos per breed; 80, the full split, is Part 2a
PB_DIR, PB_CK = RES / "ext_size", CKPT / "ext_size"
for _d in (PB_DIR / "runs", PB_DIR / "test", PB_CK):
    _d.mkdir(parents=True, exist_ok=True)
PB_YTR = np.array(cache["train"][1])        # labels of the cached training photos, in Part 1's order
PB_POS = {b: np.flatnonzero(PB_YTR == b) for b in range(len(BREEDS))}
assert all(len(v) >= max(PB_SIZES) for v in PB_POS.values())


def pb_subset(k, seed):
    """positions of the first k training photos of each breed, in an order fixed by the seed; nested across k"""
    rng = np.random.default_rng([seed, 2026])
    return sorted(int(j) for b in range(len(BREEDS)) for j in rng.permutation(PB_POS[b])[:k])


assert all(set(pb_subset(a, s)) <= set(pb_subset(b, s)) for s in SEEDS for a, b in zip(PB_SIZES, PB_SIZES[1:]))
assert all(pb_subset(80, s) == list(range(len(PB_YTR))) for s in SEEDS) or len(PB_YTR) != 80 * len(BREEDS)


def pb_ckpt(k, seed, mode, c):
    return PB_CK / f"k{k}_s{seed}_{mode}_{c}.pt"


def pb_load(k, seed, mode, c):
    """the ImageNet ResNet18 with the trained part of a Part 2b model: the head, and layer4 for the fine-tuned one"""
    m = build_seed(seed)
    sd = torch.load(pb_ckpt(k, seed, mode, c), map_location="cpu")
    assert all(q.startswith("fc.") or (c == "finetune" and q.startswith("layer4.")) for q in sd)
    _, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected
    return setup(m, c).to(device).eval()


def pb_train(k, seed, mode):
    """Part 1's protocol on k photos per breed (warm-up, lockstep, epoch rule, selection), as train_pair above"""
    rd = PB_DIR / "runs" / f"k{k}_s{seed}_{mode}"
    rd.mkdir(parents=True, exist_ok=True)
    if (rd / "summary.json").exists():
        s = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
        if all(pb_ckpt(k, seed, mode, c).exists() and sha(pb_ckpt(k, seed, mode, c)) == s["cond"][c]["sha256"] for c in CONDS):
            return dict(s, reloaded=True)
    sub = pb_subset(k, seed)
    tr_ds, va_ds = Subset(DS["train", mode], sub), DS["val", mode]
    labs = LabelsOnly([int(PB_YTR[j]) for j in sub])
    lh = lambda e: stream_hash(labs, seed, e, images=False)
    warm_device()
    t_start = time.perf_counter()
    m = setup(build_seed(seed), "warmup").to(device)
    opt = make_opt(m, "warmup")
    w_tr = []
    for e in range(WARMUP_EPOCHS):
        w_tr.append(ext_epoch(m, opt, -e, tr_ds, seed))
        assert w_tr[-1]["labels_hash"] == lh(-e)
    w_val = evaluate(m, va_ds)
    w_state = {q: v.detach().cpu().clone() for q, v in m.state_dict().items()}
    runs = {}
    for c in CONDS:
        m = build_seed(seed)
        m.load_state_dict(w_state)
        m = setup(m, c).to(device)
        s0 = evaluate(m, va_ds)
        if abs(s0["loss"] - w_val["loss"]) > 1e-2:
            raise RuntimeError(f"{c}: reloaded warm-up gives {s0['correct']} / {s0['loss']:.5f}")
        runs[c] = dict(m=m, opt=make_opt(m, c), hist=[], best=None, best_state=None, plateau=None)
    stop = None
    with open(rd / "train_log.txt", "w") as log:
        for ep in range(1, MAX_EPOCHS + 1):
            want = lh(ep)
            for c, R in runs.items():
                tr = ext_epoch(R["m"], R["opt"], ep, tr_ds, seed)
                assert tr["labels_hash"] == want
                va = evaluate(R["m"], va_ds)
                rec = dict(condition=c, epoch=ep, **tr, val_loss=va["loss"], val_acc=va["acc"], val_correct=va["correct"],
                           val_n=va["n"], val_macro_f1=va["macro_f1"])
                R["hist"].append(rec)
                if better(rec, R["best"]):
                    R["best"] = rec
                    R["best_state"] = {q: v.detach().cpu().clone() for q, v in R["m"].state_dict().items()}
                if R["plateau"] is None and ep >= MIN_EPOCHS and since_best(R["hist"]) >= PATIENCE:
                    R["plateau"] = ep
                log.write(f"{c:8s} ep {ep:3d} | train loss {tr['train_loss']:.4f} | val {va['correct']}/{va['n']} "
                          f"loss {va['loss']:.4f} | {tr['batches_hash']}\n")
            if all(R["plateau"] for R in runs.values()):
                stop = ep
                break
    stop = stop or MAX_EPOCHS
    assert [r["batches_hash"] for r in runs["frozen"]["hist"]] == [r["batches_hash"] for r in runs["finetune"]["hist"]]
    summary = dict(k=k, seed=seed, mode=mode, E=stop, plateau={c: R["plateau"] for c, R in runs.items()},
                   photos=[Path(base_dev._images[idx["train"][j]]).stem for j in sub],
                   warmup=dict(val_correct=w_val["correct"], val_loss=w_val["loss"]), minutes=(time.perf_counter() - t_start) / 60, cond={})
    for c, R in runs.items():
        sd = R["best_state"]
        changed = [q for q in sd if not torch.equal(sd[q], w_state[q])]
        assert not [q for q in changed if not q.startswith("fc.") and not (c == "finetune" and q.startswith("layer4."))]
        torch.save({q: v for q, v in sd.items() if q.startswith("fc.") or (c == "finetune" and q.startswith("layer4."))},
                   pb_ckpt(k, seed, mode, c))
        b = R["best"]
        summary["cond"][c] = dict(selected_epoch=b["epoch"], val_correct=b["val_correct"], val_loss=b["val_loss"],
                                  val_macro_f1=b["val_macro_f1"], sha256=sha(pb_ckpt(k, seed, mode, c)))
    pd.DataFrame([r for R in runs.values() for r in R["hist"]]).to_csv(rd / "history.csv", index=False)
    dump(summary, rd / "summary.json")
    del runs, w_state, m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def pb_train_all():
    out, t0 = {}, time.perf_counter()
    for k in PB_SIZES:
        for s in SEEDS:
            for md in MODES:
                out[k, s, md] = pb_train(k, s, md)
                r = out[k, s, md]
                print(f"k = {k:2d}, seed {s:2d}, {md:5s}: E = {r['E']:3d}, selected "
                      + ", ".join(f"{c} epoch {r['cond'][c]['selected_epoch']} ({r['cond'][c]['val_correct']}/120)" for c in CONDS)
                      + f"  [trained in {60 * r['minutes']:.0f}s{', reloaded' if r.get('reloaded') else ''}]")
    print(f"\n{len(out)} pairs, {sum(r['minutes'] for r in out.values()):.1f} min of training in total; this cell took "
          f"{(time.perf_counter() - t0) / 60:.1f} min (pairs trained earlier are reloaded after their hashes are checked)")
    return out


PB_SUMM = pb_train_all()
pb_val = pd.DataFrame([dict(photos_per_breed=k, seed=s, trained_on=md, model=c, epochs=PB_SUMM[k, s, md]["E"],
                            selected_epoch=PB_SUMM[k, s, md]["cond"][c]["selected_epoch"],
                            val_correct=PB_SUMM[k, s, md]["cond"][c]["val_correct"])
                       for k in PB_SIZES for s in SEEDS for md in MODES for c in CONDS])
pb_val.to_csv(PB_DIR / "validation_summary.csv", index=False)
pb_val.pivot_table(index="photos_per_breed", columns=["trained_on", "model"], values="val_correct", aggfunc="mean").round(1)


# %% code cell 34
def pb_load_test():
    rs, cc = transforms.Resize(256), transforms.CenterCrop(224)
    rs_nn = transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST)
    rgb, tri = [], []
    for j in idx["test"]:
        stem = Path(base_test._images[j]).stem
        with Image.open(base_test._images[j]) as f:
            rgb.append(torch.from_numpy(np.array(cc(rs(f.convert("RGB"))), dtype=np.uint8)).permute(2, 0, 1))
        with Image.open(Path(base_test._anns_folder) / "trimaps" / f"{stem}.png") as f:
            tri.append(np.array(cc(rs_nn(f)), dtype=np.uint8))
    return torch.stack(rgb), np.stack(tri)


PB_RGB, PB_TRI = pb_load_test()
PB_Y = req_probs["y"]
assert set(np.unique(PB_TRI)) <= {1, 2, 3}
_m, _s = torch.tensor(mean).view(1, 3, 1, 1), torch.tensor(std).view(1, 3, 1, 1)
for _pos in (0, len(PB_Y) // 2, len(PB_Y) - 1):
    assert torch.equal(((PB_RGB[_pos:_pos + 1].float().div(255) - _m) / _s)[0], DS["test", "color"][_pos][0]), _pos
PB_M, PB_S = _m.to(device), _s.to(device)
PB_REGION = {"pet": torch.from_numpy(PB_TRI != 2)[:, None].float(),             # the pet and its border
             "background": torch.from_numpy(PB_TRI == 2)[:, None].float(),
             "whole photo": torch.ones(len(PB_Y), 1, 224, 224)}
PB_CONDS = {"pet, hue +180°": ("pet", 180, 1.0), "background, hue +180°": ("background", 180, 1.0),
            "whole photo, hue +180°": ("whole photo", 180, 1.0), "whole photo, hue +90°": ("whole photo", 90, 1.0),
            "whole photo, hue −90°": ("whole photo", -90, 1.0), "whole photo, hue +30°": ("whole photo", 30, 1.0),
            "whole photo, hue −30°": ("whole photo", -30, 1.0), "whole photo, saturation 50%": ("whole photo", 0, 0.5),
            "whole photo, saturation 0%": ("whole photo", 0, 0.0)}
PB_LUMA = lambda x: 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def pb_recolor(u8, sl, cond):
    """uint8 photos -> the same photos with their chroma rotated and scaled inside a region, brightness kept, rounded to 8 bits"""
    region, angle, sat = PB_CONDS[cond]
    x = u8.to(device).float().div(255)
    Y = PB_LUMA(x)
    cb, cr = (x[:, 2:3] - Y) / 1.772, (x[:, 0:1] - Y) / 1.402
    t = math.radians(angle)
    cb2, cr2 = sat * (math.cos(t) * cb - math.sin(t) * cr), sat * (math.sin(t) * cb + math.cos(t) * cr)
    r2, b2 = Y + 1.402 * cr2, Y + 1.772 * cb2
    g2 = (Y - 0.299 * r2 - 0.114 * b2) / 0.587
    w = PB_REGION[region][sl].to(device)
    out = w * torch.cat([r2, g2, b2], 1) + (1 - w) * x
    return (out.clamp(0, 1) * 255).round().div(255)


def pb_slices(n):
    return [slice(a, min(a + EVAL_BATCH, n)) for a in range(0, n, EVAL_BATCH)]


def pb_check_luma():
    rows = []
    with torch.no_grad():
        for cond, (region, _, _) in PB_CONDS.items():
            moved = changed = clipped = inside = 0
            for sl in pb_slices(len(PB_Y)):
                x = PB_RGB[sl].to(device).float().div(255)
                y8 = pb_recolor(PB_RGB[sl], sl, cond)
                w = PB_REGION[region][sl].to(device)
                dY = (PB_LUMA(y8) - PB_LUMA(x)).abs() * 255
                moved += int(((dY > 1.0) * w).sum())
                changed += int((((y8 - x).abs().amax(1, keepdim=True) > 0.5 / 255) * w).sum())
                inside += int(w.sum())
            rows.append({"change": cond, "pixels in the region (%)": 100 * inside / (len(PB_Y) * 224 * 224),
                         "region pixels whose color changed (%)": 100 * changed / max(inside, 1),
                         "region pixels whose brightness moved by more than 1 of 255 (%)": 100 * moved / max(inside, 1)})
    return pd.DataFrame(rows).set_index("change")


def pb_fig_examples():
    show = [int(np.flatnonzero(PB_Y == BREEDS.index(b))[0]) for b in ("Abyssinian", "Pug", "Samoyed") if b in BREEDS]
    cols = ["original", "pet, hue +180°", "background, hue +180°", "whole photo, hue +180°", "whole photo, hue +30°",
            "whole photo, saturation 0%"]
    fig, ax = plt.subplots(len(show), len(cols), figsize=(2.9 * len(cols), 3.1 * len(show)), squeeze=False)
    for r, i in enumerate(show):
        for c_, cond in enumerate(cols):
            img = PB_RGB[i:i + 1].float().div(255) if cond == "original" else pb_recolor(PB_RGB[i:i + 1], slice(i, i + 1), cond).cpu()
            ax[r, c_].imshow(img[0].permute(1, 2, 0).numpy())
            ax[r, c_].set_xticks([]); ax[r, c_].set_yticks([])
            if r == 0:
                ax[r, c_].set_title(cond, fontsize=9)
        ax[r, 0].set_ylabel(f"{req_pred.image_id[i]}\n({BREEDS[PB_Y[i]]})", fontsize=9)
    fig.suptitle("changing only the color: the brightness, and so the black-and-white photo, stays the same", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "figE11_recolored_examples.png", dpi=130)
    plt.show()


pb_luma_tab = pb_check_luma()
pb_luma_tab.to_csv(PB_DIR / "recolor_brightness_check.csv")
pb_fig_examples()
pb_luma_tab.round(3)


# %% code cell 35
def pb_lock():
    models_b = {}
    for k in PB_SIZES:
        for s in SEEDS:
            for md in MODES:
                for c in CONDS:
                    info, p = PB_SUMM[k, s, md]["cond"][c], pb_ckpt(k, s, md, c)
                    assert sha(p) == info["sha256"], p
                    models_b[f"k{k}_s{s}_{md}_{c}"] = dict(k=k, seed=s, trained_on=md, condition=c, checkpoint=p.relative_to(OUT).as_posix(),
                                                           sha256=info["sha256"], selected_epoch=info["selected_epoch"],
                                                           val_correct=info["val_correct"], val_loss=info["val_loss"])
    plan = [f"need: for k in {PB_SIZES} photos per breed and both model types, accuracy of the black-and-white model on black-and-white test "
            "photos minus the color model of the same seed on color test photos, over the five seeds, with a bootstrap over seeds and photos; "
            "k = 80 is Part 2a",
            "use: for k in " + str(PB_SIZES + [80]) + " and both model types, the color models' accuracy on recolored test photos minus on the "
            "original ones, for 'pet, hue +180°' (main), 'background, hue +180°' and 'whole photo, saturation 0%'; the pet-minus-background "
            "and the wrong-minus-missing ('whole photo, hue +180°' minus 'saturation 0%') differences; same bootstrap",
            "descriptive: hue +30° and −30° (a realistic white-balance error), hue +90° and −90°, saturation 50%, models learned in "
            "black and white on color photos, per-breed results, "
            "confusions, mean probability of the true breed and log loss",
            "check: black-and-white models on the black-and-white version of recolored photos, against the original ones"]
    known = ("all test results of Part 2a (80 photos per breed, five seeds), including its per-breed results; no test result of any model "
             "with fewer photos, and no result of any model on a recolored photo")
    if PB_LOCKFILE.exists():
        lk = json.loads(PB_LOCKFILE.read_text(encoding="utf-8"))
        assert lk["models"] == models_b and lk["plan"] == plan, "models or plan differ from the lock"
        print("already locked at", lk["locked_at"])
        return lk
    lk = dict(locked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), sizes=PB_SIZES, seeds=SEEDS, models=models_b,
              plan=plan, conditions={c: list(v) for c, v in PB_CONDS.items()}, expectations=PB_EXPECT, known_before_lock=known)
    dump(lk, PB_LOCKFILE)
    print("locked at", lk["locked_at"])
    return lk


PB_EXPECT = ["with fewer photos per breed, learning without color costs more than at 80",
             "rotating the pet's colors by 180° lowers the color models' accuracy at every training size, more with fewer photos",
             "rotating the pet's colors costs more than rotating the background's",
             "the cost falls on Abyssinian, Bengal and Pug and is close to zero for the Samoyed",
             "least certain: a wrong color (whole photo, hue +180°) costs a color model more than no color (saturation 0%)"]
PB_LOCKFILE = PB_DIR / "locked_size_and_color.json"
PB_LOCKED = pb_lock()


# %% code cell 36
def pb_score_all():
    lock_sha = sha(PB_LOCKFILE)
    done = PB_DIR / "test" / "test_done.json"
    keys80 = {f"k80_s{s}_{md}_{c}": f"s{s}_{md}_{c}" for s in SEEDS for md in MODES for c in CONDS}
    if done.exists():
        d = json.loads(done.read_text(encoding="utf-8"))
        assert d["lock_sha256"] == lock_sha, "the saved results belong to a different lock"
        z, names = np.load(PB_DIR / "test" / "all_probs.npz"), json.loads((PB_DIR / "test" / "all_probs_keys.json").read_text(encoding="utf-8"))
        out = {n: z[f"a{i}"] for i, n in enumerate(names)}
        print("test results reloaded (scored " + d["time"] + ")")
        return out
    out, t0 = {}, time.perf_counter()

    def load(key):
        if key in keys80:
            info = ext_lock["models"][keys80[key]]
            assert sha(OUT / info["checkpoint"]) == info["sha256"], key
            m = build_seed(info["seed"])
            m.load_state_dict(torch.load(OUT / info["checkpoint"], map_location="cpu"))
            return setup(m, info["condition"]).to(device).eval()
        info = PB_LOCKED["models"][key]
        assert sha(OUT / info["checkpoint"]) == info["sha256"], key
        return pb_load(info["k"], info["seed"], info["trained_on"], info["condition"])

    with torch.no_grad():
        for key, info in PB_LOCKED["models"].items():          # 1) the new models on both versions of the test photos
            m = load(key)
            for tv in MODES:
                lg, yy = predict(m, DS["test", tv])
                assert np.array_equal(yy.numpy(), PB_Y)
                out[f"{key}|{tv}"] = lg.softmax(1).numpy()
            del m
        for k in PB_SIZES + [80]:                                # 2) every color model on every recolored version
            for s in SEEDS:
                for c in CONDS:
                    key = f"k{k}_s{s}_color_{c}"
                    m = load(key)
                    base = np.concatenate([m((PB_RGB[sl].to(device).float().div(255) - PB_M) / PB_S).softmax(1).cpu().numpy()
                                           for sl in pb_slices(len(PB_Y))])
                    ref = probs[keys80[key]]["color"] if k == 80 else out[f"{key}|color"]
                    assert np.abs(base - ref).max() < 1e-4, f"{key}: the rebuilt photos give different probabilities"
                    for cond in PB_CONDS:
                        out[f"{key}|{cond}"] = np.concatenate([m((pb_recolor(PB_RGB[sl], sl, cond) - PB_M) / PB_S).softmax(1).cpu().numpy()
                                                               for sl in pb_slices(len(PB_Y))])
                    del m
        to_gray = transforms.Grayscale(num_output_channels=3)    # 3) the control: black-and-white models cannot see the change
        for s in SEEDS:
            for c in CONDS:
                key = f"k80_s{s}_gray_{c}"
                m = load(key)
                ps = []
                for sl in pb_slices(len(PB_Y)):
                    x8 = (pb_recolor(PB_RGB[sl], sl, "whole photo, hue +180°") * 255).round().byte().cpu()
                    g = torch.stack([transforms.functional.pil_to_tensor(to_gray(transforms.functional.to_pil_image(im))) for im in x8])
                    ps.append(m((g.to(device).float().div(255) - PB_M) / PB_S).softmax(1).cpu().numpy())
                out[f"{key}|control"] = np.concatenate(ps)
                del m
    names = list(out)                                         # file-safe array names; the readable keys are kept next to them
    np.savez(PB_DIR / "test" / "all_probs.npz", **{f"a{i}": out[n] for i, n in enumerate(names)})
    (PB_DIR / "test" / "all_probs_keys.json").write_text(json.dumps(names, ensure_ascii=False, indent=0), encoding="utf-8")
    dump(dict(lock_sha256=lock_sha, time=datetime.now(timezone.utc).isoformat(timespec="seconds"), arrays=len(out)), done)
    print(f"{len(out)} score sets in {time.perf_counter() - t0:.0f}s")
    return out


PB_P = pb_score_all()
pb_ctrl = [dict(seed=s, model=c, changed=int((PB_P[f"k80_s{s}_gray_{c}|control"].argmax(1) != probs[f"s{s}_gray_{c}"]["gray"].argmax(1)).sum()),
                largest_p_change=float(np.abs(PB_P[f"k80_s{s}_gray_{c}|control"] - probs[f"s{s}_gray_{c}"]["gray"]).max()))
           for s in SEEDS for c in CONDS]
print("control, black-and-white models on the black-and-white version of photos whose colors were rotated by 180°:")
pd.DataFrame(pb_ctrl).set_index(["seed", "model"])


# %% code cell 37
def pb_get(k, s, md, c, tv):
    """probabilities of one model on one version of the test photos, from Part 2a (k = 80) or Part 2b"""
    return probs[f"s{s}_{md}_{c}"][tv] if k == 80 else PB_P[f"k{k}_s{s}_{md}_{c}|{tv}"]


def pb_ci(stat, *arrs, seed):
    rng = np.random.default_rng(seed)
    S, N = arrs[0].shape[:2]
    out = np.empty(B_EXT)
    for b in range(B_EXT):
        si, ii = rng.integers(0, S, S), rng.integers(0, N, N)
        out[b] = stat(*(a[si][:, ii] for a in arrs))
    return np.percentile(out, [2.5, 97.5])


def pb_need():
    ar = np.arange(len(PB_Y))
    okk = lambda P: (P.argmax(1) == PB_Y).astype(float)
    nll = lambda P: -np.log(np.clip(P[ar, PB_Y], 1e-12, None))
    rows, curves = [], {}
    for c in CONDS:
        for k in PB_SIZES + [80]:
            g = lambda md, tv, f=okk: np.stack([f(pb_get(k, s, md, c, tv)) for s in SEEDS])
            A, B, GC, CG = g("color", "color"), g("gray", "gray"), g("gray", "color"), g("color", "gray")
            LA, LB = g("color", "color", nll), g("gray", "gray", nll)
            lo, hi = pb_ci(lambda a, b: b.mean() - a.mean(), A, B, seed=SEED + k)
            llo, lhi = pb_ci(lambda a, b: b.mean() - a.mean(), LA, LB, seed=SEED + k + 1)
            curves[c, k] = {n_: (100 * a.mean(), *(100 * pb_ci(np.mean, a, seed=SEED + k + 2))) for n_, a in
                            (("color on color", A), ("B&W on B&W", B), ("B&W-trained on color", GC))}
            rows.append({"model": lab_e[c], "photos per breed": k, "color on color (%)": 100 * A.mean(), "B&W on B&W (%)": 100 * B.mean(),
                         "B&W minus color (pp)": 100 * (B.mean() - A.mean()), "CI low": 100 * lo, "CI high": 100 * hi,
                         "log loss, B&W minus color": LB.mean() - LA.mean(), "log loss CI": f"[{llo:+.3f}, {lhi:+.3f}]",
                         "B&W-trained on color photos (%)": 100 * GC.mean(), "color-trained on B&W photos (%)": 100 * CG.mean()})
    return pd.DataFrame(rows), curves


def pb_fig_need(curves):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    ks = PB_SIZES + [80]
    for a_, c in zip(ax, CONDS):
        for name, col, ls in (("color on color", "tab:red", "-"), ("B&W on B&W", "0.3", "-"), ("B&W-trained on color", "0.3", "--")):
            mu = np.array([curves[c, k][name][0] for k in ks])
            lo = np.array([curves[c, k][name][1] for k in ks])
            hi = np.array([curves[c, k][name][2] for k in ks])
            a_.plot(ks, mu, ls, marker="o", color=col, label=name)
            a_.fill_between(ks, lo, hi, color=col, alpha=0.12)
        a_.set_xscale("log")
        a_.set_xticks(ks); a_.set_xticklabels([str(k) for k in ks])
        a_.set_xlabel("training photos per breed")
        a_.set_title(lab_e[c], fontsize=10)
        a_.grid(alpha=0.3)
    ax[0].set_ylabel("test accuracy (%), mean of five seeds, 95% CI")
    ax[0].legend(fontsize=8.5)
    fig.suptitle("how much color is needed: color vs black-and-white models by training size", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "figE12_need_by_size.png", dpi=150)
    plt.show()


pb_need_tab, pb_curves = pb_need()
pb_need_tab.to_csv(PB_DIR / "need_by_size.csv", index=False)
pb_fig_need(pb_curves)
display(pb_need_tab.round(2))


def pb_ft_vs_frozen():
    """not in the lock, descriptive: the fine-tuned minus the frozen model of the same seed, both trained and tested on the
    same kind of photo, by training size; the 80-photo rows reuse Part 2a's resamples (seed 43) and so repeat its intervals"""
    okk = lambda P: (P.argmax(1) == PB_Y).astype(float)
    rows = []
    for md in MODES:
        for k in PB_SIZES + [80]:
            A = np.stack([okk(pb_get(k, s, md, "frozen", md)) for s in SEEDS])
            B = np.stack([okk(pb_get(k, s, md, "finetune", md)) for s in SEEDS])
            lo, hi = pb_ci(lambda a, b: b.mean() - a.mean(), A, B, seed=SEED + 1 if k == 80 else SEED + k + 13)
            rows.append({"trained and tested on": md, "photos per breed": k, "frozen (%)": 100 * A.mean(),
                         "fine-tuned (%)": 100 * B.mean(), "fine-tuned minus frozen (pp)": 100 * (B.mean() - A.mean()),
                         "CI low": 100 * lo, "CI high": 100 * hi})
    return pd.DataFrame(rows)


pb_ftfr_tab = pb_ft_vs_frozen()
pb_ftfr_tab.to_csv(PB_DIR / "finetune_minus_frozen_by_size.csv", index=False)
pb_ftfr_tab.round(2)


# %% code cell 38
def pb_use():
    ok = lambda P: (P.argmax(1) == PB_Y).astype(float)
    main = ["pet, hue +180°", "background, hue +180°", "whole photo, hue +180°", "whole photo, saturation 0%"]
    rows, curves = [], {}
    for c in CONDS:
        for k in PB_SIZES + [80]:
            g = lambda cond: np.stack([ok(PB_P[f"k{k}_s{s}_color_{c}|{cond}"]) for s in SEEDS])
            O = np.stack([ok(pb_get(k, s, "color", c, "color")) for s in SEEDS])
            r = {"model": lab_e[c], "photos per breed": k, "original (%)": 100 * O.mean()}
            for cond in PB_CONDS:
                X = g(cond)
                r[f"{cond} (pp)"] = 100 * (X.mean() - O.mean())
                if cond in main:
                    lo, hi = pb_ci(lambda a, b: b.mean() - a.mean(), O, X, seed=SEED + 3 * k)
                    r[f"{cond}: CI"] = f"[{100 * lo:+.2f}, {100 * hi:+.2f}]"
                    curves[c, k, cond] = (100 * (X.mean() - O.mean()), 100 * lo, 100 * hi)
            Pt, Bg, Wh, Gr = g(main[0]), g(main[1]), g(main[2]), g(main[3])
            X30 = (g("whole photo, hue +30°") + g("whole photo, hue −30°")) / 2
            lo, hi = pb_ci(lambda a, b: b.mean() - a.mean(), O, X30, seed=SEED + 7 * k)
            r["whole photo, hue ±30°, mean of both (pp)"] = 100 * (X30.mean() - O.mean())
            r["whole photo, hue ±30°: CI"] = f"[{100 * lo:+.2f}, {100 * hi:+.2f}]"
            curves[c, k, "whole photo, hue ±30° (mean of both)"] = (100 * (X30.mean() - O.mean()), 100 * lo, 100 * hi)
            for name, a, b in (("pet minus background (pp)", Bg, Pt), ("wrong minus missing color (pp)", Gr, Wh)):
                lo, hi = pb_ci(lambda x, y: y.mean() - x.mean(), a, b, seed=SEED + 5 * k)
                r[name] = 100 * (b.mean() - a.mean())
                r[name.replace(" (pp)", ": CI")] = f"[{100 * lo:+.2f}, {100 * hi:+.2f}]"
            rows.append(r)
    return pd.DataFrame(rows), curves


def pb_breeds():
    ok = lambda P: (P.argmax(1) == PB_Y).astype(float)
    rows = []
    for k in (80, PB_SIZES[0]):
        for c in CONDS:
            O = np.stack([ok(pb_get(k, s, "color", c, "color")) for s in SEEDS])
            X = np.stack([ok(PB_P[f"k{k}_s{s}_color_{c}|pet, hue +180°"]) for s in SEEDS])
            rows.append({"photos per breed": k, "model": lab_e[c],
                         **{b: 100 * (X[:, PB_Y == i].mean() - O[:, PB_Y == i].mean()) for i, b in enumerate(BREEDS)}})
    return pd.DataFrame(rows).set_index(["photos per breed", "model"])


def pb_moves(k=80, cond="pet, hue +180°"):
    rows = []
    for s in SEEDS:
        for c in CONDS:
            before, after_ = pb_get(k, s, "color", c, "color").argmax(1), PB_P[f"k{k}_s{s}_color_{c}|{cond}"].argmax(1)
            for i in np.flatnonzero((before == PB_Y) & (after_ != PB_Y)):
                rows.append((BREEDS[PB_Y[i]], BREEDS[after_[i]]))
    return pd.DataFrame([(t, p, n) for (t, p), n in Counter(rows).most_common()],
                        columns=["true breed", "called after the change", "count"])


def pb_fig_use(curves, br):
    ks = PB_SIZES + [80]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.7), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    styles = {"pet, hue +180°": ("tab:red", "-"), "background, hue +180°": ("tab:green", "-"),
              "whole photo, hue +180°": ("tab:purple", "--"), "whole photo, hue ±30° (mean of both)": ("tab:brown", "-."),
              "whole photo, saturation 0%": ("0.35", ":")}
    for a_, c in zip(ax[:2], CONDS):
        for cond, (col, ls) in styles.items():
            mu = np.array([curves[c, k, cond][0] for k in ks])
            a_.plot(ks, mu, ls, marker="o", color=col, label=cond)
            a_.fill_between(ks, [curves[c, k, cond][1] for k in ks], [curves[c, k, cond][2] for k in ks], color=col, alpha=0.12)
        a_.axhline(0, color="0.5", lw=0.8)
        a_.set_xscale("log"); a_.set_xticks(ks); a_.set_xticklabels([str(k) for k in ks])
        a_.set_xlabel("training photos per breed")
        a_.set_title(f"color models, {lab_e[c]}", fontsize=10)
        a_.grid(alpha=0.3)
    ax[0].set_ylabel("change in test accuracy (pp), 95% CI")
    ax[0].legend(fontsize=8)
    x = np.arange(len(BREEDS))
    for off, (key, c) in zip((-0.3, -0.1, 0.1, 0.3), [(k_, c_) for c_ in CONDS for k_ in (80, PB_SIZES[0])]):
        if (key, lab_e[c]) in br.index:
            ax[2].bar(x + off, br.loc[(key, lab_e[c])].to_numpy(dtype=float), 0.19, color=col_e[c],
                      alpha=0.9 if key == 80 else 0.45, edgecolor="k", lw=0.5,
                      label=f"{lab_e[c]}, {key} photo{'s' if key > 1 else ''} per breed")
    ax[2].axhline(0, color="0.5", lw=0.8)
    ax[2].set_xticks(x); ax[2].set_xticklabels(BREEDS, fontsize=9)
    ax[2].set_ylabel("change in accuracy with the pet's colors rotated (pp)")
    ax[2].set_title("by breed (mean of five seeds)", fontsize=10)
    ax[2].legend(fontsize=7.5); ax[2].grid(alpha=0.3, axis="y")
    fig.suptitle("how much color is used: color models on photos whose colors were changed, brightness kept", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "figE13_use_of_color.png", dpi=150)
    plt.show()


pb_use_tab, pb_use_curves = pb_use()
pb_breed_tab = pb_breeds()
pb_moves_tab = pb_moves()
for _t, _n in ((pb_use_tab, "use_by_size.csv"), (pb_breed_tab, "use_by_breed.csv"), (pb_moves_tab, "moves_pet_hue180_k80.csv")):
    _t.to_csv(PB_DIR / _n)
pb_fig_use(pb_use_curves, pb_breed_tab)
display(pb_use_tab.round(2))
display(pb_breed_tab.round(1))
pb_moves_tab.head(12)


# %% code cell 39
def pb_cam(m, x):
    """logits and the true breed's class activation maps (positive part, N x 7 x 7) at layer4"""
    a = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    a = m.layer4(m.layer3(m.layer2(m.layer1(a))))
    lg = m.fc(torch.flatten(m.avgpool(a), 1))
    return lg, a


def pb_fig_cam():
    ar = np.arange(len(PB_Y))
    drop = np.mean([pb_get(80, SEED, "color", c, "color")[ar, PB_Y] - PB_P[f"k80_s{SEED}_color_{c}|pet, hue +180°"][ar, PB_Y] for c in CONDS], 0)
    top = np.argsort(-drop, kind="stable")[:4]
    fig, ax = plt.subplots(len(top), 4, figsize=(13.2, 3.95 * len(top)), squeeze=False)
    with torch.no_grad():
        for col0, c in ((0, "frozen"), (2, "finetune")):
            info = ext_lock["models"][f"s{SEED}_color_{c}"]
            m = build_seed(SEED)
            m.load_state_dict(torch.load(OUT / info["checkpoint"], map_location="cpu"))
            m = setup(m, c).to(device).eval()
            W = m.fc.weight
            for r, i in enumerate(top):
                ims = [PB_RGB[i:i + 1].to(device).float().div(255), pb_recolor(PB_RGB[i:i + 1], slice(i, i + 1), "pet, hue +180°")]
                maps, ps = [], []
                for x in ims:
                    lg, feat = pb_cam(m, (x - PB_M) / PB_S)
                    maps.append(torch.einsum("nkhw,k->nhw", feat, W[int(PB_Y[i])]).clamp(min=0)[0].cpu())
                    ps.append(lg.softmax(1)[0].cpu().numpy())
                lim = max(float(mm.max()) for mm in maps) or 1.0
                for j, (x, mm, p) in enumerate(zip(ims, maps, ps)):
                    a_ = ax[r, col0 + j]
                    up = torch.nn.functional.interpolate(mm[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()
                    rgba = plt.get_cmap("jet")(up / lim)
                    rgba[..., 3] = 0.6 * (up / lim) ** 0.8
                    a_.imshow(x[0].permute(1, 2, 0).cpu().numpy())
                    a_.imshow(rgba)
                    a_.set_title(f"{lab_e[c]}, {'original' if j == 0 else 'pet recolored'}\np(true) {p[PB_Y[i]]:.2f}, "
                                 f"says {BREEDS[int(p.argmax())]}", fontsize=8.5)
                    a_.set_xticks([]); a_.set_yticks([])
                ax[r, 0].set_ylabel(f"{req_pred.image_id[i]}\n(true: {BREEDS[PB_Y[i]]})", fontsize=9)
            del m
    fig.suptitle(f"Grad-CAM of the true breed at layer4, 80 photos per breed, seed {SEED}: original photo against the same photo with "
                 "the pet's colors rotated by 180°;\neach model's two maps share one scale", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.8)
    fig.savefig(FIG / "figE14_gradcam_recolored.png", dpi=130)
    plt.show()


pb_fig_cam()


# %% code cell 40
from IPython.display import Markdown


def pb_summary():
    def rows_md(df):
        df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
        f = lambda v: f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)
        return (["| " + " | ".join(str(c_) for c_ in df.columns) + " |", "|" + "---|" * len(df.columns)]
                + ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])
    L = ["# Part 2b: fewer photos, and changing only the color (generated from the saved results)", "",
         f"locked {PB_LOCKED['locked_at']}; sizes {PB_SIZES} plus 80 (Part 2a); seeds {SEEDS}; bootstrap {B_EXT} resamples", ""]
    for title, df in (("Literal stopping rule (Part 1)", lit_tab), ("Validation, mean correct of 120", pb_val.pivot_table(
            index="photos_per_breed", columns=["trained_on", "model"], values="val_correct", aggfunc="mean").round(1)),
                      ("Brightness check of the color changes", pb_luma_tab), ("Control: black-and-white models", pd.DataFrame(pb_ctrl)),
                      ("Need, by training size", pb_need_tab),
                      ("Fine-tuned minus frozen, by training size (not in the lock)", pb_ftfr_tab), ("Use, by training size", pb_use_tab),
                      ("Use by breed: pet's colors rotated by 180° (pp)", pb_breed_tab), ("Where the recolored photos go (80 photos)", pb_moves_tab)):
        L += [f"## {title}", ""] + rows_md(df) + [""]
    text_ = "\n".join(L) + "\n"
    (PB_DIR / "SUMMARY.md").write_text(text_, encoding="utf-8")
    return text_


display(Markdown(pb_summary()))


# %% code cell 41
def md_table(df, fmt="{:.2f}"):
    out = ["| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |", "|" + "---|" * (len(df.columns) + 1)]
    for i, r in df.iterrows():
        out.append("| " + " | ".join([str(i)] + [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]) + " |")
    return out

L = ["# Extension: what is left without color? (generated from the saved results)", "",
     f"locked {ext_lock['locked_at']}, test scored {json.loads(done_path.read_text(encoding="utf-8"))['time']}, "
     f"{len(ext_lock['models'])} models, seeds {SEEDS}", "", f"## Seed {SEED} (n = {n_t} test images)", "",
     "| model | trained on | color test | gray test |", "|---|---|---|---|"]
for c in CONDS:
    for md in MODES:
        L.append(f"| {lab_e[c]} | {md} | " + " | ".join(f"{100 * OK(SEED, md, c, tv).mean():.2f}% ({int(OK(SEED, md, c, tv).sum())}/{n_t})"
                                                     for tv in MODES) + " |")
L += ["", "| comparison | difference (pp) | 95% CI | McNemar p | macro-F1 difference [95% CI] |", "|---|---|---|---|---|"]
for r in comp:
    f1s = f"{r['macro_f1_diff']:+.4f} [{r['macro_f1_ci95'][0]:+.4f}, {r['macro_f1_ci95'][1]:+.4f}]" if "macro_f1_diff" in r else ""
    L.append(f"| {r['comparison']} | {r['diff_pp']:+.2f} | [{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}] | "
             f"{r['mcnemar_p']:.4f} | {f1s} |" if "mcnemar_p" in r else
             f"| {r['comparison']} | {r['diff_pp']:+.2f} | [{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}] |  | {f1s} |")
L += [""] + [f"- {lab_e[c]}: errors {round(n_t * (1 - OK(SEED, 'color', c, 'color').mean()))} in color, "
             f"{round(n_t * (1 - OK(SEED, 'gray', c, 'gray').mean()))} in gray" for c in CONDS]
L += ["", f"## All seeds {SEEDS}", ""] + md_table(pd.concat([seed_tab, pd.DataFrame({"mean": seed_tab.mean(), "SD": seed_tab.std()}).T]))
L += ["", "95% intervals for the means (bootstrap over seeds and test images):", ""]
L += [f"- {k}: mean {seed_tab[k].mean():+.2f}, [{v[0]:+.2f}, {v[1]:+.2f}]" for k, v in ci.items()]
L += ["", "## Recall by breed (%), mean over seeds", ""] + md_table(pcs, "{:.1f}")
text = "\n".join(L) + "\n"
(EXT / "SUMMARY.md").write_text(text, encoding="utf-8")

bundle = OUT / f"d370_results_bundle{sfx}.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    for d in (RES, FIG):
        for f in sorted(d.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(OUT))
print(f"results/ext_gray/SUMMARY.md written; {bundle.name}: {bundle.stat().st_size / 2**20:.1f} MiB")
from IPython.display import Markdown
display(Markdown(text))
