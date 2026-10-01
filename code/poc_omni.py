#!/usr/bin/env python3
"""Part B: omni-embedding channel decomposition on the PoC ads (nvidia/omni-embed-nemotron-3b).

Conditions per ad (one omni embedder, content list = subset of channels):
  V (muted video), P (prosody wav, low-pass 400 Hz), T (transcript), VP, VT, PT, VPT -> 3-player Shapley
  A (raw audio incl. words), VA (original ad)                                        -> leakage references
Scores per condition:
  anchor : zero-shot, mean cos to R text anchors - mean cos to D text anchors (cross-modal, CLIP-style)
  proto  : leave-one-out prototype, cos(e_i, mean_R(-i) - mean_D(-i))  (within-condition, modality-gap-free)
  auc_loo: leave-one-out logistic regression AUC (informativeness)
Shapley: exact 3-player values on z-scored per-condition scores, v(empty) = 0; interaction = v(VPT)-v(V)-v(P)-v(T).
Conflict test: cross-party composites V_i+P_j, V_i+T_j, P_i+T_j; which source's party the composite follows.
Outputs: out/omni_emb.json (cache), out/results_B.json
"""
import csv, itertools, json, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from qwen_omni_utils import process_mm_info
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(ROOT, "data"), os.path.join(ROOT, "out")
VAR, TR = os.path.join(OUT, "var"), os.path.join(OUT, "transcripts")
CACHE = os.path.join(OUT, "omni_emb.json")
MODEL = "nvidia/omni-embed-nemotron-3b"
PROC_PATH = os.path.join(ROOT, "model_local")  # snapshot symlinks minus additional_chat_templates (transformers 4.57 loader bug)
VIDEO_KW = {"min_pixels": 32 * 14 * 14, "max_pixels": 64 * 28 * 28, "use_audio_in_video": False}
FPS, MAX_FRAMES = 1.0, 64
PLAYERS = ["V", "P", "T"]
CONDS = ["V", "P", "T", "VP", "VT", "PT", "VPT", "A", "VA"]
ANCHORS = {
    "R": ["a Republican campaign ad", "a political ad supporting Donald Trump",
          "a conservative Republican political advertisement", "a Trump 2024 campaign video"],
    "D": ["a Democratic campaign ad", "a political ad supporting Kamala Harris and Joe Biden",
          "a liberal Democratic political advertisement", "a Harris 2024 campaign video"],
}
N_SWAPS_PER_DIRECTION = 10
SEED = 0
SHAPLEY_W = {0: 1 / 3, 1: 1 / 6, 2: 1 / 3}  # |S|!(n-|S|-1)!/n! for n = 3
DEVIATIONS = [
    "n = 22 ads (10 R / 12 D) from official 2024 candidate channels; heuristic, not inferential",
    "prosody channel = low-pass 400 Hz + loudnorm (words unintelligible; pitch, rhythm, energy kept); music bass survives",
    "video sampled at 1 fps, max 64 frames, low resolution (max_pixels 64*28*28) to fit 16 GB",
    "every condition carries the constant prefix 'passage: ' so that conditions differ only by channel content",
    "LOO prototype score has the usual leave-one-out mean bias (uniform across ads); anchors are hand-written",
    "AUCs on 22 ads have SE ~ 0.12; Shapley shares are descriptive",
]


def transcript(vid):
    p = os.path.join(TR, vid + ".json")
    return json.load(open(p))["text"] if os.path.exists(p) else ""


def content(vid, cond, text=None):
    c = []
    if "V" in cond:
        c.append({"type": "video", "video": os.path.join(VAR, f"{vid}_V.mp4"), "fps": FPS, "max_frames": MAX_FRAMES})
    if "P" in cond:
        c.append({"type": "audio", "audio": os.path.join(VAR, f"{vid}_P.wav")})
    if "A" in cond:
        c.append({"type": "audio", "audio": os.path.join(OUT, "audio", f"{vid}.wav")})
    t = text if text is not None else (transcript(vid) if "T" in cond else "")
    c.append({"type": "text", "text": "passage: " + t})
    return c


def load():
    m = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                  trust_remote_code=True).to("cuda").eval()
    return m, AutoProcessor.from_pretrained(PROC_PATH, trust_remote_code=True)


@torch.no_grad()
def embed(m, p, cont):
    torch.cuda.empty_cache()
    docs = [{"role": "user", "content": cont}]
    text = p.apply_chat_template(docs, add_generation_prompt=False, tokenize=False)
    audio, images, videos = process_mm_info(docs, use_audio_in_video=False)
    batch = p(text=text, images=images, videos=videos, audio=audio, return_tensors="pt",
              text_kwargs={"truncation": True, "padding": True, "max_length": 204800},
              videos_kwargs=VIDEO_KW, audio_kwargs={"max_length": 2048000})
    batch = {k: v.to(m.device) for k, v in batch.items()}
    h = m(**batch, output_hidden_states=True).hidden_states[-1]
    mask = batch["attention_mask"]
    e = h.masked_fill(~mask[..., None].bool(), 0.0).sum(1) / mask.sum(1)[..., None]
    return F.normalize(e.float(), dim=-1)[0].cpu().numpy()


def unit(x):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def zscore(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-8 else 1.0)


def loo_proba(X, y, C=1.0):
    p = np.zeros(len(y))
    for i in range(len(y)):
        idx = np.arange(len(y)) != i
        p[i] = LogisticRegression(C=C, max_iter=2000).fit(X[idx], y[idx]).predict_proba(X[i:i + 1])[0, 1]
    return p


def proto_loo(E, y):
    s = np.zeros(len(y))
    for i in range(len(y)):
        idx = np.arange(len(y)) != i
        proto = E[idx & (y == 1)].mean(0) - E[idx & (y == 0)].mean(0)
        s[i] = E[i] @ unit(proto)
    return s


def shapley(v):
    """v: dict cond -> array over ads for V,P,T,VP,VT,PT,VPT. Exact 3-player Shapley values."""
    phi = {}
    for c in PLAYERS:
        others = [o for o in PLAYERS if o != c]
        tot = 0.0
        for k in range(3):
            for S in itertools.combinations(others, k):
                key_s = "".join(x for x in PLAYERS if x in S)
                key_sc = "".join(x for x in PLAYERS if x in S or x == c)
                tot = tot + SHAPLEY_W[k] * (v[key_sc] - (v[key_s] if key_s else 0.0))
        phi[c] = tot
    return phi


def main():
    t0 = time.time()
    rows = list(csv.DictReader(open(os.path.join(DATA, "manifest.csv"))))
    ids = [r["id"] for r in rows]
    y = np.array([1 if r["label"].upper().startswith("R") else 0 for r in rows])
    m, p = load()
    print(f"model loaded {time.time() - t0:.0f}s, vram {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

    def get(key, cont):
        if key not in cache:
            cache[key] = embed(m, p, cont).tolist()
            json.dump(cache, open(CACHE, "w"))
        return np.array(cache[key])

    # anchors as queries
    A = {k: unit(np.array([get(f"anchor:{k}:{i}", [{"type": "text", "text": "query: " + a}])
                            for i, a in enumerate(v)])) for k, v in ANCHORS.items()}
    # ad embeddings per condition
    E = {c: [] for c in CONDS}
    for n, vid in enumerate(ids):
        for c in CONDS:
            E[c].append(get(f"{vid}:{c}", content(vid, c)))
        print(f"  [{n + 1}/{len(ids)}] {vid} done, {time.time() - t0:.0f}s", flush=True)
    E = {c: unit(np.array(v)) for c, v in E.items()}

    # per-condition scores
    scores = {"anchor": {}, "proto": {}}
    auc = {"anchor": {}, "proto": {}, "logistic_loo": {}}
    for c in CONDS:
        sa = (E[c] @ A["R"].T).mean(1) - (E[c] @ A["D"].T).mean(1)
        sp = proto_loo(E[c], y)
        scores["anchor"][c], scores["proto"][c] = sa, sp
        auc["anchor"][c] = float(roc_auc_score(y, sa))
        auc["proto"][c] = float(roc_auc_score(y, sp))
        auc["logistic_loo"][c] = float(roc_auc_score(y, loo_proba(E[c], y)))

    # Shapley on z-scored scores
    shap = {}
    for kind in ("anchor", "proto"):
        v = {c: zscore(scores[kind][c]) for c in ["V", "P", "T", "VP", "VT", "PT", "VPT"]}
        phi = shapley(v)
        assert np.allclose(phi["V"] + phi["P"] + phi["T"], v["VPT"], atol=1e-6)
        inter = v["VPT"] - v["V"] - v["P"] - v["T"]
        mag = {c: float(np.abs(phi[c]).mean()) for c in PLAYERS}
        tot = sum(mag.values())
        shap[kind] = {
            "share_of_abs_contribution": {c: mag[c] / tot for c in PLAYERS},
            "auc_phi_vs_party": {c: float(roc_auc_score(y, phi[c])) for c in PLAYERS},
            "mean_phi_R": {c: float(phi[c][y == 1].mean()) for c in PLAYERS},
            "mean_phi_D": {c: float(phi[c][y == 0].mean()) for c in PLAYERS},
            "interaction_mean_abs": float(np.abs(inter).mean()),
            "interaction_auc_vs_party": float(roc_auc_score(y, inter)),
        }

    # modality gap inside the omni model: distance between condition means
    means = {c: E[c].mean(0) for c in CONDS}
    gap = {f"{a}-{b}": float(np.linalg.norm(means[a] - means[b])) for a, b in [("V", "T"), ("P", "T"), ("V", "P"), ("A", "P")]}

    # conflict test: cross-party composites, scored with full-sample prototypes of the joint condition
    rng = random.Random(SEED)
    R, D = [i for i in range(len(ids)) if y[i] == 1], [i for i in range(len(ids)) if y[i] == 0]
    pairs = [(rng.choice(R), rng.choice(D)) for _ in range(N_SWAPS_PER_DIRECTION)] + \
            [(rng.choice(D), rng.choice(R)) for _ in range(N_SWAPS_PER_DIRECTION)]
    protos = {c: unit(E[c][y == 1].mean(0) - E[c][y == 0].mean(0)) for c in ("VP", "VT", "PT")}
    conflict = {}
    for name, cond, build in [
        ("V_over_P", "VP", lambda i, j: content(ids[i], "V")[:1] + content(ids[j], "P")),
        ("V_over_T", "VT", lambda i, j: content(ids[i], "V")[:1] + [{"type": "text", "text": "passage: " + transcript(ids[j])}]),
        ("P_over_T", "PT", lambda i, j: content(ids[i], "P")[:1] + [{"type": "text", "text": "passage: " + transcript(ids[j])}]),
    ]:
        wins = []
        for i, j in pairs:
            e = unit(get(f"swap:{name}:{ids[i]}:{ids[j]}", build(i, j)))
            pred = 1 if e @ protos[cond] > 0 else 0
            wins.append(int(pred == y[i]))  # first source's party wins?
        conflict[name] = {"first_source_wins": float(np.mean(wins)), "n_pairs": len(pairs)}

    res = {"n": len(ids), "n_R": int(y.sum()), "model": MODEL, "conditions": CONDS,
           "auc": auc, "shapley": shap, "modality_gap_norm": gap, "conflict": conflict,
           "runtime_s": round(time.time() - t0), "deviations": DEVIATIONS}
    json.dump(res, open(os.path.join(OUT, "results_B.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k not in ("deviations", "conditions")}, indent=1))


if __name__ == "__main__":
    main()
