#!/usr/bin/env python3
"""PoC mini-replication of Ciccarone (2026), "An Image is Worth 1000 Words? Partisanship and
Attention in Videos", Sections 2.1-2.3, 3 (exposure) and 4.1 (SpLICE concepts).

Pipeline: frame every 3 s + matched transcript chunk -> CLIP ViT-B/32 (512-d) -> per-ad mean
-> per-modality mean-centering (modality gap, Liang et al. 2022) -> per-modality classifier
+ stacked model -> exposure curve (prefix of k seconds) -> SpLICE-lite concept decomposition.

Input : data/manifest.csv with columns id,label,video,subs (label in {R,D}; paths relative to data/)
Output: out/results.json, out/embeddings.npz, out/frames/<id>/, out/audio/<id>.wav (for part B)
"""
import csv, glob, json, os, re, subprocess, sys
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(ROOT, "data"), os.path.join(ROOT, "out")
STEP = 3.0  # paper: one frame every three seconds, transcript cut into as many chunks
EXPOSURES = [3, 6, 12, 24, 45]  # seconds of prefix, for the attention/exposure curve
CONCEPTS = [
    # topics
    "immigration", "border", "economy", "jobs", "inflation", "taxes", "healthcare", "abortion",
    "guns", "crime", "police", "climate", "education", "military", "veterans", "corruption",
    "democracy", "freedom", "family", "workers", "farmers", "small business", "energy", "china",
    # emotions / affect
    "fear", "anger", "disgust", "sadness", "hope", "joy", "pride", "empathy", "love", "threat",
    "safety", "strength", "unity", "patriotism",
    # visual scene words
    "flag", "crowd", "rally", "factory", "hospital", "children", "elderly", "soldier", "protest",
    "city", "countryside", "handshake",
]
DEVIATIONS = [
    "n ~ 20 ads from official candidate channels vs 15,427 labeled ads (2016-2020)",
    "logistic regression + leave-one-out CV instead of neural net + 80/20 split",
    "transcripts from faster-whisper large-v3 (YouTube auto-captions only as fallback); ads without speech get a constant empty-string text embedding",
    "R ads = Trump 2024 channel, D ads = Harris/Biden 2024 channels (same era; sponsor = candidate)",
    f"{len(CONCEPTS)} concepts instead of 160; sparsity tuned to ~8 active concepts, not matched reconstruction",
    "empty transcript chunks dropped from the text mean",
    "360p frames; exposure curve uses prefix means with the full-length classifier (LOO)",
]


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def extract_frames(vid, path):
    d = os.path.join(OUT, "frames", vid)
    os.makedirs(d, exist_ok=True)
    if not glob.glob(d + "/*.jpg"):
        run(["ffmpeg", "-v", "error", "-y", "-i", path, "-vf", f"fps=1/{STEP}", "-q:v", "3", d + "/f%04d.jpg"])
    return sorted(glob.glob(d + "/*.jpg"))


def extract_audio(vid, path):
    d = os.path.join(OUT, "audio")
    os.makedirs(d, exist_ok=True)
    wav = os.path.join(d, vid + ".wav")
    if not os.path.exists(wav):
        run(["ffmpeg", "-v", "error", "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000", wav])
    return wav


TS = re.compile(r"(\d+):(\d\d):(\d\d)\.(\d{3})")
WORD_TAG = re.compile(r"<(\d+:\d\d:\d\d\.\d{3})><c>\s*([^<]+?)\s*</c>")


def ts(s):
    h, m, sec, ms = TS.match(s).groups()
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000


def parse_vtt(path):
    """List of (t, word). Uses YouTube word-level timing tags; falls back to cue level."""
    words, cues, cur = [], [], None
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.rstrip("\n")
        if "-->" in line:
            a, b = line.split("-->")[:2]
            cur = (ts(a.strip()), ts(b.strip().split()[0]))
            continue
        if cur is None or not line.strip() or line.startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        if "<c>" in line:
            head = re.sub(r"<[^>]+>", "", re.split(r"<\d+:\d\d:\d\d\.\d{3}>", line)[0]).strip()
            if head:
                words.append((cur[0], head))
            words += [(ts(t), w) for t, w in WORD_TAG.findall(line)]
        else:
            txt = re.sub(r"<[^>]+>", "", line).strip()
            if txt:
                cues.append((cur[0], cur[1], txt))
    if words:
        return sorted(set(words))
    out, prev = [], None
    for s, e, txt in cues:  # cue fallback: drop repeated lines, spread words over the cue
        if txt == prev:
            continue
        prev, ws = txt, txt.split()
        out += [(s + (e - s) * i / max(len(ws), 1), w) for i, w in enumerate(ws)]
    return sorted(out)


def chunks(words, n):
    return [" ".join(w for t, w in words if k * STEP <= t < (k + 1) * STEP) for k in range(n)]


_WHISPER = None


def transcribe(wav):
    """Word-level transcript with faster-whisper large-v3 (closer to the paper's transcripts)."""
    global _WHISPER
    from faster_whisper import WhisperModel
    if _WHISPER is None:
        cuda = torch.cuda.is_available()
        _WHISPER = WhisperModel("large-v3", device="cuda" if cuda else "cpu",
                                compute_type="float16" if cuda else "int8")
    segs, _ = _WHISPER.transcribe(wav, language="en", word_timestamps=True, vad_filter=True)
    return [(w.start, w.word.strip()) for s in segs for w in (s.words or []) if w.word.strip()]


def load_clip():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
    return m, CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"), dev


@torch.no_grad()
def embed_images(m, p, dev, paths, bs=32):
    out = []
    for i in range(0, len(paths), bs):
        ims = [Image.open(x).convert("RGB") for x in paths[i:i + bs]]
        out.append(m.get_image_features(**p(images=ims, return_tensors="pt").to(dev)).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def embed_texts(m, p, dev, texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        t = p(text=texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=77).to(dev)
        out.append(m.get_text_features(**t).float().cpu().numpy())
    return np.concatenate(out)


def unit(x):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def loo_proba(X, y, C=1.0):
    """Leave-one-out predicted P(R)."""
    p = np.zeros(len(y))
    for i in range(len(y)):
        idx = np.arange(len(y)) != i
        clf = LogisticRegression(C=C, max_iter=2000).fit(X[idx], y[idx])
        p[i] = clf.predict_proba(X[i:i + 1])[0, 1]
    return p


def loo_proba_prefix(Xfull, Xprefix, y, C=1.0):
    """Train on full-length embeddings of the other ads, test on the prefix embedding of ad i."""
    p = np.zeros(len(y))
    for i in range(len(y)):
        idx = np.arange(len(y)) != i
        clf = LogisticRegression(C=C, max_iter=2000).fit(Xfull[idx], y[idx])
        p[i] = clf.predict_proba(Xprefix[i:i + 1])[0, 1]
    return p


def splice(Xc, C, n_active=8):
    """Non-negative sparse projection of centered embeddings onto a centered concept dictionary."""
    W = np.zeros((len(Xc), C.shape[1]))
    for i, x in enumerate(Xc):
        alpha = 0.01
        for _ in range(14):
            l = Lasso(alpha=alpha, positive=True, fit_intercept=False, max_iter=10000).fit(C, x)
            k = int((l.coef_ > 0).sum())
            if k > n_active * 1.5:
                alpha *= 1.6
            elif k < n_active / 1.5:
                alpha /= 1.6
            else:
                break
        W[i] = l.coef_
    return W


def main():
    rows = list(csv.DictReader(open(os.path.join(DATA, "manifest.csv"))))
    ids = [r["id"] for r in rows]
    y = np.array([1 if r["label"].strip().upper().startswith("R") else 0 for r in rows])
    m, p, dev = load_clip()
    print(f"device={dev}  ads={len(rows)}  R={int(y.sum())} D={int((1 - y).sum())}")

    img_ad, txt_ad, img_seq, txt_seq, n_words = [], [], {}, {}, []
    for r in rows:
        vid, vpath = r["id"], os.path.join(DATA, r["video"])
        frames = extract_frames(vid, vpath)
        wav = extract_audio(vid, vpath)
        src = "whisper"
        try:
            words = transcribe(wav)
        except Exception as e:  # fallback: YouTube captions if whisper is unavailable
            src, subs = "vtt", (r.get("subs") or "").strip()
            words = parse_vtt(os.path.join(DATA, subs)) if subs else []
            print(f"  whisper failed ({e!r}); using captions: {bool(subs)}")
        ch = chunks(words, len(frames))
        n_words.append(len(words))
        fi = unit(embed_images(m, p, dev, frames))
        keep = [k for k, c in enumerate(ch) if c.strip()]
        texts = [ch[k] for k in keep] if keep else [""]  # no speech: constant empty-string embedding
        ti = unit(embed_texts(m, p, dev, texts))
        keep = keep or [0]
        img_seq[vid], txt_seq[vid] = fi, (ti, keep)
        img_ad.append(fi.mean(0))
        txt_ad.append(ti.mean(0))
        print(f"  {vid}: {len(frames)} frames, {len(words)} words ({src}), {len(keep)} chunks")
    img_ad, txt_ad = unit(np.array(img_ad)), unit(np.array(txt_ad))

    # modality gap fix (Sec. 2.2): unit sphere -> subtract modality mean -> renormalize
    mu_img, mu_txt = img_ad.mean(0), txt_ad.mean(0)
    Xi, Xt = unit(img_ad - mu_img), unit(txt_ad - mu_txt)
    gap_before = float(np.linalg.norm(mu_img - mu_txt))
    gap_after = float(np.linalg.norm(Xi.mean(0) - Xt.mean(0)))

    # Sec. 2.3: per-modality classifier + stacked model (LOO)
    pi, pt = loo_proba(Xi, y), loo_proba(Xt, y)
    ps = loo_proba(np.c_[pi, pt], y, C=10.0)
    auc = {k: float(roc_auc_score(y, v)) for k, v in [("image", pi), ("text", pt), ("stacked", ps)]}
    corr_it = float(np.corrcoef(pi, pt)[0, 1])

    # Sec. 3: exposure curve, prefix of k seconds
    expo = {}
    for k in EXPOSURES:
        nk = max(1, int(round(k / STEP)))
        Pi = unit(unit(np.array([img_seq[v][:nk].mean(0) for v in ids])) - mu_img)
        Pt_raw = []
        for v in ids:
            ti, keep = txt_seq[v]
            sel = [j for j, kk in enumerate(keep) if kk < nk]
            Pt_raw.append(ti[sel].mean(0) if sel else np.zeros(Xi.shape[1]))
        Pt = unit(unit(np.array(Pt_raw)) - mu_txt)
        expo[k] = {"image": float(roc_auc_score(y, loo_proba_prefix(Xi, Pi, y))),
                   "text": float(roc_auc_score(y, loo_proba_prefix(Xt, Pt, y)))}

    # Sec. 4.1: SpLICE-lite; concept words centered with the text mean (sigma operation)
    Cw = unit(unit(embed_texts(m, p, dev, CONCEPTS)) - mu_txt).T  # (512, K)
    Wi, Wt = splice(Xi, Cw), splice(Xt, Cw)
    rec = {"image": float(np.mean([np.dot(x, unit(Cw @ w)) for x, w in zip(Xi, Wi)])),
           "text": float(np.mean([np.dot(x, unit(Cw @ w)) for x, w in zip(Xt, Wt)]))}

    def top_diff(W, k=8):
        d = W[y == 1].mean(0) - W[y == 0].mean(0)
        o = np.argsort(d)
        return {"R_minus_D": [(CONCEPTS[j], round(float(d[j]), 4)) for j in o[::-1][:k] if d[j] > 0],
                "D_minus_R": [(CONCEPTS[j], round(float(-d[j]), 4)) for j in o[:k] if d[j] < 0]}

    res = {"n": len(rows), "n_R": int(y.sum()), "auc_loo": auc, "corr_image_text_pred": corr_it,
           "modality_gap_norm": {"before": gap_before, "after": gap_after},
           "exposure_auc": expo, "splice_reconstruction_cos": rec,
           "concepts_image": top_diff(Wi), "concepts_text": top_diff(Wt),
           "mean_active_concepts": {"image": float((Wi > 0).sum(1).mean()), "text": float((Wt > 0).sum(1).mean())},
           "deviations": DEVIATIONS}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    np.savez(os.path.join(OUT, "embeddings.npz"), ids=np.array(ids), y=y, img=img_ad, txt=txt_ad,
             mu_img=mu_img, mu_txt=mu_txt, concepts=np.array(CONCEPTS), Cw=Cw, Wi=Wi, Wt=Wt)
    print(json.dumps({k: v for k, v in res.items() if k != "deviations"}, indent=1))


if __name__ == "__main__":
    main()
