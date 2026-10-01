#!/usr/bin/env python3
"""Independent recomputation of every reported number from the caches (RTA Phase 4).
Deliberately does NOT import poc_omni / poc_clip. Shapley by permutation enumeration
(not by the weight formula), AUC by Mann-Whitney rank counting, prototypes by explicit loops.
Also computes a leakage-free conflict variant (prototype without the two source ads)."""
import csv, itertools, json, math, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "manifest.csv"))))
ids = [r["id"] for r in rows]
y = np.array([1 if r["label"].upper().startswith("R") else 0 for r in rows])
cache = json.load(open(os.path.join(ROOT, "out", "omni_emb.json")))
CONDS = ["V", "P", "T", "VP", "VT", "PT", "VPT", "A", "VA"]


def unit(x):
    x = np.asarray(x, float)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def auc(yv, s):  # Mann-Whitney with tie handling
    pos, neg = s[yv == 1], s[yv == 0]
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return gt / (len(pos) * len(neg))


def loo_logistic(X, yv):
    out = np.zeros(len(yv))
    for i in range(len(yv)):
        m = np.ones(len(yv), bool); m[i] = False
        out[i] = LogisticRegression(C=1.0, max_iter=5000).fit(X[m], yv[m]).predict_proba(X[i:i + 1])[0, 1]
    return out


def loo_proto(E, yv):
    out = np.zeros(len(yv))
    for i in range(len(yv)):
        m = np.ones(len(yv), bool); m[i] = False
        p = E[m & (yv == 1)].mean(0) - E[m & (yv == 0)].mean(0)
        out[i] = float(E[i] @ (p / np.linalg.norm(p)))
    return out


E = {c: unit([cache[f"{v}:{c}"] for v in ids]) for c in CONDS}
AR = unit([cache[k] for k in sorted(cache) if k.startswith("anchor:R:")])
AD = unit([cache[k] for k in sorted(cache) if k.startswith("anchor:D:")])

scores, aucs = {"anchor": {}, "proto": {}}, {"anchor": {}, "proto": {}, "logistic_loo": {}}
for c in CONDS:
    sa = (E[c] @ AR.T).mean(1) - (E[c] @ AD.T).mean(1)
    sp = loo_proto(E[c], y)
    scores["anchor"][c], scores["proto"][c] = sa, sp
    aucs["anchor"][c] = round(float(auc(y, sa)), 3)
    aucs["proto"][c] = round(float(auc(y, sp)), 3)
    aucs["logistic_loo"][c] = round(float(auc(y, loo_logistic(E[c], y))), 3)


def z(x):
    return (x - x.mean()) / (x.std() if x.std() > 1e-12 else 1.0)


def shapley_perm(v):  # v: dict of frozenset -> array; enumerate 3! orders
    players = ["V", "P", "T"]
    phi = {p: np.zeros(len(y)) for p in players}
    for order in itertools.permutations(players):
        seen = set()
        for p in order:
            before = v[frozenset(seen)]; seen.add(p); after = v[frozenset(seen)]
            phi[p] += (after - before) / 6.0
    return phi


shap = {}
for kind in ("anchor", "proto"):
    v = {frozenset(): np.zeros(len(y))}
    for c in ["V", "P", "T", "VP", "VT", "PT", "VPT"]:
        v[frozenset(c)] = z(scores[kind][c])
    phi = shapley_perm(v)
    mag = {p: float(np.abs(phi[p]).mean()) for p in phi}
    tot = sum(mag.values())
    shap[kind] = {"share": {p: round(mag[p] / tot, 3) for p in phi},
                  "auc_phi": {p: round(float(auc(y, phi[p])), 3) for p in phi},
                  "sum_identity_max_abs_err": float(np.abs(phi["V"] + phi["P"] + phi["T"] - v[frozenset("VPT")]).max())}

means = {c: E[c].mean(0) for c in CONDS}
gap = {f"{a}-{b}": round(float(np.linalg.norm(means[a] - means[b])), 3) for a, b in [("V", "T"), ("P", "T"), ("V", "P"), ("A", "P")]}

# conflict test: full-sample prototype (as reported) and leakage-free (prototype without ads i and j)
lab = dict(zip(ids, y))
conflict = {}
for name, cond in [("V_over_P", "VP"), ("V_over_T", "VT"), ("P_over_T", "PT")]:
    keys = [k for k in cache if k.startswith(f"swap:{name}:")]
    full = E[cond][y == 1].mean(0) - E[cond][y == 0].mean(0)
    w_full, w_loo = [], []
    for k in keys:
        _, _, i, j = k.split(":")
        e = unit(cache[k])
        w_full.append(int((1 if e @ full > 0 else 0) == lab[i]))
        m = np.array([v not in (i, j) for v in ids])
        p2 = E[cond][m & (y == 1)].mean(0) - E[cond][m & (y == 0)].mean(0)
        w_loo.append(int((1 if e @ p2 > 0 else 0) == lab[i]))
    conflict[name] = {"n": len(keys), "first_wins_full_proto": sum(w_full), "first_wins_proto_without_i_j": sum(w_loo)}

# Part A from embeddings.npz
z_ = np.load(os.path.join(ROOT, "out", "embeddings.npz"), allow_pickle=True)
img, txt, yA = unit(z_["img"]), unit(z_["txt"]), z_["y"]
mu_i, mu_t = img.mean(0), txt.mean(0)
Xi, Xt = unit(img - mu_i), unit(txt - mu_t)
pi, pt = loo_logistic(Xi, yA), loo_logistic(Xt, yA)
partA = {"gap_before": round(float(np.linalg.norm(mu_i - mu_t)), 3), "gap_after": round(float(np.linalg.norm(Xi.mean(0) - Xt.mean(0))), 3),
         "angle_before_deg": round(2 * math.degrees(math.asin(min(1, np.linalg.norm(mu_i - mu_t) / 2))), 1),
         "auc_img": round(float(auc(yA, pi)), 3), "auc_txt": round(float(auc(yA, pt)), 3),
         "corr_pred": round(float(np.corrcoef(pi, pt)[0, 1]), 3),
         "auc_stacked": round(float(auc(yA, loo_logistic(np.c_[pi, pt], yA))), 3)}

out = {"n": len(ids), "n_R": int(y.sum()), "auc": aucs, "shapley": shap, "gap": gap, "conflict": conflict, "partA": partA}
print(json.dumps(out, indent=1))

# diff against reported
rb = json.load(open(os.path.join(ROOT, "out", "results_B.json")))
ra = json.load(open(os.path.join(ROOT, "out", "results.json")))
diffs = []
for k in ("anchor", "proto", "logistic_loo"):
    for c in CONDS:
        d = abs(aucs[k][c] - rb["auc"][k][c])
        if d > 0.006: diffs.append(f"B auc {k} {c}: mine {aucs[k][c]} reported {rb['auc'][k][c]:.3f}")
for kind in ("anchor", "proto"):
    for p in "VPT":
        d = abs(shap[kind]["share"][p] - rb["shapley"][kind]["share_of_abs_contribution"][p])
        if d > 0.006: diffs.append(f"B share {kind} {p}: mine {shap[kind]['share'][p]} reported {rb['shapley'][kind]['share_of_abs_contribution'][p]:.3f}")
for k in gap:
    if abs(gap[k] - rb["modality_gap_norm"][k]) > 0.006: diffs.append(f"B gap {k}")
for k in conflict:
    if conflict[k]["first_wins_full_proto"] / conflict[k]["n"] != rb["conflict"][k]["first_source_wins"]: diffs.append(f"B conflict {k}")
for k, mine in (("image", partA["auc_img"]), ("text", partA["auc_txt"]), ("stacked", partA["auc_stacked"])):
    if abs(mine - ra["auc_loo"][k]) > 0.006: diffs.append(f"A auc {k}: mine {mine} reported {ra['auc_loo'][k]:.3f}")
if abs(partA["gap_before"] - ra["modality_gap_norm"]["before"]) > 0.006 or abs(partA["gap_after"] - ra["modality_gap_norm"]["after"]) > 0.006: diffs.append("A gap")
print("DIFFS vs reported:", diffs or "none (all within 0.006)")
