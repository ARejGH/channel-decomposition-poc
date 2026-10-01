# Channel Decomposition of Political Ads: a proof of concept

**Page:** https://arejgh.github.io/channel-decomposition-poc/

An independent student proof of concept, written on 25 September 2026 after session 1 of the ETH Zurich course
851-0762 (Computational Social Science with Images and Audio) and prepared for Andrea Ciccarone. It replicates
selected measurement steps of his working paper on a fresh sample of 22 ads and extends one of them with an omni
embedder that encodes image, filtered audio and text in one model.

## Why this is public

This copy is shared as a worked example of how far an agentic AI coding workflow can get, in a few hours, in
replicating selected elements of a recent publication: locating the paper, rebuilding its pipeline, running it on new
data, reviewing the result adversarially and recomputing every number independently. The page records the
substitutions, the defects found by the reviews and what a rigorous version would need.

It is **not** a review or evaluation of the original paper, it has **not** been peer-reviewed, and it is **not**
affiliated with or endorsed by the paper's author or by ETH Zurich. At n = 22 every number is descriptive.

## Original paper

Ciccarone, Andrea (2026). *An Image is Worth 1000 Words? Partisanship and Attention in Videos.* Working paper,
version of 16 June 2026. https://andreaciccarone.github.io/Video_Partisanship_draft.pdf

## Contents

| Path | What it is |
|---|---|
| `index.html` | The page (static HTML, no libraries; the code is also embedded in appendix A.4 to A.10) |
| `code/poc_clip.py` | Section 2: CLIP mini-replication of the paper's Sections 2.1 to 2.3, 3 and 4.1 |
| `code/prep_b.py` | Channel preparation: muted video, low-passed audio (prosody), transcript |
| `code/poc_omni.py` | Section 1: omni-embedder channel subsets, Shapley decomposition, conflict test |
| `code/verify_independent.py` | Independent recomputation of every reported number from the caches |
| `code/data/manifest.csv` | The 22 ads: YouTube ids and party labels |
| `code/out/results.json`, `code/out/results_B.json` | Outputs as produced |

## Reproducing

- **Data are not included.** The manifest lists YouTube video ids of official 2024 campaign channels (Trump; Harris
  and Biden); videos and caches (`out/omni_emb.json`, `out/embeddings.npz`) are not redistributed. Obtain the videos
  only in ways permitted by YouTube's terms and place them in `code/data/` under the file names in the manifest.
- **Environment as run:** Python 3, torch 2.8 (CUDA 12.8), transformers 4.57.3, scikit-learn 1.8, faster-whisper
  1.2.1, qwen-omni-utils, ffmpeg; one RTX 5080 (16 GB).
- **`model_local`:** `poc_omni.py` loads the processor from a local copy of the model snapshot (865db1bb) without
  the `additional_chat_templates` folder, a workaround for a loader error in transformers 4.57. The model card pins
  transformers 4.51.3; with that version the Hub path should work directly (not tested).
- **Embedding:** mean of the last hidden states over the attention mask, L2-normalised, `passage: ` prefix for
  documents, as in the model card's transformers example. Anchors use a `query: ` prefix.
- The model loads with `trust_remote_code=True`; review the model repository before running it.

## How it was built

Built in zen-system, the author's own agent harness, in development since 2025.
The author set the task and approved each installation; the agent workflow
located the paper, rebuilt the pipeline, ran it, reviewed it and wrote the page.

Models used in the replication: Claude Fable 5.1 (Anthropic); GPT-6 Astra (OpenAI).

## Known limitations

Listed on the page itself (sections 5 to 8): the identity of the candidate leaks into every channel; token-count
weighting may drive the visual dominance; the prosody channel is an operationalisation; small-sample estimators; the
conflict test's prototypes include the two donor ads in the reported variant (a leakage-free variant is in
`verify_independent.py` and on the page). The defects found by the reviews are left in the code as run.

## References

- Ciccarone, A. (2026). An Image is Worth 1000 Words? Partisanship and Attention in Videos. Working paper.
  https://andreaciccarone.github.io/Video_Partisanship_draft.pdf
- Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP).
  arXiv:2103.00020
- Liang, W. et al. (2022). Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation
  Learning. arXiv:2203.02053
- Bhalla, U. et al. (2024). Interpreting CLIP with Sparse Linear Concept Embeddings (SpLiCE). arXiv:2402.10376
- Radford, A. et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision (Whisper). arXiv:2212.04356
- Xu, J. et al. (2025). Qwen2.5-Omni Technical Report. arXiv:2503.20215
- Parcalabescu, L. and Frank, A. (2023). MM-SHAP: A Performance-agnostic Metric for Measuring Multimodal
  Contributions in Vision and Language Models and Tasks. ACL 2023. arXiv:2212.08158
- Wu et al. (2025). When Language Overrules: Revealing Text Dominance in Multimodal Large Language Models.
  arXiv:2508.10552
- Rogers, P. L., Scherer, K. R. and Rosenthal, R. (1971). Content filtering human speech: a simple electronic system.
  Behavior Research Methods & Instrumentation, 3, 16-18. https://doi.org/10.3758/BF03208115
- Models: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32),
  [nvidia/omni-embed-nemotron-3b](https://huggingface.co/nvidia/omni-embed-nemotron-3b) (NVIDIA OneWay
  Noncommercial License and Qwen Research License: research use only; weights are not redistributed here),
  Systran/faster-whisper-large-v3.

## Author and licence

Aleksander Rej (University of Warsaw; exchange student at the University of Zurich, autumn 2026).

Code: MIT licence (`LICENSE`). Text of the page: CC BY 4.0. How to cite: see `CITATION.cff`.
