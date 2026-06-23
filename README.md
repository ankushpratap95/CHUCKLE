# CHUCKLE: Crowdsourced Human Understanding Curriculum for Knowledge Led Emotion Recognition

Official code for our Interspeech 2026 paper:

> **CHUCKLE — When Humans Teach AI to Learn Emotions the Easy Way**
> Ankush Pratap Singh¹, Houwei Cao¹, Yong Liu²
> ¹ Dept. of Computer Science, New York Institute of Technology
> ² Electrical and Computer Engineering Dept., New York University
> Interspeech 2026

<!-- TODO: add paper/proceedings link once available, e.g. ISCA Archive or arXiv -->

## Abstract

Curriculum learning (CL) structures training from simple to complex
samples, facilitating progressive learning. Existing CL approaches for
emotion recognition often rely on heuristic, data-driven, or model-based
definitions of sample difficulty, neglecting the difficulty for *human*
perception — a critical factor in subjective tasks like emotion recognition.
We propose **CHUCKLE** (Crowdsourced Human Understanding Curriculum for
Knowledge Led Emotion Recognition), a perception-driven CL framework that
leverages annotator agreement and alignment in crowd-sourced datasets to
define sample difficulty, under the assumption that clips challenging for
humans are similarly hard for neural networks. Experimental results show
CHUCKLE improves LSTM and Transformer performance over non-curriculum
baselines in both subject-dependent and subject-independent settings, while
reducing the number of gradient updates needed to reach comparable
performance.

## Repository Structure

```
.
├── CHUCKLE_Curriculum.ipynb   # Proposed method: perception-driven curricula (score-based + rule-based)
├── Non-Curriculum.ipynb       # Baseline: standard training on the full dataset, no staging
├── Random.ipynb               # Baseline: random curriculum (same staged schedule, random ordering)
└── README.md
```

Each notebook follows the same layout:

| Section                | Description                                                           |
| ----------------------- | ------------------------------------------------------------------------ |
| LSTM Based              | 2-layer BiLSTM classifier experiments                                  |
| &nbsp;&nbsp;↳ Subject Dependent   | 80/20 train/test split per subject & emotion (same speakers in both) |
| &nbsp;&nbsp;↳ Subject Independent | ~80/20 split by *subject* — test speakers unseen during training   |
| Transformer Based       | 2-layer Transformer encoder classifier experiments (same SD/SI structure) |

Within each subsection there are paired **Training** and **Testing** cells.

## Method Summary

CHUCKLE defines sample difficulty from annotator behavior on **CREMA-D** [Cao
et al., 2014], where each clip has one *intended* emotion label (set by the
actor) and multiple *perceived* labels (8–12 ratings per clip from
crowd-sourced annotators). Two complementary families of curricula are
proposed:

**Score-based** (continuous difficulty score → equally-sized quartile bins:
`Easy`, `Borderline Easy`, `Borderline Tough`, `Tough`):
- `intended_emotion_score` — fraction of annotators agreeing with the actor's
  intended emotion (higher = easier).
- `entropy_score` — Shannon entropy over the annotator label distribution
  (higher entropy = harder).

**Rule-based** (samples categorized into 4 groups, then ordered into the same
4 bins by 3 different rules):
- `Clear Match` — clear majority of perceived labels, and it matches the intended label.
- `Clear Mismatch` — clear majority of perceived labels, but it does **not** match the intended label.
- `Ambiguous Match` — no dominant majority, but multiple annotations are consistent with the intended label.
- `Ambiguous Mismatch` — no dominant majority, and the intended label receives no consistent support.

  | Curriculum | Bin ordering (Easy → Tough) | Rationale |
  |---|---|---|
  | `intended_perceived_agreement_1` | Clear Match → Clear Mismatch → Ambiguous Match → Ambiguous Mismatch | Prioritizes agreement *strength* over alignment with the intended label |
  | `intended_perceived_agreement_2` | Clear Match → Ambiguous Match → Ambiguous Mismatch → Clear Mismatch | Prioritizes alignment with the intended label; "confidently incorrect" (Clear Mismatch) is hardest |
  | `intended_perceived_agreement_3` | Clear Match → Ambiguous Match → Clear Mismatch → Ambiguous Mismatch | Compromise between agreement strength and alignment |

  `Intended-Perceived Agreement 1` was the best-performing curriculum in the paper.

> These bins are expected to be **precomputed** into the difficulty CSV
> (`curriculum_dataset_results_binned.csv`) before running these notebooks —
> the bin-assignment logic above is implemented in a separate
> preprocessing step that is not part of this repo's three training/eval
> notebooks. <!-- TODO: link/add the bin-generation script if you want this repo to be fully self-contained -->

**Training schedule (all three notebooks):** 4 cumulative stages, each
adding the next-hardest bin on top of the previous ones:
1. `Easy`
2. `Easy + Borderline Easy`
3. `Easy + Borderline Easy + Borderline Tough`
4. `Easy + Borderline Easy + Borderline Tough + Tough` (= full dataset)

- **CHUCKLE_Curriculum.ipynb** — stages follow one of the difficulty bins above.
- **Random.ipynb** — same 25/50/75/100% cumulative schedule, but stage membership comes from a random shuffle of the training set, isolating the effect of *staged* training from *difficulty-ordered* training.
- **Non-Curriculum.ipynb** — single stage on 100% of the data for the full epoch budget (no staging at all).

A fresh optimizer + `CosineAnnealingLR` schedule (`5×10⁻⁴ → 5×10⁻⁵`) is
initialized at the start of each stage. Model checkpoints, the fitted
`StandardScaler`, optimizer/scheduler state, and per-epoch loss are saved at
the end of every stage for later evaluation.

## Models & Training Details

| Setting | LSTM | Transformer |
|---|---|---|
| Architecture | 2-layer BiLSTM, 128-dim hidden | 2-layer Transformer encoder, 4 heads, 128-dim |
| Epochs | 200 (50 per stage) | 400 (100 per stage) |
| Optimizer | Adam | Adam |
| LR Scheduler | CosineAnnealingLR, decay 5×10⁻⁴ → 5×10⁻⁵, reset each stage | same |
| Batch size | 8 | 8 |
| GPU used in paper | RTX A6000 | RTX A6000 |
| Trials | 10 trials per setting, mean macro accuracy, paired one-sided t-test (p < 0.05) | same |

**Features:** Frame-level [HuBERT](https://arxiv.org/abs/2106.07449)-XLarge
representations (final hidden layer, no fine-tuning), extracted via Hugging
Face Transformers from 16 kHz audio. Each frame embedding is 1280-dim;
sequences are variable-length depending on clip duration. Expected as `.npy`
files under `hubert_features/`.

## Data

The code expects:

1. **CREMA-D** [Cao et al., 2014] — 7,442 clips, 91 actors, 6 emotions
   (`ANG`, `DIS`, `FEA`, `HAP`, `NEU`, `SAD`), 12 sentences, with 2,443 raters
   providing 8–12 perceived-emotion ratings per clip across audio, video,
   and multimodal conditions. Not included in this repo — see
   [the CREMA-D project](https://github.com/CheyneyComputerScience/CREMA-D)
   for access.

2. **A difficulty CSV** (`curriculum_dataset_results_binned.csv`) with:
   - `fileName` — utterance identifier (used to locate features and derive subject ID / emotion label)
   - `modality` — filtered to `"audio"` in these experiments
   - One bin column per curriculum, e.g. `bin_intended_emotion_score`, `bin_entropy_score`, `bin_intended_perceived_agreement_{1,2,3}`, valued `{Easy, Borderline Easy, Borderline Tough, Tough}`

3. **Pre-extracted HuBERT-XLarge features** as `.npy` files in `hubert_features/`, named to match each utterance's filename.

4. **Subject-organized folders** (used as an existence check linking the CSV to feature files):
   ```
   subject_ordered_features/Training/<subject_id>/Audio/<filename>.csv
   subject_ordered_features/Training_SI/<subject_id>/Audio/<filename>.csv
   subject_ordered_features/Testing/<subject_id>/Audio/<filename>.csv
   subject_ordered_features/Testing_SI/<subject_id>/Audio/<filename>.csv
   ```
   (`_SI` = subject-independent split: in the paper, train/test were split
   80/20 per-subject-and-emotion for subject-dependent, vs. ~80/20 by
   *subject* for subject-independent.)

## Setup

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

### Requirements
```
torch
pandas
numpy
scikit-learn
matplotlib
seaborn
```

A GPU is recommended (paper used an RTX A6000) but not required — `device`
falls back to CPU automatically.

## Reproducibility

Each run sets `PYTHONHASHSEED`, Python/NumPy/PyTorch seeds, and enables
deterministic cuDNN/CUDA algorithms via `torch.use_deterministic_algorithms`.
The seed used in these notebooks is `random_seed = 1737200047`; pass a
different `seed_value` to `run_trial()` to reproduce the paper's 10-trial
averages.

## Usage

Open the relevant notebook and run the cells for the model/split you want:

- **Train** the curriculum model (e.g. LSTM, subject-dependent): run the
  *"LSTM Based → Subject dependent - Training"* cell in
  `CHUCKLE_Curriculum.ipynb`. Set `difficulty_score` to one of
  `intended_emotion_score`, `entropy_score`, `intended_perceived_agreement_1`,
  `intended_perceived_agreement_2`, or `intended_perceived_agreement_3`.
- **Evaluate** the same model across all 4 stages and difficulty bins on
  held-out test data: run the matching *"...Testing"* cell directly below it.
- Swap in `Non-Curriculum.ipynb` or `Random.ipynb` for the respective
  baselines. The Testing cells report overall/macro/balanced accuracy,
  per-bin breakdowns, confusion matrices, and loss curves.

Checkpoints are written to:
```
saved_weights/{subject_dependent,subject_independent}/<scenario>/{lstm,transformer}/random_seed_<seed>/stage_<n>/model_checkpoint.pth
```

## Citation

```bibtex
@misc{singh2026chucklehumansteach,
      title={CHUCKLE -- When Humans Teach AI To Learn Emotions The Easy Way}, 
      author={Ankush Pratap Singh and Houwei Cao and Yong Liu},
      year={2026},
      eprint={2510.09382},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.09382}, 
}
```

Please also cite the CREMA-D dataset if you use it:

```bibtex
@article{cao2014crema,
  title   = {CREMA-D: Crowd-sourced emotional multimodal actors dataset},
  author  = {Cao, Houwei and Cooper, David G. and Keutmann, Michael K. and Gur, Ruben C. and Nenkova, Ani and Verma, Ragini},
  journal = {IEEE Transactions on Affective Computing},
  volume  = {5},
  number  = {4},
  pages   = {377--390},
  year    = {2014}
}
```

## Contact
For more information, email at ax2047@nyu.edu or asing213@nyit.edu
