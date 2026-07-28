# BrainGPT-mini

A from-scratch, nanoGPT-style decoder-only transformer pretrained self-supervised
(next-token prediction) on raw multi-channel EEG waveforms from PhysioNet EEGBCI,
then evaluated via a frozen linear probe on motor-imagery classification —
benchmarked against an existing supervised EEGNet baseline
(see `eegnet-motor-imagery` repo).

## Motivation

This project explores whether GPT-style self-supervised pretraining transfers
to EEG: can a transformer learn useful representations of brain signals purely
from next-token prediction, without ever seeing a label? It is **not** a claim
of a novel foundation model — it's a rigorous exploration project, evaluated
honestly against a strong supervised baseline I already built.

## Status

🚧 In progress. See `results/` for current ablations and probe accuracy.

## Repo structure

```
src/            importable modules: tokenizer, model, training loop, linear probe
notebooks/      exploration + result visualization only (imports from src/)
data/           PhysioNet EEGBCI (gitignored, download instructions below)
checkpoints/    trained model weights (gitignored)
results/        plots, tables, ablation results (tracked)
```

## Setup

```bash
pip install -r requirements.txt
```

Data: PhysioNet EEG Motor Movement/Imagery Dataset — download instructions TBD
(reusing loading pipeline from `eegnet-motor-imagery`).

## Design decisions (filled in as I go)

- **Tokenization scheme:** each subject's continuous recording concatenates 6 runs; patch tokenization can produce up to 5 boundary-straddling patches per subject out of ~[X] total (~0.1%), left uncorrected as negligible.
- **Model size:** TBD — layer count, head count, embedding dim.
- **Evaluation protocol:** LOSO cross-subject, matching EEGNet baseline for apples-to-apples comparison.

## Ablations planned

- [ ] Tokenization granularity (coarse vs. fine patches)
- [ ] Number of attention heads / layers vs. downstream probe accuracy
- [ ] Positional encoding on vs. off
- [ ] Pretraining data volume vs. downstream probe accuracy

## Results

TBD.

## Limitations

TBD — to be filled honestly as findings come in (expected: dataset is small
by transformer standards; unlikely to beat supervised EEGNet outright).
