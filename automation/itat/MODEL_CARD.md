# ITAT captcha model — model card

The target is the 6-character alphanumeric captcha on `itat.gov.in/judicial/casestatus`.
`model.onnx` will be a six-head classifier (one head per character position, each
over the full alphabet in `tools/charset.py`).

> **Status: NOT yet trained — scaffold only.** The inference path
> (`automation/itat/captcha.py`) is in place and falls back to `ddddocr`, but the
> ITAT-specific CNN does not exist yet. Live captcha data was gathered on
> 2026-11-23 (~60 images at `/tmp/itat_live/raw`), pending labelling and a
> training run. Until the CNN is exported, expect unreliable reads.

## Files

| File | Purpose |
|---|---|
| `model.onnx` | Inference model used by `automation/itat/captcha.py` (**not yet exported**) |
| `tools/charset.py` | Alphabet, `CAPTCHA_LEN`, input size `W,H`, encode/decode, `normalise`, `is_valid_code` |
| `tools/model.py` | The network definition (PyTorch, six classification heads) |
| `tools/train.py` | Trainer + ONNX exporter |
| `tools/synth.py` | Synthetic generator matching the white/light-blue-scribble ITAT style |
| `tools/preprocess.py` | Input pipeline (must match `captcha.render4`) |
| `tools/harvest.py` | Throttled, duplicate/skip-aware live captcha harvester for training data |
| `tools/inspect_sample.py` | Reports observed charset + solver accuracy and writes a captcha montage |

## Input

`float32[1, 4, H, W]` — the captcha resized to `H×W` (`182×50` in
`charset.py`) with four channels:

1. inverted greyscale,
2. absolute dark mask (`gray < 45`),
3. relative dark mask (`gray < mean × 0.72`),
4. red mark mask (R>150, G<100, B<100).

## Output

Six `float32[1, N_CLASSES]` logits (`d0` … `d5`), one per character position;
softmax per head. `N_CLASSES` is the length of `charset.CHARSET`, currently
`0-9` + `A-Z`.

## Training data

* Synthetic images from `tools/synth.py` — the real ITAT style: white background,
  dense light-blue/cyan bezier scribbles, coloured speckle, dark slanted glyphs,
  a faint blue border, randomised fonts/sizes/rotation.
* Real captchas harvested from the anonymous captcha endpoint with
  `tools/harvest.py`, labelled by hand and/or high-agreement OCR consensus. No
  third-party captcha dataset and no client data were used.

## Measured accuracy (development machine, CPU)

* Stock `ddddocr` (with blue-removal renders + multi-variant voting) reads only
  **2 / 22 full 6-character codes, 0 confident**. Ambiguous pairs (`0/O`,
  `1/I/L`, `5/S`, `2/Z`) are all present, and the light-blue scribble defeats
  simple thresholding.
* **Conclusion: a CNN trained on the real ITAT style is required.** ddddocr alone
  is not viable (unlike the GST captcha).

Captcha styles change. When confidence drops, the solver refreshes rather than
submitting a wrong read; harvest new labels with `tools/harvest.py`.

## Workflow

1. `python tools/harvest.py 200 harvest` — gather throttled real captchas and
   auto-proposed labels.
2. `python tools/inspect_sample.py harvest` — review the observed charset,
   solver accuracy, and `harvest/montage.png`; correct labels by hand.
3. `python tools/train.py <epochs> <steps> <batch> --labels harvest/labels.csv --export`
   — trains on synthetic + real data and exports `automation/itat/model.onnx`.

## Licensing

The model and training tooling are part of this repository and covered by its
AGPL-3.0-or-later licence. `ddddocr` is MIT; `onnxruntime` is MIT. No code or
weights from unlicensed captcha repositories were used.
