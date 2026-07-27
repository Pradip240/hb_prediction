# common/ — shared CPU code

Single source of truth for the configuration and DSP used by every CPU stage that
touches the rPPG signal:

- `config.py` — ROI/landmark definitions, DSP bands, quality thresholds, and the
  window/gap knobs for `prepare_dataset`.
- `signal_processing.py` — region extraction (skin mask + landmark polygon → per-region
  RGB) and the rPPG DSP (detrend, bandpass, POS/CHROM, clean-window search).

**How stages consume it.** Every stage imports these with *bare imports*
(`import config`, `import signal_processing`), which requires the two files to sit in
the stage's working directory (`/app`) at run time. So each stage's Dockerfile copies
`common/` into `/app` (`COPY common/ ./`), and for live editing you bind-mount the two
files over that copy (see the root README). Nothing here imports anything stage-specific,
so it stays a leaf dependency.

Keeping this in one place (rather than a copy inside `signal_analysis/`) means changing a
band, threshold, or ROI once updates `signal_analysis`, `prepare_dataset`,
`hr_prediction`, and the experimental `train/` identically.