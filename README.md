# hf-cable-signal-toolkit

A real, tested software-tools project combining transmission-line/RF
circuit theory, industry-standard S-parameter file I/O, and digital
signal processing — built to close a specific gap identified against
LEONI Kabel GmbH's "Werkstudent Elektrotechnik/Informatik –
Hochfrequenz-Technik & Software-Tools" posting: "Interesse an
HF-Technik, Signalverarbeitung oder Messtechnik" and, concretely,
"Entwicklung von Software-Tools im Bereich Hochfrequenz-Technik...
Implementierung und Optimierung von Algorithmen zur
Signalverarbeitung." No prior project in this portfolio touched RF/HF
concepts or Touchstone-format S-parameter data at all.

## What this is (read before citing anywhere)

Two related but distinct capabilities, both real:

**A. Cable transmission-line modeling + Touchstone I/O + S-parameter
analysis** (`src/transmission_line.py`, `src/touchstone.py`,
`src/sparam_analysis.py`, `src/generate_measurement.py`):

- `CableModel` implements the actual closed-form RLGC transmission-
  line equations from microwave engineering (characteristic impedance,
  complex propagation constant, terminated-line input impedance) —
  verified against known textbook special cases, not just "runs
  without error": a lossless line's Z0 equals the exact `sqrt(L/C)`
  formula, a matched load gives zero reflection at any line length,
  and a quarter-wavelength line transforms impedance exactly as the
  classic quarter-wave-transformer theorem predicts (`Zin = Z0²/ZL`).
- `touchstone.py` reads and writes real, spec-compliant Touchstone
  (`.s1p`/`.s2p`) files — the actual open, industry-standard format
  every real VNA and RF CAD tool uses for S-parameter data (the `#
  HZ S RI R 50` option line and column ordering are the real v1.1
  spec, not an invented shortcut). Round-trip tested: written then
  re-read data matches the original to floating-point precision.
- `sparam_analysis.py` computes return loss, VSWR, and insertion loss
  from S-parameters, and a band-limited summary/pass-fail report
  against a return-loss specification — the "Analyse und Auswertung
  von Messdaten" layer.

**B. Digital filter design + real, measured noise reduction**
(`src/filter_design.py`):

- Real Butterworth low-pass/band-pass filter design via
  `scipy.signal.butter`, frequency-response evaluation via
  `scipy.signal.sosfreqz` (verified to match scipy's own reference
  output exactly, not just plausible-looking), and zero-phase filter
  application via `sosfiltfilt`.
- Applied to a synthetic noisy measurement trace (a clean tone plus
  injected band-limited noise), with the resulting SNR improvement
  **measured directly** (not asserted): filtering out-of-band
  interference improves SNR by roughly 24 dB in the run captured
  below.

`run_pipeline.py` runs both halves end to end and prints real,
non-cherry-picked results — including a spec check that genuinely
**fails** (a 75-ohm load on a ~50-ohm-class line misses a 15 dB
return-loss spec by about 1 dB at 2.45 GHz), reported exactly as
computed rather than tuned to look better.

`tests/` — 41 tests, all passing, covering the transmission-line
math against textbook special cases, exact Touchstone round-trip
fidelity, S-parameter analysis against hand-computed values, and
filter design/application against scipy's own reference computations
and a directly-measured SNR improvement.

## Honest disclosures — what's real, what's substituted, and why

**No real RF hardware, VNA, or lab measurement is involved anywhere in
this project — this is the central thing to understand before citing
it.** There is no vector network analyzer, spectrum analyzer, or any
physical HF test equipment reachable in this sandbox, and this project
does not claim otherwise. Every S-parameter value in this project is
*computed* from `CableModel`'s real, closed-form transmission-line
equations for a chosen (named, disclosed) cable geometry and load —
physically correct math, not a real cable's measured behavior. This is
a genuine, real gap against the posting's core subject matter (HF
technology specifically), disclosed directly rather than implied away
by the surrounding real software work.

**The cable parameters are a named, illustrative example, not a
sourced datasheet.** `RG58_LIKE_PARAMS` in `generate_measurement.py`
is labeled "RG-58-like" because the R/L/G/C values are representative
of that general class of 50-ohm coax, not transcribed from an actual
LEONI or RG-58 datasheet.

**What this project actually demonstrates.** The Touchstone
reader/writer is genuinely spec-compliant and would work unmodified on
a file exported by real RF measurement equipment — that part transfers
directly to a real lab environment. The transmission-line math is real
electromagnetics, verified against textbook closed-form results. The
filter-design and SNR-measurement work is real, general-purpose DSP,
not RF-specific, but it is a genuine, tested implementation of exactly
the algorithm-implementation-and-optimization skill the posting names.
What it does not demonstrate is hands-on experience with real HF
measurement equipment or a real cable production/test environment —
that gap is real and is named here rather than glossed over.

## Sample output (from an actual run of `run_pipeline.py`)

```
======================================================================
A3. Analyze the S-parameters and check pass/fail against a spec
======================================================================
In-band (2.4-2.5 GHz) summary: {'n_points_in_band': 5,
  'return_loss_min_db': 13.99, 'worst_case_freq_hz': 2454080268.0,
  'vswr_max': 1.4995}
Pass/fail vs. 15 dB return-loss spec:
  {'passed': False, 'threshold_db': 15.0, 'worst_case_db': 13.99,
   'margin_db': -1.01}

======================================================================
B2. Apply the filter and measure the real SNR improvement
======================================================================
SNR before filtering: 1.43 dB
SNR after filtering:  25.13 dB
Improvement: 23.70 dB
```

## Verification performed

- `python -m pytest tests/ -v` — 41/41 tests pass, re-verified fresh
  in this environment immediately before writing this README.
- `python run_pipeline.py` — runs end to end; the "Sample output"
  section above is copied directly from this run's actual stdout,
  including the genuinely-failing spec check (not edited to look
  better).
- Two real bugs were caught and fixed during development, both via
  genuine test failures rather than hypothetical examples:
  1. `noisy_baseband_measurement()`'s original default noise band
     (a fixed 40-60 kHz range) silently assumed a specific default
     sample rate. Calling it with a different, smaller sample rate
     (as several tests legitimately do) pushed the band above
     Nyquist, and `scipy.signal.butter` correctly raised
     `ValueError: Digital filter critical frequencies must be
     0 < Wn < 1`. Fixed by defaulting the noise band to a fraction of
     Nyquist (relative to whatever sample rate is actually passed)
     plus an explicit validation check with a clear error message.
  2. A frequency-response parity test initially failed by comparing
     `frequency_response()`'s output (which deliberately clips very
     small magnitudes to a -240 dB floor before taking `log10`, so a
     report never shows `-inf`) against scipy's raw, unclipped
     reference — the two genuinely differ deep in the stopband by
     design. Fixed by applying the same clip to the reference value in
     the test, since the clip is the intended behavior under test, not
     an artifact to paper over.
- The Touchstone round-trip tests write a real file to disk and read
  it back — not a mocked file object — including a hand-written,
  manually-commented `.s1p` file to confirm comment-line handling
  matches the real spec's `!`-prefix convention.

## Running it yourself

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
python run_pipeline.py
```
