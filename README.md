# Photon Simulation Corrections via Conditional Flow Matching

This repository contains the training and evaluation code for correcting CMS Monte Carlo photon shower shapes to match data, using Conditional Flow Matching (CFM). The corrections are derived from Z→μμγ events collected during Run 3 (2022–2024).

---

## Physics context

Photon shower shape variables simulated by CMS Monte Carlo do not perfectly match those measured in data. This project trains a neural ODE-based flow model to transport the MC distribution toward the data distribution, conditioned on per-photon kinematics. The model is trained on photons from Z→μμγ decays (where the photon is radiated off a muon), which provide a clean, unbiased sample for learning the correction.

---

## Repository structure

```
cfm_experiment.py       # main training + evaluation script
utils.py                # model definitions (MLP, FiLMNet), loss plotting,
                        # iso variable transforms, MVA score computation
customcfm/              # CFM library with OT coupling variants
10_submit.sub           # HTCondor job description
10_submit_shell.sh      # shell wrapper called by HTCondor
basicMamba.sh           # micromamba environment setup sourced by the shell wrapper
requirements.txt        # Python dependencies
setup.py / pyproject.toml
```

---

## Input features and conditions

**16 corrected shower shape variables** (e.g. r9, σiηiη, isolation sums, H/E, preshower energy).

**5 conditioning variables**: photon pT, supercluster η, φ, event ρ (energy density), and ΔR to the nearest muon.

Photon η is used to split events into two detector regions:
- **Barrel**: |η| < 1.442
- **Endcap**: |η| > 1.566

---

## How `cfm_experiment.py` works

### 1. Preprocessing (`preprocess_data`)
- Drops NaN rows.
- Optionally appends binary barrel/endcap flags to the condition vector.
- Filters to the selected detector region.
- Applies a smooth monotonic remap to isolation variables (`apply_scale_and_smooth`) to handle the spike at zero.
- Standardizes all features and conditions using MC mean and standard deviation.

### 2. Dataset splitting
With `area_of_interest = True` (default), the validation and test sets are restricted to **signal-like photons** (pT > 25 GeV, ΔR(γ, μ) > 0.4). Photons outside this region are moved from val/test into training, so the model sees all data but is only evaluated on the clean signal region.

A 68 / 14 / 18 % train / val / test split is used.

### 3. Model
Two architectures are available, selected via `--nn_method`:
- **MLP**: a plain feed-forward network. Time `t` and conditions are concatenated to the input.
- **FiLMNet**: uses Feature-wise Linear Modulation (FiLM) layers to condition the velocity field on the kinematic variables.

Both take as input the feature vector at time `t`, the interpolation time `t`, and the condition vector, and output the velocity `v(x, t, c)`.

### 4. CFM training
At each training step:
- An MC sample (base, `x0`) and a data sample (target, `x1`) are drawn.
- The selected OT coupling pairs them optimally (or randomly, depending on `--ot_method`).
- A random time `t ~ U(0,1)` is sampled and the interpolated point `x_t` and target velocity `u_t` are computed.
- The model predicts velocity `v_t` and the loss is `||v_t - u_t||²`, optionally weighted by per-event MC weights.
- Training runs for up to 1000 epochs with early stopping (patience 15) and ReduceLROnPlateau scheduling.

Available OT methods (`--ot_method`):
| Value | Description |
|---|---|
| `None` | Standard CFM, random pairing |
| `ExactOT` | Exact balanced optimal transport |
| `UOT` | Unbalanced OT |
| `Sommer` | Custom weighted OT variant |
| `WeightedOT` | Weighted balanced OT |

### 5. Inference
The best checkpoint is loaded and the ODE `dx/dt = v(x, t, c)` is integrated from `t=0` (MC) to `t=1` (data) using the `dopri5` adaptive solver (`torchdiffeq.odeint`). The corrected features are then inverse-transformed back to physical units.

### 6. Evaluation
- Per-variable PDF plots are produced (MC raw, MC corrected, data), using χ²/ndf as a goodness-of-fit measure.
- A photon MVA ID score is computed from the corrected variables and compared to raw MC and data.
- Correlation matrices are plotted for barrel and endcap separately.
- Per-run results are written to a `.txt` summary file.

### 7. Multi-seed evaluation
The entire training and evaluation is repeated for **6 seeds** (2020–2025). The final summary reports **median ± MAD** across seeds for all metrics, and writes LaTeX-formatted tables for use in the thesis.

---

## Running locally

```bash
python cfm_experiment.py \
  --nn_method MLP \
  --ot_method ExactOT \
  --region endcap \
  --depth 5 \
  --hidden_dim 2048 \
  --batch_size 256 \
  --binary_position False \
  --loss unweighted
```

**Note:** `cfm_experiment.py` imports `flow_plotting_general_script` from the parent directory at runtime. That script is not part of this repository and must be present one level above.

---

## Running on HTCondor

```bash
condor_submit 10_submit.sub
```

The submit file transfers `cfm_experiment.py`, `10_submit_shell.sh`, and `basicMamba.sh` to the worker node, activates the `zmmg_corrections` micromamba environment, and launches the script with the specified argument combination. Logs are written to `logs_10_2026/`.

---

## Dependencies

Core: `torch`, `torchdiffeq`, `numpy`, `pandas`, `awkward`, `hist`, `mplhep`, `matplotlib`, `xgboost`, `zuko`.

The `customcfm` package in this repository provides the CFM and OT coupling implementations.
