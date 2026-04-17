import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ['PYTHONHASHSEED'] = str(42)

import random
import sys
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset, RandomSampler
from torchdiffeq import odeint
import ot as pot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import mplhep
import hist
import awkward as ak

from customcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    BinaryConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    UnbalancedOptimalTransportConditionalFlowMatcher,
    SommerConditionalFlowMatcher,
    WeightedOptimalTransportConditionalFlowMatcher,
)
plt.style.use([mplhep.style.CMS])

# Import plot utilities from sibling directory
_original_sys_path = sys.path.copy()
try:
    _parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, _parent_dir)
    import flow_plotting_general_script as plot_utils
finally:
    sys.path = _original_sys_path

from utils import (
    undo_apply_scale_and_smooth,
    apply_scale_and_smooth,
    FiLMNet,
    MLP,
    lossplot_train,
    add_corr_photonid_mva_run3_zmmg,
    add_mc_photonid_mva_run3_zmmg,
    add_data_photonid_mva_run3_zmmg,
    plot_correlation_matrices,
)


# ---------------------------------------------------------------------------
# Physics variable name lists
# ---------------------------------------------------------------------------

ISOLATION_VARIABLES = [
    "photon_hoe",
    "photon_hcalPFClusterIso",
    "photon_ecalPFClusterIso",
    "photon_trkSumPtHollowConeDR03",
    "photon_trkSumPtSolidConeDR04",
    "photon_pfChargedIsoWorstVtx",
    "photon_pfChargedIso",
    "photon_esEffSigmaRR",
    "photon_esEnergyOverRawE",
]

VARIABLES_CMS_NAMES = [
    "photon_r9",
    "photon_sieie",
    "photon_etaWidth",
    "photon_phiWidth",
    "photon_sieip",
    "photon_s4",
    "photon_energyErr",
    "photon_hoe",
    "photon_hcalPFClusterIso",
    "photon_ecalPFClusterIso",
    "photon_trkSumPtHollowConeDR03",
    "photon_trkSumPtSolidConeDR04",
    "photon_pfChargedIsoWorstVtx",
    "photon_pfChargedIso",
    "photon_esEffSigmaRR",
    "photon_esEnergyOverRawE",
]

VARIABLES_MC_NAMES = [
    "photon_raw_r9",
    "photon_raw_sieie",
    "photon_raw_etaWidth",
    "photon_raw_phiWidth",
    "photon_raw_sieip",
    "photon_raw_s4",
    "photon_raw_energyErr",
    "photon_raw_hoe",
    "photon_raw_hcalPFClusterIso",
    "photon_raw_ecalPFClusterIso",
    "photon_raw_trkSumPtHollowConeDR03",
    "photon_raw_trkSumPtSolidConeDR04",
    "photon_raw_pfChargedIsoWorstVtx",
    "photon_raw_pfChargedIso",
    "photon_raw_esEffSigmaRR",
    "photon_raw_esEnergyOverRawE",
]

CONDITIONS_NAMES = [
    "photon_pt",
    "photon_ScEta",
    "photon_phi",
    "Rho_fixedGridRhoAll",
    "photon_muon_near_dR",
]


# ---------------------------------------------------------------------------
# Statistics utilities
# ---------------------------------------------------------------------------

def safe_mean(lst):
    filtered = [x for x in lst if x is not None]
    return float(np.median(filtered)) if filtered else None


def safe_std(lst):
    filtered = [x for x in lst if x is not None]
    if not filtered:
        return None
    x = np.asanyarray(filtered)
    return float(np.median(np.abs(x - np.median(x))))


def safe_arithmetic_std_propagation(lst):
    filtered = [x for x in lst if x is not None]
    return np.sqrt(np.sum(np.array(filtered) ** 2)) / len(filtered) if filtered else None


def safe_argmin(lst):
    filtered = [x for x in lst if x is not None]
    return int(np.argmin(filtered)) if filtered else None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train conditional flow model")
    parser.add_argument("--nn_method",        type=str,   default="MLP",     help="Neural network type")
    parser.add_argument("--ot_method",        type=str,   default="None",    help="OT method to use")
    parser.add_argument("--sigma",            type=float, default=0.0,       help="Sigma for CFM")
    parser.add_argument("--region",           type=str,   default="endcap",  help="Region to correct")
    parser.add_argument("--depth",            type=int,   default=7,         help="Number of layers")
    parser.add_argument("--hidden_dim",       type=int,   default=512,       help="Hidden dimensions")
    parser.add_argument("--batch_size",       type=int,   default=2048,      help="Batch size")
    parser.add_argument("--binary_position",  type=str,   default="False",   help="Use binary conditions")
    parser.add_argument("--loss",             type=str,   default="weighted")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Model & CFM construction
# ---------------------------------------------------------------------------

def build_model(nn_method, feature_dim, cond_dim, hidden_dim, depth, device):
    if nn_method == "MLP":
        model = MLP(input_dim=feature_dim, cond_dim=cond_dim,
                    hidden_dim=hidden_dim, depth=depth, use_time=True)
    elif nn_method == "FiLM":
        model = FiLMNet(input_dim=feature_dim, cond_dim=cond_dim,
                        hidden_dim=hidden_dim, depth=depth, use_time=True)
    else:
        raise ValueError(f"Unknown nn_method: {nn_method}")
    return model.to(device)


def build_cfm(ot_method, sigma):
    cfm_map = {
        "None":                ConditionalFlowMatcher,
        "None_binary_unmatch": BinaryConditionalFlowMatcher,
        "UOT":                 UnbalancedOptimalTransportConditionalFlowMatcher,
        "Sommer":              SommerConditionalFlowMatcher,
        "ExactOT":             ExactOptimalTransportConditionalFlowMatcher,
        "WeightedOT":          WeightedOptimalTransportConditionalFlowMatcher,
    }
    if ot_method not in cfm_map:
        raise ValueError(f"Unknown ot_method: {ot_method}")
    return cfm_map[ot_method](sigma=sigma)


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_data(year):
    base = "/home/home1/institut_3a/tsommer/bsc-sommer-project/conditioned_reg_cfm/data_read_and_store"
    data_df = pd.read_pickle(f"{base}/data_df_{year}.pkl")
    mc_df   = pd.read_pickle(f"{base}/mc_df_{year}.pkl")
    return data_df, mc_df


def preprocess_data(
    data_df, mc_df,
    variables_cms_names, variables_mc_names, conditions_names,
    isolation_variables, binary_position, region,
):
    """
    Drop NaNs, apply binary flags, filter region, log-transform,
    iso scale/smooth, and standardize.  Returns preprocessed arrays
    and the stats needed to invert the standardization later.
    """
    data_df = data_df.dropna(subset=variables_cms_names + conditions_names).copy(deep=True)
    mc_df   = mc_df.dropna(subset=variables_mc_names   + conditions_names).copy(deep=True)

    if binary_position:
        for df in [data_df, mc_df]:
            df["is_barrel"] = (np.abs(df["photon_ScEta"].values) < 1.442).astype(float)
            df["is_endcap"] = (np.abs(df["photon_ScEta"].values) > 1.566).astype(float)
        conditions_names = conditions_names + ["is_barrel", "is_endcap"]

    def _region_mask(df):
        eta = np.abs(df["photon_ScEta"].values)
        if region == "barrel":
            return eta < 1.442
        elif region == "endcap":
            return eta > 1.566
        return np.ones(len(df), dtype=bool)

    data_df = data_df.iloc[_region_mask(data_df)].copy(deep=True)
    mc_df   = mc_df.iloc[_region_mask(mc_df)].copy(deep=True)
    print(data_df.shape, mc_df.shape)

    data_inputs     = data_df[variables_cms_names].values
    data_conditions = data_df[conditions_names].values
    mc_inputs       = mc_df[variables_mc_names].values
    mc_conditions   = mc_df[conditions_names].values

    # NOTE: variables_cms_names is a list; `list == str` evaluates to False,
    # making this index a no-op — preserving identical behaviour to 9.py.
    data_inputs[variables_cms_names == "photon_energyErr"] = np.log(
        data_inputs[variables_cms_names == "photon_energyErr"]
    )
    mc_inputs[variables_mc_names == "photon_energyErr"] = np.log(
        mc_inputs[variables_mc_names == "photon_energyErr"]
    )

    mc_inputs, data_inputs, iso_data, iso_mc = apply_scale_and_smooth(
        mc_inputs, data_inputs, variables_cms_names, isolation_variables
    )

    inputs_mean     = np.mean(np.nan_to_num(mc_inputs),     axis=0)
    inputs_std      = np.std( np.nan_to_num(mc_inputs),     axis=0)
    condition_means = np.mean(np.nan_to_num(mc_conditions), axis=0)
    conditions_std  = np.std( np.nan_to_num(mc_conditions), axis=0)

    data_inputs     = (data_inputs     - inputs_mean)    / inputs_std
    mc_inputs       = (mc_inputs       - inputs_mean)    / inputs_std
    data_conditions = (data_conditions - condition_means) / conditions_std
    mc_conditions   = (mc_conditions   - condition_means) / conditions_std

    return (
        data_df, mc_df,
        data_inputs, mc_inputs,
        data_conditions, mc_conditions,
        inputs_mean, inputs_std,
        condition_means, conditions_std,
        iso_data, iso_mc,
        conditions_names,
    )


# ---------------------------------------------------------------------------
# Dataset & DataLoader construction
# ---------------------------------------------------------------------------

def make_tensor_datasets(mc_inputs, data_inputs, mc_conditions, data_conditions,
                         weights_tensor, device):
    base_dataset = TensorDataset(
        torch.tensor(mc_inputs,       dtype=torch.float32, device=device),
        torch.tensor(mc_conditions,   dtype=torch.float32, device=device),
        weights_tensor,
    )
    target_dataset = TensorDataset(
        torch.tensor(data_inputs,     dtype=torch.float32, device=device),
        torch.tensor(data_conditions, dtype=torch.float32, device=device),
    )
    return base_dataset, target_dataset


def split_dataset(dataset, n, train_frac, val_frac, seed_mult, SEED):
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed_mult * SEED))
    train_idx = perm[:int(n * train_frac)].tolist()
    val_idx   = perm[int(n * train_frac):int(n * (train_frac + val_frac))].tolist()
    test_idx  = perm[int(n * (train_frac + val_frac)):].tolist()
    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
        train_idx, val_idx, test_idx,
    )


def split_dataset_area_of_interest(dataset, df, n, train_frac, val_frac, seed_mult, SEED):
    """Like split_dataset, but val/test contain only photons with pt>25 and dR>0.4.
    Photons excluded from val/test are added to the training set."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed_mult * SEED))

    raw_val_idx  = perm[int(n * train_frac):int(n * (train_frac + val_frac))].tolist()
    raw_test_idx = perm[int(n * (train_frac + val_frac)):].tolist()

    def _is_relevant(i):
        row = df.iloc[i]
        return row["photon_pt"] > 25 and row["photon_muon_near_dR"] > 0.4

    val_idx  = [i for i in raw_val_idx  if _is_relevant(i)]
    test_idx = [i for i in raw_test_idx if _is_relevant(i)]
    extra_train = [i for i in raw_test_idx + raw_val_idx if not _is_relevant(i)]
    train_idx = extra_train + perm[:int(n * train_frac)].tolist()

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
        train_idx, val_idx, test_idx,
    )


def build_dataloaders(base_train, base_val, base_test,
                      target_train, target_val, batch_size, SEED):
    dl_base_train = DataLoader(
        base_train, batch_size=batch_size,
        sampler=RandomSampler(base_train, replacement=True, num_samples=len(base_train),
                              generator=torch.Generator().manual_seed(10 * SEED)),
        generator=torch.Generator().manual_seed(44 * SEED),
    )
    dl_base_val = DataLoader(
        base_val, batch_size=batch_size, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(45 * SEED),
    )
    dl_base_test    = DataLoader(base_test, batch_size=batch_size, shuffle=False, drop_last=False)
    dl_target_train = DataLoader(
        target_train, batch_size=batch_size, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(46 * SEED),
    )
    dl_target_val = DataLoader(
        target_val, batch_size=batch_size, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(47 * SEED),
    )
    return dl_base_train, dl_base_val, dl_base_test, dl_target_train, dl_target_val


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _run_epoch(model, cfm, base_loader, target_loader,
               feature_dim, ot_rescale, loss_type, optimizer=None):
    """One training or validation pass.  Pass optimizer=None for eval mode."""
    total_loss, n_batches = 0.0, 0

    for base_batch, target_batch in zip(base_loader, target_loader):
        base_features, base_conditions, weights_batch = base_batch
        target_features, target_conditions            = target_batch

        mean_weight   = weights_batch.mean().item()
        weights_batch = weights_batch / mean_weight

        base   = torch.cat([base_features,   base_conditions],   dim=1) * ot_rescale
        target = torch.cat([target_features, target_conditions], dim=1) * ot_rescale

        t, xt, ut, weights_batch, _ = cfm.guided_sample_location_and_conditional_flow(
            base, target, y0=weights_batch
        )
        ut = ut[:, :feature_dim] / ot_rescale
        xt = xt / ot_rescale

        vt = model(xt[:, :feature_dim], t, xt[:, feature_dim:])

        if loss_type == "weighted":
            loss = ((vt - ut).pow(2) * weights_batch.unsqueeze(1)).mean()
        else:
            loss = ((vt - ut).pow(2)).mean()

        total_loss += float(loss) * mean_weight
        n_batches  += 1

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return total_loss / n_batches if n_batches else float("nan")


def train_model(model, cfm,
                dl_base_train, dl_base_val,
                dl_target_train, dl_target_val,
                feature_dim, ot_rescale, loss_type,
                optimizer, scheduler,
                n_epochs, early_stop_patience, outdir):
    train_losses, val_losses = [], []
    best_val_loss     = float("inf")
    best_val_epoch    = 0
    epochs_no_improve = 0
    start = time.time()

    for epoch in range(n_epochs):
        model.train()
        train_loss = _run_epoch(model, cfm, dl_base_train, dl_target_train,
                                feature_dim, ot_rescale, loss_type, optimizer=optimizer)
        train_losses.append(train_loss)

        with torch.no_grad():
            model.eval()
            val_loss = _run_epoch(model, cfm, dl_base_val, dl_target_val,
                                  feature_dim, ot_rescale, loss_type, optimizer=None)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        end = time.time()

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            best_val_epoch    = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(outdir, "best_model.pth"))
        else:
            epochs_no_improve += 1

        print(
            f"[Epoch {epoch+1}] Loss: {train_loss:.6f}  Val: {val_loss:.6f}  "
            f"Best: {best_val_loss:.6f} @ {best_val_epoch}  "
            f"({end - start:.1f}s)  Patience left: {early_stop_patience - epochs_no_improve}"
        )
        start = end

        if epochs_no_improve >= early_stop_patience:
            print("Early stopping triggered.")
            break

    lossplot_train(train_losses, val_losses, epoch + 1, best_val_epoch, outdir, "loss_plot.png")
    return best_val_epoch


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, dl_base_test, device):
    """ODE-integrate the flow over the full test set.
    Returns corrected_features tensor and trajectory from the first batch."""
    model.eval()
    corrected_features = None
    traj_plot = None

    with torch.no_grad():
        for batch_mc in dl_base_test:
            batch_features_mc, batch_conditions_mc, _ = batch_mc

            def ode_func(t, x):
                x_flat   = x.squeeze(1)
                t_tensor = t * torch.ones(x_flat.size(0), 1, device=x.device)
                v = model(x_flat, t_tensor, batch_conditions_mc)
                if torch.isnan(v).any():
                    raise RuntimeError(f"ODE returned NaN at t={t.item():.4f}")
                return v.unsqueeze(1)

            t_span = torch.linspace(0, 1, 100, device=device)
            traj   = odeint(ode_func, batch_features_mc.unsqueeze(1), t_span,
                            atol=1e-7, rtol=1e-4, method="dopri5")
            batch_out = traj[-1].squeeze(1)

            if traj_plot is None:
                traj_plot = traj

            corrected_features = (
                batch_out if corrected_features is None
                else torch.cat([corrected_features, batch_out], dim=0)
            )

    return corrected_features, traj_plot


# ---------------------------------------------------------------------------
# EMD calculation
# ---------------------------------------------------------------------------

def _emd(x0, x1):
    a = torch.full((x0.shape[0],), 1.0 / x0.shape[0], device=x0.device)
    b = torch.full((x1.shape[0],), 1.0 / x1.shape[0], device=x1.device)
    M = pot.dist(x0.reshape(x0.shape[0], -1), x1.reshape(x1.shape[0], -1))
    return pot.emd2(a, b, M, numItermax=100_000_000)


def compute_all_emds(corrected_features, mc_inputs_np, data_inputs_np,
                     base_test_indices, target_test_indices,
                     mc_df_test, data_df_test, region, device):
    """Returns six EMD values (None where not applicable to the region)."""
    print("Starting EMD calculation...")
    x1 = torch.tensor(data_inputs_np[target_test_indices], dtype=torch.float32, device=device)
    x0 = torch.tensor(mc_inputs_np[base_test_indices],     dtype=torch.float32, device=device)

    EMD_mc_all = EMD_corr_all = None
    EMD_mc_barrel = EMD_corr_barrel = None
    EMD_mc_endcap = EMD_corr_endcap = None

    if region == "all":
        b_data = np.abs(data_df_test["photon_ScEta"].values) < 1.442
        e_data = np.abs(data_df_test["photon_ScEta"].values) > 1.566
        b_mc   = np.abs(mc_df_test["photon_ScEta"].values)   < 1.442
        e_mc   = np.abs(mc_df_test["photon_ScEta"].values)   > 1.566

        for label, x0i, xi_c, x1i in [
            ("all",    x0,         corrected_features,         x1),
            ("barrel", x0[b_mc],   corrected_features[b_mc],   x1[b_data]),
            ("endcap", x0[e_mc],   corrected_features[e_mc],   x1[e_data]),
        ]:
            print(f"  EMD {label}: {x0i.shape}, {x1i.shape}")
            emd_mc, emd_corr = _emd(x0i, x1i), _emd(xi_c, x1i)
            if label == "all":
                EMD_mc_all,    EMD_corr_all    = emd_mc, emd_corr
            elif label == "barrel":
                EMD_mc_barrel, EMD_corr_barrel = emd_mc, emd_corr
            else:
                EMD_mc_endcap, EMD_corr_endcap = emd_mc, emd_corr

    elif region == "barrel":
        EMD_mc_barrel,  EMD_corr_barrel  = _emd(x0, x1), _emd(corrected_features, x1)
    elif region == "endcap":
        EMD_mc_endcap,  EMD_corr_endcap  = _emd(x0, x1), _emd(corrected_features, x1)

    return (EMD_mc_all, EMD_corr_all,
            EMD_mc_barrel, EMD_corr_barrel,
            EMD_mc_endcap, EMD_corr_endcap)


# ---------------------------------------------------------------------------
# MVA computation
# ---------------------------------------------------------------------------

def compute_mva(mc_df_test, data_df_test, variables_cms_names, variables_mc_names):
    """Add corrected MVA, raw MVA, and data MVA columns in-place."""
    extra = ["photon_ScEta", "photon_energyRaw", "Rho_fixedGridRhoAll"]

    corrected_ak = ak.Array(mc_df_test[variables_cms_names + extra].to_dict(orient="records"))
    mc_ak        = ak.Array(mc_df_test[variables_mc_names  + extra].to_dict(orient="records"))
    data_ak      = ak.Array(data_df_test[variables_cms_names + extra].to_dict(orient="records"))

    mc_df_test["mva"]     = ak.to_numpy(add_data_photonid_mva_run3_zmmg(corrected_ak, process=None))
    mc_df_test["raw_mva"] = ak.to_numpy(add_mc_photonid_mva_run3_zmmg(mc_ak,          process=None))
    data_df_test["mva"]   = ak.to_numpy(add_data_photonid_mva_run3_zmmg(data_ak,       process=None))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def get_bin_edges(entry, mean, std):
    if "Iso" in entry:
        return hist.new.Reg(20, 0.0, mean + 4.5 * std, overflow=True)
    elif "es" in entry:
        return hist.new.Reg(20, 0.0, mean + 2.8 * std, overflow=True)
    elif "hoe" in entry:
        return hist.new.Reg(20, 0.0, 0.08, overflow=True)
    elif "DR" in entry:
        return hist.new.Reg(20, 0.0, 6.0, overflow=True)
    elif "energy" in entry:
        return hist.new.Reg(20, 0.0, mean + 2.0 * std, overflow=True)
    elif "pt" in entry or "Rho" in entry:
        return hist.new.Reg(20, mean - 4 * std, mean + 4 * std, overflow=True)
    elif "eta" in entry and "idth" not in entry:
        return hist.new.Reg(20, -2.5, 2.5, overflow=True)
    elif "sieie" in entry:
        return hist.new.Reg(20, mean - 3.0 * std, mean + 3.0 * std, overflow=True)
    elif "sieip" in entry:
        return hist.new.Reg(20, mean - 3.0 * std, mean + 3.0 * std, overflow=True)
    elif "r9" in entry:
        return hist.new.Reg(20, 0.5, 1.1, overflow=True)
    elif "mva" in entry:
        return hist.new.Reg(20, -0.9, 1.0, overflow=True)
    elif "s4" in entry:
        return hist.new.Reg(20, mean - 4.5 * std, mean + 2.5 * std, overflow=True)
    elif "dimuon_mass" in entry:
        return hist.new.Reg(42, 35, 80, overflow=True)
    else:
        return hist.new.Reg(20, mean - 2.0 * std, mean + 3.0 * std, overflow=True)


def _plot_one(entry, entry_corr, entry_data,
              data_df, mc_df, xbins, out_name,
              total_lumi, endcap, barrel,
              data_mask=None, mc_mask=None):
    dv = data_df[entry_data].values
    mv = mc_df[entry].values
    cv = mc_df[entry_corr].values
    wv = mc_df["weights"].values
    if data_mask is not None: dv = dv[data_mask]
    if mc_mask   is not None: mv, cv, wv = mv[mc_mask], cv[mc_mask], wv[mc_mask]
    weights = len(dv) * wv

    dh, mh, ch = xbins.Weight(), xbins.Weight(), xbins.Weight()
    dh.fill(dv)
    mh.fill(mv, weight=weights)
    ch.fill(cv, weight=weights)

    return plot_utils.plott(
        dh, mh, ch, out_name,
        physics_process=r"$Z\rightarrow \mu \mu \gamma$",
        lumi_label=total_lumi, zmmg=True, xlabel=str(entry),
        postEE=True, endcap=endcap, barrel=barrel,
    )


_REGION_KEY = {"all": "all", "barrel": "eb", "endcap": "ee"}


def _update_tracker(tracker, entry, region, result):
    rk = _REGION_KEY.get(region, region)
    if "mva" in entry:
        tracker[f"mva_mc_{rk}"]   = result[0]
        tracker[f"mva_corr_{rk}"] = result[1]
    elif "energyErr" in entry:
        tracker[f"energyErr_mc_{rk}"]   = result[0]
        tracker[f"energyErr_corr_{rk}"] = result[1]


def plot_all_variables(variables_mc_names, variables_cms_names,
                       data_df_test, mc_df_test,
                       region, outdir, total_lumi):
    """
    Plot every variable for the target region.
    For region=='all', also produces barrel and endcap sub-plots.
    Returns (goodness_list, tracker_dict).
    """
    tracker = {f"{m}_{s}_{r}": None
               for m in ("mva", "energyErr")
               for s in ("mc", "corr")
               for r in ("all", "eb", "ee")}
    goodness = []

    for entry, entry_corr, entry_data in zip(variables_mc_names, variables_cms_names, variables_cms_names):
        print(f"Plotting: {entry_data}  (n={len(mc_df_test)})")
        if len(data_df_test) == 0 or len(mc_df_test) == 0:
            continue

        mean = np.nanmean(data_df_test[entry_data].values)
        std  = max(np.nanstd(data_df_test[entry_data].values), 0.1)
        if "energy" in entry:
            print(mean, std)
        xbins = get_bin_edges(entry, mean, std)

        result = _plot_one(
            entry, entry_corr, entry_data,
            data_df_test, mc_df_test, xbins,
            os.path.join(outdir, f"{region}_{entry}.pdf"),
            total_lumi,
            endcap=(region == "endcap"), barrel=(region == "barrel"),
        )
        goodness.append(result)
        _update_tracker(tracker, entry, region, result)

        if region == "all":
            for region_i, d_eta_cut, m_eta_cut in [
                ("barrel", np.abs(data_df_test["photon_ScEta"].values) < 1.442,
                           np.abs(mc_df_test["photon_ScEta"].values)   < 1.442),
                ("endcap", np.abs(data_df_test["photon_ScEta"].values) > 1.566,
                           np.abs(mc_df_test["photon_ScEta"].values)   > 1.566),
            ]:
                sub_result = _plot_one(
                    entry, entry_corr, entry_data,
                    data_df_test, mc_df_test, xbins,
                    os.path.join(outdir, f"{region_i}_{entry}.pdf"),
                    total_lumi,
                    endcap=(region_i == "endcap"),
                    barrel=(region_i == "barrel"),
                    data_mask=d_eta_cut, mc_mask=m_eta_cut,
                )
                goodness.append(sub_result)
                _update_tracker(tracker, entry, region_i, sub_result)

    return goodness, tracker


# ---------------------------------------------------------------------------
# Trajectory plotting  (defined for completeness; not called from main)
# ---------------------------------------------------------------------------

def plot_trajectories(cfm, variables_cms_names, target_dataset_test, base_dataset_test,
                      traj, inputs_mean, inputs_std, outdir,
                      matching_batch=32, matching_batch_visual=128):
    with torch.no_grad():
        N = 256
        colors = np.zeros((N, 4))
        colors[:, :3] = np.array([0, 1, 0])
        colors[:, 3]  = np.linspace(0, 1, N)
        green_alpha_cmap = ListedColormap(colors)

        n_tail   = len(traj)
        tail_idx = range(-1, -n_tail - 1, -1)

        f_data = torch.stack([target_dataset_test[i][0] for i in tail_idx])
        c_data = torch.stack([target_dataset_test[i][1] for i in tail_idx])
        f_mc   = torch.stack([base_dataset_test[i][0]   for i in tail_idx])
        c_mc   = torch.stack([base_dataset_test[i][1]   for i in tail_idx])
        w_mc   = torch.stack([base_dataset_test[i][2]   for i in tail_idx])

        mc   = torch.cat([f_mc,   c_mc],   dim=1) * 1e-2
        data = torch.cat([f_data, c_data], dim=1) * 1e-2
        mc, data, ut, _, _ = cfm.guided_sample_location_and_conditional_flow(
            mc[:matching_batch], data[:matching_batch], y0=w_mc[:matching_batch]
        )
        ut     = ut[:, :f_data.shape[1]].detach().cpu().numpy() / 1e-2
        f_data = f_data.detach().cpu().numpy() * inputs_std + inputs_mean
        f_mc   = f_mc.detach().cpu().numpy()   * inputs_std + inputs_mean
        traj   = traj.squeeze(2).detach().cpu().numpy() * inputs_std + inputs_mean

        outdir = os.path.join(outdir, "trajectories")
        os.makedirs(outdir, exist_ok=True)
        for i, fi in enumerate(variables_cms_names):
            for j, fj in enumerate(variables_cms_names):
                if i >= j:
                    continue
                plt.title("CFM Trajectory")
                plt.xlabel(fi); plt.ylabel(fj)
                h, xe, ye = np.histogram2d(f_data[:, i], f_data[:, j], bins=1000, density=True)
                plt.xlim(inputs_mean[i] - 5*inputs_std[i], inputs_mean[i] + 5*inputs_std[i])
                plt.ylim(inputs_mean[j] - 5*inputs_std[j], inputs_mean[j] + 5*inputs_std[j])
                plt.imshow(h.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
                           aspect="auto", cmap=green_alpha_cmap, label="Data Density")
                plt.scatter(traj[0,  :matching_batch_visual, i], traj[0,  :matching_batch_visual, j],
                            s=10, alpha=0.3, c="blue", label="Simulation")
                plt.scatter(traj[:,  :matching_batch_visual, i], traj[:,  :matching_batch_visual, j],
                            s=0.2, alpha=0.15, c="olive")
                plt.scatter(traj[-1, :matching_batch_visual, i], traj[-1, :matching_batch_visual, j],
                            s=4, alpha=0.4, c="green", label="Flow")
                plt.legend(); plt.xticks([]); plt.yticks([])
                plt.savefig(os.path.join(outdir, f"trajectory_{fi}_{fj}.png"), dpi=300)
                plt.close()


# ---------------------------------------------------------------------------
# Summary writing
# ---------------------------------------------------------------------------

def write_per_run_summary(outdir, region, ot_method, nn_method, binary_position,
                          g_mc, g_morph, tracker,
                          EMD_mc_all, EMD_corr_all,
                          EMD_mc_barrel, EMD_corr_barrel,
                          EMD_mc_endcap, EMD_corr_endcap):
    path = os.path.join(outdir, f"chi2_over_ndf_{g_mc:.3f}->{g_morph:.3f}.txt")

    def _opt(val, fmt=".3f"):
        return f"{val:{fmt}}" if val is not None else "N/A"

    with open(path, "w") as f:
        f.write(f"Region: {region}\nOT method: {ot_method}\n"
                f"NN method: {nn_method}\nBinary position: {binary_position}\n\n")
        f.write(f"Average goodness for MC: {g_mc}\n")
        f.write(f"MVA chi2/ndf for MC: {_opt(tracker.get('mva_mc_all'))}\n")
        f.write(f"MVA chi2/ndf for MC (barrel): {_opt(tracker.get('mva_mc_eb'))}\n")
        f.write(f"MVA chi2/ndf for MC (endcap): {_opt(tracker.get('mva_mc_ee'))}\n")
        f.write(f"\nAverage goodness for corrected: {g_morph}\n")
        f.write(f"MVA chi2/ndf for corrected: {_opt(tracker.get('mva_corr_all'))}\n")
        f.write(f"MVA chi2/ndf for corrected (barrel): {_opt(tracker.get('mva_corr_eb'))}\n")
        f.write(f"MVA chi2/ndf for corrected (endcap): {_opt(tracker.get('mva_corr_ee'))}\n\n")
        if EMD_mc_all    is not None: f.write(f"EMD for MC (all): {EMD_mc_all:.3f}\n")
        if EMD_mc_barrel is not None: f.write(f"EMD for MC (barrel): {EMD_mc_barrel:.3f}\n")
        if EMD_mc_endcap is not None: f.write(f"EMD for MC (endcap): {EMD_mc_endcap:.3f}\n\n")
        if EMD_corr_all    is not None: f.write(f"EMD for corrected (all): {EMD_corr_all:.3f}\n")
        if EMD_corr_barrel is not None: f.write(f"EMD for corrected (barrel): {EMD_corr_barrel:.3f}\n")
        if EMD_corr_endcap is not None: f.write(f"EMD for corrected (endcap): {EMD_corr_endcap:.3f}\n")


def _latex_header(hidden_dim, depth, ot_method, sigma, batch_size):
    ot_label = {"ExactOT": "Balanced", "UOT": "Unbalanced"}.get(ot_method, "None")
    return (
        f"\\multirow{{2}}{{*}}{{(${hidden_dim}$)$^{depth}$}} & "
        f"\\multirow{{2}}{{*}}{{{ot_label}}}    &   "
        f"\\multirow{{2}}{{*}}{{{sigma}}} &   "
        f"\\multirow{{2}}{{*}}{{{batch_size}}}"
    )


def write_aggregate_summary(outdir0, start_time, args, metric_lists):
    """
    Write the multi-seed aggregate summary with LaTeX tables.
    `metric_lists` maps metric name -> Python list of per-seed values.
    """
    def s(key):
        return safe_mean(metric_lists[key]), safe_std(metric_lists[key])

    mva_mc_all,    mva_mc_all_std    = s("mva_mc_all")
    mva_corr_all,  mva_corr_all_std  = s("mva_corr_all")
    mva_mc_eb,     mva_mc_eb_std     = s("mva_mc_eb")
    mva_corr_eb,   mva_corr_eb_std   = s("mva_corr_eb")
    mva_mc_ee,     mva_mc_ee_std     = s("mva_mc_ee")
    mva_corr_ee,   mva_corr_ee_std   = s("mva_corr_ee")

    EMD_mc_all,      EMD_mc_all_std      = s("EMD_mc_all")
    EMD_corr_all,    EMD_corr_all_std    = s("EMD_corr_all")
    EMD_mc_barrel,   EMD_mc_barrel_std   = s("EMD_mc_barrel")
    EMD_corr_barrel, EMD_corr_barrel_std = s("EMD_corr_barrel")
    EMD_mc_endcap,   EMD_mc_endcap_std   = s("EMD_mc_endcap")
    EMD_corr_endcap, EMD_corr_endcap_std = s("EMD_corr_endcap")

    energyErr_mc_all,    energyErr_mc_all_std    = s("energyErr_mc_all")
    energyErr_corr_all,  energyErr_corr_all_std  = s("energyErr_corr_all")
    energyErr_mc_eb,     energyErr_mc_eb_std     = s("energyErr_mc_eb")
    energyErr_corr_eb,   energyErr_corr_eb_std   = s("energyErr_corr_eb")
    energyErr_mc_ee,     energyErr_mc_ee_std     = s("energyErr_mc_ee")
    energyErr_corr_ee,   energyErr_corr_ee_std   = s("energyErr_corr_ee")

    def rel_improve(corr_key, mc_key):
        c, m = metric_lists[corr_key], metric_lists[mc_key]
        return [(ci - mi) / mi if mi else None for ci, mi in zip(c, m)]

    opt_mva_all = safe_argmin(rel_improve("mva_corr_all",       "mva_mc_all"))
    opt_mva_eb  = safe_argmin(rel_improve("mva_corr_eb",        "mva_mc_eb"))
    opt_mva_ee  = safe_argmin(rel_improve("mva_corr_ee",        "mva_mc_ee"))
    opt_emd_all = safe_argmin(rel_improve("EMD_corr_all",       "EMD_mc_all"))
    opt_emd_eb  = safe_argmin(rel_improve("EMD_corr_barrel",    "EMD_mc_barrel"))
    opt_emd_ee  = safe_argmin(rel_improve("EMD_corr_endcap",    "EMD_mc_endcap"))
    opt_err_all = safe_argmin(rel_improve("energyErr_corr_all", "energyErr_mc_all"))
    opt_err_eb  = safe_argmin(rel_improve("energyErr_corr_eb",  "energyErr_mc_eb"))
    opt_err_ee  = safe_argmin(rel_improve("energyErr_corr_ee",  "energyErr_mc_ee"))

    hdr = _latex_header(args.hidden_dim, args.depth, args.ot_method, args.sigma, args.batch_size)
    ml  = metric_lists

    def _at(key, idx):
        return ml[key][idx] if idx is not None else float("nan")

    with open(os.path.join(outdir0, f"{start_time}.txt"), "w") as f:
        f.write(
            "Table format: hidden_dim^depth, OT method, sigma, batch_size, "
            "evaluation metric, ee+eb photons, eb photons, ee photons "
            "(mc metric left, corr metric right)\n\n"
        )

        if args.region == "all":
            f.write(f"EB+EB MVA min = EB MVA min?  {opt_mva_eb == opt_mva_all}\n")
            f.write(f"EB+EE MVA min = EE MVA min?  {opt_mva_ee == opt_mva_all}\n")
            f.write(f"EB MVA min = EE MVA min?     {opt_mva_ee == opt_mva_eb}\n\n")
            f.write(f"EB+EB EMD min = EB EMD min?  {opt_emd_eb == opt_emd_all}\n")
            f.write(f"EB+EE EMD min = EE EMD min?  {opt_emd_ee == opt_emd_all}\n")
            f.write(f"EB EMD min = EE EMD min?     {opt_emd_ee == opt_emd_eb}\n")
        else:
            f.write("-\n-\n-\n\n-\n-\n-\n")

        f.write("\n")
        if mva_mc_all is not None and not np.isnan(mva_mc_all):
            f.write(f"All Min Corr MVA same as All Min Corr EMD:       {opt_mva_all == opt_emd_all}\n")
        else:
            f.write("All Min Gen MVA same as All Min Gen EMD: N/A\n")
        if mva_mc_eb is not None and not np.isnan(mva_mc_eb):
            f.write(f"Barrel Min Corr MVA same as Barrel Min Corr EMD: {opt_mva_eb == opt_emd_eb}\n")
        else:
            f.write("Barrel Min Gen MVA same as Barrel Min Gen EMD: N/A\n")
        if mva_mc_ee is not None and not np.isnan(mva_mc_ee):
            f.write(f"Endcap Min Corr MVA same as Endcap Min Corr EMD: {opt_mva_ee == opt_emd_ee}\n")
        else:
            f.write("Endcap Min Gen MVA same as Endcap Min Gen EMD: N/A\n")

        # Average performance table
        f.write("\n\nLatex table average performance:\n")
        f.write(
            hdr + "    &   MVA   &   "
            f"{mva_mc_all:.2f} ± {mva_mc_all_std:.2f}  &  {mva_corr_all:.2f} ± {mva_corr_all_std:.2f}  &  "
            f"{mva_mc_eb:.2f} ± {mva_mc_eb_std:.2f}    &  {mva_corr_eb:.2f} ± {mva_corr_eb_std:.2f}  &  "
            f"{mva_mc_ee:.2f} ± {mva_mc_ee_std:.2f}    &  {mva_corr_ee:.2f} ± {mva_corr_ee_std:.2f} \\\\\n"
            f"    &&&& EMD &   "
            f"{EMD_mc_all:.2f} ± {EMD_mc_all_std:.2f}  &  {EMD_corr_all:.2f} ± {EMD_corr_all_std:.2f}  &  "
            f"{EMD_mc_barrel:.2f} ± {EMD_mc_barrel_std:.2f}  &  {EMD_corr_barrel:.2f} ± {EMD_corr_barrel_std:.2f}  &  "
            f"{EMD_mc_endcap:.2f} ± {EMD_mc_endcap_std:.2f}  &  {EMD_corr_endcap:.2f} ± {EMD_corr_endcap_std:.2f} \\\\\n"
            f"    &&&& energyErr &   "
            f"{energyErr_mc_all:.2f} ± {energyErr_mc_all_std:.2f}  &  {energyErr_corr_all:.2f} ± {energyErr_corr_all_std:.2f}  &  "
            f"{energyErr_mc_eb:.2f} ± {energyErr_mc_eb_std:.2f}    &  {energyErr_corr_eb:.2f} ± {energyErr_corr_eb_std:.2f}  &  "
            f"{energyErr_mc_ee:.2f} ± {energyErr_mc_ee_std:.2f}    &  {energyErr_corr_ee:.2f} ± {energyErr_corr_ee_std:.2f} \\\\\n\\hline\n"
        )

        # Per-focus optimal tables
        focus_configs = [
            ("EB+EE", opt_mva_all, opt_emd_all, opt_err_all),
            ("EB",    opt_mva_eb,  opt_emd_eb,  opt_err_eb),
            ("EE",    opt_mva_ee,  opt_emd_ee,  opt_err_ee),
        ]
        for focus, mva_idx, emd_idx, err_idx in focus_configs:
            for table_name, idx in [
                (f"{focus} optimal MVA",       mva_idx),
                (f"{focus} optimal EMD",       emd_idx),
                (f"{focus} optimal energyErr", err_idx),
            ]:
                if idx is None:
                    continue
                f.write(f"\nLatex table {table_name} performance:\n")
                f.write(
                    hdr + "    &   MVA   &   "
                    f"{_at('mva_mc_all',idx):.2f}  &  {_at('mva_corr_all',idx):.2f}  &  "
                    f"{_at('mva_mc_eb',idx):.2f}   &  {_at('mva_corr_eb',idx):.2f}   &  "
                    f"{_at('mva_mc_ee',idx):.2f}   &  {_at('mva_corr_ee',idx):.2f} \\\\\n"
                    f"    &&&& \\textit{{EMD}} &   "
                    f"{_at('EMD_mc_all',idx):.2f}  &  {_at('EMD_corr_all',idx):.2f}  &  "
                    f"{_at('EMD_mc_barrel',idx):.2f}  &  {_at('EMD_corr_barrel',idx):.2f}  &  "
                    f"{_at('EMD_mc_endcap',idx):.2f}  &  {_at('EMD_corr_endcap',idx):.2f} \\\\\n"
                    f"    &&&& \\textit{{energyErr}} &   "
                    f"{_at('energyErr_mc_all',idx):.2f}  &  {_at('energyErr_corr_all',idx):.2f}  &  "
                    f"{_at('energyErr_mc_eb',idx):.2f}  &  {_at('energyErr_corr_eb',idx):.2f}  &  "
                    f"{_at('energyErr_mc_ee',idx):.2f}  &  {_at('energyErr_corr_ee',idx):.2f} \\\\\n\\hline\n"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args=None, SEED=42):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device =", device)

    # --- configuration ---
    year        = "2022_2023_2024_pure_rw"
    total_lumi  = 108.96
    train_frac  = 0.68
    val_frac    = 0.14
    n_epochs    = 1000
    scheduler_patience  = 7
    early_stop_patience = scheduler_patience * 2+1
    lr          = 1e-3
    ot_rescale  = 1e-2
    area_of_interest = True    # val/test contain only pt>25 & dR>0.4 photons; EMD disabled

    if args.binary_position not in ("True", "False"):
        raise ValueError("binary_position must be 'True' or 'False'")
    binary_position = args.binary_position == "True"

    variables_cms_names = list(VARIABLES_CMS_NAMES)
    variables_mc_names  = list(VARIABLES_MC_NAMES)
    conditions_names    = list(CONDITIONS_NAMES)

    feature_dim = len(variables_mc_names)
    cond_dim    = len(conditions_names) + (2 if binary_position else 0)

    method = f"{args.nn_method}_{args.ot_method}_{args.region}"
    if binary_position:
        method += "_binary"

    # --- model & optimiser ---
    model = build_model(args.nn_method, feature_dim, cond_dim,
                        args.hidden_dim, args.depth, device)
    cfm   = build_cfm(args.ot_method, args.sigma)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=scheduler_patience,
    )

    # --- output directory ---
    print(time.strftime("%Y-%m-%d %H:%M"))
    outdir0 = (
        f"/home/home1/institut_3a/tsommer/bsc-sommer-project/caios_dnf/CFM/"
        f"results10/full_set/{time.strftime('%Y-%m-%d')}/"
        f"year_{year}/{method}/loss_{args.loss}/depth_{args.depth}/"
        f"hidden_dim_{args.hidden_dim}/sigma_{args.sigma}/lr_{lr}/batch_size_{args.batch_size}"
    )
    start_time = time.strftime("%H:%M")
    outdir = os.path.join(outdir0, start_time)
    os.makedirs(outdir, exist_ok=True)

    # --- data ---
    data_df, mc_df = load_data(year)
    (
        data_df, mc_df,
        data_inputs, mc_inputs,
        data_conditions, mc_conditions,
        inputs_mean, inputs_std,
        condition_means, conditions_std,
        iso_data, iso_mc,
        conditions_names,
    ) = preprocess_data(
        data_df, mc_df,
        variables_cms_names, variables_mc_names, conditions_names,
        ISOLATION_VARIABLES, binary_position, args.region,
    )

    weights = torch.tensor(mc_df["weights"].values, dtype=torch.float32, device=device).abs()
    weights = weights / weights.mean().item()

    base_dataset, target_dataset = make_tensor_datasets(
        mc_inputs, data_inputs, mc_conditions, data_conditions, weights, device
    )

    n_base   = len(base_dataset)
    n_target = len(target_dataset)
    print(n_target, "NUMBER OF DATA DATAPOINTS")

    if area_of_interest:
        (base_train,   base_val,   base_test,
         base_train_idx, _base_val_idx, base_test_idx) = split_dataset_area_of_interest(
            base_dataset, mc_df, n_base, train_frac, val_frac, 42, SEED
        )
        (target_train, target_val, target_test,
         target_train_idx, _target_val_idx, target_test_idx) = split_dataset_area_of_interest(
            target_dataset, data_df, n_target, train_frac, val_frac, 43, SEED
        )
    else:
        (base_train,   base_val,   base_test,
         base_train_idx, _base_val_idx, base_test_idx) = split_dataset(
            base_dataset, n_base, train_frac, val_frac, 42, SEED
        )
        (target_train, target_val, target_test,
         target_train_idx, _target_val_idx, target_test_idx) = split_dataset(
            target_dataset, n_target, train_frac, val_frac, 43, SEED
        )

    mc_df_test   = mc_df.iloc[base_test_idx].copy(deep=True)
    data_df_test = data_df.iloc[target_test_idx].copy(deep=True)

    print(
        f"Data split:  {len(target_train)} train  {len(target_val)} val  "
        f"{len(target_test)} test  (of {n_target})\n"
        f"MC split:    {len(base_train)} train  {len(base_val)} val  "
        f"{len(base_test)} test  (of {n_base})"
    )

    (dl_base_train, dl_base_val, dl_base_test,
     dl_target_train, dl_target_val) = build_dataloaders(
        base_train, base_val, base_test,
        target_train, target_val, args.batch_size, SEED
    )

    # --- training ---
    train_model(
        model, cfm,
        dl_base_train, dl_base_val,
        dl_target_train, dl_target_val,
        feature_dim, ot_rescale, args.loss,
        optimizer, scheduler,
        n_epochs, early_stop_patience, outdir,
    )

    # --- inference ---
    model.load_state_dict(torch.load(os.path.join(outdir, "best_model.pth")))
    corrected_features, _ = run_inference(model, dl_base_test, device)

    # --- optional EMD ---
    EMD_mc_all = EMD_corr_all = None
    EMD_mc_barrel = EMD_corr_barrel = None
    EMD_mc_endcap = EMD_corr_endcap = None

    if False:  # EMD disabled; area_of_interest only affects the data split
        (EMD_mc_all, EMD_corr_all,
         EMD_mc_barrel, EMD_corr_barrel,
         EMD_mc_endcap, EMD_corr_endcap) = compute_all_emds(
            corrected_features, mc_inputs, data_inputs,
            base_test_idx, target_test_idx,
            mc_df_test, data_df_test, args.region, device,
        )

    # --- invert standardization & iso transforms ---
    corrected_np = corrected_features.detach().cpu().numpy() * inputs_std + inputs_mean
    # NOTE: same list-comparison no-op as 9.py (preserving identical behaviour)
    corrected_np[variables_cms_names == "photon_energyErr"] = np.exp(
        corrected_np[variables_cms_names == "photon_energyErr"]
    )
    corrected_np = undo_apply_scale_and_smooth(
        corrected_np, iso_mc, variables_cms_names, ISOLATION_VARIABLES
    )

    mc_df_test.loc[:, variables_cms_names] = corrected_np
    print("total test events:", mc_df_test.shape[0])
    print("  barrel:", (mc_df_test["photon_ScEta"].abs() < 1.442).sum(),
          "  endcap:", (mc_df_test["photon_ScEta"].abs() > 1.566).sum())

    # --- correlation matrices ---
    if args.region == "barrel":
        var_list_barrel = [x for x in variables_cms_names
                           if x not in ("photon_esEffSigmaRR", "photon_esEnergyOverRawE")]
        var_list_endcap = []
    elif args.region == "endcap":
        var_list_barrel = []
        var_list_endcap = list(variables_cms_names)
    else:
        var_list_barrel = [x for x in variables_cms_names
                           if x not in ("photon_esEffSigmaRR", "photon_esEnergyOverRawE")]
        var_list_endcap = list(variables_cms_names)

    plot_correlation_matrices(
        data_df_test.copy(deep=True), mc_df_test.copy(deep=True),
        var_list_barrel, var_list_endcap, args.region,
        path=f"{outdir}/correlation_matrices/",
    )

    # --- MVA ---
    compute_mva(mc_df_test, data_df_test, variables_cms_names, variables_mc_names)
    variables_mc_names  = variables_mc_names  + ["raw_mva"]
    variables_cms_names = variables_cms_names + ["mva"]

    # --- plots ---
    goodness, tracker = plot_all_variables(
        variables_mc_names, variables_cms_names,
        data_df_test, mc_df_test,
        args.region, outdir, total_lumi,
    )

    g_mc    = np.mean(goodness, axis=0)[0]
    g_morph = np.mean(goodness, axis=0)[1]
    print(f"Average goodness before correcting: {g_mc}")
    print(f"Average goodness after  correcting: {g_morph}")

    write_per_run_summary(
        outdir, args.region, args.ot_method, args.nn_method, binary_position,
        g_mc, g_morph, tracker,
        EMD_mc_all, EMD_corr_all,
        EMD_mc_barrel, EMD_corr_barrel,
        EMD_mc_endcap, EMD_corr_endcap,
    )

    os.remove(os.path.join(outdir, "best_model.pth"))

    return (
        outdir0, start_time,
        tracker.get("mva_mc_all"),        tracker.get("mva_corr_all"),
        tracker.get("mva_mc_eb"),         tracker.get("mva_corr_eb"),
        tracker.get("mva_mc_ee"),         tracker.get("mva_corr_ee"),
        EMD_mc_all, EMD_corr_all, EMD_mc_barrel, EMD_corr_barrel, EMD_mc_endcap, EMD_corr_endcap,
        tracker.get("energyErr_mc_all"),  tracker.get("energyErr_corr_all"),
        tracker.get("energyErr_mc_eb"),   tracker.get("energyErr_corr_eb"),
        tracker.get("energyErr_mc_ee"),   tracker.get("energyErr_corr_ee"),
    )


# ---------------------------------------------------------------------------
# Entry point: multi-seed loop & aggregate reporting
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args  = parse_args()
    SEEDS = [2020, 2021, 2022, 2023, 2024, 2025]

    start_time = time.strftime("%H:%M")

    METRIC_KEYS = [
        "mva_mc_all",   "mva_corr_all",
        "mva_mc_eb",    "mva_corr_eb",
        "mva_mc_ee",    "mva_corr_ee",
        "EMD_mc_all",   "EMD_corr_all",
        "EMD_mc_barrel","EMD_corr_barrel",
        "EMD_mc_endcap","EMD_corr_endcap",
        "energyErr_mc_all",  "energyErr_corr_all",
        "energyErr_mc_eb",   "energyErr_corr_eb",
        "energyErr_mc_ee",   "energyErr_corr_ee",
    ]
    metric_lists = {k: [] for k in METRIC_KEYS}
    outdir0 = None

    for seed in SEEDS:
        result = main(args=args, SEED=seed)
        outdir0 = result[0]
        for i, key in enumerate(METRIC_KEYS):
            metric_lists[key].append(result[2 + i])

    write_aggregate_summary(outdir0, start_time, args, metric_lists)

    # Append one-line performance record
    perf_path = "/home/home1/institut_3a/tsommer/bsc-sommer-project/performance_of_interest_df.txt"
    base_cols = (f"{args.ot_method}, {args.loss}, {args.sigma}, {args.batch_size}, "
                 f"{args.depth}, {args.hidden_dim}")

    def _perf_line(label, mva_key, err_key):
        mva_list = metric_lists[mva_key]
        err_list = metric_lists[err_key]
        return (
            f"{label}, {base_cols}, "
            f"{safe_mean(mva_list)}, {safe_std(mva_list)}, {min(mva_list)}, {max(mva_list)}, "
            f"{safe_mean(err_list)}, {safe_std(err_list)}, {min(err_list)}, {max(err_list)}\n"
        )

    with open(perf_path, "a") as f:
        if args.region == "all":
            f.write(_perf_line("EB+EE",    "mva_corr_all", "energyErr_corr_all"))
            f.write(_perf_line("EB+EE_EB", "mva_corr_eb",  "energyErr_corr_eb"))
            f.write(_perf_line("EB+EE_EE", "mva_corr_ee",  "energyErr_corr_ee"))
        elif args.region == "barrel":
            f.write(_perf_line("EB", "mva_corr_eb", "energyErr_corr_eb"))
        elif args.region == "endcap":
            f.write(_perf_line("EE", "mva_corr_ee", "energyErr_corr_ee"))
