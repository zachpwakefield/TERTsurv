"""AS survival analysis utilities with N/event-annotated forest plots.

This script mirrors the SNP survival workflow but is tailored to alternative
splicing (ψ) events. Forest plots include a right-hand column showing the
sample/event counts for interaction and exposure-derived terms when training
data are available.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceError
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test

mpl.rcParams["svg.fonttype"] = "none"  # keep text editable in SVGs
SAVEFIG_KW = dict(format="svg", bbox_inches="tight", pad_inches=0.3)
IMAGE_EXT = ".svg"

# ================== USER CONFIG ==================
ID_COL = "id_new"           # event identifier
SAMPLE = "File.ID"
TIME = "OS.time"
EVENT = "OS.event"
CANCER = "Project.ID"
SEX = "gender_code"        # can be string or numeric; coerced to categorical
AGE = "age_at_diagnosis"
RACE = "race"              # coerced to categorical
PSI = "psi"

PENALIZER = 0.5             # ridge for numerical stability
KM_MIN_PER_GRP = 5          # skip KM if either group too small
ALPHA_INT = 0.05            # significance threshold for psi×cancer
BASELINE_CANCER = None      # set to a cancer name to force baseline; else first alphabetical present
RANDOM_SEED = 123

# ================== HELPERS ==================
def ensure_columns(df: pd.DataFrame) -> None:
    need = {ID_COL, SAMPLE, TIME, EVENT, CANCER, SEX, AGE, RACE, PSI}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def drop_nonfinite_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=[TIME, EVENT, PSI, AGE, SEX, RACE, CANCER, ID_COL])
    d[TIME] = pd.to_numeric(d[TIME], errors="coerce")
    d[EVENT] = pd.to_numeric(d[EVENT], errors="coerce").astype(int)
    d[AGE] = pd.to_numeric(d[AGE], errors="coerce")
    d = d.dropna(subset=[TIME, EVENT, AGE])
    d = d.sort_values([ID_COL, CANCER, SAMPLE])
    return d


def encode_categoricals(dfe: pd.DataFrame) -> pd.DataFrame:
    d = dfe.copy()
    for col in [SEX, RACE, CANCER]:
        d[col] = d[col].astype(str)
        d[col] = pd.Categorical(d[col])
    return d


def cancers_with_nonconstant_psi(dfe: pd.DataFrame) -> set[str]:
    g = dfe.groupby(CANCER, observed=False)[PSI].agg(lambda x: np.nanvar(x.astype(float), ddof=0))
    keep = set(g[g > 0].index.astype(str))
    return keep


def _save_training_metadata(cph: CoxPHFitter, design: pd.DataFrame, event_col: str, exposure: str | None) -> None:
    """Store training design/event info for downstream plots."""
    try:
        cph._tertsurv_training = design.copy()
        cph._tertsurv_event_col = event_col
        cph._tertsurv_exposure = exposure
    except Exception:
        pass


def design_matrix_with_interactions(
    dfe: pd.DataFrame, baseline_cancer: str | None
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Builds X with:
      - continuous: AGE, PSI
      - categorical main effects: SEX, RACE, CANCER (one-hot, drop baseline)
      - interactions: PSI * CANCER_dummies (for non-baseline cancers)
    Returns: X, cancer_levels (full order), cancer_dummies (the columns created)
    """
    d = dfe.copy()

    # choose baseline cancer (for identifiability)
    baseline = baseline_cancer
    if baseline is None:
        baseline = sorted(d[CANCER].cat.categories)[0]
    if baseline not in list(d[CANCER].cat.categories):
        baseline = list(d[CANCER].cat.categories)[0]

    cats = [baseline] + [c for c in d[CANCER].cat.categories if c != baseline]
    d[CANCER] = d[CANCER].cat.reorder_categories(cats, ordered=True)

    sex_d = pd.get_dummies(d[SEX].astype("category"), prefix="sex", drop_first=True)
    race_d = pd.get_dummies(d[RACE].astype("category"), prefix="race", drop_first=True)
    cancer_d = pd.get_dummies(d[CANCER].astype("category"), prefix="cancer", drop_first=True)

    base = pd.DataFrame({
        "age": pd.to_numeric(d[AGE], errors="coerce"),
        "psi": pd.to_numeric(d[PSI], errors="coerce"),
    }, index=d.index)

    inter_cols = {f"{col}:psi": base["psi"] * cancer_d[col] for col in cancer_d.columns}
    inter = pd.DataFrame(inter_cols, index=d.index) if inter_cols else pd.DataFrame(index=d.index)

    X = pd.concat([base, sex_d, race_d, cancer_d, inter], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    cancer_levels = cats
    cancer_dummy_cols = list(cancer_d.columns)
    return X, cancer_levels, cancer_dummy_cols


def tidy_summary(cph: CoxPHFitter) -> pd.DataFrame:
    """Return a tidy Cox summary with consistent column names."""
    s = cph.summary.copy()

    idx_name = s.index.name or "index"
    s = s.reset_index().rename(columns={idx_name: "term"})

    # Handle p-value naming differences
    if "p" not in s.columns and "p_value" in s.columns:
        s = s.rename(columns={"p_value": "p"})

    s["HR"] = np.exp(s["coef"])
    s["HR_lower"] = np.exp(s["coef lower 95%"])
    s["HR_upper"] = np.exp(s["coef upper 95%"])

    se_col = "se(coef)" if "se(coef)" in s.columns else next(
        (c for c in s.columns if c.lower().startswith("se(")), None
    )

    keep_cols: list[str] = ["term", "HR", "HR_lower", "HR_upper", "p"]
    if se_col:
        keep_cols.append(se_col)
    return s[keep_cols]


def forest_plot(
    cph: CoxPHFitter,
    out_svg: Path,
    title: str,
    alpha: float = 0.05,
    hide_cancer_main: bool = True,
    interactions_only: bool = False,
    training_df: pd.DataFrame | None = None,
    exposure: str | None = None,
) -> None:
    """Render a forest plot with HRs, p-values, and optional N/event counts."""

    def fmt_p(p: Any) -> str:
        if p is None or np.isnan(p):
            return "NA"
        return f"{p:.1e}" if p < 1e-4 else f"{p:.4f}"

    summ = tidy_summary(cph).copy()

    # Try to recover training design + event column for counts
    if training_df is None:
        training_df = getattr(cph, "_tertsurv_training", None)
    if training_df is None:
        training_df = getattr(cph, "_training_data", None)

    event_col = getattr(cph, "_tertsurv_event_col", None)
    if event_col is None:
        event_col = getattr(cph, "event_col_", getattr(cph, "event_col", None))
    if exposure is None:
        exposure = getattr(cph, "_tertsurv_exposure", None)

    # hide main-effect cancer rows; keep interactions
    if hide_cancer_main:
        is_cancer_main = (
            summ["term"].str.startswith("cancer_")
            | summ["term"].str.startswith("cancer.type_")
            | summ["term"].str.startswith("cancer.type:")
            | summ["term"].str.startswith("cancer:")
        ) & ~summ["term"].str.contains(":psi")
        summ = summ.loc[~is_cancer_main].reset_index(drop=True)

    if interactions_only:
        summ = summ.loc[summ["term"].str.contains(":")].reset_index(drop=True)

    def pretty_term(t: str) -> str:
        if t == "age":
            return "Age"
        if t == "psi":
            return "ψ"
        if t.startswith("sex_"):
            return "Sex: " + t[len("sex_"):]
        if t.startswith("race_"):
            return "Race: " + t[len("race_"):]
        if exposure and (t == exposure or t.startswith(f"{exposure}_")):
            lbl = t[len(exposure) + 1 :] if t.startswith(f"{exposure}_") else t
            return f"{exposure}={lbl}" if "=" not in lbl else lbl
        if ":psi" in t and t.startswith("cancer_"):
            cn = t.split(":")[0].replace("cancer_", "", 1)
            cn = cn.replace("TCGA-", "")
            return f"ψ × {cn}"
        return t

    summ["pretty"] = summ["term"].map(pretty_term)

    def term_group(t: str) -> int:
        if t == "age":
            return 0
        if t == "psi":
            return 1
        if ":" in t:
            return 3
        return 2

    summ = summ.sort_values(by=["term"], key=lambda col: [(term_group(t), t) for t in col])
    if summ.empty:
        return

    HR = summ["HR"].astype(float).to_numpy()
    L = np.minimum(summ["HR_lower"].astype(float).to_numpy(), HR)
    U = np.maximum(summ["HR_upper"].astype(float).to_numpy(), HR)
    xerr = np.vstack([HR - L, U - HR])
    pvals = summ["p"].astype(float).to_numpy()
    labels = summ["pretty"].astype(str).tolist()
    raw = summ["term"].astype(str).tolist()
    sig = pvals < alpha

    n = len(labels)
    y = np.arange(n)[::-1]
    fig_h = max(4.0, 0.42 * n)
    max_lab = max(len(s) for s in labels) if labels else 10
    fig_w = min(13.0, max(9.0, 0.12 * max_lab + 7.5))

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[2.2, 1.0],
        left=0.08,
        right=0.98,
        bottom=0.12,
        top=0.92,
        wspace=0.02,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_txt = fig.add_subplot(gs[0, 1])

    for i in range(n):
        if i % 2 == 1:
            ax.axhspan(y[i] - 0.5, y[i] + 0.5, color=(0, 0, 0, 0.03), zorder=0)

    colors = np.where(sig, "C3", "C0")
    for i in range(n):
        ax.errorbar(
            HR[i],
            y[i],
            xerr=xerr[:, i:i + 1],
            fmt="o",
            ms=5.5,
            capsize=3,
            lw=1.3,
            color=colors[i],
            ecolor=colors[i],
            zorder=3,
        )

    ax.axvline(1.0, linestyle="--", linewidth=1, color="gray", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel("Hazard ratio (log scale)")

    try:
        xmin = np.nanmin(L[L > 0]) if np.any(L > 0) else 0.2
        xmax = np.nanmax(U[np.isfinite(U)]) if np.any(np.isfinite(U)) else 5.0
        lo = max(0.2, xmin / 1.3)
        hi = min(20.0, xmax * 1.3)
        ax.set_xlim(lo, hi)
    except Exception:
        pass

    ax.grid(axis="x", which="both", linestyle=":", linewidth=0.8, alpha=0.35)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left")

    ax_txt.set_axis_off()
    ax_txt.set_ylim(ax.get_ylim())

    counts: dict[str, tuple[int, int] | None] = {}
    train_cols = set(training_df.columns) if training_df is not None else set()
    show_counts = training_df is not None and event_col is not None and event_col in train_cols

    def _match_training_col(term: str) -> str | None:
        if training_df is None:
            return None
        if term in training_df.columns:
            return term
        variants = set()
        if term.startswith("cancer_"):
            variants.add(term.replace("cancer_", "cancer.type_", 1))
        variants.add(term.replace(":", "_"))
        for var in variants:
            if var in training_df.columns:
                return var
        for col in training_df.columns:
            if col.endswith(term) or col.endswith(term.replace(":", "_")):
                return col
        return None

    def _is_exposure_term(term: str) -> bool:
        if exposure is None:
            return False
        return term == exposure or term.startswith(f"{exposure}_") or term.startswith(f"{exposure}:")

    if show_counts:
        for term in summ["term"]:
            if ":" not in term and not _is_exposure_term(term):
                continue
            col = _match_training_col(term)
            if col is None:
                counts[term] = None
                continue
            mask = training_df[col] > 0
            n_pos = int(mask.sum())
            ev_pos = int(training_df.loc[mask, event_col].sum())
            counts[term] = (n_pos, ev_pos)

    ax_txt.text(0.00, 1.02, "HR [95% CI]", transform=ax_txt.transAxes,
                ha="left", va="bottom", fontsize=11, fontweight="bold")
    ax_txt.text(0.55, 1.02, "p", transform=ax_txt.transAxes,
                ha="left", va="bottom", fontsize=11, fontweight="bold")
    if show_counts:
        ax_txt.text(0.80, 1.02, "N / events", transform=ax_txt.transAxes,
                    ha="left", va="bottom", fontsize=11, fontweight="bold")

    for i in range(n):
        hr_txt = f"{HR[i]:.2f} [{L[i]:.2f}, {U[i]:.2f}]"
        p_txt = fmt_p(pvals[i])
        ax_txt.text(0.00, y[i], hr_txt, ha="left", va="center",
                    fontsize=10, color=("C3" if sig[i] else "black"))
        ax_txt.text(0.55, y[i], p_txt, ha="left", va="center",
                    fontsize=10, color=("C3" if sig[i] else "black"))

        if show_counts:
            ct_txt = ""
            if ":" in raw[i] or _is_exposure_term(raw[i]):
                if raw[i] in counts:
                    val = counts[raw[i]]
                    if val is not None:
                        n_pos, ev_pos = val
                        ct_txt = f"{n_pos} / {ev_pos}"
                    else:
                        ct_txt = "—"
                else:
                    ct_txt = "—"
            ax_txt.text(0.80, y[i], ct_txt, ha="left", va="center",
                        fontsize=10, color=("C3" if sig[i] else "black"))

    from matplotlib.lines import Line2D
    leg_handles = [
        Line2D([0], [0], marker="o", color="C0", label=f"p ≥ {alpha}", markersize=6, linestyle="None"),
        Line2D([0], [0], marker="o", color="C3", label=f"p < {alpha}", markersize=6, linestyle="None"),
    ]
    ax.legend(handles=leg_handles, loc="lower left", frameon=False)

    plt.tight_layout(rect=[0.00, 0.00, 1.00, 0.98])
    fig.savefig(out_svg.with_suffix(".svg"), **SAVEFIG_KW)
    plt.close(fig)


__all__ = ["forest_plot", "tidy_summary", "SAVEFIG_KW", "process"]


# ================== MODEL FITTING ==================
def fit_cox_for_event(df_event: pd.DataFrame, penalizer: float, baseline_cancer: str | None):
    keep_cancers = cancers_with_nonconstant_psi(df_event)
    df_filt = df_event[df_event[CANCER].astype(str).isin(keep_cancers)].copy()
    if df_filt[CANCER].nunique() < 1:
        return None, None, None, None, None, None

    y_time = df_filt[TIME].astype(float)
    y_event = df_filt[EVENT].astype(int)

    df_filt = encode_categoricals(df_filt)
    X, cancer_levels, cancer_dummy_cols = design_matrix_with_interactions(df_filt, baseline_cancer)
    y_time = y_time.loc[X.index]
    y_event = y_event.loc[X.index]

    if y_event.sum() == 0 or len(X) < 10:
        return None, None, None, None, None, None

    design = pd.concat([X, y_time.rename("T"), y_event.rename("E")], axis=1)
    design = _clean_design_matrix(design, protect=("T", "E"))

    # ensure required covariates remain after cleaning
    if "psi" not in design.columns or design["E"].sum() == 0 or len(design) < 10:
        return None, None, None, None, None, None

    try:
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(design, duration_col="T", event_col="E", show_progress=False)
    except ConvergenceError:
        # retry with progressively stronger ridge / small elastic net
        cph = _fit_cox_with_fallbacks(design, penalizer_init=penalizer)

    _save_training_metadata(cph, design, event_col="E", exposure="psi")
    return cph, design.drop(columns=["T", "E"], errors="ignore"), y_time, y_event, cancer_levels, cancer_dummy_cols


def tidy_per_cancer_counts(df_event: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for ev, sub in df_event.groupby(ID_COL, observed=True):
        nonzero = sub.loc[sub[PSI] > 0, PSI]
        cancers = sub.loc[sub[PSI] > 0, CANCER].unique().tolist()
        stats.append({
            "id_new": ev,
            "n_total": len(sub),
            "n_nonzero": len(nonzero),
            "frac_nonzero": len(nonzero) / len(sub) if len(sub) else np.nan,
            "psi_mean_nonzero": nonzero.mean() if len(nonzero) else np.nan,
            "psi_median_nonzero": nonzero.median() if len(nonzero) else np.nan,
            "psi_max": sub[PSI].max(),
            "psi_min": sub[PSI].min(),
            "n_cancers_nonzero": len(cancers),
            "cancers_nonzero": ", ".join(sorted(map(str, cancers))),
        })
    return pd.DataFrame(stats)


def find_significant_interactions(cph: CoxPHFitter, alpha: float = 0.05) -> list[str]:
    summ = tidy_summary(cph)
    mask = summ["term"].str.contains(":psi") & (summ["p"] < alpha)
    sig_terms = summ.loc[mask, "term"].tolist()
    cancers = [t.split(":")[0].replace("cancer_", "", 1) for t in sig_terms]
    return cancers


def safe_name(s: str) -> str:
    return "".join([c if c.isalnum() or c in "-._" else "_" for c in str(s)])


# ---------- KM PLOTS ----------
def km_plot_within_cancer(
    df_event: pd.DataFrame,
    cancer_name: str,
    out_png: Path,
    event_label: str,
    figsize=(9, 7.5),
    dpi=250,
    base_fontsize=12,
    bottom_pad=0.25,
):
    d = df_event[df_event[CANCER].astype(str) == cancer_name].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[TIME, EVENT, PSI])
    if len(d) < 2:
        return

    m = d[PSI].mean()
    d["psi_group"] = np.where(d[PSI] >= m, "High ψ", "Low ψ")

    vc = d["psi_group"].value_counts()
    if vc.min() < KM_MIN_PER_GRP or len(vc) < 2:
        return

    old_size = mpl.rcParams.get("font.size", 10)
    mpl.rcParams["font.size"] = base_fontsize

    d_hi = d[d["psi_group"] == "High ψ"]
    d_lo = d[d["psi_group"] == "Low ψ"]

    try:
        lr = logrank_test(
            d_hi[TIME].astype(float), d_lo[TIME].astype(float),
            event_observed_A=d_hi[EVENT].astype(int),
            event_observed_B=d_lo[EVENT].astype(int),
        )
        pval = float(lr.p_value)
    except Exception:
        pval = np.nan

    def fmt_p(p):
        if np.isnan(p):
            return "NA"
        return f"{p:.1e}" if p < 1e-4 else f"{p:.4f}"

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for grp, sub in d.groupby("psi_group"):
        kmf.fit(
            durations=sub[TIME].astype(float),
            event_observed=sub[EVENT].astype(int),
            label=f"{grp} (n={len(sub)})",
        )
        kmf.plot_survival_function(ax=ax)

    ax.set_title(f"{event_label}\n{cancer_name}: KM (High vs Low ψ) — log-rank p={fmt_p(pval)}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.legend(loc="best", frameon=False)
    ax.margins(x=0.02)

    used_pad = dict(rect=[0, bottom_pad, 1, 1])
    try:
        kmf_high = KaplanMeierFitter().fit(d_hi[TIME].astype(float), d_hi[EVENT].astype(int), label="High ψ")
        kmf_low = KaplanMeierFitter().fit(d_lo[TIME].astype(float), d_lo[EVENT].astype(int), label="Low ψ")
        add_at_risk_counts(kmf_high, kmf_low, ax=ax)
    except Exception:
        used_pad = dict()

    plt.tight_layout(**used_pad)
    fig.savefig(out_png.with_suffix(IMAGE_EXT), **SAVEFIG_KW)
    plt.close(fig)
    mpl.rcParams["font.size"] = old_size


# ---------- STRATIFIED + PER-CANCER ----------
def design_matrix_stratified(dfe: pd.DataFrame) -> pd.DataFrame:
    d = dfe.copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[TIME, EVENT, PSI, AGE, SEX, RACE, CANCER])
    d = encode_categoricals(d)

    X = pd.DataFrame({
        "age": pd.to_numeric(d[AGE], errors="coerce"),
        "psi": pd.to_numeric(d[PSI], errors="coerce"),
        "T": pd.to_numeric(d[TIME], errors="coerce"),
        "E": pd.to_numeric(d[EVENT], errors="coerce").astype(int),
        CANCER: d[CANCER].astype(str),
    }, index=d.index)
    sex_d = pd.get_dummies(d[SEX].astype("category"), prefix="sex", drop_first=True, dtype=float)
    race_d = pd.get_dummies(d[RACE].astype("category"), prefix="race", drop_first=True, dtype=float)
    X = pd.concat([X, sex_d, race_d], axis=1)
    X = X.dropna()
    return X


def fit_cox_stratified_for_event(df_event: pd.DataFrame, penalizer: float):
    keep = cancers_with_nonconstant_psi(df_event)
    dfe = df_event[df_event[CANCER].astype(str).isin(keep)].copy()
    if dfe.empty:
        return None, None

    X = design_matrix_stratified(dfe)
    if X["E"].sum() < 5 or len(X) < 10:
        return None, None

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(X, duration_col="T", event_col="E", strata=[CANCER], show_progress=False)
    _save_training_metadata(cph, X, event_col="E", exposure="psi")
    return cph, X


def _clean_design_matrix(X: pd.DataFrame, protect: Iterable[str] = ("T", "E")) -> pd.DataFrame:
    X = X.copy()
    X = X.dropna(axis=1, how="all")
    protect = tuple(protect)
    feat_cols = [c for c in X.columns if c not in protect]

    for c in feat_cols:
        if not np.issubdtype(X[c].dtype, np.number):
            v = pd.to_numeric(X[c], errors="coerce")
            if v.notna().mean() > 0.99:
                X[c] = v
            else:
                X[c] = X[c].astype("category").cat.codes.astype(float)

    feat_cols = [c for c in X.columns if c not in protect]
    const = [c for c in feat_cols if X[c].nunique(dropna=True) <= 1]
    if const:
        X = X.drop(columns=const, errors="ignore")

    feat_cols = [c for c in X.columns if c not in protect]
    for c in feat_cols:
        v = pd.to_numeric(X[c], errors="coerce")
        if v.nunique(dropna=True) > 2:
            m = v.mean(); sd = v.std(ddof=0)
            if np.isfinite(sd) and sd > 0:
                X[c] = (v - m) / sd

    feat_cols = [c for c in X.columns if c not in protect]
    R = X[feat_cols].apply(pd.to_numeric, errors="coerce").round(10)
    dup_mask = R.T.duplicated(keep="first")
    if dup_mask.any():
        X = X.drop(columns=R.columns[dup_mask], errors="ignore")
    return X


def _fit_cox_with_fallbacks(X, penalizer_init=0.5):
    tries = [
        dict(penalizer=penalizer_init, l1_ratio=0.0),
        dict(penalizer=2.0, l1_ratio=0.0),
        dict(penalizer=10.0, l1_ratio=0.0),
        dict(penalizer=2.0, l1_ratio=0.1),
    ]
    last_err = None
    for kw in tries:
        try:
            cph = CoxPHFitter(**kw)
            cph.fit(X, duration_col="T", event_col="E", show_progress=False)
            return cph
        except ConvergenceError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err


def fit_per_cancer_psi(df_event: pd.DataFrame, penalizer: float = 0.5) -> pd.DataFrame:
    rows = []
    for cn, d in df_event.groupby(CANCER, observed=False):
        d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[TIME, EVENT, PSI, AGE, SEX, RACE])
        if len(d) < 10 or d[EVENT].sum() < 5 or d[PSI].nunique() < 2:
            continue

        d[SEX] = d[SEX].astype(str); d[RACE] = d[RACE].astype(str)

        X = pd.get_dummies(
            d[[AGE, PSI, SEX, RACE]].assign(
                **{AGE: pd.to_numeric(d[AGE], errors="coerce"),
                   PSI: pd.to_numeric(d[PSI], errors="coerce")}
            ),
            columns=[SEX, RACE], drop_first=True, dtype=float,
        )
        X["T"] = pd.to_numeric(d[TIME], errors="coerce")
        X["E"] = pd.to_numeric(d[EVENT], errors="coerce").astype(int)
        X = X.dropna()

        if len(X) < 10 or X["E"].sum() < 5:
            continue

        X = _clean_design_matrix(X, protect=("T", "E"))
        if "psi" not in X.columns:
            continue

        try:
            cph = _fit_cox_with_fallbacks(X, penalizer_init=penalizer)
        except ConvergenceError:
            rows.append({
                "cancer": str(cn),
                "coef": np.nan, "se": np.nan, "z": np.nan, "p": np.nan,
                "HR": np.nan, "HR_lower": np.nan, "HR_upper": np.nan,
                "n": int(len(X)), "events": int(X["E"].sum()),
                "note": "convergence_failed",
            })
            continue

        if "psi" in cph.summary.index:
            r = cph.summary.loc["psi"]
            se_name = "se(coef)" if "se(coef)" in cph.summary.columns else next(
                (c for c in cph.summary.columns if c.lower().startswith("se(")), None
            )
            rows.append({
                "cancer": str(cn),
                "coef": float(r["coef"]),
                "se": float(r[se_name]) if se_name else np.nan,
                "z": float(r["z"]),
                "p": float(r["p"]),
                "HR": float(np.exp(r["coef"])),
                "HR_lower": float(np.exp(r["coef lower 95%"])),
                "HR_upper": float(np.exp(r["coef upper 95%"])),
                "n": int(len(X)),
                "events": int(X["E"].sum()),
                "note": "",
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("cancer")
    return out


def per_cancer_psi_forest(dfpsi: pd.DataFrame, out_png: Path, title: str, alpha: float = 0.05):
    if dfpsi is None or dfpsi.empty:
        return
    df = dfpsi.copy()
    df = df.sort_values("cancer")
    y = np.arange(len(df))[::-1]
    HR = df["HR"].to_numpy(float); L = df["HR_lower"].to_numpy(float); U = df["HR_upper"].to_numpy(float)
    p = df["p"].to_numpy(float); labs = df["cancer"].astype(str).tolist()
    L = np.minimum(L, HR); U = np.maximum(U, HR)
    xerr = np.vstack([HR - L, U - HR])
    sig = p < alpha
    colors = np.where(sig, "C3", "C0")

    fig, ax = plt.subplots(figsize=(10.5, max(4, 0.38*len(labs))))
    for i in range(len(labs)):
        ax.errorbar(HR[i], y[i], xerr=xerr[:, i:i+1], fmt='o', capsize=3, color=colors[i])
    ax.axvline(1.0, ls="--", lw=1, color="gray")
    ax.set_yticks(y); ax.set_yticklabels(labs)
    ax.set_xscale("log"); ax.set_xlabel("Hazard Ratio for ψ (log scale)")
    ax.set_title(title)

    xmin, xmax = ax.get_xlim()
    xtxt = xmax * (xmax / xmin) ** 0.02
    for i in range(len(labs)):
        txt = f"HR {HR[i]:.2f} [{L[i]:.2f}, {U[i]:.2f}]   p={(('%.1e'%p[i]) if p[i]<1e-4 else ('%.4f'%p[i]))}"
        ax.text(xtxt, y[i], txt, va="center", ha="left", fontsize=9, color=("C3" if sig[i] else "black"))
    ax.set_xlim(xmin, xmax * (xmax / xmin) ** 0.25)

    from matplotlib.lines import Line2D
    ax.legend([Line2D([0],[0], marker='o', color='C0', ls=''), Line2D([0],[0], marker='o', color='C3', ls='')],
              [f"p ≥ {alpha}", f"p < {alpha}"], loc="lower left", frameon=False)
    plt.tight_layout()
    fig.savefig(out_png.with_suffix(IMAGE_EXT), **SAVEFIG_KW); plt.close(fig)


# ================== MAIN PIPELINE ==================
def process(df: pd.DataFrame, out_root: Path):
    np.random.seed(RANDOM_SEED)
    ensure_columns(df)
    df = drop_nonfinite_and_sort(df)
    df = encode_categoricals(df)

    out_root.mkdir(parents=True, exist_ok=True)

    for event_id, dfe in df.groupby(ID_COL, sort=False):
        event_dir = out_root / safe_name(str(event_id))
        event_dir.mkdir(exist_ok=True, parents=True)

        cph, X, y_time, y_event, cancer_levels, cancer_dummy_cols = fit_cox_for_event(
            dfe, penalizer=PENALIZER, baseline_cancer=BASELINE_CANCER
        )
        if cph is None:
            (event_dir / "SKIPPED.txt").write_text("Insufficient variation/events after filtering constant-psi cancers.")
            continue

        summ = tidy_summary(cph)
        summ_path = event_dir / "cox_summary.csv"
        summ.to_csv(summ_path, index=False)

        forest_path = event_dir / "cox_forest.svg"
        forest_plot(cph, forest_path, title=f"{event_id} — Cox PH with ψ×Cancer", exposure="psi")

        cph_strat, Xs = fit_cox_stratified_for_event(dfe, penalizer=PENALIZER)
        if cph_strat is not None:
            strat_csv = event_dir / "stratified_cox_summary.csv"
            tidy_summary(cph_strat).to_csv(strat_csv, index=False)
            strat_png = event_dir / "stratified_cox_forest.svg"
            forest_plot(cph_strat, strat_png, title=f"{event_id} — Cox PH (stratified by cancer)",
                        alpha=ALPHA_INT, hide_cancer_main=True, exposure="psi")

        dfpsi = fit_per_cancer_psi(dfe, penalizer=PENALIZER)
        if dfpsi is not None and not dfpsi.empty:
            dfpsi.to_csv(event_dir / "per_cancer_psi.csv", index=False)
            per_cancer_psi_forest(dfpsi, event_dir / "per_cancer_psi_forest.svg",
                                  title=f"{event_id} — ψ effect per cancer (adjusted)", alpha=ALPHA_INT)

        sig_cancers = find_significant_interactions(cph, alpha=ALPHA_INT)
        for cn in sig_cancers:
            sub = dfe.copy()
            if cn not in set(sub[CANCER].astype(str).unique()):
                continue
            if sub.loc[sub[CANCER].astype(str) == cn, PSI].nunique() < 2:
                continue
            km_path = event_dir / f"KM_{safe_name(cn)}.svg"
            km_plot_within_cancer(dfe, cn, km_path, event_label=str(event_id))


def savefig_svg(fig, out_path: Path):
    out_path = out_path.with_suffix(IMAGE_EXT)
    fig.savefig(out_path, **SAVEFIG_KW)
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    ap = argparse.ArgumentParser(description="Per-event Cox forest & within-cancer KM (significant psi×cancer).")
    ap.add_argument("--input", required=True, help="Path to long CSV/Parquet with required columns.")
    ap.add_argument("--format", choices=["csv", "parquet"], default="csv")
    ap.add_argument("--outdir", default="./output", help="Output root directory.")
    args = ap.parse_args()

    in_path = Path(args.input)
    if args.format == "csv":
        df = pd.read_csv(in_path)
    else:
        df = pd.read_parquet(in_path)

    process(df, Path(args.outdir))


def run_notebook_workflow(df: pd.DataFrame, outdir: str | Path = "./output") -> Path:
    """Run the AS survival workflow from a Jupyter notebook.

    The quick-start usage from a notebook cell looks like this::

        import pandas as pd
        from pathlib import Path
        from temp_surv_as import run_notebook_workflow

        df = pd.read_csv("AS_long_joined.csv")
        outdir = run_notebook_workflow(df, Path("./notebook_outputs"))
        outdir

    Parameters
    ----------
    df : pd.DataFrame
        Long-format input with columns including id_new, OS.time, OS.event,
        Project.ID, gender_code, age_at_diagnosis, race, and psi.
    outdir : str | Path, optional
        Directory where outputs will be written. Created if missing.

    Returns
    -------
    Path
        The resolved output directory.
    """
    out_path = Path(outdir)
    process(df, out_path)
    return out_path


def run_notebook_from_path(input_path: str | Path, outdir: str | Path = "./output", fmt: str | None = None) -> Path:
    """Load input data and run the workflow with a one-liner in notebooks.

    Example
    -------
    >>> from temp_surv_as import run_notebook_from_path
    >>> run_notebook_from_path("AS_long_joined.parquet", "./nb_out")

    Parameters
    ----------
    input_path : str | Path
        CSV or Parquet file to read. Format is inferred from the suffix unless
        ``fmt`` is provided.
    outdir : str | Path, optional
        Output directory where figures and summaries are written.
    fmt : {"csv", "parquet"}, optional
        Explicit format override; defaults to the file suffix.

    Returns
    -------
    Path
        The resolved output directory.
    """
    in_path = Path(input_path)
    fmt = (fmt or in_path.suffix.lstrip(".")).lower()
    if fmt not in {"csv", "parquet"}:
        raise ValueError("fmt must be 'csv' or 'parquet'")

    if fmt == "csv":
        df = pd.read_csv(in_path)
    else:
        df = pd.read_parquet(in_path)

    return run_notebook_workflow(df, outdir)


if __name__ == "__main__":
    main()
