#!/usr/bin/env python
"""
tert_snp_survival.py  – Survival analysis for TERT promoter SNPs
Outputs: /projectnb/evolution/zwakefield/tcga/TERTsnp_yunwei/
"""

import os, sys
from pathlib import Path
from lifelines.plotting import add_at_risk_counts
import matplotlib as mpl

from lifelines.exceptions import ConvergenceError

mpl.rcParams.update({
    "pdf.fonttype": 42,     # TrueType fonts in PDFs (not Type 3 outlines)
    "ps.fonttype": 42,
    "svg.fonttype": "none", # keep text as <text> in SVG
    "pdf.use14corefonts": False,
    "path.simplify": False, # (optional) keep paths intact, fewer weird masks
})
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, statistics, CoxPHFitter
from lifelines.statistics import proportional_hazard_test

import warnings
warnings.filterwarnings("ignore", message="Calling float on a single element")

# ────────────────────────── paths & constants ──────────────────────────
OUTROOT   = Path("/projectnb/evolution/zwakefield/tcga/TERTsnp_yunwei")
# SNP_XLSX  = OUTROOT / "TCGA_TERT_SNP_IDs.xlsx"
SNP_XLSX  = OUTROOT / "tert_data_12_8.tsv"
CLIN_CSV  = "/projectnb2/evolution/zwakefield/tcga/sir_analysis/harmonized/clinical_harmonized_numeric.csv"
MIN_PER_GROUP = 6
MIN_PER_CANCER = 6
FIG_EXT = ".svg"
covars_init = ["gender", "race", "age_at_diagnosis"]#, "stage_code"]

# ─────────────────────────── helpers ───────────────────────────────────
def safe_mkdir(p: Path): p.mkdir(parents=True, exist_ok=True)

def logrank_p(df, t, e, g):
    if df[g].nunique() == 2:
        g1, g2 = df[g].unique()
        lr = statistics.logrank_test(
            df.loc[df[g]==g1, t], df.loc[df[g]==g2, t],
            df.loc[df[g]==g1, e], df.loc[df[g]==g2, e]
        )
    else:
        lr = statistics.multivariate_logrank_test(df[t], df[g], df[e])
    return lr.test_statistic, lr.p_value

def km_plot(df, t, e, g, title, png):
    kmf = KaplanMeierFitter(); sns.set_style("whitegrid"); sns.set_context("talk", 0.8)
    fig, ax = plt.subplots(figsize=(6,4))
    # for grp, sub in df.groupby(g, observed=True, sort=False):
    #     kmf.fit(sub[t], sub[e], label=str(grp))
    #     kmf.plot_survival_function(ax=ax, ci_show=True)

    kmfs = []  # keep the fitted KM objects for the risk table
    for grp, sub in df.groupby(g, observed=True, sort=False):
        kmf = KaplanMeierFitter()
        kmf.fit(sub[t], sub[e], label=str(grp))
        kmf.plot_survival_function(ax=ax, ci_show=True)
        kmfs.append(kmf)

    # one call, after plotting, with all fitted KM objects
    add_at_risk_counts(*kmfs, ax=ax)
    # add_at_risk_counts(kmf, ax=ax)
    _, p = logrank_p(df, t, e, g)
    ax.text(0.98, 0.04, f"log-rank p = {p:.3g}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, lw=0))
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Survival probability")
    ax.set_title(title); ax.legend(title=g, frameon=False)
    fig.tight_layout(); fig.savefig(png.with_suffix(FIG_EXT), dpi=120); plt.close(fig)

def cox_survival_plot(
    cph: CoxPHFitter,
    cov_df: pd.DataFrame,
    title: str,
    out_png: Path,
):
    """
    Plot model-based survival curves for one or more covariate rows.

    Parameters
    ----------
    cph      : the fitted CoxPHFitter
    cov_df   : DataFrame where each row is a scenario you want to plot
               (same columns and dtypes as the model’s design matrix).
    title    : plot title
    out_png  : output stem – extension added automatically
    """
    import matplotlib.pyplot as plt
    sns.set_style("whitegrid"); sns.set_context("talk", 0.8)

    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, row in cov_df.iterrows():
        sf = cph.predict_survival_function(row.to_frame().T)
        sf.plot(ax=ax, label=str(idx))

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Model-based survival probability")
    ax.set_title(title)
    ax.legend(title="Scenario", frameon=False)
    fig.tight_layout()
    fig.savefig(out_png.with_suffix(FIG_EXT), dpi=120)
    plt.close(fig)


# def cox_summary(df, t, e, exposure, adjust, *, strata=None, out_tsv):
#     X = pd.get_dummies(df[[exposure] + adjust], drop_first=True)
#     if strata is not None:
#         X[strata] = df[strata]
#     cph = CoxPHFitter(penalizer=0.01)
#     cph.fit(pd.concat([df[[t, e]], X], axis=1),
#             duration_col=t, event_col=e, strata=strata, robust=True)
#     cph.summary.to_csv(out_tsv, sep="\t")
#     return cph.concordance_index_, cph

def cox_summary(df, t_col, e_col, exposure, adjust, *, strata=None, out_tsv):
    """
    df      : DataFrame with all variables
    t_col   : name of time column in df (e.g. 'OS.time')
    e_col   : name of event column in df (e.g. 'OS.event')
    exposure: main variable of interest (string; can be categorical)
    adjust  : list of covariate names (numeric or categorical)
    strata  : column name or None
    out_tsv : path for summary output
    """
    # 1. Build covariate design matrix from raw df
    covars = [exposure] + list(adjust)
    X = df[covars].copy()

    # One-hot encode categoricals (drop first level)
    X = pd.get_dummies(X, drop_first=True)

    # 2. If strata is used, bring it in *before* any row filtering
    if strata is not None:
        X[strata] = df[strata]

    # 3. Combine with survival columns, aligned on index
    df_fit = pd.concat([df[[t_col, e_col]], X], axis=1)

    # 4. Replace inf with NaN and drop any rows with missing/NaN
    df_fit = df_fit.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    # 5. Drop constant columns (zero variance) except time/event
    nunique = df_fit.nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index.tolist()
    const_cols = [c for c in const_cols if c not in (t_col, e_col)]
    if const_cols:
        print(f"[cox_summary] Dropping constant covariates: {const_cols}")
        df_fit = df_fit.drop(columns=const_cols)

    # 6. Quick sanity checks: at least one event; at least one covariate
    if df_fit[e_col].sum() == 0:
        print("[cox_summary] No events in this subset – skipping Cox fit.")
        return np.nan, None
    if df_fit.shape[1] <= 2:  # only time + event
        print("[cox_summary] No usable covariates after filtering – skipping Cox fit.")
        return np.nan, None

    # 7. Fit Cox model with slightly stronger ridge regularization
    cph = CoxPHFitter(penalizer=0.1)  # stronger than 0.01

    try:
        cph.fit(
            df_fit,
            duration_col=t_col,
            event_col=e_col,
            strata=strata,
            robust=True,
        )
    except ConvergenceError as err:
        print("[cox_summary] ConvergenceError:", err)
        return np.nan, None

    cph.summary.to_csv(out_tsv, sep="\t")
    return cph.concordance_index_, cph

def cox_cancer_interaction(
    df          : pd.DataFrame,
    t           : str,
    e           : str,
    snp_flag    : str,          # column with 0/1
    base_covars : list[str],    # the usual gender / race / age …
    out_tsv     : Path,
    penalizer   : float = 0.05, # a bit more ridge for many dummies
):
    """
    Cancer type is treated like any other categorical covariate and every
    dummy is interacted with the SNP flag.

    Returns the fitted CoxPHFitter (or None if convergence fails).
    """
    df = df.copy()

    # 1. one-hot cancer.type  (reference level dropped automatically)
    df = pd.get_dummies(df, columns=["cancer.type"], drop_first=True)
    cancer_dummies = [c for c in df.columns if c.startswith("cancer.type_")]

    # 2. make sure the SNP flag is numeric 0/1
    df["SNP_FLAG"] = df[snp_flag].astype(int)

    THRESH_N_SNP_POS = 3
    
    # 3. interaction columns
    inter_cols = []
    for cd in cancer_dummies:
        n_pos = (df[cd] & df["SNP_FLAG"]).sum()    # #samples in that cancer + SNP=1
        if n_pos < THRESH_N_SNP_POS:
            print(f"  · skip {cd}:SNP_FLAG  (only {n_pos} SNP-positive)")
            continue
        cname = f"{cd}:SNP_FLAG"
        prod  = df[cd] * df["SNP_FLAG"]
        if prod.var() == 0:            # drop zero-variance terms
            continue
        df[cname] = prod
        inter_cols.append(cname)

    # 4. final design matrix
    design = (
        ["SNP_FLAG"]
        + cancer_dummies
        + inter_cols
        + base_covars           # continuous or to-be-dummied later
    )
    X = pd.get_dummies(df[design], drop_first=True)

    # 5. fit
    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(
            pd.concat([df[[t, e]], X], axis=1),
            duration_col=t, event_col=e, robust=True
        )
    except Exception as err:
        print(f"[WARN] interaction model failed: {err}")
        return None

    cph.summary.to_csv(out_tsv, sep="\t")
    return cph

def forest_plot(
    cox_tsv: Path,
    out_stem: Path,
    title: str,
    annotate: bool = True,
    star: bool = True,
    exposure_prefix: tuple[str, ...] = ("group", "PAT_LABEL", "SNP_PATTERN", "Any_SNP"),
):
    """
    Draw a forest plot from a lifelines Cox summary.

    • Blue rows = exposures of interest   (label starts with exposure_prefix)
    • Grey rows = adjustment covariates   (e.g. gender, race)
    • log-scale x-axis centred on HR = 1
    • Optional annotation: “HR  (p=…)” + significance stars
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    summ = pd.read_csv(cox_tsv, sep="\t", index_col=0)
    if summ.empty:
        return

    # keep order as in table
    hr   = summ["exp(coef)"]
    lo   = summ["exp(coef) lower 95%"]
    hi   = summ["exp(coef) upper 95%"]
    pval = summ["p"]

    yticks = summ.index.tolist()
    n      = len(yticks)

    # ── adaptive figure size ────────────────────────────────────────────
    fig_w = max(5.0, 0.10 * max(map(len, yticks)) + 4.0)
    fig_h = 0.6 * n + 1.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ── loop row-by-row so we can colour exposures vs covariates ───────
    for y, name in enumerate(yticks):
        is_exposure = name.startswith(exposure_prefix)
        color       = "tab:blue" if is_exposure else "grey"

        ax.errorbar(
            hr[name],
            y,
            xerr=[[hr[name] - lo[name]], [hi[name] - hr[name]]],
            fmt="o",
            capsize=3,
            color=color,
            elinewidth=1.2,
        )

        if annotate and is_exposure:
            stars = ""
            if star:
                pv = pval[name]
                if   pv < 0.001: stars = " ***"
                elif pv < 0.01:  stars = " **"
                elif pv < 0.05:  stars = " *"

            ax.annotate(
                f"{hr[name]:.2f}  (p={pval[name]:.3g}){stars}",
                xy=(float(hr[name]), y),
                xytext=(4, 5),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=8,
                color=color,
            )
        else:
            stars = ""
            if star:
                pv = pval[name]
                if   pv < 0.001: stars = " ***"
                elif pv < 0.01:  stars = " **"
                elif pv < 0.05:  stars = " *"

            ax.annotate(
                f"{hr[name]:.2f}  (p={pval[name]:.3g}){stars}",
                xy=(float(hr[name]), y),
                xytext=(4, 5),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=8,
                color="black",
            )

    # ── cosmetics ───────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.axvline(1, ls="--", lw=1.2, color="black")
    ax.grid(axis="x", ls=":", lw=0.6)

    ax.set_yticks(range(n))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()

    lo_min = lo.min()
    xmin   = 0.05 if lo_min < 0.05 else lo_min * 0.8
    ax.set_xlim(xmin, hi.max()*1.3)
    
    # xmin = max(0.3, lo.min() * 0.8)
    # xmax = hi.max() * 1.3
    # ax.set_xlim(xmin, xmax)

    ax.set_xlabel("Hazard ratio (log scale)")
    ax.set_title(title)
    fig.tight_layout()

    fig.savefig(out_stem.with_suffix(FIG_EXT), bbox_inches="tight")
    plt.close(fig)

def pattern_label(bits: str, snp_cols: list[str]) -> str:
    """
    Turn a '10100000' bit-string into a readable label.
      • 'None' if all zeros
      • otherwise join the SNP column names whose bit = 1 with ' + '
    """
    if set(bits) <= {"0"}:
        return "None"
    hits = [col for bit, col in zip(bits, snp_cols) if bit == "1"]
    return " + ".join(hits)

def informative_strata_mask(df: pd.DataFrame, snp_col: str) -> pd.Series:
    """Keep rows from cancer types that have at least one TRUE and one FALSE for snp_col."""
    by_cancer = df.groupby("cancer.type")[snp_col]
    has_true  = by_cancer.transform("any")
    has_false = (~by_cancer.transform("all"))
    return has_true & has_false
    
# ───────────────────────── main pipeline ───────────────────────────────
def main():
    print("Reading clinical and SNP tables …")
    clin = pd.read_csv(CLIN_CSV)
    snps = pd.read_csv(SNP_XLSX, sep='\t')
    snps = snps[snps['indiv_has_wgs'] == True]
    # snp_cols = [c for c in snps.columns if "chr5" in c.lower()]
    # snp_cols = [c for c in snp_cols if snps[c].sum() > 10]

    clin   = clin[clin["Sample.Type"] != "Solid Tissue Normal"]
    df   = clin.merge(snps, left_on="File.ID", right_on="run_id", how="inner")
    df["race"] = df["race"].replace({"Asian":"Other", "Black":"Other", "is_missing":"Other"})
    snp_cols = [c for c in df.columns if "chr5" in c.lower()]
    snp_cols = [c for c in snp_cols if snps[c].sum() > 10]
    # print(snp_cols)
    print(f"Found {len(snp_cols)} SNP flag columns:", ", ".join(snp_cols))

    for snp in snp_cols:

        # --- Simple overall counts for this SNP subset ---
        sub = df.dropna(subset=[snp]).copy()
        
        # booleans
        is_snp     = sub[snp].astype(bool)
        is_male    = sub["gender"].astype(str).str.lower().eq("male")
        is_white   = sub["race"].astype(str).str.lower().eq("white")
        
        counts = pd.DataFrame({
            "category": ["SNP_TRUE", "SNP_FALSE", "male", "nonMale", "white", "nonWhite"],
            "n": [
                is_snp.sum(),
                (~is_snp).sum(),
                is_male.sum(),
                (~is_male).sum(),
                is_white.sum(),
                (~is_white).sum(),
            ],
        })
        
        print(f"\n=== Overall counts for {snp} ===")
        print(counts.to_string(index=False))
        
        ct   = df.groupby("cancer.type")[snp].agg(["sum", "count"])
        ct["absent"] = ct["count"] - ct["sum"]
        ct.rename(columns={"sum": "present"}, inplace=True)
        ct = ct.sort_values("count", ascending=False)      # most samples first
        
        # ------------------ 1) Stacked 100-% bar plot -------------------------
        plt.figure(figsize=(0.25*len(ct) + 4, 4))
        bottom = np.zeros(len(ct))
        for status, color in [("present", "tab:blue"), ("absent", "lightgrey")]:
            plt.bar(ct.index, ct[status], bottom=bottom, label=status.capitalize(), color=color)
            bottom += ct[status]
        
        plt.ylabel("Number of samples")
        plt.title(f"{snp} – presence vs absence by cancer type")
        plt.xticks(rotation=90, ha="center")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(OUTROOT / "summary_plots" / f"{snp.replace(':','_')}_stacked_bar.svg")
        plt.close()
        
        # ------------------ 2) Two-tone heat-map (optional) -------------------
        heat2 = ct[["present","absent"]].T      # rows = status, cols = cancer types
        fig_h = 2.2; fig_w = 0.25*heat2.shape[1] + 3
        plt.figure(figsize=(fig_w, fig_h))
        sns.heatmap(
            heat2,
            cmap="Blues",#sns.color_palette(["lightgrey","tab:blue"], as_cmap=True),
            annot=True, fmt="d", linewidths=.3, linecolor="white",
            cbar=False
        )
        plt.yticks(rotation=0)
        plt.title(f"{snp} – counts in each cancer type")
        plt.xlabel("Cancer type")
        plt.tight_layout()
        plt.savefig(OUTROOT / "summary_plots" / f"{snp.replace(':','_')}_presence_absence_heat.svg")
        plt.close()
    
    # 1️⃣  overall counts ----------------------------------------------------
    totals = df[snp_cols].sum().sort_values(ascending=False)      # Series
    
    plt.figure(figsize=(0.25*len(totals) + 4, 4))
    sns.barplot(x=totals.index.str.replace("chr5:", ""),          # shorter labels
                y=totals.values,
                palette="ch:.25")
    plt.xticks(rotation=90, ha="center")
    plt.ylabel("Number of samples (TRUE)")
    plt.title("Overall prevalence of TERT-promoter SNPs")
    plt.tight_layout()
    plt.savefig(OUTROOT / "summary_plots" / "snp_counts_overall.svg")
    plt.close()
    
    # 2️⃣  per-cancer heat-map ----------------------------------------------
    heat_true = (
        df.groupby("cancer.type")[snp_cols]
          .sum()                                  # TRUE counts per tumour type
          .T                                      # rows = SNPs
          .loc[totals.index]                      # keep SNP ordering from the barplot
    )
    
    # NEW: one extra row = #samples with NONE of these SNPs in that cancer type
    none_row = (
        (~df[snp_cols].any(axis=1))               # boolean: no SNP across all given SNP columns
        .groupby(df["cancer.type"])
        .sum()                                    # count per cancer type
    )
    # Align the cancer-type columns to heat_true, convert to frame, row name = 'None'
    none_row = none_row.reindex(heat_true.columns).fillna(0).astype(int)
    heat_full = pd.concat([heat_true, none_row.to_frame(name="None").T], axis=0)
    # heat_full = pd.concat([heat_true, none_row.to_frame(name="None").T], axis=0)

    # ➜ Add a column that is the sum across cancers for each row (SNP or None)
    heat_full = heat_full.copy()
    heat_full["TOTAL"] = heat_full.sum(axis=1).astype(int)
    
    # Make sure the TOTAL column is the last one
    cols = [c for c in heat_full.columns if c != "TOTAL"] + ["TOTAL"]
    heat_full = heat_full[cols]
    # Figure size adapts to one extra row
    fig_h = 0.22 * heat_full.shape[0] + 2
    fig_w = 0.40 * heat_full.shape[1] + 3
    
    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(
        heat_full,
        cmap="Blues",
        linewidths=.3,
        linecolor="lightgrey",
        cbar_kws=dict(label="#TRUE samples"),
        square=False,
        annot=True, fmt=".0f"
    )
    plt.ylabel("SNP (last row = None/absent)")
    plt.xlabel("Cancer type")
    plt.title("SNP prevalence within each cancer type (+ None row)")
    plt.tight_layout()
    plt.savefig(OUTROOT / "summary_plots" / "snp_counts_by_cancer_plus_none.svg")
    plt.close()
    
    print("✓ SNP count visuals written to:",
          OUTROOT / "snp_counts_overall.svg",
          "and snp_counts_by_cancer.svg")

    t, e = "OS.time", "OS.event"
    if df[[t, e]].isna().any().any():
        raise RuntimeError("Missing OS.time / OS.event values.")

    # ───────────── 1. Per-SNP analyses ─────────────
    for snp in snp_cols:
        snp_dir = OUTROOT / "per_snp" / snp.replace(":", "_"); safe_mkdir(snp_dir)

        ## 1a pan-cancer
        sub = df.dropna(subset=[snp])

        ##
        # sub = sub.loc[informative_strata_mask(sub, snp)].copy()
        

        
        if sub[snp].nunique()==2 and sub[snp].value_counts().min()>=MIN_PER_GROUP:
            sub["group"] = sub[snp].map({True:"TRUE", False:"FALSE"})
            km_plot(sub, t, e, "group", f"{snp} (pan-cancer)", snp_dir/"km_pan")
            stat, p = logrank_p(sub, t, e, "group")
            pd.DataFrame({"stat":[stat],"p":[p]}).to_csv(snp_dir/"logrank_pan.tsv", sep="\t", index=False)

            ### NEW / CHANGED – adjusted Cox & forest
            covars = covars_init
            cidx, cph   = cox_summary(sub, t, e, "group", covars,
                                      strata="cancer.type", 
                                      out_tsv=snp_dir/"cox_pan.tsv")

            (snp_dir/"_cindex_pan.txt").write_text(f"{cidx:.3f}\n")

            forest_plot(snp_dir/"cox_pan.tsv", snp_dir/"forest_pan",
                        f"{snp} • adjusted Cox (pan)")

                        # === Extra figs (pan-cancer) ===
            if cph is not None:

                cox_scenarios_plot_marginal(cph, 
                                            base_df=sub[["group"] + covars_init + [t, e, "cancer.type"]].copy(), 
                                            group_col="group", 
                                            title=f"{snp} • model-based survival marginal (pan)", 
                                            out_png=snp_dir / "cox_model_survival_pan_marginal"
                                           )
                # Model-based S(t), TRUE vs FALSE — NOTE: include strata col
                cox_scenarios_plot_simple(
                    cph,
                    sub[["group"] + covars_init + [t, e, "cancer.type"]].copy(),
                    group_col="group",
                    title=f"{snp} • model-based survival (pan)",
                    out_png=snp_dir / "cox_model_survival_pan"
                )
                # Calibration at 3y/5y
                for yrs in (3.0, 5.0):
                    calibration_curve_at(
                        cph,
                        sub[["group"] + covars_init + [t, e, "cancer.type"]].copy(),
                        t, e, yrs,
                        title=f"{snp} • calibration {int(yrs)}y (pan)",
                        out_png=snp_dir / f"calibration_{int(yrs)}y_pan"
                    )
                    bs, ici = brier_ici_at(
                        cph,
                        sub[["group"] + covars_init + [t, e, "cancer.type"]].copy(),
                        t, e, yrs
                    )
                    (snp_dir / f"calibration_metrics_{int(yrs)}y.txt").write_text(
                        f"Brier={bs:.4f}\nICI={ici:.4f}\n"
                    )
                # LP distribution by group
                risk_score_violin(
                    cph,
                    sub[["group"] + covars_init + [t, e, "cancer.type"]].copy(),
                    group_col="group", t=t, e=e,
                    out_png=snp_dir / "risk_violin_pan"
                )
                # RMST bars at 5y (non-model based)
                res = rmst_bar_with_ci(dframe=sub.assign(group=sub["group"].astype(str)), 
                                       group_col="group", 
                                       t=t, 
                                       e=e, 
                                       years=5.0, 
                                       out_png=snp_dir / "rmst_5y_pan" )
                # res = rmst_bar_with_ci(
                #     dframe=sub.assign(group=sub["group"].astype(str)),
                #     group_col="group", t=t, e=e, years=5.0,
                #     out_png=snp_dir / "rmst_5y_pan",
                #     n_boot=0,
                #     seed=1,
                #     strata_col="cancer.type",     # <- good for pan-cancer; drop for per-cancer
                #     also_permutation=False        # set True if you want a permutation p-value too
                # )
            
            # KM grid for top cancers (doesn't need cph)
            km_grid_by_cancer(df.copy(), snp, t, e,
                              out_png=snp_dir / "km_grid_by_cancer",
                              max_panels=12, min_per=MIN_PER_CANCER)
            Xsub = pd.get_dummies(sub[["group"] + covars_init], drop_first=True)

            # 2. add survival columns
            Xsub[t] = sub[t].values
            Xsub[e] = sub[e].values
            
            # 3. add every stratum column the model used
            if cph.strata:                                 # empty tuple / None → False
                strata_cols = list(cph.strata) if isinstance(cph.strata, (list, tuple)) else [cph.strata]
                for s in strata_cols:
                    Xsub[s] = sub[s].values
            
            # 4. PH test
            from lifelines.statistics import proportional_hazard_test
            ph_test = proportional_hazard_test(cph, Xsub, time_transform="rank")
            print(snp)
            print(ph_test.summary)
            ph_test.summary.to_csv(snp_dir / "ph_test.tsv", sep="\t")
            cph_inter = cox_cancer_interaction(
                sub,                    # the same subset you already built
                t, e,
                snp_flag=snp,           # the original boolean column
                base_covars=covars_init,
                out_tsv=snp_dir / "cox_inter.tsv",
            )
            
            if cph_inter is not None:
                forest_plot(
                    snp_dir / "cox_inter.tsv",
                    snp_dir / "forest_inter",
                    f"{snp} • cancer covariate + SNP×cancer interaction",
                )
            
        ## 1b within each cancer
        for ctype, d in df.groupby("cancer.type", observed=True):
            d = d.dropna(subset=[snp])
            if d[snp].nunique()==2 and d[snp].value_counts().min()>=MIN_PER_CANCER:
                cdir = snp_dir/"by_cancer"/ctype; safe_mkdir(cdir)
                d["group"] = d[snp].map({True:"TRUE", False:"FALSE"})
                km_plot(d, t, e, "group", f"{snp} · {ctype}", cdir/"km")
                stat,p = logrank_p(d, t, e, "group")
                pd.DataFrame({"stat":[stat],"p":[p]}).to_csv(cdir/"logrank.tsv", sep="\t", index=False)

                covars_cancer_specific = covars_init
                cidx, cph   = cox_summary(d, t, e, "group", covars_cancer_specific,
                                     strata=None, 
                                     out_tsv=cdir/"cox.tsv")
                (cdir/"_cindex.txt").write_text(f"{cidx:.3f}\n")
                forest_plot(cdir/"cox.tsv", cdir/"forest",
                            f"{snp} • {ctype} • Cox")

                

    # ───────────── 2. Any-SNP indicator ─────────────
    df["ANY_SNP"] = df[snp_cols].any(axis=1)
    any_dir = OUTROOT/"any_snp"; safe_mkdir(any_dir)

    def indicator_analysis(dframe, tag, out_dir):
        if dframe["ANY_SNP"].value_counts().min() < MIN_PER_GROUP: return
        dframe["group"] = dframe["ANY_SNP"].map({True:"True_SNP", False:"False_SNP"})
        km_plot(dframe, t, e, "group", f"Any SNP • {tag}", out_dir/"km")
        stat,p = logrank_p(dframe, t, e, "group")
        pd.DataFrame({"stat":[stat],"p":[p]}).to_csv(out_dir/"logrank.tsv", sep="\t", index=False)

        covars = covars_init
        strata = "cancer.type" if tag=="pan-cancer" else None
        cidx, cph   = cox_summary(dframe, t, e, "group", covars,
                             strata= strata, 
                             out_tsv=out_dir/"cox.tsv")
        (out_dir/"_cindex.txt").write_text(f"{cidx:.3f}\n")
        forest_plot(out_dir/"cox.tsv", out_dir/"forest",
                    f"Any SNP • {tag} • Cox")

    indicator_analysis(df.copy(), "pan-cancer", any_dir)
    for c, d in df.groupby("cancer.type", observed=True):
        if d["ANY_SNP"].nunique()==2 and d["ANY_SNP"].value_counts().min()>=MIN_PER_GROUP:
            cdir = any_dir/"by_cancer"/c; safe_mkdir(cdir)
            indicator_analysis(d.copy(), c, cdir)

    # ───────────── 3. SNP-pattern combos ───────────────────────────────
    pat_dir = OUTROOT / "pattern_combo"; safe_mkdir(pat_dir)
    
    df["SNP_PATTERN"] = df[snp_cols].astype(int).astype(str).agg("".join, axis=1)
    counts = df["SNP_PATTERN"].value_counts()
    counts.to_csv(pat_dir / "pattern_counts.tsv", sep="\t")
    
    zero = "0" * len(snp_cols)
    keep = counts[(counts >= MIN_PER_GROUP) | (counts.index == zero)].index
    dpat = df[df["SNP_PATTERN"].isin(keep)].copy()
    
    # ── NEW: rename levels to readable labels ───────────────────────────
    label_map = {bits: pattern_label(bits, snp_cols) for bits in keep}
    dpat["PAT_LABEL"] = dpat["SNP_PATTERN"].map(label_map)
    
    # Make 'None' the reference level by ordering categories
    cats = sorted(label_map.values(), key=lambda s: (s != "None", s))
    dpat["PAT_LABEL"] = pd.Categorical(dpat["PAT_LABEL"], categories=cats)
    
    if dpat["PAT_LABEL"].nunique() > 1:
        km_plot(dpat, t, e, "PAT_LABEL",
                "SNP combinations (pan-cancer)", pat_dir / "km")
        stat, p = logrank_p(dpat, t, e, "PAT_LABEL")
        pd.DataFrame({"stat": [stat], "p": [p]}).to_csv(
            pat_dir / "logrank.tsv", sep="\t", index=False
        )
    
        covars = covars_init
        cidx, cph = cox_summary(
            dpat, t, e,
            exposure="PAT_LABEL",
            adjust=covars,
            strata="cancer.type",
            out_tsv=pat_dir / "cox.tsv"
        )
        (pat_dir / "_cindex.txt").write_text(f"{cidx:.3f}\n")
        forest_plot(
            pat_dir / "cox.tsv",
            pat_dir / "forest",
            "SNP combo • pan-cancer • Cox"
        )

    print("✓ All analyses complete ➜", OUTROOT)









# ───────────────────────── extra figure helpers (strata-safe) ─────────────────────────
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

YEARS = 365.25

def _ensure_strata_columns(cph, Xcov: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    """
    If the Cox model was fit with strata (e.g., strata='cancer.type'), lifelines expects
    those raw columns to be present in X for predict_* calls. This re-attaches them.
    """
    if not getattr(cph, "strata", None):
        return Xcov
    strata_cols = list(cph.strata) if isinstance(cph.strata, (list, tuple)) else [cph.strata]
    Xcov = Xcov.copy()
    for s in strata_cols:
        if s in Xcov.columns:
            continue
        if s in base_df.columns and base_df[s].notna().any():
            # Use the most common observed level from base_df
            try:
                val = base_df[s].mode(dropna=True).iloc[0]
            except Exception:
                val = base_df[s].dropna().iloc[0]
        else:
            # Fallback: take any level from baseline_cumulative_hazard_ index (if present)
            bch = getattr(cph, "baseline_cumulative_hazard_", None)
            if hasattr(bch, "index") and s in getattr(bch.index, "names", []):
                level_pos = bch.index.names.index(s)
                val = bch.index.levels[level_pos][0]
            else:
                val = None
        Xcov[s] = val
    return Xcov

def _pred_surv_at(cph, X, t_days):
    """Predict S(t) at a single time t_days for all rows of X."""
    sf = cph.predict_survival_function(X, times=[t_days])
    return sf.iloc[0, :].values

def _rmst_from_km(time, event, tau):
    """Restricted mean survival time (RMST) up to tau from KM."""
    km = KaplanMeierFitter().fit(time, event)
    timeline = km.survival_function_.index.values
    surv = km.survival_function_["KM_estimate"].values
    # ensure [0, tau] coverage
    if timeline[0] > 0:
        timeline = np.insert(timeline, 0, 0.0)
        surv = np.insert(surv, 0, 1.0)
    if timeline[-1] < tau:
        timeline = np.append(timeline, tau)
        surv = np.append(surv, surv[-1])
    timeline = np.clip(timeline, 0, tau)
    return np.trapz(surv, timeline)

def km_grid_by_cancer(d, snp_col, t, e, out_png, max_panels=12, min_per=10):
    """Small-multiples KM by top cancers where both groups meet min_per."""
    ok = []
    for ctype, sub in d.groupby("cancer.type", observed=True):
        sub = sub.dropna(subset=[snp_col])
        vc = sub[snp_col].value_counts()
        if (len(vc) == 2) and (vc.min() >= min_per):
            ok.append((ctype, len(sub)))
    ok = sorted(ok, key=lambda x: x[1], reverse=True)[:max_panels]
    if not ok:
        return
    n = len(ok); ncols = min(4, n); nrows = int(np.ceil(n / ncols))
    sns.set_style("whitegrid"); sns.set_context("talk", 0.8)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.3*nrows), squeeze=False)
    kmf = KaplanMeierFitter()
    from lifelines.statistics import logrank_test
    for i, (ctype, _) in enumerate(ok):
        ax = axes[i//ncols, i%ncols]
        sub = d[d["cancer.type"] == ctype].dropna(subset=[snp_col]).copy()
        sub["group"] = sub[snp_col].map({True: "TRUE", False: "FALSE"})
        for g, dd in sub.groupby("group", observed=True, sort=False):
            kmf.fit(dd[t], dd[e], label=str(g))
            kmf.plot_survival_function(ax=ax, ci_show=True)
        g1 = sub[sub["group"] == "TRUE"]; g0 = sub[sub["group"] == "FALSE"]
        lr = logrank_test(g1[t], g0[t], g1[e], g0[e])
        ax.set_title(f"{ctype}  (p={lr.p_value:.2g})")
        ax.set_xlabel("Days"); ax.set_ylabel("S(t)"); ax.legend(frameon=False)
    # hide unused panels
    for j in range(i+1, nrows*ncols):
        axes[j//ncols, j%ncols].axis("off")
    fig.suptitle(f"{snp_col} • KM by cancer", y=1.02)
    fig.tight_layout()
    fig.savefig(Path(out_png).with_suffix(FIG_EXT), dpi=120, bbox_inches="tight")
    plt.close(fig)

def cox_scenarios_plot_simple(cph, base_df, group_col, title, out_png):
    """
    Model-based survival curves for two scenarios (group TRUE vs FALSE),
    other covariates fixed at typical values from base_df.
    NOTE: base_df must include any strata columns used in the fit (e.g., 'cancer.type').
    """
    df = base_df.copy()
    # build two scenarios
    typical = {}
    skip = {group_col, "OS.time", "OS.event"}
    for c in df.columns:
        if c in skip: 
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            typical[c] = np.nanmedian(pd.to_numeric(df[c], errors="coerce"))
        else:
            if df[c].notna().any():
                typical[c] = df[c].mode(dropna=True).iloc[0]
    scenarios = [{group_col: "TRUE", **typical}, {group_col: "FALSE", **typical}]
    cov_df = pd.DataFrame(scenarios, index=[f"{group_col}=TRUE", f"{group_col}=FALSE"])

    # dummy & align to model params
    Xcov = pd.get_dummies(cov_df, drop_first=True)
    for col in cph.params_.index:
        if col not in Xcov.columns:
            Xcov[col] = 0.0
    Xcov = Xcov[cph.params_.index]
    # ensure strata columns present
    Xcov = _ensure_strata_columns(cph, Xcov, base_df)

    sns.set_style("whitegrid"); sns.set_context("talk", 0.8)
    fig, ax = plt.subplots(figsize=(6, 4))
    times = np.linspace(0, float(df["OS.time"].max()), 150)
    sf = cph.predict_survival_function(Xcov, times=times)
    
    for scen in sf.columns:               # scen names = cov_df index
        ax.plot(sf.index, sf[scen], label=scen)
        
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Model-based S(t)")
    ax.set_title(title); ax.legend(title="Scenario", frameon=False)
    fig.tight_layout(); fig.savefig(Path(out_png).with_suffix(FIG_EXT), dpi=120); plt.close(fig)

def calibration_curve_at(
    cph,
    dframe: pd.DataFrame,
    t: str,
    e: str,
    horizon_years: float,
    title: str,
    out_png: Path,
    bins: int = 10,
    min_bin_n: int = 15,
):
    """
    Risk calibration at a fixed horizon (e.g., 3y/5y) using quantile bins built with
    numpy (no pandas Categorical, no qcut). Safe for stratified Cox models.

    • cph: lifelines CoxPHFitter (possibly with strata)
    • dframe: must include predictors used in the model (+ strata if any) and [t, e]
    """
    df = dframe.copy()

    # ---- build model design matrix (exclude time/event), align to model ----
    drop_cols = [t, e]
    X = pd.get_dummies(df.drop(columns=drop_cols, errors="ignore"), drop_first=True)

    # add missing coeff columns with 0; order to model param index
    for col in cph.params_.index:
        if col not in X.columns:
            X[col] = 0.0
    X = X[cph.params_.index]

    # ensure strata columns are present for predict_*
    X = _ensure_strata_columns(cph, X, df)

    # ---- predict risk at horizon ----
    YEARS = 365.25
    tau = float(horizon_years) * YEARS
    pred_S = _pred_surv_at(cph, X, tau)
    pred_R = pd.Series(1.0 - pred_S, index=df.index, name="predR")

    # degenerate predictions → nothing to plot
    if pred_R.nunique() < 3:
        return

    # ---- build quantile edges robustly (unique & strictly increasing) ----
    q = np.linspace(0, 1, bins + 1)
    edges = np.quantile(pred_R.values, q, method="linear")
    edges = np.unique(edges)  # drop duplicate edges if predictions cluster
    if edges.size < 4:
        # not enough distinct edges to form ≥3 bins
        return

    # Assign integer bins with np.digitize; bins = 0..len(edges)-2
    # right=True makes the bins closed on the right.
    bin_idx = np.digitize(pred_R.values, edges[1:-1], right=True)
    df["_calib_bin"] = bin_idx

    # ---- observed risk per bin via KM(tau) ----
    kmf = KaplanMeierFitter()
    mids, obs = [], []
    for b in range(edges.size - 1):
        sub = df[df["_calib_bin"] == b]
        if len(sub) < min_bin_n:
            continue
        kmf.fit(sub[t], sub[e])
        obs_R = 1.0 - float(kmf.survival_function_at_times(tau))
        mid = (edges[b] + edges[b + 1]) / 2.0
        mids.append(mid); obs.append(obs_R)

    if len(mids) < 3:
        return

    # ---- plot ----
    sns.set_style("whitegrid"); sns.set_context("talk", 0.8)
    fig, ax = plt.subplots(figsize=(5.8, 5.8), constrained_layout=False)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="black", label="ideal")
    ax.scatter(mids, obs, s=30)
    ax.set_xlabel(f"Predicted risk at {int(horizon_years)}y")
    ax.set_ylabel("Observed risk (KM)")
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    plt.subplots_adjust(top=0.88)  # extra headroom for title
    fig.savefig(Path(out_png).with_suffix(FIG_EXT), dpi=120)
    plt.close(fig)
    
def risk_score_violin(cph, dframe, group_col, t, e, out_png):
    """
    Distribution of Cox risk scores by group (TRUE/FALSE).
    Uses log(partial hazard) as the linear predictor so column alignment
    and strata are handled by lifelines.
    """
    df = dframe.copy()

    # --- build design matrix aligned to the fitted model ---
    drop_cols = [t, e]
    X0 = pd.get_dummies(df.drop(columns=drop_cols, errors="ignore"), drop_first=True)
    for col in cph.params_.index:
        if col not in X0.columns:
            X0[col] = 0.0
    X0 = X0[cph.params_.index]

    # add back strata columns for predict_* calls
    X = _ensure_strata_columns(cph, X0.copy(), df)

    # --- risk scores (linear predictor up to an additive constant) ---
    # partial hazard = exp(Xβ); so log(partial hazard) = Xβ
    ph = cph.predict_partial_hazard(X).astype(float).values.reshape(-1)
    lp = np.log(ph, where=np.isfinite(ph))
    # guard against degenerate cases
    if np.all(~np.isfinite(lp)):
        return

    dd = pd.DataFrame({
        group_col: df[group_col].astype(str),
        "lp": lp,
        "event": df[e].astype(int),
    })

    # --- plot ---
    sns.set_style("whitegrid"); sns.set_context("talk", 0.9)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    sns.violinplot(data=dd, x=group_col, y="lp", inner="box", cut=0, ax=ax)

    # faint event rugs (points) per group
    order = sorted(dd[group_col].unique())
    x_pos = {g:i for i,g in enumerate(order)}
    for g, sub in dd.groupby(group_col, observed=True):
        evt = sub.loc[sub["event"] == 1, "lp"]
        if len(evt):
            ax.scatter([x_pos[g]]*len(evt), evt, s=6, alpha=0.35)

    ax.set_title(f"Risk score by {group_col}")
    ax.set_ylabel("log(partial hazard)")
    fig.tight_layout()
    plt.subplots_adjust(top=0.88)  # give the title more room (reduces tight_layout warnings)
    fig.savefig(Path(out_png).with_suffix(FIG_EXT), dpi=120)
    plt.close(fig)

def rmst_bar_with_ci(dframe, group_col, t, e, years, out_png, n_boot=300, seed=1):
    """RMST(τ) bars with bootstrap 95% CI; τ = years."""
    df = dframe.copy()
    df = df[df[group_col].isin(["TRUE", "FALSE"])].copy()
    tau = years * YEARS
    rng = np.random.default_rng(seed)
    stats = {}
    for g, sub in df.groupby(group_col, observed=True):
        r = _rmst_from_km(sub[t].values, sub[e].values, tau)
        bs = []
        idx = np.arange(len(sub))
        for _ in range(n_boot):
            ii = rng.choice(idx, size=len(idx), replace=True)
            bs.append(_rmst_from_km(sub[t].values[ii], sub[e].values[ii], tau))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        stats[g] = (r, lo, hi)
    order = ["FALSE", "TRUE"] if "FALSE" in stats and "TRUE" in stats else list(stats.keys())
    vals = [stats[g][0] for g in order]; los = [stats[g][1] for g in order]; his = [stats[g][2] for g in order]
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(order))
    ax.bar(x, vals, yerr=[np.array(vals)-np.array(los), np.array(his)-np.array(vals)], capsize=4, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel(f"RMST ≤ {years:.0f} y (days)")
    ax.set_title(f"Restricted mean survival time by {group_col}")
    fig.tight_layout(); fig.savefig(Path(out_png).with_suffix(FIG_EXT), dpi=120); plt.close(fig)

def per_cancer_hr_lollipop(cox_dir_by_cancer, out_png, title):
    """Summarize per-cancer Cox (by_cancer/cox.tsv) as a lollipop HR plot."""
    rows = []
    for cdir in sorted(Path(cox_dir_by_cancer).glob("*/cox.tsv")):
        ctype = cdir.parent.name
        summ = pd.read_csv(cdir, sep="\t", index_col=0)
        match = summ[summ.index.str.startswith("group")]
        if match.empty:
            continue
        hr = float(match["exp(coef)"].iloc[0])
        lo = float(match["exp(coef) lower 95%"].iloc[0])
        hi = float(match["exp(coef) upper 95%"].iloc[0])
        p  = float(match["p"].iloc[0])
        rows.append((ctype, hr, lo, hi, p))
    if not rows:
        return
    tab = pd.DataFrame(rows, columns=["cancer", "hr", "lo", "hi", "p"]).sort_values("hr")
    fig, ax = plt.subplots(figsize=(6, 0.45*len(tab)+1.5))
    y = np.arange(len(tab))
    ax.hlines(y, tab["lo"], tab["hi"], lw=2)
    ax.plot(tab["hr"], y, "o")
    ax.axvline(1, ls="--", color="black", lw=1)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(tab["cancer"])
    ax.set_xlabel("Hazard ratio (group TRUE vs FALSE)")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(Path(out_png).with_suffix(FIG_EXT), dpi=120); plt.close(fig)
# ─────────────────────── end extra helpers ───────────────────────
# put near FIG_EXT
FIG_EXT = ".svg"

def savefig_multi(fig, out_stem: Path, exts=(".svg", ".png"), dpi=150):
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in exts:
        fig.savefig(out_stem.with_suffix(ext), dpi=dpi, bbox_inches="tight")

def brier_ici_at(cph, dframe, t, e, horizon_years):
    """
    IPCW Brier score and ICI at a fixed horizon (e.g., 3y/5y), strata-safe.

    Fixes:
      • Only test numeric design-matrix columns for finiteness (strata may be strings)
      • Explicitly drop rows with NA/inf in X, t, e, or missing strata
      • Ensures predictions, weights, and outcomes all have matching length
    """
    df = dframe.copy()
    YEARS = 365.25
    tau = float(horizon_years) * YEARS

    # --- Build design matrix aligned to the model (no time/event cols) ---
    X = pd.get_dummies(df.drop(columns=[t, e], errors="ignore"), drop_first=True)
    for col in cph.params_.index:
        if col not in X.columns:
            X[col] = 0.0
    X = X[cph.params_.index]

    # Add back strata columns that lifelines needs for predict_*
    X = _ensure_strata_columns(cph, X, df)

    # --- Row-level keep mask ---
    # 1) numeric predictors must be finite
    num_X = X.select_dtypes(include=[np.number, "bool"]).astype(float)
    mask_num = np.isfinite(num_X.to_numpy()).all(axis=1)

    # 2) time/event must be present and numeric
    times = pd.to_numeric(df[t], errors="coerce").to_numpy()
    events = pd.to_numeric(df[e], errors="coerce").to_numpy()
    mask_te = np.isfinite(times) & np.isfinite(events)

    # 3) strata columns (if any) must be non-null
    mask_strata = np.ones(len(df), dtype=bool)
    if getattr(cph, "strata", None):
        strata_cols = list(cph.strata) if isinstance(cph.strata, (list, tuple)) else [cph.strata]
        for s in strata_cols:
            if s in df.columns:
                mask_strata &= df[s].notna().to_numpy()

    keep = mask_num & mask_te & mask_strata
    if keep.sum() < 10:
        return np.nan, np.nan  # not enough rows to evaluate

    # Filter everything to the same rows
    X = X.loc[keep]
    times = times[keep].astype(float)
    events = events[keep].astype(int)
    df = df.loc[keep].copy()

    # --- Predictions at tau ---
    p_surv = _pred_surv_at(cph, X, tau)   # length == len(df)
    p = 1.0 - p_surv                      # predicted risk

    # --- IPCW weights using KM of censoring ---
    # Censoring indicator = 1 - event
    kmc = KaplanMeierFitter().fit(times, 1 - events)
    G_t = np.clip(kmc.predict(times), 1e-8, 1.0)
    G_tau = np.clip(float(kmc.survival_function_at_times(tau)), 1e-8, 1.0)

    # Observed event by tau
    y = ((events == 1) & (times <= tau)).astype(float)

    at_tau = (times >= tau).astype(float)
    event_before_tau = ((times < tau) & (events == 1)).astype(float)
    w = at_tau / G_tau + event_before_tau / G_t

    # --- Metrics ---
    bs = float(np.mean(w * (y - p) ** 2))
    ici = float(np.mean(w * np.abs(y - p)))
    return bs, ici


# def rmst_bar_with_ci_and_test(
#     dframe,
#     group_col,
#     t,
#     e,
#     years,
#     out_png,
#     n_boot=1000,
#     seed=1,
#     strata_col=None,          # e.g., "cancer.type" to stratify resampling
#     also_permutation=False,   # set True to add a label-permutation p-value
#     n_perm=2000
# ):
#     """
#     RMST(τ) by group with bootstrap 95% CI and a bootstrap p-value for ΔRMST.
#     Optionally, compute a permutation p-value by shuffling group labels.

#     Returns dict with point estimates, CIs, and p-values.
#     """
#     df = dframe.copy()
#     df = df[df[group_col].isin(["TRUE", "FALSE"])].copy()
#     tau = years * YEARS
#     rng = np.random.default_rng(seed)

#     # --- point estimates by group ---
#     stats = {}
#     for g, sub in df.groupby(group_col, observed=True):
#         r = _rmst_from_km(sub[t].values, sub[e].values, tau)
#         stats[g] = {"rmst": r, "n": len(sub)}
#     if not {"TRUE", "FALSE"} <= set(stats):
#         return None  # need both groups

#     # --- bootstrap helper (optionally stratified) ---
#     def _boot_once():
#         if strata_col and strata_col in df.columns:
#             # resample within each (group, stratum)
#             boot_rows = []
#             for (g, s), sub in df.groupby([group_col, strata_col], observed=True):
#                 if len(sub) == 0:
#                     continue
#                 ii = rng.choice(sub.index.values, size=len(sub), replace=True)
#                 boot_rows.append(df.loc[ii])
#             bdf = pd.concat(boot_rows, axis=0)
#         else:
#             # resample within each group
#             parts = []
#             for g, sub in df.groupby(group_col, observed=True):
#                 ii = rng.choice(sub.index.values, size=len(sub), replace=True)
#                 parts.append(df.loc[ii])
#             bdf = pd.concat(parts, axis=0)

#         # RMST per group in the bootstrap sample
#         out = {}
#         for g, sub in bdf.groupby(group_col, observed=True):
#             r = _rmst_from_km(sub[t].values, sub[e].values, tau)
#             out[g] = r
#         # ensure both present
#         if "TRUE" in out and "FALSE" in out:
#             return out["TRUE"] - out["FALSE"]
#         return np.nan

#     # --- bootstrap ΔRMST and CI ---
#     boot = []
#     for _ in range(n_boot):
#         d = _boot_once()
#         if np.isfinite(d):
#             boot.append(d)
#     boot = np.array(boot)
#     if boot.size < max(50, n_boot // 5):
#         # not enough stable bootstrap reps
#         return None

#     delta = stats["TRUE"]["rmst"] - stats["FALSE"]["rmst"]
#     lo, hi = np.percentile(boot, [2.5, 97.5])
#     # two-sided bootstrap p-value: proportion of |Δ*| >= |Δ_obs|
#     p_boot = (np.sum(np.abs(boot) >= abs(delta)) + 1.0) / (boot.size + 1.0)

#     # --- optional permutation p-value (shuffle labels) ---
#     p_perm = None
#     if also_permutation:
#         perm_d = []
#         for _ in range(n_perm):
#             shuf = df.copy()
#             shuf[group_col] = rng.permutation(shuf[group_col].values)
#             out = {}
#             for g, sub in shuf.groupby(group_col, observed=True):
#                 r = _rmst_from_km(sub[t].values, sub[e].values, tau)
#                 out[g] = r
#             if "TRUE" in out and "FALSE" in out:
#                 perm_d.append(out["TRUE"] - out["FALSE"])
#         perm_d = np.asarray(perm_d, dtype=float)
#         if perm_d.size > 20:
#             p_perm = (np.sum(np.abs(perm_d) >= abs(delta)) + 1.0) / (perm_d.size + 1.0)

#     # --- plot bars with CIs (pointwise by group) ---
#     order = ["FALSE", "TRUE"]
#     vals = [stats[g]["rmst"] for g in order]

#     # per-group bootstrap CIs re-using the bootstrap samples:
#     # compute RMST per group across bootstrap draws for intervals
#     # (this is a little heavier; for speed we’ll approximate using normal CI from boot deltas for annotation,
#     #  and keep your original per-group CI code if you prefer exact per-group intervals.)
#     fig, ax = plt.subplots(figsize=(5.2, 4.2))
#     x = np.arange(2)
#     # reuse your earlier per-group CI code for clarity
#     # quick per-group CI via bootstrap:
#     def _boot_group(g):
#         bvals = []
#         for _ in range(min(n_boot, 600)):  # cap for speed
#             if strata_col and strata_col in df.columns:
#                 parts = []
#                 for s, sub in df[df[group_col] == g].groupby(strata_col, observed=True):
#                     ii = rng.choice(sub.index.values, size=len(sub), replace=True)
#                     parts.append(df.loc[ii])
#                 bdf = pd.concat(parts, axis=0)
#             else:
#                 sub = df[df[group_col] == g]
#                 ii = rng.choice(sub.index.values, size=len(sub), replace=True)
#                 bdf = df.loc[ii]
#             bvals.append(_rmst_from_km(bdf[t].values, bdf[e].values, tau))
#         return np.percentile(bvals, [2.5, 97.5])

#     lo_gF, hi_gF = _boot_group("FALSE")
#     lo_gT, hi_gT = _boot_group("TRUE")
#     los = [lo_gF, lo_gT]; his = [hi_gF, hi_gT]
#     yerr = [np.array(vals) - np.array(los), np.array(his) - np.array(vals)]
#     ax.bar(x, vals, yerr=yerr, capsize=4, alpha=0.9)
#     ax.set_xticks(x); ax.set_xticklabels(order)
#     ax.set_ylabel(f"RMST ≤ {years:.0f} y (days)")

#     # annotate Δ and p
#     subtitle = f"ΔRMST (TRUE−FALSE) = {delta:.1f} days;  p_boot = {p_boot:.3g}"
#     if p_perm is not None:
#         subtitle += f";  p_perm = {p_perm:.3g}"
#     ax.set_title(f"Restricted mean survival time by {group_col}\n{subtitle}")

#     fig.tight_layout()
#     fig.savefig(Path(out_png).with_suffix(FIG_EXT), dpi=120, bbox_inches="tight")
#     plt.close(fig)

#     return {
#         "rmst_TRUE": stats["TRUE"]["rmst"],
#         "rmst_FALSE": stats["FALSE"]["rmst"],
#         "delta": float(delta),
#         "delta_CI": (float(lo), float(hi)),
#         "p_boot": float(p_boot),
#         "p_perm": (None if p_perm is None else float(p_perm)),
#         "n_TRUE": int(stats["TRUE"]["n"]),
#         "n_FALSE": int(stats["FALSE"]["n"]),
#         "tau_days": float(tau),
#     }

def cox_scenarios_plot_marginal(cph, base_df, group_col, title, out_png,
                                horizon_quantile=0.90, n_grid=400):
    """
    Direct-adjusted survival: for every subject in base_df, predict survival twice
    (group=TRUE and group=FALSE), then average. Works with strata.
    """
    df = base_df.copy()

    # two counterfactual copies
    df_true  = df.copy();  df_true[group_col]  = "TRUE"
    df_false = df.copy();  df_false[group_col] = "FALSE"

    # build design matrices aligned to the fitted model
    def _design(d):
        X = pd.get_dummies(d.drop(columns=["OS.time","OS.event"], errors="ignore"),
                           drop_first=True)
        for col in cph.params_.index:
            if col not in X.columns:
                X[col] = 0.0
        X = X[cph.params_.index]
        return _ensure_strata_columns(cph, X, d)

    X_true  = _design(df_true)
    X_false = _design(df_false)

    # union of baseline time points across strata, then cap to 90th pct of follow-up
    if getattr(cph, "baseline_cumulative_hazard_", None) is not None:
        bch = cph.baseline_cumulative_hazard_
        if isinstance(bch.index, pd.MultiIndex):
            t0 = np.unique(bch.index.get_level_values(-1).values)
        else:
            t0 = bch.index.values
    else:
        t0 = df["OS.time"].values
    t_cap = np.nanquantile(df["OS.time"], horizon_quantile)
    t_dense = np.linspace(0.0, float(t_cap), n_grid)

    # predict survival and average across subjects for each scenario
    Sf_true  = cph.predict_survival_function(X_true,  times=t0).T.mean(axis=0)
    Sf_false = cph.predict_survival_function(X_false, times=t0).T.mean(axis=0)

    # re-sample onto dense grid for a nicer plot (step-hold)
    last_true  = float(Sf_true.iloc[-1])
    last_false = float(Sf_false.iloc[-1])
    
    S_true  = np.interp(t_dense, t0,  Sf_true.values,  left=1.0, right=last_true)
    S_false = np.interp(t_dense, t0,  Sf_false.values, left=1.0, right=last_false)
    
    sns.set_style("whitegrid"); sns.set_context("talk", 0.85)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(t_dense, S_true,  label="group=TRUE")
    ax.plot(t_dense, S_false, label="group=FALSE")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Model-based S(t)")
    ax.set_title(title)
    ax.legend(title="Scenario", frameon=False)
    fig.tight_layout()
    fig.savefig(Path(out_png).with_suffix(FIG_EXT), dpi=120, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
