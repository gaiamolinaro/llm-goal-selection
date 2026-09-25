"""
Helper functions for analyzing human behavioral data from Holton et al. 2024.

Provides data restructuring, loading, logistic fitting, and plotting utilities
for replicating Figures 2b and 2c from the paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats


ITEM_NAMES = {1: "fish", 2: "octopus", 3: "crab"}


# ---------------------------------------------------------------------------
# Data restructuring
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_family(model):
    family = ""
    if model.startswith("gpt-"):
        family = "openai"
    elif model.startswith("gemini"):
        family = "google"
    elif model.startswith("claude"):
        family = "anthropic"
    elif model.startswith("centaur"):
        family = "centaur"
    elif model.startswith("qwen"):
        family = "qwen"
    return family



def prepare_learn_df(df: pd.DataFrame, subj_col: str = "run") -> pd.DataFrame:
    """Clean an LLM DataFrame into learn_df format.

    - Renames *subj_col* to ``participant``.
    - Drops every participant that has **any** NaN in the ``choice`` column.
    - Renumbers participant IDs to 0 .. N-1.
    """
    df = df.copy()
    if subj_col in df.columns and subj_col != "participant":
        df.rename(columns={subj_col: "participant"}, inplace=True)

    bad = df.loc[df["choice"].isna(), "participant"].unique()
    df = df[~df["participant"].isin(bad)].reset_index(drop=True)

    old_ids = sorted(df["participant"].unique())
    id_map = {old: new for new, old in enumerate(old_ids)}
    df["participant"] = df["participant"].map(id_map)
    return df


# ---------------------------------------------------------------------------
# Tree-search sampler (Monte Carlo)
# ---------------------------------------------------------------------------
# Parameters from extend_configs.py / the Holton generative model.

# Parameters match Holton et al. 2024 (Methods, "Schedules"), calibrated
# against the paper's released data:
#   - Gaussian walk with std = 0.8 (paper says "sigma^2 = 0.8" but the
#     std measured from released data is ~0.87, consistent with 0.8 being
#     used as std rather than variance).
#   - On each trial, each option independently has p_jump_up=p_jump_lo=0.1
#     of jumping. Jumps *replace* the offer with starting_mean +/-
#     uniform(jump_min, jump_max).
SAMPLER_PARAMS = {
    "var": 0.8,
    "xmin": -6.0,
    "xmax": 18.0,
    "starting_mean": 6.0,
    "p_jump_up": 0.1,
    "p_jump_lo": 0.1,
    "jump_min": 3.0,   # paper: "between 3 and 9 points higher or lower"
    "jump_max": 9.0,
    "cut_off": 100,
}

ITEM_TO_IDX = {"fish": 0, "octopus": 1, "crab": 2}


def _walk_with_jumps(current_mag, n_steps, params, n_itr, rng):
    """Simulate *n_itr* future trajectories for one item (vectorised).

    Either a Gaussian walk step or a jump that *resets* the offer to
    starting_mean +/- Uniform(jump_min, jump_max), per the paper.
    """
    var = params["var"]
    xmin, xmax = params["xmin"], params["xmax"]
    sm = params["starting_mean"]
    p_up, p_lo = params["p_jump_up"], params["p_jump_lo"]
    p_jump = p_up + p_lo
    jmin, jmax = params["jump_min"], params["jump_max"]

    noise = rng.normal(0, var, (n_itr, n_steps))
    rolls = rng.random((n_itr, n_steps))
    jump_mag = rng.uniform(jmin, jmax, (n_itr, n_steps))

    is_jump_up = rolls < p_up
    is_jump_dn = (rolls >= p_up) & (rolls < p_jump)

    mags = np.empty((n_itr, n_steps))
    cur = np.full(n_itr, current_mag)
    for t in range(n_steps):
        rw = cur + noise[:, t]
        up = sm + jump_mag[:, t]
        dn = sm - jump_mag[:, t]
        cur = np.where(is_jump_up[:, t], up,
                       np.where(is_jump_dn[:, t], dn, rw))
        cur = np.clip(cur, xmin, xmax)
        mags[:, t] = cur
    return mags  # (n_itr, n_steps)


def _completion_time(future_mags, accumulated, goal_size):
    """Trials to fill the net, counting the current trial (vectorised).

    ``accumulated`` already includes the current trial's chosen offer, so
    a return value of 1 means the current trial alone filled the net. A
    value of ``cut_off + 1`` means the net was never filled within the
    horizon.
    """
    reached_now = accumulated >= goal_size
    cumsum = np.cumsum(future_mags, axis=1) + accumulated[:, None]
    reached_future = cumsum >= goal_size
    any_future = reached_future.any(axis=1)
    # argmax=0 -> fm[0] is the first *future* trial, i.e. trial 2.
    future_time = np.where(any_future,
                           reached_future.argmax(axis=1) + 2,
                           future_mags.shape[1] + 1)
    return np.where(reached_now, 1, future_time)


def _evaluate_options(offers, net_contents, net_item_idx, goal_size,
                      params, n_itr, rng):
    """Sampler values for three options on a single trial."""
    cut_off = params["cut_off"]
    vals = np.empty(3)
    for opt in range(3):
        fm = _walk_with_jumps(offers[opt], cut_off, params, n_itr, rng)
        acc = net_contents + offers[opt] if opt == net_item_idx else offers[opt]
        times = _completion_time(fm, np.full(n_itr, acc), goal_size)
        vals[opt] = -np.mean(times)
    return vals


def compute_sampler_switch_value(df, params=None, n_itr=1000, seed=42):
    """Add ``sampler_switch_value`` to *df* via Monte-Carlo tree search.

    For every non-first trial the sampler simulates future offer trajectories
    and estimates how quickly the net would be filled under each option.
    ``sampler_switch_value = best_alt_value - goal_item_value``.

    First trials of each block (no existing net) get NaN.
    """
    if params is None:
        params = dict(SAMPLER_PARAMS)
    rng = np.random.default_rng(seed)

    df = df.copy()
    # The item in the net *before* the current choice = previous trial's choice.
    df["_prev_choice"] = df.groupby(
        ["participant", "block_id"])["net_type"].shift(1)

    switch_vals = np.full(len(df), np.nan)
    for i in range(len(df)):
        row = df.iloc[i]
        prev = row["_prev_choice"]
        if pd.isna(prev):
            continue
        offers = np.array([row["offer_fish"],
                           row["offer_octopus"],
                           row["offer_crab"]])
        net_idx = ITEM_TO_IDX[prev]
        sv = _evaluate_options(offers, row["net_amount_before"],
                               net_idx, row["net_size"],
                               params, n_itr, rng)
        switch_vals[i] = np.max(np.delete(sv, net_idx)) - sv[net_idx]

    df["sampler_switch_value"] = switch_vals
    df.drop(columns="_prev_choice", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def print_mannwhitney_results(arr1, arr2, nan_policy="omit"):
    t, p = stats.mannwhitneyu(arr1, arr2, nan_policy=nan_policy)
    z = np.abs(stats.norm.ppf(p / 2))
    n1 = np.count_nonzero(~np.isnan(arr1))
    n2 = np.count_nonzero(~np.isnan(arr2))
    m1, se1 = np.round(np.nanmean(arr1), 2), np.round(np.nanstd(arr1) / np.sqrt(n1), 2)
    m2, se2 = np.round(np.nanmean(arr2), 2), np.round(np.nanstd(arr2) / np.sqrt(n2), 2)
    md1 = np.round(np.nanmedian(arr1), 2)
    md2 = np.round(np.nanmedian(arr2), 2)
    r = z / np.sqrt(n1 + n2)
    print(f"M1={m1}±{se1}, Md1 = {md1}, M2={m2}±{se2}, Md2 = {md2}, U({n1}, {n2})={np.round(t, 2)}, p{('=' + str(np.round(p, 3))) if p > 0.001 else '<0.001'}, r = {np.round(r, 2)}")


def print_kstest_results(arr1, arr2, nan_policy="omit"):
    res = stats.kstest(arr1, arr2, nan_policy=nan_policy)
    m1, se1 = np.round(np.nanmean(arr1), 2), np.round(np.nanstd(arr1) / np.sqrt(np.count_nonzero(~np.isnan(arr1))), 2)
    m2, se2 = np.round(np.nanmean(arr2), 2), np.round(np.nanstd(arr2) / np.sqrt(np.count_nonzero(~np.isnan(arr2))), 2)
    print(f"M1={m1}±{se1}, M2={m2}±{se2}, D={np.round(res.statistic, 2)}, p{('=' + str(np.round(res.pvalue, 3))) if res.pvalue > 0.001 else '<0.001'}")



# ---------------------------------------------------------------------------
# Comparison tables
# ---------------------------------------------------------------------------

def _stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _fmt_summary(vals):
    """Format mean +/- std (median) for an array, ignoring NaN/inf."""
    v = vals[np.isfinite(vals)]
    if len(v) == 0:
        return "---"
    return f"{np.mean(v):.2f} ± {np.std(v):.2f} ({np.median(v):.2f})"


def build_comparison_tables(plotters, models, prefixes, model_names_dict,
                            include_human_descriptives=True,
                            exclude_per_metric=None):
    """Build descriptive and statistical comparison DataFrames.

    Parameters
    ----------
    plotters : dict of Plotter objects keyed by model name.
    models : list of model names to compare against human.
    prefixes : dict mapping metric key -> display name.
    model_names_dict : dict mapping model key -> display name.
    include_human_descriptives : whether to include a column for humans.
    exclude_per_metric : optional dict mapping metric key -> list of model
        names to mark as "---" (not enough data to compare).

    Returns
    -------
    desc_df, stats_df : pandas DataFrames.
    """
    exclude_per_metric = exclude_per_metric or {}
    human_name = model_names_dict["human"]
    display_names = [model_names_dict[m] for m in models]
    metric_names = list(prefixes.values())

    desc_data = {}
    stats_data = {}

    for metric, metric_name in prefixes.items():
        human_vals = plotters["human"].metrics[metric].astype(float)
        human_clean = human_vals[np.isfinite(human_vals)]
        excluded = set(exclude_per_metric.get(metric, []))

        if include_human_descriptives:
            desc_data.setdefault(human_name, {})[metric_name] = _fmt_summary(human_vals)

        for model, display_name in zip(models, display_names):
            if model in excluded:
                desc_data.setdefault(display_name, {})[metric_name] = "---"
                stats_data.setdefault((display_name, "Md."), {})[metric_name] = "---"
                stats_data.setdefault((display_name, "Dist."), {})[metric_name] = "---"
                continue

            model_vals = plotters[model].metrics[metric].astype(float)
            model_clean = model_vals[np.isfinite(model_vals)]

            desc_data.setdefault(display_name, {})[metric_name] = _fmt_summary(model_vals)

            if len(model_clean) == 0 or len(human_clean) == 0:
                stats_data.setdefault((display_name, "Md."), {})[metric_name] = "---"
                stats_data.setdefault((display_name, "Dist."), {})[metric_name] = "---"
                continue

            U, p_mwu = stats.mannwhitneyu(model_clean, human_clean)
            stats_data.setdefault((display_name, "Md."), {})[metric_name] = (
                f"{U:.1f}{_stars(p_mwu)}"
            )

            ks_stat, p_dist = stats.ks_2samp(model_clean, human_clean)
            stats_data.setdefault((display_name, "Dist."), {})[metric_name] = (
                f"{ks_stat:.3f}{_stars(p_dist)}"
            )

    desc_df = pd.DataFrame(desc_data, index=metric_names)
    desc_df.columns.name = "Model"
    desc_df.index.name = "Metric"

    stats_df = pd.DataFrame(stats_data, index=metric_names)
    stats_df.columns = pd.MultiIndex.from_tuples(stats_df.columns)
    stats_df.index.name = "Metric"

    return desc_df, stats_df


# ---------------------------------------------------------------------------
# Logistic fitting utilities
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Standard logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


def fit_logistic(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit P(y=1) = sigmoid(b0 + b1 * x) via GLM. Returns (b0, b1)."""
    X = sm.add_constant(x)
    res = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    return float(res.params[0]), float(res.params[1])


# ---------------------------------------------------------------------------
# Plotter
# ---------------------------------------------------------------------------

class Plotter():
    """Data plotter for Holton et al. 2024 goal-commitment analyses."""

    def __init__(self, df,
                 model_color="#6a3d9a",
                 bias_color="#2ca02c",
                 main_color=None,
                 marker="o",
                 progress_palette="Purples",
                 plot_range=(-20, 20),
                 plotting_mode="paper"):
        """
        Parameters
        ----------
        df : DataFrame (restructured + sampler values).
        model_color : colour for sigmoid curves.
        bias_color : colour for indifference-point dots.
        main_color : colour used in CDF / comparison plots.
                     Defaults to *model_color*.
        marker : marker style for CDF / comparison plots.
        progress_palette : seaborn palette name for the progress-bin sigmoids.
        plot_range : (min, max) x-axis range for abandonment-value plots.
        plotting_mode : 'paper', 'poster', or 'presentation'.
        """
        self.df = df
        self.n_participants = df["participant"].nunique()
        self.participants = sorted(df["participant"].unique())

        self.model_color = model_color
        self.bias_color = bias_color
        self.main_color = main_color if main_color is not None else model_color
        self.marker = marker
        self.progress_palette = progress_palette
        self.plot_range = np.arange(plot_range[0], plot_range[1] + 0.01, 0.1)

        self.bin_edges = [0.0, 0.25, 0.50, 0.75, np.inf]
        self.bin_labels = ["<25", "25-50", "50-75", ">75"]

        self.plotting_mode = plotting_mode
        self._set_plot_params()
        self._compute_metrics()

    # ---- metrics ----------------------------------------------------------

    def _compute_metrics(self):
        """Compute per-participant summary metrics.

        Populates ``self.metrics`` (dict of arrays) and
        ``self.fits`` (per-participant logistic fit parameters).
        """
        data = self.df.dropna(subset=["sampler_switch_value"]).copy()
        data["switched_int"] = data["switched"].astype(int)

        fits = {}
        ips = []
        for p in self.participants:
            sub = data[data["participant"] == p]
            n_switches = sub["switched_int"].sum()
            if len(sub) == 0 or n_switches < 2:
                fits[p] = (np.nan, np.nan, np.inf)
                ips.append(np.inf)
                continue
            b0, b1 = fit_logistic(
                sub["sampler_switch_value"].to_numpy(),
                sub["switched_int"].to_numpy(),
            )
            IP = -b0 / b1 if b1 != 0 else np.inf
            fits[p] = (b0, b1, IP)
            ips.append(IP)

        self.fits = fits

        nets_filled = np.array([
            self.df.loc[self.df["participant"] == p, "points_scored"].max()
            for p in self.participants
        ])

        # Goal completion time: mean trials per completed net
        # (only count blocks where the net was actually filled)
        block_stats = self.df.groupby(["participant", "block_id"]).agg(
            n_trials=("switched", "count"),
            filled=("net_filled", "any"),
        )
        completed = block_stats[block_stats["filled"]]
        mean_trials_per_goal = completed.groupby("participant")["n_trials"].mean()
        goal_completion_time = np.array([
            mean_trials_per_goal.get(p, np.nan) for p in self.participants
        ])

        # Best/better alternative chosen on switch trials
        # Compare chosen alternative's offer to the third item's offer.
        better_alt = []
        best_alt = []
        for p in self.participants:
            sub = self.df[self.df["participant"] == p].copy()
            sub = sub.sort_values(["block_id", "trial_in_block"]).reset_index(drop=True)
            sub["prev_choice"] = sub.groupby("block_id")["choice"].shift(1)

            switches = sub[(sub["switched"] == True) & sub["prev_choice"].notna()].copy()
            if len(switches) == 0:
                better_alt.append(np.nan)
                best_alt.append(np.nan)
                continue

            # Offer of the abandoned (current goal) item
            goal_offer = np.where(switches["prev_choice"] == "fish", switches["offer_fish"],
                          np.where(switches["prev_choice"] == "octopus", switches["offer_octopus"],
                          np.where(switches["prev_choice"] == "crab", switches["offer_crab"], np.nan)))
            # Offer of the third (unchosen alternative) item
            sum_offers = (switches["offer_fish"] + switches["offer_octopus"]
                          + switches["offer_crab"])
            unchosen_alt_offer = sum_offers - switches["choice_offer"] - goal_offer

            picked_better = switches["choice_offer"] > unchosen_alt_offer
            better_alt.append(picked_better.mean())

            # Tempted switches: current goal's offer below at least one alternative
            tempted_mask = (goal_offer < switches["choice_offer"]) | (goal_offer < unchosen_alt_offer)
            if tempted_mask.sum() == 0:
                best_alt.append(np.nan)
            else:
                best_alt.append(picked_better[tempted_mask].mean())

        self.metrics = {
            "persistence_bias": np.array(ips),
            "nets_filled": nets_filled,
            "goal_completion_time": goal_completion_time,
            "better_alternative_chosen": np.array(better_alt),
            "best_alternative_chosen": np.array(best_alt),
        }

    # ---- style configuration ----------------------------------------------

    def _set_plot_params(self):
        """Set matplotlib rcParams based on plotting_mode."""
        if self.plotting_mode == "paper":
            self.font_size = 14
            self.k_color = "k"
            self.bg_color = "w"
            plt.rcParams["errorbar.capsize"] = 5
        elif self.plotting_mode == "poster":
            self.font_size = 20
            self.k_color = "w"
            self.bg_color = "#333333"
            plt.rcParams["errorbar.capsize"] = 10
        elif self.plotting_mode == "presentation":
            self.font_size = 20
            self.k_color = "k"
            self.bg_color = "w"
            plt.rcParams["errorbar.capsize"] = 10

        self.legend_font_size = self.font_size - 2
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.size"] = self.font_size

        plt.rcParams["lines.color"] = self.k_color
        plt.rcParams["lines.markeredgecolor"] = self.k_color
        plt.rcParams["lines.markerfacecolor"] = self.k_color
        plt.rcParams["axes.edgecolor"] = self.k_color
        plt.rcParams["text.color"] = self.k_color
        plt.rcParams["axes.labelcolor"] = self.k_color
        plt.rcParams["xtick.color"] = self.k_color
        plt.rcParams["ytick.color"] = self.k_color
        plt.rcParams["axes.facecolor"] = self.bg_color
        plt.rcParams["figure.facecolor"] = self.bg_color

        plt.rcParams["legend.frameon"] = False
        plt.rcParams["legend.labelspacing"] = 0
        plt.rcParams["legend.columnspacing"] = 0.7
        plt.rcParams["legend.handletextpad"] = 0.1

        plt.rcParams["lines.linewidth"] = 2
        plt.rcParams["axes.spines.right"] = False
        plt.rcParams["axes.spines.top"] = False
        plt.rcParams["mathtext.fontset"] = "custom"
        plt.rcParams["mathtext.default"] = "regular"
        plt.rcParams["mathtext.rm"] = "Arial"
        plt.rcParams["mathtext.it"] = "Arial:italic"
        plt.rcParams["mathtext.bf"] = "Arial:bold"
        self.fade = 0.4

    # ---- Figure 2b --------------------------------------------------------

    def plot_fig2b(self, ax=None):
        """Plot Figure 2b: abandonment probability vs tree-search value.

        Per-participant logistic sigmoids (thin, transparent) with
        indifference points, plus an aggregate pooled sigmoid (bold).

        Returns
        -------
        fig, ax, summary : figure, axes, and dict with 'fits' / 'aggregate'.
        """
        data = self.df.dropna(subset=["sampler_switch_value"]).copy()
        data["switched_int"] = data["switched"].astype(int)

        # Aggregate fit
        agg_b0, agg_b1 = fit_logistic(
            data["sampler_switch_value"].to_numpy(),
            data["switched_int"].to_numpy(),
        )

        # Plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 3.5))
        else:
            fig = ax.figure

        for b0, b1, IP in self.fits.values():
            if np.isnan(b0):
                continue
            probs = sigmoid(b0 + b1 * self.plot_range)
            ax.plot(self.plot_range, probs,
                    linewidth=0.5, alpha=0.25, color=self.model_color)
            if np.isfinite(IP):
                ax.scatter(IP, 0.5, color=self.bias_color,
                           s=15, alpha=0.6, zorder=3)

        agg_probs = sigmoid(agg_b0 + agg_b1 * self.plot_range)
        ax.plot(self.plot_range, agg_probs,
                linewidth=2.2, color=self.model_color,
                label="aggregate fit")

        ax.axvline(0, color=self.k_color, linewidth=0.75, linestyle="--")
        ax.set_xlim(self.plot_range[0], self.plot_range[-1])
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(np.arange(self.plot_range[0], self.plot_range[-1] + 1, 10))
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_xlabel("abandonment value (a.u.)")
        ax.set_ylabel("abandonment\nprobability")

        summary = {
            "fits": self.fits,
            "aggregate": (agg_b0, agg_b1, -agg_b0 / agg_b1),
        }
        return fig, ax, summary

    # ---- Figure 2c --------------------------------------------------------

    def plot_fig2c(self, ax=None):
        """Plot Figure 2c: abandonment by goal-progress quartile.

        Four logistic sigmoids (light to dark purple) for progress bins
        <25%, 25-50%, 50-75%, >75%, with dashed indifference-point lines.

        Returns
        -------
        fig, ax, summary : figure, axes, and dict with per-bin fit results.
        """
        data = self.df.copy()
        data["progress_before"] = data["net_amount_before"] / data["net_size"]

        data = data[data["sampler_switch_value"].notna()].copy()
        data["switched_int"] = data["switched"].astype(int)

        data["progress_bin"] = pd.cut(
            data["progress_before"],
            bins=self.bin_edges, labels=self.bin_labels,
            include_lowest=True, right=False,
        )

        fits = {}
        for label in self.bin_labels:
            sub = data[data["progress_bin"] == label]
            n = len(sub)
            if n < 10 or sub["switched_int"].nunique() < 2:
                fits[label] = (np.nan, np.nan, np.nan, n)
                continue
            b0, b1 = fit_logistic(
                sub["sampler_switch_value"].to_numpy(),
                sub["switched_int"].to_numpy(),
            )
            IP = -b0 / b1 if b1 != 0 else np.nan
            fits[label] = (b0, b1, IP, n)

        # Plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 3.5))
        else:
            fig = ax.figure

        palette = sns.color_palette(self.progress_palette)[2:6]

        for label, color in zip(self.bin_labels, palette):
            b0, b1, IP, _ = fits[label]
            if not np.isfinite(b0):
                continue
            probs = sigmoid(b0 + b1 * self.plot_range)
            ax.plot(self.plot_range, probs,
                    linewidth=2, color=color, label=label)
            if np.isfinite(IP):
                ax.axvline(IP, color=color, alpha=0.8, linestyle="--",
                           linewidth=1, label="_nolegend_")

        ax.axvline(0, color=self.k_color, linewidth=1,
                   linestyle="--", label="_nolegend_")
        ax.set_xlim(self.plot_range[0], self.plot_range[-1])
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(np.arange(self.plot_range[0], self.plot_range[-1] + 1, 10))
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_xlabel("abandonment value (a.u.)")
        ax.set_ylabel("abandonment\nprobability")
        ax.legend(title="goal progress (%)", frameon=False,
                  loc="upper left", fontsize=self.legend_font_size,
                  title_fontsize=self.legend_font_size)

        summary = {"fits": fits}
        return fig, ax, summary
