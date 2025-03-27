from __future__ import annotations

import typing
from pathlib import Path

import numpy as np
import seaborn as sns  # type: ignore
import typer
from matplotlib import pyplot as plt

from granite_tools.config import DEFAULT_LIMIT_MULTIPLIERS, read_config
from granite_tools.scorer import get_trigram_data_from_files
from granite_tools.scorer.scorer import create_log_m_func, get_limit_funcs

try:
    from typing import Annotated
except ImportError:
    # For older python versions
    from typing_extensions import Annotated  # type: ignore

if typing.TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

ARG_CONFIG_FILE = Annotated[
    Path | None,
    typer.Argument(
        help="The path to the Granite configuration (YAML) file.",
        show_default=False,
    ),
]

ARG_TRIGRAM_SCORING_FILE = Annotated[
    Path | None,
    typer.Argument(
        help="The path to the trigram relative scoring file.",
        show_default=False,
    ),
]

ARG_BIGRAM_RANKING_FILE = Annotated[
    Path | None,
    typer.Argument(
        help="The path to the bigram (+unigram) ranking file. Created with create_ngram_ranking.py",
        show_default=False,
    ),
]

ARG_NGRAM_ANCHOR_SCORES_JSON = Annotated[
    Path | None,
    typer.Argument(
        help="The path to the anchor scores (JSON) file. Created with granite_tools/scripts/scoreratios_fit.py",
        show_default=False,
    ),
]


ARG_ONLY_LIMITS = Annotated[
    bool,
    typer.Option(
        "--only-limits",
        help="Just show the limits; i.e. does not plot any trigram scoring data, but just the limits. In this mode, only accepts the [CONFIG_FILE] as an argument (other args are disregarded, and may be omitted).",
    ),
]

ARG_DASHBOARD = Annotated[
    bool,
    typer.Option(
        "--dashboard",
        help="Create a dashboard",
    ),
]


def main(
    config_file: ARG_CONFIG_FILE = None,
    bigram_ranking_file: ARG_BIGRAM_RANKING_FILE = None,
    trigram_scoring_file: ARG_TRIGRAM_SCORING_FILE = None,
    ngram_anchor_scores_file: ARG_NGRAM_ANCHOR_SCORES_JSON = None,
    only_limits: ARG_ONLY_LIMITS = False,
    dashboard: ARG_DASHBOARD = False,
) -> None:

    if config_file is not None:
        config = read_config(config_file)
        limit_multipliers = config.limit_multipliers
    else:
        limit_multipliers = DEFAULT_LIMIT_MULTIPLIERS

    plotter: BasePlotter

    if (
        config_file is None
        or bigram_ranking_file is None
        or trigram_scoring_file is None
        or ngram_anchor_scores_file is None
    ):
        if not only_limits:
            raise typer.BadParameter(
                "All arguments are required, except for --only-limits. "
                "Please provide all arguments.",
            )

        plotter = AcceptanceLimitPlotter(limit_multipliers)
        plotter.run()
        plt.tight_layout()
        plt.show()
        raise typer.Exit()

    df = get_trigram_data_from_files(
        config_file,
        bigram_ranking_file,
        trigram_scoring_file,
        ngram_anchor_scores_file,
    ).reset_index()

    if dashboard:
        create_dashboard(df)
        raise typer.Exit()

    plotter = ModelCheckPlotter(limit_multipliers, df)
    plotter.run()

    plt.tight_layout()
    plt.show()


def print_sse(df):
    for col, name in zip(
        [
            "score_ratio_resid",
            "score_ratio_scaled_resid_pred",
            "score_ratio_scaled_resid_actual",
        ],
        ["resid", "scaled_resid_pred", "scaled_resid_act"],
    ):
        resid_mask = df[col] >= 0
        SSE_pos = sum(df[resid_mask][col] ** 2)
        SSE_neg = sum(df[~resid_mask][col] ** 2)
        SSE = SSE_pos + SSE_neg
        print(name)
        print(f"SSE: {SSE:.2f}, SSE_pos: {SSE_pos:.2f}, SSE_neg: {SSE_neg:.2f}")
        print(
            f"SSE >= 0: {SSE_pos / SSE:.2%}, SSE < 0: {SSE_neg / SSE:.2%}", end="\n\n"
        )


def create_dashboard(df: pd.DataFrame):

    import panel as pn
    from panel_gwalker import GraphicWalker

    print(sorted(df.columns))
    pn.extension()

    dashboard_json = (
        Path(__file__).parent.parent.parent.parent
        / "examples"
        / "dashboard.config.json"
    )
    spec = dashboard_json if dashboard_json.exists() else None
    app = GraphicWalker(df, spec=spec).servable()
    pn.serve(app, port=5006, show=True)


class BasePlotter:

    def __init__(self, limit_multipliers: dict[float, float]):

        self.limit_multipliers = limit_multipliers
        self.r_range_start = min(limit_multipliers.keys())
        self.r_range_end = max(limit_multipliers.keys())
        self.r = get_r_for_plot(self.r_range_start, self.r_range_end)

        self.get_lower_limit, self.get_upper_limit = get_limit_funcs(limit_multipliers)
        self.lower_limit = self.get_lower_limit(self.r)
        self.upper_limit = self.get_upper_limit(self.r)

    def run(self):
        raise NotImplementedError

    def plot_acceptance_limits_on_ax(
        self,
        ax: Axes,
        r: np.ndarray,
        ref: np.ndarray,
        lower_limit: np.ndarray,
        upper_limit: np.ndarray,
        ylabel: str,
        color: str = "tab:blue",
        xlabel: str = "Score ratio (r)",
    ):

        self._plot_with_extrapolate_on(ax, lower_limit, upper_limit, color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot(r, ref, color="tab:red", ls="--")
        self._finalize_ax(ax)

    def _plot_with_extrapolate_on(
        self,
        ax,
        y_upper,
        y_lower,
        color: str,
        color_extrapolate="gray",
        alpha_extrapolate=0.6,
        ls_extrapolate="--",
    ):

        extrapolate_mask_left = self.r < self.r_range_start
        extrapolate_mask_right = self.r > self.r_range_end
        idx_extrapolate_left = np.where(extrapolate_mask_left)[0]
        idx_extrapolate_right = np.where(extrapolate_mask_right)[0]
        idx = np.where(~(extrapolate_mask_left | extrapolate_mask_right))[0]

        ax.plot(self.r[idx], y_lower[idx], color=color)
        ax.plot(self.r[idx], y_upper[idx], color=color)
        kwargs = dict(
            color=color_extrapolate,
            alpha=alpha_extrapolate,
            ls=ls_extrapolate,
            lw=0.44,
        )
        ax.plot(self.r[idx_extrapolate_left], y_lower[idx_extrapolate_left], **kwargs)
        ax.plot(self.r[idx_extrapolate_left], y_upper[idx_extrapolate_left], **kwargs)
        ax.plot(self.r[idx_extrapolate_right], y_lower[idx_extrapolate_right], **kwargs)
        ax.plot(self.r[idx_extrapolate_right], y_upper[idx_extrapolate_right], **kwargs)

    @staticmethod
    def _finalize_ax(ax, aspect: str = "equal"):
        ax.axvline(x=1, color="black", ls="-", lw=0.8)
        ax.grid(ls="--", color="gray", alpha=0.5, lw=0.5)
        ax.set_aspect(aspect)
        ax.set_box_aspect(1)


class AcceptanceLimitPlotter(BasePlotter):
    def run(self):

        plt.figure(figsize=(8, 8))
        self.ax_m = plt.subplot2grid((4, 4), (0, 1), 2, 2)
        self.ax_r = plt.subplot2grid((4, 4), (2, 0), 2, 2)
        self.ax_d = plt.subplot2grid((4, 4), (2, 2), 2, 2)

        self.axes = [self.ax_m, self.ax_r, self.ax_d]

        self.plot_log_m_on_ax(self.ax_m, self.r)
        self.plot_acceptance_limits_on_ax(
            self.ax_r,
            self.r,
            ref=self.r,
            lower_limit=self.lower_limit,
            upper_limit=self.upper_limit,
            ylabel="Acceptance limits of r",
        )
        self.plot_acceptance_limits_on_ax(
            self.ax_d,
            self.r,
            ref=np.zeros(len(self.r)),
            lower_limit=self.lower_limit - self.r,
            upper_limit=self.upper_limit - self.r,
            ylabel="Acceptance limits of residual",
        )

    def plot_log_m_on_ax(self, ax: Axes, r: np.ndarray):

        get_log_m = create_log_m_func(self.limit_multipliers)
        log_m = np.vectorize(get_log_m)(r)

        mlist_sorted = sorted(
            ((k, v) for k, v in self.limit_multipliers.items()), key=lambda x: x[0]
        )

        given_r, given_m = list(zip(*mlist_sorted))
        given_log_m = np.log(given_m)
        self._plot_with_extrapolate_on(ax, -log_m, log_m, "tab:purple")
        ax.scatter(given_r, given_log_m, color="black", s=7, zorder=10)
        ax.scatter(given_r, -given_log_m, color="black", s=7, zorder=10)
        ax.set_xlabel("Score ratio (r)")
        ax.set_ylabel("Log(multiplier)")
        ax.axhline(y=0, color="tab:red", ls="--")
        self._finalize_ax(ax)


class ModelCheckPlotter(BasePlotter):
    def __init__(self, limit_multipliers: dict[float, float], df: pd.DataFrame):
        super().__init__(limit_multipliers)
        self.df = df

    def run(self):

        fig, axes = plt.subplots(ncols=3, figsize=(18, 6))
        fig2, axes2 = plt.subplots(ncols=2, figsize=(12, 6))
        fig3, axes3 = plt.subplots(ncols=3, figsize=(12, 4))

        (
            self.ax_pred_vs_actual,
            self.ax_resid_actual,
            self.ax_resid_scaled_actual,
        ) = axes.flatten()

        (
            self.ax_resid_fit,
            self.ax_resid_scaled_fit,
        ) = axes2.flatten()

        self.plot_acceptance_limits_on_ax(
            self.ax_pred_vs_actual,
            self.r,
            ref=self.r,
            lower_limit=self.lower_limit,
            upper_limit=self.upper_limit,
            ylabel="r (predicted)",
            xlabel="r (actual)",
        )

        for trigram_type, df_type in self.df.groupby("trigram_type"):
            r = df_type["score_ratio_actual"].to_numpy()
            r_est = df_type["score_ratio_pred"].to_numpy()
            kwargs = dict(label=trigram_type, alpha=0.5, s=10)
            self.ax_pred_vs_actual.scatter(r, r_est, **kwargs)

        self.do_plot(self.ax_resid_fit, self.ax_resid_scaled_fit, use_actual_r=False)
        self.do_plot(
            self.ax_resid_actual, self.ax_resid_scaled_actual, use_actual_r=True
        )

        for ax in np.concat((axes.flatten(), axes2.flatten())):
            self._finalize_ax(
                ax,
                aspect=(
                    "auto"
                    if ax in (self.ax_resid_scaled_fit, self.ax_resid_scaled_actual)
                    else "equal"
                ),
            )
            ax.legend()

        sns.histplot(self.df["score_ratio_resid"], ax=axes3[0])
        sns.histplot(self.df["score_ratio_scaled_resid_pred"], ax=axes3[1])
        sns.histplot(self.df["score_ratio_scaled_resid_actual"], ax=axes3[2])
        print_sse(self.df)

    def do_plot(self, ax_resid, ax_scaled_resid, use_actual_r=False):

        xlabel = "r (actual)" if use_actual_r else "r (predicted)"

        self.plot_acceptance_limits_on_ax(
            ax_resid,
            self.r,
            ref=np.zeros(len(self.r)),
            lower_limit=self.lower_limit - self.r,
            upper_limit=self.upper_limit - self.r,
            ylabel="residual (actual - predicted)",
            xlabel=xlabel,
        )
        self.plot_acceptance_limits_on_ax(
            ax_scaled_resid,
            self.r,
            ref=np.zeros(len(self.r)),
            lower_limit=-np.ones(len(self.r)),
            upper_limit=np.ones(len(self.r)),
            ylabel="residual (scaled)",
            xlabel=xlabel,
        )

        for trigram_type, df_type in self.df.groupby("trigram_type"):
            r = df_type["score_ratio_actual"].to_numpy()
            r_est = df_type["score_ratio_pred"].to_numpy()
            resid = r_est - r
            kwargs = dict(label=trigram_type, alpha=0.5, s=10)

            idx_neg = np.where(resid < 0)[0]
            idx_pos = np.where(resid >= 0)[0]

            score_ratio = r if use_actual_r else r_est

            ax_resid.scatter(score_ratio, resid, **kwargs)
            ul = self.get_upper_limit(score_ratio) - score_ratio
            ll = score_ratio - self.get_lower_limit(score_ratio)  # this is > 0
            score_ratio_scaled = np.hstack([score_ratio[idx_neg], score_ratio[idx_pos]])

            scaled_resid_lower = resid[idx_neg] / ll[idx_neg]
            scaled_resid_upper = resid[idx_pos] / ul[idx_pos]
            resid_scaled = np.hstack([scaled_resid_lower, scaled_resid_upper])
            ax_scaled_resid.scatter(score_ratio_scaled, resid_scaled, **kwargs)


def print_table(get_log_m):
    # Helper function to print a table of limits
    # fmt: off
    r_table = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0)
    # fmt: on
    m_table = np.exp([get_log_m(ri) for ri in r_table])
    lower_limit_table = r_table / m_table
    upper_limit_table = r_table * m_table
    for r, lower, upper in zip(r_table, lower_limit_table, upper_limit_table):
        print(
            f"| {lower:.2f} (-{r-lower:.2f}) | {r:.2f} | {upper:.2f} (+{upper-r:.2f})|"
        )


def get_r_for_plot(min_r: float, max_r: float) -> np.ndarray:
    r_range = max_r - min_r
    min_r_plot = max(0.001, min_r - r_range / 10)
    max_r_plot = max_r + r_range / 10
    r = np.linspace(min_r_plot, max_r_plot, 1000)
    return r


if __name__ == "__main__":

    app = typer.Typer(pretty_exceptions_enable=False)
    app.command()(main)
    app()
