from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "SOURCE_SUBREDDIT",
    "TARGET_SUBREDDIT",
    "POST_ID",
    "TIMESTAMP",
    "POST_LABEL",
    "POST_PROPERTIES",
]

# The 86 post features, in file order (from the SNAP page; verify the order
# against the page, since a wrong order would silently mislabel every feature).
PROPERTY_NAMES = [
    "n_chars", "n_chars_no_space", "frac_alpha", "frac_digits", "frac_upper",
    "frac_space", "frac_special", "n_words", "n_unique_words", "n_long_words",
    "avg_word_len", "n_unique_stopwords", "frac_stopwords", "n_sentences",
    "n_long_sentences", "avg_chars_per_sentence", "avg_words_per_sentence",
    "readability_ari", "vader_pos", "vader_neg", "vader_compound",
] + [
    "LIWC_" + name for name in (
        "Funct Pronoun Ppron I We You SheHe They Ipron Article Verbs AuxVb "
        "Past Present Future Adverbs Prep Conj Negate Quant Numbers Swear "
        "Social Family Friends Humans Affect Posemo Negemo Anx Anger Sad "
        "CogMech Insight Cause Discrep Tentat Certain Inhib Incl Excl "
        "Percept See Hear Feel Bio Body Health Sexual Ingest Relativ Motion "
        "Space Time Work Achiev Leisure Home Money Relig Death Assent "
        "Dissent Nonflu Filler"
    ).split()
]
assert len(PROPERTY_NAMES) == 86, len(PROPERTY_NAMES)

# Reference dates for annotating time plots (verify before using in a report).
EVENTS = {
    "Charlie Hebdo": "2015-01-07",
    "Paris attacks": "2015-11-13",
    "US election": "2016-11-08",
}

# --------------------------------------------------------------------------
# Structure and quality
# --------------------------------------------------------------------------
def overview(links: pd.DataFrame) -> pd.Series:
    """One-glance summary of size and coverage."""
    src, tgt = links["SOURCE_SUBREDDIT"], links["TARGET_SUBREDDIT"]
    all_subs = pd.unique(pd.concat([src.astype(str), tgt.astype(str)]))
    return pd.Series(
        {
            "rows (hyperlinks)": len(links),
            "unique source subreddits": src.nunique(),
            "unique target subreddits": tgt.nunique(),
            "unique subreddits overall": len(all_subs),
            "unique POST_IDs": links["POST_ID"].nunique(),
            "first timestamp": links["TIMESTAMP"].min(),
            "last timestamp": links["TIMESTAMP"].max(),
            "self-links (source == target)": int(
                (src.astype(str) == tgt.astype(str)).sum()
            ),
            "missing values": int(links.isna().sum().sum()),
        }
    )

def missing_report(links: pd.DataFrame, props: pd.DataFrame | None = None):
    """Missing values per column (links table, and optionally the features)."""
    parts = [links.isna().sum().rename("missing")]
    if props is not None:
        parts.append(props.isna().sum().rename("missing"))
    out = pd.concat(parts).to_frame()
    total = len(links)
    out["share"] = (out["missing"] / total).round(4)
    return out[out["missing"] > 0] if (out["missing"] > 0).any() else out.head(0)


def duplicate_report(links: pd.DataFrame) -> pd.Series:
    """Duplicate checks: whole rows, and repeated POST_IDs."""
    key = ["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "POST_ID", "TIMESTAMP"]
    return pd.Series(
        {
            "fully duplicated rows": int(links.duplicated().sum()),
            "duplicated (source, target, post, time)": int(links.duplicated(key).sum()),
            "rows sharing a POST_ID with another row": int(
                links["POST_ID"].duplicated(keep=False).sum()),
            "POST_IDs with different link sentiments": int(
                (links.groupby("POST_ID")["LINK_SENTIMENT"].nunique() > 1).sum()
            ),
            "total unique posts": int(len(links["POST_ID"].unique())),
            
        }
    )
def fuse_og_datasets(links1: pd.DataFrame, links2: pd.DataFrame) -> pd.DataFrame:
    """Combine two links tables, keeping only the columns in EXPECTED_COLUMNS."""
    links1["HYPERLINK_LOCATION"] = "body"
    links2["HYPERLINK_LOCATION"] = "title"
    out = pd.concat([links1, links2], axis=0, ignore_index=True)

    return out.copy()

# --------------------------------------------------------------------------
# Distributions
# --------------------------------------------------------------------------
def sentiment_balance(links: pd.DataFrame) -> pd.DataFrame:
    counts = links["POST_LABEL"].value_counts().sort_index()
    return pd.DataFrame({"links": counts, "share": (counts / counts.sum()).round(4)})


def top_subreddits(links: pd.DataFrame, col: str = "SOURCE_SUBREDDIT", n: int = 15):
    """Most frequent sources/targets, with their share of negative links."""
    g = links.groupby(col, observed=True)["POST_LABEL"]
    out = pd.DataFrame({"links": g.size(), "neg_share": g.apply(lambda s: (s == -1).mean())})
    return out.sort_values("links", ascending=False).head(n).round(3)

def degree_table(links: pd.DataFrame) -> pd.DataFrame:
    """Out/in link counts per subreddit, and negative out/in counts."""
    neg = links["POST_LABEL"] == -1
    out_deg = links.groupby("SOURCE_SUBREDDIT", observed=True).size().rename("out")
    in_deg = links.groupby("TARGET_SUBREDDIT", observed=True).size().rename("in")
    out_neg = links[neg].groupby("SOURCE_SUBREDDIT", observed=True).size().rename("out_neg")
    in_neg = links[neg].groupby("TARGET_SUBREDDIT", observed=True).size().rename("in_neg")
    table = pd.concat([out_deg, in_deg, out_neg, in_neg], axis=1).fillna(0).astype(int)
    table["total"] = table["out"] + table["in"]
    return table.sort_values("total", ascending=False)

def period_counts(links: pd.DataFrame, period: str = "W") -> pd.DataFrame:
    """Links per period, negative links per period and negative share."""
    week = links["TIMESTAMP"].dt.to_period(period).dt.start_time
    neg = links["POST_LABEL"] == -1
    out = pd.DataFrame({"links": links.groupby(week).size(), "neg": neg.groupby(week).sum()})
    out["neg_share"] = out["neg"] / out["links"]
    return out


def activity_thresholds(links: pd.DataFrame, minimums=(1, 2, 3, 5, 10)) -> pd.DataFrame:
    """How many (source subreddit, week) cells have at least m outgoing links?

    Directly shows how much data survives a minimum-activity rule.
    """
    week = links["TIMESTAMP"].dt.to_period("W")
    cells = links.groupby(["SOURCE_SUBREDDIT", week], observed=True).size()
    rows = []
    for m in minimums:
        keep = cells >= m
        rows.append(
            {
                "min links per week (m)": m,
                "subreddit-weeks kept": int(keep.sum()),
                "share of all cells": round(float(keep.mean()), 3),
                "subreddits with >=1 kept week": int(
                    keep[keep].index.get_level_values(0).nunique()
                ),
            }
        )
    return pd.DataFrame(rows)


def property_summary(props: pd.DataFrame) -> pd.DataFrame:
    """Per-feature mean, spread, range, zero share and skewness."""
    desc = props.describe().T[["mean", "std", "min", "50%", "max"]]
    desc["zero_share"] = (props == 0).mean()
    desc["skew"] = props.skew()
    return desc.round(3)


def property_sentiment_gap(props: pd.DataFrame, sentiment: pd.Series, top: int = 15):
    """Correlation of each feature with 'link is negative' (1/0)."""
    is_neg = (sentiment == -1).astype(float).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):  # constant columns give NaN
        corr = props.apply(lambda col: np.corrcoef(col.to_numpy(), is_neg)[0, 1])
    corr = corr.rename("corr_with_negative").dropna()
    return corr.reindex(corr.abs().sort_values(ascending=False).index).head(top).round(3)




# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _mark_events(ax, events: dict | None):
    for name, date in (events or {}).items():
        ax.axvline(pd.Timestamp(date), color="grey", ls="--", lw=0.8)
        ax.text(pd.Timestamp(date), ax.get_ylim()[1], f" {name}", rotation=90,
                va="top", fontsize=7, color="grey")


def plot_periodicaly(links: pd.DataFrame, events: dict | None = EVENTS):
    wk = period_counts(links, period="W")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(wk.index, wk["links"])
    axes[0].set_ylabel("links per week")
    axes[1].plot(wk.index, wk["neg_share"], color="tab:red")
    axes[1].set_ylabel("share negative")
    for ax in axes:
        _mark_events(ax, events)
    fig.tight_layout()
    return fig


def plot_degree_distribution(table: pd.DataFrame):
    """Log-log histogram of out- and in-link counts per subreddit."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col in zip(axes, ["out", "in"]):
        values = table.loc[table[col] > 0, col]
        bins = np.unique(np.logspace(0, np.log10(values.max() + 1), 40).astype(int))
        ax.hist(values, bins=bins)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{col}-links per subreddit")
        ax.set_ylabel("number of subreddits")
    fig.tight_layout()
    return fig


def plot_property_distributions(props: pd.DataFrame, columns=None, ncols: int = 4):
    columns = list(columns) if columns is not None else list(props.columns[:12])
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.4 * nrows))
    for ax, col in zip(np.ravel(axes), columns):
        ax.hist(props[col].dropna(), bins=40)
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in np.ravel(axes)[len(columns):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(props: pd.DataFrame, sample: int = 200_000):
    data = props if len(props) <= sample else props.sample(sample, random_state=0)
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Correlation between text features")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig