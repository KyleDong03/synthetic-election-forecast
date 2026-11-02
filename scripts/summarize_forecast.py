"""Summarize the saved answers into one forecast row per Senate race."""

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "03_silicon_sampled_data"

SHORT_NAMES = {"anthropic/claude-sonnet-5.5": "claude", "openai/gpt-6-luna": "gpt"}
COLUMNS = ["dem", "rep", "other", "would_not_vote", "dem_perc", "rep_perc", "other_perc",
           "would_not_vote_perc", "leader"]


def summarize(answers):
    """Creates a summary of the Senate race forecast from the saved answers."""
    per_model = (
        answers.with_columns(pl.col("model").replace_strict(SHORT_NAMES))
        .group_by("race_id", "model")
        .agg(
            (pl.col("vote") == "Democratic").sum().alias("dem"),
            (pl.col("vote") == "Republican").sum().alias("rep"),
            ((pl.col("vote") == "Other") | pl.col("vote").is_null()).sum().alias("other"),
            (pl.col("vote") == "Would not vote").sum().alias("would_not_vote"),
            pl.len().alias("total"),
        )
        .with_columns(
            (100 * pl.col("dem") / pl.col("total")).alias("dem_perc"),
            (100 * pl.col("rep") / pl.col("total")).alias("rep_perc"),
            (100 * pl.col("other") / pl.col("total")).alias("other_perc"),
            (100 * pl.col("would_not_vote") / pl.col("total")).alias("would_not_vote_perc"),
        )
        .with_columns(
            pl.when(pl.col("dem") > pl.col("rep")).then(pl.lit("Democratic"))
            .when(pl.col("dem") < pl.col("rep")).then(pl.lit("Republican"))
            .otherwise(pl.lit("Tie"))
            .alias("leader")
        )
    )

    wide = None
    for name in SHORT_NAMES.values():
        part = per_model.filter(pl.col("model") == name).select(
            "race_id",
            *[pl.col(column).alias(f"{name}_{column}") for column in COLUMNS],
            pl.col("total").alias(f"{name}_total"),
        )
        wide = part if wide is None else wide.join(part, on="race_id", how="full", coalesce=True)

    first, second = SHORT_NAMES.values()
    return wide.with_columns(
        (pl.col(f"{first}_dem_perc") - pl.col(f"{second}_dem_perc")).alias("pp_gap"),
    ).sort("race_id")


def main():
    answers = pl.read_parquet(DATA / "answers.parquet")
    summary = summarize(answers)
    # Each race has 25 personas per model, so every count should add up to 25.
    assert (summary.select(pl.col("^.*_total$")) == 25).to_numpy().all()
    assert summary.height == answers["race_id"].n_unique()
    summary.write_csv(DATA / "forecast_summary.csv")
    print(summary)
    print(f"Saved {summary.height} races to {DATA / 'forecast_summary.csv'}")


if __name__ == "__main__":
    main()
