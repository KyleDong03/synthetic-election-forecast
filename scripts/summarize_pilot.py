"""Pilot summaries."""

from collections import Counter
import json
import math
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = {
    "vote": ["Democratic", "Republican", "Other", "Would not vote"],
    "track": ["Right direction", "Wrong track", "Not sure"],
    "attention": ["Very closely", "Somewhat closely", "Not too closely", "Not at all"],
}


def add_age_groups(table):
    return table.with_columns(
        pl.when(pl.col("age") < 35).then(pl.lit("18-34"))
        .when(pl.col("age") < 65).then(pl.lit("35-64"))
        .otherwise(pl.lit("65+")).alias("age_group")
    )


def read_answers(cache, personas, races):
    files = sorted(cache.glob("*.json"))
    if not files:
        raise ValueError("Cache: empty")
    if personas["persona_id"].n_unique() != personas.height:
        raise ValueError("Duplicate persona IDs")
    people = {row["persona_id"]: row for row in personas.iter_rows(named=True)}
    race_ids = {row["state_name"]: row["race_id"] for row in races.iter_rows(named=True)}
    if len(race_ids) != races.height:
        raise ValueError("Duplicate state races")
    seen = set()
    rows = []
    for path in files:
        record = json.loads(path.read_text())
        person = people.get(record["persona_id"])
        if person is None or record["persona"] != person["persona"]:
            raise ValueError(f"Persona mismatch: {path.name}")
        if person["state"] not in race_ids:
            raise ValueError("Unknown persona state")
        if f"Senate in {person['state']}" not in record["question"]:
            raise ValueError("Question/state mismatch")
        repeat = record.get("repeat_id", record.get("draw", 0))
        key = (record["model_requested"], record["persona_id"], repeat)
        if key in seen:
            raise ValueError("Duplicate model/persona/repeat")
        seen.add(key)
        for question, options in OPTIONS.items():
            if record[question] is not None and record[question] not in options:
                raise ValueError(f"Invalid {question} answer in {path.name}")
        confidence = record["confidence"]
        if confidence is not None and (type(confidence) is not int or not 0 <= confidence <= 100):
            raise ValueError("Confidence: integer 0-100 or null")
        cost = record["cost"]
        if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
            raise ValueError("Invalid call cost")
        rows.append({
            "race_id": race_ids[person["state"]], "state": person["state"],
            "model": record["model_requested"], "model_answered": record["model_answered"],
            "persona_id": record["persona_id"], "repeat_id": str(repeat),
            "vote": record["vote"], "track": record["track"],
            "attention": record["attention"], "confidence": confidence, "cost_usd": float(cost),
        })
    return pl.DataFrame(rows)


def summarize_answers(answers):
    rows = []
    groups = answers.select("race_id", "state", "model").unique().sort("race_id", "model")
    for group in groups.iter_rows(named=True):
        sample = answers.filter((pl.col("race_id") == group["race_id"]) & (pl.col("model") == group["model"]))
        for question, options in OPTIONS.items():
            counts = Counter(sample[question].to_list())
            for option in options + [None]:
                rows.append({**group, "question": question,
                             "answer": "Missing" if option is None else option,
                             "count": counts[option], "total_answers": sample.height,
                             "share": counts[option] / sample.height})
    categorical = pl.DataFrame(rows)
    confidence = answers.group_by("race_id", "state", "model").agg(
        pl.col("confidence").count().alias("valid_answers"),
        pl.col("confidence").null_count().alias("missing_answers"),
        pl.col("confidence").mean().alias("mean_confidence"),
        pl.col("confidence").median().alias("median_confidence"),
    ).sort("race_id", "model")
    costs = answers.group_by("model", "model_answered").agg(
        pl.len().alias("cached_calls"), pl.col("persona_id").n_unique().alias("unique_personas"),
        pl.col("race_id").n_unique().alias("races"),
        pl.col("repeat_id").n_unique().alias("repeat_ids"),
        pl.col("cost_usd").sum().alias("recorded_cost_usd"),
    ).sort("model", "model_answered")
    return categorical, confidence, costs


def compare_demographics(personas, ces):
    if (ces["weight"] <= 0).any() or ces["weight"].null_count() or not ces["weight"].is_finite().all():
        raise ValueError("Invalid CES weights")
    personas = add_age_groups(personas)
    ces = add_age_groups(ces)
    rows = []
    variables = ["age_group", "gender", "race", "education", "party", "ideology", "registered"]
    for state in sorted(personas["state"].unique().to_list()):
        sample = personas.filter(pl.col("state") == state)
        source = ces.filter(pl.col("state") == state)
        if source.is_empty():
            raise ValueError(f"Missing CES state: {state}")
        for variable in variables:
            sample_values = sample[variable].fill_null("Missing")
            source_values = source[variable].fill_null("Missing")
            categories = sorted(set(sample_values.to_list()) | set(source_values.to_list()))
            for category in categories:
                count = int((sample_values == category).sum())
                source_share = float(source.filter(source_values == category)["weight"].sum() / source["weight"].sum())
                share = count / sample.height
                rows.append({"state": state, "variable": variable, "category": category,
                             "persona_count": count, "persona_total": sample.height,
                             "persona_share": share, "ces_weighted_share": source_share,
                             "difference_pp": 100 * (share - source_share)})
    return pl.DataFrame(rows)


def compare_models(summary):
    models = sorted(summary["model"].unique().to_list())
    if len(models) != 2:
        raise ValueError("Model count: expected 2")
    keys = ["race_id", "state", "question", "answer"]
    first = summary.filter(pl.col("model") == models[0]).select(keys + ["share"]).rename({"share": "share_a"})
    second = summary.filter(pl.col("model") == models[1]).select(keys + ["share"]).rename({"share": "share_b"})
    comparison = first.join(second, on=keys, how="full", coalesce=True)
    if comparison["share_a"].null_count() or comparison["share_b"].null_count():
        raise ValueError("Model coverage mismatch")
    return comparison.with_columns(
        pl.lit(models[0]).alias("model_a"), pl.lit(models[1]).alias("model_b"),
        (100 * (pl.col("share_a") - pl.col("share_b"))).alias("difference_pp_a_minus_b"),
    ).sort(keys)


def main():
    from build_personas import label_respondents

    ces_path = ROOT / "data" / "01_raw_data" / "ces_common_2025.parquet"
    if not ces_path.exists():
        raise SystemExit("Required: uv run python scripts/download_data.py")
    personas = pl.read_parquet(ROOT / "data" / "02_processed_data" / "personas.parquet")
    races = pl.read_csv(ROOT / "data" / "02_processed_data" / "senate_races.csv")
    answers = read_answers(ROOT / "data" / "03_silicon_sampled_data" / "cache", personas, races)
    categorical, confidence, costs = summarize_answers(answers)
    comparisons = compare_models(categorical)
    ces = label_respondents(pl.read_parquet(ces_path))
    demographics = compare_demographics(personas, ces)
    output = ROOT / "outputs" / "pilot"
    output.mkdir(parents=True, exist_ok=True)
    for name, table in [("question_summary", categorical), ("confidence_summary", confidence),
                        ("cost_summary", costs), ("persona_demographics", demographics),
                        ("model_comparison", comparisons)]:
        table.write_csv(output / f"{name}.csv")
        table.write_parquet(output / f"{name}.parquet")
    print(f"- Cached calls: {answers.height:,}")
    print("- Pilot summaries; not forecasts or national estimates")
    print(costs)
    print(f"- Output: {output}")


if __name__ == "__main__":
    main()
