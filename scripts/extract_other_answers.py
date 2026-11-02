"""Save the other question's answers from the cache to one parquet file."""

import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "03_silicon_sampled_data" / "cache"
OUTPUT = ROOT / "data" / "03_silicon_sampled_data"


def main():
    # Each cache file holds one persona's answer from one model. Cache records have no race ID,
    # so it is looked up from the persona's state.
    records = [json.loads(path.read_text()) for path in CACHE.glob("*.json")]
    answers = pl.DataFrame(records).select(
        "persona_id", "model_requested", "confidence", "track", "attention"
    ).rename({"model_requested": "model"})

    personas = pl.read_parquet(ROOT / "data" / "02_processed_data" / "personas.parquet")
    races = pl.read_csv(ROOT / "data" / "02_processed_data" / "senate_races.csv")
    race_of_persona = personas.join(races.select("state_name", "race_id"),
                                    left_on="state", right_on="state_name", how="left")
    answers = answers.join(race_of_persona.select("persona_id", "race_id"), on="persona_id", how="left")
    assert answers["race_id"].null_count() == 0

    answers = answers.select("race_id", "persona_id", "model", "confidence", "track", "attention").sort("model", "race_id", "persona_id")
    answers.write_parquet(OUTPUT / "other_answers.parquet")
    print(f"Saved {answers.height:,} answers to {OUTPUT / 'other_answers.parquet'}")


if __name__ == "__main__":
    main()
