"""Tests on the saved Senate responses."""

import json

import polars as pl

from scripts.query_personas import MODELS, ROOT

RESPONSES = ROOT / "data" / "03_silicon_sampled_data" / "responses.parquet"
CACHE = ROOT / "data" / "03_silicon_sampled_data" / "cache"


def test_response_format_matches_simulated_responses():
    responses = pl.read_parquet(RESPONSES)
    reference = pl.read_parquet(ROOT / "data" / "00_simulated_data" / "responses.parquet")
    assert responses.schema == reference.schema


def test_one_row_per_persona_and_model():
    responses = pl.read_parquet(RESPONSES)
    personas = pl.read_parquet(ROOT / "data" / "02_processed_data" / "personas.parquet")
    assert responses.height == personas.height * len(MODELS)
    assert responses["response_id"].n_unique() == responses.height
    assert responses.group_by("model", "cell_id").len()["len"].max() == 1


def test_every_senate_race_has_each_persona_once_per_model():
    responses = pl.read_parquet(RESPONSES)
    races = pl.read_csv(ROOT / "data" / "02_processed_data" / "senate_races.csv")
    assert responses["race_id"].n_unique() == races.height
    counts = responses.group_by("race_id", "model").len()
    assert counts["len"].to_list() == [25] * (races.height * len(MODELS))


def test_models_are_the_two_queried_models():
    responses = pl.read_parquet(RESPONSES)
    assert set(responses["model"].to_list()) == set(MODELS)


def test_each_persona_is_in_its_own_state_race():
    responses = pl.read_parquet(RESPONSES)
    personas = pl.read_parquet(ROOT / "data" / "02_processed_data" / "personas.parquet")
    races = pl.read_csv(ROOT / "data" / "02_processed_data" / "senate_races.csv")
    expected = personas.join(races.select("state_name", "race_id"),
                             left_on="state", right_on="state_name", how="left")
    matched = responses.join(expected.select("persona_id", "race_id"),
                             left_on="cell_id", right_on="persona_id", how="left",
                             suffix="_expected")
    assert matched.height == responses.height
    assert (matched["race_id"] == matched["race_id_expected"]).all()


def test_votes_are_one_zero_or_missing():
    responses = pl.read_parquet(RESPONSES)
    assert responses["vote_a"].drop_nulls().is_in([0, 1]).all()


def test_missing_votes_match_non_major_party_and_failed_answers():
    # A null vote_a should come from Other, Would not vote, or a failed call in the cache.
    responses = pl.read_parquet(RESPONSES)
    records = [json.loads(path.read_text()) for path in CACHE.glob("*.json")]
    non_major = sum(1 for record in records if record["vote"] not in ("Democratic", "Republican"))
    assert responses["vote_a"].null_count() == non_major


def test_responses_are_flagged_as_simulated():
    responses = pl.read_parquet(RESPONSES)
    assert responses["simulated"].all()
