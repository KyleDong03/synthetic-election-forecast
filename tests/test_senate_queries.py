"""Check question wording and the response format without API calls or network access."""

import polars as pl

from scripts.query_personas import ROOT, make_question, to_responses_format


def test_question_names_state():
    assert "election for the United States Senate in Alabama" in make_question("Alabama")


def test_question_asks_about_parties_not_candidate_names():
    question = make_question("Alabama")
    assert "the Democratic candidate" in question
    assert "the Republican candidate" in question


def test_output_matches_responses_parquet_schema():
    records = [
        {"race_id": "2026-AL-S-regular", "persona_id": "p1", "model_requested": "m",
         "vote": "Democratic"},
        {"race_id": "2026-AL-S-regular", "persona_id": "p2", "model_requested": "m",
         "vote": "Republican"},
    ]
    responses = to_responses_format(records)
    reference = pl.read_parquet(ROOT / "data" / "00_simulated_data" / "responses.parquet")
    assert responses.schema == reference.schema


def test_votes_map_to_one_zero_or_missing():
    records = [
        {"race_id": "r1", "persona_id": "p1", "model_requested": "m", "vote": "Democratic"},
        {"race_id": "r1", "persona_id": "p2", "model_requested": "m", "vote": "Republican"},
        {"race_id": "r1", "persona_id": "p3", "model_requested": "m", "vote": "Other"},
        {"race_id": "r1", "persona_id": "p4", "model_requested": "m", "vote": "Would not vote"},
        {"race_id": "r1", "persona_id": "p5", "model_requested": "m", "vote": None},
    ]
    assert to_responses_format(records)["vote_a"].to_list() == [1, 0, None, None, None]


def test_persona_becomes_cell_and_response_ids_count_up():
    records = [
        {"race_id": "r1", "persona_id": "p1", "model_requested": "m", "vote": "Democratic"},
        {"race_id": "r1", "persona_id": "p2", "model_requested": "m", "vote": "Democratic"},
    ]
    responses = to_responses_format(records)
    assert responses["cell_id"].to_list() == ["p1", "p2"]
    assert responses["response_id"].to_list() == [0, 1]
