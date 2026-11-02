"""Pilot summary tests."""

import json

import numpy as np
import polars as pl
import pytest

from scripts.summarize_pilot import ROOT, compare_demographics, compare_models, read_answers, summarize_answers


def example_answers():
    return pl.DataFrame({
        "race_id": ["r1"] * 4, "state": ["Alabama"] * 4,
        "model": ["m1"] * 3 + ["m2"], "model_answered": ["m1"] * 3 + ["m2"],
        "persona_id": ["p1", "p2", "p3", "p1"], "repeat_id": ["0"] * 4,
        "vote": ["Democratic", "Other", "Would not vote", "Republican"],
        "track": ["Right direction", "Wrong track", None, "Not sure"],
        "attention": ["Very closely", "Not at all", None, "Somewhat closely"],
        "confidence": [80, 40, None, 60], "cost_usd": [0.1, 0.2, 0.3, 0.4],
    })


def test_categories_use_all_answers_and_keep_non_voters():
    summary, _, _ = summarize_answers(example_answers())
    votes = summary.filter((pl.col("model") == "m1") & (pl.col("question") == "vote"))
    for answer in ["Democratic", "Other", "Would not vote"]:
        assert votes.filter(pl.col("answer") == answer)["share"][0] == pytest.approx(1 / 3)
    assert votes.filter(pl.col("answer") == "Republican")["count"][0] == 0
    missing = summary.filter((pl.col("model") == "m1") & (pl.col("question") == "track") & (pl.col("answer") == "Missing"))
    assert missing["count"][0] == 1
    assert missing["total_answers"][0] == 3
    totals = summary.group_by("race_id", "model", "question").agg(pl.col("share").sum())
    assert np.allclose(totals["share"], 1)


def test_confidence_and_costs_have_separate_model_totals():
    _, confidence, costs = summarize_answers(example_answers())
    first = confidence.filter(pl.col("model") == "m1").row(0, named=True)
    assert first["valid_answers"] == 2
    assert first["missing_answers"] == 1
    assert first["mean_confidence"] == 60
    assert costs.filter(pl.col("model") == "m1")["recorded_cost_usd"][0] == pytest.approx(0.6)
    assert costs.filter(pl.col("model") == "m2")["cached_calls"][0] == 1


def test_model_comparison_has_correct_direction_and_units():
    summary, _, _ = summarize_answers(example_answers())
    comparison = compare_models(summary)
    democratic = comparison.filter((pl.col("question") == "vote") & (pl.col("answer") == "Democratic")).row(0, named=True)
    assert democratic["model_a"] == "m1"
    assert democratic["model_b"] == "m2"
    assert democratic["difference_pp_a_minus_b"] == pytest.approx(100 / 3)


def test_demographics_use_weights_within_each_state():
    ces = pl.DataFrame({
        "state": ["Alabama", "Alabama", "Alaska"], "age": [25, 70, 40],
        "gender": ["woman", "man", "man"], "race": ["White"] * 3,
        "education": ["degree", "no degree", "degree"],
        "party": ["Independent"] * 3, "ideology": [None, "Moderate", "Liberal"],
        "registered": ["Yes"] * 3, "weight": [8.0, 2.0, 100.0],
    })
    personas = ces[[0, 0, 1, 2]].drop("weight")
    summary = compare_demographics(personas, ces)
    women = summary.filter((pl.col("state") == "Alabama") & (pl.col("variable") == "gender") & (pl.col("category") == "woman")).row(0, named=True)
    assert women["persona_share"] == pytest.approx(2 / 3)
    assert women["ces_weighted_share"] == pytest.approx(0.8)
    assert women["difference_pp"] == pytest.approx(100 * (2 / 3 - 0.8))
    assert summary.filter(pl.col("category") == "Missing").height == 1
    totals = summary.group_by("state", "variable").agg(pl.col("persona_share").sum(), pl.col("ces_weighted_share").sum())
    assert np.allclose(totals["persona_share"], 1)
    assert np.allclose(totals["ces_weighted_share"], 1)


def write_example_cache(folder):
    record = {"persona_id": "p1", "persona": "Example persona",
              "model_requested": "m1", "model_answered": "m1",
              "question": "The election for the United States Senate in Alabama",
              "vote": "Democratic", "confidence": 80,
              "track": "Right direction", "attention": "Very closely", "cost": 0.1}
    (folder / "answer.json").write_text(json.dumps(record))
    people = pl.DataFrame({"persona_id": ["p1"], "persona": ["Example persona"], "state": ["Alabama"]})
    races = pl.DataFrame({"state_name": ["Alabama"], "race_id": ["r1"]})
    return record, people, races


def test_stale_duplicate_cache_is_rejected_but_repeats_are_allowed(tmp_path):
    record, people, races = write_example_cache(tmp_path)
    (tmp_path / "duplicate.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Duplicate"):
        read_answers(tmp_path, people, races)
    record["repeat_id"] = 1
    (tmp_path / "duplicate.json").write_text(json.dumps(record))
    assert read_answers(tmp_path, people, races).height == 2


def test_invalid_answers_and_changed_personas_are_rejected(tmp_path):
    record, people, races = write_example_cache(tmp_path)
    record["vote"] = "invalid"
    (tmp_path / "answer.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Invalid vote"):
        read_answers(tmp_path, people, races)
    record["vote"] = "Democratic"
    record["persona"] = "Old profile"
    (tmp_path / "answer.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Persona mismatch"):
        read_answers(tmp_path, people, races)


def test_actual_pilot_cache_has_expected_coverage_and_totals():
    people = pl.read_parquet(ROOT / "data/02_processed_data/personas.parquet")
    races = pl.read_csv(ROOT / "data/02_processed_data/senate_races.csv")
    answers = read_answers(ROOT / "data/03_silicon_sampled_data/cache", people, races)
    assert set(answers["race_id"]) == set(races["race_id"])
    for model in answers["model"].unique():
        sample = answers.filter(pl.col("model") == model)
        assert set(sample["persona_id"]) == set(people["persona_id"])
    summary, _, costs = summarize_answers(answers)
    for model in answers["model"].unique():
        source = answers.filter(pl.col("model") == model)
        for question in ["vote", "track", "attention"]:
            result = summary.filter((pl.col("model") == model) & (pl.col("question") == question))
            assert result["count"].sum() == source.height
            assert result.filter(pl.col("answer") == "Missing")["count"].sum() == source[question].null_count()
    assert costs["recorded_cost_usd"].sum() == pytest.approx(answers["cost_usd"].sum())
