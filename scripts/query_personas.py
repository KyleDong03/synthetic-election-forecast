"""Ask each 2026 Senate persona about its own state's race, using one OpenAI and one Anthropic model."""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "no-key-set"),
    timeout=60,
    max_retries=2,
)

MODELS = ["anthropic/claude-sonnet-5.5", "openai/gpt-6-luna"]
SCHEMA = {
    "type": "object",
    "properties": {
        "vote": {"type": "string", "enum": ["Democratic", "Republican", "Other", "Would not vote"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "track": {"type": "string", "enum": ["Right direction", "Wrong track", "Not sure"]},
        "attention": {"type": "string", "enum": ["Very closely", "Somewhat closely",
                                                 "Not too closely", "Not at all"]},
    },
    "required": ["vote", "confidence", "track", "attention"],
    "additionalProperties": False,
}

BUDGET_USD = 20.00
CACHE = ROOT / "data" / "03_silicon_sampled_data" / "cache"
OUTPUT = ROOT / "data" / "03_silicon_sampled_data"
spent = 0.0


# Creates the question for one state's Senate race.
def make_question(state):
    return (
        f"The election for the United States Senate in {state} is on Tuesday 3 November 2026. "
        "If it were held today, would you vote for the Democratic candidate, the Republican "
        "candidate, someone else, or would you not vote? How confident are you, from 0 to 100? "
        "Generally, would you say things in the United States are headed in the right direction "
        "or off on the wrong track? "
        f"How closely are you following the Senate race in {state}: very closely, "
        "somewhat closely, not too closely, or not at all?"
    )

# Asks one persona about its state's Senate race, using the specified model.
def ask(model, persona_id, persona, question):
    global spent
    raw = f"{model}|{persona_id}|{persona}|{question}"
    path = CACHE / (hashlib.sha256(raw.encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text())
    if spent >= BUDGET_USD:
        raise RuntimeError(f"Budget of ${BUDGET_USD} reached")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": persona}, {"role": "user", "content": question}],
        temperature=1.0,
        max_tokens=1000,  # some models count reasoning tokens against this limit
        response_format={"type": "json_schema",
                         "json_schema": {"name": "answers", "strict": True, "schema": SCHEMA}},
    )
    spent += response.usage.cost
    try:
        answer = json.loads(response.choices[0].message.content)
        answer = {"vote": answer["vote"], "confidence": answer["confidence"],
                  "track": answer["track"], "attention": answer["attention"]}
    except (TypeError, ValueError, KeyError):
        answer = {"vote": None, "confidence": None, "track": None, "attention": None}
    record = {
        "model_requested": model,
        "model_answered": response.model,
        "persona_id": persona_id,
        "persona": persona,
        "question": question,
        "cost": response.usage.cost,
        **answer,
    }
    path.write_text(json.dumps(record))
    return record


# Asks one persona about its race, tagged with the race ID.
def run(job):
    model, row = job
    question = make_question(row["state"])
    return {"race_id": row["race_id"], **ask(model, row["persona_id"], row["persona"], question)}


# Reshapes the answers to a better format for analysis.
# Each persona is one cell. Democratic is 1, Republican is 0, and Other or Would not vote is missing.
def to_responses_format(records):
    return pl.DataFrame(records).select(
        pl.int_range(pl.len(), dtype=pl.Int64).alias("response_id"),
        pl.col("race_id"),
        pl.col("persona_id").alias("cell_id"),
        pl.col("model_requested").alias("model"),
        pl.when(pl.col("vote") == "Democratic").then(1)
        .when(pl.col("vote") == "Republican").then(0)
        .otherwise(None).cast(pl.Int64).alias("vote_a"),
        pl.lit(True).alias("simulated"),
    )


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    personas = pl.read_parquet(ROOT / "data" / "02_processed_data" / "personas.parquet")
    races = pl.read_csv(ROOT / "data" / "02_processed_data" / "senate_races.csv")
    personas = personas.join(races.select("state_name", "race_id"),
                             left_on="state", right_on="state_name", how="left")

    jobs = [(model, row) for model in MODELS for row in personas.iter_rows(named=True)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(run, jobs))
    to_responses_format(records).write_parquet(OUTPUT / "responses.parquet")
    print(f"Spent ${spent:.4f}")


if __name__ == "__main__":
    main()
