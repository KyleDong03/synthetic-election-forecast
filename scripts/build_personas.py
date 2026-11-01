"""Make 25 personas for each 2026 Senate race from the 2025 CES respondents. Created based on the weighted distribution of respondents in each state, to reflect the CES."""

from pathlib import Path

import numpy as np
import polars as pl

CES_PATH = Path("data/01_raw_data/ces_common_2025.parquet")
OUTPUT_PATH = Path("data/02_processed_data/personas.parquet")
ELECTION_YEAR = 2026
PERSONAS_PER_STATE = 25
SEED = 42

# State FIPS codes, converted to the actual states, as the CES stores them for the 35 states with a 2026 senate election
SENATE_STATES = {
    "1": "Alabama", "2": "Alaska", "5": "Arkansas", "8": "Colorado", "10": "Delaware",
    "12": "Florida", "13": "Georgia", "16": "Idaho", "17": "Illinois", "19": "Iowa",
    "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine", "25": "Massachusetts",
    "26": "Michigan", "27": "Minnesota", "28": "Mississippi", "30": "Montana", "31": "Nebraska",
    "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico", "37": "North Carolina",
    "39": "Ohio", "40": "Oklahoma", "41": "Oregon", "44": "Rhode Island", "45": "South Carolina",
    "46": "South Dakota", "47": "Tennessee", "48": "Texas", "51": "Virginia", "54": "West Virginia",
    "56": "Wyoming",
}

# Conversion of the CES codes to the actual meanings
GENDER = {"1": "man", "2": "woman", "3": "non-binary person", "4": "person of another gender"}
RACE = {"1": "White", "2": "Black", "3": "Hispanic", "4": "Asian", "5": "Native American",
        "6": "two or more races", "7": "other", "8": "Middle Eastern"}
EDUCATION = {"1": "no high school diploma", "2": "a high school diploma",
             "3": "some college", "4": "a 2-year degree", "5": "a 4-year degree",
             "6": "a postgraduate degree"}
PARTY = {"1": "Strong Democrat", "2": "Not very strong Democrat", "3": "Lean Democrat",
         "4": "Independent", "5": "Lean Republican", "6": "Not very strong Republican",
         "7": "Strong Republican", "8": "Not sure"}
IDEOLOGY = {"1": "Very liberal", "2": "Liberal", "3": "Moderate", "4": "Conservative",
            "5": "Very conservative", "6": "Not sure"}
REGISTERED = {"1": "Yes", "2": "No", "3": "Don't know"}

# Creates the persona text for each respondent, based on the CES responses.
def make_persona(row):
    return (
        f"You are a {row['age']}-year-old {row['gender']} who identifies as {row['race']} "
        f"and lives in {row['state']}. Your highest education is {row['education']}. "
        f"Party identification: {row['party']}. Political viewpoint: {row['ideology']}. "
        f"Registered to vote: {row['registered']}."
    )

# Converts the CES responses to a smaller table of respondents with the relevant information for making personas.
def label_respondents(ces):
    return ces.select(
        pl.col("caseid").alias("respondent_id"),
        pl.col("inputstate").replace_strict(SENATE_STATES, default=None).alias("state"),
        (ELECTION_YEAR - pl.col("birthyr").cast(pl.Int64)).alias("age"),
        pl.col("gender4").replace_strict(GENDER, default=None).alias("gender"),

        # Counts hispanic origin as Hispanic even when race was answered as something else.
        pl.when((pl.col("race") == "3") | (pl.col("hispanic") == "1")).then(pl.lit("Hispanic"))
        .otherwise(pl.col("race").replace_strict(RACE, default=None)).alias("race"),

        pl.col("educ").replace_strict(EDUCATION, default=None).alias("education"),
        pl.col("pid7").replace_strict(PARTY, default=None).alias("party"),
        pl.col("ideo5").replace_strict(IDEOLOGY, default=None).alias("ideology"),
        pl.col("votereg").replace_strict(REGISTERED, default=None).alias("registered"),
        pl.col("commonweight").cast(pl.Float64).alias("weight"),
        pl.col("inputstate"),
    )


def main():
    ces = pl.read_parquet(CES_PATH).filter(pl.col("inputstate").is_in(list(SENATE_STATES)))
    respondents = label_respondents(ces)
    rng = np.random.default_rng(SEED)

    pieces = []
    for fips, state in SENATE_STATES.items():
        pool = respondents.filter(pl.col("inputstate") == fips)
        # Respondents with larger weights are drawn more often.
        probability = pool["weight"].to_numpy() / pool["weight"].sum()
        picks = rng.choice(pool.height, size=PERSONAS_PER_STATE, replace=True, p=probability)
        pieces.append(pool[picks.tolist()])

    personas = pl.concat(pieces).drop("inputstate", "weight")
    personas = personas.with_columns(
        pl.Series("persona_id", [f"persona-{i + 1:03d}" for i in range(personas.height)]),
        pl.Series("persona", [make_persona(row) for row in personas.iter_rows(named=True)]),
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    personas.write_parquet(OUTPUT_PATH)


if __name__ == "__main__":
    main()
