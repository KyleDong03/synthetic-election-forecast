"""Build the race list and download an FEC candidate lookup, not a final ballot."""

import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
FEC_URL = "https://www.fec.gov/files/bulk-downloads/2026/cn26.zip"
FEC_COLUMNS = [
    "candidate_id", "candidate_name", "party", "election_year", "state",
    "office", "district", "incumbent_status", "fec_status", "committee_id",
    "street_1", "street_2", "city", "mailing_state", "zip_code",
]


def load_races():
    races = pl.read_csv(ROOT / "config" / "senate_races.csv")
    if races["state"].n_unique() != races.height:
        raise ValueError("This roster expects one Senate race per listed state.")
    return races.with_columns(
        (pl.lit("2026-") + pl.col("state") + pl.lit("-S-") + pl.col("election_type")).alias("race_id"),
        pl.lit("2026-11-03").alias("election_date"),
    ).sort("state")


def select_candidates(archive, races):
    with ZipFile(BytesIO(archive)) as zipped:
        candidates = pl.read_csv(
            zipped.read("cn.txt"), separator="|", has_header=False,
            new_columns=FEC_COLUMNS, infer_schema=False,
        )
    # FEC records can include primary losers, withdrawals, and other non-ballot
    # candidates. Neither election year nor FEC status proves ballot eligibility.
    candidates = candidates.filter(
        (pl.col("office") == "S") & (pl.col("election_year") == "2026")
    ).join(races.select("state", "race_id"), on="state", how="inner")
    candidates = candidates.select(
        "race_id", "state", "candidate_id", "candidate_name", "party", "fec_status"
    ).with_columns(pl.lit(False).alias("ballot_verified")).sort("race_id", "candidate_name")
    if candidates.is_empty():
        raise ValueError("No matching Senate candidate records were found.")
    if candidates["candidate_id"].null_count() or candidates["candidate_id"].n_unique() != candidates.height:
        raise ValueError("Candidate IDs must be present and unique.")
    return candidates


def main():
    raw = ROOT / "data" / "01_raw_data"
    output = ROOT / "data" / "02_processed_data"
    raw.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = raw / "fec_candidates_2026.zip"
    # Reuse the saved snapshot so reruns do not silently change candidate records.
    if not archive_path.exists():
        with urlopen(FEC_URL, timeout=60) as response:
            archive = response.read()
        candidates = select_candidates(archive, load_races())
        archive_path.write_bytes(archive)
        metadata = {
            "source_url": FEC_URL,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(archive).hexdigest(),
        }
        (raw / "fec_candidates_2026_source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    else:
        candidates = select_candidates(archive_path.read_bytes(), load_races())

    races = load_races()
    races.write_csv(output / "senate_races.csv")
    races.write_parquet(output / "senate_races.parquet")
    candidates.write_csv(output / "senate_candidate_lookup.csv")
    candidates.write_parquet(output / "senate_candidate_lookup.parquet")
    print(f"Saved {races.height} races and {candidates.height} FEC candidate lookup records.")
    print("These candidate records are NOT a certified general-election ballot list.")
    print("Verify candidates against state election-office lists before writing prompts.")
    print(f"Files: {output}")


if __name__ == "__main__":
    main()
