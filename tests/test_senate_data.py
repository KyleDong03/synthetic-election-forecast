"""Check race coverage and FEC filtering without network requests."""

from io import BytesIO
from zipfile import ZipFile

from scripts.download_senate_data import load_races, select_candidates


def test_race_roster():
    races = load_races()
    assert races.height == 35
    assert races["race_id"].n_unique() == 35
    assert races["state_fips"].n_unique() == 35
    assert races.filter(races["election_type"] == "special")["state"].to_list() == ["FL", "OH"]
    assert races["election_date"].unique().to_list() == ["2026-11-03"]


def test_candidate_filter_does_not_claim_ballot_verification():
    # Four example FEC records: keep only the 2026 Senate record in a listed state.
    records = [
        "S1AL00001|EXAMPLE, ONE|DEM|2026|AL|S|00|C|C||||||",
        "H1AL00001|EXAMPLE, TWO|REP|2026|AL|H|01|C|C||||||",
        "S1AL00002|EXAMPLE, THREE|REP|2028|AL|S|00|C|C||||||",
        "S1CA00001|EXAMPLE, FOUR|REP|2026|CA|S|00|C|C||||||",
    ]
    buffer = BytesIO()
    with ZipFile(buffer, "w") as zipped:
        zipped.writestr("cn.txt", "\n".join(records))
    candidates = select_candidates(buffer.getvalue(), load_races())
    assert candidates.height == 1
    assert candidates["candidate_id"][0] == "S1AL00001"
    assert candidates["race_id"][0] == "2026-AL-S-regular"
    assert candidates["ballot_verified"][0] is False
