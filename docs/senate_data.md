# Senate race and candidate data: first stage

## Run

```sh
uv run python scripts/download_senate_data.py
uv run pytest -q
```

No API key or additional dependency is needed. Run from the project folder.

## Race roster

`config/senate_races.csv` is a manually transcribed roster of the 35 contests:
33 regular races and the Florida and Ohio special elections. State codes are
two-letter postal codes; `state_fips` is a numeric state code that can be matched
to the CES `inputstate` after converting that column to an integer.
Each race gets an ID such as `2026-FL-S-special`. The main election date is
2026-11-03; this stage does not model later runoffs or ranked-choice transfers.

Sources checked on 2026-10-31:

- FEC schedule, states marked S and general-election date (schedule dated 2026-05-18):
  https://www.fec.gov/resources/cms-content/documents/2026pdates.pdf
- USPS list of all 35 Senate states:
  https://about.usps.com/postal-bulletin/2026/pb22712/html/cover_003.htm
- Ohio special-election writ:
  https://www.ohiosos.gov/assets/writ-of-election-us-senate-may-2026.pdf
- Florida Senate special election:
  https://ballotpedia.org/United_States_Senate_special_election_in_Florida%2C_2026
- CES state-code labels: the `inputstate` section of CES25_Common.pdf,
  https://doi.org/10.7910/DVN/DFUGJR

If a new special election is added, update the roster and its coverage test.
The current script expects one contest per listed state; that would also need
adjustment if a state held two Senate contests in this cycle.

## FEC candidate lookup

The script downloads and retains the official candidate master ZIP:
https://www.fec.gov/files/bulk-downloads/2026/cn26.zip

It reads the pipe-separated `cn.txt` using the FEC's documented columns:
https://www.fec.gov/campaign-finance-data/candidate-master-file-description/

It retains Senate records with election year 2026 in the roster states. The output
keeps FEC IDs, names, party codes, and statutory-candidate status, excluding addresses.
Raw downloads are ignored by Git. A JSON file alongside the ZIP records retrieval
time, source URL, and SHA-256. Reruns reuse that ZIP. The live FEC file changes over
time, so fresh downloads can differ from this snapshot.

## Outputs

Under `data/02_processed_data/`:

- `senate_races.csv` and `.parquet`: one row per contest, including state/FIPS,
  election type, race ID, and election date.
- `senate_candidate_lookup.csv` and `.parquet`: one row per matching FEC candidate.
  `party` and `fec_status` retain the original FEC codes. `ballot_verified` is false.

IMPORTANT: this is a candidate lookup, NOT the final list of ballot candidates.
FEC records include primary losers, withdrawals, and others not on the November
ballot. Status C means statutory candidate, not primary winner or ballot-certified.
Do not feed this table directly into survey prompts.

## Next step

Check every race against its state election office's certified candidate list.
Create a separate final ballot table with display names, parties, source URLs,
verification dates, and any withdrawal/write-in policy. Include independents and
minor parties according to a documented rule. FEC IDs are useful for matching but
should not be required when an official ballot candidate has no matching FEC record.
Also record race-specific election rules before calculating actual win probabilities.
