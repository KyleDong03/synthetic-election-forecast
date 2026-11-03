# Synthetic Election Forecast

## Overview

This project forecasts the 2026 U.S. Senate elections using silicon sampling. The 2025 Cooperative Election Study (CES) is used to build 25 personas for each state with a 2026 Senate election (33 regular elections and 2 special elections). Two large language models, Claude Sonnet 5.5, from Anthropic, and GPT-6 Luna, from OpenAI, are asked how each persona would vote in its own state's race. The answers summarized into one forecast per race, and the results are displayed on the website.

## File Structure

The repo is structured as:

-   `config/` contains `senate_races.csv`, the list of the 35 Senate races (state, state FIPS, and whether the race is regular or special).
-   `data/00_simulated_data` contains simulated population and response data. No real data or API calls are required.
-   `data/01_raw_data` contains the raw CES 2025 file downloaded from Harvard Dataverse. It is not committed because of its size. `scripts/download_data.py` may be used to recreate it.
-   `data/02_processed_data` contains the race list, the FEC candidate lookup, and the 25 personas per state.
-   `data/03_silicon_sampled_data` contains the answers from both models. `cache/` holds one JSON file per persona per model with model responses, so a rerun does not occur for paid API calls. `answers.parquet`, `responses.parquet`, `other_answers.parquet`, and `forecast_summary.csv` are built from the cache.
-   `outputs/` contains summary tables from the pilot run (`pilot/`) and from the simulated example (`simulated/`).
-   `scripts/` contains the Python scripts used to download, summarize, and plot the data, and build and query the personas.
-   `sketches/` contains a sketch of the website and data flow.
-   `tests/` contains the pytest suite for the data, query, and forecast code.
-   `website/` contains the Quarto website (`website.qmd`), its styles and footer, the generated figures, and the references.

## Statement on LLM usage

We used Claude Code, with Claude Opus 5.5, to help write: scripts\build_personas.py due to the repetative nature of the code, scripts\extract_other_answers.py, scripts\plot_forecast.py, scripts\summarize_forecast.py,scripts\summarize_pilot.py, and the tests. These script and tests were all manually reviewed, before they were ran and confirmed to be working as intended. 

Claude Code, with Claude Haiku 5.5, was used in the website to apply the themes, troubleshooting the figures, and to check for typos in what was wrote. HOWEVER, AI was NOT used in creating the design of the figures, the design of the theme, or for the written prose. AI was only used to help shape the figures into what we wanted them to look like. 