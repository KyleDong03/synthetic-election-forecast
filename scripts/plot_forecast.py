"""Draw one donut chart per Senate race and write the website's summary table and figure list."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without opening windows.
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

# Chart text uses a serif font to match the report. The report's own font, Libre Caslon Text, is
# loaded from Google Fonts in the browser, so Georgia (the fallback in styles.css) is used here.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Georgia", "DejaVu Serif"]

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "03_silicon_sampled_data"
SITE = ROOT / "website"
FIGURES = SITE / "figures"

# Colours: blue for Democratic, red for Republican, grey for everything else.
DEM_COLOR = "#1f40d6"
REP_COLOR = "#e10600"
OTHER_COLOR = "#9e9e9e"
TITLE_COLOR = "#111111"  # Same black as the page's --ink colour in styles.css.
LEADER_COLOR = "#ffe38a"
# Column prefix for each model in forecast_summary.csv, and the title shown above its donut.
MODELS = {"claude": "Claude Sonnet 5.5", "gpt": "GPT-6 Luna"}


def race_title(row):
    """Return a readable race name, such as 'Alabama' or 'Florida (special election)'."""
    if row["election_type"] == "special":
        return f"{row['state_name']} (special election)"
    return row["state_name"]


def draw_donut(ax, model_name, dem, rep, other):
    """Draw the Democratic, Republican and Other shares as a full ring, with percentage pills.

    The shares sum to 100, so the ring is full. Would not vote is not drawn.
    """
    ax.pie([dem, rep, other], colors=[DEM_COLOR, REP_COLOR, OTHER_COLOR],
           startangle=90, counterclock=False, wedgeprops={"width": 0.35, "edgecolor": "white"})
    ax.set_title(model_name, color=TITLE_COLOR, fontsize=11, fontweight="bold")
    # Limits are set just past the ring and the leader pill, so no empty margin is left around them.
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.1, 1.35)
    ax.axis("equal")

    # Put each percentage pill at the middle of its slice, measured clockwise from the top.
    start = 90
    for share, colour in [(dem, DEM_COLOR), (rep, REP_COLOR), (other, OTHER_COLOR)]:
        if share == 0:
            continue  # Skip empty slices so no 0.0% pill is shown.
        angle = math.radians(start - 1.8 * share)
        ax.text(0.82 * math.cos(angle), 0.82 * math.sin(angle), f"{share:.1f}%", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold",
                bbox={"boxstyle": "round,pad=0.3", "fc": colour, "ec": "white"})
        start -= 3.6 * share

    # Leader pill above the ring: party name and margin in percentage points.
    margin = dem - rep
    if margin > 0:
        text = f"Democratic +{margin:.1f}"
    elif margin < 0:
        text = f"Republican +{-margin:.1f}"
    else:
        text = "Tie"
    ax.text(0, 1.22, text, ha="center", va="center", fontsize=9, fontweight="bold", color=TITLE_COLOR,
            bbox={"boxstyle": "round,pad=0.3", "fc": LEADER_COLOR, "ec": "none"})


def draw_race(row):
    """Save one figure with a donut for each model, the race name, and a shared legend."""
    # A narrower canvas than before, so the two donuts fill more of each image on the page.
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.75))
    for ax, (short, model_name) in zip(axes, MODELS.items()):
        draw_donut(ax, model_name, row[f"{short}_dem_perc"], row[f"{short}_rep_perc"],
                   row[f"{short}_other_perc"])
    fig.text(0.03, 0.96, race_title(row), fontsize=16, fontweight="bold", color=TITLE_COLOR, va="top")
    # The colour key is written once above the whole grid, so each chart has no legend of its own.
    fig.subplots_adjust(top=0.80, bottom=0.0, wspace=0.15)
    # A transparent background lets the charts sit directly on the page colour.
    fig.savefig(FIGURES / f"{row['race_id']}.png", dpi=120, bbox_inches="tight", pad_inches=0.3,
                transparent=True)
    plt.close(fig)


def draw_seats(summary):
    """Save a grouped bar chart of how many races each model gives to the Democrats and to the Republicans.

    A race counts for a party when that party has the larger share in the model's answers.
    Ties (for example, 10 and 10 personas) are not counted, so a bar pair can add up to less than 35.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(MODELS))
    width = 0.35
    dem_seats = [(summary[f"{short}_leader"] == "Democratic").sum() for short in MODELS]
    rep_seats = [(summary[f"{short}_leader"] == "Republican").sum() for short in MODELS]
    dem_bars = ax.bar([i - width / 2 for i in x], dem_seats, width, color=DEM_COLOR, label="Democratic")
    rep_bars = ax.bar([i + width / 2 for i in x], rep_seats, width, color=REP_COLOR, label="Republican")
    ax.bar_label(dem_bars, padding=3, color=TITLE_COLOR, fontweight="bold")
    ax.bar_label(rep_bars, padding=3, color=TITLE_COLOR, fontweight="bold")
    ax.set_xticks(list(x), list(MODELS.values()))
    ax.set_ylabel("Races won")
    # Leave headroom above the tallest bar so the legend does not cover it.
    ax.set_ylim(0, summary.height + 5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", ncol=2)
    fig.savefig(FIGURES / "seats.png", dpi=120, bbox_inches="tight", pad_inches=0.3, transparent=True)
    plt.close(fig)


def html_table(summary):
    """Return the summary as an HTML table, with each model's shares side by side.

    Democratic is blue, Republican is red, and Other is grey. The Would not vote count is a number
    of personas, not a share, and is not part of the percentages.
    """
    sub_headers = [("dem", "Democratic"), ("rep", "Republican"), ("other", "Other"),
                   ("wnv", "Would not vote")]
    # A narrow spacer column separates the two models.
    header = ('<tr><th rowspan="2">State</th>'
              + f'<th colspan="{len(sub_headers)}">{list(MODELS.values())[0]}</th>'
              + '<th rowspan="2" class="spacer"></th>'
              + f'<th colspan="{len(sub_headers)}">{list(MODELS.values())[1]}</th></tr>')
    sub_header = ("<tr>" + "".join(f'<th class="{css}">{label}</th>' for css, label in sub_headers)
                  + "".join(f'<th class="{css}">{label}</th>' for css, label in sub_headers) + "</tr>")
    rows = []
    for row in summary.iter_rows(named=True):
        cells = [f"<td>{row['state_name']}</td>"]
        for index, short in enumerate(MODELS):
            if index == 1:
                cells.append('<td class="spacer"></td>')
            cells.append(f'<td class="dem">{row[f"{short}_dem_perc"]:.1f}%</td>')
            cells.append(f'<td class="rep">{row[f"{short}_rep_perc"]:.1f}%</td>')
            cells.append(f'<td class="other">{row[f"{short}_other_perc"]:.1f}%</td>')
            cells.append(f'<td class="wnv">{row[f"{short}_would_not_vote"]}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return ('<table class="forecast">\n<thead>\n' + header + "\n" + sub_header + "\n</thead>\n<tbody>\n"
            + "\n".join(rows) + "\n</tbody>\n</table>\n")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = pl.read_csv(DATA / "forecast_summary.csv")
    races = pl.read_csv(ROOT / "data" / "02_processed_data" / "senate_races.csv").select(
        "race_id", "state_name", "election_type")
    summary = summary.join(races, on="race_id", how="left").sort("state_name")
    assert summary["state_name"].null_count() == 0
    # Each state should appear once on the page.
    assert summary["state_name"].n_unique() == summary.height

    # The race name appears only in the image title, so the page does not repeat it.
    figure_lines = []
    for row in summary.iter_rows(named=True):
        draw_race(row)
        # fig-alt gives screen readers a description without adding a caption under the chart.
        figure_lines.append(
            f"![](figures/{row['race_id']}.png){{fig-alt=\"Donut charts for {race_title(row)}\"}}\n")
    # Wrap the charts in a div so styles.css can lay them out in two columns (one on phones).
    # The one colour key sits above the grid as HTML, so it uses the page font and colours.
    key = ('<div class="race-key">'
           '<span class="key-item"><span class="key-dot dem"></span>Democratic</span>'
           '<span class="key-item"><span class="key-dot rep"></span>Republican</span>'
           '<span class="key-item"><span class="key-dot other"></span>Other</span>'
           '</div>\n\n')
    figure_block = key + "::: {.race-grid}\n\n" + "\n".join(figure_lines) + "\n:::\n"
    (SITE / "race_figures.md").write_text(figure_block, encoding="utf-8")

    (SITE / "forecast_table.md").write_text(html_table(summary), encoding="utf-8")
    draw_seats(summary)
    print(f"Saved {summary.height} race figures to {FIGURES}")
    print(f"Saved the summary table to {SITE / 'forecast_table.md'}")


if __name__ == "__main__":
    main()
