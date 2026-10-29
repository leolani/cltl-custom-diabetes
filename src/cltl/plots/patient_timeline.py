"""A timeline of one patient's activities and conditions, as recorded in the knowledge graph.

Each activity the patient took part in (an `n2mu:activity` with the patient as `sem:hasActor`, or
one mentioned in an utterance the patient said) becomes one entry on the lane of its type, e.g.
exercise, take_drink or mental_condition. Activities (things the patient does) and conditions
(states the patient is in) are drawn in two colors and in two groups of lanes: in one plot when
the lanes fit, or in two plots on a shared time axis when they don't (see `split`).

When something was the case is only known roughly. The n2mu:time/* roles of an activity decide
how it is drawn:

- dateTime: a point (or a bar, for several different dates).
- rangeTime: a bar from that date to when it was mentioned ("since last week").
- recurringTime, or an nl/eckg/EventSeries: a hatched bar, a habit from that date to the mention.
- vagueTime only: a faint bar from that date to the mention ("lately").
- no time role: a hollow point on the date of the utterance it was mentioned in.

Usage, with a LongTermMemory brain (see cltl.kg_chat.catch_up_from_kg.connect_brain()):

    entries = fetch_timeline_entries(brain, "Jan")
    fig = plot_patient_timeline(entries, "Jan", current_date=datetime.now(), days=60)

The figure remembers which mark stands for which entry (`fig.timeline_marks`), so a viewer can
make it interactive: mark_regions() gives each mark's box in the rendered image, and
describe_entry() the text to show for it (see notebooks/kg_chat_gui.py's timeline).
"""
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cltl.gaps_from_kg.get_temporal_containers import _parse_event_time

N2MU = "http://cltl.nl/leolani/n2mu/"

# Type local names that only say "this is an activity", not which kind.
GENERIC_TYPES = {"activity", "nl/eckg/EventSeries"}

# Lane order: the catch-up topics (chat_sessions.DEFAULT_GAP_ACTIVITY_TYPES) first, then the other
# types found in the graph, alphabetically. "other" is used for activities without a specific type.
ACTIVITY_LANES = ("exercise", "take_food", "take_drink", "diet", "sleep", "medication",
                  "take_medicine", "treatment", "measurement", "social")
CONDITION_LANES = ("symptom", "physical_condition", "mental_condition", "social_condition",
                   "condition")

# Colors from the dataviz reference palette (categorical slots 1 and 2), and its chart chrome.
ACTIVITY_COLOR = "#2a78d6"
CONDITION_COLOR = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# With more lanes than this, activities and conditions get a plot each (split="auto").
MAX_LANES_IN_ONE_PLOT = 12

# Entries of one lane that overlap in time are spread over at most this many rows.
MAX_ROWS_PER_LANE = 3

# A point is drawn at least this wide as a bar, so single-day conditions stay visible.
_MIN_BAR = timedelta(hours=12)

FIGURE_WIDTH = 11
ROW_INCHES = 0.32
# The width of one character of a 7pt label, to reserve room for labels.
CHAR_INCHES = 0.055
# Room for one legend entry, and the height of a row of them.
LEGEND_ENTRY_INCHES = 1.45
LEGEND_ROW_INCHES = 0.22


def timeline_query(human: str) -> str:
    """One row per activity of `human`: its types, a label, its n2mu:time/* values as
    "role=value" pairs and the earliest timestamp of an utterance it was mentioned in.

    The patient is matched on the local name of the actor or speaker, ignoring case, since the
    same person is both <friends/Jan> (the speaker) and <world/Jan> (extracted from text)."""
    name = human.lower().replace('"', '')
    return """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX sem: <http://semanticweb.cs.vu.nl/2009/11/sem/>
        PREFIX gaf: <http://groundedannotationframework.org/gaf#>
        PREFIX grasp: <http://groundedannotationframework.org/grasp#>
        SELECT ?id
               (GROUP_CONCAT(DISTINCT STRAFTER(STR(?type), "%(n2mu)s"); SEPARATOR="|") AS ?types)
               (SAMPLE(?label) AS ?activity_label)
               (GROUP_CONCAT(DISTINCT CONCAT(STRAFTER(STR(?time_p), "%(n2mu)stime/"), "=", STR(?time));
                             SEPARATOR=" ") AS ?times)
               (MIN(?said) AS ?mentioned)
        WHERE {
            ?id a <%(n2mu)sactivity>, ?type .
            FILTER(STRSTARTS(STR(?type), "%(n2mu)s"))
            { ?id sem:hasActor ?actor .
              FILTER(LCASE(REPLACE(STR(?actor), "^.*/", "")) = "%(name)s") }
            UNION
            { ?id gaf:denotedIn ?own_utterance .
              ?own_utterance grasp:wasAttributedTo ?speaker .
              FILTER(LCASE(REPLACE(STR(?speaker), "^.*/", "")) = "%(name)s") }
            OPTIONAL { ?id rdfs:label ?label . }
            OPTIONAL { ?id ?time_p ?time . FILTER(STRSTARTS(STR(?time_p), "%(n2mu)stime/")) }
            OPTIONAL { ?id gaf:denotedIn ?utterance .
                       ?utterance sem:hasBeginTimeStamp ?said . FILTER(isLiteral(?said)) }
        }
        GROUP BY ?id
    """ % {"n2mu": N2MU, "name": name}


def _value(row: Dict, key: str) -> str:
    return row[key]["value"] if key in row else ""


def _parse_mentioned(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def entry_from_row(row: Dict) -> Optional[Dict]:
    """One timeline entry from a timeline_query() result row (SPARQL JSON bindings), or None
    when the row has no date at all.

    An entry is {"id", "chat", "type", "group", "label", "start", "end", "kind", "times",
    "mentioned"}, where "group" is "condition" or "activity", "kind" is one of "point", "range",
    "recurring", "vague" or "mentioned" (see the module docstring), "times" maps each n2mu:time/*
    role the activity has to its dates, and "mentioned" is when it was first mentioned."""
    activity_id = _value(row, "id")
    types = [t for t in _value(row, "types").split("|") if t]
    specific = [t for t in types if t not in GENERIC_TYPES]
    activity_type = specific[0] if specific else "other"

    times = {"dateTime": [], "rangeTime": [], "recurringTime": [], "vagueTime": []}
    for pair in _value(row, "times").split():
        role, _, value = pair.partition("=")
        date = _parse_event_time(value) if value else None
        if role in times and date is not None:
            times[role].append(date)
    mentioned = _parse_mentioned(_value(row, "mentioned"))

    since = times["rangeTime"] + times["recurringTime"]
    if since:
        start = min(since)
        end = max(times["dateTime"] + ([mentioned] if mentioned else []) + since)
        recurring = times["recurringTime"] or "nl/eckg/EventSeries" in types
        kind = "recurring" if recurring else "range"
    elif times["dateTime"]:
        start, end = min(times["dateTime"]), max(times["dateTime"])
        kind = "point" if start.date() == end.date() else "range"
    elif times["vagueTime"]:
        start = min(times["vagueTime"])
        end = max([mentioned or start] + times["vagueTime"])
        kind = "vague"
    elif mentioned:
        start = end = mentioned
        kind = "mentioned"
    else:
        return None
    if start > end:
        start, end = end, start

    local_name = activity_id.rsplit("/", 1)[-1]
    label = _value(row, "activity_label")
    # Activities without a label of their own are labelled with their id.
    if label == local_name:
        label = ""
    chat = local_name.split(".", 1)[0] if local_name.startswith("chat") else None
    return {
        "id": activity_id,
        "chat": chat,
        "type": activity_type,
        "group": "condition" if _is_condition(activity_type) else "activity",
        "label": label,
        "start": start,
        "end": end,
        "kind": kind,
        "times": {role: sorted(dates) for role, dates in times.items() if dates},
        "mentioned": mentioned,
    }


def _is_condition(activity_type: str) -> bool:
    return activity_type in CONDITION_LANES or activity_type.endswith("condition")


def fetch_timeline_entries(brain, human: str) -> List[Dict]:
    """All dated timeline entries for `human` in the knowledge graph behind `brain` (a
    cltl.brain LongTermMemory), sorted by start date."""
    rows = brain._submit_query(timeline_query(human))
    entries = [entry for entry in (entry_from_row(row) for row in rows) if entry]
    return sorted(entries, key=lambda entry: entry["start"])


def _lanes(entries: Iterable[Dict], group: str) -> List[str]:
    order = ACTIVITY_LANES if group == "activity" else CONDITION_LANES
    present = {entry["type"] for entry in entries if entry["group"] == group}
    known = [lane for lane in order if lane in present]
    return known + sorted(present - set(known))


def _short(label: str, length: int = 24) -> str:
    return label if len(label) <= length else label[:length - 1].rstrip() + "…"


def _layout(entries: List[Dict], lanes: List[str], window_start: datetime, window_end: datetime,
            annotate: bool, width_inches: float):
    """Pack the entries of each lane into at most MAX_ROWS_PER_LANE rows, so entries (and their
    labels) that overlap in time are drawn above each other instead of on top of each other.

    Returns ([(entry, lane, row, label or None), ...], {lane: number of rows}). A label is kept
    only when it fits in its row; an entry that fits in no row goes in the row that frees up
    first, without its label."""
    window_days = max((window_end - window_start).total_seconds() / 86400, 1)
    days_per_inch = window_days / width_inches
    marker_days = timedelta(days=0.08 * days_per_inch)
    placed, rows_used = [], {}
    for lane in lanes:
        row_ends: List[datetime] = []
        seen_labels = set()
        for entry in sorted((e for e in entries if e["type"] == lane), key=lambda e: e["start"]):
            start = max(entry["start"], window_start)
            end = max(entry["end"], start + marker_days)
            label = _short(entry["label"]) if annotate and entry["label"] else None
            if label and label.lower() in seen_labels:
                label = None
            label_end = start + timedelta(days=(len(label) * CHAR_INCHES + 0.1) * days_per_inch) \
                if label else start
            needed = max(end, label_end)
            row = next((i for i, row_end in enumerate(row_ends) if row_end <= start), None)
            if row is None and len(row_ends) < MAX_ROWS_PER_LANE:
                row = len(row_ends)
                row_ends.append(needed)
            elif row is None:
                row = min(range(len(row_ends)), key=lambda i: row_ends[i])
                label, needed = None, end
                row_ends[row] = max(row_ends[row], needed)
            else:
                row_ends[row] = needed
            if label:
                seen_labels.add(label.lower())
            placed.append((entry, lane, row, label))
        rows_used[lane] = max(1, len(row_ends))
    return placed, rows_used


def _draw_lanes(ax, entries: List[Dict], lanes: List[str], color: Dict[str, str],
                highlight_chat: Optional[str], annotate: bool, window_start: datetime,
                window_end: datetime, width_inches: float, marks: List) -> int:
    """Draw `lanes` top to bottom on `ax`; "" in `lanes` is an empty separator row. `color` maps
    an entry's group to its color. Appends an (entry, artist) pair per drawn mark to `marks`.
    Returns the number of rows drawn."""
    placed, rows_used = _layout(entries, [lane for lane in lanes if lane], window_start,
                                window_end, annotate, width_inches)
    # Each lane is a block of rows; row 0 is the top row of its block.
    lane_top, y = {}, 0
    for lane in lanes:
        lane_top[lane] = y
        y += rows_used.get(lane, 1) if lane else 0.6
    total = y

    for entry, lane, row, label in placed:
        y = total - lane_top[lane] - row - 0.5
        start, end, kind = entry["start"], entry["end"], entry["kind"]
        fill = color[entry["group"]]
        highlighted = highlight_chat is not None and entry["chat"] == highlight_chat
        if kind in ("point", "mentioned"):
            artist, = ax.plot(start, y, marker="o", markersize=8, linestyle="none", zorder=4,
                    markerfacecolor=fill if kind == "point" else SURFACE,
                    markeredgecolor=INK if highlighted else fill,
                    markeredgewidth=2 if highlighted else 1.5)
        else:
            artist = ax.barh(y, max(end - start, _MIN_BAR), left=start, height=0.4, zorder=3,
                             color=fill, alpha=0.35 if kind == "vague" else 0.85,
                             hatch="///" if kind == "recurring" else None,
                             edgecolor=INK if highlighted else SURFACE,
                             linewidth=1.5 if highlighted else 1).patches[0]
        marks.append((entry, artist))
        if label:
            ax.annotate(label, (max(start, window_start), y + 0.22), xytext=(0, 1),
                        textcoords="offset points", fontsize=7, color=INK_SECONDARY,
                        va="bottom", ha="left", zorder=5, clip_on=True)

    ticks = [(total - lane_top[lane] - rows_used[lane] / 2, lane) for lane in lanes if lane]
    ax.set_yticks([tick for tick, _ in ticks])
    ax.set_yticklabels([lane.replace("_", " ") for _, lane in ticks])
    # A hairline between lanes, so the rows of one lane read as a block.
    for lane in lanes[1:]:
        if lane:
            ax.axhline(total - lane_top[lane], color=GRID, linewidth=0.6, zorder=1)
    ax.set_ylim(0, total)
    return total


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)


def _mark_dates(ax, marks: Dict[str, datetime], with_labels: bool, window_start: datetime,
                window_end: datetime):
    """Dashed lines at `marks`, each labelled to its left: the last one (now) is near the right
    edge, and so neighboring labels don't run into each other. A label close to the next mark's
    (e.g. a long period, where the last conversation is just before now) goes a line higher. A
    mark before `window_start` is left out, rather than its label sticking out of the plot."""
    shown = [(name, date) for name, date in marks.items() if date >= window_start]
    window = (window_end - window_start).total_seconds()
    for i, (name, date) in enumerate(shown):
        ax.axvline(date, color=INK_MUTED, linestyle="--", linewidth=1, zorder=2)
        if not with_labels:
            continue
        close = i + 1 < len(shown) and (shown[i + 1][1] - date).total_seconds() / window < 0.25
        ax.annotate(name, (date, 1), xycoords=("data", "axes fraction"), xytext=(-3, 13 if close else 2),
                    textcoords="offset points", fontsize=8, color=INK_SECONDARY,
                    va="bottom", ha="right")


def _legend(fig, groups: List[str], width: float) -> int:
    """Add the legend below the plot(s), in as many columns as fit in `width` inches. Returns
    the number of rows it takes."""
    handles = []
    if "activity" in groups:
        handles.append(Patch(color=ACTIVITY_COLOR, label="activity"))
    if "condition" in groups:
        handles.append(Patch(color=CONDITION_COLOR, label="condition"))
    handles += [
        Line2D([], [], marker="o", linestyle="none", color=INK_MUTED, markersize=7,
               label="on a date"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=SURFACE,
               markeredgecolor=INK_MUTED, markersize=7, label="date of mention only"),
        Patch(facecolor=INK_MUTED, alpha=0.85, label="since a date"),
        Patch(facecolor=INK_MUTED, alpha=0.85, hatch="///", edgecolor=SURFACE,
              label="recurring since a date"),
        Patch(facecolor=INK_MUTED, alpha=0.35, label="vaguely, since a date"),
    ]
    columns = max(1, min(len(handles), int(width / LEGEND_ENTRY_INCHES)))
    fig.legend(handles=handles, loc="lower center", ncol=columns, frameon=False,
               fontsize=8, labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 0), handlelength=1.5)
    return -(-len(handles) // columns)


def plot_patient_timeline(entries: List[Dict], human: str, current_date: Optional[datetime] = None,
                          days: Optional[int] = 60, last_conversation_date: Optional[datetime] = None,
                          highlight_chat=None, annotate: bool = True, split="auto",
                          title: Optional[str] = None, width: float = FIGURE_WIDTH):
    """Plot `entries` (see fetch_timeline_entries()) as a timeline with a lane per type.

    The time axis runs from `days` days before `current_date` (default: now) to just after it;
    days=None shows everything. `current_date` and `last_conversation_date` are marked with a
    dashed line. Entries of chat `highlight_chat` (a chat id such as the notebooks' CHAT_ID, or
    "chat<id>") get a dark outline, e.g. to show what the current conversation added. With
    `annotate`, entries are labelled with their activity label where there is room.

    split: True draws activities and conditions in two plots on a shared time axis, False in one
    plot, "auto" in one plot unless there are more than MAX_LANES_IN_ONE_PLOT lanes.

    width: of the figure in inches; its height follows from the number of rows. Labels and the
    legend are laid out for that width, so a narrow figure is not just a scaled-down wide one.

    Returns the matplotlib Figure."""
    current_date = current_date or datetime.now()
    if highlight_chat is not None and not str(highlight_chat).startswith("chat"):
        highlight_chat = f"chat{highlight_chat}"
    if days is None:
        window_start = min([e["start"] for e in entries] + [current_date]) - timedelta(days=1)
    else:
        window_start = current_date - timedelta(days=days)
    window_end = current_date + timedelta(days=max(1, (current_date - window_start).days // 30))
    shown = [e for e in entries if e["end"] >= window_start and e["start"] <= window_end]

    activity_lanes = _lanes(shown, "activity")
    condition_lanes = _lanes(shown, "condition")
    if split == "auto":
        split = len(activity_lanes) + len(condition_lanes) > MAX_LANES_IN_ONE_PLOT
    if split and activity_lanes and condition_lanes:
        plots = [(activity_lanes, "activities"), (condition_lanes, "conditions")]
    else:
        # One plot: the activity lanes above, a separator, then the condition lanes.
        separator = [""] if activity_lanes and condition_lanes else []
        plots = [(activity_lanes + separator + condition_lanes, None)]

    colors = {"activity": ACTIVITY_COLOR, "condition": CONDITION_COLOR}
    # Roughly the width left for the time axis, next to the lane names.
    width_inches = max(width - 2, 1.5)
    # The row counts are only known after the layout, so draw first and size the figure after.
    fig, axes = plt.subplots(len(plots), 1, sharex=True, squeeze=False, figsize=(width, 4))
    axes = list(axes[:, 0])
    heights = []
    fig.timeline_marks = []
    for ax, (lanes, ylabel) in zip(axes, plots):
        _style_axis(ax)
        if lanes:
            heights.append(_draw_lanes(ax, shown, lanes, colors, highlight_chat, annotate,
                                       window_start, window_end, width_inches, fig.timeline_marks))
        else:
            heights.append(2)
            ax.set_yticks([])
            ax.text(0.5, 0.5, f"No dated activities or conditions for {human} in this period",
                    transform=ax.transAxes, ha="center", va="center", color=INK_SECONDARY)
        if ylabel:
            ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
        for label in ax.get_yticklabels():
            label.set_color(INK)
            label.set_fontsize(9)
    if len(axes) > 1:
        axes[0].get_gridspec().set_height_ratios(heights)

    marks = {"now": current_date}
    if last_conversation_date is not None:
        marks = {"last conversation": last_conversation_date, **marks}
    for i, ax in enumerate(axes):
        _mark_dates(ax, marks, with_labels=i == 0, window_start=window_start, window_end=window_end)
    axes[-1].set_xlim(window_start, window_end)
    # About one date per 0.6 inch of time axis, so short periods in a narrow plot stay readable.
    locator = mdates.AutoDateLocator(minticks=3, maxticks=max(3, min(10, int(width_inches / 0.6))))
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(title or f"What was the case for {human}, roughly", x=0.01, ha="left",
                 fontsize=12, color=INK)
    legend_rows = _legend(fig, sorted({e["group"] for e in shown}), width)
    legend_height = LEGEND_ROW_INCHES * legend_rows + 0.2
    height = ROW_INCHES * sum(heights) + 1.4 + legend_height
    fig.set_size_inches(width, height)
    fig.tight_layout(rect=(0, legend_height / height, 1, 1 - 0.35 / height))
    return fig


def mark_regions(fig, renderer, min_size: float = 12) -> List:
    """[(entry, (x0, y0, x1, y1)), ...]: the box of every mark in `fig` (see
    plot_patient_timeline()) in pixels of the image `renderer` drew, from its top left corner,
    topmost mark first. Call it after drawing the figure at the resolution it is shown at.

    A bar that starts before the window is cut off at the edge of its plot, as drawn. A box is
    at least `min_size` pixels in each direction, so a small dot is still easy to point at."""
    height = fig.bbox.height
    regions = []
    for entry, artist in reversed(getattr(fig, "timeline_marks", [])):
        box = artist.get_window_extent(renderer)
        plot_box = artist.axes.bbox
        x0, x1 = max(box.x0, plot_box.x0), min(box.x1, plot_box.x1)
        if x1 < x0:
            continue
        y0, y1 = height - box.y1, height - box.y0
        grow_x, grow_y = max(0, min_size - (x1 - x0)) / 2, max(0, min_size - (y1 - y0)) / 2
        regions.append((entry, (x0 - grow_x, y0 - grow_y, x1 + grow_x, y1 + grow_y)))
    return regions


def _day(date: datetime) -> str:
    return date.strftime("%a %d %b %Y")


def describe_when(entry: Dict) -> str:
    """When `entry` was the case, in words, e.g. "since Mon 21 Sep 2026, until mentioned on
    Mon 28 Sep 2026"."""
    start, end, kind = entry["start"], entry["end"], entry["kind"]
    until = f", until mentioned on {_day(end)}"
    if kind == "point":
        return f"on {_day(start)}"
    if kind == "mentioned":
        return f"no date given; mentioned on {_day(start)}"
    if kind == "recurring":
        return f"recurring since {_day(start)}{until}"
    if kind == "vague":
        return f"vaguely since {_day(start)}{until}"
    if "rangeTime" in entry.get("times", {}):
        return f"since {_day(start)}{until}"
    return f"from {_day(start)} to {_day(end)}"


def describe_entry(entry: Dict) -> str:
    """A few lines on `entry`: what it is, when it was the case and where it comes from."""
    lines = [f"{entry['type'].replace('_', ' ')} ({entry['group']})"]
    if entry["label"]:
        lines.append(f"\u201c{entry['label']}\u201d")
    lines.append(describe_when(entry))
    if entry.get("chat"):
        lines.append(f"from {entry['chat']}")
    return "\n".join(lines)


def plot_patient_timeline_from_kg(brain, human: str, **kwargs):
    """fetch_timeline_entries() and plot_patient_timeline() in one call; `kwargs` go to the
    latter. Returns the matplotlib Figure."""
    return plot_patient_timeline(fetch_timeline_entries(brain, human), human, **kwargs)
