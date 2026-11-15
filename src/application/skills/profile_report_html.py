"""Render a :class:`ProfileReport` as one self-contained HTML page.

The profile mode of the person report: same page style as
``person_report_html`` (inline CSS and SVG, no request, printable,
phone width, light or dark, every colour doubled by a shape and a
word), with one strip of sessions per competency.
"""

import math
from html import escape
from typing import List

from src.application.skills.person_report import (
    RECOMMENDED_SITUATIONS,
    note,
)
from src.application.skills.person_report_html import (
    STYLE,
    _legend_icon,
    _mark,
    _text,
)
from src.application.skills.person_report_texts import TEXTS
from src.application.skills.profile_report import (
    Cell,
    ProfileReport,
    ProfileRow,
    RepeatedPoint,
    decimal,
    level_text,
    quote_when,
    reliability_word,
)
from src.application.skills.profile_report_texts import PROFILE_TEXTS

PROFILE_STYLE = """
.strip{grid-column:1 / -1;display:flex;flex-wrap:wrap;gap:6px;
list-style:none;padding:0;margin:6px 0 2px}
.cell{width:54px;min-height:60px;border:2px solid var(--line);
border-radius:8px;background:var(--card);text-align:center;
padding:2px 0 4px;display:flex;flex-direction:column;
justify-content:space-between}
.cell .n{font-size:.8rem;color:var(--muted);line-height:1.3}
.cell .v{font-size:1.2rem;font-weight:700;line-height:1.2;display:flex;
align-items:center;justify-content:center;gap:3px}
.cell.none{border-style:dashed;background:transparent}
.cell.none .v{color:var(--none);font-weight:400}
.cell.strong{border-color:var(--strong)}.cell.weak{border-color:var(--weak)}
.tag.building{color:var(--muted)}
.mini{flex:none;width:22px;height:22px;border:2px dashed var(--none);
border-radius:5px;display:inline-flex;align-items:center;
justify-content:center;color:var(--none);font-size:.8rem;line-height:1}
.mini.building{border-style:solid;border-color:var(--line)}
.unseen{margin-top:16px}.unseen ul{margin:4px 0 0;padding-left:1.2em}
.row .lines{grid-column:1 / -1;margin:4px 0 0;padding:0;list-style:none;
font-size:.95rem}
.row .lines li{margin:2px 0}
.sessions{padding-left:1.4em}.sessions li{margin:4px 0}
@media print{.cell{break-inside:avoid}}
"""


def _cell(cell: Cell, report: ProfileReport) -> str:
    """Return one session's box in a competency strip."""
    words = PROFILE_TEXTS[report.language]
    if cell.rating is None:
        label = words["cell_none"].format(number=cell.number)
        value = "–"
    else:
        label = words["cell_level"].format(
            number=cell.number, rating=cell.rating,
            band=TEXTS[report.language][f"band_{cell.band}"].lower(),
        )
        icon = (
            f'<svg width="14" height="14" viewBox="0 0 22 22" '
            f'aria-hidden="true">{_mark(cell.band, 11, 12, 8)}</svg>'
        )
        value = f"{icon}{cell.rating}"
    return (
        f'<li class="cell {cell.band}" aria-label="{escape(label)}" '
        f'title="{escape(label)}"><span class="n" aria-hidden="true">'
        f'{cell.number}</span><span class="v" aria-hidden="true">{value}'
        f'</span></li>'
    )


def _lines(row: ProfileRow, report: ProfileReport) -> List[str]:
    """Return the plain sentences under a competency strip."""
    words = PROFILE_TEXTS[report.language]
    lines: List[str] = []
    if row.low is not None and row.high is not None and row.rated >= 2:
        key = (
            "same" if row.low == row.high
            else "consistent" if row.consistent else "varies"
        )
        lines.append(words[key].format(
            low=row.low, high=row.high, rated=row.rated
        ))
    if row.trend in ("up", "down", "flat"):
        lines.append(words[f"trend_{row.trend}"])
    if row.expected_reliability is not None:
        value = row.expected_reliability
        key = (
            "reliability_ok" if value >= 0.70 or
            row.rated >= row.situations_for_target else "reliability"
        )
        text = words[key].format(
            n6=note(6), word=reliability_word(value, report.language),
            value=decimal(value, report.language),
            target=row.situations_for_target,
        )
        if row.measured_reliability is not None:
            text += words["measured"].format(
                value=decimal(row.measured_reliability, report.language)
            )
        lines.append(text)
    if row.kinds and row.rated:
        lines.append(words["kinds_seen"].format(kinds=", ".join(row.kinds)))
    return lines


def _row(row: ProfileRow, report: ProfileReport) -> str:
    """Return one competency: overall level, strip and what it means."""
    words = PROFILE_TEXTS[report.language]
    if row.status == "aggregated" and row.aggregate is not None:
        score = (
            f"{level_text(row.aggregate, report.language)} "
            f"<small>/ {report.maximum}</small>"
        )
        tag_class: str = row.band
        tag = (
            f'{words["status_aggregated"]}: '
            f'{TEXTS[report.language]["band_" + row.band]}'
        )
        meta = words["aggregate_of"].format(rated=row.rated)
    elif row.status == "building":
        score, tag_class = "—", "building"
        tag = words["status_building"]
        meta = words[
            "building_text_one" if row.missing == 1 else "building_text"
        ].format(
            rated=row.rated, count=len(report.sessions),
            missing=row.missing,
        )
    else:
        score, tag_class = "—", "none"
        tag = words["status_none"]
        meta = words["none_text"]
    cells = "".join(_cell(cell, report) for cell in row.cells)
    lines = "".join(f"<li>{_text(line)}</li>" for line in _lines(row, report))
    return (
        f'<div class="row"><div class="name">{escape(row.name)}</div>'
        f'<div class="score">{score}</div>'
        f'<ol class="strip">{cells}</ol>'
        f'<div class="meta"><span class="tag {tag_class}">{escape(tag)}'
        f'</span><span>{escape(meta)}</span></div>'
        + (f'<ul class="lines">{lines}</ul>' if lines else "")
        + "</div>"
    )


def _meaning(report: ProfileReport) -> str:
    """Return the tiles, what the overall level is and the limits."""
    words = PROFILE_TEXTS[report.language]
    base = TEXTS[report.language]
    tiles = [
        (len(report.sessions), words["tile_situations"],
         words["tile_of"].format(count=RECOMMENDED_SITUATIONS)),
        (len(report.kinds), words["tile_kinds"], words["tile_kinds_of"]),
        (len(report.aggregated), words["tile_aggregated"],
         words["tile_aggregated_of"].format(count=len(report.rows))),
    ]
    tile_html = "".join(
        f'<div class="tile"><div>{escape(title)}</div>'
        f'<div class="big">{value}</div><div class="of">{escape(of)}</div>'
        f'</div>'
        for value, title, of in tiles
    )
    aggregate = words["aggregate_text"].format(
        needed=report.min_incidents, n1=note(1),
        situations=report.min_situations,
    )
    level = base["level_text"].format(
        minimum=report.minimum, maximum=report.maximum, n4=note(4),
        needed=report.min_incidents,
    )
    limits = words["limits_text"].format(
        seen=len(report.sessions), recommended=RECOMMENDED_SITUATIONS,
        target=report.target,
    )
    return (
        f'<div class="tiles">{tile_html}</div>'
        f'<h3>{escape(base["level"])}</h3><p>{_text(level)}</p>'
        f'<h3>{escape(words["aggregate_title"])}</h3>'
        f'<p>{_text(aggregate)}</p>'
        f'<div class="limits"><strong>{escape(base["limits_title"])}'
        f'</strong><p>{escape(limits)}</p>'
        f'<p>{escape(words["limits_change"])}</p>'
        f'<p>{escape(base["limits_more"])}</p></div>'
    )


def _charts(report: ProfileReport) -> str:
    """Return the legend and one strip per competency."""
    words = PROFILE_TEXTS[report.language]
    base = TEXTS[report.language]
    low = math.floor(report.midpoint) + 1
    high = math.ceil(report.midpoint) - 1
    legend = [
        ("strong", base["legend_strong"].format(low=low)),
        ("weak", base["legend_weak"].format(high=high)),
    ]
    if report.midpoint == int(report.midpoint):
        legend.insert(1, ("middle", base["legend_middle"].format(
            mid=int(report.midpoint)
        )))
    items = "".join(
        f"<li>{_legend_icon(kind)}<span>{escape(text)}</span></li>"
        for kind, text in legend
    )
    building = words["legend_building"].format(
        needed=report.min_situations
    )
    items += (
        f'<li><span class="mini" aria-hidden="true">–</span><span>'
        f'{escape(words["legend_empty"])}</span></li>'
        f'<li><span class="mini building" aria-hidden="true">…</span>'
        f'<span>{escape(building)}</span></li>'
    )
    rows = "".join(
        _row(row, report) for row in report.rows if row.status != "none"
    )
    empty = [row.name for row in report.rows if row.status == "none"]
    unseen = (
        f'<div class="unseen"><h3>{escape(words["none_list_title"])}</h3>'
        f'<p class="small">{escape(words["none_list_text"])}</p><ul>'
        + "".join(f"<li>{escape(name)}</li>" for name in empty)
        + "</ul></div>"
    ) if empty else ""
    return (
        f'<h3>{escape(base["legend"])}</h3><ul class="legend">{items}</ul>'
        f'<p class="small">{escape(words["strip_text"])}</p>{rows}{unseen}'
    )


def _point(item: RepeatedPoint, report: ProfileReport, weak: bool) -> str:
    """Return a repeated strength or point to improve with its quotes."""
    words = PROFILE_TEXTS[report.language]
    base = TEXTS[report.language]
    count = words["repeated_count_weak" if weak else "repeated_count"].format(
        count=item.count, rated=item.rated
    )
    quotes = "".join(
        f'<blockquote class="{"neg" if weak else "pos"}">'
        f'<p>«{escape(q.quote.text)}»</p><footer>'
        f'{escape(quote_when(q, report.language))} · '
        f'{escape(q.quote.indicator)}</footer></blockquote>'
        for q in item.quotes
    )
    practice = "".join(f"<li>{escape(line)}</li>" for line in item.practice)
    effect = base["hurts" if weak else "helps"]
    return (
        f'<div class="card"><h3>{escape(item.name)}</h3>'
        f'<p class="lvl">{escape(count)}</p>'
        f'<p class="small">{escape(item.definition)}</p>'
        + (f'<p><strong>{escape(base["seen"])}</strong></p>{quotes}'
           if quotes else "")
        + (f'<p><strong>{escape(effect)}</strong></p><ul>{practice}</ul>'
           if practice else "")
        + "</div>"
    )


def _sessions(report: ProfileReport) -> str:
    """Return the list of conversations and those left out."""
    words = PROFILE_TEXTS[report.language]
    items = []
    for session in report.sessions:
        parts = [session.date or words["no_date"]]
        if session.kind:
            parts.append(session.kind)
        parts.append(f"«{session.session_id}»" if report.language == "es"
                     else f"“{session.session_id}”")
        parts.append(words["session_turns"].format(turns=session.turns))
        items.append(f"<li>{escape(' · '.join(parts))}</li>")
    excluded = "".join(f"<li>{escape(line)}</li>" for line in report.excluded)
    return (
        f'<ol class="sessions">{"".join(items)}</ol>'
        + (f'<h3>{escape(words["excluded_title"])}</h3><ul>{excluded}</ul>'
           if excluded else "")
    )


def _notes(report: ProfileReport) -> str:
    """Return the footnotes that explain the terms used."""
    words = PROFILE_TEXTS[report.language]
    base = TEXTS[report.language]
    labels = "; ".join(
        f"{point} = {label}" for point, label in report.scale_labels.items()
    )
    level = (
        base["note_level"].format(labels=labels) if labels
        else base["note_level_plain"].format(
            minimum=report.minimum, maximum=report.maximum
        )
    )
    notes = [
        words["note_aggregate"], base["note_evaluator"],
        base["note_situation"], level, base["note_evidence"],
        words["note_reliability"],
    ]
    items = "".join(
        f'<li id="nota-{i}">{escape(text)}</li>'
        for i, text in enumerate(notes, start=1)
    )
    return f'<ol class="notes">{items}</ol>'


def render_profile_html(report: ProfileReport) -> str:
    """Return the profile as one HTML page with no external resource."""
    words = PROFILE_TEXTS[report.language]
    base = TEXTS[report.language]
    title = words["title"].format(participant=report.participant)
    dates = (
        "" if not report.first
        else words["subtitle_date"].format(first=report.first)
        if report.first == report.last
        else words["subtitle_dates"].format(
            first=report.first, last=report.last
        )
    )
    count = len(report.sessions)
    subtitle = (
        words["subtitle_one"].format(dates=dates) if count == 1
        else words["subtitle"].format(count=count, dates=dates)
    )
    banner = (
        f'<p class="banner" role="note">{escape(base["synthetic"])}</p>'
        if report.synthetic else ""
    )
    summary = "".join(f"<p>{_text(s)}</p>" for s in report.summary)
    strengths = "".join(
        _point(p, report, False) for p in report.strengths
    ) or f'<p>{escape(words["none_repeated_strengths"])}</p>'
    weaknesses = "".join(
        _point(p, report, True) for p in report.weaknesses
    ) or f'<p>{escape(words["none_repeated_weaknesses"])}</p>'
    made = words["how_made"].format(
        catalogue=report.catalogue, version=report.catalogue_version,
        judges=", ".join(report.judges), person=report.person_id,
    )
    return (
        f'<!doctype html>\n<html lang="{report.language}"><head>'
        f'<meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width, initial-scale=1">'
        f'<title>{escape(title)}</title>'
        f'<style>{STYLE}{PROFILE_STYLE}</style></head>'
        f'<body><main><h1>{escape(title)}</h1>'
        f'<p class="sub">{escape(subtitle)}</p>{banner}'
        f'<section><h2>{escape(base["summary"])}</h2>'
        f'<div class="lead">{summary}</div></section>'
        f'<section><h2>{escape(base["meaning"])}</h2>{_meaning(report)}'
        f'</section>'
        f'<section><h2>{escape(words["strip_title"])}</h2>'
        f'{_charts(report)}</section>'
        f'<section><h2>{escape(words["repeated_strengths"])}</h2>'
        f'{strengths}</section>'
        f'<section><h2>{escape(words["repeated_weaknesses"])}</h2>'
        f'{weaknesses}</section>'
        f'<section><h2>{escape(words["sessions"])}</h2>{_sessions(report)}'
        f'</section>'
        f'<section><h2>{escape(base["notes"])}</h2>{_notes(report)}'
        f'<p class="small">{escape(made)}</p></section>'
        f'</main></body></html>\n'
    )
