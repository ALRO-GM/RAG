"""Render a :class:`PersonReport` as one self-contained HTML page.

Inline CSS and SVG only: the page makes no request, prints on paper,
reads on a phone and follows the light or dark setting of the reader.
Colours come from the Okabe-Ito palette, which stays distinct with the
common colour-vision deficiencies, and every colour is doubled by a
shape and a written label.
"""

import math
from html import escape
from typing import Dict, List, Optional

from src.application.skills.person_report import (
    NOTE_MARK,
    RADAR_MINIMUM,
    RECOMMENDED_SITUATIONS,
    PersonReport,
    Point,
    Quote,
    Review,
    Row,
    note,
)
from src.application.skills.person_report_texts import (
    TEXTS,
    conversation_type_label,
)
from src.domain.skills.speech import clock
from src.domain.skills.synthesis import (
    Citable,
    PersonSynthesis,
    SynthesisPoint,
)

STYLE = """
:root{--bg:#ffffff;--fg:#1c1c1c;--muted:#555555;--card:#f4f4f2;
--line:#c8c8c8;--strong:#0072B2;--weak:#D55E00;--middle:#5a5a5a;
--none:#8a8a8a;--warn-bg:#fff4d6;--warn-fg:#5c4300;--fill:rgba(0,114,178,.18)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#161616;--fg:#eeeeee;--muted:#b4b4b4;--card:#242424;--line:#4a4a4a;
--strong:#56B4E9;--weak:#F08A4B;--middle:#c4c4c4;--none:#8f8f8f;
--warn-bg:#3d3214;--warn-fg:#ffe7a3;--fill:rgba(86,180,233,.22)}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:18px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:1.9rem;line-height:1.2;margin:0 0 4px}
h2{font-size:1.4rem;margin:40px 0 12px;padding-top:12px;
border-top:2px solid var(--line)}
h3{font-size:1.15rem;margin:0 0 6px}
p{margin:0 0 12px}
.sub{color:var(--muted);margin-bottom:16px}
.banner{background:var(--warn-bg);color:var(--warn-fg);
border:2px solid currentColor;border-radius:8px;padding:12px 14px;
font-weight:600;margin:12px 0 20px}
.lead p{font-size:1.1rem;margin-bottom:8px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
gap:12px;margin:16px 0}
.tile{background:var(--card);border-radius:10px;padding:12px 14px}
.tile .big{font-size:2.4rem;font-weight:700;line-height:1.1}
.tile .of{color:var(--muted)}
.limits{border:2px solid var(--weak);border-radius:10px;padding:12px 14px;
margin:16px 0}
.limits strong{display:block;margin-bottom:4px}
.legend{list-style:none;padding:0;margin:0 0 16px;display:grid;gap:6px}
.legend li{display:flex;gap:10px;align-items:center}
.legend svg{flex:none}
.row{display:grid;grid-template-columns:1fr auto;gap:2px 12px;
padding:12px 0;border-bottom:1px solid var(--line)}
.row .name{font-weight:600}
.row .score{font-size:1.8rem;font-weight:700;text-align:right;
line-height:1.1}
.row .score small{font-size:1rem;color:var(--muted);font-weight:400}
.row .track{grid-column:1 / -1;width:100%;max-width:460px;height:auto}
.row .meta{grid-column:1 / -1;display:flex;flex-wrap:wrap;gap:4px 14px;
color:var(--muted);font-size:.95rem}
.tag{font-weight:700}
.tag.strong{color:var(--strong)}.tag.weak{color:var(--weak)}
.tag.middle{color:var(--middle)}.tag.none{color:var(--none)}
.tag.provisional{color:var(--fg);font-style:italic}
.card.review{background:var(--bg);border:2px dashed var(--line)}
.why{border-left:4px solid var(--line);padding:4px 12px;margin:12px 0}
.why ul{margin:4px 0 0;padding-left:1.2em}
.dots{letter-spacing:2px}
.radar{display:grid;grid-template-columns:minmax(0,340px) 1fr;gap:16px;
align-items:center}
.radar ol{margin:0;padding-left:1.4em}
.card{background:var(--card);border-radius:10px;padding:14px 16px;
margin:12px 0;break-inside:avoid}
.card .lvl{font-weight:700}
blockquote{margin:8px 0;padding:6px 12px;border-left:4px solid var(--line);
background:var(--bg);border-radius:4px}
blockquote.pos{border-left-color:var(--strong)}
blockquote.neg{border-left-color:var(--weak)}
blockquote p{margin:0}
blockquote footer{color:var(--muted);font-size:.9rem}
.card ul{margin:4px 0 0;padding-left:1.2em}
.small{color:var(--muted);font-size:.95rem}
sup a{color:inherit;text-decoration:none;font-weight:700}
.notes{font-size:.95rem}
.synthesis{border:2px solid var(--strong);border-radius:10px;
padding:4px 16px 12px;margin:16px 0}
.synthesis h2{border-top:0;margin-top:12px}
.basis{font-size:.9rem;color:var(--muted);font-style:italic}
svg text{fill:var(--muted);font:12px system-ui,sans-serif}
@media (max-width:560px){.radar{grid-template-columns:1fr}
body{font-size:17px}h1{font-size:1.6rem}}
@media print{:root{--bg:#fff;--fg:#000;--card:#f2f2f2}
body{font-size:12pt}main{max-width:none;padding:0}
h2{break-after:avoid}.row,.tile,.limits{break-inside:avoid}
*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
"""

SHAPES = {
    "strong": "circle", "middle": "square", "weak": "triangle",
    "provisional": "ring",
}


def _text(value: str) -> str:
    """Escape ``value`` and turn its footnote marks into links."""
    return NOTE_MARK.sub(
        lambda m: f'<sup><a href="#nota-{m.group(1)}">{m.group(1)}</a></sup>',
        escape(value),
    )


def _mark(kind: str, x: float, y: float, size: float = 7) -> str:
    """Return the SVG shape of a band: circle, square, triangle or ring.

    The ring (hollow) marks a provisional estimate.
    """
    colour = f"var(--{kind})"
    if SHAPES.get(kind) == "ring":
        return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size:.1f}" '
                f'fill="none" stroke="var(--fg)" stroke-width="2.5"/>')
    if SHAPES.get(kind) == "circle":
        return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size:.1f}" '
                f'fill="{colour}"/>')
    if SHAPES.get(kind) == "square":
        side = size * 1.7
        return (f'<rect x="{x - side / 2:.1f}" y="{y - side / 2:.1f}" '
                f'width="{side:.1f}" height="{side:.1f}" fill="{colour}"/>')
    top, base = y - size * 1.1, y + size * 0.9
    return (f'<polygon points="{x:.1f},{top:.1f} {x - size:.1f},{base:.1f} '
            f'{x + size:.1f},{base:.1f}" fill="{colour}"/>')


def _legend_icon(kind: str) -> str:
    """Return the small SVG used in the legend for a band."""
    if kind == "none":
        body = ('<line x1="2" y1="11" x2="20" y2="11" stroke="var(--none)" '
                'stroke-width="3" stroke-dasharray="4 3"/>')
    else:
        body = _mark(kind, 11, 11)
    return (f'<svg width="22" height="22" viewBox="0 0 22 22" '
            f'aria-hidden="true">{body}</svg>')


def _track(row: Row, report: PersonReport) -> str:
    """Return the scale of one competency with its level marked."""
    width, left, right = 300, 14, 286
    points = list(range(report.minimum, report.maximum + 1))
    span = report.maximum - report.minimum

    def at(value: int) -> float:
        """Return the x position of a scale point."""
        return left + (value - report.minimum) / span * (right - left)

    dashed = ' stroke-dasharray="6 5"' if row.rating is None else ""
    parts = [
        f'<line x1="{left}" y1="16" x2="{right}" y2="16" '
        f'stroke="var(--line)" stroke-width="4"{dashed}/>'
    ]
    for value in points:
        parts.append(
            f'<line x1="{at(value):.1f}" y1="10" x2="{at(value):.1f}" '
            f'y2="22" stroke="var(--line)" stroke-width="2"/>'
            f'<text x="{at(value):.1f}" y="38" text-anchor="middle">'
            f'{value}</text>'
        )
    if row.rating is not None:
        parts.append(_mark(row.band, at(row.rating), 16, 9))
    elif row.estimate is not None:
        parts.append(_mark("provisional", at(row.estimate), 16, 8))
    label = escape(_band_label(row, report))
    return (f'<svg class="track" viewBox="0 0 {width} 42" '
            f'role="img" aria-label="{label}">'
            f'{"".join(parts)}</svg>')


def _band_label(row: Row, report: PersonReport) -> str:
    """Return the written level of a row, e.g. ``Punto fuerte``."""
    words = TEXTS[report.language]
    name = words[f"band_{row.band}"]
    if row.estimate is not None:
        label = _estimate_label(row.estimate, row.confidence, report)
        return f"{row.name}: {label}"
    if row.rating is None:
        return f"{row.name}: {name}"
    level = words["level_label"].format(
        rating=row.rating, maximum=report.maximum
    )
    return f"{row.name}: {level}, {name}"


def _estimate_label(
    rating: int, confidence: str, report: PersonReport
) -> str:
    """Return e.g. ``Estimación provisional — revisar: nivel 3 de 5…``."""
    words = TEXTS[report.language]
    return words["estimate_label"].format(
        rating=rating, maximum=report.maximum,
        confidence=words.get(f"confidence_{confidence}", confidence),
    )


def _dots(count: int, needed: int) -> str:
    """Return filled dots for the evidence, hollow ones up to the minimum."""
    shown = min(count, 10)
    return "●" * shown + "○" * max(needed - shown, 0)


def _row(row: Row, report: PersonReport) -> str:
    """Return one competency of the chart."""
    words = TEXTS[report.language]
    needed = report.min_incidents
    if row.rating is not None:
        score = f"{row.rating} <small>/ {report.maximum}</small>"
    elif row.estimate is not None:
        score = (
            escape(words["estimate_score"].format(rating=row.estimate))
            + f" <small>/ {report.maximum}</small>"
        )
    else:
        score = "—"
    tag = (
        _estimate_label(row.estimate, row.confidence, report)
        if row.estimate is not None else words["band_" + row.band]
    )
    evidence = words[
        "evidence_enough" if row.incidents >= needed else "evidence"
    ].format(count=row.incidents, needed=needed)
    return (
        f'<div class="row"><div class="name">{escape(row.name)}</div>'
        f'<div class="score">{score}</div>{_track(row, report)}'
        f'<div class="meta"><span class="tag {row.band}">'
        f'{escape(tag)}</span>'
        f'<span><span class="dots" aria-hidden="true">'
        f'{_dots(row.incidents, needed)}</span> {escape(evidence)}</span>'
        f'</div></div>'
    )


def _radar(report: PersonReport) -> str:
    """Return the radar of the rated competencies, numbered axes."""
    words = TEXTS[report.language]
    rows = report.rated
    size, centre, radius = 340, 170.0, 120.0
    span = report.maximum - report.minimum

    def point(index: int, value: float) -> str:
        """Return the x,y of a value on an axis."""
        angle = 2 * math.pi * index / len(rows) - math.pi / 2
        distance = radius * (value - report.minimum + 1) / (span + 1)
        return (f"{centre + distance * math.cos(angle):.1f},"
                f"{centre + distance * math.sin(angle):.1f}")

    parts: List[str] = []
    for level in range(report.minimum, report.maximum + 1):
        ring = " ".join(point(i, level) for i in range(len(rows)))
        parts.append(f'<polygon points="{ring}" fill="none" '
                     f'stroke="var(--line)" stroke-width="1"/>')
    for index in range(len(rows)):
        outer = point(index, report.maximum).split(",")
        parts.append(f'<line x1="{centre}" y1="{centre}" x2="{outer[0]}" '
                     f'y2="{outer[1]}" stroke="var(--line)"/>')
        label = point(index, report.maximum + 0.75).split(",")
        parts.append(
            f'<circle cx="{label[0]}" cy="{label[1]}" r="11" '
            f'fill="var(--card)" stroke="var(--line)"/>'
            f'<text x="{label[0]}" y="{float(label[1]) + 4:.1f}" '
            f'text-anchor="middle" style="fill:var(--fg);font-weight:700">'
            f'{index + 1}</text>'
        )
    shape = " ".join(
        point(i, row.rating or report.minimum) for i, row in enumerate(rows)
    )
    parts.append(f'<polygon points="{shape}" fill="var(--fill)" '
                 f'stroke="var(--strong)" stroke-width="3"/>')
    for i, row in enumerate(rows):
        x, y = point(i, row.rating or report.minimum).split(",")
        parts.append(_mark(row.band, float(x), float(y), 6))
    items = "".join(
        f"<li>{escape(row.name)}: <strong>{row.rating}</strong> / "
        f"{report.maximum} ({escape(words['band_' + row.band])})</li>"
        for row in rows
    )
    return (
        f'<h3>{escape(words["radar_title"])}</h3><div class="radar">'
        f'<svg viewBox="0 0 {size} {size}" width="100%" role="img" '
        f'aria-label="{escape(words["radar_title"])}">{"".join(parts)}'
        f'</svg><div><ol>{items}</ol><p class="small">'
        f'{escape(words["radar_note"])}</p></div></div>'
    )


def _quote(quote: Quote, language: str) -> str:
    """Return a verified quote with who said it and when."""
    kind = "pos" if quote.positive else "neg"
    source = (
        f" · {TEXTS[language]['source.' + quote.source]}"
        if quote.source else ""
    )
    sign = "" if not quote.source else "(+) " if quote.positive else "(−) "
    return (
        f'<blockquote class="{kind}"><p>«{escape(quote.text)}»</p>'
        f'<footer>{escape(quote.when(language))} · '
        f'{escape(sign + quote.indicator + source)}</footer></blockquote>'
    )


def _card(item: Point, report: PersonReport, effect: str) -> str:
    """Return a strength, point to improve or signal with its quotes."""
    words = TEXTS[report.language]
    level = "" if item.rating is None else (
        ' — <span class="lvl">' + escape(words["level_label"].format(
            rating=item.rating, maximum=report.maximum
        )) + "</span>"
    )
    quotes = "".join(_quote(q, report.language) for q in item.quotes)
    other = "".join(_quote(q, report.language) for q in item.other)
    practice = "".join(f"<li>{escape(line)}</li>" for line in item.practice)
    return (
        f'<div class="card"><h3>{escape(item.name)}{level}</h3>'
        f'<p class="small">{escape(item.definition)}</p>'
        + (f'<p><strong>{escape(words["seen"])}</strong></p>{quotes}'
           if quotes else "")
        + (f'<p><strong>{escape(words["also_for"])}</strong></p>{other}'
           if other else "")
        + (f'<p><strong>{escape(words[effect])}</strong></p>'
           f'<ul>{practice}</ul>' if practice else "")
        + "</div>"
    )


def _notes(report: PersonReport) -> str:
    """Return the footnotes that explain the terms used."""
    words = TEXTS[report.language]
    labels = "; ".join(
        f"{point} = {label}" for point, label in report.scale_labels.items()
    )
    level = (
        words["note_level"].format(labels=labels) if labels
        else words["note_level_plain"].format(
            minimum=report.minimum, maximum=report.maximum
        )
    )
    notes = [
        words["note_competency"], words["note_evaluator"],
        words["note_situation"], level, words["note_evidence"],
    ]
    items = "".join(
        f'<li id="nota-{i}">{escape(text)}</li>'
        for i, text in enumerate(notes, start=1)
    )
    return f'<ol class="notes">{items}</ol>'


def _meaning(report: PersonReport) -> str:
    """Return what was observed, what a level means and the limits."""
    words = TEXTS[report.language]
    minutes = report.minutes
    if minutes is None:
        duration_text, big, unit = "", "—", words["tile_unknown"]
    elif minutes < 1:
        duration_text = words["observed_duration_short"]
        big, unit = "&lt;1", words["tile_minutes"]
    else:
        duration_text = words["observed_duration"].format(
            minutes=round(minutes)
        )
        big, unit = str(round(minutes)), words["tile_minutes"]
    observed = words["observed_text"].format(
        n3=note(3), session=report.session_id,
        duration_text=duration_text, participant=report.participant,
        turns=report.turns, words=report.words,
    )
    level = words["level_text"].format(
        minimum=report.minimum, maximum=report.maximum, n4=note(4),
        needed=report.min_incidents,
    )
    limits = words["limits_text"].format(
        seen=report.situations, recommended=RECOMMENDED_SITUATIONS,
        plural="" if report.situations == 1 else (
            "es" if report.language == "es" else "s"
        ),
    )
    tiles = [
        (str(report.situations), words["tile_situations"],
         words["tile_of"].format(count=RECOMMENDED_SITUATIONS)),
        (str(len(report.rated)), words["tile_rated"],
         words["tile_rated_of"].format(count=len(report.rows))),
        (str(len(report.provisional)), words["tile_provisional"],
         words["tile_observations"].format(count=report.observations)),
        (big, words["tile_duration"], unit),
    ]
    tile_html = "".join(
        f'<div class="tile"><div>{escape(title)}</div>'
        f'<div class="big">{value}</div><div class="of">{escape(of)}</div>'
        f'</div>'
        for value, title, of in tiles
    )
    return (
        f'<div class="tiles">{tile_html}</div>'
        f'<h3>{escape(words["observed"])}</h3><p>{_text(observed)}</p>'
        f'<h3>{escape(words["level"])}</h3><p>{_text(level)}</p>'
        f'<div class="limits"><strong>{escape(words["limits_title"])}'
        f'</strong><p>{escape(limits)}</p>'
        f'<p>{escape(words["limits_more"])}</p>'
        f'<p>{escape(words["limits_provisional"])}</p></div>'
    )


def _why(report: PersonReport) -> str:
    """Return why there are few firm levels, or nothing."""
    if not report.why:
        return ""
    words = TEXTS[report.language]
    values = {
        "participant": report.participant, "words": report.words,
        "needed": report.min_incidents,
    }
    items = "".join(
        f"<li>{escape(words['why.' + code].format(**values))}</li>"
        for code in report.why
    )
    return (
        f'<div class="why"><strong>{escape(words["why_title"])}</strong>'
        f'<ul>{items}</ul></div>'
    )


def _review(item: Review, report: PersonReport) -> str:
    """Return a provisional estimate or a set of observations."""
    words = TEXTS[report.language]
    heading = (
        _estimate_label(item.estimate, item.confidence, report)
        if item.estimate is not None else words["observations_only"]
    )
    reasons = "".join(
        f"<li>{escape(words['reason.' + code].format(**item.details))}</li>"
        for code in item.reasons
    )
    quotes = "".join(
        _quote(q, report.language) for q in item.observations
    )
    return (
        f'<div class="card review"><h3>{escape(item.name)} — '
        f'<span class="tag provisional">{escape(heading)}</span></h3>'
        f'<p class="small">{escape(item.definition)}</p>'
        + (f'<p><strong>{escape(words["why_not_firm"])}</strong></p>'
           f'<ul>{reasons}</ul>' if reasons else "")
        + (f'<p><strong>{escape(words["observed_items"])}</strong></p>'
           f'{quotes}' if quotes else "")
        + "</div>"
    )


def _reviews(report: PersonReport) -> str:
    """Return the section of provisional estimates and observations."""
    words = TEXTS[report.language]
    if not report.reviews:
        return f'<p>{escape(words["review_none"])}</p>'
    return (
        f'<p class="small">{escape(words["review_text"])}</p>'
        + "".join(_review(item, report) for item in report.reviews)
    )


def _charts(report: PersonReport) -> str:
    """Return the legend, one row per competency and the radar."""
    words = TEXTS[report.language]
    low = math.floor(report.midpoint) + 1
    high = math.ceil(report.midpoint) - 1
    legend: Dict[str, str] = {
        "strong": words["legend_strong"].format(low=low),
        "weak": words["legend_weak"].format(high=high),
        "provisional": words["legend_provisional"],
        "none": words["legend_none"],
    }
    if report.midpoint == int(report.midpoint):
        legend["middle"] = words["legend_middle"].format(
            mid=int(report.midpoint)
        )
    order = ["strong", "middle", "weak", "provisional", "none"]
    items = "".join(
        f"<li>{_legend_icon(kind)}<span>{escape(legend[kind])}</span></li>"
        for kind in order if kind in legend
    )
    evidence = words["legend_evidence"].format(n5=note(5))
    items += ('<li><span class="dots" aria-hidden="true">●●○</span>'
              f'<span>{_text(evidence)}</span></li>')
    rows = "".join(_row(row, report) for row in report.rows)
    missing = words["no_radar"].format(
        needed=RADAR_MINIMUM, rated=len(report.rated)
    )
    radar = (
        _radar(report) if report.radar
        else f'<p class="small">{escape(missing)}</p>'
    )
    return (f'<h3>{escape(words["legend"])}</h3><ul class="legend">{items}'
            f'</ul>{rows}{radar}')


def _context(report: PersonReport) -> str:
    """Return the conversation's declared type and goal and the
    participant's conversation-event counts, or nothing."""
    words = TEXTS[report.language]
    parts = []
    if report.conversation_type:
        parts.append(words["context_type"].format(
            type=conversation_type_label(
                report.language, report.conversation_type
            )
        ))
    if report.conversation_goal:
        parts.append(words["context_goal"].format(
            goal=report.conversation_goal
        ))
    events = (
        f'<p class="sub context">'
        f'{escape(" ".join(report.conversation_lines))}</p>'
        if report.conversation_lines else ""
    )
    if not parts:
        return events
    return (
        f'<p class="sub context">{escape(" ".join(parts))} '
        f'<span class="small">{escape(words["context_note"])}</span></p>'
        + events
    )


def _cited(item: Citable, names: Dict[str, str], language: str) -> str:
    """Return one cited quote or event with its minute and source."""
    words = TEXTS[language]
    when = "" if item.start is None else clock(item.start)
    when = (when[3:] if when.startswith("00:") else when) or "–"
    if item.kind == "event":
        return f'<li class="small">[{escape(when)}] {escape(item.text)}</li>'
    kind = "neg" if item.polarity == "negative" else "pos"
    source = names.get(item.competency_id, item.competency_id)
    basis = words["syn_basis_" + item.kind]
    return (
        f'<blockquote class="{kind}"><p>«{escape(item.text)}»</p>'
        f'<footer>{escape(when)} · {escape(source)} · {escape(basis)}'
        f'</footer></blockquote>'
    )


def _point(
    point: SynthesisPoint, names: Dict[str, str], language: str
) -> str:
    """Return one point of the synthesis with what it cites."""
    words = TEXTS[language]
    quotes = "".join(
        _cited(c, names, language) for c in point.cites if c.kind != "event"
    )
    events = "".join(
        _cited(c, names, language) for c in point.cites if c.kind == "event"
    )
    basis = words["syn_basis_" + point.basis]
    return (
        f'<div class="card"><p>{escape(point.text)} '
        f'<span class="basis">({escape(basis)})</span></p>{quotes}'
        + (f'<ul>{events}</ul>' if events else "") + "</div>"
    )


def _synthesis(
    report: PersonReport, profile: Optional[PersonSynthesis], job: str
) -> str:
    """Return the «Síntesis» section on top of the report, or nothing."""
    if profile is None:
        return ""
    language = report.language
    words = TEXTS[language]
    names = {row.competency_id: row.name for row in report.rows}
    parts = [
        f'<section class="synthesis"><h2>{escape(words["syn_title"])}</h2>'
        f'<p class="banner" role="note">{escape(words["syn_note"])}</p>'
    ]
    if job:
        parts.append(
            f'<p class="small">{escape(words["syn_job"].format(job=job))}</p>'
        )
    if profile.status == "skipped":
        reason = (
            words["syn_no_evidence"] if profile.reason == "no_evidence"
            else profile.reason
        )
        text = words["syn_skipped"].format(reason=reason)
        return "".join(parts) + f"<p>{escape(text)}</p></section>"
    none = f'<p>{escape(words["syn_none"])}</p>'
    strengths = "".join(
        _point(p, names, language) for p in profile.strengths
    ) or none
    fit = "".join(_point(p, names, language) for p in profile.fit) or none
    unobserved = (
        "<ul>" + "".join(f"<li>{escape(n)}</li>" for n in profile.unobserved)
        + "</ul>" if profile.unobserved
        else f'<p>{escape(words["syn_unobserved_none"])}</p>'
    )
    questions = "".join(f"<li>{escape(q)}</li>" for q in profile.questions)
    level = words["syn_conf_" + profile.confidence]
    reason = words["syn_conf_reason"].format(
        **profile.confidence_basis.model_dump()
    )
    dropped = sum(profile.dropped.values())
    fit_title = words["syn_fit" if job else "syn_tasks"]
    parts += [
        f'<h3>{escape(words["syn_strengths"])}</h3>{strengths}',
        f'<h3>{escape(words["syn_unobserved"])}</h3>{unobserved}',
        f'<h3>{escape(fit_title)}</h3>{fit}',
    ]
    if questions:
        parts.append(
            f'<h3>{escape(words["syn_questions"])}</h3><ul>{questions}</ul>'
        )
    parts.append(
        f'<h3>{escape(words["syn_confidence"].format(level=level))}</h3>'
        f'<p>{escape(reason)}</p>'
    )
    if dropped:
        text = words["syn_dropped"].format(count=dropped)
        parts.append(f'<p class="small">{escape(text)}</p>')
    parts.append(
        f'<p class="small">{escape(words["limits_more"])} '
        f'{escape(words["limits_provisional"])}</p>'
    )
    if profile.model:
        text = words["syn_model"].format(model=profile.model)
        parts.append(f'<p class="small">{escape(text)}</p>')
    return "".join(parts) + "</section>"


def render_person_html(
    report: PersonReport,
    synthesis: Optional[PersonSynthesis] = None,
    job: str = "",
) -> str:
    """Return the report as one HTML page with no external resource.

    ``synthesis``, when given, is the person's explanatory profile of
    step i (#113), shown on top with the job sought (``job``).
    """
    words = TEXTS[report.language]
    title = words["title"].format(participant=report.participant)
    subtitle = words["subtitle"].format(session=report.session_id)
    banner = (
        f'<p class="banner" role="note">{escape(words["synthetic"])}</p>'
        if report.synthetic else ""
    )
    summary = "".join(f"<p>{_text(s)}</p>" for s in report.summary)
    tiers = "".join(
        f'<p class="tiers"><strong>{escape(s)}</strong></p>'
        if i == 0 else f'<p>{escape(s)}</p>'
        for i, s in enumerate(report.tiers)
    )
    strengths = "".join(
        _card(p, report, "helps") for p in report.strengths
    ) or f'<p>{escape(words["none_strengths"])}</p>'
    weaknesses = "".join(
        _card(p, report, "hurts") for p in report.weaknesses
    ) or f'<p>{escape(words["none_weaknesses"])}</p>'
    signals = (
        f'<h3>{escape(words["signals"])}</h3>'
        f'<p class="small">{escape(words["signals_text"])}</p>'
        + "".join(_card(p, report, "hurts") for p in report.signals)
        if report.signals else ""
    )
    made = words["how_made"].format(
        catalogue=report.catalogue, version=report.catalogue_version,
        judge=report.judge_model,
    )
    return (
        f'<!doctype html>\n<html lang="{report.language}"><head>'
        f'<meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width, initial-scale=1">'
        f'<title>{escape(title)}</title><style>{STYLE}</style></head>'
        f'<body><main><h1>{escape(title)}</h1>'
        f'<p class="sub">{escape(subtitle)}</p>{_context(report)}{banner}'
        f'{_synthesis(report, synthesis, job)}'
        f'<section><h2>{escape(words["summary"])}</h2>'
        f'<div class="lead">{summary}</div>{tiers}{_why(report)}</section>'
        f'<section><h2>{escape(words["meaning"])}</h2>{_meaning(report)}'
        f'</section>'
        f'<section><h2>{escape(words["charts"])}</h2>{_charts(report)}'
        f'</section>'
        f'<section><h2>{escape(words["strengths"])}</h2>{strengths}'
        f'</section>'
        f'<section><h2>{escape(words["weaknesses"])}</h2>{weaknesses}'
        f'{signals}</section>'
        f'<section><h2>{escape(words["review"])}</h2>{_reviews(report)}'
        f'</section>'
        f'<section><h2>{escape(words["notes"])}</h2>{_notes(report)}'
        f'<p class="small">{escape(made)}</p></section>'
        f'</main></body></html>\n'
    )
