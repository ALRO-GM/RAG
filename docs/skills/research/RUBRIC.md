# Scoring rubric (applied uniformly to every model / instrument / method)

Three separate scores, 0–100 each, never merged into one number, plus a
confidence label. Every sub-score needs a one-line justification with
source ids from `sources/*.jsonl`. "No evidence found" scores 0 on that
line and is written as *not found*; evidence AGAINST is written as
*negative* — keep the two apart. Use only figures present in the source
records; a figure marked secondary or abstract-only may be used but
lowers the confidence label.

## A. Credibility of the evidence (0–100)
| Line | Max | Anchors |
|---|---|---|
| A1 Reliability (name the type: internal consistency, test-retest, inter-rater, across situations) | 20 | 0 none · 8 one type, single study or vendor only · 14 several types or independent · 20 meta-analytic / generalizability study, adequate values |
| A2 Construct validity (structural, convergent/discriminant, nomological) | 15 | 0 none · 6 structural only or contested · 10 convergent+discriminant shown · 15 replicated independently |
| A3 Criterion validity against work outcomes (job/team performance, not student grades or impressions) | 25 | 0 none · 8 single/small or non-work criterion · 15 several studies · 20 meta-analysis · 25 meta-analysis, re-estimated/robust, r≥.25 |
| A4 Independence and replication (not only the owner/vendor) | 20 | 0 vendor only · 10 mixed · 20 mostly independent, incl. critical tests survived |
| A5 Scale and recency (samples, work populations, evidence from the last ~15 years) | 10 | |
| A6 Known problems (subtract up to 10 from the total: failed replications, faking, bias/adverse impact, exercise effects…) | −10 | list them |

## B. Acceptance in organisations (0–100)
| Line | Max | Anchors |
|---|---|---|
| B1 Measured adoption (independent surveys > vendor surveys > vendor claims) | 35 | |
| B2 Position of professional bodies and test-review commissions (SIOP, BPS, COP, EFPA, ITC) | 25 | |
| B3 Legal/defensibility and candidate acceptance | 20 | |
| B4 Availability to a company in Spain (Spanish version, publisher, cost) | 20 | |

## C. Fit to this product (0–100): scoring individuals from a recorded, transcribed conversation
| Line | Max | Anchors |
|---|---|---|
| C1 Native validated assessment mode | 35 | 35 trained-observer rating/coding of interaction · 22 other-report (peers/followers) · 12 interview/simulation scored by assessors on other material · 5 self-report or test |
| C2 Evidence that it survives in transcripts/audio or automatic scoring | 25 | 0 none · 10 indirect · 18 one study with human-criterion agreement · 25 replicated |
| C3 Unit of analysis is the individual | 15 | 15 individual · 7 dyad · 3 team only |
| C4 Content can be reused (licence of indicators/codes) | 15 | 15 open licence · 8 cite/paraphrase only · 0 gated |
| C5 Covers the prototype scope (personality, negotiation, leadership) or core teamwork | 10 | |

## Confidence label
- **High**: the decisive sources were read in full text here.
- **Medium**: mixture of full text and abstracts.
- **Low**: mostly abstract-only / secondary; scores may move when the papers are obtained.

## Terms (use them strictly)
Reliability = consistency of measurement. Validity types are named
individually. "Meta-analysis" only for a quantitative synthesis.
