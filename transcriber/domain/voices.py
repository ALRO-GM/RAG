"""Voices: clustering voice embeddings and putting names on the groups.

Diarization embeds short windows of a recording and groups them into
anonymous voices (:func:`cluster_speakers`, :func:`voice_centroids`,
:func:`nearest_voice`, copied from ``src/domain/skills/speech.py``).
:func:`match_voices` then pairs each group with at most one enrolled
:class:`~transcriber.domain.models.Voiceprint`; the groups left over get
anonymous labels ``S1``, ``S2``, ...
When voiceprints are available, :func:`seed_voices` is the other way
(measured better in TR-05): the voiceprints claim windows first and only
the unclaimed windows are clustered, into anonymous voices
(:func:`name_seeded` names the groups).

How embeddings are normalised before they are compared is a parameter
(:data:`Normalisation`): centring on the mean of a recording, which
helps inside one recording, is meaningless for a one-voice enrolment
sample. Ticket TR-05 compared raw unit vectors, centring on the
session, on a fixed mean of other readers and on the enrolled
voiceprints: none beat raw unit vectors on held-out readers, which stay
the default (``docs/transcriber/IDENTIFICATION.md``). The match and
single-voice thresholds were set there on synthetic read speech.
"""

import math
from typing import (
    AbstractSet,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment

from transcriber.domain.errors import InvalidInputError
from transcriber.domain.models import Voiceprint

FloatArray = npt.NDArray[np.float32]

# A way to normalise session voices and enrolled voiceprints before
# they are compared by cosine: it receives both matrices (one vector
# per row) and returns them, transformed, in the same order.
Normalisation = Callable[
    [FloatArray, FloatArray], Tuple[FloatArray, FloatArray]
]

# Cosine similarity (raw unit vectors) a group needs to take an
# enrolled id. Measured in TR-05 (docs/transcriber/IDENTIFICATION.md):
# raw cosines between different readers' voices reach 0.95 and more, so
# 0.5 named nearly every group; 0.95 was best on the tuning readers.
DEFAULT_MATCH_THRESHOLD = 0.95

# Fewest session groups whose mean :func:`centre_on_session` subtracts.
# Below it the mean says nothing about the room: one group minus its own
# mean is the zero vector, and two groups centred on their mean become
# exact opposites, so every voiceprint's cosine with one is minus its
# cosine with the other. A rule of the arithmetic, not a measurement.
MIN_GROUPS_TO_CENTRE = 3

ANONYMOUS_PREFIX = "S"


def cluster_speakers(
    vectors: FloatArray,
    num_speakers: Optional[int] = None,
    threshold: float = 0.5,
    max_speakers: int = 10,
) -> List[int]:
    """Group voice embeddings into speakers (agglomerative, cosine).

    Average-linkage clustering: clusters are merged by mean similarity
    until ``num_speakers`` remain or, when the number is unknown, until
    no two clusters are more similar than ``threshold`` (at most
    ``max_speakers`` remain).
    Labels are numbered by first appearance, so the first voice heard
    is speaker 0.
    """
    count = int(vectors.shape[0])
    if num_speakers is not None and num_speakers < 1:
        raise InvalidInputError("num_speakers must be at least 1.")
    if count == 0:
        return []
    if count == 1:
        return [0]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = (vectors / np.where(norms > 0, norms, 1.0)).astype(np.float64)
    # Cosine distance, computed here so that zero vectors (silence) are
    # simply far from everything instead of undefined.
    distance = np.clip(1.0 - unit @ unit.T, 0.0, 2.0)
    tree = linkage(
        distance[np.triu_indices(count, k=1)], method="average"
    )
    if num_speakers is not None:
        raw = fcluster(tree, t=num_speakers, criterion="maxclust")
    else:
        raw = fcluster(tree, t=1.0 - threshold, criterion="distance")
        if len(set(raw.tolist())) > max_speakers:
            raw = fcluster(tree, t=max_speakers, criterion="maxclust")
    first_seen: Dict[int, int] = {}
    labels: List[int] = []
    for value in raw.tolist():
        labels.append(first_seen.setdefault(int(value), len(first_seen)))
    return labels


def unit_rows(vectors: FloatArray) -> FloatArray:
    """Scale every row to unit length (zero rows stay zero)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    out: FloatArray = (
        vectors / np.where(norms > 0, norms, 1.0)
    ).astype(np.float32)
    return out


def centre(
    vectors: FloatArray, mean: Optional[FloatArray] = None
) -> FloatArray:
    """Subtract the mean voice of the recording and re-normalise.

    Voice embeddings of one recording share a large common component
    (the room, the microphone, the model's own bias). Removing it is
    what makes cosines between different people small: measured in
    ``src`` on synthetic four-speaker meetings (not re-measured here),
    attribution rose from 55% to 78% with 1.5 s windows and thresholds
    became stable across meetings.
    """
    if vectors.shape[0] == 0:
        return vectors
    centred = vectors - (vectors.mean(axis=0) if mean is None else mean)
    return unit_rows(centred)


def voice_centroids(
    vectors: FloatArray, labels: Sequence[int]
) -> FloatArray:
    """Return one unit-length mean vector per speaker label (0..n-1)."""
    count = max(labels) + 1 if labels else 0
    rows = []
    for label in range(count):
        members = vectors[[i for i, v in enumerate(labels) if v == label]]
        mean = members.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        rows.append(mean / norm if norm > 0 else mean)
    if not rows:
        return np.zeros((0, vectors.shape[1]), dtype=np.float32)
    return np.vstack(rows).astype(np.float32)


def nearest_voice(vectors: FloatArray, centroids: FloatArray) -> List[int]:
    """Label each embedding with its most similar speaker centroid."""
    if vectors.shape[0] == 0 or centroids.shape[0] == 0:
        return []
    return [int(i) for i in np.argmax(vectors @ centroids.T, axis=1)]


def is_single_voice(
    vectors: FloatArray, threshold: float, quantile: float = 0.0
) -> bool:
    """Tell whether every row of ``vectors`` is one and the same voice.

    The rows are taken as raw unit vectors (no centring): they are one
    voice when the ``quantile`` of their cosines with their mean
    direction reaches ``threshold``. With ``quantile`` 0 that is the
    lowest cosine, so every window must agree; with 0.1 about one window
    in ten may stray (a cough, a noisy window) without the recording
    counting as several voices. Measured in ``docs/transcriber/
    IDENTIFICATION.md``: one stray window was enough for the minimum to
    call a one-voice recording several. Centring cannot answer this
    question: with one voice, subtracting the mean leaves only the
    voice's own variation, which clustering then splits into several
    "people". No rows, or a mean of zero length, is not one voice.
    Raises :class:`InvalidInputError` for a ``quantile`` outside
    ``[0, 1)``.
    """
    if not 0.0 <= quantile < 1.0:
        raise InvalidInputError("The quantile must be in [0, 1).")
    if vectors.shape[0] == 0:
        return False
    unit = unit_rows(vectors).astype(np.float64)
    mean = unit.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        return False
    cosines = unit @ (mean / norm)
    return bool(float(np.quantile(cosines, quantile)) >= threshold)


class VoiceGroups(NamedTuple):
    """The voices of a recording found by :func:`group_voices`.

    ``coarse`` and ``fine`` hold the group of each long and each short
    window (``fine`` is empty without short windows) and ``vectors``
    one raw unit-length mean per group (row ``i`` is group ``i``), the
    vector compared with enrolled voiceprints. ``fine_scores`` holds,
    for each short window (row) and each group (column), the cosine the
    window's group was chosen by (the largest of its row); it is
    ``None`` when nothing was compared (one voice, or no windows).
    """

    coarse: List[int]
    fine: List[int]
    vectors: FloatArray
    fine_scores: Optional[FloatArray] = None


def group_voices(
    coarse: FloatArray,
    fine: FloatArray,
    num_speakers: Optional[int] = None,
    threshold: float = 0.5,
    max_speakers: int = 10,
    centre_windows: bool = True,
    single_voice_threshold: Optional[float] = None,
    single_voice_quantile: float = 0.0,
) -> VoiceGroups:
    """Find the voices of a recording in two passes.

    1. *Who is there*: the long-window embeddings ``coarse`` are
       clustered (:func:`cluster_speakers` with ``num_speakers``,
       ``threshold`` and ``max_speakers``). With ``num_speakers`` 1, or
       when the number is unknown and :func:`is_single_voice` finds
       one voice at ``single_voice_threshold`` and
       ``single_voice_quantile`` (a threshold of ``None`` skips the
       test), every window is one group and nothing is clustered.
    2. *Who speaks when*: each short-window embedding of ``fine`` goes
       to the nearest group centroid, which keeps the time resolution
       long windows lack.

    With ``centre_windows`` both passes work on vectors centred on the
    mean of the long windows (:func:`centre`), as the analyzer's
    diarizer does. The group vectors returned for matching are not
    centred: they are the means of the raw unit long-window vectors, so
    a normalisation can be chosen when they meet voiceprints.
    """
    if coarse.ndim != 2 or fine.ndim != 2:
        raise InvalidInputError("Window embeddings must be 2-D arrays.")
    if coarse.shape[0] == 0:
        width = int(coarse.shape[1])
        return VoiceGroups([], [], np.zeros((0, width), dtype=np.float32))
    if fine.shape[0] and fine.shape[1] != coarse.shape[1]:
        raise InvalidInputError("Long and short windows differ in size.")
    if num_speakers is not None and num_speakers < 1:
        raise InvalidInputError("num_speakers must be at least 1.")
    single = num_speakers == 1 or (
        num_speakers is None and single_voice_threshold is not None
        and is_single_voice(
            coarse, single_voice_threshold, single_voice_quantile
        )
    )
    if single:
        return VoiceGroups(
            [0] * int(coarse.shape[0]), [0] * int(fine.shape[0]),
            voice_centroids(unit_rows(coarse), [0] * int(coarse.shape[0])),
        )
    if centre_windows:
        mean: FloatArray = coarse.mean(axis=0).astype(np.float32)
        long, short = centre(coarse, mean), centre(fine, mean)
    else:
        long, short = unit_rows(coarse), unit_rows(fine)
    labels = cluster_speakers(long, num_speakers, threshold, max_speakers)
    centroids = voice_centroids(long, labels)
    assigned = nearest_voice(short, centroids)
    return VoiceGroups(
        labels, assigned, voice_centroids(unit_rows(coarse), labels),
        _scores(short, centroids),
    )


def unit_vectors(
    session: FloatArray, enrolled: FloatArray
) -> Tuple[FloatArray, FloatArray]:
    """Normalisation that compares raw unit vectors (the default)."""
    return unit_rows(session), unit_rows(enrolled)


def centre_on_session(
    session: FloatArray, enrolled: FloatArray
) -> Tuple[FloatArray, FloatArray]:
    """Normalisation that removes the session's mean from both sides.

    The mean of the session vectors is subtracted from the session and
    from the voiceprints alike, then every row is re-normalised. With
    fewer than :data:`MIN_GROUPS_TO_CENTRE` groups nothing is centred
    and the raw unit vectors are compared (:func:`unit_vectors`): one
    group centred on itself is a zero vector that matches nobody, and
    two groups become exact opposites. Even with more groups the mean
    is a poor estimate of the room; this is one of the options TR-05
    compares, not a recommendation.
    """
    if session.shape[0] < MIN_GROUPS_TO_CENTRE:
        return unit_vectors(session, enrolled)
    mean: FloatArray = session.mean(axis=0).astype(np.float32)
    return centre(session, mean), centre(enrolled, mean)


class Attribution(BaseModel):
    """Who voice group ``group`` of a session is.

    ``speaker`` is an enrolled user id (``enrolled`` true) or an
    anonymous label ``S<n>``. ``similarity`` is the cosine between the
    group and the voiceprint it was given, ``None`` when anonymous.
    ``nearest`` and ``nearest_similarity`` record the closest
    voiceprint whatever the decision, so thresholds can be studied.
    """

    model_config = ConfigDict(frozen=True)

    group: int = Field(ge=0)
    speaker: str = Field(min_length=1)
    enrolled: bool
    similarity: Optional[float] = Field(default=None, ge=-1, le=1)
    nearest: Optional[str] = None
    nearest_similarity: Optional[float] = Field(default=None, ge=-1, le=1)


def match_voices(
    groups: FloatArray,
    voiceprints: Sequence[Voiceprint],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    normalisation: Normalisation = unit_vectors,
    model: Optional[str] = None,
    taken: AbstractSet[str] = frozenset(),
) -> List[Attribution]:
    """Attribute the voice groups of a session to enrolled people.

    ``groups`` is a 2-D array with one finite vector per group (row
    ``i`` is group ``i``). Anonymous labels are numbered in row order,
    so the caller must pass the groups in order of first appearance
    (in the final, attributed words) for ``S1`` to be the first unknown
    voice heard. Both sides go through ``normalisation`` and are
    compared by cosine.

    The rule: the one-to-one assignment maximises the *total*
    similarity of the accepted pairs, those at or above ``threshold``
    (``linear_sum_assignment`` on similarities with the pairs below the
    threshold set to zero), so a pair below the threshold never
    displaces an accepted one. It is not greedy: when two groups are
    near the same voiceprint and one of them also has an acceptable
    second voiceprint, both get an id even if that means the group
    most similar to the first voiceprint takes its second choice
    (g0: P 0.75, Q 0.65; g1: P 0.70 gives g0 Q, g1 P). When there is
    no acceptable alternative only the better group gets the id. A
    person is given to at most one group. TR-05 compares this rule with
    a greedy one.

    Groups left without a voiceprint are labelled ``S1``, ``S2``, ...
    in group order, skipping labels that are ids of ``voiceprints`` or
    in ``taken`` (callers pass every enrolled id, candidate or not, so
    an anonymous voice never wears a person's id). When
    ``model`` is given, every voiceprint must come from that voice
    model. Raises :class:`InvalidInputError` for a ``groups`` array
    that is not 2-D, a vector that is not finite, or voiceprints that
    do not fit.
    """
    ids = [v.speaker_id for v in voiceprints]
    if len(set(ids)) != len(ids):
        raise InvalidInputError("Each person may have one voiceprint.")
    if model is not None:
        other = sorted({v.model for v in voiceprints if v.model != model})
        if other:
            raise InvalidInputError(
                f"Voiceprints of model {', '.join(other)} cannot be"
                f" compared with {model}."
            )
    if groups.ndim != 2:
        raise InvalidInputError(
            "Session voices must be a 2-D array, one row per group."
        )
    if not bool(np.isfinite(groups).all()) or not all(
        math.isfinite(x) for v in voiceprints for x in v.vector
    ):
        raise InvalidInputError("Voice vectors must be finite numbers.")
    count = int(groups.shape[0])
    width = int(groups.shape[1])
    if any(len(v.vector) != width for v in voiceprints) and count:
        raise InvalidInputError(
            "Voiceprints and session voices have different sizes."
        )
    if count == 0:
        return []
    matched: Dict[int, Tuple[str, float]] = {}
    nearest: Dict[int, Tuple[str, float]] = {}
    if voiceprints:
        enrolled = np.array([v.vector for v in voiceprints], dtype=np.float32)
        session, prints = normalisation(groups.astype(np.float32), enrolled)
        similarity = np.clip(session @ prints.T, -1.0, 1.0).astype(np.float64)
        if not bool(np.isfinite(similarity).all()):
            raise InvalidInputError(
                "The normalisation gave voice vectors that are not finite."
            )
        for group in range(count):
            best = int(np.argmax(similarity[group]))
            nearest[group] = (ids[best], float(similarity[group, best]))
        gain = np.where(similarity >= threshold, similarity, 0.0)
        rows, cols = linear_sum_assignment(gain, maximize=True)
        for row, col in zip(rows.tolist(), cols.tolist()):
            if similarity[row, col] >= threshold:
                matched[row] = (ids[col], float(similarity[row, col]))
    anonymous = _anonymous_labels(count - len(matched), set(ids) | taken)
    attributions: List[Attribution] = []
    for group in range(count):
        near_id, near_sim = nearest.get(group, (None, None))
        if group in matched:
            speaker, score = matched[group]
            attributions.append(Attribution(
                group=group, speaker=speaker, enrolled=True,
                similarity=score, nearest=near_id,
                nearest_similarity=near_sim,
            ))
        else:
            attributions.append(Attribution(
                group=group, speaker=anonymous.pop(0), enrolled=False,
                nearest=near_id, nearest_similarity=near_sim,
            ))
    return attributions


class SeededVoices(NamedTuple):
    """The voices of a recording found by :func:`seed_voices`.

    ``coarse``, ``fine`` and ``vectors`` are as in :class:`VoiceGroups`
    (group ``i`` is row ``i`` of ``vectors``). ``people`` gives the
    enrolled id of each group a voiceprint claimed; the other groups are
    voices nobody enrolled. ``fine_scores`` is as in
    :class:`VoiceGroups`.
    """

    coarse: List[int]
    fine: List[int]
    vectors: FloatArray
    people: Dict[int, str]
    fine_scores: Optional[FloatArray] = None


def seed_voices(
    coarse: FloatArray,
    fine: FloatArray,
    voiceprints: Sequence[Voiceprint],
    num_speakers: Optional[int] = None,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    normalisation: Normalisation = unit_vectors,
    group_threshold: float = 0.5,
    max_speakers: int = 10,
    one_voice: Optional[Callable[[FloatArray], bool]] = None,
) -> SeededVoices:
    """Find the voices of a recording with the voiceprints as seeds.

    The method TR-05 measured best (``docs/transcriber/
    IDENTIFICATION.md``, "voiceprints as seeds"):

    1. each long window (``coarse``) takes the enrolled person it is
       most similar to, when the cosine after ``normalisation`` (of the
       windows and the voiceprints) reaches ``threshold``; every person
       claimed by at least one window is a group;
    2. the windows nobody claims are clustered as :func:`group_voices`
       does, on vectors centred on the recording's mean: into the
       people still missing when ``num_speakers`` is known, else by
       ``group_threshold`` (at most ``max_speakers``), or as one voice
       when ``one_voice`` says so. With nobody missing (``num_speakers``
       known and reached) they go to the nearest claimed person;
    3. each short window (``fine``) takes the nearest group centroid.

    ``num_speakers`` is a lower bound here, not a cap: every enrolled
    person at least one window claims is a group, so more people than
    ``num_speakers`` can be named (two voiceprints that clear
    ``threshold`` with ``num_speakers=1`` give two named groups). The
    count only says how many voices the unclaimed windows make. This is
    the method TR-05 measured; the voiceprints winning over a wrong
    count is what makes it better there.

    Groups are numbered claimed people first (in voiceprint order),
    then the unclaimed voices. ``vectors`` are the raw unit means of the
    groups' long windows. Raises :class:`InvalidInputError` like
    :func:`group_voices`, or without any voiceprint.
    """
    if not voiceprints:
        raise InvalidInputError("Seeding needs at least one voiceprint.")
    if coarse.ndim != 2 or fine.ndim != 2:
        raise InvalidInputError("Window embeddings must be 2-D arrays.")
    if num_speakers is not None and num_speakers < 1:
        raise InvalidInputError("num_speakers must be at least 1.")
    width = int(coarse.shape[1])
    if coarse.shape[0] == 0:
        return SeededVoices([], [], np.zeros((0, width), np.float32), {})
    if fine.shape[0] and fine.shape[1] != width:
        raise InvalidInputError("Long and short windows differ in size.")
    if any(len(v.vector) != width for v in voiceprints):
        raise InvalidInputError(
            "Voiceprints and session voices have different sizes."
        )
    ids = [v.speaker_id for v in voiceprints]
    matrix = np.array([v.vector for v in voiceprints], dtype=np.float32)
    windows, enrolled = normalisation(coarse.astype(np.float32), matrix)
    similarity = windows @ enrolled.T
    best = similarity.argmax(axis=1).tolist()
    claimed = (similarity.max(axis=1) >= threshold).tolist()
    people = sorted({int(b) for b, c in zip(best, claimed) if c})
    labels = [people.index(int(b)) if c else -1
              for b, c in zip(best, claimed)]
    free = [i for i, c in enumerate(claimed) if not c]
    mean: FloatArray = coarse.mean(axis=0).astype(np.float32)
    centred = centre(coarse, mean)
    if free:
        missing = None if num_speakers is None else num_speakers - len(people)
        if missing is not None and missing <= 0 and people:
            anchors = np.vstack([
                centred[[i for i, v in enumerate(labels) if v == k]].mean(
                    axis=0) for k in range(len(people))
            ])
            for i, k in zip(free, nearest_voice(centred[free],
                                                unit_rows(anchors))):
                labels[i] = k
        else:
            if missing is None and one_voice is not None and one_voice(
                    coarse[free]):
                others = [0] * len(free)
            else:
                others = cluster_speakers(
                    centred[free],
                    None if missing is None else max(missing, 1),
                    group_threshold, max_speakers,
                )
            for i, a in zip(free, others):
                labels[i] = len(people) + a
    centroids = voice_centroids(centred, labels)
    fine_centred = centre(fine, mean) if fine.shape[0] else fine
    short = nearest_voice(fine_centred, centroids)
    return SeededVoices(
        labels, short, voice_centroids(unit_rows(coarse), labels),
        {k: ids[people[k]] for k in range(len(people))},
        _scores(fine_centred, centroids),
    )


def _scores(
    windows: FloatArray, centroids: FloatArray
) -> Optional[FloatArray]:
    """Cosines of unit ``windows`` (rows) with unit ``centroids``.

    ``None`` when there is no window or no centroid to compare.
    """
    if windows.shape[0] == 0 or centroids.shape[0] == 0:
        return None
    out: FloatArray = (windows @ centroids.T).astype(np.float32)
    return out


def name_seeded(
    order: Sequence[int],
    seeded: SeededVoices,
    voiceprints: Sequence[Voiceprint],
    normalisation: Normalisation = unit_vectors,
    taken: AbstractSet[str] = frozenset(),
) -> List[Attribution]:
    """Attribute the groups of :func:`seed_voices` in ``order``.

    ``order`` lists group numbers in order of first appearance in the
    final words; attribution ``i`` is for ``order[i]`` (its ``group``
    is ``i``, as with :func:`match_voices`). A group a voiceprint
    claimed gets that person, with the cosine between the group's
    vector and the voiceprint (after ``normalisation``) as its
    similarity; the others get ``S1``, ``S2``, ... in order, skipping
    the ids of ``voiceprints`` and ``taken``.
    """
    ids = [v.speaker_id for v in voiceprints]
    anonymous = _anonymous_labels(
        sum(1 for g in order if g not in seeded.people), set(ids) | taken
    )
    similarity = np.zeros((len(order), len(ids)), dtype=np.float64)
    if order and ids:
        matrix = np.array([v.vector for v in voiceprints], dtype=np.float32)
        groups, prints = normalisation(seeded.vectors[list(order)], matrix)
        similarity = np.clip(groups @ prints.T, -1.0, 1.0).astype(np.float64)
    attributions: List[Attribution] = []
    for row, group in enumerate(order):
        near = int(np.argmax(similarity[row])) if ids else None
        near_id = ids[near] if near is not None else None
        near_sim = float(similarity[row, near]) if near is not None else None
        person = seeded.people.get(group)
        if person is not None:
            attributions.append(Attribution(
                group=row, speaker=person, enrolled=True,
                similarity=float(similarity[row, ids.index(person)]),
                nearest=near_id, nearest_similarity=near_sim,
            ))
        else:
            attributions.append(Attribution(
                group=row, speaker=anonymous.pop(0), enrolled=False,
                nearest=near_id, nearest_similarity=near_sim,
            ))
    return attributions


def anonymous_label(number: int) -> str:
    """Return the anonymous label of the ``number``-th unknown voice."""
    return f"{ANONYMOUS_PREFIX}{number}"


def _anonymous_labels(count: int, taken: set[str]) -> List[str]:
    """Return ``count`` labels ``S1``, ``S2``, ... not in ``taken``."""
    labels: List[str] = []
    number = 1
    while len(labels) < count:
        label = anonymous_label(number)
        if label not in taken:
            labels.append(label)
        number += 1
    return labels
