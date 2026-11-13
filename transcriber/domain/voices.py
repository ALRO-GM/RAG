"""Voices: clustering voice embeddings and putting names on the groups.

Diarization embeds short windows of a recording and groups them into
anonymous voices (:func:`cluster_speakers`, :func:`voice_centroids`,
:func:`nearest_voice`, copied from ``src/domain/skills/speech.py``).
:func:`match_voices` then pairs each group with at most one enrolled
:class:`~transcriber.domain.models.Voiceprint`; the groups left over get
anonymous labels ``S1``, ``S2``, ...

How embeddings are normalised before they are compared is a parameter
(:data:`Normalisation`): centring on the mean of a recording, which
helps inside one recording, is meaningless for a one-voice enrolment
sample, and the right choice across recordings is still to be measured
(ticket TR-05). The default compares raw unit vectors. The thresholds
here are parameters with untuned defaults for the same reason.
"""

import math
from typing import (
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

# Cosine similarity a group needs to take an enrolled id. A starting
# value, not a measured one: ticket TR-05 measures and sets it.
DEFAULT_MATCH_THRESHOLD = 0.5

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


class VoiceGroups(NamedTuple):
    """The voices of a recording found by :func:`group_voices`.

    ``coarse`` and ``fine`` hold the group of each long and each short
    window (``fine`` is empty without short windows) and ``vectors``
    one raw unit-length mean per group (row ``i`` is group ``i``), the
    vector compared with enrolled voiceprints.
    """

    coarse: List[int]
    fine: List[int]
    vectors: FloatArray


def group_voices(
    coarse: FloatArray,
    fine: FloatArray,
    num_speakers: Optional[int] = None,
    threshold: float = 0.5,
    max_speakers: int = 10,
    centre_windows: bool = True,
) -> VoiceGroups:
    """Find the voices of a recording in two passes.

    1. *Who is there*: the long-window embeddings ``coarse`` are
       clustered (:func:`cluster_speakers` with ``num_speakers``,
       ``threshold`` and ``max_speakers``).
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
    if centre_windows:
        mean: FloatArray = coarse.mean(axis=0).astype(np.float32)
        long, short = centre(coarse, mean), centre(fine, mean)
    else:
        long, short = unit_rows(coarse), unit_rows(fine)
    labels = cluster_speakers(long, num_speakers, threshold, max_speakers)
    assigned = nearest_voice(short, voice_centroids(long, labels))
    return VoiceGroups(
        labels, assigned, voice_centroids(unit_rows(coarse), labels)
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
    in group order, skipping labels that are enrolled ids. When
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
    anonymous = _anonymous_labels(count - len(matched), set(ids))
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
