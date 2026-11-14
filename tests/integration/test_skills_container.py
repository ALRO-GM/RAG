"""SkillsContainer wiring and the model commands run in-process."""

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

import src.skills_bootstrap as bootstrap
from src.adapters.inbound.cli import safe_command
from src.adapters.inbound.lab_cli import LabCLI
from src.adapters.inbound.skills_cli import SkillsCLI, fmt, parse_names
from src.adapters.outbound.anthropic_chat_model import AnthropicChatModel
from src.adapters.outbound.bm25_index import BM25Builder
from src.adapters.outbound.transformers_chat_model import (
    TransformersChatModel,
)
from src.config import SkillsSettings
from src.domain.errors import InvalidInputError, NotFoundError
from src.domain.experiments import TrialResult
from src.domain.skills.assessment import SessionAssessment
from src.skills_bootstrap import SkillsContainer, SkillsLabFactory
from tests.fakes import FakeChatModel, FakeEmbedder, judge_reply
from tests.samples import write_json
from tests.skills_samples import (
    EN,
    ES,
    GOLD,
    RUBRICS,
    TRANSCRIPTS,
    write_form,
    write_rubrics,
    write_transcripts,
)

REPLIES = {
    ("Maya", "listening"): judge_reply(
        [(0, "What do you mean exactly", "L1"),
         (3, "So if I understood correctly, the big file must work", "L2")],
        rating=4, rationale="Asks and summarises.",
    ),
    ("Tom", "planning"): judge_reply(
        [(4, "Planning is a waste of time", "P2"),
         (7, "I will finish it by Friday", "P1"),
         (7, "Tom promised to help everybody", "P1")], rating=3,
    ),
    ("Luis", "planning"): judge_reply(
        [(4, "Planificar es perder el tiempo", "P2"),
         (7, "lo termino el viernes entonces", "P1")], rating=2,
    ),
}


def judge() -> FakeChatModel:
    """Return a judge scripted for the sample meetings."""

    def responder(system: str, user: str) -> str:
        """Reply by participant and competency of the prompt."""
        who = re.search(r"Participant to assess: (.+)", user)
        what = re.search(r"\((\w+)\)", user.splitlines()[0])
        assert who is not None and what is not None
        return REPLIES.get((who.group(1), what.group(1)), "{}")

    return FakeChatModel(responder=responder)


def identity(function: Callable[..., Any]) -> Callable[..., Any]:
    """A guard that lets errors through, to assert on them."""
    return function


@pytest.fixture()
def skills_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write rubrics and transcripts under ``tmp_path`` and chdir there."""
    write_rubrics(tmp_path)
    write_transcripts(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def settings(**overrides: Any) -> SkillsSettings:
    """Return settings for the tmp project with explicit thresholds."""
    values: Dict[str, Any] = {
        "rubric_dir": RUBRICS, "transcripts_dir": TRANSCRIPTS,
        "output_dir": "out", "cache_dir": "cache", "lab_dir": "lab",
        "progress": False, "min_incidents": 2, "min_episodes": 1,
        "min_participant_words": 0, "min_similarity": 0.0,
        "candidates": 8, "min_words": 4, "semantic_weight": 1.0,
    }
    return SkillsSettings.model_validate({**values, **overrides})


def container(**overrides: Any) -> SkillsContainer:
    """Return a container with a fake embedder and the scripted judge."""
    return SkillsContainer(
        settings(**overrides), embedder=FakeEmbedder(dim=256), judge=judge()
    )


def cli(app: SkillsContainer, guard: Any = identity) -> Any:
    """Return the skills CLI bound to an already built container."""
    commands = SkillsCLI(guard, {})
    commands._container = app
    return commands


class CountingEmbedder(FakeEmbedder):
    """A fake embedder that counts its constructions."""

    created: List[str] = []

    def __init__(
        self, model_name: str, max_seq_length: int = 0, plan: object = None
    ):
        """Record the construction."""
        super().__init__(model_name, 256)
        CountingEmbedder.created.append(f"{model_name}|{max_seq_length}")


@pytest.fixture()
def fake_models(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Replace the real embedder class of the composition root."""
    CountingEmbedder.created = []
    monkeypatch.setattr(
        bootstrap, "SentenceTransformerEmbedder", CountingEmbedder
    )
    return CountingEmbedder.created


# ---------------------------------------------------------------- wiring


def test_listing_a_catalogue_builds_no_model(
    skills_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catalogues, transcripts and agreement need no embedder or judge."""
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        """Fail if a model adapter is constructed."""
        raise AssertionError("a model adapter was constructed")

    for name in ("SentenceTransformerEmbedder", "TransformersChatModel",
                 "AnthropicChatModel", "SqliteCache"):
        monkeypatch.setattr(bootstrap, name, forbidden)
    app = SkillsContainer(settings())
    assert app.catalogues.load("core", "es").language == "es"
    assert app.rubric("en").name == "core"
    assert app.transcripts.read(EN).session_id == "s1"
    assert not {"embedder", "judge", "retriever", "assessment", "cache",
                "experiments"} & set(vars(app))


def test_container_uses_the_shared_embedder_and_judge(
    skills_project: Path,
) -> None:
    """Injected models are the ones the services use."""
    embedder, model = FakeEmbedder(), FakeChatModel()
    app = SkillsContainer(settings(), embedder=embedder, judge=model)
    assert app.embedder is embedder and app.judge is model
    assert app.retriever.embedder is embedder
    assert app.assessment.judge is model
    assert app.assessment.retriever is app.retriever


def test_container_builds_lazy_model_adapters(skills_project: Path) -> None:
    """The default adapters are configured but load nothing."""
    local = SkillsContainer(settings(llm="some/local-model"))
    assert isinstance(local.judge, TransformersChatModel)
    assert local.judge.model_name == "some/local-model"
    assert local.judge._model is None
    api = SkillsContainer(settings(
        judge="api", api_model="claude-haiku-5-5", api_effort="low"
    ))
    assert isinstance(api.judge, AnthropicChatModel)
    assert (api.judge.model_name, api.judge.effort) == (
        "claude-haiku-5-5", "low",
    )
    assert api.judge._client is None
    assert local.embedder.model_name == settings().embedding_model


def test_container_retriever_follows_the_settings(
    skills_project: Path,
) -> None:
    """Retrieval settings reach the retriever; lexical mode drops dense."""
    app = container(
        semantic_weight=0.25, min_words=6, query_prefix="q: ",
        passage_prefix="p: ", bm25_k1=0.9, bm25_b=0.3,
    )
    retriever = app.retriever
    assert (retriever.semantic_weight, retriever.min_words) == (0.25, 6)
    assert (retriever.query_prefix, retriever.passage_prefix) == (
        "q: ", "p: ",
    )
    assert isinstance(retriever.lexical_builder, BM25Builder)
    assert retriever.embedder is app.embedder
    assert container(semantic=False).retriever.embedder is None


def test_container_assessment_follows_the_settings(
    skills_project: Path,
) -> None:
    """Every threshold of the settings reaches the use case."""
    service = container(
        candidates=5, min_incidents=3, min_episodes=2, episode_gap=4,
        min_participant_words=33, min_similarity=0.45, max_new_tokens=99,
    ).assessment
    assert (service.candidates, service.min_incidents) == (5, 3)
    assert (service.min_episodes, service.episode_gap) == (2, 4)
    assert (service.min_words, service.min_similarity) == (33, 0.45)
    assert service.max_tokens == 99


def test_container_rubric_language(skills_project: Path) -> None:
    """``auto`` follows the transcript; a fixed language overrides it."""
    assert container().rubric("es").language == "es"
    assert container(rubric_language="en").rubric("es").language == "en"
    (skills_project / RUBRICS / "core.es.json").unlink()
    assert container().rubric("es").language == "en"
    with pytest.raises(NotFoundError):
        container(catalogue="other").rubric("en")


def test_settings_reject_invalid_values() -> None:
    """Out-of-range thresholds do not validate."""
    for bad in ({"candidates": 0}, {"min_incidents": 0}, {"min_words": 0},
                {"judge": "cloud"}, {"min_similarity": 1.5},
                {"semantic_weight": -1.0}, {"api_effort": "extreme"},
                {"min_episodes": 0}, {"episode_gap": -1}):
        with pytest.raises(ValueError):
            settings(**bad)


def test_skills_options_do_not_collide_with_the_global_options() -> None:
    """No ``skills`` option shares its name with a global RAG option."""
    import inspect

    from src.adapters.inbound.cli import RagCLI

    globals_ = set(inspect.signature(RagCLI.__init__).parameters)
    own = set(inspect.signature(RagCLI.skills).parameters)
    assert (globals_ & own) - {"self"} == set()


def test_cli_helpers() -> None:
    """Name lists and optional statistics are parsed and formatted."""
    assert parse_names("Maya, Tom ,") == ["Maya", "Tom"]
    assert parse_names(("Maya", " Tom")) == ["Maya", "Tom"]
    assert parse_names(7) == ["7"]
    for empty in (None, "", (), " , "):
        assert parse_names(empty) is None
    assert fmt(None) == "n/a"
    assert fmt(0.12345) == "0.123"


# ------------------------------------------------- assess (in-process)


def test_assess_writes_scorecards_and_prints_ratings(
    skills_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``assess`` saves JSON and Markdown and summarises the ratings."""
    app = container()
    cli(app).assess(EN)
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].split() == ["Maya", "listening=4", "planning=-"]
    assert lines[1].split() == ["Tom", "listening=-", "planning=3"]
    assert lines[2].split() == ["Priya", "listening=-", "planning=-"]
    assert "out/s1.assessment.json" in lines[3]
    assert "out/s1.scorecard.md" in lines[3]
    saved = SessionAssessment.model_validate_json(
        (skills_project / "out/s1.assessment.json").read_text("utf-8")
    )
    assert saved.ratings()["Maya"] == {"listening": 4, "planning": None}
    assert saved.judge_model == "fake-judge"
    assert saved.embedding_model == "fake-embedder"
    tom = saved.scorecard("Tom")
    assert tom is not None
    planning = tom.assessment("planning")
    assert planning is not None
    assert (planning.proposed, planning.rejected) == (3, 1)
    assert all(e.similarity is not None for e in planning.evidence)
    report = (skills_project / "out/s1.scorecard.md").read_text("utf-8")
    assert "# Session s1" in report
    assert "| Active listening | 4 / 5 | 2 |" in report
    assert '"What do you mean exactly" — turn 0, 00:00.' in report
    assert "insufficient evidence" in report
    assert "(match " in report


def test_assess_selection_and_save_directory(
    skills_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Participants, competencies and the output directory can be set."""
    app = container()
    cli(app).assess(EN, participants="Tom", competencies="planning",
                    save_directory="elsewhere")
    assert capsys.readouterr().out.splitlines()[0].split() == [
        "Tom", "planning=3",
    ]
    assert (skills_project / "elsewhere/s1.assessment.json").is_file()
    assert not (skills_project / "out").exists()
    assert isinstance(app.judge, FakeChatModel)
    assert len(app.judge.calls) == 1


def test_assess_spanish_session_uses_the_spanish_rubric(
    skills_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The rubric follows the detected language of the transcript."""
    app = container()
    cli(app).assess(ES, participants="Luis")
    assert "planning=2" in capsys.readouterr().out
    assert isinstance(app.judge, FakeChatModel)
    system, user, _ = app.judge.calls[0]
    assert "sentences in Spanish" in system
    assert "Escucha activa" in user
    report = (skills_project / "out/es1.scorecard.md").read_text("utf-8")
    assert "language: es" in report and "Planificación" in report
    assert "turn 4. *P2" in report


def test_assess_twice_reuses_the_sqlite_cache(
    skills_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A second run with a fresh container asks the judge nothing."""
    first = container()
    cli(first).assess(EN)
    assert isinstance(first.judge, FakeChatModel)
    assert len(first.judge.calls) == 6
    second = container()
    cli(second).assess(EN)
    assert isinstance(second.judge, FakeChatModel)
    assert second.judge.calls == []
    outputs = capsys.readouterr().out.splitlines()
    assert outputs[:3] == outputs[4:7]
    uncached = container(use_cache=False)
    cli(uncached).assess(EN)
    assert isinstance(uncached.judge, FakeChatModel)
    assert len(uncached.judge.calls) == 6


def test_assess_thresholds_change_the_outcome(
    skills_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stricter settings withhold ratings the loose ones gave."""
    cli(container(min_episodes=2, use_cache=False)).assess(EN)
    strict = capsys.readouterr().out.splitlines()
    assert strict[0].split()[1] == "listening=-"
    saved = SessionAssessment.model_validate_json(
        (skills_project / "out/s1.assessment.json").read_text("utf-8")
    )
    maya = saved.scorecard("Maya")
    assert maya is not None
    listening = maya.assessment("listening")
    assert listening is not None and listening.flags == ["single_episode"]
    cli(container(min_participant_words=1000, use_cache=False)).assess(EN)
    floor = capsys.readouterr().out
    assert "=4" not in floor and "=3" not in floor


def test_assess_errors(skills_project: Path) -> None:
    """Unknown files, participants and competencies are user errors."""
    commands = cli(container())
    with pytest.raises(NotFoundError, match="Transcript not found"):
        commands.assess("missing.txt")
    with pytest.raises(InvalidInputError, match="Unknown participant"):
        commands.assess(EN, participants="Zoe")
    with pytest.raises(InvalidInputError, match="Unknown competency"):
        commands.assess(EN, competencies="charisma")
    assert not (skills_project / "out").exists()


def test_assess_error_through_the_real_guard(
    skills_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With the CLI guard an error is one stderr line and exit code 1."""
    commands = cli(container(), safe_command)
    with pytest.raises(SystemExit) as exit_info:
        commands.assess(EN, participants="Zoe")
    assert exit_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("Error: Unknown participant(s): Zoe.")
    assert captured.err.count("\n") == 1 and captured.out == ""


# ----------------------------------------------- evidence (in-process)


def test_evidence_lists_the_closest_turns(
    skills_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``evidence`` prints turn, score, indicator and text; no judge."""
    app = container()
    cli(app).evidence(EN, "Tom", "planning", k=2)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert re.match(r"\[4\] \d+\.\d{3} P2: Planning is a waste", lines[0])
    assert re.match(r"\[7\] \d+\.\d{3} P1: Fine, I will finish", lines[1])
    assert isinstance(app.judge, FakeChatModel) and app.judge.calls == []


def test_evidence_lexical_only_and_spanish(
    skills_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Lexical mode works on a Spanish session with the Spanish rubric."""
    cli(container(semantic=False)).evidence(ES, "Luis", "planning")
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("[4] ") and " P2: Planificar" in lines[0]


def test_evidence_errors_and_empty_result(
    skills_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unknown names are errors; k = 0 says there is nothing."""
    commands = cli(container())
    with pytest.raises(InvalidInputError, match="Unknown participant 'Zoe'"):
        commands.evidence(EN, "Zoe", "planning")
    with pytest.raises(InvalidInputError, match="Unknown competency"):
        commands.evidence(EN, "Tom", "charisma")
    with pytest.raises(NotFoundError):
        commands.evidence("missing.txt", "Tom", "planning")
    commands.evidence(EN, "Tom", "planning", k=0)
    assert capsys.readouterr().out == "No eligible turn.\n"


# ----------------------------------------------- evaluate (in-process)


def test_evaluate_assessment_against_reference(
    skills_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The saved assessment is scored against a reference file."""
    commands = cli(container())
    commands.assess(EN)
    write_form(skills_project / "ref.json", "consensus", GOLD)
    capsys.readouterr()
    commands.evaluate("out/s1.assessment.json", "ref.json",
                      output="out/eval.json")
    out = capsys.readouterr().out
    assert out.startswith("6 units in 1 session(s)\n")
    assert "retrieval recall 1.000, precision 1.000" in out
    assert "ratings (2 rated by both): exact 1.000" in out
    assert "rated-or-not: both 2, neither 2, system only 0, reference only 2" \
        in out
    assert "quotes verified: 0.800" in out
    saved = json.loads((skills_project / "out/eval.json").read_text("utf-8"))
    assert saved["units"] == 6
    assert saved["evidence"]["recall"] == pytest.approx(4 / 6)
    assert sorted(saved["per_competency"]) == ["listening", "planning"]


# ------------------------------------------------------ the skills lab


def test_lab_factory_rejects_unknown_parameters(skills_project: Path) -> None:
    """A parameter that is not a skills setting is a user error."""
    factory = SkillsLabFactory(settings())
    with pytest.raises(InvalidInputError, match=r"Unknown parameter\(s\): "
                       "file_boost, nope"):
        factory.create({"nope": 1, "file_boost": 0.5, "semantic": False})


@pytest.mark.parametrize("params", [
    {"candidates": 0}, {"semantic_weight": -1}, {"min_words": "many"},
    {"judge": "cloud"}, {"min_similarity": 3},
])
def test_lab_factory_rejects_invalid_values(
    skills_project: Path, params: Dict[str, Any]
) -> None:
    """An invalid value is reported in one line."""
    with pytest.raises(InvalidInputError, match="Invalid parameters") as err:
        SkillsLabFactory(settings()).create(params)
    assert "\n" not in str(err.value)


def test_lab_factory_memoises_pipelines_per_parameter_set(
    skills_project: Path, fake_models: List[str]
) -> None:
    """Equal parameters give the same pipeline; different ones do not."""
    factory = SkillsLabFactory(settings())
    first = factory.create({"semantic_weight": 0.5})
    assert factory.create({"semantic_weight": 0.5}) is first
    assert factory.create({"semantic_weight": 0.5, "progress": True}) is first
    assert factory.create({}) is not first
    assert factory.create({"semantic_weight": 0.7}) is not first
    assert first.retriever.semantic_weight == 0.5
    assert first.catalogue == "core" and first.rubric_language == "auto"


def test_lab_factory_shares_the_embedder_between_parameter_sets(
    skills_project: Path, fake_models: List[str]
) -> None:
    """One embedder per (model, max tokens), whatever else changes."""
    factory = SkillsLabFactory(settings(embedding_model="model-a"))
    a = factory.create({"semantic_weight": 0.5})
    b = factory.create({"semantic_weight": 2.0, "min_words": 2})
    c = factory.create({"embedding_model": "model-b"})
    d = factory.create({"embedding_max_tokens": 64})
    assert a.retriever.embedder is b.retriever.embedder
    assert c.retriever.embedder is not a.retriever.embedder
    assert d.retriever.embedder is not a.retriever.embedder
    assert fake_models == ["model-a|128", "model-b|128", "model-a|64"]
    lexical = factory.create({"semantic": False})
    assert lexical.retriever.embedder is None


def test_container_lab_uses_the_shared_embedder(skills_project: Path) -> None:
    """The lab of a container reuses the embedder it was given."""
    shared = FakeEmbedder(dim=256)
    app = SkillsContainer(settings(), embedder=shared)
    factory = app.experiments.factory
    assert isinstance(factory, SkillsLabFactory)
    assert factory.create({}).retriever.embedder is shared


def test_lab_factory_shared_embedder_serves_the_base_model_only(
    skills_project: Path, fake_models: List[str]
) -> None:
    """Another embedding model in a trial gets its own embedder."""
    shared = FakeEmbedder(dim=256)
    factory = SkillsLabFactory(settings(), embedder=shared)
    assert factory.create({"semantic_weight": 2.0}).retriever.embedder \
        is shared
    other = factory.create({"embedding_model": "model-b"})
    assert other.retriever.embedder is not shared
    assert fake_models == ["model-b|128"]


def write_lab_files(project: Path) -> None:
    """Write a dataset from the gold annotations and a two-trial suite."""
    form = write_form(project / "ref.json", "consensus", GOLD)
    commands = cli(SkillsContainer(settings()))
    commands.lab_dataset(str(form), "lab/evidence.json")
    write_json(project / "lab/suites/modes.json", {
        "name": "modes", "base": {"semantic": True},
        "factors": {"semantic": [False]},
    })
    write_json(project / "lab/suites/unknown.json", {
        "name": "unknown", "base": {"nope": 1},
    })


def test_lab_experiment_runs_on_evidence_retrieval(
    skills_project: Path, fake_models: List[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``skills lab experiment`` measures retrieval and stores trials."""
    write_lab_files(skills_project)
    app = SkillsContainer(settings())
    lab = LabCLI(lambda: app, identity, default_datasets="")
    capsys.readouterr()
    lab.experiment("modes", datasets="lab/evidence.json")
    out = capsys.readouterr().out
    assert out.startswith("Suite modes: 2 trials\n")
    assert out.count("evidence=1.000") == 2
    lines = (skills_project / "lab/results/trials.jsonl").read_text(
        "utf-8"
    ).splitlines()
    trials = [TrialResult.model_validate_json(line) for line in lines]
    assert [t.params["semantic"] for t in trials] == [True, False]
    assert all(t.n_chunks == 16 for t in trials)
    assert fake_models == [f"{settings().embedding_model}|128"]
    lab.experiment("modes", datasets="lab/evidence.json")
    assert len((skills_project / "lab/results/trials.jsonl").read_text(
        "utf-8"
    ).splitlines()) == 2


def test_lab_errors_are_clean(
    skills_project: Path, fake_models: List[str]
) -> None:
    """No dataset, unknown parameters and the router give user errors."""
    write_lab_files(skills_project)
    app = SkillsContainer(settings())
    lab = LabCLI(lambda: app, identity, default_datasets="")
    with pytest.raises(InvalidInputError, match="--datasets must list"):
        lab.experiment("modes")
    with pytest.raises(InvalidInputError, match="Unknown parameter"):
        lab.experiment("unknown", datasets="lab/evidence.json")
    with pytest.raises(NotFoundError, match="Suite not found"):
        lab.experiment("nosuite", datasets="lab/evidence.json")
    with pytest.raises(InvalidInputError, match="no query router to train"):
        lab.train_router()
    with pytest.raises(InvalidInputError, match="no query router to train"):
        lab.train_router(datasets="lab/evidence.json")


def test_skills_cli_lab_command_hosts_the_lab(skills_project: Path) -> None:
    """``skills lab`` returns the lab commands bound to the container."""
    commands = cli(SkillsContainer(settings()))
    lab = commands.lab()
    assert isinstance(lab, LabCLI)
    with pytest.raises(InvalidInputError, match="no query router to train"):
        lab.train_router()


# -- sentence tags ---------------------------------------

def test_sentences_and_tags_of_a_stored_transcript(
    skills_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``sentences`` lists them; ``tag`` saves candidates per sentence."""
    commands = cli(container(min_similarity=-1.0, tags_per_sentence=1))
    commands.sentences(EN)
    listed = capsys.readouterr().out.splitlines()
    assert listed[0].startswith("[00:00:")
    assert " t0.s0 Maya: " in listed[0]
    commands.tag(EN)
    out = capsys.readouterr().out
    assert "with candidate tags" in out
    assert "Saved out/s1.tags.json" in out
    saved = json.loads(Path("out/s1.tags.json").read_text(encoding="utf-8"))
    assert saved["session_id"] == "s1" and saved["catalogue"] == "core"
    tags = {
        s["sentence"]["text"]: [t["indicator_id"] for t in s["tags"]]
        for s in saved["sentences"]
    }
    nonsense = next(v for k, v in tags.items() if "nonsense" in k)
    assert nonsense == ["L3"]
    commands.tag(EN, save_directory="other", show=False)
    quiet = capsys.readouterr().out
    assert "Maya:" not in quiet and "Saved other/s1.tags.json" in quiet


def test_container_wires_the_tagger(skills_project: Path) -> None:
    """The tagger takes its limits from the settings and loads no model."""
    app = container(tags_per_sentence=3)
    assert app.tagger.top == 3


def test_tag_confirm_asks_the_judge_and_saves_decisions(
    skills_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``tag --confirm`` adds a decision per candidate sentence."""
    app = SkillsContainer(
        settings(min_similarity=-1.0, tags_per_sentence=1),
        embedder=FakeEmbedder(dim=256),
        judge=FakeChatModel(responder=lambda system, user: json.dumps({
            "indicator": "L3" if "nonsense" in user.split("TARGET")[1]
            else None,
            "reason": "said so",
        })),
    )
    cli(app).tag(EN, confirm=True)
    out = capsys.readouterr().out
    assert "Judge (fake-judge): 1 confirmed" in out
    assert "0 not understood" in out
    assert "[-L3]" in out and "[none]" in out
    saved = json.loads(Path("out/s1.tags.json").read_text(encoding="utf-8"))
    decisions = [
        s["decision"] for s in saved["sentences"] if "decision" in s
    ]
    confirmed = [d for d in decisions if d["status"] == "confirmed"]
    assert confirmed == [{
        "status": "confirmed", "indicator_id": "L3",
        "competency_id": "listening", "polarity": "negative",
        "reason": "said so", "model": "fake-judge",
    }]
    assert all(d["status"] == "rejected" for d in decisions
               if d not in confirmed)
