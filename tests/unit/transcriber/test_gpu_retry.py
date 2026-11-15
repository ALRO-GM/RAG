"""A stage that runs out of GPU memory is freed and tried again (#117)."""

import gc
import weakref
from typing import Any, List

import pytest

from tests.unit.transcriber.fakes import (
    FakeGpuMemory,
    FakeOutOfMemory,
)
from tests.unit.transcriber.names_fakes import FakeChatModel
from tests.unit.transcriber.test_speaker_names import Kit, understands
from tests.unit.transcriber.test_transcribe import World, world
from transcriber.application.gpu import (
    GPU_RETRY_NOTE,
    drop_frames,
    retry_on_gpu_memory,
)
from transcriber.application.speaker_names import FindSpeakerNames
from transcriber.application.transcribe import (
    QueueTranscription,
    TranscribeSession,
)
from transcriber.domain.errors import GpuBusyError, ModelLoadError
from transcriber.domain.models import SessionState


class Weights:
    """Stands for a model's tensors."""


def failing(errors: List[Exception], result: str = "ok") -> Any:
    """Return work raising each of ``errors`` in turn, then ``result``."""
    def work() -> str:
        if errors:
            raise errors.pop(0)
        return result
    return work


class TestRetryOnGpuMemory:
    def test_without_a_gpu_the_work_just_runs(self) -> None:
        with pytest.raises(FakeOutOfMemory):
            retry_on_gpu_memory(None, "x", failing([FakeOutOfMemory()]),
                                print)

    def test_success_frees_nothing(self) -> None:
        gpu = FakeGpuMemory()
        notes: List[str] = []
        assert retry_on_gpu_memory(gpu, "x", failing([]), notes.append) == (
            "ok"
        )
        assert gpu.freed == 0 and notes == []

    def test_other_errors_are_raised_without_freeing(self) -> None:
        gpu = FakeGpuMemory()
        notes: List[str] = []
        with pytest.raises(ValueError):
            retry_on_gpu_memory(gpu, "x", failing([ValueError("no")]),
                                notes.append)
        assert gpu.freed == 0 and notes == []

    def test_one_out_of_memory_frees_and_tries_again(self) -> None:
        gpu = FakeGpuMemory()
        notes: List[str] = []
        result = retry_on_gpu_memory(
            gpu, "recognise", failing([FakeOutOfMemory()]), notes.append
        )
        assert result == "ok" and gpu.freed == 1
        assert notes == [GPU_RETRY_NOTE.format(stage="recognise")]

    def test_two_say_the_gpu_is_busy_and_who_holds_it(self) -> None:
        gpu = FakeGpuMemory()
        notes: List[str] = []
        with pytest.raises(GpuBusyError) as caught:
            retry_on_gpu_memory(
                gpu, "recognise",
                failing([FakeOutOfMemory(), FakeOutOfMemory()]),
                notes.append,
            )
        message = str(caught.value)
        assert message.startswith("The GPU is busy: recognise")
        assert FakeGpuMemory.HOLDERS in message
        assert gpu.freed == 2 and len(notes) == 1

    def test_the_failed_frames_no_longer_hold_the_model(self) -> None:
        """Freed before the retry, even if something keeps the error."""
        kept: List[BaseException] = []
        refs: List["weakref.ref[Weights]"] = []

        def work() -> str:
            weights = Weights()
            refs.append(weakref.ref(weights))
            if len(refs) == 1:
                try:
                    raise FakeOutOfMemory()
                except FakeOutOfMemory as exc:
                    kept.append(exc)
                    raise
            return "ok"

        gc.disable()
        try:
            gpu = FakeGpuMemory()
            alive = []
            gpu.release = lambda: alive.append(refs[0]() is not None)
            assert retry_on_gpu_memory(gpu, "x", work, print) == "ok"
        finally:
            gc.enable()
        assert alive == [False] and kept


def test_drop_frames_follows_the_causes() -> None:
    refs: List["weakref.ref[Weights]"] = []

    def inner() -> None:
        weights = Weights()
        refs.append(weakref.ref(weights))
        raise FakeOutOfMemory()

    try:
        try:
            inner()
        except FakeOutOfMemory as exc:
            raise ModelLoadError("cannot load") from exc
    except ModelLoadError as outer:
        error = outer
    gc.disable()
    try:
        assert refs[0]() is not None
        drop_frames(error)
        assert refs[0]() is None
    finally:
        gc.enable()


def with_gpu(w: World, gpu: FakeGpuMemory) -> World:
    """Rewire ``w``'s use cases with ``gpu``."""
    w.transcribe = TranscribeSession(
        w.sessions, w.voiceprints, w.decoder, w.recogniser, w.encoder,
        w.locks, gpu=gpu,
    )
    w.queue = QueueTranscription(
        w.sessions, w.voiceprints, w.runner, w.transcribe, w.locks
    )
    return w


class TestTranscriptionStages:
    def test_recognise_is_tried_again_after_freeing(self) -> None:
        gpu = FakeGpuMemory()
        w = with_gpu(world(), gpu)
        w.recogniser.failures.append(FakeOutOfMemory())
        transcript = w.run()
        assert transcript.utterances and gpu.freed == 1
        assert len(w.recogniser.languages) == 2
        status = w.runner.status("s1")
        assert status is not None
        assert status.message == GPU_RETRY_NOTE.format(stage="recognise")

    def test_twice_out_of_memory_fails_saying_the_gpu_is_busy(self) -> None:
        gpu = FakeGpuMemory()
        w = with_gpu(world(), gpu)
        w.recogniser.failures += [FakeOutOfMemory(), FakeOutOfMemory()]
        w.upload()
        w.queue("s1")
        w.runner.run_all()
        assert isinstance(w.runner.errors["s1"], GpuBusyError)
        stored = w.sessions.sessions["s1"]
        assert stored.state is SessionState.FAILED
        assert stored.failure is not None
        assert stored.failure.startswith("recognise: The GPU is busy")
        assert FakeGpuMemory.HOLDERS in stored.failure
        assert gpu.freed == 2

    def test_another_failure_is_not_retried(self) -> None:
        gpu = FakeGpuMemory()
        w = with_gpu(world(), gpu)
        w.recogniser.failures.append(RuntimeError("broken"))
        w.upload()
        w.queue("s1")
        w.runner.run_all()
        assert str(w.runner.errors["s1"]) == "broken" and gpu.freed == 0

    def test_the_voices_stage_is_tried_again(self) -> None:
        gpu = FakeGpuMemory()
        w = with_gpu(world(), gpu)
        embed = w.encoder.embed
        failures = [FakeOutOfMemory()]

        def flaky(*args: Any) -> Any:
            if failures:
                raise failures.pop()
            return embed(*args)

        w.encoder.embed = flaky  # type: ignore[method-assign,assignment]
        assert w.run().utterances and gpu.freed == 1

    def test_scoring_out_of_memory_twice_keeps_the_words(self) -> None:
        """Scores are an extra: the job goes on without them."""
        gpu = FakeGpuMemory()
        w = with_gpu(world(), gpu)
        w.recogniser.score_failures += [FakeOutOfMemory(), FakeOutOfMemory()]
        transcript = w.run()
        assert transcript.utterances and gpu.freed == 2

    def test_the_gpu_is_logged_at_each_stage(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        w = with_gpu(world(), FakeGpuMemory())
        with caplog.at_level("DEBUG", "transcriber.application.transcribe"):
            w.run()
        lines = [r.getMessage() for r in caplog.records]
        assert "GPU memory when recognise starts: fake GPU: 0.00 GB" \
            " allocated" in lines
        assert any("stages ended" in line for line in lines)


class TestNames:
    def kit(self, model: FakeChatModel) -> "tuple[Kit, FakeGpuMemory]":
        """A kit whose search frees the model through a fake GPU."""
        kit = Kit(model=model)
        gpu = FakeGpuMemory(release=model.release)
        kit.find = FindSpeakerNames(
            kit.sessions, kit.store, kit.locks, kit.rename, kit.unname,
            kit.find._clock, model, free_models=lambda: None, gpu=gpu,
        )
        return kit, gpu

    def test_a_reading_out_of_memory_is_tried_again(self) -> None:
        model = FakeChatModel(understands)
        failures = [FakeOutOfMemory()]
        complete = model.complete

        def flaky(system: str, user: str, tokens: int) -> str:
            if failures:
                raise failures.pop()
            return complete(system, user, tokens)

        model.complete = flaky  # type: ignore[method-assign,assignment]
        kit, gpu = self.kit(model)
        record = kit.find("s1")
        assert [a.name for a in record.auto] == ["Ana", "Pedro"]
        assert record.note == GPU_RETRY_NOTE.format(stage="names")
        assert gpu.freed == 1
        # Released by the free before the retry and after the search.
        assert model.released == 2 and not model.loaded

    def test_twice_leaves_suggestions_and_says_the_gpu_is_busy(
        self,
    ) -> None:
        model = FakeChatModel(fail=FakeOutOfMemory())
        kit, gpu = self.kit(model)
        record = kit.find("s1")
        assert record.auto == [] and record.model is None
        assert record.note is not None and "The GPU is busy" in record.note
        assert FakeGpuMemory.HOLDERS in record.note
        assert gpu.freed == 2 and model.released == 3

    def test_the_model_is_released_after_any_failure(self) -> None:
        model = FakeChatModel(fail=RuntimeError("bad weights"))
        kit, gpu = self.kit(model)
        kit.find("s1")
        assert gpu.freed == 0 and model.released == 1
