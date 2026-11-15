"""Every model leaves memory after each job, even a failed one (#117).

The real job runner and GPU-memory adapter, around the transcription
and name-search use cases with models that hold stand-in weights: after
each job (done, failed in a stage, failed in the names step, a waited-for
call that failed) no weights are left alive. No GPU is used.
"""

import gc
import weakref
from types import SimpleNamespace
from typing import Any, List, Optional

import pytest

from tests.unit.transcriber.fakes import FakeOutOfMemory
from tests.unit.transcriber.names_fakes import FakeChatModel
from tests.unit.transcriber.test_speaker_names import Kit, understands
from tests.unit.transcriber.test_transcribe import world
from transcriber.adapters.outbound.thread_job_runner import ThreadJobRunner
from transcriber.adapters.outbound.torch_gpu_memory import (
    TorchGpuMemory,
    gpu_processes,
)
from transcriber.adapters.outbound.torch_runtime import empty_cuda_cache
from transcriber.adapters.outbound.whisper_recogniser import (
    without_encoder_attentions,
)
from transcriber.application.speaker_names import FindSpeakerNames
from transcriber.application.transcribe import (
    QueueTranscription,
    TranscribeSession,
)
from transcriber.domain.errors import GpuBusyError, ModelLoadError

WAIT = 10.0


class Weights:
    """Stands for a model's tensors."""


class Network:
    """A model whose forward fails deep inside, holding its weights."""

    def __init__(self) -> None:
        """Make the weights; patch forward like the Whisper adapter."""
        self.weights = Weights()
        without_encoder_attentions(self)

    def get_encoder(self) -> "Network":
        """The encoder is the network itself."""
        return self

    def forward(self, fail: Optional[Exception], **kwargs: Any) -> None:
        """Fail with ``fail``, as a CUDA kernel would mid-decoding."""
        cache = [self.weights]  # noqa: F841 - kept by the frame
        if fail is not None:
            raise fail


class Holder:
    """A model adapter that loads a :class:`Network` when used."""

    def __init__(self) -> None:
        """Nothing loaded."""
        self.network: Optional[Network] = None
        self.refs: List["weakref.ref[Weights]"] = []
        self.fail: Optional[Exception] = None

    def use(self) -> None:
        """Load the network and run it (failing with :attr:`fail`)."""
        if self.network is None:
            self.network = Network()
            self.refs.append(weakref.ref(self.network.weights))
        network = self.network
        network.forward(self.fail)

    def release(self) -> None:
        """Drop the network (no collection here: the adapter does it)."""
        self.network = None

    def alive(self) -> int:
        """How many networks it loaded are still in memory."""
        return sum(r() is not None for r in self.refs)


@pytest.fixture()
def no_gc() -> Any:
    """Automatic collection off: only the adapter's collect may free."""
    gc.collect()
    gc.disable()
    yield
    gc.enable()


def wired(recogniser: Holder, names: Holder) -> Any:
    """A world whose recogniser and name model run :class:`Holder`s."""
    w = world()
    recognise = w.recogniser.recognise

    def heard(audio: Any, language: Optional[str] = None) -> Any:
        recogniser.use()
        return recognise(audio, language)

    w.recogniser.recognise = heard  # type: ignore[method-assign]
    model = FakeChatModel(understands)
    complete = model.complete

    def answered(system: str, user: str, tokens: int) -> str:
        names.use()
        return complete(system, user, tokens)

    model.complete = answered  # type: ignore[method-assign,assignment]

    def release_models() -> None:
        recogniser.release()
        names.release()

    gpu = TorchGpuMemory(release_models, listing=lambda: "")
    runner = ThreadJobRunner(after_job=gpu.free)
    transcribe = TranscribeSession(
        w.sessions, w.voiceprints, w.decoder, w.recogniser, w.encoder,
        w.locks, gpu=gpu,
    )
    kit = Kit(model=model)
    find = FindSpeakerNames(
        w.sessions, kit.store, w.locks, kit.rename, kit.unname,
        kit.find._clock, model, gpu=gpu,
    )
    queue = QueueTranscription(
        w.sessions, w.voiceprints, runner, transcribe, w.locks,
        after=find.after_transcription,
    )
    return SimpleNamespace(w=w, runner=runner, queue=queue)


def test_a_done_job_leaves_no_model(no_gc: Any) -> None:
    recogniser, names = Holder(), Holder()
    s = wired(recogniser, names)
    s.w.upload()
    s.queue("s1")
    assert s.runner.wait(WAIT)
    assert s.runner.error("s1") is None
    assert len(recogniser.refs) == 1
    assert recogniser.alive() == 0 and names.alive() == 0


def test_a_stage_failing_inside_a_model_leaves_no_model(no_gc: Any) -> None:
    recogniser, names = Holder(), Holder()
    recogniser.fail = RuntimeError("cuDNN error")
    s = wired(recogniser, names)
    s.w.upload()
    s.queue("s1")
    assert s.runner.wait(WAIT)
    assert s.runner.error("s1") == "cuDNN error"
    assert recogniser.alive() == 0


def test_out_of_memory_twice_frees_every_model(no_gc: Any) -> None:
    recogniser, names = Holder(), Holder()
    recogniser.fail = FakeOutOfMemory()
    s = wired(recogniser, names)
    s.w.upload()
    s.queue("s1")
    assert s.runner.wait(WAIT)
    error = s.runner.error("s1")
    assert error is not None and "The GPU is busy" in error
    assert len(recogniser.refs) == 2 and recogniser.alive() == 0


def test_a_failing_name_model_is_freed(no_gc: Any) -> None:
    names = Holder()
    names.fail = RuntimeError("bitsandbytes error")
    model = FakeChatModel(understands)
    complete = model.complete

    def answered(system: str, user: str, tokens: int) -> str:
        names.use()
        return complete(system, user, tokens)

    model.complete = answered  # type: ignore[method-assign,assignment]
    gpu = TorchGpuMemory(names.release, listing=lambda: "")
    runner = ThreadJobRunner(after_job=gpu.free)
    kit = Kit(model=model)
    find = FindSpeakerNames(
        kit.sessions, kit.store, kit.locks, kit.rename, kit.unname,
        kit.find._clock, model, gpu=gpu,
    )
    runner.submit("s1", lambda p: find.after_transcription("s1", p))
    assert runner.wait(WAIT)
    assert runner.error("s1") is None
    record = kit.store.names("s1")
    assert record is not None and record.note is not None
    assert "bitsandbytes error" in record.note
    assert len(names.refs) == 1 and names.alive() == 0
    runner.close(WAIT)


def test_a_waited_call_that_failed_keeps_no_model(no_gc: Any) -> None:
    """An enrolment that fails: its error goes to the caller, lean."""
    holder = Holder()
    holder.fail = RuntimeError("boom")
    gpu = TorchGpuMemory(holder.release, listing=lambda: "")
    runner = ThreadJobRunner(after_job=gpu.free)
    with pytest.raises(RuntimeError) as caught:
        runner.run(holder.use)
    assert holder.alive() == 0
    assert caught.value.__traceback__ is not None  # still printable
    runner.close(WAIT)


def test_without_encoder_attentions_makes_no_reference_cycle(
    no_gc: Any,
) -> None:
    network = Network()
    ref = weakref.ref(network.weights)
    network.forward(None, output_attentions=True)
    del network
    assert ref() is None


class FakeCuda:
    """The parts of ``torch.cuda`` the adapter reads."""

    OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})

    def __init__(self, available: bool = True) -> None:
        """Nothing emptied yet."""
        self.available = available
        self.emptied = 0
        self.calls: List[str] = []

    def is_available(self) -> bool:
        """Whether there is a GPU."""
        return self.available

    def empty_cache(self) -> None:
        """Count it."""
        self.emptied += 1
        self.calls.append("empty_cache")

    def mem_get_info(self, device: int) -> Any:
        """1.5 GB free of 8."""
        return 1.5e9, 8.0e9

    def memory_allocated(self, device: int) -> int:
        """0.25 GB."""
        return 250_000_000

    def memory_reserved(self, device: int) -> int:
        """0.5 GB."""
        return 500_000_000


def fake_torch(cuda: FakeCuda, workspaces: bool = True) -> Any:
    """A torch module with ``cuda`` and its out-of-memory error.

    ``workspaces``: it can drop cuBLAS's workspaces.
    """
    native = SimpleNamespace()
    if workspaces:
        native._cuda_clearCublasWorkspaces = (
            lambda: cuda.calls.append("workspaces")
        )
    return SimpleNamespace(
        cuda=cuda, OutOfMemoryError=cuda.OutOfMemoryError, _C=native,
    )


class TestTorchGpuMemory:
    def test_tells_out_of_memory_from_other_errors(self) -> None:
        cuda = FakeCuda()
        gpu = TorchGpuMemory(torch_module=lambda: fake_torch(cuda))
        assert gpu.out_of_memory(cuda.OutOfMemoryError("x"))
        assert gpu.out_of_memory(RuntimeError("CUDA out of memory. Tried"))
        assert not gpu.out_of_memory(RuntimeError("shape mismatch"))
        assert not gpu.out_of_memory(MemoryError("batch_size"))
        try:
            try:
                raise cuda.OutOfMemoryError("while loading")
            except RuntimeError as exc:
                raise ModelLoadError("Cannot load Qwen") from exc
        except ModelLoadError as error:
            assert gpu.out_of_memory(error)

    def test_without_torch_only_the_message_tells(self) -> None:
        gpu = TorchGpuMemory(torch_module=lambda: None)
        assert gpu.out_of_memory(RuntimeError("CUDA error: out of memory"))
        assert not gpu.out_of_memory(GpuBusyError("busy"))
        assert gpu.usage() == "no CUDA in use"

    def test_free_releases_collects_and_empties_the_cache(self) -> None:
        cuda, released = FakeCuda(), []
        gpu = TorchGpuMemory(lambda: released.append(1),
                             torch_module=lambda: fake_torch(cuda))
        gpu.free()
        assert released == [1] and cuda.emptied == 1
        # The workspaces pin a model-sized segment unless dropped first.
        assert cuda.calls == ["workspaces", "empty_cache"]

    def test_an_older_torch_still_empties_the_cache(self) -> None:
        cuda = FakeCuda()
        empty_cuda_cache(fake_torch(cuda, workspaces=False))
        empty_cuda_cache(None)
        assert cuda.calls == ["empty_cache"]

    def test_free_empties_the_cache_even_if_a_release_fails(self) -> None:
        cuda = FakeCuda()

        def broken() -> None:
            raise RuntimeError("release failed")

        gpu = TorchGpuMemory(broken, torch_module=lambda: fake_torch(cuda))
        with pytest.raises(RuntimeError):
            gpu.free()
        assert cuda.emptied == 1

    def test_no_cuda_no_cache_to_empty(self) -> None:
        cuda = FakeCuda(available=False)
        gpu = TorchGpuMemory(torch_module=lambda: fake_torch(cuda))
        gpu.free()
        assert cuda.emptied == 0 and gpu.usage() == "no CUDA in use"

    def test_usage_and_holders(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cuda = FakeCuda()
        listing = (
            "586484, /opt/google/chrome/chrome --type=gpu-process, 55\n"
            "735694, .venv/bin/python, 6082\n"
            "not a line\n"
        )
        gpu = TorchGpuMemory(
            listing=lambda: listing, torch_module=lambda: fake_torch(cuda),
            own_pid=735694,
        )
        usage = (
            "this process 0.25 GB allocated, 0.50 GB reserved;"
            " GPU 1.50 of 8.00 GB free"
        )
        assert gpu.usage() == usage
        assert gpu.holders() == (
            "chrome (pid 586484): 55 MiB; this service (pid 735694):"
            f" 6082 MiB ({usage})"
        )
        with caplog.at_level(
            "DEBUG", "transcriber.adapters.outbound.torch_gpu_memory"
        ):
            gpu.free()
        assert [r.getMessage() for r in caplog.records] == [
            f"GPU memory before freeing: {usage}",
            f"GPU memory after freeing: {usage}",
        ]

    def test_holders_when_nvidia_smi_cannot_run(self) -> None:
        gpu = TorchGpuMemory(listing=lambda: "", torch_module=lambda: None)
        assert gpu.holders() == (
            "the processes cannot be listed (no CUDA in use)"
        )

    def test_gpu_processes_skips_what_does_not_parse(self) -> None:
        assert gpu_processes("12, /bin/a --x, 3\nx, y, 1\n, ,\n", 1) == [
            "a (pid 12): 3 MiB",
        ]
