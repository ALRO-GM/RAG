"""Try a model stage again when the GPU ran out of memory (#117).

The GPU is shared with the RAG and another project's worker, so a stage
may find too little memory free. :func:`retry_on_gpu_memory` runs the
stage; if it fails with the GPU out of memory, the frames of the failure
are cleared (they hold the model's tensors), every model of the service
is freed and the stage runs once more, with a note saying so. A second
out-of-memory failure becomes a :class:`~transcriber.domain.errors.
GpuBusyError` whose message says the GPU is busy and who holds it.
"""

import logging
import traceback
from typing import Callable, Optional, Set, TypeVar

from transcriber.application.ports import GpuMemory
from transcriber.domain.errors import GpuBusyError

LOG = logging.getLogger(__name__)

T = TypeVar("T")

GPU_RETRY_NOTE = (
    "The GPU ran out of memory in {stage}; the models were freed and"
    " {stage} was tried again."
)
GPU_BUSY_MESSAGE = (
    "The GPU is busy: {stage} ran out of GPU memory again after this"
    " service freed its models. Holding the GPU: {holders}. Try again"
    " when it is free."
)


def retry_on_gpu_memory(
    gpu: Optional[GpuMemory],
    stage: str,
    work: Callable[[], T],
    note: Callable[[str], None],
) -> T:
    """Return ``work()``, trying once more after freeing the GPU.

    Without ``gpu`` (fake models) ``work`` simply runs. Any failure
    other than the GPU running out of memory is raised as it is. The
    retry is told through ``note``; a second out-of-memory failure
    raises :class:`GpuBusyError` after freeing the models again.
    """
    if gpu is None:
        return work()
    for attempt in range(2):
        try:
            return work()
        except Exception as exc:
            if not gpu.out_of_memory(exc):
                raise
            drop_frames(exc)
        gpu.free()
        if attempt == 0:
            LOG.warning("Out of GPU memory in %s; trying it again.", stage)
            note(GPU_RETRY_NOTE.format(stage=stage))
    raise GpuBusyError(
        GPU_BUSY_MESSAGE.format(stage=stage, holders=gpu.holders())
    )


def drop_frames(error: BaseException) -> None:
    """Clear the locals of the frames ``error`` (and its causes) passed.

    A failure inside a model keeps its frames, and with them the
    model's tensors, for as long as the exception lives; the traceback
    stays printable without them.
    """
    seen: Set[int] = set()
    current: Optional[BaseException] = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        current = current.__cause__ or current.__context__
