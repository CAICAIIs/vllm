# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import importlib.util
import json
import os
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.engine import (
    MemoryRegion,
    RangeRequest,
    TransferBackend,
    TransferEngineKind,
    TransferEvent,
    TransferOp,
    TransferState,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.tent_config import (
    build_tent_config,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

# Status poll interval. TENT settles tasks on its own progress thread, so a
# waiter starts short and backs off to this cap.
POLL_INTERVAL_MIN_S = 0.00005
POLL_INTERVAL_MAX_S = 0.001

# Neighbouring ranges travel as one task up to this length, so a request of many
# small blocks does not pay one Python object per block.
MAX_MERGED_LENGTH = 1 << 20

# How long `shutdown` waits for canceled transfers to settle.
SHUTDOWN_TIMEOUT_S = 5.0


@dataclass
class _Task:
    """One TENT task covering `count` neighbouring `RangeRequest`s."""

    local_addr: int
    remote_addr: int
    length: int
    count: int


@dataclass
class _Submission:
    """A batch and the outcome of every task in it.

    `results` keeps one slot per task so events leave in range order.
    """

    tasks: list[_Task]
    results: list[TransferState | None]
    errors: list[str | None]
    lock: threading.Lock


class TentTransferBackend(TransferBackend):
    """Wraps TENT behind the backend interface.

    A submission keeps its batch until every task is terminal, which is also
    when the KV memory behind it can be released.
    """

    def __init__(
        self,
        *,
        hostname: str,
        protocol: str,
        device_name: str,
        metrics_port: int,
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self._tent = _import_tent_module()
        config = build_tent_config(
            hostname=hostname,
            protocol=protocol,
            device_name=device_name,
            metrics_port=metrics_port,
            overrides=overrides,
        )
        # TENT builds its segment name from this hostname, so a name that does
        # not resolve would surface much later as an engine peers cannot reach.
        socket.getaddrinfo(hostname, None)
        config_path = _write_config_file(config)
        try:
            self._engine = self._tent.TransferEngine(config_path)
        finally:
            os.unlink(config_path)
        if not self._engine.available():
            raise RuntimeError(
                f"The TENT transfer engine is not available over {protocol!r} on "
                f"{hostname!r}."
            )
        rpc_port = int(self._engine.get_rpc_server_port())
        advertised = self._engine.get_segment_name()
        expected = _peer_address(hostname, rpc_port)
        if advertised != expected:
            raise RuntimeError(
                f"TENT advertises segment {advertised!r} while peers reach this "
                f"worker as {expected!r}; set rpc_server_hostname through "
                "tent_config."
            )
        self._rpc_port = rpc_port
        self._segments: dict[str, int] = {}
        self._segment_lock = threading.Lock()
        self._batches: dict[int, _Submission] = {}
        self._batches_lock = threading.Lock()
        self._closed = threading.Event()

    @property
    def kind(self) -> TransferEngineKind:
        return TransferEngineKind.TENT

    @property
    def rpc_port(self) -> int:
        return self._rpc_port

    def register_regions(self, regions: Sequence[MemoryRegion]) -> None:
        if not regions:
            return
        self._engine.register_local_memory_batch(
            [region.base_addr for region in regions],
            [region.length for region in regions],
        )

    def unregister_regions(self, regions: Sequence[MemoryRegion]) -> None:
        if not regions:
            return
        self._engine.unregister_local_memory_batch(
            [region.base_addr for region in regions],
            [region.length for region in regions],
        )

    def transfer(
        self, peer: str, op: TransferOp, ranges: Sequence[RangeRequest]
    ) -> list[TransferEvent]:
        if not ranges:
            return []
        if self._closed.is_set():
            raise RuntimeError("The TENT transfer engine is shutting down.")
        batch_id = self._start_batch(self._segment(peer), op, ranges)
        return self._wait(batch_id)

    def _start_batch(
        self, segment: int, op: TransferOp, ranges: Sequence[RangeRequest]
    ) -> int:
        tasks = _merge_ranges(ranges)
        batch_id = int(self._engine.allocate_transfer_batch(len(tasks)))
        if not batch_id:
            raise RuntimeError(
                f"TENT could not allocate a batch of {len(tasks)} transfers."
            )
        opcode = (
            self._tent.OpCode.WRITE
            if op is TransferOp.WRITE
            else self._tent.OpCode.READ
        )
        requests = [
            self._tent.Request(
                opcode,
                task.local_addr,
                segment,
                task.remote_addr,
                task.length,
            )
            for task in tasks
        ]
        # The batch is registered before the submission so that a shutdown
        # racing with it still sees the tasks it has to wait for.
        with self._batches_lock:
            self._batches[batch_id] = _Submission(
                tasks=tasks,
                results=[None] * len(tasks),
                errors=[None] * len(tasks),
                lock=threading.Lock(),
            )
        submitted = False
        try:
            self._engine.submit_transfer(batch_id, requests)
            submitted = True
        finally:
            if not submitted:
                with self._batches_lock:
                    self._batches.pop(batch_id, None)
                self._engine.free_transfer_batch(batch_id)
        return batch_id

    def _wait(self, batch_id: int) -> list[TransferEvent]:
        """Waits on the aggregated batch status.

        One status object covers the batch, so the per-task list is read only
        once the batch reports a terminal state.
        """
        delay = POLL_INTERVAL_MIN_S
        while True:
            overall = self._engine.get_transfer_status_overall(batch_id)
            if self._event_state(overall.state) is not None:
                events = self._poll(batch_id)
                if events:
                    return events
            if self._closed.is_set():
                raise RuntimeError(
                    "The TENT transfer engine shut down while a transfer was in flight."
                )
            with self._batches_lock:
                known = batch_id in self._batches
            if not known:
                raise RuntimeError(
                    f"TENT dropped batch {batch_id} before its ranges settled."
                )
            time.sleep(delay)
            delay = min(delay * 2, POLL_INTERVAL_MAX_S)

    def _poll(self, batch_id: int) -> list[TransferEvent]:
        """Records what settled and returns the events once the batch is done.

        An empty list means a task is still running, so events always line up
        with the ranges the caller submitted.
        """
        with self._batches_lock:
            submission = self._batches.get(batch_id)
        if submission is None:
            return []
        with submission.lock:
            if all(result is not None for result in submission.results):
                return []
            statuses = self._engine.get_transfer_status_list(batch_id)
            if len(statuses) != len(submission.tasks):
                raise RuntimeError(
                    f"TENT returned {len(statuses)} statuses for "
                    f"{len(submission.tasks)} tasks."
                )
            pending = False
            settled: list[TransferEvent] = []
            for index, task in enumerate(submission.tasks):
                state = submission.results[index]
                if state is None:
                    state = self._event_state(statuses[index].state)
                    if state is None:
                        pending = True
                        continue
                    submission.results[index] = state
                    if state is not TransferState.COMPLETED:
                        submission.errors[index] = (
                            f"TENT reported {statuses[index].state}"
                        )
                event = TransferEvent(state=state, error=submission.errors[index])
                settled.extend([event] * task.count)
            if pending:
                return []
            events = settled
        if not self._finish(batch_id, submission):
            return []
        return events

    def _finish(self, batch_id: int, submission: _Submission) -> bool:
        """Removes a settled batch, freeing it once.

        The submission lock stops a concurrent poll from asking the engine about
        a batch that was already freed.
        """
        with submission.lock, self._batches_lock:
            if self._batches.pop(batch_id, None) is None:
                return False
        self._engine.free_transfer_batch(batch_id)
        return True

    def shutdown(self) -> None:
        """Cancels what is still in flight, then closes the peer segments.

        A batch cannot be freed while its tasks run, and the KV memory behind it
        cannot be released before that.
        """
        self._closed.set()
        with self._batches_lock:
            in_flight = {
                batch_id: [
                    index
                    for index, result in enumerate(submission.results)
                    if result is None
                ]
                for batch_id, submission in self._batches.items()
            }
        for batch_id, outstanding in in_flight.items():
            for index in outstanding:
                if not self._cancel(batch_id, index):
                    break
        try:
            self._drain()
        finally:
            with self._segment_lock:
                for segment in self._segments.values():
                    self._engine.close_segment(segment)
                self._segments.clear()

    def _cancel(self, batch_id: int, index: int) -> bool:
        """Cancels one task.

        Returns False when TENT refuses, leaving the batch to the drain deadline.
        """
        unsupported = getattr(self._tent, "NotImplemented", None)
        try:
            self._engine.cancel_transfer(batch_id, index)
        except Exception as error:
            if unsupported is None or not isinstance(error, unsupported):
                raise
            logger.warning(
                "TENT cannot cancel task %d of batch %d: %s", index, batch_id, error
            )
            return False
        return True

    def _drain(self) -> None:
        deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
        while True:
            with self._batches_lock:
                pending = list(self._batches)
            if not pending:
                return
            for batch_id in pending:
                self._poll(batch_id)
            with self._batches_lock:
                remaining = len(self._batches)
            if remaining == 0:
                return
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"TENT still holds {remaining} batches {SHUTDOWN_TIMEOUT_S}s "
                    "after shutting down, so the KV memory behind them cannot be "
                    "released."
                )
            time.sleep(POLL_INTERVAL_MAX_S)

    def _event_state(self, state: Any) -> TransferState | None:
        status = self._tent.TransferStatusEnum
        if state == status.COMPLETED:
            return TransferState.COMPLETED
        if state == status.CANCELED:
            return TransferState.CANCELED
        if state in (status.FAILED, status.TIMEOUT, status.INVALID):
            return TransferState.FAILED
        return None

    def _segment(self, peer: str) -> int:
        segment = self._segments.get(peer)
        if segment is not None:
            return segment
        with self._segment_lock:
            if self._closed.is_set():
                raise RuntimeError(
                    "The TENT transfer engine shut down while a peer was opened."
                )
            segment = self._segments.get(peer)
            if segment is None:
                segment = int(self._engine.open_segment(peer))
                self._segments[peer] = segment
        return segment


def _merge_ranges(ranges: Sequence[RangeRequest]) -> list[_Task]:
    """Folds contiguous ranges into one task.

    TENT builds one object per task and reports one status per task.
    """
    tasks: list[_Task] = []
    for request in ranges:
        if tasks:
            last = tasks[-1]
            if (
                last.length < MAX_MERGED_LENGTH
                and last.local_addr + last.length == request.local_addr
                and last.remote_addr + last.length == request.remote_addr
            ):
                last.length += request.length
                last.count += 1
                continue
        tasks.append(
            _Task(
                local_addr=request.local_addr,
                remote_addr=request.remote_addr,
                length=request.length,
                count=1,
            )
        )
    return tasks


def _import_tent_module() -> Any:
    # An already imported module is returned as is, so callers that loaded the
    # extension themselves keep working.
    module = sys.modules.get("tent")
    if module is not None:
        return module
    if importlib.util.find_spec("tent") is None:
        raise RuntimeError(
            "transfer_engine='tent' requires the 'tent' Python module, which this "
            "Mooncake build does not ship. Build Mooncake with -DUSE_TENT=ON, make "
            "the resulting 'tent' module importable, or select "
            "transfer_engine='classic'."
        )
    return importlib.import_module("tent")


def _peer_address(hostname: str, port: int) -> str:
    """TENT brackets IPv6 addresses when it builds a segment name."""
    if ":" in hostname:
        return f"[{hostname}]:{port}"
    return f"{hostname}:{port}"


def _write_config_file(config: Mapping[str, Any]) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w", prefix="vllm-tent-", suffix=".json", delete=False
    ) as handle:
        json.dump(config, handle)
        return handle.name
