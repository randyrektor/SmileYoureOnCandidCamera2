"""Runs videos through a process pool and relays their progress to the UI.

A "job" is a list of videos. A single-video run is just a one-item job, so
the frontend has one code path for "Process One" and "Process All".
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import queue as queue_mod
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ProcessPoolExecutor

from .worker import init_worker, run_video_job

logger = logging.getLogger(__name__)

Broadcast = Callable[[dict], Awaitable[None]]


class JobManager:
    def __init__(self, broadcast: Broadcast, max_workers: int):
        self._broadcast = broadcast
        self.max_workers = max_workers
        self._pool: ProcessPoolExecutor | None = None
        self._manager = None
        self._queue = None
        self._stop_event = None
        self._task: asyncio.Task | None = None
        self._pumping = False
        self.current: dict | None = None

    # ------------------------------------------------------------ properties

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> dict:
        return {
            "running": self.is_running,
            "job": self.current,
            "workers": self.max_workers,
        }

    # -------------------------------------------------------------- lifecycle

    def _ensure_pool(self) -> None:
        if self._pool is not None:
            return
        ctx = multiprocessing.get_context("spawn")
        self._manager = ctx.Manager()
        self._queue = self._manager.Queue()
        self._stop_event = self._manager.Event()
        self._pool = ProcessPoolExecutor(
            max_workers=self.max_workers, mp_context=ctx, initializer=init_worker
        )
        logger.info("Worker pool ready (%d workers)", self.max_workers)

    async def shutdown(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None and not self._task.done():
            self._task.cancel()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
        if self._manager is not None:
            self._manager.shutdown()
        self._pool = self._manager = self._queue = self._stop_event = None

    # ------------------------------------------------------------------- jobs

    async def start(self, specs: list[dict]) -> dict:
        if self.is_running:
            raise RuntimeError("A job is already running")
        self._ensure_pool()
        self._stop_event.clear()
        names = [spec["video_name"] for spec in specs]
        self.current = {"videos": names, "started": time.time()}
        futures = [
            self._pool.submit(run_video_job, spec, self._queue, self._stop_event) for spec in specs
        ]
        self._task = asyncio.get_running_loop().create_task(self._run(futures))
        return self.current

    async def stop(self) -> None:
        if not self.is_running:
            return
        self._stop_event.set()
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=30)
        except TimeoutError:
            logger.warning("Job did not stop within 30s")

    async def _run(self, futures) -> None:
        await self._broadcast({"type": "job_started", **self.current})
        pump = asyncio.create_task(self._pump_queue())
        results: list[dict] = []
        try:
            for fut in asyncio.as_completed([asyncio.wrap_future(f) for f in futures]):
                result = await fut
                results.append(result)
                await self._broadcast({"type": "video_complete", **result})
        finally:
            self._pumping = False
            await pump
            self._drain_queue_sync()
            saved = sum(r.get("saved", 0) for r in results)
            errors = [r for r in results if r.get("error")]
            stopped = any(r.get("stopped") for r in results)
            await self._broadcast(
                {
                    "type": "complete",
                    "saved": saved,
                    "videos": len(results),
                    "errors": errors,
                    "stopped": stopped,
                    "elapsed": round(time.time() - self.current["started"]),
                }
            )
            self.current = None

    async def _pump_queue(self) -> None:
        """Relay worker messages to the websocket without blocking the loop."""
        self._pumping = True
        loop = asyncio.get_running_loop()
        while self._pumping:
            msg = await loop.run_in_executor(None, self._get_message, 0.25)
            if msg is not None:
                await self._broadcast(msg)
        # Flush anything that arrived while the last worker was finishing.
        while (msg := self._get_message(0.0)) is not None:
            await self._broadcast(msg)

    def _get_message(self, timeout: float):
        try:
            return self._queue.get(timeout=timeout)
        except queue_mod.Empty:
            return None

    def _drain_queue_sync(self) -> None:
        while self._get_message(0.0) is not None:
            pass
