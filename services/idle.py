"""Own the model's lifetime across inference requests and idle periods."""

import asyncio
import logging
from contextlib import asynccontextmanager
from time import monotonic


class IdleModel:
    def __init__(self, load, unload, seconds, *, serialize=False):
        if seconds < 0:
            raise ValueError("MODEL_IDLE_SECONDS must be nonnegative")
        self._load = load
        self._unload = unload
        self._seconds = seconds
        self._serialize = serialize
        self._loaded = False
        self._loading = None
        self._lock = asyncio.Lock()
        self._requests = set()
        self._last_finished = 0.0

    @asynccontextmanager
    async def lifespan(self):
        await asyncio.to_thread(self._load)
        self._loaded = True
        self._last_finished = monotonic()
        stop = asyncio.Event()
        watcher = asyncio.create_task(self._watch(stop)) if self._seconds else None
        try:
            yield
        finally:
            stop.set()
            if watcher:
                await watcher
            if self._requests:
                await asyncio.gather(*self._requests, return_exceptions=True)

    async def run(self, infer, *args):
        task = asyncio.create_task(self._run(infer, args))
        self._requests.add(task)
        task.add_done_callback(self._finished)
        # An interrupted HTTP wait must not end the lifetime of native inference.
        return await asyncio.shield(task)

    def _finished(self, task):
        self._requests.discard(task)
        self._last_finished = monotonic()
        if not task.cancelled():
            task.exception()

    async def _run(self, infer, args):
        if not self._seconds:
            return await asyncio.to_thread(infer, *args)
        if not self._loaded:
            if self._loading is None:
                self._loading = asyncio.create_task(self._reload())
            await self._loading
        if self._serialize:
            async with self._lock:
                return await asyncio.to_thread(infer, *args)
        return await asyncio.to_thread(infer, *args)

    async def _reload(self):
        try:
            async with self._lock:
                await asyncio.to_thread(self._load)
                self._loaded = True
        finally:
            self._loading = None

    async def _watch(self, stop):
        log = logging.getLogger("uvicorn.error")
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), min(self._seconds, 1))
                break
            except TimeoutError:
                pass
            if self._requests or not self._loaded:
                continue
            async with self._lock:
                if self._requests or monotonic() - self._last_finished < self._seconds:
                    continue
                self._loaded = False
                try:
                    await asyncio.to_thread(self._unload)
                except Exception:
                    log.exception("Idle model cleanup failed")
                else:
                    log.info("Model unloaded after %g seconds idle", self._seconds)
