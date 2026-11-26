"""Load a model on demand, run one inference at a time, unload it when idle.

Vendored byte for byte into every service by ai-audio-jobs `deploy/vendored.py sync`.
Edit ai-audio-jobs `shared/idle_model.py`, never a copy: `deploy/vendored.py check`
refuses to start or bundle a service whose copy differs. The protocol is model
checked in `formal/IdleModel.tla`.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from time import monotonic


class IdleModel:
    """Own one model's lifetime across inference requests and idle periods.

    Inference, loading and unloading all run under one asyncio lock, and each
    takes an executor thread only after it holds that lock. So the model holds
    at most one executor thread, and queued requests wait in the event loop.
    """

    def __init__(self, load, unload, seconds):
        """Set the load and unload callables and the idle timeout.

        Args:
            load: Blocking callable that loads the model.
            unload: Blocking callable that frees the model.
            seconds: Idle seconds before unloading; 0 never unloads.

        Raises:
            ValueError: If seconds is negative.
        """
        if seconds < 0:
            raise ValueError("MODEL_IDLE_SECONDS must be nonnegative")
        self._load = load
        self._unload = unload
        self._seconds = seconds
        self._loaded = False
        self._loading = None
        self._lock = asyncio.Lock()
        self._requests = set()
        self._last_finished = 0.0

    @asynccontextmanager
    async def lifespan(self, *, preload):
        """Run the idle watcher for the app's lifetime.

        Args:
            preload: Load the model before the app serves requests.

        Yields:
            None, while the app serves requests.
        """
        if preload:
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
        """Run one inference on the loaded model, loading it first if needed.

        A cancelled caller does not cancel the inference: native code keeps
        running, and the model stays loaded until it returns.

        Args:
            infer: Blocking callable that runs the inference.
            *args: Arguments passed to infer.

        Returns:
            What infer returns.
        """
        task = asyncio.create_task(self._run(infer, args))
        self._requests.add(task)
        task.add_done_callback(self._finished)
        return await asyncio.shield(task)

    def _finished(self, task):
        """Forget a finished request and restart the idle clock.

        Args:
            task: The finished request task.
        """
        self._requests.discard(task)
        self._last_finished = monotonic()
        if not task.cancelled():
            task.exception()

    async def _run(self, infer, args):
        """Load if needed, then run infer under the lock.

        Args:
            infer: Blocking callable that runs the inference.
            args: Arguments passed to infer.

        Returns:
            What infer returns.
        """
        if not self._loaded:
            if self._loading is None:
                self._loading = asyncio.create_task(self._reload())
            await self._loading
        async with self._lock:
            return await asyncio.to_thread(infer, *args)

    async def _reload(self):
        """Load the model under the lock."""
        try:
            async with self._lock:
                await asyncio.to_thread(self._load)
                self._loaded = True
        finally:
            self._loading = None

    async def _watch(self, stop):
        """Unload the model once no request has run for the idle timeout.

        Args:
            stop: Event set when the app shuts down.
        """
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
