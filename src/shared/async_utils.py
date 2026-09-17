"""Async helpers shared by model-backed services."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar


ResultT = TypeVar("ResultT")
_background_tasks: set[asyncio.Task[Any]] = set()


def _consume_background_result(task: asyncio.Task[Any]) -> None:
    """Keep timed-out inference tasks alive and consume their final result."""

    _background_tasks.discard(task)
    if task.cancelled():
        return
    task.exception()


async def wait_for_inference(
    coroutine: Coroutine[Any, Any, ResultT], *, timeout_seconds: float
) -> ResultT:
    """Wait for inference without cancelling its worker when the caller gives up.

    Local model calls run in worker threads. Cancelling the coroutine does not stop
    that thread, and releasing its model semaphore early could allow unsafe
    concurrent GPU access. Shielding lets the worker finish while the caller gets
    a timely timeout response.

    A task the event loop only references weakly can be garbage collected while it
    is still pending, which would leak the provider semaphore. Both giving-up paths
    reach the ``finally`` block: the deadline expiring, and the caller itself being
    cancelled when a client disconnects.
    """

    task = asyncio.create_task(coroutine)
    try:
        return await asyncio.wait_for(
            asyncio.shield(task), timeout=timeout_seconds
        )
    finally:
        if not task.done():
            _background_tasks.add(task)
            task.add_done_callback(_consume_background_result)
