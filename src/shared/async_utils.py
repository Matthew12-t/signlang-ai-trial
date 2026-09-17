"""Async helpers shared by model-backed services."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar


ResultT = TypeVar("ResultT")
_background_tasks: set[asyncio.Task[object]] = set()


def _consume_background_result(task: asyncio.Task[object]) -> None:
    """Keep timed-out inference tasks alive and consume their final result."""

    _background_tasks.discard(task)
    if task.cancelled():
        return
    task.exception()


async def wait_for_inference(
    awaitable: Awaitable[ResultT], *, timeout_seconds: float
) -> ResultT:
    """Wait for inference without cancelling its worker when the client times out.

    Local model calls run in worker threads. Cancelling the coroutine does not stop
    that thread, and releasing its model semaphore early could allow unsafe
    concurrent GPU access. Shielding lets the worker finish while the caller gets
    a timely timeout response.
    """

    task = asyncio.create_task(awaitable)
    try:
        return await asyncio.wait_for(
            asyncio.shield(task), timeout=timeout_seconds
        )
    except TimeoutError:
        background_task = task  # Help type checkers with the invariant Task type.
        _background_tasks.add(background_task)
        background_task.add_done_callback(_consume_background_result)
        raise
