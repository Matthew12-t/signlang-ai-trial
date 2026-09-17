import asyncio

import pytest

from src.shared.async_utils import _background_tasks, wait_for_inference


@pytest.mark.anyio
async def test_inference_finishes_after_the_caller_times_out() -> None:
    finished = asyncio.Event()

    async def slow_inference() -> str:
        await asyncio.sleep(0.05)
        finished.set()
        return "transcript"

    with pytest.raises(asyncio.TimeoutError):
        await wait_for_inference(slow_inference(), timeout_seconds=0.01)

    assert len(_background_tasks) == 1
    await asyncio.wait_for(finished.wait(), timeout=1)
    await asyncio.sleep(0)
    assert _background_tasks == set()


@pytest.mark.anyio
async def test_inference_finishes_after_the_caller_is_cancelled() -> None:
    finished = asyncio.Event()

    async def slow_inference() -> str:
        await asyncio.sleep(0.05)
        finished.set()
        return "transcript"

    caller = asyncio.create_task(
        wait_for_inference(slow_inference(), timeout_seconds=10)
    )
    await asyncio.sleep(0.01)
    caller.cancel()

    with pytest.raises(asyncio.CancelledError):
        await caller

    assert len(_background_tasks) == 1
    await asyncio.wait_for(finished.wait(), timeout=1)
    await asyncio.sleep(0)
    assert _background_tasks == set()


@pytest.mark.anyio
async def test_completed_inference_is_not_tracked() -> None:
    async def fast_inference() -> str:
        return "transcript"

    assert await wait_for_inference(fast_inference(), timeout_seconds=1) == "transcript"
    assert _background_tasks == set()
