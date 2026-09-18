import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Pin anyio tests to asyncio.

    The services use asyncio primitives directly (``asyncio.Lock``,
    ``asyncio.create_task``, ``asyncio.to_thread``), so the trio backend that
    anyio parametrizes by default cannot run them.
    """

    return "asyncio"
