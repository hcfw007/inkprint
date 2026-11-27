"""The offline guard itself: real network must still fail loudly.

The guard blocks connect/sendto (not the socket() constructor) so that
asyncio socketpair self-pipes — and thus FastAPI's TestClient — keep
working. These tests pin both halves of that contract.
"""

import socket

import pytest


def test_real_connections_are_blocked():
    with pytest.raises(RuntimeError, match="network access is blocked"):
        socket.create_connection(("127.0.0.1", 1))
    with pytest.raises(RuntimeError, match="network access is blocked"):
        socket.socket().connect(("127.0.0.1", 1))


def test_socketpair_still_works_for_event_loops():
    a, b = socket.socketpair()
    try:
        a.send(b"x")
        assert b.recv(1) == b"x"
    finally:
        a.close()
        b.close()
