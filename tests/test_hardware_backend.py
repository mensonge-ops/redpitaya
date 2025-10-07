from __future__ import annotations

import pathlib
import sys
from typing import Deque, List

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import time
from collections import deque

import pytest

from version1 import RedPitayaBackend


class FakeSocket:
    def __init__(self, responses: List[bytes]) -> None:
        self._responses: Deque[bytes] = deque(responses)
        self._pending: bytes | None = None
        self.sent: List[str] = []
        self.closed = False
        self.timeout = None

    def settimeout(self, value: float) -> None:  # pragma: no cover - simple setter
        self.timeout = value

    def sendall(self, data: bytes) -> None:
        command = data.decode().strip()
        self.sent.append(command)
        if command.endswith("?"):
            try:
                self._pending = self._responses.popleft()
            except IndexError:  # pragma: no cover - indicates a broken test
                raise AssertionError("No response prepared for query command")

    def recv(self, _: int) -> bytes:
        if self._pending is None:
            return b""
        chunk = self._pending
        self._pending = None
        return chunk

    def close(self) -> None:  # pragma: no cover - trivial
        self.closed = True


@pytest.mark.parametrize("channels", [2, 4])
def test_redpitaya_backend_cycle(monkeypatch, channels: int) -> None:
    fake_socket = FakeSocket([b"RP,MODEL\n", b"0.125\n"])

    def fake_create_connection(address, timeout=None):
        assert address[0] == "127.0.0.1"
        return fake_socket

    monkeypatch.setattr("socket.create_connection", fake_create_connection)
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    backend = RedPitayaBackend("127.0.0.1", n_channels=channels, rate_hz=5_000.0)
    idn = backend.check_connection()
    assert idn == "RP,MODEL"
    backend.configure()

    measurement = backend.read_int1()
    assert measurement == pytest.approx(0.125)

    control = [0.1 for _ in range(channels)]
    out1, out2 = backend.apply_control(control)
    assert out1 == pytest.approx(0.1)
    if channels > 1:
        assert out2 == pytest.approx(0.1)
    else:
        assert out2 == pytest.approx(0.0)

    backend.close()
    assert fake_socket.closed is True
    assert "SOUR1:ENABLE 0" in fake_socket.sent
    assert "SOUR2:ENABLE 0" in fake_socket.sent
    assert "*IDN?" in fake_socket.sent
