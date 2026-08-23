# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Wire-format tests for the session-run reserve_io op.

No board and no c7x_compute: C7xSession is driven against a fake process, so
these run anywhere.

This covers the one seam in the reserve_io path that nothing else can reach
without hardware -- the JSON field names, which are agreed by convention
between smollm_board.py (writer) and cmd_session_run() in
c7x_compute_cli.cpp (reader). A rename on one side alone fails only at the
first prefill on a real board, and looks like a capacity bug rather than a
protocol one, so the expected keys are asserted literally here rather than
rebuilt from the code under test.

The reader side is `json_get_ll(header_buf, "input_bytes", ...)` and
`json_get_ll(header_buf, "output_bytes", ...)`, dispatched on an `op` of
exactly "reserve_io".
"""

import json

import pytest
from smollm_board import C7xSession


class _FakeStdin:
    def __init__(self):
        self.written = b""

    def write(self, data):
        self.written += data

    def flush(self):
        pass


class _FakeStdout:
    """Yields one queued line per readline(); C7xSession reads via select()
    on a real fd normally, so _read_json_line is stubbed out instead."""

    def __init__(self, lines):
        self._lines = list(lines)

    def pop(self):
        return self._lines.pop(0)


class _FakeProc:
    def __init__(self, responses):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(responses)

    def poll(self):
        return None


def _session(monkeypatch, responses):
    """A C7xSession wired to a fake process, bypassing __init__'s subprocess
    launch and readiness handshake."""
    sess = C7xSession.__new__(C7xSession)
    sess._proc = _FakeProc(responses)  # noqa: SLF001
    monkeypatch.setattr(
        "smollm_board._read_json_line",
        lambda stream, _timeout_s: stream.pop(),
    )
    return sess


@pytest.mark.quick
def test_reserve_io_emits_the_keys_the_cli_parses(monkeypatch):
    """Exact header the C++ side parses, for SmolLM prefill's real figures."""
    sess = _session(monkeypatch, ['{"status":"ok"}'])
    sess.reserve_io(11_802_496, 12_583_040)

    written = sess._proc.stdin.written  # noqa: SLF001
    assert written.endswith(b"\n"), "header must be newline-terminated (fgets)"
    assert written.count(b"\n") == 1, "no payload follows a reserve_io header"

    sent = json.loads(written)
    assert sent == {
        "op": "reserve_io",
        "input_bytes": 11_802_496,
        "output_bytes": 12_583_040,
    }


@pytest.mark.quick
def test_reserve_io_does_not_use_the_infer_field_names(monkeypatch):
    """Deliberate: an older c7x_compute reaching the infer branch must fail
    its input_size/shape/dtype parse and reply "Invalid request format"
    WITHOUT trying to read a binary payload that was never sent. Naming a
    field input_size would desync the stream instead."""
    sess = _session(monkeypatch, ['{"status":"ok"}'])
    sess.reserve_io(4096, 8192)

    sent = json.loads(sess._proc.stdin.written)  # noqa: SLF001
    for infer_only in ("input_size", "num_inputs", "shape", "dtype"):
        assert infer_only not in sent


@pytest.mark.quick
def test_reserve_io_raises_on_error_status(monkeypatch):
    """A rejected reservation (-EBUSY after capacity locked, -ENOMEM when the
    carveout genuinely can't provide it) must not be swallowed -- silently
    continuing would surface later as an unexplained inference failure."""
    sess = _session(
        monkeypatch,
        ['{"status":"error","error":"Device or resource busy"}'],
    )
    with pytest.raises(RuntimeError, match="reserve_io failed"):
        sess.reserve_io(4096, 8192)


@pytest.mark.quick
def test_reserve_io_coerces_to_int(monkeypatch):
    """metadata.json round-trips through JSON, so values may arrive as floats;
    json_get_ll() parses with strtoll and would stop at the decimal point."""
    sess = _session(monkeypatch, ['{"status":"ok"}'])
    sess.reserve_io(12_583_040.0, 196_736.0)

    sent = json.loads(sess._proc.stdin.written)  # noqa: SLF001
    assert sent["input_bytes"] == 12_583_040
    assert sent["output_bytes"] == 196_736
    assert isinstance(sent["input_bytes"], int)
    assert isinstance(sent["output_bytes"], int)
