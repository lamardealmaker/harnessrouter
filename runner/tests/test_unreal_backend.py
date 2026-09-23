"""Pinned Unreal contracts plus optional real-binary tests against a local Responses server.

HR_UNREAL_TEST_BIN=/path/to/unreal-agent-runner enables native integration tests.
No provider credentials, network model calls, or paid services are used.
"""
import base64
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server as rs
from unreal_driver import Events, operation_result, session_file


def record(seq, kind, data):
    return {"Sequence": seq, "Kind": kind, "Data": data}


def response(seq, outputs, **kw):
    return record(seq, "model_response", {"Response": {"Output": outputs, "Stop": "complete", **kw}})


def call(cid, name="Bash"):
    return {"Type": "tool_call", "Data": {"CallID": cid, "Name": name, "Arguments": '{"command":"true"}'}}


def status(seq, cid, state="completed", exit_code=0):
    return record(seq, "tool_call_status", {"CallID": cid, "Status": {"WaitingFor": [cid]}, "Operations": [
        {"ID": cid, "Type": "shell", "Status": state,
         "State": {"Result": {"Out": cid, "Err": "", "ExitCode": exit_code}}}]})


def text(value):
    return {"Type": "message", "Data": {"Text": value}}


def test_async_calls_are_paired_once_and_only_at_completion():
    ev = Events()
    assert len(ev.feed(response(1, [call("slow"), call("fast")]))) == 2
    assert ev.feed(status(2, "slow", "awaiting")) == []
    fast = ev.feed(status(3, "fast"))[0]["message"]["content"][0]
    assert fast["tool_use_id"] == "fast" and fast["content"] == "fast"
    ev.feed(response(4, [text("still working")]))
    assert ev.finish(0)["is_error"]  # a model response does not finish outstanding work
    assert ev.feed(status(5, "fast")) == []
    assert ev.feed(status(6, "slow"))[0]["message"]["content"][0]["tool_use_id"] == "slow"


def test_final_text_and_failures_are_not_inferred_from_prose():
    ev = Events()
    ev.feed(response(1, [text("working")]))
    ev.feed(response(2, [text("done")]))
    assert ev.finish(0)["result"] == "done"
    ev.feed({"type": "error", "message": "provider denied request"})
    assert ev.finish(0)["is_error"]
    assert ev.finish(0)["result"] == "provider denied request"
    assert Events().finish(0)["is_error"]


@pytest.mark.parametrize("stop", ["max_output_tokens", "refused"])
def test_incomplete_model_response_is_not_success(stop):
    ev = Events()
    ev.feed(response(1, [text("partial")], Stop=stop))
    result = ev.finish(0)
    assert result["subtype"] == "incomplete" and result["reason"] == stop


def test_error_status_and_nonzero_shell_exit_are_visible():
    ev = Events()
    ev.feed(response(1, [call("x")]))
    result = ev.feed(status(2, "x", exit_code=7))[0]["message"]["content"][0]
    assert result["is_error"] and "Exit code: 7" in result["content"]
    assert operation_result({"Type": "skill_use", "Status": "completed", "State": {
        "Content": base64.b64encode(b"skill instructions").decode()}}) == ("skill instructions", False)


def test_bad_records_fail_closed_and_reasoning_replay_state_is_not_exposed():
    ev = Events()
    output = ev.feed(response(1, [{"Type": "reasoning", "Data": {"Summary": ["public summary"], "Raw": "opaque"}}]))
    assert "opaque" not in json.dumps(output)
    with pytest.raises(ValueError, match="sequence"):
        ev.feed(record(1, "turn", {}))
    with pytest.raises(ValueError, match="unknown tool"):
        ev.feed(status(2, "missing"))


def test_builder_uses_relay_checkpointed_state_and_current_instructions(tmp_path):
    rs._write_agent_doc(str(tmp_path), "unreal", "Follow these instructions", [])
    env = {}
    cmd = rs._build_unreal("openai", rs.Auth(api_key="provider-secret", base_url="https://example.test/v1"),
                           "chosen-model", "hello", str(tmp_path), env, idempotency_key="repeat-me")
    job = json.loads(cmd[-1])
    assert "Follow these instructions" in job["system_prompt"] and "ONLY place" in job["system_prompt"]
    assert "provider-secret" not in json.dumps(job) and "provider-secret" not in json.dumps(env)
    assert env["UNREAL_HARNESS_LLM_API_KEY"].startswith("hr-relay-")
    assert rs._HERMES_RELAY["routes"][env["UNREAL_HARNESS_LLM_API_KEY"]][0] == "https://example.test/v1"
    assert rs._caller_env({"UNREAL_HARNESS_LLM_PROVIDER": "evil"}) == {}
    cmd2 = rs._build_unreal("openai", rs.Auth(api_key="k"), "m", "hello", str(tmp_path), {}, idempotency_key="repeat-me")
    assert json.loads(cmd2[-1])["message_id"] == job["message_id"]


def test_missing_resume_and_unsupported_capabilities_are_rejected(tmp_path):
    args = ("openai", rs.Auth(api_key="k"), "m", "hi", str(tmp_path), {})
    with pytest.raises(rs.HTTPException, match="missing"):
        rs._build_unreal(*args, resume_session_id="lost")
    with pytest.raises(rs.HTTPException, match="MCP"):
        rs._build_unreal(*args, mcp_servers=[{"url": "https://example.test/mcp"}])
    with pytest.raises(rs.HTTPException, match="unknown Unreal tools"):
        rs._build_unreal(*args, tools_disabled=["not-a-tool"])
    with pytest.raises(ValueError):
        session_file(str(tmp_path), "../../other")
    with pytest.raises(rs.HTTPException) as invalid:
        rs._build_unreal(*args, resume_session_id="../../other")
    assert invalid.value.status_code == 400


@pytest.fixture
def native(monkeypatch):
    binary = os.environ.get("HR_UNREAL_TEST_BIN")
    if not binary:
        pytest.skip("Set HR_UNREAL_TEST_BIN to the verified v0.1.1 executable")
    monkeypatch.setenv("HR_UNREAL_BIN", str(Path(binary).resolve()))
    monkeypatch.setattr(rs, "_SESSION_UIDS", False)
    return binary


def message(value):
    return {"id": "msg-" + uuid.uuid4().hex, "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": value}]}


def tool(cid, name, args):
    return {"id": "fc-" + cid, "type": "function_call", "call_id": cid, "name": name,
            "status": "completed", "arguments": json.dumps(args)}


@pytest.fixture
def model():
    class Fake(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.server.requests.append(req)
            self.server.auths.append(self.headers.get("Authorization"))
            response = {"id": "resp-" + uuid.uuid4().hex, "object": "response", "status": "completed",
                        "model": req["model"], "output": self.server.answer(req),
                        "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13,
                                  "input_tokens_details": {"cached_tokens": 2}}}
            body = ("event: response.completed\ndata: " + json.dumps({"type": "response.completed", "response": response}) + "\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    server.requests, server.auths = [], []
    server.answer = lambda req: [message("ok")]
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    server.url = f"http://127.0.0.1:{server.server_port}/v1"
    yield server
    server.shutdown()
    server.server_close()
    worker.join()


def execute(cwd, model, prompt="hello", sid=None, name="model-a", **kwargs):
    env = {**os.environ}
    rs._write_agent_doc(str(cwd), "unreal", "Test instructions", [])
    rs._produced_ack(str(cwd))  # collect setup before measuring the turn's own artifacts
    cmd = rs._build_unreal("openai", rs.Auth(api_key="test-upstream-key", base_url=model.url),
                           name, prompt, str(cwd), env, resume_session_id=sid, **kwargs)
    tid = "turn" + uuid.uuid4().hex
    rs._turns[tid] = {"status": "running", "events": [], "done": False}
    rs._run_turn_bg(tid, cmd, env, str(cwd), rs._unreal_to_claude, name, timeout_seconds=15)
    return rs._turns.pop(tid)


def test_native_five_scenarios_and_usage(native, model, tmp_path):
    # Real native session persistence, real shell execution, real event and relay paths.
    # Model judgment is deliberately deterministic: this is not a live-provider matrix.
    def answer(req):
        calls = [i for i in req["input"] if i.get("type") == "function_call"]
        if not calls:
            return [tool("artifact", "Bash", {"command": "printf proof > proof.txt"})]
        return [message("remembered seed-829" if "seed-829" in json.dumps(req["input"]) else "lost seed")]
    model.answer = answer
    first = execute(tmp_path, model, "seed-829; create proof")
    assert first["status"] == "done", first
    sid = first["session_id"]
    assert (tmp_path / "proof.txt").read_text() == "proof"
    assert session_file(str(tmp_path), sid).is_file()
    assert {f["path"] for f in rs._produced_list(str(tmp_path))} == {"proof.txt"}
    rs._git(str(tmp_path), "add", "-A")
    assert ".harness/unreal/" not in rs._git(str(tmp_path), "ls-files").stdout
    for selected in ("model-a", "model-b", "model-a"):
        follow = execute(tmp_path, model, "recall", sid=sid, name=selected)
        assert follow["status"] == "done" and follow["result"] == "remembered seed-829", follow
        result = next(e for e in follow["events"] if e["type"] == "result")
        assert result["model"] == selected and result["usage"]["output_tokens"] > 0
        assert result["usage"]["input_tokens"] == 8 and result["usage"]["cache_read_tokens"] == 2
    restored = tmp_path.parent / (tmp_path.name + "-restored")
    restored.mkdir()
    archive = subprocess.run(["tar", "-cf", "-", *[f"--exclude={p}" for p in rs.CHECKPOINT_EXCLUDE],
                              "-C", str(tmp_path), "."], capture_output=True, check=True)
    subprocess.run(["tar", "-xf", "-", "-C", str(restored)], input=archive.stdout, check=True)
    recycled = execute(restored, model, "recall after restore", sid=sid)
    assert recycled["status"] == "done" and recycled["result"] == "remembered seed-829"
    assert all(a == "Bearer test-upstream-key" for a in model.auths)
    assert not (tmp_path / "logs").exists()


def test_native_skill_loading_and_hard_disabled_tool(native, model, tmp_path):
    rs._write_skills(str(tmp_path), [{"name": "probe", "files": [
        {"path": "SKILL.md", "content": "---\nname: probe\ndescription: Probe skill\n---\nSecret skill stamp 429."}]}], "unreal")
    def answer(req):
        assert "Bash" not in [t.get("name") for t in req["tools"]]
        if not any(i.get("type") == "function_call" for i in req["input"]):
            return [tool("skill", "SkillUse", {"name": "probe"}), tool("blocked", "Bash", {"command": "touch forbidden"})]
        return [message("skill-loaded")]
    model.answer = answer
    result = execute(tmp_path, model, tools_disabled=["Bash"])
    assert result["status"] == "done", result
    assert not (tmp_path / "forbidden").exists()
    tools = [b for e in result["events"] for b in e.get("message", {}).get("content", []) if b.get("type") == "tool_result"]
    assert any(b["tool_use_id"] == "blocked" and b["is_error"] for b in tools)
    assert any("429" in b["content"] for b in tools)


def test_native_request_limit_stops_before_forwarding(native, model, tmp_path):
    model.answer = lambda req: [tool("call-" + uuid.uuid4().hex, "Bash", {"command": "true"})]
    result = execute(tmp_path, model, max_turns=2)
    assert result["status"] == "max_turns", result
    assert len(model.requests) == 2


def test_native_redelivery_returns_persisted_answer_without_model_or_tools(native, model, tmp_path):
    first = execute(tmp_path, model, idempotency_key="delivery-1")
    assert first["status"] == "done", first
    count = len(model.requests)
    again = execute(tmp_path, model, sid=first["session_id"], idempotency_key="delivery-1")
    assert again["status"] == "done" and again["result"] == first["result"], again
    assert len(model.requests) == count


def test_native_view_image_reaches_model_without_base64_in_ui(native, model, tmp_path):
    # A valid one-pixel PNG; image decoding and the provider payload are native code.
    (tmp_path / "pixel.png").write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="))
    def answer(req):
        if not any(i.get("type") == "function_call" for i in req["input"]):
            return [tool("image", "ViewImage", {"path": "pixel.png"})]
        assert "data:image/" in json.dumps(req["input"])
        return [message("image received")]
    model.answer = answer
    result = execute(tmp_path, model)
    assert result["status"] == "done", result
    transcript = json.dumps(result["events"])
    assert "Viewed image" in transcript and "data:image/" not in transcript


def test_native_cancellation_stops_shell_children(native, model, tmp_path):
    model.answer = lambda req: [tool("long", "Bash", {"command": "echo $$ > child.pid; sleep 60"})] if not any(
        i.get("type") == "function_call" for i in req["input"]) else [message("waiting")]
    env = {**os.environ}
    cmd = rs._build_unreal("openai", rs.Auth(api_key="k", base_url=model.url), "model-a", "wait", str(tmp_path), env)
    tid = "turn" + uuid.uuid4().hex
    rec = rs._turns[tid] = {"status": "running", "events": [], "done": False}
    thread = threading.Thread(target=rs._run_turn_bg, args=(tid, cmd, env, str(tmp_path), rs._unreal_to_claude, "model-a", 15))
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "child.pid").exists() and not rec["done"] and time.monotonic() < deadline:
            threading.Event().wait(0.05)
        assert (tmp_path / "child.pid").exists(), rec
        pid = int((tmp_path / "child.pid").read_text())
        rec["cancelled"] = True
        rs._kill_proc_tree(rec["proc"], tid)
        thread.join(5)
        assert not thread.is_alive() and rec["status"] == "cancelled"
        stat = Path(f"/proc/{pid}/stat")
        try:
            child_state = stat.read_text().split()[2]
        except (FileNotFoundError, ProcessLookupError):
            child_state = None  # child may disappear between lookup and read
        assert child_state in (None, "Z")
    finally:
        if rec.get("proc"):
            rs._kill_proc_tree(rec["proc"], tid)
        thread.join(5)
        rs._turns.pop(tid, None)
