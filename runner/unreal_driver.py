"""Run pinned Unreal Agent v0.1.1 and translate its persisted JSONL records.

One process per HarnessRouter turn. The parent runner owns timeout/cancellation,
session uid isolation, credential relay, usage accounting, and artifact collection.
Unreal owns the agent loop and asynchronous operation execution.
"""
from __future__ import annotations

import base64
import json
import signal
from pathlib import Path
import re
import subprocess
import sys
import uuid

TERMINAL = {"completed", "failed", "canceled"}
REQUEST_LIMIT = "hr_unreal_request_limit"


def session_file(cwd: str, sid: str) -> Path:
    # Never let an untrusted resume id name a different session's file.
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", sid):
        raise ValueError("Invalid Unreal session ID")
    return Path(cwd) / ".harness" / "unreal" / "sessions" / (sid + ".session.jsonl")


def _emit(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def _assistant(block: dict) -> dict:
    return {"type": "assistant", "message": {"content": [block]}}


class Events:
    """Pair async results by CallID; a submitted operation is not a completed tool."""

    def __init__(self) -> None:
        self.sequence = 0
        self.calls: dict[str, dict] = {}
        self.emitted: set[str] = set()
        self.completed: set[str] = set()
        self.final = ""
        self.error = ""
        self.stop = ""
        self.responses = 0
        self.last_input_id = ""
        self.previous_response: dict = {}

    def restore_calls(self, path: Path) -> None:
        # Only call metadata is restored. Do not re-emit old assistant text or usage.
        # The native store validates the complete history before it executes anything.
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break                 # upstream ignores an uncommitted trailing record
                record = json.loads(line)
                item = (record.get("data") or {}).get("Item") or {}
                if item.get("Kind") == "input" and item["Data"].get("Kind") == "external":
                    self.last_input_id = item["Data"]["ID"]
                if item.get("Kind") == "model_response":
                    self.previous_response = item["Data"]["Response"]
                    for out in item["Data"]["Response"].get("Output", []):
                        if out["Type"] == "tool_call":
                            call = out["Data"]
                            self.calls[call["CallID"]] = call

    def _call(self, call: dict) -> list[dict]:
        cid = call["CallID"]
        if cid in self.emitted:
            return []
        self.emitted.add(cid)
        try:
            args = json.loads(call["Arguments"])
        except (ValueError, TypeError):
            args = {"raw_arguments": call.get("Arguments")}
        if not isinstance(args, dict):
            args = {"raw_arguments": call.get("Arguments")}
        return [_assistant({"type": "tool_use", "id": cid, "name": call["Name"], "input": args})]

    def feed(self, item: dict) -> list[dict]:
        if item.get("type") == "error":
            self.error = str(item.get("message") or "Unreal runner failed")
            return []
        seq = item["Sequence"]
        if not isinstance(seq, int) or seq <= self.sequence:
            raise ValueError("Unreal event sequence must increase")
        self.sequence = seq
        kind, data = item["Kind"], item["Data"]
        if kind in ("input", "turn", "fork"):
            return []
        if kind == "model_response":
            response = data["Response"]
            self.responses += 1
            self.stop = response.get("Stop") or ""
            if response.get("Failure"):
                failure = response["Failure"]
                self.error = str(failure.get("Message") or failure.get("Code") or "Model request failed")
            out = []
            texts = []
            for entry in response.get("Output", []):
                value = entry["Data"]
                if entry["Type"] == "message":
                    text = str(value.get("Text") or "")
                    if text:
                        texts.append(text)
                        out.append(_assistant({"type": "text", "text": text}))
                elif entry["Type"] == "reasoning":
                    # Only the provider's public summary, never its opaque replay state.
                    summary = "\n".join(value.get("Summary") or [])
                    if summary:
                        out.append(_assistant({"type": "thinking", "thinking": summary}))
                elif entry["Type"] == "tool_call":
                    self.calls[value["CallID"]] = value
                    out.extend(self._call(value))
                else:
                    raise ValueError(f"Unsupported Unreal output: {entry['Type']}")
            # Never return an earlier 'working...' message as the final answer.
            self.final = "\n\n".join(texts)
            return out
        if kind == "tool_call_status":
            cid, status = data["CallID"], data["Status"]
            if cid in self.completed:
                return []
            if cid not in self.calls:
                raise ValueError(f"Unreal result references unknown tool call {cid}")
            out = self._call(self.calls[cid])
            error = status.get("Error") or ""
            waiting = status.get("WaitingFor") or []
            ops = {op["ID"]: op for op in data.get("Operations", [])}
            if not error:
                if not waiting or any(oid not in ops for oid in waiting):
                    raise ValueError("Unreal tool status has missing operations")
                if any(ops[oid]["Status"] not in TERMINAL for oid in waiting):
                    return out
            parts = []
            failed = bool(error)
            if error:
                parts.append(error)
            else:
                for oid in waiting:
                    text, bad = operation_result(ops[oid])
                    parts.append(text)
                    failed |= bad
            self.completed.add(cid)
            out.append({"type": "user", "message": {"content": [{
                "type": "tool_result", "tool_use_id": cid, "is_error": failed,
                "content": "\n".join(parts),
            }]}})
            return out
        raise ValueError(f"Unsupported Unreal record: {kind}")

    def finish(self, rc: int) -> dict:
        reason = ""
        if REQUEST_LIMIT in self.error:
            subtype, bad, reason = "error_max_turns", False, "max_steps"
        elif self.error or rc != 0:
            subtype, bad = "error", True
        elif self.stop == "max_output_tokens":
            subtype, bad, reason = "incomplete", False, "max_output_tokens"
        elif self.stop == "refused":
            subtype, bad, reason = "incomplete", False, "refused"
        elif not self.responses or self.emitted - self.completed:
            subtype, bad = "error", True
            self.error = "Unreal exited without a completed response or with unresolved tool calls"
        else:
            subtype, bad = "success", False
        return {"type": "result", "subtype": subtype, "is_error": bad,
                "result": (self.error or f"Unreal exited {rc}") if bad else self.final,
                "reason": reason, "usage": {}}  # the parent stamps relay usage and served model


def operation_result(op: dict) -> tuple[str, bool]:
    state = op.get("State") or {}
    error = str(state.get("TerminalError") or "")
    failed = op["Status"] != "completed" or bool(error)
    if op["Type"] == "shell":
        result = state.get("Result")
        if result is None:
            if not failed:
                raise ValueError("Completed Unreal shell operation has no result")
            return error or "Shell operation " + op["Status"], True
        parts = [str(result.get("Out") or "")]
        if result.get("Err"):
            parts.append("Stderr:\n" + str(result["Err"]))
        if result.get("ExitCode"):
            parts.append(f"Exit code: {result['ExitCode']}")
            failed = True
        if error:
            parts.append(error)
        return "\n".join(p for p in parts if p) or "(no output)", failed
    if op["Type"] == "skill_use":
        if failed:
            return error or "Skill operation " + op["Status"], True
        return base64.b64decode(state.get("Content") or "", validate=True).decode("utf-8", errors="replace"), False
    if op["Type"] == "view_image":
        result = state.get("Result") or {}
        if failed or result.get("Error"):
            return str(result.get("Error") or error or "Image operation " + op["Status"]), True
        if not result.get("Content") or result.get("EncodedMIMEType") not in ("image/png", "image/jpeg"):
            raise ValueError("Completed Unreal image operation has no valid image")
        # The image reaches Unreal's model natively. Avoid copying megabytes of base64 into
        # the UI transcript; the original attachment remains in the workspace file surface.
        return f"Viewed image ({result.get('OriginalWidth', '?')} × {result.get('OriginalHeight', '?')})", False
    raise ValueError(f"Unsupported Unreal operation: {op['Type']}")


def run_turn(job: dict, emit=_emit) -> dict:
    cwd = str(Path(job["cwd"]).resolve())
    sid = job.get("resume_session_id") or str(uuid.uuid4())
    path = session_file(cwd, sid)
    events = Events()
    if job.get("resume_session_id"):
        if not path.is_file():
            raise ValueError("Unreal session history is missing; refusing to silently start a new conversation")
        events.restore_calls(path)
    emit({"type": "system", "subtype": "init", "session_id": sid, "model": job["model"]})
    request = {"session_id": sid, "model": job["model"],
               "messages": [{"role": "user", "content": job["prompt"], "message_id": job["message_id"]}],
               "system_prompt": job["system_prompt"], "disallowed_tools": job.get("tools_disabled") or [],
               "max_attempts": 2}
    cmd = [job["binary"], "-workspace", cwd, "-session-directory", str(path.parent),
           "-log-directory", str(path.parent.parent / "logs")]
    # Inherit the parent's process group and turn marker so cancellation also kills Bash children.
    proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace")
    try:
        proc.stdin.write(json.dumps(request))
        proc.stdin.close()
        for line in proc.stdout:
            if line.strip():
                for ev in events.feed(json.loads(line)):
                    emit(ev)
        rc = proc.wait()
    except Exception:
        # Give the native coordinator a chance to cancel its operations on parse/output
        # failure, too. Normal cancellation is the parent's process-group kill + sweep.
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise
    finally:
        proc.stdout.close()
    if (rc == 0 and not events.responses and events.last_input_id == job["message_id"]
            and events.previous_response and not events.error):
        # The native inbox correctly deduplicated a redelivery after a restart. Return
        # its already-persisted answer rather than rerunning tools or declaring failure.
        prior = events.previous_response
        events.responses = 1
        events.stop = prior.get("Stop") or ""
        if prior.get("Failure"):
            events.error = str(prior["Failure"].get("Message") or "Previous model request failed")
        events.final = "\n\n".join(o["Data"].get("Text", "") for o in prior.get("Output", []) if o["Type"] == "message")
    result = events.finish(rc)
    emit(result)
    return result


def main() -> int:
    try:
        if len(sys.argv) != 2:
            raise ValueError("Expected one JSON job")
        result = run_turn(json.loads(sys.argv[1]))
    except Exception as exc:
        _emit({"type": "result", "subtype": "error", "is_error": True,
               "result": f"{type(exc).__name__}: {exc}", "usage": {}})
        return 1
    return 1 if result.get("is_error") else 0


if __name__ == "__main__":
    sys.exit(main())
