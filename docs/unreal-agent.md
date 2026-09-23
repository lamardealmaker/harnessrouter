# Unreal Agent

The `unreal` backend runs [Unreal Agent](https://github.com/unreallabsai/unreal-agent)
v0.1.1 as a native process. Unreal owns the asynchronous agent loop and tool execution;
HarnessRouter owns authentication, session isolation, cancellation, workspace persistence,
artifact collection, and the public API.

## Install and select

Build an image from this branch using the [self-hosting guide](self-hosting-guide.md).
Unreal is included in the default `HR_BACKENDS` list. If you override that list, include
`unreal`. Startup downloads the pinned Linux amd64 or arm64 release, verifies its SHA-256,
and installs the executable and MIT license under the tools volume. The installer is
`docker/install-unreal.sh`; updating the version requires updating both archive digests.

Connect an OpenAI API key in the Console, then select **Unreal Agent** from **Agent harnesses**.
Custom connections must use the **Responses** API format. Chat Completions, Anthropic,
Google, and the built-in OpenRouter connection are not exposed for this backend.
The initial model menu contains `gpt-5.4`; custom Responses connections can supply their
own model IDs. This is an initial integration, with live-provider verification still pending
as described below.

The existing Responses-compatible API selects it with `metadata.harness_id`:

```bash
curl --fail-with-body -sS "$HARNESSROUTER_BASE_URL/v1/responses" \
  -H "Authorization: Bearer ${HARNESSROUTER_API_KEY:?}" \
  -H 'content-type: application/json' \
  -d '{
    "model": "gpt-5.4",
    "input": "Create hello.txt containing hello and tell me where you saved it.",
    "metadata": {"harness_id": "unreal"},
    "stream": true
  }'
```

Use the usual session continuation API for follow-up turns. The internal Unreal session ID
maps to `.harness/unreal/sessions/<id>.session.jsonl` in the task workspace. This state moves
with workspace checkpoints; a missing resume file produces an error instead of silently
starting over. Native logs live under `.harness/unreal/logs`. Both are internal workspace
state, excluded from public artifact listings.

## Capabilities and limits

| Capability | Behavior |
|---|---|
| Tools | Native `Bash`, `ViewImage`, and `SkillUse` |
| Tool policy | Native enforcement of disabled tool names; unknown names rejected |
| Instructions | Current `AGENTS.md` content passed as the system prompt on each turn |
| Skills | Existing skill bundles written to `.harness/skills` |
| MCP | Unsupported in v0.1.1; enabled servers rejected, including plugin servers |
| Streaming | Completed native records; no token-by-token deltas from this release |
| Async tools | Tool results emitted only when the referenced operations finish, paired by call ID |
| Reasoning | Public provider summaries only; opaque replay state stays internal |
| Usage | Responses usage and served model captured by the credential relay |
| Request limit | `max_turns` limits upstream model requests, including retries; default 400 |
| Cancellation | Existing runner process-group termination and child-process sweep |

Disabling `ViewImage` or `SkillUse` disables that named native tool. An enabled `Bash` can
still read files and execute programs; tool switches are not a filesystem sandbox.
Provider credentials stay in the parent relay. The native process receives a per-turn relay
token, and caller environment variables cannot override `UNREAL_*` provider configuration.

## Verification

The runner CI job installs the same checksum-pinned executable used by the container and runs
`runner/tests/test_unreal_backend.py`. Native tests use a local deterministic Responses server
and real Unreal tool execution. They cover first turn, follow-up, model switching, file creation,
restoring a checkpoint archive, request limits, skills, disabled tools, image viewing, duplicate
delivery, usage accounting, and cancellation of shell children. Parser tests cover asynchronous
operation ordering, provider failures, incomplete responses, and reasoning privacy. Gateway
tests verify routing, configuration rejection, and advertised capabilities.

To reproduce the native tests without a container:

```bash
export TOOLS="$(mktemp -d)"
source docker/install-unreal.sh
install_unreal
HR_UNREAL_TEST_BIN="$TOOLS/bin/unreal-agent-runner" \
  python -m pytest runner/tests/test_unreal_backend.py
```

These deterministic tests do **not** establish live model quality, rendered artifact cards,
or end-to-end sandbox recycling. Before merging/releasing, build the container, walk the
documented self-hosted flow, and run the [support matrix](harness-verification.md) for each
offered model on a real OpenAI connection. Also run the custom-harness probe with `BASES=unreal`;
it tests skill execution and tool policy, and records MCP as unsupported. Record the live
results before expanding the model or provider menu.
