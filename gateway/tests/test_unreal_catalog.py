"""Unreal's routing and advertised capabilities agree with its pinned runner."""
import asyncio
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import app as gw


def test_unreal_routes_only_to_responses_connections():
    assert gw._backend_of_builtin("unreal") == "unreal"
    assert gw._INTEGRATION_WIRING[("openai", "unreal")] == "openai"
    assert gw._integration_serves_backend({"provider": "openai"}, "unreal")
    assert gw._integration_serves_backend({"provider": "custom", "config": {"api_format": "responses"}}, "unreal")
    for fmt in ("openai", "anthropic"):
        assert not gw._integration_serves_backend({"provider": "custom", "config": {"api_format": fmt}}, "unreal")
    for provider in ("anthropic", "google", "typesafe", "bedrock", "openrouter"):
        assert not gw._integration_serves_backend({"provider": provider}, "unreal")


def test_base_advertises_native_tools_and_skills():
    base = gw._BASE_CATALOG["unreal"]
    assert {name for name, _ in base["tools"]} == {"Bash", "ViewImage", "SkillUse"}
    assert base["tool_enforcement"] == "hard" and base["mcp"] is False
    assert gw._base_takes_skills("unreal")
    assert "unreal" not in gw.CHAT_ONLY_BACKENDS


@pytest.mark.parametrize("plugins,servers", [([], [{"name": "x", "url": "https://example.test/mcp"}]),
                                             ([{"enabled": True, "mcpServers": [{"name": "x"}]}], [])])
def test_enabled_mcp_is_rejected_at_configuration(plugins, servers):
    with pytest.raises(gw.HTTPException) as err:
        gw._harness_props(gw.HarnessBody(name="u", base="unreal", plugins=plugins, mcp_servers=servers))
    assert err.value.status_code == 422
    assert "does not support MCP" in str(err.value.detail)


def test_disabled_mcp_and_skill_only_plugins_are_allowed():
    props = gw._harness_props(gw.HarnessBody(name="u", base="unreal", disabled_tools=["Bash"],
        mcp_servers=[{"name": "x", "enabled": False}], plugins=[{"enabled": True, "skills": [{"name": "s"}]}]))
    assert json.loads(props["disabled_tools"]) == ["Bash"]
    with pytest.raises(gw.HTTPException):
        gw._harness_props(gw.HarnessBody(name="u", base="unreal", disabled_tools=["made-up"]))
    with pytest.raises(gw.HTTPException):
        gw._env_clean({"UNREAL_HARNESS_LLM_BASE_URL": "https://wrong.test"})


def test_out_of_box_unreal_receives_builtin_skill_files(monkeypatch):
    monkeypatch.setattr(gw, "_builtin_skills", lambda: {"probe": {
        "default_enabled": True, "files": [{"path": "SKILL.md", "content": "read me"}]}})
    _, skills, _, _, _ = asyncio.run(gw._harness_plugins("unreal", "local", None, hv=None))
    assert [s["name"] for s in skills] == ["probe"]


def test_base_endpoint_exposes_mcp_capability(monkeypatch):
    async def owner(_):
        return "local", "owner"

    async def models(*_):
        return {"gpt-5.4"}

    monkeypatch.setattr(gw, "_pub_org_member", owner)
    monkeypatch.setattr(gw, "_servable_models", models)
    monkeypatch.setattr(gw, "_builtin_skills", lambda: {})
    result = asyncio.run(gw.list_bases(None))
    unreal = next(b for b in result["bases"] if b["id"] == "unreal")
    assert unreal["takesMcp"] is False and unreal["takesSkills"] is True
    assert next(b for b in result["bases"] if b["id"] == "codex")["takesMcp"] is True
