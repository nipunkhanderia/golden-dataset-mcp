"""
Protocol-level smoke tests against a real fastmcp.Client connected to the
real server — validates the MCP-facing contract (tool discovery, schemas,
connection lifecycle) independent of business logic correctness.
"""

import asyncio

import pytest
from fastmcp import Client

from golden_dataset_mcp.server import mcp

EXPECTED_TOOLS = {
    "init_dataset",
    "add_entry",
    "update_entry",
    "delete_entry",
    "list_entries",
    "commit_version",
    "diff_versions",
    "evaluate_answers",
    "dataset_status",
}


@pytest.fixture
def run():
    def _run(coro):
        return asyncio.run(coro)

    return _run


class TestToolDiscovery:
    def test_all_expected_tools_registered(self, run):
        async def _check():
            async with Client(mcp) as client:
                tools = await client.list_tools()
                return {t.name for t in tools}

        names = run(_check())
        assert EXPECTED_TOOLS.issubset(names), f"Missing: {EXPECTED_TOOLS - names}"

    def test_every_tool_has_description_and_schema(self, run):
        async def _check():
            async with Client(mcp) as client:
                return await client.list_tools()

        tools = run(_check())
        for tool in tools:
            assert tool.description and len(tool.description) > 10
            assert tool.inputSchema.get("type") == "object"


class TestConnectionLifecycle:
    def test_connect_disconnect_reconnect(self, run):
        async def _check():
            async with Client(mcp) as client:
                await client.list_tools()
            async with Client(mcp) as client:
                tools = await client.list_tools()
                return len(tools) > 0

        assert run(_check()) is True


class TestEndToEndViaClient:
    """Exercise a full init -> add -> commit -> evaluate flow through the
    actual MCP protocol layer, not just direct Python calls, to catch any
    serialization issues that direct calls in test_lifecycle.py wouldn't."""

    def test_full_flow_through_mcp_client(self, run, tmp_path):
        dataset_path = str(tmp_path)

        async def _check():
            async with Client(mcp) as client:
                await client.call_tool(
                    "init_dataset",
                    {"input": {"dataset_path": dataset_path, "name": "e2e-test"}},
                )
                await client.call_tool(
                    "add_entry",
                    {
                        "input": {
                            "dataset_path": dataset_path,
                            "question": "What is the capital of France?",
                            "answer": "The capital of France is Paris.",
                        }
                    },
                )
                commit_result = await client.call_tool(
                    "commit_version",
                    {"input": {"dataset_path": dataset_path, "description": "v1"}},
                )
                eval_result = await client.call_tool(
                    "evaluate_answers",
                    {
                        "input": {
                            "dataset_path": dataset_path,
                            "actual_answers": ["The capital of France is Paris."],
                        }
                    },
                )
                return commit_result, eval_result

        commit_result, eval_result = run(_check())
        assert commit_result.data.version == "1.0"
        assert eval_result.data.passed is True
