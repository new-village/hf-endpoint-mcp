import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    params = StdioServerParameters(command=sys.executable, args=["-m", "hf_endpoint_mcp.server"], env={})
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            expected = {"start", "stop", "status", "select_model", "report"}
            assert names == expected, names
            result = await session.call_tool("status", {})
            assert result.isError or "HF_MCP_CONFIG" in str(result), result
            print("discovery and missing-config fail-safe verified:", ", ".join(sorted(names)))


if __name__ == "__main__":
    asyncio.run(main())
