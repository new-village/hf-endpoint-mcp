"""Stdio MCP interface. Mutation tools are never invoked automatically."""
from mcp.server.fastmcp import FastMCP

from . import ops

mcp = FastMCP("hf-endpoint")


@mcp.tool()
def status() -> dict:
    """Read endpoint status without waking the GPU."""
    return ops.status(ops.config())


@mcp.tool()
def start() -> dict:
    """Explicitly resume a billable endpoint, verify real inference, and switch the agent model. Requires user authorization."""
    return ops.start(ops.config())


@mcp.tool()
def stop() -> dict:
    """Switch to fallback first, pause endpoint, verify pause, then disable monitoring."""
    return ops.shutdown(ops.config(), force=True)


@mcp.tool()
def select_model(repository: str) -> dict:
    """Explicit billable-risk redeployment of a different Hub model; switches to fallback first. Requires user authorization."""
    return ops.select_model(ops.config(), repository)


@mcp.tool()
def report(hours: int = 24) -> dict:
    """Estimate running hours and USD cost from replica metrics; does not wake GPU."""
    return ops.report(ops.config(), hours)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
