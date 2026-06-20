"""golden-dataset-studio-mcp: MCP server wrapping golden-dataset-studio."""

__version__ = "0.1.0"

from golden_dataset_mcp.server import main, mcp

__all__ = ["main", "mcp", "__version__"]
