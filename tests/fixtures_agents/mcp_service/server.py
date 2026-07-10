from mcp.server import Server
import mcp

app = Server("my-tools")

@mcp.tool()
def search_docs(query: str) -> str:
    """Search internal documentation."""
    return "results"

@mcp.tool()
def create_ticket(title: str) -> str:
    """Create a support ticket."""
    return "ticket-123"
