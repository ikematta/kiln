"""Real MCP server for the e2e suite, built on the official `mcp` SDK
(FastMCP) — the reference server implementation of the protocol, so the
gateway's client is proven against genuine ecosystem framing, not a mock.

Stdio transport by default (the gateway spawns this script as a child per
its `[[mcp_server]]` command); `--transport http` serves the
streamable-HTTP transport on `--port` instead (the test spawns it and
points a `url =` config at http://127.0.0.1:PORT/mcp).

`--sleep-weather N` makes get_weather hang N seconds before answering —
the deliberately slow/hung server of the timeout scenarios. `--tag X` is
an inert argv marker so tests can pgrep this exact instance.
"""

from __future__ import annotations

import argparse
import json
import time

from mcp.server.fastmcp import FastMCP


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="stdio", choices=["stdio", "http"])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sleep-weather", type=float, default=0.0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    server = FastMCP("kiln-e2e", host="127.0.0.1", port=args.port)

    # Same name/description/schema as the request-level tool the Phase 7
    # e2e uses, so the pinned models call it just as reliably — but now the
    # definition comes from THIS server and the gateway executes the call.
    @server.tool()
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        if args.sleep_weather:
            time.sleep(args.sleep_weather)
        return json.dumps({"temperature_c": 21, "sky": "clear", "city": city})

    server.run(transport="streamable-http" if args.transport == "http" else "stdio")


if __name__ == "__main__":
    main()
