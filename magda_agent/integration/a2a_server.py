import json
from typing import Any, Dict, Optional
from fastapi import FastAPI, Request, Response
from magda_agent.planning.planner import Planner

class A2AServer:
    """
    A JSON-RPC 2.0 Server interface for the A2A (Agent-to-Agent) Protocol.
    Receives and parses A2A task delegation requests and routes them to the Planner.
    """
    def __init__(self, planner: Planner) -> None:
        """Initializes the A2AServer with a planner module."""
        self.planner = planner
        self.app = FastAPI(title="A2A JSON-RPC Server")

        @self.app.post("/rpc")
        async def handle_rpc(request: Request) -> Response:
            """Endpoint that delegates processing to handle_request."""
            return await self.handle_request(request)

    async def handle_request(self, request: Request) -> Response:
        """Handles the JSON-RPC request."""
        try:
            data = await request.json()
        except Exception:
            return Response(content=json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": "Parse error"},
                "id": None
            }), media_type="application/json", status_code=400)

        if not isinstance(data, dict):
            return Response(content=json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32600, "message": "Invalid Request"},
                "id": None
            }), media_type="application/json", status_code=400)

        is_notification = "id" not in data
        req_id = data.get("id")
        method = data.get("method")
        params = data.get("params", {})

        if data.get("jsonrpc") != "2.0" or not method:
            # If it doesn't have jsonrpc="2.0", it's an invalid request, not a notification
            return Response(content=json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32600, "message": "Invalid Request"},
                "id": req_id
            }), media_type="application/json", status_code=400)

        if method == "delegate_task":
            if not isinstance(params, dict):
                if is_notification:
                    return Response(status_code=204)
                return Response(content=json.dumps({
                    "jsonrpc": "2.0",
                    "error": {"code": -32602, "message": "Invalid params"},
                    "id": req_id
                }), media_type="application/json", status_code=400)

            task = params.get("task")
            if not task:
                if is_notification:
                    return Response(status_code=204)
                return Response(content=json.dumps({
                    "jsonrpc": "2.0",
                    "error": {"code": -32602, "message": "Invalid params"},
                    "id": req_id
                }), media_type="application/json", status_code=400)

            try:
                # Route to the Planner. user_id is arbitrarily passed as "A2A_User"
                await self.planner.generate_plan(user_input=task, user_id="A2A_User")
                if is_notification:
                    return Response(status_code=204)
                return Response(content=json.dumps({
                    "jsonrpc": "2.0",
                    "result": {"status": "accepted", "task": task},
                    "id": req_id
                }), media_type="application/json")
            except Exception as e:
                if is_notification:
                    return Response(status_code=204)
                return Response(content=json.dumps({
                    "jsonrpc": "2.0",
                    "error": {"code": -32000, "message": str(e)},
                    "id": req_id
                }), media_type="application/json", status_code=500)
        else:
            if is_notification:
                return Response(status_code=204)
            return Response(content=json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": "Method not found"},
                "id": req_id
            }), media_type="application/json", status_code=404)
