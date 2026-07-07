import json
import logging
from typing import Any, Dict, Optional, List
from fastapi import FastAPI, Request, Response
from magda_agent.planning.planner import Planner
from magda_agent.integration.a2a_tracing import A2ATracer
from magda_agent.integration.a2a_security import A2ASecurityContext

class A2AServer:
    """
    A JSON-RPC 2.0 Server interface for the A2A (Agent-to-Agent) Protocol.
    Receives and parses A2A task delegation requests and routes them to the Planner.
    """
    def __init__(self, planner: Planner, security_context: Optional[A2ASecurityContext] = None) -> None:
        """Initializes the A2AServer with a planner module and optional security context."""
        self.planner = planner
        self.security_context = security_context
        self.app = FastAPI(title="A2A JSON-RPC Server")

        @self.app.post("/rpc")
        async def handle_rpc(request: Request) -> Response:
            """Endpoint that delegates processing to handle_request."""
            return await self.handle_request(request)

    async def handle_request(self, request: Request) -> Response:
        """Handles the JSON-RPC request."""
        # Extract trace ID from headers
        headers = dict(request.headers)
        trace_id = A2ATracer.extract_from_headers(headers)
        if trace_id:
            A2ATracer.set_trace_id(trace_id)
            logging.info(f"[A2A SERVER] Received request with TraceID: {trace_id}")
        else:
            # Optionally start a new trace if none provided
            trace_id = A2ATracer.get_or_create_trace_id()
            logging.info(f"[A2A SERVER] Starting new TraceID: {trace_id}")

        # Token validation if security context is provided
        if self.security_context:
            auth_header = headers.get("authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return Response(content=json.dumps({
                    "jsonrpc": "2.0",
                    "error": {"code": -32001, "message": "Unauthorized: Missing or invalid token"},
                    "id": None
                }), media_type="application/json", status_code=401)

            token = auth_header.split(" ")[1]
            if not self.security_context.validate_token(token):
                return Response(content=json.dumps({
                    "jsonrpc": "2.0",
                    "error": {"code": -32001, "message": "Unauthorized: Invalid token"},
                    "id": None
                }), media_type="application/json", status_code=401)

            self.security_context.trace_action("rpc_request_received", {"method": request.method, "path": request.url.path})

        try:
            data = await request.json()
            if self.security_context and isinstance(data, dict):
                self.security_context.trace_action("rpc_payload_received", {"rpc_method": data.get("method")})
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

        if method in ["delegate_task", "execute_subplan"]:
            if not isinstance(params, dict):
                if is_notification:
                    return Response(status_code=204)
                return Response(content=json.dumps({
                    "jsonrpc": "2.0",
                    "error": {"code": -32602, "message": "Invalid params"},
                    "id": req_id
                }), media_type="application/json", status_code=400)

            task = params.get("task") or params.get("context")
            if isinstance(task, dict):
                # If context is a step dict, use its description
                task = task.get("description", str(task))

            if not task:
                if is_notification:
                    return Response(status_code=204)
                return Response(content=json.dumps({
                    "jsonrpc": "2.0",
                    "error": {"code": -32602, "message": "Invalid params: task or context required"},
                    "id": req_id
                }), media_type="application/json", status_code=400)

            try:
                # Route to the Planner. user_id is arbitrarily passed as "A2A_User"
                await self.planner.generate_plan(user_input=str(task), user_id="A2A_User")
                if is_notification:
                    return Response(status_code=204)
                return Response(content=json.dumps({
                    "jsonrpc": "2.0",
                    "result": {"status": "accepted", "method": method},
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
