#!/usr/bin/env python3
"""
WebSocket Proxy Server (Production)
Container → Here → Client (WebSocket) → Real LLM → Client → Here → Container
"""
import asyncio
import uuid
import json
from datetime import datetime
from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
import uvicorn

def log(msg):
    """Log with timestamp (local time + UTC)"""
    local_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    utc_time = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
    print(f"[{local_time}][UTC {utc_time}] {msg}", flush=True)

app = FastAPI()

# Global variables
connected_client: Optional[WebSocket] = None  # Current connected Client
pending_requests: Dict[str, dict] = {}  # request_id -> request_data
responses: Dict[str, dict] = {}  # request_id -> response_data
cancelled_requests: set = set()  # Cancelled request ID blacklist

# ===== WebSocket Management =====

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection endpoint (only one Client allowed)"""
    global connected_client

    await websocket.accept()

    # Safe get client address (prevent websocket.client from being None)
    try:
        if websocket.client is not None:
            client_addr = f"{websocket.client.host}:{websocket.client.port}"
        else:
            client_addr = "Unknown"
    except Exception:
        client_addr = "Unknown"

    log(f"[Server] Client connected: {client_addr}")

    # Check if there is already a Client connected
    if connected_client is not None:
        log(f"[Server] Reject connection (already a Client): {client_addr}")
        try:
            await websocket.send_json({"type": "error", "message": "Another client is already connected"})
            await websocket.close()
        except Exception:
            pass
        return

    connected_client = websocket

    # Create two tasks
    task_handle_messages = None
    task_push_requests = None

    try:
        # Use create_task instead of gather,这样可以更好地控制取消
        task_handle_messages = asyncio.create_task(handle_client_messages(websocket))
        task_push_requests = asyncio.create_task(push_requests_to_client(websocket))

        # Wait for any task to complete (usually because of connection closed)
        done, pending = await asyncio.wait(
            [task_handle_messages, task_push_requests],
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel incomplete tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                log(f"[Server] Error canceling task: {e}")

        # Check exceptions of completed tasks
        for task in done:
            try:
                task.result()
            except WebSocketDisconnect:
                log(f"[Server] Client disconnected: {client_addr}")
            except Exception as e:
                log(f"[Server] Task exception: {e}")

    except Exception as e:
        log(f"[Server] WebSocket error: {e}")
        import traceback
        log(f"[Server] Stack: {traceback.format_exc()}")
    finally:
        # Ensure cleanup (this will always execute)
        connected_client = None

        # Cancel all possible running tasks
        for task in [task_handle_messages, task_push_requests]:
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except:
                    pass

        log(f"[Server] Client cleanup completed: {client_addr}")

async def handle_client_messages(websocket: WebSocket):
    """Handle messages from Client (responses, heartbeats, etc.)"""
    while True:
        try:
            # Add timeout: must receive message within 90 seconds (heartbeat interval is 30 seconds, allow 2 missed)
            message = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=90.0
            )
            msg_type = message.get("type")

            if msg_type == "response":
                # Client returned response
                request_id = message.get("request_id")
                response_data = message.get("data")

                # Check if in cancelled list (blacklist)
                if request_id in cancelled_requests:
                    log(f"[Server] Ignore response for cancelled request: {request_id}")
                    cancelled_requests.discard(request_id)  # Remove from blacklist
                    continue  # Discard response, not process

                responses[request_id] = response_data
                log(f"[Server] Received client response: {request_id}, status code: {response_data.get('status_code')}")

            elif msg_type == "heartbeat":
                # Heartbeat response, add send timeout protection
                try:
                    await asyncio.wait_for(
                        websocket.send_json({"type": "heartbeat_ack"}),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    log(f"[Server] Send heartbeat response timeout")
                    raise  # Throw exception to trigger cleanup

            else:
                log(f"[Server] Unknown message type: {msg_type}")

        except asyncio.TimeoutError:
            log(f"[Server] Receive message timeout (90 seconds no message), Client may be disconnected")
            raise  # Trigger connection cleanup
        except WebSocketDisconnect:
            log(f"[Server] handle_client_messages: Client主动断开")
            raise  # Propagate upwards
        except Exception as e:
            log(f"[Server] handle_client_messages error: {e}")
            import traceback
            log(f"[Server] Stack: {traceback.format_exc()}")
            raise  # Trigger cleanup

async def push_requests_to_client(websocket: WebSocket):
    """Continuously check queue, push new requests immediately to Client"""
    while True:
        try:
            # Check if there are pending requests to push
            to_push = [rid for rid, req in pending_requests.items() if not req.get("pushed")]

            if to_push:
                # Batch push (maximum 100)
                batch = to_push[:100]
                requests_to_send = []

                for request_id in batch:
                    req_data = pending_requests[request_id]
                    req_data["pushed"] = True
                    # Add push timestamp for diagnosis
                    req_data["_server_push_time"] = datetime.utcnow().isoformat()
                    requests_to_send.append(req_data)

                log(f"[Server] Push {len(requests_to_send)} requests: {batch}")

                # Add send timeout protection (10 seconds)
                try:
                    send_start = datetime.utcnow()
                    await asyncio.wait_for(
                        websocket.send_json({
                            "type": "new_requests",
                            "requests": requests_to_send
                        }),
                        timeout=10.0
                    )
                    send_duration = (datetime.utcnow() - send_start).total_seconds()
                    if send_duration > 0.1:  # If more than 100ms, record warning
                        log(f"[Server] ⚠️  Push duration {send_duration:.3f} seconds (可能被 TCP 流控阻塞)")
                except asyncio.TimeoutError:
                    log(f"[Server] Push request timeout, connection may有问题")
                    raise  # Trigger cleanup

            await asyncio.sleep(0.1)  # Check every 100ms

        except asyncio.TimeoutError:
            log(f"[Server] push_requests_to_client: Send timeout")
            raise  # Trigger cleanup
        except WebSocketDisconnect:
            log(f"[Server] push_requests_to_client: Client disconnected")
            raise
        except Exception as e:
            log(f"[Server] push_requests_to_client error: {e}")
            import traceback
            log(f"[Server] Stack: {traceback.format_exc()}")
            raise

# ===== HTTP API =====

async def _handle_proxy_request(request: Request, request_data: dict, endpoint: str):
    """Common handler for proxy requests (both /chat/completions and /responses)"""
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    log(f"[Server] Received {endpoint} request {request_id}")

    # Check if there is a Client connected
    if connected_client is None:
        log(f"[Server] No available Client")
        return JSONResponse(
            content={
                "error": {
                    "message": "No client connected to proxy server",
                    "type": "service_unavailable",
                    "code": "no_client"
                }
            },
            status_code=503
        )

    # Add to pending queue with endpoint information
    pending_requests[request_id] = {
        "request_id": request_id,
        "pushed": False,
        "_endpoint": endpoint,  # Add endpoint metadata
        **request_data
    }

    # Wait for response (maximum 10 minutes)
    for i in range(1200):  # 1200 * 0.5 = 600 秒
        # Check if the caller is disconnected (check every 10 seconds to reduce overhead)
        if i % 20 == 0:  # 20 * 0.5 = 10 秒
            if await request.is_disconnected():
                log(f"[Server] Caller disconnected {request_id}, stop waiting")
                pending_requests.pop(request_id, None)
                responses.pop(request_id, None)
                cancelled_requests.add(request_id)  # Add to cancelled blacklist
                # Do not return anything, connection is disconnected
                return

        # Check if response is received
        if request_id in responses:
            resp_data = responses.pop(request_id)
            pending_requests.pop(request_id, None)  # Clean up

            log(f"[Server] Got response {request_id}, status code: {resp_data.get('status_code')}")

            return JSONResponse(
                content=resp_data["body"],
                status_code=resp_data["status_code"]
            )

        await asyncio.sleep(0.5)

    # Timeout
    pending_requests.pop(request_id, None)
    log(f"[Server] Request timeout {request_id}")
    return JSONResponse(
        content={
            "error": {
                "message": "Request timed out after 600 seconds (10 minutes)",
                "type": "timeout",
                "code": "timeout"
            }
        },
        status_code=504
    )

@app.post("/v1/chat/completions")
async def proxy_chat(request: Request, request_data: dict):
    """Proxy for OpenAI Chat Completions API"""
    return await _handle_proxy_request(request, request_data, "/chat/completions")

@app.post("/v1/responses")
async def proxy_responses(request: Request, request_data: dict):
    """Proxy for OpenAI Responses API"""
    return await _handle_proxy_request(request, request_data, "/responses")

@app.get("/")
async def root():
    """Health check"""
    return {
        "service": "WebSocket Proxy Server",
        "client_connected": connected_client is not None,
        "pending_requests": len(pending_requests),
        "status": "running"
    }

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

    print("="*50)
    print("WebSocket Proxy Server (Production)")
    print(f"Port: {port}")
    print(f"WebSocket: ws://0.0.0.0:{port}/ws")
    print("="*50)

    uvicorn.run(app, host="0.0.0.0", port=port)
