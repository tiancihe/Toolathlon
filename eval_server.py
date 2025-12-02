#!/usr/bin/env python3
"""
Toolathlon Remote Evaluation Server

This server allows remote clients to submit evaluation tasks.
Only one task can run at a time, with IP rate limiting (3 tasks per 24 hours).
"""

import asyncio
import os
import sys
import time
import uuid
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
from collections import defaultdict

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import uvicorn

def log(msg):
    """Log with timestamp (local time + UTC)"""
    local_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    utc_time = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
    print(f"[{local_time}][UTC {utc_time}] {msg}", flush=True)

app = FastAPI(title="Toolathlon Eval Server")

# ===== Global State =====
current_job: Optional[Dict[str, Any]] = None
ip_submission_history: Dict[str, list] = defaultdict(list)
ws_proxy_process = None  # Global WebSocket proxy process
ws_proxy_log_file = None  # Global log file handle

# ===== Configuration =====
TIMEOUT_SECONDS = 240 * 60  # 240 minutes
MAX_SUBMISSIONS_PER_IP = 3
RATE_LIMIT_HOURS = 24
DUMPS_DIR = "./dumps_public_service"
SERVER_PORT = 8080  # Will be updated in main
WS_PROXY_PORT = 8081  # Will be updated in main

# ===== Request/Response Models =====

class SubmitEvaluationRequest(BaseModel):
    mode: str  # "public" or "private"
    base_url: str
    api_key: Optional[str] = None
    model_name: str
    workers: int = 10
    custom_job_id: Optional[str] = None  # Allow custom job_id

class SubmitEvaluationResponse(BaseModel):
    status: str
    job_id: str
    client_id: Optional[str] = None
    message: str
    warning: Optional[str] = None  # Warning if job_id already exists

# ===== Helper Functions =====

def check_job_id_exists(job_id: str) -> bool:
    """Check if job_id already exists in dumps directory or is currently running"""
    # Check if currently running
    if current_job and current_job.get("job_id") == job_id:
        return True

    # Check if directory exists in dumps
    job_dir = Path(DUMPS_DIR) / job_id
    return job_dir.exists()

def check_ip_rate_limit(ip: str) -> tuple[bool, str]:
    """Check if IP has exceeded rate limit"""
    now = datetime.now()
    cutoff = now - timedelta(hours=RATE_LIMIT_HOURS)

    # Clean old records
    ip_submission_history[ip] = [
        ts for ts in ip_submission_history[ip]
        if ts > cutoff
    ]

    count = len(ip_submission_history[ip])
    if count >= MAX_SUBMISSIONS_PER_IP:
        oldest = ip_submission_history[ip][0]
        retry_after = oldest + timedelta(hours=RATE_LIMIT_HOURS)
        return False, f"Rate limit exceeded: {MAX_SUBMISSIONS_PER_IP} tasks per {RATE_LIMIT_HOURS} hours. Retry after {retry_after.isoformat()}"

    return True, ""

async def run_command_async(cmd: list, env: dict, log_file: str):
    """Run command asynchronously and capture output to log file"""
    with open(log_file, 'a') as log_f:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=log_f,
            stderr=asyncio.subprocess.STDOUT,
            env=env
        )
        return process

# ===== Background Task Executor =====

async def execute_evaluation(job_id: str, mode: str, config: Dict[str, Any]):
    """Execute evaluation task in background"""
    global current_job

    start_time = time.time()
    job_dir = Path(DUMPS_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    log_file = str(job_dir / "server_stdout.log")
    ws_proxy_process = None

    try:
        with open(log_file, 'w') as f:
            f.write(f"=== Toolathlon Evaluation Server Log ===\n")
            f.write(f"Job ID: {job_id}\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"Model: {config['model_name']}\n")
            f.write(f"Workers: {config['workers']}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"{'='*50}\n\n")

        # Step 1: Deploy containers
        with open(log_file, 'a') as f:
            f.write("=== Step 1: Deploying local containers ===\n")
            f.flush()

        deploy_process = await run_command_async(
            ["bash", "global_preparation/deploy_containers.sh", "true"],
            env=os.environ.copy(),
            log_file=log_file
        )

        await deploy_process.wait()

        with open(log_file, 'a') as f:
            f.write("\n=== Step 2: Running parallel tests ===\n")
            f.flush()

        # Step 2: Run tests
        env = os.environ.copy()

        if mode == "public":
            env["TOOLATHLON_OPENAI_BASE_URL"] = config["base_url"]
            env["TOOLATHLON_OPENAI_API_KEY"] = config.get("api_key", "")
        else:  # private
            env["TOOLATHLON_OPENAI_BASE_URL"] = f"http://localhost:{WS_PROXY_PORT}/v1"
            env["TOOLATHLON_OPENAI_API_KEY"] = "dummy"

        run_process = await run_command_async(
            [
                "bash", "scripts/run_parallel.sh",
                config["model_name"],
                str(job_dir),
                "unified",
                str(config["workers"])
            ],
            env=env,
            log_file=log_file
        )

        current_job["process"] = run_process

        # Monitor execution with timeout
        while run_process.returncode is None:
            elapsed = time.time() - start_time

            if elapsed > TIMEOUT_SECONDS:
                with open(log_file, 'a') as f:
                    f.write(f"\n\n!!! TIMEOUT: Task exceeded {TIMEOUT_SECONDS//60} minutes !!!\n")

                run_process.kill()
                await run_process.wait()

                current_job["status"] = "timeout"
                current_job["error"] = f"Task exceeded {TIMEOUT_SECONDS//60} minutes"
                log(f"[Server] Job {job_id} timed out after {elapsed//60:.1f} minutes")
                return

            await asyncio.sleep(5)

        with open(log_file, 'a') as f:
            f.write("\n=== Step 3: Task completed ===\n")
            f.write(f"Finished: {datetime.now().isoformat()}\n")

        # Read results
        eval_stats_file = job_dir / "eval_stats.json"
        if eval_stats_file.exists():
            with open(eval_stats_file, 'r') as f:
                eval_stats = json.load(f)
            current_job["eval_stats"] = eval_stats
        else:
            current_job["eval_stats"] = {"error": "eval_stats.json not found"}

        # Read traj_log_all.jsonl
        traj_log_file = job_dir / "traj_log_all.jsonl"
        if traj_log_file.exists():
            with open(traj_log_file, 'r') as f:
                current_job["traj_log_all"] = f.read()
        else:
            current_job["traj_log_all"] = None

        current_job["status"] = "completed"
        log(f"[Server] Job {job_id} completed successfully")

    except Exception as e:
        error_msg = str(e)
        with open(log_file, 'a') as f:
            f.write(f"\n\n!!! ERROR: {error_msg} !!!\n")

        current_job["status"] = "failed"
        current_job["error"] = error_msg
        log(f"[Server] Job {job_id} failed: {error_msg}")

    finally:
        # Keep job info for 60 seconds for client to retrieve
        await asyncio.sleep(60)
        if current_job and current_job.get("job_id") == job_id:
            current_job = None

# ===== API Endpoints =====

@app.get("/")
async def root():
    return {
        "service": "Toolathlon Remote Evaluation Server",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/check_server_status")
async def check_server_status():
    """Check if server is busy (public endpoint)"""
    if current_job:
        return {
            "busy": True,
            "job_id": current_job.get("job_id"),
            "mode": current_job.get("mode"),
            "model": current_job.get("model_name"),
            "started_at": current_job.get("started_at")
        }
    else:
        return {
            "busy": False,
            "message": "Server is idle and ready to accept tasks"
        }

@app.post("/submit_evaluation")
async def submit_evaluation(request: Request, data: SubmitEvaluationRequest):
    """Submit an evaluation task"""
    global current_job

    client_ip = request.client.host

    # Check IP rate limit
    allowed, error_msg = check_ip_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail=error_msg)

    # Check if server is busy
    if current_job is not None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Server is busy",
                "message": "Server is currently processing another task. Please try again later.",
                "current_job_started_at": current_job.get("started_at")
            }
        )

    # Validate mode
    if data.mode not in ["public", "private"]:
        raise HTTPException(status_code=400, detail="Mode must be 'public' or 'private'")

    # Generate or use custom job_id
    warning_msg = None
    if data.custom_job_id:
        job_id = data.custom_job_id
        # Check if job_id already exists
        if check_job_id_exists(job_id):
            warning_msg = f"Job ID '{job_id}' already exists in the system."
            log(f"[Server] WARNING: Job ID {job_id} already exists, but accepting request (possible resume)")
    else:
        job_id = f"job_{uuid.uuid4().hex[:12]}"

    client_id = f"client_{uuid.uuid4().hex[:8]}" if data.mode == "private" else None

    # Record IP submission
    ip_submission_history[client_ip].append(datetime.now())

    # Initialize job
    current_job = {
        "job_id": job_id,
        "client_id": client_id,
        "client_ip": client_ip,
        "mode": data.mode,
        "model_name": data.model_name,
        "workers": data.workers,
        "status": "running",
        "started_at": datetime.now().isoformat()
    }

    # Start background task
    config = {
        "base_url": data.base_url,
        "api_key": data.api_key,
        "model_name": data.model_name,
        "workers": data.workers
    }

    asyncio.create_task(execute_evaluation(job_id, data.mode, config))

    log(f"[Server] Accepted job {job_id} from {client_ip} (mode: {data.mode})")

    response = {
        "status": "accepted",
        "job_id": job_id,
        "client_id": client_id,
        "message": "Task accepted and started"
    }

    if warning_msg:
        response["warning"] = warning_msg

    return response

@app.get("/poll_job_status")
async def poll_job_status(job_id: str):
    """Poll job status"""
    if not current_job or current_job.get("job_id") != job_id:
        # Check if job exists in dumps (completed job)
        job_dir = Path(DUMPS_DIR) / job_id
        eval_stats_file = job_dir / "eval_stats.json"

        if eval_stats_file.exists():
            with open(eval_stats_file, 'r') as f:
                eval_stats = json.load(f)

            # Also read traj_log_all.jsonl if exists
            traj_log_file = job_dir / "traj_log_all.jsonl"
            traj_log_all = None
            if traj_log_file.exists():
                with open(traj_log_file, 'r') as f:
                    traj_log_all = f.read()

            return {
                "status": "completed",
                "eval_stats": eval_stats,
                "traj_log_all": traj_log_all
            }
        else:
            return {
                "status": "cancelled",
                "error": "Job not found or has been cleaned up"
            }

    status = current_job.get("status", "running")
    response = {"status": status}

    if status == "completed":
        response["eval_stats"] = current_job.get("eval_stats", {})
        response["traj_log_all"] = current_job.get("traj_log_all")
    elif status in ["failed", "timeout"]:
        response["error"] = current_job.get("error", "Unknown error")

    return response

@app.get("/get_server_log")
async def get_server_log(job_id: str, offset: int = 0):
    """Get server execution log with incremental reading"""
    log_file = Path(DUMPS_DIR) / job_id / "server_stdout.log"

    if not log_file.exists():
        return {
            "error": "Log file not found",
            "content": "",
            "offset": 0,
            "size": 0,
            "complete": False
        }

    try:
        with open(log_file, 'r') as f:
            f.seek(offset)
            new_content = f.read()
            new_offset = f.tell()

        file_size = log_file.stat().st_size

        # Check if job is complete
        job_complete = False
        if current_job and current_job.get("job_id") == job_id:
            if current_job.get("status") in ["completed", "failed", "timeout", "cancelled"]:
                job_complete = True
        else:
            job_complete = True

        return {
            "content": new_content,
            "offset": new_offset,
            "size": file_size,
            "complete": job_complete
        }
    except Exception as e:
        return {
            "error": str(e),
            "content": "",
            "offset": offset,
            "size": 0,
            "complete": False
        }

@app.post("/cancel_job")
async def cancel_job(job_id: str):
    """Cancel a running job"""
    global current_job

    if not current_job or current_job.get("job_id") != job_id:
        raise HTTPException(status_code=404, detail="Job not found or not running")

    # Kill process if exists
    if "process" in current_job:
        process = current_job["process"]
        try:
            # Kill the main bash process
            process.kill()
            await process.wait()

            # Kill all related Python processes by name
            import subprocess
            try:
                # Kill all run_parallel.py processes
                subprocess.run(['pkill', '-9', '-f', 'run_parallel.py'], check=False)
                # Kill all run_single_containerized.sh processes
                subprocess.run(['pkill', '-9', '-f', 'run_single_containerized.sh'], check=False)
            except:
                pass
        except:
            pass

    # Stop and remove all toolathlon docker containers
    import subprocess
    try:
        log(f"[Server] Cleaning up toolathlon docker containers...")
        # Get all containers with name prefix "toolathlon"
        result = subprocess.run(
            ['docker', 'ps', '-a', '--filter', 'name=toolathlon', '--format', '{{.ID}}'],
            capture_output=True,
            text=True,
            check=False
        )
        container_ids = result.stdout.strip().split('\n')
        container_ids = [cid for cid in container_ids if cid]

        if container_ids:
            # Force remove containers immediately
            subprocess.run(['docker', 'rm', '-f'] + container_ids, check=False, timeout=10)
            log(f"[Server] Force removed {len(container_ids)} toolathlon containers")
    except Exception as e:
        log(f"[Server] Warning: Failed to clean up docker containers: {e}")

    current_job["status"] = "cancelled"

    # Clean up after a short delay
    await asyncio.sleep(10)
    if current_job and current_job.get("job_id") == job_id:
        current_job = None

    log(f"[Server] Job {job_id} cancelled")

    return {"status": "cancelled", "job_id": job_id}

# ===== Main =====

def cleanup_on_shutdown():
    """Cleanup running jobs and docker containers on server shutdown"""
    global current_job, ws_proxy_process, ws_proxy_log_file

    print("\n" + "="*60)
    print("Server shutdown requested (Ctrl+C)")
    print("="*60)

    # Clean up WebSocket proxy
    if ws_proxy_process:
        try:
            print(f"Stopping WebSocket proxy (PID: {ws_proxy_process.pid})...")
            ws_proxy_process.kill()
            ws_proxy_process.wait()
            print("  ✓ WebSocket proxy killed")
        except Exception as e:
            print(f"  ✗ Failed to kill WebSocket proxy: {e}")

        if ws_proxy_log_file:
            try:
                ws_proxy_log_file.close()
            except:
                pass

    if current_job:
        job_id = current_job.get("job_id")
        print(f"Cleaning up running job: {job_id}")

        # Kill process
        if "process" in current_job:
            process = current_job["process"]
            try:
                print(f"  - Killing process (PID: {process.pid})...")
                process.kill()
                print(f"  ✓ Process killed")

                # Kill all related Python processes by name
                import subprocess
                try:
                    print(f"  - Killing run_parallel.py processes...")
                    subprocess.run(['pkill', '-9', '-f', 'run_parallel.py'], check=False)
                    print(f"  - Killing run_single_containerized.sh processes...")
                    subprocess.run(['pkill', '-9', '-f', 'run_single_containerized.sh'], check=False)
                    print(f"  ✓ All related processes killed")
                except Exception as e:
                    print(f"  ✗ Failed to kill related processes: {e}")
            except Exception as e:
                print(f"  ✗ Failed to kill process: {e}")
    else:
        print("No running jobs to clean up")

    # Clean up docker containers
    import subprocess
    try:
        print("Cleaning up toolathlon docker containers...")
        result = subprocess.run(
            ['docker', 'ps', '-a', '--filter', 'name=toolathlon', '--format', '{{.ID}}'],
            capture_output=True,
            text=True,
            check=False,
            timeout=10
        )
        container_ids = result.stdout.strip().split('\n')
        container_ids = [cid for cid in container_ids if cid]

        if container_ids:
            print(f"  - Force removing {len(container_ids)} containers...")
            # Use docker rm -f to force remove immediately without graceful shutdown
            subprocess.run(['docker', 'rm', '-f'] + container_ids, check=False, timeout=10)
            print(f"  ✓ Cleaned up {len(container_ids)} containers")
        else:
            print("  - No containers to clean up")
    except Exception as e:
        print(f"  ✗ Warning: Failed to clean up docker containers: {e}")

    print("="*60)
    print("Server shutdown complete")
    print("="*60)

if __name__ == "__main__":
    import sys

    server_port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    ws_proxy_port = int(sys.argv[2]) if len(sys.argv) > 2 else 8081

    # Update global variables
    SERVER_PORT = server_port
    WS_PROXY_PORT = ws_proxy_port

    print(f"""
{'='*60}
Toolathlon Remote Evaluation Server
{'='*60}
Server Port: {server_port}
WebSocket Proxy Port: {ws_proxy_port} (for private mode)
Max tasks per IP: {MAX_SUBMISSIONS_PER_IP} per {RATE_LIMIT_HOURS} hours
Timeout: {TIMEOUT_SECONDS//60} minutes
Output directory: {DUMPS_DIR}
{'='*60}
    """)

    # Start WebSocket proxy server
    print(f"Starting WebSocket proxy server on port {ws_proxy_port}...")
    ws_log_path = Path(DUMPS_DIR) / "ws_proxy.log"
    ws_log_path.parent.mkdir(parents=True, exist_ok=True)

    ws_proxy_log_file = open(str(ws_log_path), 'w', buffering=1)

    import subprocess
    ws_proxy_process = subprocess.Popen(
        [sys.executable, "simple_server_ws.py", str(ws_proxy_port)],
        stdout=ws_proxy_log_file,
        stderr=subprocess.STDOUT
    )

    print(f"✓ WebSocket proxy started (PID: {ws_proxy_process.pid})")
    print(f"  Log: {ws_log_path}")
    print("="*60)

    try:
        uvicorn.run(app, host="0.0.0.0", port=server_port)
    except KeyboardInterrupt:
        cleanup_on_shutdown()
        print("\nExiting...")

