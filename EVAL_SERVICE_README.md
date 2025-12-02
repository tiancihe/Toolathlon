# Toolathlon Remote Evaluation Service
Besides configuring Toolathlon evaluation on your own machine, we also provide Toolathlon evaluation as a service on public servers, where we have setup all the required MCP accounts and you don't need to worry about the setup -- you don't even need to install any MCP-related dependencies, evaluation can be ran by just communicating with our public server.

> We have set up a public Toolathlon evaluation service on 47.253.6.47, this is mainly for you to quickly play with our evaluation without any setup. However, due to the potential evaluation conflict from multiple users, we have constrained this public service to be 3 evaluation requests per IP per 24 hours. If you find this public service crowded, you have a few other options to do the evaluation: 
> 1. Setup your own Toolathlon evaluation service on your own machine following the main readme, which would take like 20-30 minutes.
> 2. If you are a major user that will use Toolathlon evaluation a lot, please contact us (jlini@cse.ust.hk / junxianh@cse.ust.hk), we may be able to provide a dedicated evaluation service for you (for free). 
> 3. If you have an API endpoint and just want to test your model, please contact us (jlini@cse.ust.hk / junxianh@cse.ust.hk) and we are happy to help you run evaluation on Toolathlon with your given API endpoint.


## Quick Start
If you want to test **your inhouse locally deployed model that is not publicly accessible**, just simply put `eval_client.py` and `simple_client_ws.py` together under a folder on your own machine (they are already there if you cloned this repo), then install the client-side dependencies:

```bash
pip install httpx typer websockets
```

Then run the following command directly under this folder:

```bash
# Configuration:
# - base-url: your local openai-compatible endpoint
# - api-key: this argument can be ignored if your model endpoint does not need an API key
# - workers: suggested # of parallel workers
# - output-file, log-file, server-log, traj-log: any file paths you prefer
# - server-host, server-port, ws-proxy-port: our public server addresses

python eval_client.py run \
  --mode private \
  --base-url http://localhost:8001/v1 \
  --api-key dummy \
  --model-name your-model-name \
  --workers 10 \
  --output-file ./results/eval_stats.json \
  --log-file ./results/client.log \
  --server-log ./results/server.log \
  --traj-log ./results/traj_log_all.jsonl \
  --server-host 47.253.6.47 \
  --server-port 8080 \
  --ws-proxy-port 8081
```
If the server is idle, your task will be submitted and you will find the results later on under the `./results` directory. Otherwise, please wait for a while and check again later via ``python eval_client.py check --server-host 47.253.6.47 --server-port 8080``.

If you have ready-to-use public API endpoind and API key, please use the public mode as follows which should run a bit faster:

```bash
python eval_client.py run \
  --mode public \
  --base-url your-puclic-endpoint \
  --api-key sk-your-key \
  --model-name your-model-name \
  --workers 10 \
  --output-file ./results/eval_stats.json \
  --log-file ./results/client.log \
  --server-log ./results/server.log \
  --traj-log ./results/traj_log_all.jsonl \
  --server-host 47.253.6.47 \
  --server-port 8080
```
It will return the results exactly the same as in private mode, we won't save your API keys locally. You can find the running log in the input `[server-log]`, and after the eval is done, you can find the results and trajectory logs in the input `[output-file]` and `[traj-log]`.

#### Resume Incomplete Tasks

If your evaluation was interrupted (e.g., network issues, client/server crash), you can resume it by providing the same job ID:

```bash
python eval_client.py run \
  -- ... # all other arguments you entered previously
  --job-id job_abc123def456  # Use the same job ID to resume
```

**Note:**
- You can find your job ID from the initial submission output or from your log files
- The server will output a warning if the job ID already exists, but will accept the request
- Results will be written to the same output directory on server
- This works for both public and private modes

If you meet any trouble, please feel free to contact us (jlini@cse.ust.hk / junxianh@cse.ust.hk), e.g. we may help you testing your model if provided with your public API endpoint and API key or set up a special channel for you.

# Implemention Details

## Architecture

### Public Mode
```
Client → Server → OpenAI/Anthropic/etc. API
```
Client submits task with API credentials. Server runs evaluation using the public API.

### Private Mode
```
Client + Local LLM ←→ WebSocket ←→ Server
```
Server runs evaluation but forwards LLM requests back to client via WebSocket. Your LLM credentials never leave your machine.

**How Private Mode Works:**
1. Server starts `simple_server_ws.py` (WebSocket proxy on port 8081)
2. Client starts `simple_client_ws.py` (connects to proxy)
3. When server needs LLM inference, request flows: Server → WebSocket → Client → Your LLM
4. Response flows back: Your LLM → Client → WebSocket → Server

---

## More Usage Info

#### Check Server Status

```bash
python eval_client.py check \
  --server-host <host> \
  --server-port 8080
```

**Output (idle):**
```
✓ Server is idle and ready to accept tasks
```

**Output (busy):**
```
⏳ Server is currently busy
   Job ID: job_ab*****56
   Mode: public
   Started: 2025-11-28T10:30:45.123456

Please try again later.
```

#### Monitor Progress

```bash
# Watch client log
tail -f ./results/client.log

# Watch server execution log (synced in real-time)
tail -f ./results/server.log
```

#### Check Task Status

```bash
python eval_client.py status \
  --job-id <job_id> \
  --server-host <host> \
  --server-port 8080
```

#### Cancel Running Task

```bash
python eval_client.py cancel <job_id> \
  --server-host <host> \
  --server-port 8080
```

This will:
- Kill the evaluation process
- Stop and remove all Docker containers
- Clean up server resources

---

#### Output Files

When a task completes, you'll have:

1. **eval_stats.json** (`--output-file`)
   - Evaluation statistics and results
   - Contains pass/fail status for all tasks

2. **traj_log_all.jsonl** (`--traj-log`, optional)
   - Complete trajectory logs for all tasks
   - One JSON object per line, each representing one task

3. **client.log** (`--log-file`)
   - Client-side execution log with timestamps
   - Shows task submission, status polling, completion

4. **server.log** (`--server-log`)
   - Server-side execution log (synced in real-time)
   - Shows container deployment, parallel test execution

5. **ws_client.log** (private mode only)
   - WebSocket client log (in same directory as client.log)
   - Shows WebSocket connection status and request handling

---

## Privacy & Security

### Public Mode
- ⚠️ API keys are sent to server, but we do not store them
- 💡 Use HTTPS in production to protect credentials
- Server uses your API key to call LLM directly

### Private Mode
- ✓ LLM URL and API key stay on client
- ✓ Server never sees your credentials
- ✓ Only inference requests/responses transmitted
- Server's `TOOLATHLON_OPENAI_BASE_URL` points to local WebSocket proxy

### Job ID Anonymization
When checking server status, job IDs are anonymized:
- `job_abc123def456` → `job_ab*****56`
- First 6 and last 2 characters shown
- Protects running task privacy

---


<details>
<summary><h2>Server Setup (This is mainly for developers on the server side, you can skip this if you just want to use the service)</h2></summary>

### Prerequisites

Server requires full Toolathlon environment:

```bash
# Install dependencies
bash global_preparation/install_env_minimal.sh true

# Deploy local services (Canvas, email, etc.)
bash global_preparation/deploy_containers.sh true

# Install server dependencies
pip install fastapi uvicorn websockets
```

### Start Server

```bash
python eval_server.py <server_port> <ws_proxy_port>
```

**Default ports:**
- Server: 8080
- WebSocket proxy: 8081

**Example:**
```bash
python eval_server.py 8080 8081
```

Server output:
```
============================================================
Toolathlon Remote Evaluation Server
============================================================
Server Port: 8080
WebSocket Proxy Port: 8081 (for private mode)
Max tasks per IP: 3 per 24 hours
Timeout: 240 minutes
Output directory: ./dumps_public_service
============================================================
✓ WebSocket proxy started (PID: 12345)
  Log: ./dumps_public_service/ws_proxy.log
============================================================
```

</details>