# HF Endpoint MCP

A single Python package with a stdio MCP server and a systemd user timer for a Hugging Face Inference Endpoint. `start` is **explicit and billable**; the watcher never starts a GPU. `status` and `report` are non-waking management reads. No automatic recovery, proxy, or prompt replay.

## Installation

Python 3.11+; install into a dedicated virtualenv with `pip install .`. Set `HF_MCP_CONFIG` to an operator-owned JSON file and supply `HF_TOKEN` securely to both the MCP subprocess and the systemd user service. Never commit the token, live endpoint details, machine paths, or agent identity. Example operator JSON (replace every placeholder):

```json
{
  "namespace": "YOUR_NAMESPACE",
  "endpoint": "YOUR_EXISTING_ENDPOINT",
  "timer": "hf-endpoint-watch.timer",
  "lock_file": "/PATH/TO/PRIVATE-RUNTIME/endpoint.lock",
  "switch_command": ["/PATH/TO/VERIFIED-LOCAL-MODEL-SWITCH-ADAPTER"],
  "price_per_hour_usd": 1.0
}
```

The external adapter receives one argument, `hf` or `sol`. It must set the target agent's complete provider/model/API configuration, read it back, and exit nonzero if verification fails. Keep the agent on its fallback until the endpoint passes `/health`, authenticated `/v1/models`, and authenticated chat inference. Configure native provider fallback to the existing model separately; never rely on fallback to wake the GPU. The adapter and its agent-specific settings are private operator material, not part of this public repo.

Install `systemd/hf-endpoint-watch.service` and `.timer` under your user unit directory. Adapt the service's example `EnvironmentFile` and `ExecStart` to your private installation. The environment file should be mode 0600, owned by the service user, containing `HF_MCP_CONFIG` and `HF_TOKEN` (or use your local credential mechanism); do not expose it to other users. Run `systemctl --user daemon-reload`; verify `systemctl --user is-enabled hf-endpoint-watch.timer` reports disabled. Do not enable the timer at install time.

Configure the client to execute `python -m hf_endpoint_mcp.server` with `HF_MCP_CONFIG` and the credential supplied securely to that process. MCP tools: `status`, `start`, `stop`, `select_model`, `report`. Only call mutating operations after explicit user authorization. `select_model` may trigger a redeployment and charges, and only changes the repository; verify that the new model is compatible with the existing image/hardware and set its actual serving model ID in the private adapter before starting.

## Lifecycle and rollback

`start` first checks one-replica/zero-minimum scaling, enables the 5-minute timer and reads it back, resumes if needed, waits for ready, performs authenticated inference, then invokes the verified model adapter. On failure after enabling the timer, the timer stays active: investigate and pause safely. The watcher performs only a state read while running. At scaled-to-zero/paused it switches the agent to fallback, pauses and confirms the endpoint (to prevent accidental HTTP wake), then disables the timer and reads back disabled. `stop` does the same ordering intentionally. There is no forced session duration cap; endpoint-side billing limits remain the operator's responsibility.

To roll back: ensure the agent adapter selects `sol` and verifies readback; pause and confirm HF state through its management API; only then disable the timer. Remove the MCP client entry and units, reload systemd, and uninstall the package. **Do not disable an active timer until the endpoint is confirmed paused**. If the endpoint or adapter is unavailable, preserve the timer and configuration for recovery. Model changes in an existing conversation may require a new agent session depending on client behavior.

Report is an hourly metric-bin estimate based on `replicasRunning`, not an HF invoice. Poll windows are one hour to avoid coarse bins. Clock-skew/bin-boundary errors remain possible. No token values or response bodies are logged by the HF HTTP wrapper.

## Offline checks

`python -m unittest discover -s tests -v`; MCP client `list_tools` and read-only tool error path can be exercised without an HF credential. A real `start` and successful inference need explicit permission to incur GPU charges.
