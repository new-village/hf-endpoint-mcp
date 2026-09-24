"""Shared safety-critical orchestration. No automatic GPU resume in reconcile."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

API = "https://api.endpoints.huggingface.cloud/v2"


def config():
    path = os.environ.get("HF_MCP_CONFIG")
    if not path:
        raise RuntimeError("HF_MCP_CONFIG is required")
    cfg = json.loads(Path(path).read_text())
    for key in ("namespace", "endpoint", "switch_command", "timer", "lock_file"):
        if not cfg.get(key):
            raise RuntimeError(f"missing configuration: {key}")
    return cfg


def request(token, method, url, payload=None, timeout=30):
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                                 headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HF API HTTP {exc.code}") from None
    except urllib.error.URLError:
        raise RuntimeError("HF API network failure") from None


def endpoint_url(cfg):
    return f"{API}/endpoint/{cfg['namespace']}/{cfg['endpoint']}"


def token():
    value = os.environ.get("HF_TOKEN", "")
    if not value:
        raise RuntimeError("HF_TOKEN must be provided securely in the process environment")
    return value


def state(ep):
    return ep.get("status", {}).get("state")


def snapshot(ep):
    s = ep.get("status", {})
    return {"state": s.get("state"), "ready": s.get("readyReplica"),
            "target": s.get("targetReplica"), "repository": ep.get("model", {}).get("repository")}


def run(argv, *, check=True):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    if check and result.returncode:
        raise RuntimeError(f"command failed: {argv[0]} (exit {result.returncode})")
    return result


def timer(cfg, action):
    unit = cfg["timer"]
    if not unit.endswith(".timer") or "/" in unit:
        raise RuntimeError("invalid timer unit")
    if action == "enable":
        run(["systemctl", "--user", "enable", "--now", unit])
        if run(["systemctl", "--user", "is-enabled", unit]).stdout.strip() != "enabled":
            raise RuntimeError("timer is not enabled")
        if run(["systemctl", "--user", "is-active", unit]).stdout.strip() != "active":
            raise RuntimeError("timer is not active")
    else:
        run(["systemctl", "--user", "disable", "--now", unit])
        if run(["systemctl", "--user", "is-enabled", unit], check=False).stdout.strip() != "disabled":
            raise RuntimeError("timer is not disabled")
        if run(["systemctl", "--user", "is-active", unit], check=False).stdout.strip() != "inactive":
            raise RuntimeError("timer is still active")


def switch(cfg, mode):
    # Operator-supplied adapter MUST set and then read back the effective model.
    # It must exit nonzero on verification failure. Never execute through a shell.
    command = cfg["switch_command"]
    if not isinstance(command, list) or not command or any(not isinstance(x, str) for x in command):
        raise RuntimeError("switch_command must be an argv array")
    run([*command, mode])


def get(cfg, tok):
    return request(tok, "GET", endpoint_url(cfg))


@contextmanager
def locked(cfg):
    path = Path(cfg["lock_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as fp:
        fcntl.flock(fp, fcntl.LOCK_EX)
        yield


def status(cfg):
    return snapshot(get(cfg, token()))


def smoke(cfg, tok, ep):
    url = ep.get("status", {}).get("url")
    if not url or not url.startswith("https://"):
        raise RuntimeError("endpoint has no HTTPS serving URL")
    request(tok, "GET", url + "/health", timeout=60)
    models = request(tok, "GET", url + "/v1/models", timeout=60)
    ids = [x.get("id") for x in models.get("data", []) if x.get("id")]
    if not ids:
        raise RuntimeError("no serving model")
    reply = request(tok, "POST", url + "/v1/chat/completions", {
        "model": ids[0], "messages": [{"role": "user", "content": "Reply briefly with OK"}],
        "max_tokens": 64, "stream": False}, timeout=120)
    choices = reply.get("choices", [])
    if not choices or not choices[0].get("message", {}).get("content", "").strip():
        raise RuntimeError("authenticated inference yielded no response")
    return {"model": ids[0], "inference_verified": True}


def start(cfg, timeout=2700):
    with locked(cfg):
        tok = token()
        ep = get(cfg, tok)
        compute = ep.get("compute", {})
        scaling = compute.get("scaling", {})
        if (scaling.get("maxReplica") != 1 or scaling.get("minReplica", 0) != 0
                or scaling.get("scaleToZeroTimeout") != 15):
            raise RuntimeError("endpoint must have minReplica=0, maxReplica=1, 15-minute scale-to-zero")
        if state(ep) == "failed":
            raise RuntimeError("endpoint is failed; investigate before separately authorized recovery")
        timer(cfg, "enable")  # fail closed: no unmonitored GPU resume
        if state(ep) != "running":
            request(tok, "POST", endpoint_url(cfg) + "/resume")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ep = get(cfg, tok)
            if state(ep) == "running" and ep.get("status", {}).get("readyReplica", 0) == 1:
                result = smoke(cfg, tok, ep)
                switch(cfg, "hf")
                return {"state": "running", **result, "timer": "enabled"}
            if state(ep) in {"failed", "paused"}:
                raise RuntimeError(f"endpoint entered {state(ep)}; timer remains enabled for recovery")
            time.sleep(15)
        raise RuntimeError("ready timeout; timer remains enabled for recovery")


def shutdown(cfg, *, force=False):
    with locked(cfg):
        tok = token()
        switch(cfg, "sol")  # fallback FIRST; do not access serving URL after scale-to-zero
        ep = get(cfg, tok)
        if force and state(ep) not in {"paused", "scaledToZero"}:
            request(tok, "POST", endpoint_url(cfg) + "/pause")
        elif state(ep) == "scaledToZero":
            request(tok, "POST", endpoint_url(cfg) + "/pause")
        elif state(ep) != "paused":
            return {"state": state(ep), "timer": "enabled"}
        ep = get(cfg, tok)
        if state(ep) != "paused":
            raise RuntimeError("pause unconfirmed; timer remains enabled")
        timer(cfg, "disable")
        return {"state": "paused", "timer": "disabled", "model": "sol"}


def reconcile(cfg):
    ep = get(cfg, token())  # no HF access when timer disabled (called only from timer)
    if state(ep) in {"scaledToZero", "paused"}:
        return shutdown(cfg)
    return {"state": state(ep), "timer": "enabled"}


def select_model(cfg, repository):
    if not repository or "/" not in repository or any(c.isspace() for c in repository):
        raise ValueError("expected owner/model repository")
    with locked(cfg):
        tok = token()
        switch(cfg, "sol")
        ep = get(cfg, tok)
        if state(ep) not in {"paused", "scaledToZero"}:
            request(tok, "POST", endpoint_url(cfg) + "/pause")
            if state(get(cfg, tok)) != "paused":
                raise RuntimeError("pause unconfirmed; model unchanged")
        # An update can redeploy automatically; arm monitoring before the mutation.
        timer(cfg, "enable")
        request(tok, "PUT", endpoint_url(cfg), {"model": {"repository": repository}})
        ep = get(cfg, tok)
        if ep.get("model", {}).get("repository") != repository:
            raise RuntimeError("repository update unconfirmed")
        return {"repository": repository, "state": state(ep),
                "note": "model is on fallback; update may redeploy and incur charges; monitor stays enabled"}


def report(cfg, hours=24):
    if not 1 <= hours <= 168:
        raise ValueError("hours must be 1..168")
    tok = token()
    now = datetime.now(timezone.utc)
    total = 0.0
    rows = []
    for n in range(hours):
        end = now - timedelta(hours=n)
        begin = end - timedelta(hours=1)
        body = request(tok, "POST", endpoint_url(cfg) + "/metrics", {
            "start": begin.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "stop": end.strftime("%Y-%m-%dT%H:%M:%SZ")}, timeout=90)
        width = int(body.get("timeWindowSeconds", 0))
        if width <= 0:
            raise RuntimeError("metrics missing bin width")
        bins = [p for s in body.get("replicasRunning", {}).get("series", [])
                if s.get("status") == "running" for p in s.get("data", [])]
        seconds = sum(width * min(1, max(0, float(p.get("y", 0)))) for p in bins)
        total += seconds
        rows.append({"hour_start_utc": begin.strftime("%Y-%m-%dT%H:%M:%SZ"), "running_seconds": seconds})
    rate = float(cfg["price_per_hour_usd"])
    return {"hours": hours, "running_hours_estimate": round(total / 3600, 3),
            "cost_usd_estimate": round(total / 3600 * rate, 3), "hourly": rows,
            "warning": "metric-bin estimate, not an invoice"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["reconcile", "status"])
    args = parser.parse_args()
    try:
        cfg = config()
        print(json.dumps(reconcile(cfg) if args.action == "reconcile" else status(cfg)))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
