#!/usr/bin/env python3
"""HTTP-level load generator for scripts/bench.sh (SPEC §11.3).

Measures, against a running Kiln gateway over /v1/chat/completions (SSE):
  - single-stream lane: TTFT and decode tok/s, sequential requests
  - batch lane: aggregate tok/s and TTFT under N concurrent streams

Prompts embed the request index so repeated runs measure real prefill,
not radix prefix-cache hits. Greedy (temperature 0) for run-to-run
stability. Token counts come from the server's own usage block
(stream_options.include_usage), never from chunk counting.

Standard library only, deliberately: this must run on a bare macOS
python3, with no venv, both from a checkout and from an installed kiln.
"""

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

# Roughly 60 tokens of neutral English prose; the {seed} keeps prompts
# distinct across requests so prefix caching cannot skip prefill.
PROMPT = (
    "Request {seed}: Write a short, factual paragraph explaining how a kiln "
    "fires clay into ceramic. Cover the drying stage, the gradual temperature "
    "ramp, the chemical changes in the clay body, and why the cooling rate "
    "matters just as much as the peak temperature."
)


def one_stream(base, model, api_key, max_tokens, seed, timeout):
    """One streaming chat completion; returns dict with ttft/decode timing."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(seed=seed)}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers,
    )
    started = time.monotonic()
    first_token = None
    last_token = None
    usage = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("delta", {}).get("content"):
                    now = time.monotonic()
                    if first_token is None:
                        first_token = now
                    last_token = now
    finished = time.monotonic()
    if first_token is None or usage is None:
        raise RuntimeError("stream produced no content or no usage block")
    completion = usage["completion_tokens"]
    decode_seconds = max(last_token - first_token, 1e-9)
    return {
        "ttft_ms": (first_token - started) * 1e3,
        # First token anchors the decode window, so it is excluded from the
        # numerator: N tokens span N-1 decode intervals.
        "decode_tps": (completion - 1) / decode_seconds if completion > 1 else 0.0,
        "completion_tokens": completion,
        "wall_s": finished - started,
    }


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return ordered[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="gateway base URL")
    parser.add_argument("--model", required=True, help="served model id")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    run = lambda seed: one_stream(  # noqa: E731
        args.base, args.model, args.api_key, args.max_tokens, seed, args.timeout
    )

    # Warmup: first request pays worker spin-up/first-eval costs.
    run("warmup")

    single = [run(f"single-{i}") for i in range(args.requests)]

    batch_started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        batch = list(pool.map(lambda i: run(f"batch-{i}"), range(args.concurrency)))
    batch_wall = time.monotonic() - batch_started

    single_tps = statistics.median(r["decode_tps"] for r in single)
    result = {
        "model": args.model,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": {
            "requests": args.requests,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
        },
        "single_stream": {
            "ttft_ms_p50": round(percentile([r["ttft_ms"] for r in single], 0.5), 1),
            "ttft_ms_p95": round(percentile([r["ttft_ms"] for r in single], 0.95), 1),
            "decode_tps_median": round(single_tps, 1),
        },
        "batch": {
            "aggregate_tps": round(
                sum(r["completion_tokens"] for r in batch) / batch_wall, 1
            ),
            "ttft_ms_p50": round(percentile([r["ttft_ms"] for r in batch], 0.5), 1),
            "ttft_ms_p95": round(percentile([r["ttft_ms"] for r in batch], 0.95), 1),
            "wall_s": round(batch_wall, 2),
        },
    }
    result["batch"]["speedup_vs_single"] = (
        round(result["batch"]["aggregate_tps"] / single_tps, 2) if single_tps else None
    )
    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    try:
        main()
    except (urllib.error.URLError, RuntimeError, json.JSONDecodeError) as err:
        print(f"bench.py: {err}", file=sys.stderr)
        sys.exit(1)
