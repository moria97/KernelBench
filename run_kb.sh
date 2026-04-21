#!/usr/bin/env bash
# run_kb.sh - Run all KernelBench tasks in a given level and report aggregate stats.
#
# Usage:
#   ./run_kb.sh <level> [extra_pydra_args...]
#
# Example:
#.  ./run_kb.sh 3 server_type=anthropic model_name=anthropic/aws.claude-opus-4-6 dataset_src=local  gpu_arch=Ampere
#   ./run_kb.sh 1 server_type=openai model_name=openai/glm-5 dataset_src=local gpu_arch=Turing
#
# Extra args are forwarded verbatim to generate_and_eval_single_sample.py (pydra).
# Optional env vars:
#   RUN_DIR=/path/to/logs   # per-problem log directory (default: results/run_kb_<level>_<ts>)
#   PARALLEL=1              # run problems sequentially (default); set >1 for parallel xargs
#
# Provider adapter (optional — enable by exporting MODEL_API_KEY):
#   MODEL_PROVIDER=anthropic|openai       # sets ANTHROPIC_* / OPENAI_* env vars
#   MODEL_API_KEY=<key>                   # key fed to provider env vars
#   MODEL_BASE_URL=<url>                  # custom gateway; e.g. dashscope compatible-mode
#   NUM_RETRIES=30                        # exported as LITELLM_NUM_RETRIES
# When MODEL_API_KEY is set, a sitecustomize.py is auto-injected via PYTHONPATH that
# rewrites dashscope-style /v1/messages -> /v1/chat/completions and converts the
# Anthropic x-api-key header into Authorization: Bearer (for aws.claude-* on dashscope).

set -u -o pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <level> [extra pydra args...]" >&2
    exit 1
fi

LEVEL="$1"
shift || true

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

LEVEL_DIR="$REPO_DIR/KernelBench/level${LEVEL}"
if [[ ! -d "$LEVEL_DIR" ]]; then
    echo "Level dir not found: $LEVEL_DIR" >&2
    exit 1
fi

# Auto-detect problem ids from filenames like "<id>_<name>.py" if not overridden.
if [[ -z "${PROBLEM_IDS:-}" ]]; then
    PROBLEM_IDS="$(
        ls "$LEVEL_DIR" \
            | grep -E '^[0-9]+_.*\.py$' \
            | sed -E 's/^([0-9]+)_.*/\1/' \
            | sort -n \
            | tr '\n' ' '
    )"
fi

IFS=' ' read -r -a PROBLEM_ID_ARR <<< "$PROBLEM_IDS"
NUM_PROBLEMS="${#PROBLEM_ID_ARR[@]}"
if [[ "$NUM_PROBLEMS" -eq 0 ]]; then
    echo "No problems detected in level $LEVEL" >&2
    exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$REPO_DIR/results/run_kb_l${LEVEL}_${TS}}"
mkdir -p "$RUN_DIR"
SUMMARY_CSV="$RUN_DIR/summary.csv"
echo "problem_id,compiled,correctness,runtime_us,ref_runtime_us,speedup,log" > "$SUMMARY_CSV"

# ================= Provider adapter (optional) =================
# Only activates when MODEL_API_KEY is set. Mirrors the logic from run_l1.sh:
#   - Provider env var setup (anthropic / openai)
#   - Non-ASCII API key guard (prevents httpx UnicodeEncodeError)
#   - sitecustomize.py patching litellm's HTTPHandler for dashscope compatible-mode
PROVIDER_ADAPTER_ENABLED=0
if [[ -n "${MODEL_API_KEY:-}" ]]; then
    PROVIDER_ADAPTER_ENABLED=1
    MODEL_PROVIDER="${MODEL_PROVIDER:-anthropic}"
    MODEL_BASE_URL="${MODEL_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
    NUM_RETRIES="${NUM_RETRIES:-30}"

    case "${MODEL_PROVIDER}" in
        anthropic)
            export ANTHROPIC_API_KEY="${MODEL_API_KEY}"
            export ANTHROPIC_AUTH_TOKEN="${MODEL_API_KEY}"
            export ANTHROPIC_API_BASE="${MODEL_BASE_URL}"
            unset OPENAI_API_KEY OPENAI_API_BASE OPENAI_BASE_URL
            ;;
        openai|*)
            export OPENAI_API_KEY="${MODEL_API_KEY}"
            export OPENAI_API_BASE="${MODEL_BASE_URL}"
            ;;
    esac
    export LITELLM_NUM_RETRIES="${NUM_RETRIES}"

    # API key 纯 ASCII 校验 (防止 httpx UnicodeEncodeError)
    python3.12 - <<'PY' || { echo "API key 含非 ASCII 字符,请检查" >&2; exit 1; }
import os, sys
bad = False
for n in ("ANTHROPIC_API_KEY","ANTHROPIC_AUTH_TOKEN","ANTHROPIC_API_BASE",
         "OPENAI_API_KEY","OPENAI_API_BASE"):
    v = os.environ.get(n)
    if not v: continue
    try:
        v.encode("ascii")
    except UnicodeEncodeError as e:
        print(f"[check] {n} NON-ASCII at {e.start}-{e.end}: {v[e.start:e.end]!r}", file=sys.stderr)
        bad = True
sys.exit(1 if bad else 0)
PY

    # litellm URL rewrite patch injected via PYTHONPATH/sitecustomize.py
    PATCH_DIR="$RUN_DIR/patches"
    mkdir -p "$PATCH_DIR"
    cat > "$PATCH_DIR/sitecustomize.py" <<'PY'
import os, sys

def _rewrite_url(url: str) -> str:
    if not isinstance(url, str) or "compatible-mode" not in url:
        return url
    if "/compatible-mode/v1/v1/messages" in url:
        return url.replace("/compatible-mode/v1/v1/messages",
                           "/compatible-mode/v1/chat/completions")
    if url.endswith("/v1/messages"):
        return url[:-len("/v1/messages")] + "/v1/chat/completions"
    return url

def _rewrite_headers(url: str, headers):
    if not isinstance(url, str) or "compatible-mode" not in url:
        return headers
    if headers is None:
        return headers
    try:
        items = dict(headers)
    except Exception:
        return headers
    api_key = items.pop("x-api-key", None) or items.pop("X-Api-Key", None)
    items.pop("anthropic-version", None)
    items.pop("Anthropic-Version", None)
    items.pop("anthropic-beta", None)
    items.pop("Anthropic-Beta", None)
    if api_key and "Authorization" not in items and "authorization" not in items:
        items["Authorization"] = f"Bearer {api_key}"

    # 兜底:strip 掉所有 header value 里的非 ASCII 字符
    cleaned = {}
    for k, v in items.items():
        if isinstance(v, bytes) or not isinstance(v, str):
            cleaned[k] = v
            continue
        try:
            v.encode("ascii")
            cleaned[k] = v
        except UnicodeEncodeError:
            sanitized = v.encode("ascii", "ignore").decode("ascii")
            print(f"[sitecustomize] header {k!r} sanitized {len(v)}->{len(sanitized)} chars",
                  file=sys.stderr)
            cleaned[k] = sanitized
    return cleaned

try:
    from litellm.llms.custom_httpx import http_handler as _hh

    for _cls_name in ("HTTPHandler", "AsyncHTTPHandler"):
        _cls = getattr(_hh, _cls_name, None)
        if _cls is None or not hasattr(_cls, "post"):
            continue
        _orig = _cls.post
        _is_async = _cls_name.startswith("Async")

        if _is_async:
            async def _patched(self, url, *a, __orig=_orig, **kw):
                new_url = _rewrite_url(url)
                if "headers" in kw:
                    kw["headers"] = _rewrite_headers(new_url, kw["headers"])
                return await __orig(self, new_url, *a, **kw)
        else:
            def _patched(self, url, *a, __orig=_orig, **kw):
                new_url = _rewrite_url(url)
                if "headers" in kw:
                    kw["headers"] = _rewrite_headers(new_url, kw["headers"])
                return __orig(self, new_url, *a, **kw)

        setattr(_cls, "post", _patched)
    print("[sitecustomize] litellm dashscope-anthropic URL rewrite installed",
          file=sys.stderr)
except Exception as _e:
    print(f"[sitecustomize] litellm URL patch skipped: {_e}", file=sys.stderr)
PY

    # Prepend PATCH_DIR so Python auto-imports sitecustomize.py on startup
    export PYTHONPATH="${PATCH_DIR}:${PYTHONPATH:-}"
fi

echo "=== run_kb.sh ==="
echo "Level           : $LEVEL"
echo "Num problems    : $NUM_PROBLEMS"
echo "Run dir         : $RUN_DIR"
echo "Extra pydra args: $*"
if [[ "$PROVIDER_ADAPTER_ENABLED" -eq 1 ]]; then
    echo "Provider        : ${MODEL_PROVIDER} (base=${MODEL_BASE_URL}, retries=${NUM_RETRIES})"
    echo "PYTHONPATH      : ${PYTHONPATH}"
fi
echo "================="

PARALLEL="${PARALLEL:-1}"

run_one() {
    local pid="$1"
    shift
    local log="$RUN_DIR/problem_${pid}.log"
    echo "[L${LEVEL} P${pid}] running -> $log"
    # Never abort the whole sweep on a single failure.
    python3.12 scripts/generate_and_eval_single_sample.py \
        level="$LEVEL" problem_id="$pid" "$@" \
        >"$log" 2>&1 || true

    # Extract the "Evaluation result" line (single line pydantic repr).
    local result_line
    result_line="$(
        awk '
            /^Evaluation result for level/ { capture=1; next }
            capture==1 { print; exit }
        ' "$log" 2>/dev/null || true
    )"

    local compiled="False" correctness="False" runtime="-1" ref_runtime="-1" speedup=""
    if [[ -n "$result_line" ]]; then
        compiled="$(printf '%s' "$result_line"  | grep -oE 'compiled=(True|False)'    | head -1 | cut -d= -f2)"
        correctness="$(printf '%s' "$result_line" | grep -oE 'correctness=(True|False)' | head -1 | cut -d= -f2)"
        runtime="$(printf '%s' "$result_line"    | grep -oE '(^| )runtime=-?[0-9.]+'   | head -1 | sed -E 's/.*=//')"
        ref_runtime="$(printf '%s' "$result_line" | grep -oE 'ref_runtime=-?[0-9.]+'   | head -1 | cut -d= -f2)"
    fi
    : "${compiled:=False}" "${correctness:=False}" "${runtime:=-1}" "${ref_runtime:=-1}"

    if [[ "$correctness" == "True" ]]; then
        speedup="$(awk -v r="$runtime" -v rr="$ref_runtime" 'BEGIN{ if (r+0>0 && rr+0>0) printf "%.4f", rr/r; }')"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s\n' \
        "$pid" "$compiled" "$correctness" "$runtime" "$ref_runtime" "${speedup:-}" "$log" \
        >> "$SUMMARY_CSV"
    echo "  -> compiled=$compiled correctness=$correctness runtime=${runtime}us ref=${ref_runtime}us speedup=${speedup:-N/A}"
}

if [[ "$PARALLEL" -gt 1 ]]; then
    # Parallel execution via xargs (still appends to the same CSV; append is line-atomic for small lines).
    export -f run_one
    export RUN_DIR LEVEL SUMMARY_CSV
    printf '%s\n' "${PROBLEM_ID_ARR[@]}" \
        | xargs -n1 -P "$PARALLEL" -I{} bash -c 'run_one "$@"' _ {} "$@"
else
    for pid in "${PROBLEM_ID_ARR[@]}"; do
        run_one "$pid" "$@"
    done
fi

# ------------------ aggregate stats ------------------
echo
echo "=== Aggregate Stats (level ${LEVEL}) ==="
python3.12 - "$SUMMARY_CSV" <<'PY'
import csv, sys, statistics
path = sys.argv[1]
rows = []
with open(path) as f:
    for r in csv.DictReader(f):
        rows.append(r)
n = len(rows)
if n == 0:
    print("No rows in summary; nothing to aggregate.")
    sys.exit(0)

def as_bool(x): return str(x).strip().lower() == "true"
def as_float(x):
    try: return float(x)
    except: return None

compiled  = sum(1 for r in rows if as_bool(r["compiled"]))
correct   = sum(1 for r in rows if as_bool(r["correctness"]))
speedups  = [as_float(r["speedup"]) for r in rows if as_bool(r["correctness"]) and as_float(r["speedup"]) is not None]
sped_gt_1 = sum(1 for s in speedups if s > 1.0)

def pct(x, total): return f"{x}/{total} ({100.0*x/total:.2f}%)"

print(f"Total problems       : {n}")
print(f"Compile success rate : {pct(compiled, n)}")
print(f"Correctness rate     : {pct(correct, n)}")
print(f"Speedup>1 rate       : {pct(sped_gt_1, n)}   (over all problems)")
if speedups:
    print(f"Speedup stats (correct-only, n={len(speedups)}): "
          f"mean={statistics.mean(speedups):.4f} "
          f"median={statistics.median(speedups):.4f} "
          f"max={max(speedups):.4f} "
          f"min={min(speedups):.4f}")
else:
    print("Speedup stats        : no correct runs with valid timing")
PY

echo
echo "Per-problem CSV: $SUMMARY_CSV"
echo "Per-problem logs: $RUN_DIR/problem_<id>.log"
