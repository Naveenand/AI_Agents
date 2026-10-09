"""
Save points for the ETL agent.

1. Every successful, verified transform is recorded as a DRAFT run (#1, #2, #3 ...).
2. Nothing becomes a pipeline until the user says so ("save this", "save #2", "save the grass one").
3. A saved pipeline is a normal Python script that re-runs WITHOUT the AI.
"""
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from utils.etl_tools import PROJECT_ROOT, rel, resolve_path

PIPELINE_DIR = PROJECT_ROOT / "pipelines"
REGISTRY = PIPELINE_DIR / "registry.json"
RUNS_FILE = PROJECT_ROOT / ".agent" / "runs.json"

PIPELINE_TEMPLATE = '''"""
Pipeline : {name}
Created  : {created} by the ETL Analyst agent (from run #{run_id})
Request  : "{request}"
Input    : {input}  ({rows_in} rows)
Output   : {output}  ({rows_out} rows)

Re-run:  python pipelines/{name}.py --input <file> --output <file>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


{transform_def}


def read_table(path):
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".json":
        return pd.read_json(path, lines=True)
    return pd.read_parquet(path)


def write_table(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext == ".json":
        df.to_json(path, orient="records", lines=True)
    else:
        df.to_parquet(path, index=False)


def main(input_path, output_path):
    df = read_table(input_path)
    out = transform(df.copy())
    write_table(out, output_path)
    print(f"{{len(df)}} rows in -> {{len(out)}} rows out -> {{output_path}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="{input}")
    parser.add_argument("--output", default="{output}")
    args = parser.parse_args()
    main(args.input, args.output)
'''


# ------------------------------------------------------------------ storage

def _read_json(path: Path) -> list:
    return json.loads(path.read_text()) if path.exists() else []


def _write_json(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "pipeline"


# ------------------------------------------------------------------ drafts

def record_run(request: str, result: dict) -> int:
    """Store a successful transform as a draft. Returns the run number."""
    runs = _read_json(RUNS_FILE)
    run = {"id": len(runs) + 1, "request": request,
           "transform_def": result["transform_def"],
           "input": result["input"], "output": result["output"],
           "rows_in": result["rows_in"], "rows_out": result["rows_out"],
           "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "saved_as": None}
    _write_json(RUNS_FILE, runs + [run])
    return run["id"]


def list_runs() -> str:
    runs = _read_json(RUNS_FILE)
    if not runs:
        return "No transform runs yet."
    lines = []
    for r in runs:
        status = f"saved as '{r['saved_as']}'" if r["saved_as"] else "draft"
        lines.append(f"#{r['id']} [{r['time']}] {r['request']}  "
                     f"({r['rows_in']} -> {r['rows_out']} rows, {r['output']})  [{status}]")
    return "\n".join(lines)


def resolve_run(ref: str) -> dict:
    """'this'/'last' -> newest, 'that'/'previous' -> one before, '#2' -> run 2, words -> best match."""
    runs = _read_json(RUNS_FILE)
    if not runs:
        raise ValueError("There are no runs yet. Run a transform first.")
    text = ref.strip().lower()

    if text in {"this", "this one", "last", "latest", "it"}:
        return runs[-1]
    if text in {"that", "that one", "previous", "the one before", "the previous one"}:
        if len(runs) < 2:
            raise ValueError("There is only one run, so there is no previous one.")
        return runs[-2]
    number = re.fullmatch(r"#?\s*(\d+)", text)
    if number:
        for r in runs:
            if r["id"] == int(number.group(1)):
                return r
        raise ValueError(f"There is no run #{number.group(1)}.\n{list_runs()}")

    stop_words = {"the", "one", "pipeline", "run", "save", "this", "that", "a", "an"}
    words = set(re.findall(r"[a-z0-9]+", text)) - stop_words
    scored = [(len(words & set(re.findall(r"[a-z0-9]+", r["request"].lower()))), r) for r in runs]
    best = max(score for score, _ in scored)
    if best == 0:
        raise ValueError(f"No run matches '{ref}'.\n{list_runs()}")
    matches = [r for score, r in scored if score == best]
    if len(matches) > 1:
        options = "\n".join(f"  #{r['id']} {r['request']}" for r in matches)
        raise ValueError(f"'{ref}' matches more than one run. Ask the user which one:\n{options}")
    return matches[0]


# ------------------------------------------------------------------ pipelines

def save_run(ref: str, name: str) -> str:
    """Turn a draft run into a pipeline script. Never overwrites: adds _v2, _v3 ..."""
    run = resolve_run(ref)
    registry = _read_json(REGISTRY)
    existing = {p["name"] for p in registry}
    base = _slug(name)
    final, version = base, 1
    while final in existing:
        version += 1
        final = f"{base}_v{version}"

    path = PIPELINE_DIR / f"{final}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PIPELINE_TEMPLATE.format(
        name=final, created=datetime.now().strftime("%Y-%m-%d %H:%M"), run_id=run["id"],
        request=run["request"].replace('"', "'"), input=run["input"], output=run["output"],
        rows_in=run["rows_in"], rows_out=run["rows_out"], transform_def=run["transform_def"]))

    registry.append({"name": final, "file": path.relative_to(PROJECT_ROOT).as_posix(),
                     "request": run["request"], "from_run": run["id"],
                     "input": run["input"], "output": run["output"],
                     "created": datetime.now().isoformat(timespec="minutes")})
    _write_json(REGISTRY, registry)

    runs = _read_json(RUNS_FILE)
    for r in runs:
        if r["id"] == run["id"]:
            r["saved_as"] = final
    _write_json(RUNS_FILE, runs)

    note = f" ('{base}' already existed, so this is version {version})" if version > 1 else ""
    return (f"OK. Saved run #{run['id']} (\"{run['request']}\") as pipeline '{final}'{note}.\n"
            f"File: pipelines/{final}.py\n"
            f"Re-run any time without AI: python pipelines/{final}.py --input <file> --output <file>")


def list_pipelines() -> str:
    registry = _read_json(REGISTRY)
    if not registry:
        return "No saved pipelines yet."
    return "\n".join(f"- {p['name']}: {p['request']}  (default input {p['input']})" for p in registry)


def run_pipeline(name: str, input_path: str = "", output_path: str = "") -> str:
    """Re-run a saved pipeline. No LLM involved."""
    registry = {p["name"]: p for p in _read_json(REGISTRY)}
    if name not in registry:
        return f"ERROR: no pipeline called '{name}'.\n{list_pipelines()}"
    cmd = [sys.executable, str(PROJECT_ROOT / registry[name]["file"])]
    try:
        if input_path:
            cmd += ["--input", rel(resolve_path(input_path))]
        if output_path:
            cmd += ["--output", rel(resolve_path(output_path))]
    except ValueError as e:
        return f"ERROR: {e}"
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        return f"ERROR: pipeline '{name}' failed:\n{proc.stderr.strip()[-800:]}"
    return f"OK. {proc.stdout.strip()}"
