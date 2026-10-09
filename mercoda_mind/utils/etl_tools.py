"""
ETL tools used by the ETL Analyst agent.

Every function returns a plain-text message, because that text is what the AI reads.
Success messages always include facts (rows, columns, file path, preview) so the AI
reports real numbers instead of guessing. Failures start with "ERROR:" and say how to fix them.
"""
import ast
import json
import re
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SUPPORTED_FORMATS = (".csv", ".json", ".parquet")

# Remembers every API download: which URL, which file, how many rows, when.
CATALOG = DATA_DIR / "extract" / ".catalog.json"
REUSE_HOURS = 24          # a download younger than this is reused instead of downloaded again


# ------------------------------------------------------------------ helpers

def resolve_path(path: str) -> Path:
    """Turn 'data/extract/x.csv' into a full path, and refuse paths outside the project."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()
    if p != PROJECT_ROOT and PROJECT_ROOT not in p.parents:
        raise ValueError(f"Path must be inside the project folder ({PROJECT_ROOT}): {path}")
    return p


def rel(path: Path) -> str:
    """Short, readable path for messages: data/extract/pokemon.csv"""
    return path.relative_to(PROJECT_ROOT).as_posix()


def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".json":
        return pd.read_json(path, lines=True)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type '{ext}'. Use one of {SUPPORTED_FORMATS}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext == ".json":
        df.to_json(path, orient="records", lines=True)
    elif ext == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use one of {SUPPORTED_FORMATS}")


def describe(df: pd.DataFrame, sample_rows: int = 5) -> str:
    """What the AI needs to know about a table: size, columns, types, nulls, sample rows."""
    lines = [f"Rows: {len(df)}   Columns: {len(df.columns)}", "Columns (type, nulls):"]
    for col in df.columns:
        lines.append(f"  - {col} ({df[col].dtype}, {int(df[col].isna().sum())} nulls)")
    with pd.option_context("display.max_columns", 30, "display.width", 200,
                           "display.max_colwidth", 40):
        lines.append(f"First {min(sample_rows, len(df))} rows:\n{df.head(sample_rows).to_string(index=False)}")
    return "\n".join(lines)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "extracted_data"


# ------------------------------------------------------------------ download catalog

def _normalize_url(url: str) -> str:
    """So 'HTTPS://PokeAPI.co/api/v2/pokemon/?limit=100' and
    'https://pokeapi.co/api/v2/pokemon?limit=100' count as the same download."""
    parts = urlsplit(url.strip())
    query = urlencode(sorted(parse_qsl(parts.query)))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), query, ""))


def _load_catalog() -> list:
    try:
        return json.loads(CATALOG.read_text()) if CATALOG.exists() else []
    except json.JSONDecodeError:
        return []


def _save_catalog(entries: list) -> None:
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    CATALOG.write_text(json.dumps(entries, indent=2))


def _age_text(iso_time: str) -> str:
    minutes = int((datetime.now() - datetime.fromisoformat(iso_time)).total_seconds() // 60)
    if minutes < 60:
        return f"{minutes} min ago"
    if minutes < 48 * 60:
        return f"{minutes // 60} h ago"
    return f"{minutes // (24 * 60)} days ago"


def find_download(url: str, records_key: str = None) -> dict | None:
    """Newest catalog entry for this URL whose file still exists, or None."""
    key = _normalize_url(url)
    for entry in reversed(_load_catalog()):
        if (entry["url_key"] == key and entry.get("records_key") == records_key
                and (PROJECT_ROOT / entry["file"]).exists()):
            return entry
    return None


# ------------------------------------------------------------------ extract

def _find_records(data, records_key=None):
    """Find the list of records inside an API response, whatever it is called."""
    if isinstance(data, list):
        return data, "(top-level list)"
    if isinstance(data, dict):
        if records_key:
            if records_key not in data:
                raise KeyError(f"records_key '{records_key}' not found. Keys in response: {list(data)}")
            return data[records_key], records_key
        # pick the biggest list of objects, e.g. 'results', 'data', 'items'
        candidates = [(k, v) for k, v in data.items()
                      if isinstance(v, list) and v and isinstance(v[0], dict)]
        if candidates:
            key, records = max(candidates, key=lambda kv: len(kv[1]))
            return records, key
        return [data], "(single object)"
    raise ValueError(f"Response is not JSON records: {type(data).__name__}")


def extract_api(url: str, output_name: str = "extracted_data", format: str = "csv",
                records_key: str = None, max_pages: int = 20, refresh: bool = False) -> str:
    """Download JSON from an API (following 'next' page links) and save it as a table.
    If the same URL was downloaded less than REUSE_HOURS ago, reuse that file instead."""
    fmt = "." + format.lower().lstrip(".")
    if fmt not in SUPPORTED_FORMATS:
        return f"ERROR: format must be csv, json or parquet, not '{format}'."
    out = DATA_DIR / "extract" / f"{_safe_name(output_name)}{fmt}"

    # ---- 1. Check the catalog before touching the network ----
    previous = None if refresh else find_download(url, records_key)
    if previous:
        age_hours = (datetime.now() - datetime.fromisoformat(previous["downloaded_at"])).total_seconds() / 3600
        if age_hours < REUSE_HOURS:
            cached = PROJECT_ROOT / previous["file"]
            df = read_table(cached)
            copied = ""
            if out.resolve() != cached.resolve():      # asked for another name/format: copy, don't download
                write_table(df, out)
                copied = f"\nAlso saved a copy as {rel(out)} (no download needed)."
                entries = [e for e in _load_catalog() if e["file"] != rel(out)]
                _save_catalog(entries + [{**previous, "file": rel(out)}])
            return (f"REUSED: this URL was already downloaded {_age_text(previous['downloaded_at'])} "
                    f"to {previous['file']}, so nothing was downloaded.{copied}\n"
                    f"If the user wants the latest data, call again with refresh=true.\n{describe(df)}")

    # ---- 2. Download (first time, too old, file deleted, or refresh=true) ----
    frames, pages, next_url, key = [], 0, url, None
    try:
        while next_url and pages < max_pages:
            response = requests.get(next_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            records, key = _find_records(data, records_key)
            frames.append(pd.json_normalize(records))
            pages += 1
            next_url = data.get("next") if isinstance(data, dict) else None
    except requests.exceptions.RequestException as e:
        return f"ERROR: could not download {url}: {e}"
    except (ValueError, KeyError) as e:
        return f"ERROR: could not read the API response: {e}"

    df = pd.concat(frames, ignore_index=True)
    write_table(df, out)

    # ---- 3. Remember this download ----
    entries = [e for e in _load_catalog() if e["file"] != rel(out)]
    entries.append({"url": url, "url_key": _normalize_url(url), "records_key": records_key,
                    "file": rel(out), "rows": len(df), "columns": list(map(str, df.columns)),
                    "pages": pages, "complete": next_url is None,
                    "downloaded_at": datetime.now().isoformat(timespec="seconds")})
    _save_catalog(entries)

    note = ""
    if next_url:
        note = (f"\nNOTE: stopped after {max_pages} pages; more data is available. "
                f"Call again with a larger max_pages to get everything.")
    action = "Downloaded fresh data" if refresh else "Downloaded"
    return (f"OK. {action}: {pages} page(s) using records key '{key}', saved to {rel(out)}.\n"
            f"{describe(df)}{note}")


# ------------------------------------------------------------------ inspect

def profile_file(path: str) -> str:
    """Show a file's real columns, types, nulls and sample rows."""
    try:
        p = resolve_path(path)
        if not p.exists():
            return f"ERROR: file not found: {path}. Call list_files to see what exists."
        return f"File: {rel(p)}\n{describe(read_table(p))}"
    except Exception as e:
        return f"ERROR: could not read {path}: {e}"


def list_files(folder: str = "data") -> str:
    """List data files the agent can work with."""
    try:
        base = resolve_path(folder)
    except ValueError as e:
        return f"ERROR: {e}"
    files = sorted(f for f in base.rglob("*")
                   if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
                   and not any(part.startswith(".") for part in f.relative_to(base).parts))
    if not files:
        return f"No csv/json/parquet files in {folder}."
    sources = {e["file"]: e for e in _load_catalog()}
    lines = []
    for f in files:
        line = f"{rel(f)}  ({f.stat().st_size / 1024:.1f} KB)"
        src = sources.get(rel(f))
        if src:
            line += f"  <- downloaded from {src['url']} {_age_text(src['downloaded_at'])}, {src['rows']} rows"
        lines.append(line)
    return "\n".join(lines)


# ------------------------------------------------------------------ transform

ALLOWED_IMPORTS = {"pandas", "numpy", "math", "re", "datetime"}
BLOCKED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals",
                 "locals", "vars", "getattr", "setattr", "delattr", "breakpoint",
                 "exit", "quit", "os", "sys", "subprocess", "shutil", "socket"}


def build_transform_def(code: str) -> str:
    """Accept either a full `def transform(df):` or just its body, and return a full def."""
    code = textwrap.dedent(code).strip()
    code = re.sub(r"^```(?:python)?\s*|\s*```$", "", code).strip()   # remove markdown fences
    if re.search(r"^def\s+transform\s*\(", code, re.MULTILINE):
        return code
    return "def transform(df):\n" + textwrap.indent(code, "    ")


def check_transform_code(transform_def: str) -> str | None:
    """Static safety check. Returns an error message, or None if the code looks safe."""
    try:
        tree = ast.parse(transform_def)
    except SyntaxError as e:
        return f"Python syntax error on line {e.lineno}: {e.msg}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                if name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"import '{name}' is not allowed. Only {sorted(ALLOWED_IMPORTS)}."
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            return f"'{node.id}' is not allowed in transform code."
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return f"'{node.attr}' is not allowed in transform code."
            if node.attr.startswith(("to_", "read_")) and node.attr not in ("to_numeric", "to_datetime", "to_timedelta", "to_dict", "to_list"):
                return (f"'.{node.attr}' is not allowed: transform(df) must only change the "
                        f"DataFrame and return it. Reading and saving files is done for you.")
    return None


# Runs in a separate Python process, so a crash or endless loop can't take the agent down.
_RUNNER = r'''
import json, sys, traceback
import numpy as np, pandas as pd
job = json.load(sys.stdin)
sys.path.insert(0, job["utils_dir"])
from etl_tools import read_table, write_table
from pathlib import Path
try:
    df = read_table(Path(job["input"]))
    ns = {"pd": pd, "np": np}
    exec(job["code"], ns)
    out = ns["transform"](df.copy())
    if isinstance(out, pd.Series):
        out = out.to_frame()
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"transform(df) must return a DataFrame, got {type(out).__name__}")
    write_table(out, Path(job["output"]))
    print(json.dumps({"ok": True, "rows_in": len(df), "rows_out": len(out)}))
except Exception as e:
    tb = traceback.extract_tb(e.__traceback__)
    line = next((f.lineno for f in reversed(tb) if f.filename == "<string>"), None)
    print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}", "line": line,
                      "columns": list(df.columns) if "df" in dir() else []}))
'''


def run_transform(input_path: str, output_path: str, transform_code: str,
                  timeout_seconds: int = 60) -> dict:
    """
    Run transform(df) on input_path and save the result to output_path.
    Returns a dict: {"ok": bool, "message": str, ...run details}.
    """
    try:
        src, dst = resolve_path(input_path), resolve_path(output_path)
    except ValueError as e:
        return {"ok": False, "message": f"ERROR: {e}"}
    if not src.exists():
        return {"ok": False, "message": f"ERROR: input file not found: {input_path}. Call list_files."}
    if dst.suffix.lower() not in SUPPORTED_FORMATS:
        return {"ok": False, "message": f"ERROR: output must end with {SUPPORTED_FORMATS}."}
    if dst == src:
        return {"ok": False, "message": "ERROR: output_path must be different from input_path "
                                        "(never overwrite the source data)."}

    transform_def = build_transform_def(transform_code)
    problem = check_transform_code(transform_def)
    if problem:
        return {"ok": False, "message": f"ERROR: code rejected by safety check: {problem}\n"
                                        f"Rewrite transform(df) and call run_transform again."}

    job = {"input": str(src), "output": str(dst), "code": transform_def,
           "utils_dir": str(Path(__file__).resolve().parent)}
    try:
        proc = subprocess.run([sys.executable, "-c", _RUNNER], input=json.dumps(job),
                              capture_output=True, text=True, timeout=timeout_seconds,
                              cwd=PROJECT_ROOT)
    except subprocess.TimeoutExpired:
        return {"ok": False, "message": f"ERROR: transform took longer than {timeout_seconds}s "
                                        f"and was stopped. Avoid Python loops over rows."}
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"ok": False, "message": f"ERROR: transform process crashed:\n{proc.stderr[-800:]}"}

    if not result["ok"]:
        where = f" (line {result['line']} of transform)" if result.get("line") else ""
        return {"ok": False, "message": (
            f"ERROR in transform code{where}: {result['error']}\n"
            f"Columns available in the input: {result['columns']}\n"
            f"Your code was:\n{transform_def}\n"
            f"Fix the code and call run_transform again.")}

    # Verify: read the saved file back instead of trusting the code.
    saved = read_table(dst)
    warning = "\nWARNING: the result is empty (0 rows). Check the filter values." if saved.empty else ""
    return {
        "ok": True,
        "rows_in": result["rows_in"], "rows_out": len(saved),
        "input": rel(src), "output": rel(dst), "transform_def": transform_def,
        "message": (f"OK. {result['rows_in']} rows in -> {len(saved)} rows out. "
                    f"Saved to {rel(dst)}.\n{describe(saved)}{warning}"),
    }
