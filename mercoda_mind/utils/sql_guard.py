"""Deterministic SQL safety check for LLM-generated queries.
 
No LLM, no database: it parses the SQL with sqlglot and allows only a single
read-only SELECT. Kept in its own module so it can be unit-tested and reused
by any agent that runs generated SQL.
"""
import re
 
import sqlglot
from sqlglot import exp
 
 
# Commands that change data, structure, or permissions. Blocked anywhere in the query,
# including inside CTEs. exp.Command catches statements sqlglot doesn't fully model (e.g. COPY).
BANNED_STATEMENTS = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.Merge, exp.TruncateTable, exp.Grant, exp.Command,
)
 
# Functions that are valid inside a SELECT but have side effects.
BANNED_FUNCTIONS = {
    "pg_sleep", "pg_terminate_backend", "pg_cancel_backend",
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir",
    "dblink", "dblink_exec", "lo_import", "lo_export", "set_config",
}
 
 
def clean_sql(raw: str) -> str:
    """Remove Markdown code fences (```sql ... ```) that models sometimes add."""
    text = str(raw).strip()
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    return text
 
 
def check_sql(sql: str) -> tuple[bool, str]:
    """Check that `sql` is a single read-only SELECT, using a SQL parser.
 
    Returns (is_safe, reason). Fails closed: anything it can't parse or
    doesn't recognise as a plain read is rejected.
    """
    if not sql or not sql.strip():
        return False, "The query is empty."
 
    # 1. Parse with Postgres grammar. Unparseable SQL is rejected.
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except sqlglot.errors.ParseError as e:
        return False, f"The SQL could not be parsed: {str(e).splitlines()[0]}"
 
    # 2. Exactly one statement (blocks "SELECT 1; DROP TABLE users;").
    if len(statements) != 1:
        return False, f"Expected exactly 1 statement, found {len(statements)}."
 
    # 3. Allowlist: the statement must be a SELECT (WITH ... SELECT counts) or a UNION of SELECTs.
    stmt = statements[0]
    if not isinstance(stmt, (exp.Select, exp.Union)):
        return False, f"Only SELECT queries are allowed, but this is a {type(stmt).__name__} statement."
 
    # 4. SELECT ... INTO creates a new table.
    if stmt.find(exp.Into):
        return False, "SELECT ... INTO creates a new table and is not allowed."
 
    # 5. Search the whole tree for write commands (catches DELETE hidden inside a CTE).
    for node in stmt.find_all(*BANNED_STATEMENTS):
        return False, f"The query contains a {type(node).__name__} command, which changes the database."
 
    # 6. Block functions with side effects.
    for func in stmt.find_all(exp.Func):
        name = (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).lower()
        if name in BANNED_FUNCTIONS:
            return False, f"The query calls {name}(), which has side effects and is not allowed."
 
    return True, "The query is a single read-only SELECT statement."