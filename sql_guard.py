"""
Read-only guard for LLM-generated SQL.

The LLM writes the SQL we run, so a bad (or manipulated) question could produce
DELETE, DROP, EXEC, etc. This module only lets a single SELECT statement through.
It is a safety net, not a replacement for connecting as a read-only DB user.
"""

import re
from typing import Dict, Tuple

from llama_index.core import SQLDatabase


class UnsafeSQLError(ValueError):
    """Raised when generated SQL is anything other than a single SELECT."""


# Keywords that can change data, schema, permissions or server state, or reach
# outside the database. T-SQL does not need semicolons between statements, so
# "SELECT 1 DELETE FROM t" is two statements; scanning for these catches that.
_FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "MERGE", "TRUNCATE", "DROP", "ALTER", "CREATE",
    "GRANT", "REVOKE", "DENY", "EXEC", "EXECUTE", "SP_EXECUTESQL", "INTO",
    "BULK", "OPENROWSET", "OPENQUERY", "OPENDATASOURCE", "OPENXML", "DBCC",
    "BACKUP", "RESTORE", "SHUTDOWN", "KILL", "RECONFIGURE", "WAITFOR", "USE",
    "DECLARE", "SET", "BEGIN", "COMMIT", "ROLLBACK", "TRAN", "TRANSACTION",
    "DISABLE", "ENABLE", "UPDATETEXT", "WRITETEXT",
}

_ALLOWED_FIRST_KEYWORDS = {"SELECT", "WITH"}

# String literals ('...' with '' escapes, N'...'), quoted identifiers ("..." and
# [...]), and comments (-- line, /* block */).
_STRIP_PATTERN = re.compile(
    r"""
      N?'(?:[^']|'')*'      # string literal
    | "(?:[^"]|"")*"        # double-quoted identifier
    | \[(?:[^\]]|\]\])*\]   # bracketed identifier
    | --[^\n]*              # line comment
    | /\*.*?\*/             # block comment
    """,
    re.VERBOSE | re.DOTALL,
)


def _strip_literals_and_comments(sql: str) -> str:
    # Replace with a space so tokens on either side stay separate.
    return _STRIP_PATTERN.sub(" ", sql)


def validate_select(sql: str) -> str:
    """Return the SQL unchanged if it is a single SELECT, else raise UnsafeSQLError."""
    if not sql or not sql.strip():
        raise UnsafeSQLError("Blocked query: the generated SQL was empty.")

    stripped = _strip_literals_and_comments(sql).strip()
    # Anything left over here was never closed, so we can't tell what is code.
    if any(mark in stripped for mark in ("'", '"', "[", "/*", "*/")):
        raise UnsafeSQLError(
            "Blocked query: the generated SQL has an unterminated string, "
            "identifier or comment."
        )

    # Allow a single trailing semicolon, nothing after it.
    stripped = stripped.rstrip().rstrip(";").strip()
    if ";" in stripped:
        raise UnsafeSQLError(
            "Blocked query: only one SQL statement is allowed, but the generated SQL "
            "contains several."
        )

    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_@#$]*", stripped)
    if not tokens or tokens[0].upper() not in _ALLOWED_FIRST_KEYWORDS:
        first = tokens[0].upper() if tokens else "nothing"
        raise UnsafeSQLError(
            f"Blocked query: only SELECT statements are allowed, but the generated "
            f"SQL starts with {first}."
        )

    upper_tokens = {t.upper() for t in tokens}
    forbidden = sorted(upper_tokens & _FORBIDDEN_KEYWORDS)
    if forbidden:
        raise UnsafeSQLError(
            "Blocked query: only read-only SELECT statements are allowed, but the "
            f"generated SQL contains {', '.join(forbidden)}."
        )

    procs = sorted(t for t in upper_tokens if t.startswith(("XP_", "SP_")))
    if procs:
        raise UnsafeSQLError(
            "Blocked query: calling stored procedures is not allowed "
            f"({', '.join(procs)})."
        )

    return sql


def blocked_reason(sql: str | None) -> str | None:
    """Return the guard's error message if this SQL would be blocked, else None."""
    if sql is None:
        return None
    try:
        validate_select(sql)
    except UnsafeSQLError as e:
        return str(e)
    return None


class ReadOnlySQLDatabase(SQLDatabase):
    """SQLDatabase that refuses to run anything but a single SELECT."""

    def run_sql(self, command: str) -> Tuple[str, Dict]:
        validate_select(command)
        return super().run_sql(command)
