"""Unit tests for utils/sql_guard.py. Run from the project root with:  pytest -v"""
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.sql_guard import check_sql, clean_sql


SAFE_QUERIES = [
    "SELECT name FROM customers LIMIT 10;",
    "SELECT * FROM orders WHERE order_status = 'deleted';",    # 'deleted' is just a value
    "WITH t AS (SELECT 1 AS x) SELECT x FROM t;",
    "SELECT a FROM t UNION SELECT b FROM u;",
    "SELECT 'UNANSWERABLE: no weather data' AS error;",
    "SELECT c.customer_state, SUM(oi.price) AS total_sales "
    "FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
    "JOIN customers c ON c.customer_id = o.customer_id "
    "GROUP BY c.customer_state ORDER BY total_sales DESC LIMIT 5;",
]

UNSAFE_QUERIES = [
    ("SELECT 1; DROP TABLE users;", "Expected exactly 1 statement"),
    ("DELETE FROM orders;", "Only SELECT"),
    ("WITH r AS (DELETE FROM orders RETURNING *) SELECT * FROM r;", "Delete command"),
    ("SELECT * INTO backup FROM orders;", "SELECT ... INTO"),
    ("SELECT pg_sleep(600);", "pg_sleep"),
    ("SELECT pg_terminate_backend(123);", "pg_terminate_backend"),
    ("TRUNCATE orders;", "Only SELECT"),
    ("GRANT ALL ON orders TO bob;", "Only SELECT"),
    ("this is not sql", "Only SELECT"),
    ("", "empty"),
]


@pytest.mark.parametrize("sql", SAFE_QUERIES)
def test_safe_queries_are_allowed(sql):
    ok, reason = check_sql(sql)
    assert ok, reason


@pytest.mark.parametrize("sql, reason_part", UNSAFE_QUERIES)
def test_unsafe_queries_are_blocked_with_a_reason(sql, reason_part):
    ok, reason = check_sql(sql)
    assert not ok
    assert reason_part in reason


def test_clean_sql_removes_code_fences():
    assert clean_sql("```sql\nSELECT 1;\n```") == "SELECT 1;"
    assert clean_sql("SELECT 1;") == "SELECT 1;"