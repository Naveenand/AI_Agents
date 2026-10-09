import psycopg2
from psycopg2 import sql
 
 
def _plain(value):
    """Show a value the way a person would read it: 58.90, not Decimal('58.90')."""
    if value is None:
        return "NULL"
    return str(value).strip()  # strip pads from fixed-width char columns
 
 
class Databaseutil:
    """Small helper around a PostgreSQL connection.
 
    The connection is opened READ-ONLY with a statement timeout, so even if a
    bad query slips past every other check, Postgres refuses to change data and
    stops queries that run too long.
 
    Use it with `with`, so the connection is always closed:
        with Databaseutil(config) as db:
            result = db.execute_sql("SELECT 1;")
    """
 
    def __init__(self, db_config, read_only=True, timeout_ms=10000, max_rows=5):
        self.db_config = db_config
        self.max_rows = max_rows
        self.connection_error = None
        try:
            self.connection = psycopg2.connect(
                **db_config,
                options=f"-c statement_timeout={timeout_ms}",  # kill queries after 10 s
            )
            if read_only:
                self.connection.set_session(readonly=True)    # Postgres rejects any write
        except Exception as e:
            print(f"Error Connection to Database: {e}")
            self.connection = None
            self.connection_error = str(e).strip()
 
    def __enter__(self):
        return self
 
    def __exit__(self, *exc):
        self.close()
 
    def schema_details(self, schema_name):
        if self.connection is None:
            print("No database connection")
            return None
 
        schema_info_context = f"Database schema : {schema_name}\n"
        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = %s;",
                (schema_name,),
            )
            for (table_name,) in cursor.fetchall():
                schema_info_context += f"\n Table: {table_name}\n"
 
                cursor.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position;",
                    (schema_name, table_name),
                )
                for column_name, data_type in cursor.fetchall():
                    schema_info_context += f" columns:{column_name} data_types : {data_type}\n"
 
                cursor.execute(
                    sql.SQL("SELECT * FROM {}.{} LIMIT 2;").format(
                        sql.Identifier(schema_name), sql.Identifier(table_name)
                    )
                )
                schema_info_context += " sample_data: \n"
                for row in cursor.fetchall():
                    # Plain values, so the LLM sees 58.90 and 2017-10-02 10:56:33,
                    # not Decimal('58.90') and datetime.datetime(2017, 10, 2, ...)
                    schema_info_context += "     (" + ", ".join(_plain(v) for v in row) + ")\n"
 
            self.connection.rollback()  # end the read-only transaction
        except Exception as e:
            self.connection.rollback()
            print(f"Error fetching schema details : {e}")
            return None
        finally:
            if cursor:
                cursor.close()
 
        return schema_info_context
 
    def execute_sql(self, query):
        """Run a query and return the result as a small text table with column names.
 
        Errors are RAISED (not hidden), so the agent can show the real Postgres
        error to the LLM and let it fix the query.
        """
        if self.connection is None:
            raise ConnectionError(f"No database connection: {self.connection_error}")
 
        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
 
            if cursor.description is None:
                return "The query returned no rows."
 
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchmany(self.max_rows + 1)   # one extra, to know if we cut it off
            truncated = len(rows) > self.max_rows
            rows = rows[: self.max_rows]
 
            lines = [" | ".join(columns)]
            lines += [" | ".join(_plain(v) for v in row) for row in rows]
            if not rows:
                lines.append("(no rows)")
            elif truncated:
                lines.append(f"(showing the first {self.max_rows} rows only)")
            return "\n".join(lines)
        except Exception:
            self.connection.rollback()
            raise                                         # let the agent see the real error
        finally:
            if cursor:
                cursor.close()
            if self.connection is not None:
                self.connection.rollback()                # end the read-only transaction
 
    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None