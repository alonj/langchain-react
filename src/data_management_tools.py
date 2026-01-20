from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import RunContext, Tool


@dataclass
class DataManager:
    sqlite_connections: dict[str, sqlite3.Connection] = field(default_factory=dict)
    dataframes: dict[str, pd.DataFrame] = field(default_factory=dict)

    def get_connection(self, database: str) -> sqlite3.Connection:
        if database not in self.sqlite_connections:
            connection = sqlite3.connect(database)
            connection.row_factory = sqlite3.Row
            self.sqlite_connections[database] = connection
        return self.sqlite_connections[database]


class SQLiteFTSRequest(BaseModel):
    action: Literal["execute", "query", "fts_query", "list_tables"] = Field(
        ...,
        description="Execute SQL, run a query, run an FTS5 MATCH query, or list tables.",
    )
    database: str = Field(..., description="SQLite database name or path.")
    sql: str | None = Field(default=None, description="SQL statement for execute/query actions.")
    parameters: list[Any] | None = Field(default=None, description="Parameters for the SQL statement.")
    table: str | None = Field(default=None, description="FTS5 table name for match queries.")
    query: str | None = Field(default=None, description="FTS5 search query string.")
    limit: int = Field(default=50, description="Maximum rows to return.")


class DataFrameRequest(BaseModel):
    action: Literal["create", "update", "query", "list", "delete", "describe"] = Field(
        ...,
        description="Create, update, query, list, delete, or describe a dataframe.",
    )
    name: str | None = Field(default=None, description="Name of the dataframe.")
    data: Any | None = Field(default=None, description="Data used for creating a dataframe.")
    append: list[dict[str, Any]] | None = Field(default=None, description="Rows to append to a dataframe.")
    assign: dict[str, Any] | None = Field(default=None, description="Column assignments to apply.")
    drop: list[str] | None = Field(default=None, description="Columns to drop from a dataframe.")
    query: str | None = Field(default=None, description="Pandas query string to filter rows.")
    columns: list[str] | None = Field(default=None, description="Subset of columns to return.")
    limit: int = Field(default=50, description="Maximum rows to return for query results.")


def _rows_from_cursor(cursor: sqlite3.Cursor, limit: int) -> list[dict[str, Any]]:
    rows = cursor.fetchmany(limit)
    return [dict(row) for row in rows]


async def manage_sqlite_fts5(ctx: RunContext[Any], request: SQLiteFTSRequest) -> dict[str, Any]:
    """Create, update, and query SQLite databases with FTS5 support."""
    connection = ctx.deps.data_manager.get_connection(request.database)
    cursor = connection.cursor()

    if request.action == "list_tables":
        cursor.execute("SELECT name, type, sql FROM sqlite_master WHERE type IN ('table', 'view')")
        rows = _rows_from_cursor(cursor, request.limit)
        return {"tables": rows}

    if request.action == "execute":
        if not request.sql:
            raise ValueError("SQL is required for execute action.")
        cursor.execute(request.sql, request.parameters or [])
        connection.commit()
        return {"status": "ok", "rowcount": cursor.rowcount}

    if request.action == "query":
        if not request.sql:
            raise ValueError("SQL is required for query action.")
        cursor.execute(request.sql, request.parameters or [])
        rows = _rows_from_cursor(cursor, request.limit)
        return {"rows": rows}

    if request.action == "fts_query":
        if not request.table or request.query is None:
            raise ValueError("Table and query are required for fts_query action.")
        sql = f"SELECT rowid, * FROM {request.table} WHERE {request.table} MATCH ? LIMIT ?"
        cursor.execute(sql, [request.query, request.limit])
        rows = _rows_from_cursor(cursor, request.limit)
        return {"rows": rows}

    raise ValueError(f"Unsupported action: {request.action}")


async def manage_dataframes(ctx: RunContext[Any], request: DataFrameRequest) -> dict[str, Any]:
    """Create, update, and query pandas dataframes."""
    dataframes = ctx.deps.data_manager.dataframes

    if request.action == "list":
        return {"dataframes": sorted(dataframes.keys())}

    if request.action in {"create", "update", "query", "delete", "describe"} and not request.name:
        raise ValueError("A dataframe name is required for this action.")

    name = request.name
    if request.action == "create":
        if request.data is None:
            raise ValueError("Data is required to create a dataframe.")
        dataframes[name] = pd.DataFrame(request.data)
        df = dataframes[name]
        return {
            "status": "created",
            "shape": list(df.shape),
            "columns": list(df.columns),
            "preview": df.head(request.limit).to_dict(orient="records"),
        }

    if name not in dataframes:
        raise ValueError(f"Dataframe '{name}' does not exist.")

    if request.action == "delete":
        del dataframes[name]
        return {"status": "deleted", "name": name}

    df = dataframes[name]

    if request.action == "update":
        if request.append:
            df = pd.concat([df, pd.DataFrame(request.append)], ignore_index=True)
        if request.assign:
            for column, value in request.assign.items():
                df[column] = value
        if request.drop:
            df = df.drop(columns=request.drop)
        dataframes[name] = df
        return {
            "status": "updated",
            "shape": list(df.shape),
            "columns": list(df.columns),
            "preview": df.head(request.limit).to_dict(orient="records"),
        }

    if request.action == "describe":
        description = df.describe(include="all").to_dict()
        return {"description": description}

    if request.action == "query":
        result = df
        if request.query:
            result = result.query(request.query)
        if request.columns:
            result = result[request.columns]
        return {
            "rows": result.head(request.limit).to_dict(orient="records"),
            "shape": list(result.shape),
        }

    raise ValueError(f"Unsupported action: {request.action}")


def build_data_management_tools() -> dict[str, Tool]:
    return {
        "manage_sqlite_fts5": Tool(manage_sqlite_fts5),
        "manage_dataframes": Tool(manage_dataframes),
    }
