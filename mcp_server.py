"""
MCP Server for Radian Reporting (MySQL)

This module exposes read-only reporting tools from the
`radianreporting` MySQL database using MCP.

Purpose:
- Allow an AI (Claude / ChatGPT) to safely discover reporting tables
- Connects using a read-only database user (mcp_read)
- No write operations are permitted

This server runs locally during development and can later
be pointed to the India production database by changing .env values.
"""

import os
from dotenv import load_dotenv
import mysql.connector
from mcp.server.fastmcp import FastMCP

# Load environment variables from .env file
load_dotenv()

# Create the MCP server instance
# The name is what the AI will see as the tool provider
mcp = FastMCP("radian-mcp")


def get_conn():
    """
    Create and return a MySQL database connection.

    This function:
    - Reads database credentials from environment variables
    - Connects using a read-only MySQL user
    - Returns a live connection object

    Returns:
        mysql.connector.connection.MySQLConnection:
            An active connection to the radianreporting database.
    """
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "mcp_read"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "radianreporting"),
    )


@mcp.tool()
def list_reporting_tables():
    """
    List all tables in the radianreporting database.

    This is the first MCP tool and acts as a safe discovery mechanism.
    It allows the AI to understand what data is available without
    seeing any sensitive records.

    Returns:
        list[str]:
            A list of table names in the radianreporting schema.
    """
    # Open a database connection
    cnx = get_conn()

    # Create a cursor to run SQL queries
    cur = cnx.cursor()

    # Execute a safe, read-only query
    cur.execute("SHOW TABLES;")

    # Extract table names from the result
    tables = [row[0] for row in cur.fetchall()]

    # Clean up database resources
    cur.close()
    cnx.close()

    return tables

@mcp.tool()
def describe_reporting_table(table: str):
    """
    Describe the columns of a given table.

    Simple meaning:
    - You tell me a table name (example: 'patients')
    - I return the column names + types (so the AI knows what fields exist)

    Args:
        table (str): The table name to describe.

    Returns:
        list[dict]: Each row describes one column (Field, Type, Null, Key, Default, Extra).
    """
    cnx = get_conn()
    cur = cnx.cursor(dictionary=True)

    # Using DESCRIBE to get column metadata safely (no data returned)
    cur.execute(f"DESCRIBE `{table}`;")
    rows = cur.fetchall()

    cur.close()
    cnx.close()
    return rows

@mcp.tool()
def sample_reporting_rows(table: str, limit: int = 5):
    """
    Return a small sample of rows from a table.

    Simple meaning:
    - You give a table name
    - You optionally give how many rows (default = 5)
    - I return only a tiny sample so it’s safe

    Args:
        table (str): Table name (example: 'patients')
        limit (int): Number of rows to return (max 20)

    Returns:
        list[dict]: Sample rows from the table
    """
    # Hard safety cap so no one can pull too much data
    limit = min(int(limit), 20)

    cnx = get_conn()
    cur = cnx.cursor(dictionary=True)

    cur.execute(f"SELECT * FROM `{table}` LIMIT {limit};")
    rows = cur.fetchall()

    cur.close()
    cnx.close()
    return rows

@mcp.tool()
def count_table_rows(table: str):
    """
    Count how many rows exist in a given table.

    Simple meaning:
    - You give a table name (example: 'visits')
    - I return how many rows are inside it
    - This helps the AI avoid sampling empty tables

    Args:
        table (str): The table name to count.

    Returns:
        dict: {"table": <name>, "row_count": <number>}
    """
    cnx = get_conn()
    cur = cnx.cursor()

    cur.execute(f"SELECT COUNT(*) FROM `{table}`;")
    row_count = cur.fetchone()[0]

    cur.close()
    cnx.close()

    return {"table": table, "row_count": row_count}


if __name__ == "__main__":
    mcp.run()
