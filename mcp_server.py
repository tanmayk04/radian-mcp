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

import re
from typing import Any, Dict, List, Optional

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _safe_ident(name: str) -> str:
    """Validate table/column identifier to reduce SQL injection risk in dynamic SQL."""
    if not name or not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Unsafe identifier: {name!r}")
    return name


def _get_columns(cnx, table: str) -> List[str]:
    """Return column names for a table using DESCRIBE."""
    table = _safe_ident(table)
    cur = cnx.cursor()
    cur.execute(f"DESCRIBE `{table}`;")
    cols = [row[0] for row in cur.fetchall()]  # Field name is first column
    cur.close()
    return cols


def _pick_col(existing_cols: List[str], candidates: List[str]) -> Optional[str]:
    """Pick the first candidate that exists (case-insensitive)."""
    m = {c.lower(): c for c in existing_cols}
    for cand in candidates:
        if cand.lower() in m:
            return m[cand.lower()]
    return None


def _select_list(cols: List[str], wanted: List[Optional[str]]) -> List[str]:
    """Return a list of backticked column selects for columns that exist."""
    out = []
    existing = set(cols)
    for c in wanted:
        if c and c in existing and c not in out:
            out.append(f"`{c}`")
    return out


@mcp.tool()
def patient_profile(patient_query: str, max_visits: int = 10) -> Dict[str, Any]:
    """
    Show a complete profile for a single patient.

    Simple meaning:
    - You give a patient name or ID
    - I return:
        * basic patient details
        * recent imaging visits (if linkable)
        * whether each visit has a report (if linkable)

    Args:
        patient_query (str): Patient identifier (name, MRN, or ID)
        max_visits (int): Maximum visits to return (default 10, max 25)

    Returns:
        dict: { found, patient, visits, summary, schema_notes }
    """
    if not patient_query or not str(patient_query).strip():
        return {"ok": False, "error": "patient_query is required."}

    max_visits = max(1, min(int(max_visits), 25))
    q = str(patient_query).strip()

    cnx = get_conn()
    try:
        # --- Load columns (so we don't reference columns that don't exist) ---
        p_cols = _get_columns(cnx, "patients")
        v_cols = _get_columns(cnx, "visits")
        r_cols = _get_columns(cnx, "reports")

        # --- Detect key patient columns ---
        p_pk = _pick_col(p_cols, ["id", "patient_id", "pat_id", "person_id", "pat_person_nbr", "pat_person_id"])
        p_mrn = _pick_col(p_cols, ["mrn", "medical_record", "medical_record_number", "patient_mrn", "record_number"])
        p_fname = _pick_col(p_cols, ["first_name", "firstname", "given_name"])
        p_lname = _pick_col(p_cols, ["last_name", "lastname", "family_name", "surname"])
        p_name = _pick_col(p_cols, ["name", "full_name"])
        p_dob = _pick_col(p_cols, ["dob", "date_of_birth", "birth_date"])
        p_gender = _pick_col(p_cols, ["gender", "sex"])

        # --- Build patient search WHERE based on what exists ---
        where = []
        params: List[Any] = []

        # Numeric query? try exact matches
        if q.isdigit():
            if p_pk:
                where.append(f"CAST(`{p_pk}` AS CHAR) = %s")
                params.append(q)
            if p_mrn:
                where.append(f"CAST(`{p_mrn}` AS CHAR) = %s")
                params.append(q)

        # Name-like matches
        like = f"%{q}%"
        if p_name:
            where.append(f"`{p_name}` LIKE %s")
            params.append(like)
        if p_fname:
            where.append(f"`{p_fname}` LIKE %s")
            params.append(like)
        if p_lname:
            where.append(f"`{p_lname}` LIKE %s")
            params.append(like)

        if not where:
            return {
                "ok": False,
                "error": "Could not determine searchable columns in patients table.",
                "hint": "Run describe_reporting_table('patients') and confirm name/id columns.",
            }

        # Only select useful columns (that exist)
        patient_select = _select_list(
            p_cols,
            [p_pk, p_mrn, p_name, p_fname, p_lname, p_dob, p_gender]
        )
        if not patient_select:
            patient_select = ["*"]

        patient_sql = f"""
            SELECT {", ".join(patient_select)}
            FROM `patients`
            WHERE {" OR ".join(where)}
            LIMIT 1;
        """

        cur = cnx.cursor(dictionary=True)
        cur.execute(patient_sql, params)
        patient = cur.fetchone()
        cur.close()

        if not patient:
            return {"ok": True, "found": False, "message": f"No patient found for '{q}'."}

        # Pick a patient identifier value to link visits
        patient_id_value = None
        if p_pk and p_pk in patient:
            patient_id_value = patient[p_pk]
        elif p_mrn and p_mrn in patient:
            patient_id_value = patient[p_mrn]

        # --- Detect visit columns / keys ---
        v_pk = _pick_col(v_cols, ["id", "visit_id", "appt_id", "appointment_id", "encounter_id", "encounter_nbr"])
        v_patient_fk = _pick_col(v_cols, ["patient_id", "pat_id", "person_id", "pat_person_nbr", "patient_number", "pat_person_id"])
        v_date = _pick_col(v_cols, ["visit_date", "appt_date", "appointment_date", "scheduled_date", "created_at", "appt_create_date", "create_date"])
        v_status = _pick_col(v_cols, ["status", "appt_status", "visit_status", "appt_status_desc", "status_desc"])
        v_modality = _pick_col(v_cols, ["modality", "appt_class", "procedure", "reason_desc", "appt_svc_cntr_name", "appt_svc_cntr_code"])

        visits: List[Dict[str, Any]] = []
        if v_patient_fk and patient_id_value is not None:
            visit_select = []
            # Always include primary keys if possible
            for c in [v_pk, v_date, v_status, v_modality, v_patient_fk]:
                if c and c in v_cols and c not in visit_select:
                    visit_select.append(f"`{c}`")

            if not visit_select:
                visit_select = ["*"]

            order_by = f"ORDER BY `{v_date}` DESC" if v_date else (f"ORDER BY `{v_pk}` DESC" if v_pk else "")
            visits_sql = f"""
                SELECT {", ".join(visit_select)}
                FROM `visits`
                WHERE `{v_patient_fk}` = %s
                {order_by}
                LIMIT %s;
            """

            cur = cnx.cursor(dictionary=True)
            cur.execute(visits_sql, (patient_id_value, max_visits))
            visits = cur.fetchall()
            cur.close()

        # --- Attach report info if possible ---
        # Detect report linkage keys
        r_visit_fk = _pick_col(r_cols, ["visit_id", "appt_id", "appointment_id", "encounter_id", "encounter_nbr"])
        r_status = _pick_col(r_cols, ["status", "report_status", "final_status", "status_desc"])
        r_created = _pick_col(r_cols, ["created_at", "create_date", "report_date", "dt_created"])
        r_signed = _pick_col(r_cols, ["signed_at", "finalized_at", "completed_at", "dt_signed"])

        reports_by_visit: Dict[Any, List[Dict[str, Any]]] = {}
        if visits and v_pk and r_visit_fk:
            visit_ids = [v.get(v_pk) for v in visits if v.get(v_pk) is not None]
            if visit_ids:
                # Select only a few report columns
                report_select_cols = [r_visit_fk, r_status, r_created, r_signed]
                report_select_cols = [c for c in report_select_cols if c and c in r_cols]
                if not report_select_cols:
                    report_select_cols = [r_visit_fk]

                placeholders = ", ".join(["%s"] * len(visit_ids))
                reports_sql = f"""
                    SELECT {", ".join([f"`{c}`" for c in report_select_cols])}
                    FROM `reports`
                    WHERE `{r_visit_fk}` IN ({placeholders});
                """

                cur = cnx.cursor(dictionary=True)
                cur.execute(reports_sql, tuple(visit_ids))
                report_rows = cur.fetchall()
                cur.close()

                for r in report_rows:
                    k = r.get(r_visit_fk)
                    reports_by_visit.setdefault(k, []).append(r)

        # Build final visits with report status
        visits_missing_report = 0
        final_visits: List[Dict[str, Any]] = []
        if visits and v_pk:
            for v in visits:
                vid = v.get(v_pk)
                linked = reports_by_visit.get(vid, [])
                has_report = len(linked) > 0
                if not has_report:
                    visits_missing_report += 1

                v2 = dict(v)
                v2["report"] = {
                    "has_report": has_report,
                    "report_count": len(linked),
                    "sample": linked[:1],  # keep it small/safe
                }
                final_visits.append(v2)

        # Nice patient display fields
        display_name = None
        if p_name and p_name in patient:
            display_name = patient.get(p_name)
        else:
            fn = patient.get(p_fname, "") if p_fname else ""
            ln = patient.get(p_lname, "") if p_lname else ""
            display_name = f"{fn} {ln}".strip() or None

        summary = {
            "patient_name": display_name,
            "total_visits_returned": len(final_visits),
            "visits_with_report": len(final_visits) - visits_missing_report,
            "visits_missing_report": visits_missing_report,
            "note": None,
        }

        if not v_patient_fk:
            summary["note"] = "Could not link visits to patient (missing patient FK in visits table)."
        elif not r_visit_fk:
            summary["note"] = "Could not link reports to visits (missing visit FK in reports table)."

        return {
            "ok": True,
            "found": True,
            "patient": patient,
            "visits": final_visits,
            "summary": summary,
            "schema_notes": {
                "patients_pk": p_pk,
                "patients_mrn": p_mrn,
                "visits_pk": v_pk,
                "visits_patient_fk": v_patient_fk,
                "visits_date_col": v_date,
                "reports_visit_fk": r_visit_fk,
            },
        }

    finally:
        cnx.close()

if __name__ == "__main__":
    mcp.run()
