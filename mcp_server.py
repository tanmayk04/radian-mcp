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
    - You give a patient name or ID (or PID like ECC609Y)
    - I return:
        * best matching patient (prefers exact PID/ID, then First+Last, then partials)
        * recent imaging visits
        * whether each visit has a report (if linkable)

    Why this version:
    - Fixes the "multiple patients with same name" problem by ranking matches
      and preferring the patient who actually has visits.

    Args:
        patient_query (str): Patient identifier (name, PID/MRN, or ID)
        max_visits (int): Maximum visits to return (default 10, max 25)

    Returns:
        dict: { ok, found, patient, visits, summary, schema_notes }
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
        p_pid = _pick_col(p_cols, ["pid", "mrn", "medical_record", "medical_record_number", "patient_mrn", "record_number"])
        p_fname = _pick_col(p_cols, ["first_name", "firstname", "given_name"])
        p_lname = _pick_col(p_cols, ["last_name", "lastname", "family_name", "surname"])
        p_name = _pick_col(p_cols, ["name", "full_name"])
        p_dob = _pick_col(p_cols, ["dob", "dob_old", "date_of_birth", "birth_date"])
        p_gender = _pick_col(p_cols, ["gender", "sex"])

        if not p_pk:
            return {"ok": False, "error": "Could not detect primary key column in patients table."}

        # --- Detect visit columns / keys (use your real schema if present) ---
        v_pk = _pick_col(v_cols, ["visit_no", "id", "visit_id", "appt_id", "appointment_id", "encounter_id", "encounter_nbr"])
        v_patient_fk = _pick_col(v_cols, ["patient_id", "pat_id", "person_id", "pat_person_nbr", "patient_number", "pat_person_id"])
        v_date = _pick_col(v_cols, ["visit_start", "visit_date", "appt_date", "appointment_date", "scheduled_date", "created_at", "appt_create_date", "create_date"])
        v_status = _pick_col(v_cols, ["order_status", "status", "appt_status", "visit_status", "appt_status_desc", "status_desc"])
        v_exam = _pick_col(v_cols, ["exam_description", "exam_code", "modality", "procedure", "reason_desc", "appt_class"])

        # --- Build ranked patient search ---
        like_full = f"%{q}%"
        parts = q.split()
        first_part = parts[0] if len(parts) >= 1 else ""
        last_part = parts[-1] if len(parts) >= 2 else ""

        where_clauses: List[str] = []
        where_params: List[Any] = []

        def _add_where(clause: Optional[str], *vals: Any):
            if clause:
                where_clauses.append(clause)
                where_params.extend(vals)

        # Candidate match clauses (OR)
        if q.isdigit():
            _add_where(f"CAST(p.`{p_pk}` AS CHAR) = %s", q)

        if p_pid:
            _add_where(f"CAST(p.`{p_pid}` AS CHAR) = %s", q)

        if p_name:
            _add_where(f"p.`{p_name}` LIKE %s", like_full)

        if first_part and last_part and p_fname and p_lname:
            _add_where(f"(p.`{p_fname}` LIKE %s AND p.`{p_lname}` LIKE %s)", f"%{first_part}%", f"%{last_part}%")

        if p_fname:
            _add_where(f"p.`{p_fname}` LIKE %s", like_full)

        if p_lname:
            _add_where(f"p.`{p_lname}` LIKE %s", like_full)

        if not where_clauses:
            return {
                "ok": False,
                "error": "Could not determine searchable columns in patients table.",
                "hint": "Run describe_reporting_table('patients') and confirm name/id columns.",
            }

        # Match score (higher = better), used to pick the correct patient when duplicates exist
        score_parts: List[str] = []
        score_params: List[Any] = []

        if q.isdigit():
            score_parts.append(f"WHEN CAST(p.`{p_pk}` AS CHAR) = %s THEN 100")
            score_params.append(q)

        if p_pid:
            score_parts.append(f"WHEN CAST(p.`{p_pid}` AS CHAR) = %s THEN 90")
            score_params.append(q)

        if first_part and last_part and p_fname and p_lname:
            score_parts.append(f"WHEN (p.`{p_fname}` LIKE %s AND p.`{p_lname}` LIKE %s) THEN 80")
            score_params.extend([f"%{first_part}%", f"%{last_part}%"])

        if p_name:
            score_parts.append(f"WHEN p.`{p_name}` LIKE %s THEN 70")
            score_params.append(like_full)

        if p_fname:
            score_parts.append(f"WHEN p.`{p_fname}` LIKE %s THEN 60")
            score_params.append(like_full)

        if p_lname:
            score_parts.append(f"WHEN p.`{p_lname}` LIKE %s THEN 60")
            score_params.append(like_full)

        score_case = "CASE " + " ".join(score_parts) + " ELSE 0 END"

        # Select useful patient columns
        patient_select = _select_list(p_cols, [p_pk, p_pid, p_name, p_fname, p_lname, p_dob, p_gender])
        if not patient_select:
            patient_select_sql = "p.*"
        else:
            # patient_select items look like `col` -> prefix with p.
            patient_select_sql = ", ".join([s.replace("`", "p.`", 1) for s in patient_select])

        # If visits table doesn't have a linkable patient FK, we can still return patient info
        if not v_patient_fk or not v_pk:
            patient_sql = f"""
                SELECT {patient_select_sql}, {score_case} AS match_score
                FROM patients p
                WHERE {" OR ".join(where_clauses)}
                ORDER BY match_score DESC, p.`{p_pk}` DESC
                LIMIT 1;
            """
            cur = cnx.cursor(dictionary=True)
            cur.execute(patient_sql, score_params + where_params)
            patient = cur.fetchone()
            cur.close()

            if not patient:
                return {"ok": True, "found": False, "message": f"No patient found for '{q}'."}

            return {
                "ok": True,
                "found": True,
                "patient": patient,
                "visits": [],
                "summary": {
                    "patient_name": (patient.get(p_name) if p_name else None)
                    or f"{patient.get(p_fname,'') if p_fname else ''} {patient.get(p_lname,'') if p_lname else ''}".strip()
                    or None,
                    "total_visits_returned": 0,
                    "visits_with_report": 0,
                    "visits_missing_report": 0,
                    "note": "Visits table could not be linked to patients with detected keys.",
                },
                "schema_notes": {
                    "patients_pk": p_pk,
                    "patients_pid": p_pid,
                    "visits_pk": v_pk,
                    "visits_patient_fk": v_patient_fk,
                },
            }

        # Ranked patient query that prefers patients WITH visits when names collide
        patient_sql = f"""
            SELECT
                {patient_select_sql},
                COUNT(v.`{v_pk}`) AS visit_count,
                {score_case} AS match_score
            FROM patients p
            LEFT JOIN visits v
              ON v.`{v_patient_fk}` = p.`{p_pk}`
            WHERE {" OR ".join(where_clauses)}
            GROUP BY p.`{p_pk}`
            ORDER BY match_score DESC, visit_count DESC, p.`{p_pk}` DESC
            LIMIT 1;
        """

        cur = cnx.cursor(dictionary=True)
        cur.execute(patient_sql, score_params + where_params)
        patient = cur.fetchone()
        cur.close()

        if not patient:
            return {"ok": True, "found": False, "message": f"No patient found for '{q}'."}

        # patient id to link visits (always use p_pk value)
        patient_id_value = patient.get(p_pk)

        # --- Fetch recent visits for this patient ---
        visit_select_cols: List[str] = []
        for c in [v_pk, v_patient_fk, v_date, v_status, v_exam]:
            if c and c in v_cols and f"`{c}`" not in visit_select_cols:
                visit_select_cols.append(f"`{c}`")

        if not visit_select_cols:
            visit_select_cols = ["*"]

        order_by = f"ORDER BY `{v_date}` DESC" if v_date else (f"ORDER BY `{v_pk}` DESC" if v_pk else "")
        visits_sql = f"""
            SELECT {", ".join(visit_select_cols)}
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
        r_visit_fk = _pick_col(r_cols, ["visit_no", "visit_id", "appt_id", "appointment_id", "encounter_id", "encounter_nbr"])
        r_status = _pick_col(r_cols, ["status", "report_status", "final_status", "status_desc"])
        r_created = _pick_col(r_cols, ["created_at", "create_date", "report_date", "dt_created"])
        r_signed = _pick_col(r_cols, ["signed_at", "finalized_at", "completed_at", "dt_signed"])

        reports_by_visit: Dict[Any, List[Dict[str, Any]]] = {}

        if visits and v_pk and r_visit_fk:
            visit_ids = [v.get(v_pk) for v in visits if v.get(v_pk) is not None]
            if visit_ids:
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
                    key = r.get(r_visit_fk)
                    reports_by_visit.setdefault(key, []).append(r)

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
                    "sample": linked[:1],
                }
                final_visits.append(v2)

        # Display name
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

        if not r_visit_fk:
            summary["note"] = "Could not link reports to visits (missing visit key in reports table)."

        return {
            "ok": True,
            "found": True,
            "patient": patient,
            "visits": final_visits,
            "summary": summary,
            "schema_notes": {
                "patients_pk": p_pk,
                "patients_pid": p_pid,
                "visits_pk": v_pk,
                "visits_patient_fk": v_patient_fk,
                "visits_date_col": v_date,
                "reports_visit_fk": r_visit_fk,
            },
        }

    finally:
        cnx.close()

from datetime import datetime, date, timedelta
import calendar
from typing import Dict, Any, List, Optional


def _to_date(x) -> date:
    """Convert MySQL datetime/date/str to python date."""
    if x is None:
        raise ValueError("Cannot convert None to date")
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    # fallback: parse string
    return datetime.fromisoformat(str(x)).date()


def _month_range(d: date) -> (date, date):
    """Return first and last day of month for date d."""
    first = d.replace(day=1)
    last_day = calendar.monthrange(d.year, d.month)[1]
    last = d.replace(day=last_day)
    return first, last


def _quarter(d: date) -> int:
    return ((d.month - 1) // 3) + 1


def _quarter_range(d: date) -> (date, date):
    """Return first and last day of quarter for date d."""
    q = _quarter(d)
    start_month = (q - 1) * 3 + 1
    end_month = start_month + 2
    start = date(d.year, start_month, 1)
    end_last_day = calendar.monthrange(d.year, end_month)[1]
    end = date(d.year, end_month, end_last_day)
    return start, end


def _shift_month(d: date, months: int) -> date:
    """Shift date by N months, clamping day to month length."""
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    day = min(d.day, calendar.monthrange(y, m)[1])
    return date(y, m, day)


def _period_to_range(snapshot_end: date, period: str) -> (date, date):
    """
    Interpret period relative to snapshot_end (latest date in dump).
    Returns (start_date, end_date) inclusive.
    """
    p = (period or "").strip().lower()

    if p in ("last_10_days", "last10days", "last-10-days"):
        return snapshot_end - timedelta(days=9), snapshot_end

    if p in ("last_30_days", "last30days", "last-30-days"):
        return snapshot_end - timedelta(days=29), snapshot_end

    if p in ("this_month", "month_to_date", "mtd"):
        start = snapshot_end.replace(day=1)
        return start, snapshot_end

    if p in ("last_month",):
        prev = _shift_month(snapshot_end.replace(day=1), -1)
        start, end = _month_range(prev)
        return start, end

    if p in ("this_quarter", "quarter_to_date", "qtd"):
        q_start, _ = _quarter_range(snapshot_end)
        return q_start, snapshot_end

    if p in ("last_quarter",):
        # take the first day of current quarter, go back 1 day, then get that quarter range
        this_q_start, _ = _quarter_range(snapshot_end)
        prev_q_day = this_q_start - timedelta(days=1)
        return _quarter_range(prev_q_day)

    if p in ("this_year", "year_to_date", "ytd"):
        start = date(snapshot_end.year, 1, 1)
        return start, snapshot_end

    if p in ("last_year",):
        start = date(snapshot_end.year - 1, 1, 1)
        end = date(snapshot_end.year - 1, 12, 31)
        return start, end

    raise ValueError(
        "Unsupported period. Use one of: "
        "last_10_days, last_30_days, this_month, last_month, this_quarter, last_quarter, this_year, last_year"
    )


@mcp.tool()
def ops_summary(period: str = "last_10_days", top_n_exams: int = 10) -> Dict[str, Any]:
    """
    Operations summary for a time period (snapshot-based).

    Simple meaning:
    - You give a period like:
        * last_10_days
        * last_month
        * last_year
        * this_quarter
        * last_quarter
    - I summarize radiology activity for that period:
        * total visits
        * visits by status
        * top exams
        * daily volume trend
        * reports pending vs available (based on report_id)

    Important note (because you have a DB dump):
    - "last_month / last_10_days" is calculated relative to the latest visit_start in your dump,
      not today's real date.

    Args:
        period (str): One of the supported period values.
        top_n_exams (int): How many top exams to return (max 25).

    Returns:
        dict: A clean ops summary for the requested period.
    """
    top_n_exams = max(1, min(int(top_n_exams), 25))

    cnx = get_conn()
    try:
        cur = cnx.cursor(dictionary=True)

        # 1) Find snapshot end date (latest date in dump)
        cur.execute("SELECT MAX(visit_start) AS max_visit_start FROM visits;")
        row = cur.fetchone()
        if not row or row["max_visit_start"] is None:
            return {"ok": False, "error": "visits table has no data (MAX(visit_start) is NULL)."}

        snapshot_end = _to_date(row["max_visit_start"])

        # 2) Convert 'period' -> date range (inclusive)
        try:
            start_date, end_date = _period_to_range(snapshot_end, period)
        except ValueError as e:
            return {"ok": False, "error": str(e)}

        # 3) Total visits + report availability (using visits.report_id)
        cur.execute(
            """
            SELECT
              COUNT(*) AS total_visits,
              SUM(CASE WHEN report_id IS NULL OR report_id = '' THEN 1 ELSE 0 END) AS visits_missing_report,
              SUM(CASE WHEN report_id IS NOT NULL AND report_id <> '' THEN 1 ELSE 0 END) AS visits_with_report
            FROM visits
            WHERE DATE(visit_start) BETWEEN %s AND %s;
            """,
            (start_date, end_date),
        )
        totals = cur.fetchone() or {}

        # 4) Visits by status
        cur.execute(
            """
            SELECT
              order_status,
              COUNT(*) AS count
            FROM visits
            WHERE DATE(visit_start) BETWEEN %s AND %s
            GROUP BY order_status
            ORDER BY count DESC;
            """,
            (start_date, end_date),
        )
        by_status = cur.fetchall()

        # 5) Top exams (code + description)
        cur.execute(
            f"""
            SELECT
              exam_code,
              exam_description,
              COUNT(*) AS count
            FROM visits
            WHERE DATE(visit_start) BETWEEN %s AND %s
            GROUP BY exam_code, exam_description
            ORDER BY count DESC
            LIMIT {top_n_exams};
            """,
            (start_date, end_date),
        )
        top_exams = cur.fetchall()

        # 6) Daily volume trend + busiest day
        cur.execute(
            """
            SELECT
              DATE(visit_start) AS day,
              COUNT(*) AS visit_count
            FROM visits
            WHERE DATE(visit_start) BETWEEN %s AND %s
            GROUP BY DATE(visit_start)
            ORDER BY day ASC;
            """,
            (start_date, end_date),
        )
        daily = cur.fetchall()

        busiest_day = None
        if daily:
            busiest_day = max(daily, key=lambda x: x.get("visit_count", 0))

        cur.close()

        return {
            "ok": True,
            "period": period,
            "snapshot_latest_visit_date": str(snapshot_end),
            "date_range": {"start": str(start_date), "end": str(end_date)},
            "totals": {
                "total_visits": int(totals.get("total_visits") or 0),
                "visits_with_report": int(totals.get("visits_with_report") or 0),
                "visits_missing_report": int(totals.get("visits_missing_report") or 0),
            },
            "by_status": by_status,
            "top_exams": top_exams,
            "daily_volume": daily,
            "busiest_day": busiest_day,
        }

    finally:
        cnx.close()

from typing import Dict, Any, List, Optional


@mcp.tool()
def missing_reports_queue(
    period: str = "last_10_days",
    status_filter: Optional[List[str]] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Work queue of visits that are missing reports (snapshot-based).

    Simple meaning:
    - You give a time period like last_10_days / last_month
    - I return a ranked list of imaging visits where the report is not available yet
    - This is the "follow-up list" for ops staff to chase completion

    Important note (DB dump):
    - The queue is "as of the dump" (snapshot), not real-time.

    Args:
        period (str):
            One of: last_10_days, last_30_days, this_month, last_month,
                    this_quarter, last_quarter, this_year, last_year
        status_filter (list[str] | None):
            Optional list of order_status values to include (example: ["IP", "In Progress"]).
            If None, includes all statuses.
        limit (int):
            Max rows to return (default 50, max 200)

    Returns:
        dict:
            {
              "date_range": {...},
              "total_missing_reports_in_range": <int>,
              "queue": [ ... ],
              "notes": ...
            }
    """
    limit = max(1, min(int(limit), 200))

    cnx = get_conn()
    try:
        cur = cnx.cursor(dictionary=True)

        # 1) Find snapshot end (latest date in dump)
        cur.execute("SELECT MAX(visit_start) AS max_visit_start FROM visits;")
        row = cur.fetchone()
        if not row or row["max_visit_start"] is None:
            return {"ok": False, "error": "visits table has no data (MAX(visit_start) is NULL)."}

        snapshot_end = _to_date(row["max_visit_start"])

        # 2) period -> date range (inclusive)
        try:
            start_date, end_date = _period_to_range(snapshot_end, period)
        except ValueError as e:
            return {"ok": False, "error": str(e)}

        # 3) Build optional status filter SQL
        status_sql = ""
        params: List[Any] = [start_date, end_date]

        if status_filter:
            # normalize and keep only non-empty
            cleaned = [s for s in status_filter if s and str(s).strip()]
            if cleaned:
                placeholders = ", ".join(["%s"] * len(cleaned))
                status_sql = f" AND v.order_status IN ({placeholders}) "
                params.extend(cleaned)

        # 4) Count total missing reports in range
        cur.execute(
            f"""
            SELECT COUNT(*) AS cnt
            FROM visits v
            WHERE DATE(v.visit_start) BETWEEN %s AND %s
              AND (v.report_id IS NULL OR v.report_id = '')
              {status_sql}
            """,
            tuple(params),
        )
        total_cnt = (cur.fetchone() or {}).get("cnt", 0)

        # 5) Pull queue rows (oldest first)
        cur.execute(
            f"""
            SELECT
              v.visit_start,
              v.visit_no,
              v.exam_code,
              v.exam_description,
              v.order_status,
              v.machine,
              v.patient_id,
              p.pid AS patient_pid,
              p.first_name,
              p.last_name
            FROM visits v
            JOIN patients p ON p.id = v.patient_id
            WHERE DATE(v.visit_start) BETWEEN %s AND %s
              AND (v.report_id IS NULL OR v.report_id = '')
              {status_sql}
            ORDER BY v.visit_start ASC
            LIMIT {limit};
            """,
            tuple(params),
        )
        rows = cur.fetchall()

        cur.close()

        # 6) Format queue a bit nicer
        queue = []
        for r in rows:
            queue.append({
                "visit_start": str(r.get("visit_start")),
                "visit_no": r.get("visit_no"),
                "exam": f"{r.get('exam_description')} ({r.get('exam_code')})".strip(),
                "order_status": r.get("order_status"),
                "machine": r.get("machine"),
                "patient": {
                    "id": r.get("patient_id"),
                    "pid": r.get("patient_pid"),
                    "name": f"{r.get('first_name','')} {r.get('last_name','')}".strip()
                }
            })

        notes = "Queue is snapshot-based (as of latest date in dump)."

        if status_filter:
            notes += f" Status filter applied: {status_filter}"

        return {
            "ok": True,
            "period": period,
            "snapshot_latest_visit_date": str(snapshot_end),
            "date_range": {"start": str(start_date), "end": str(end_date)},
            "total_missing_reports_in_range": int(total_cnt),
            "returned": len(queue),
            "queue": queue,
            "notes": notes,
        }

    finally:
        cnx.close()


if __name__ == "__main__":
    mcp.run()
