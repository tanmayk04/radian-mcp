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

from datetime import datetime, date, timedelta
import calendar
from typing import Dict, Any, List, Optional


# -----------------------------
# Period helpers (calendar-safe)
# -----------------------------
def _shift_month(d: date, months: int) -> date:
    """Shift date by N months, clamping day to month length."""
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    day = min(d.day, calendar.monthrange(y, m)[1])
    return date(y, m, day)


def _month_range(d: date) -> (date, date):
    """First and last day of the month for a given date."""
    first = d.replace(day=1)
    last_day = calendar.monthrange(d.year, d.month)[1]
    last = d.replace(day=last_day)
    return first, last


def _quarter(d: date) -> int:
    return ((d.month - 1) // 3) + 1


def _quarter_range(d: date) -> (date, date):
    """First and last day of the quarter for a given date."""
    q = _quarter(d)
    start_month = (q - 1) * 3 + 1
    end_month = start_month + 2
    start = date(d.year, start_month, 1)
    end_last_day = calendar.monthrange(d.year, end_month)[1]
    end = date(d.year, end_month, end_last_day)
    return start, end


def _resolve_period_to_range(period: str, anchor: Optional[date] = None) -> (str, str):
    """
    Convert a friendly period string to (date_from, date_to) in 'YYYY-MM-DD'.

    For production/live DB:
      anchor defaults to today's date.

    For DB dump/demo:
      you can pass anchor as the latest visit date in the dump
      (MAX(visit_start)) so "last_10_days" means last 10 days of the data.

    Supported:
      today, yesterday,
      last_10_days, last_30_days,
      this_month, last_month,
      this_quarter, last_quarter,
      this_year, last_year
    """
    d = anchor or datetime.now().date()
    p = (period or "").strip().lower()

    if p == "today":
        start = end = d

    elif p == "yesterday":
        start = end = d - timedelta(days=1)

    elif p in ("last_10_days", "last10days", "last-10-days"):
        start, end = d - timedelta(days=9), d

    elif p in ("last_30_days", "last30days", "last-30-days"):
        start, end = d - timedelta(days=29), d

    elif p in ("this_month", "month_to_date", "mtd"):
        start, end = d.replace(day=1), d

    elif p == "last_month":
        prev_month_first = _shift_month(d.replace(day=1), -1)
        start, end = _month_range(prev_month_first)

    elif p in ("this_quarter", "quarter_to_date", "qtd"):
        q_start, _ = _quarter_range(d)
        start, end = q_start, d

    elif p == "last_quarter":
        this_q_start, _ = _quarter_range(d)
        prev_q_day = this_q_start - timedelta(days=1)
        start, end = _quarter_range(prev_q_day)

    elif p in ("this_year", "year_to_date", "ytd"):
        start, end = date(d.year, 1, 1), d

    elif p == "last_year":
        start, end = date(d.year - 1, 1, 1), date(d.year - 1, 12, 31)

    else:
        raise ValueError(
            "Unsupported period. Use one of: "
            "today, yesterday, last_10_days, last_30_days, this_month, last_month, "
            "this_quarter, last_quarter, this_year, last_year"
        )

    return start.isoformat(), end.isoformat()


@mcp.tool()
def worklist(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    period: Optional[str] = None,    # NEW: friendly time windows
    status: str = "pending",         # "pending" | "unsigned" | "signed" | "all"
    modality: Optional[str] = None,  # optional filter (ex: "CT", "US", "CR", "MR")
    site: Optional[str] = None,      # optional filter (best-effort, depends on schema)
    limit: int = 100,
    use_snapshot_anchor: bool = True # for DB dumps: anchor period to MAX(visit_start)
) -> Dict[str, Any]:
    """
    Radiologist worklist (smart to-do list) for a date range OR friendly period.

    Simple meaning:
    - You can pass:
        A) explicit dates: date_from="YYYY-MM-DD", date_to="YYYY-MM-DD"
        OR
        B) a period: "today", "last_10_days", "last_month", "last_year", etc.
    - You choose which bucket you want:
        * pending  -> no report exists yet
        * unsigned -> report exists but not signed
        * signed   -> report exists and is signed
        * all      -> everything
    - Optional: filter by modality (CT/US/CR/MR) and site

    How it works:
    - Pull visits in the date range
    - Join patients for name / PID
    - Determine report status:
        * PENDING  : visit has no report (best effort; see schema_notes)
        * UNSIGNED : report exists but no signature found
        * SIGNED   : signature exists

    Notes:
    - Works even if dictations are empty.
    - For DB dump demos, set use_snapshot_anchor=True (default) so "last_10_days"
      uses the latest date in your dataset, not today's real date.

    Args:
        date_from (str|None): "YYYY-MM-DD"
        date_to   (str|None): "YYYY-MM-DD"
        period (str|None): today | yesterday | last_10_days | last_month | last_year | ...
        status (str): pending | unsigned | signed | all
        modality (str|None): Optional modality filter like "CT", "US", "CR", "MR"
        site (str|None): Optional site/machine filter (schema-dependent)
        limit (int): Max results (default 100, max 500)
        use_snapshot_anchor (bool): If True, anchor period to MAX(visits.visit_start)

    Returns:
        dict: { ok, filters, row_count, rows, schema_notes }
    """
    # -----------------------------
    # Validate/normalize inputs
    # -----------------------------
    limit = max(1, min(int(limit), 500))
    status_norm = (status or "pending").strip().lower()
    if status_norm not in {"pending", "unsigned", "signed", "all"}:
        return {"ok": False, "error": "status must be one of: pending, unsigned, signed, all"}

    modality_norm = (modality or "").strip().upper() or None
    site_norm = (site or "").strip() or None

    cnx = get_conn()
    try:
        # -----------------------------
        # Discover columns (schema-safe)
        # -----------------------------
        v_cols = _get_columns(cnx, "visits")
        p_cols = _get_columns(cnx, "patients")
        r_cols = _get_columns(cnx, "reports")
        s_cols = _get_columns(cnx, "signatures")

        # Visits essentials
        v_patient_fk = _pick_col(v_cols, ["patient_id", "pat_id", "person_id", "pat_person_nbr"])
        v_visit_no = _pick_col(v_cols, ["visit_no", "visit_id", "id", "appt_id", "encounter_nbr"])
        v_start = _pick_col(v_cols, ["visit_start", "visit_date", "appt_date", "created_at"])
        v_end = _pick_col(v_cols, ["visit_end"])
        v_exam_code = _pick_col(v_cols, ["exam_code"])
        v_exam_desc = _pick_col(v_cols, ["exam_description", "procedure", "exam_desc"])
        v_order_status = _pick_col(v_cols, ["order_status", "status", "visit_status"])
        v_machine = _pick_col(v_cols, ["machine", "site", "location", "facility"])
        v_report_id = _pick_col(v_cols, ["report_id", "report_fk", "reportid"])

        # Patients essentials
        p_pk = _pick_col(p_cols, ["id", "patient_id"])
        p_pid = _pick_col(p_cols, ["pid", "mrn", "medical_record_number"])
        p_fname = _pick_col(p_cols, ["first_name", "firstname", "given_name"])
        p_lname = _pick_col(p_cols, ["last_name", "lastname", "family_name", "surname"])

        if not (v_patient_fk and p_pk and v_start):
            return {
                "ok": False,
                "error": "Could not link visits to patients or find visit date column.",
                "schema_notes": {
                    "visits_patient_fk": v_patient_fk,
                    "patients_pk": p_pk,
                    "visits_start_col": v_start,
                },
            }

        # -----------------------------
        # Resolve date range
        # -----------------------------
        snapshot_anchor = None
        if period:
            if use_snapshot_anchor:
                # For DB dumps: anchor "last_10_days" etc to latest visit date in your data.
                cur = cnx.cursor(dictionary=True)
                cur.execute(f"SELECT MAX(`{v_start}`) AS max_dt FROM visits;")
                row = cur.fetchone()
                cur.close()
                if row and row.get("max_dt"):
                    max_dt = row["max_dt"]
                    # max_dt might be datetime/date/str; normalize
                    if isinstance(max_dt, datetime):
                        snapshot_anchor = max_dt.date()
                    elif isinstance(max_dt, date):
                        snapshot_anchor = max_dt
                    else:
                        snapshot_anchor = datetime.fromisoformat(str(max_dt)).date()

            try:
                date_from, date_to = _resolve_period_to_range(period, anchor=snapshot_anchor)
            except ValueError as e:
                return {"ok": False, "error": str(e)}

        if not date_from or not date_to:
            return {"ok": False, "error": "Provide either period OR both date_from and date_to."}

        # -----------------------------
        # Reports/signatures linkage (best effort)
        # -----------------------------
        r_pk = _pick_col(r_cols, ["id", "report_id"])
        s_report_fk = _pick_col(s_cols, ["report_id", "report_fk", "reportid"])
        r_visit_fk = _pick_col(r_cols, ["visit_no", "visit_id", "appt_id", "encounter_nbr"])
        s_visit_fk = _pick_col(s_cols, ["visit_no", "visit_id", "appt_id", "encounter_nbr"])

        report_join = ""
        report_has_expr = "0"
        sig_join = ""
        sig_has_expr = "0"

        # Report exists?
        if v_report_id and r_pk:
            # visits.report_id -> reports.id
            report_join = f"LEFT JOIN reports r ON r.`{r_pk}` = v.`{v_report_id}`"
            report_has_expr = (
                f"CASE WHEN v.`{v_report_id}` IS NOT NULL AND v.`{v_report_id}` <> '' "
                f"AND r.`{r_pk}` IS NOT NULL THEN 1 ELSE 0 END"
            )
        elif v_visit_no and r_visit_fk:
            # visits.visit_no -> reports.visit_no
            report_join = f"LEFT JOIN reports r ON r.`{r_visit_fk}` = v.`{v_visit_no}`"
            report_has_expr = f"CASE WHEN r.`{r_visit_fk}` IS NOT NULL THEN 1 ELSE 0 END"

        # Signed?
        if s_report_fk and v_report_id:
            # signatures.report_id -> visits.report_id
            sig_join = f"LEFT JOIN signatures s ON s.`{s_report_fk}` = v.`{v_report_id}`"
            sig_has_expr = f"CASE WHEN s.`{s_report_fk}` IS NOT NULL THEN 1 ELSE 0 END"
        elif s_report_fk and r_pk and report_join:
            # signatures.report_id -> reports.id
            sig_join = f"LEFT JOIN signatures s ON s.`{s_report_fk}` = r.`{r_pk}`"
            sig_has_expr = f"CASE WHEN s.`{s_report_fk}` IS NOT NULL THEN 1 ELSE 0 END"
        elif s_visit_fk and v_visit_no:
            # signatures.visit_no -> visits.visit_no
            sig_join = f"LEFT JOIN signatures s ON s.`{s_visit_fk}` = v.`{v_visit_no}`"
            sig_has_expr = f"CASE WHEN s.`{s_visit_fk}` IS NOT NULL THEN 1 ELSE 0 END"

        # -----------------------------
        # Build WHERE filters
        # -----------------------------
        where = [f"DATE(v.`{v_start}`) BETWEEN %s AND %s"]
        params: List[Any] = [date_from, date_to]

        if site_norm and v_machine:
            where.append(f"v.`{v_machine}` LIKE %s")
            params.append(f"%{site_norm}%")

        if modality_norm:
            modality_clauses = []
            if v_visit_no:
                # Often visit_no ends like 4762672-CR -> modality = CR
                modality_clauses.append(f"UPPER(SUBSTRING_INDEX(v.`{v_visit_no}`, '-', -1)) = %s")
                params.append(modality_norm)
            if v_exam_code:
                # Fallback: exam_code contains modality somewhere
                modality_clauses.append(f"UPPER(v.`{v_exam_code}`) LIKE %s")
                params.append(f"%{modality_norm}%")
            if modality_clauses:
                where.append("(" + " OR ".join(modality_clauses) + ")")

        # Status bucket filter
        if status_norm == "pending":
            where.append(f"({report_has_expr}) = 0")
        elif status_norm == "unsigned":
            where.append(f"({report_has_expr}) = 1 AND ({sig_has_expr}) = 0")
        elif status_norm == "signed":
            where.append(f"({report_has_expr}) = 1 AND ({sig_has_expr}) = 1")
        # "all" -> no extra clause

        # -----------------------------
        # Build SELECT list
        # -----------------------------
        select_cols = [f"v.`{v_start}` AS visit_start"]

        if v_visit_no:
            select_cols.append(f"v.`{v_visit_no}` AS visit_no")
        if v_end:
            select_cols.append(f"v.`{v_end}` AS visit_end")
        if v_exam_code:
            select_cols.append(f"v.`{v_exam_code}` AS exam_code")
        if v_exam_desc:
            select_cols.append(f"v.`{v_exam_desc}` AS exam_description")
        if v_order_status:
            select_cols.append(f"v.`{v_order_status}` AS order_status")
        if v_machine:
            select_cols.append(f"v.`{v_machine}` AS machine")

        select_cols.append(f"p.`{p_pk}` AS patient_id")
        if p_pid:
            select_cols.append(f"p.`{p_pid}` AS patient_pid")
        if p_fname:
            select_cols.append(f"p.`{p_fname}` AS first_name")
        if p_lname:
            select_cols.append(f"p.`{p_lname}` AS last_name")

        select_cols.append(f"{report_has_expr} AS has_report")
        select_cols.append(f"{sig_has_expr} AS is_signed")

        if v_visit_no:
            select_cols.append(f"UPPER(SUBSTRING_INDEX(v.`{v_visit_no}`, '-', -1)) AS modality_guess")
        elif v_exam_code:
            select_cols.append(f"UPPER(v.`{v_exam_code}`) AS modality_guess")
        else:
            select_cols.append("NULL AS modality_guess")

        # -----------------------------
        # Final query
        # -----------------------------
        sql = f"""
            SELECT
              {", ".join(select_cols)}
            FROM visits v
            JOIN patients p
              ON p.`{p_pk}` = v.`{v_patient_fk}`
            {report_join}
            {sig_join}
            WHERE {" AND ".join(where)}
            ORDER BY v.`{v_start}` ASC
            LIMIT {limit};
        """

        cur = cnx.cursor(dictionary=True)
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
        cur.close()

        # -----------------------------
        # Post-format output rows
        # -----------------------------
        out_rows = []
        for r in rows:
            has_report = int(r.get("has_report") or 0)
            is_signed = int(r.get("is_signed") or 0)

            if has_report == 0:
                wl_status = "PENDING"
            elif has_report == 1 and is_signed == 0:
                wl_status = "UNSIGNED"
            else:
                wl_status = "SIGNED"

            out_rows.append({
                "visit_start": str(r.get("visit_start")),
                "visit_no": r.get("visit_no"),
                "patient": {
                    "id": r.get("patient_id"),
                    "pid": r.get("patient_pid"),
                    "name": f"{(r.get('first_name') or '').strip()} {(r.get('last_name') or '').strip()}".strip()
                },
                "modality": r.get("modality_guess"),
                "exam_code": r.get("exam_code"),
                "exam_description": r.get("exam_description"),
                "order_status": r.get("order_status"),
                "worklist_status": wl_status,
                "machine_or_site": r.get("machine"),
            })

        return {
            "ok": True,
            "filters": {
                "period": period,
                "date_from": date_from,
                "date_to": date_to,
                "status": status_norm,
                "modality": modality_norm,
                "site": site_norm,
                "limit": limit,
                "use_snapshot_anchor": use_snapshot_anchor,
                "snapshot_anchor_date_used": snapshot_anchor.isoformat() if snapshot_anchor else None,
            },
            "row_count": len(out_rows),
            "rows": out_rows,
            "schema_notes": {
                "visits_patient_fk": v_patient_fk,
                "visits_visit_no": v_visit_no,
                "visits_start": v_start,
                "visits_report_id": v_report_id,
                "reports_pk": r_pk,
                "reports_visit_fk": r_visit_fk,
                "signatures_report_fk": s_report_fk,
                "signatures_visit_fk": s_visit_fk,
            },
        }

    finally:
        cnx.close()

from typing import Dict, Any, List, Optional


@mcp.tool()
def visit_case_summary(visit_id: str, include_attachments: bool = False) -> Dict[str, Any]:
    """
    Summarize a single imaging case (visit) in one response.

    Simple meaning:
    - You give a visit identifier (visit_id / visit_no / id depending on schema)
    - I return:
        * patient + visit snapshot
        * booked study details (if linkable)
        * report status (exists? signed? who/when if available)
        * missing pieces (what's still not done)

    Uses (best effort):
      visits, patients, booked_study, reports, signatures, attachments (optional)

    Args:
        visit_id (str): Visit identifier. Works with numeric id OR visit_no strings like "4762672-CR".
        include_attachments (bool): If True, return a small attachment summary (count + few filenames) if linkable.

    Returns:
        dict: { ok, found, patient, visit, booked_study, report, signature, missing, schema_notes }
    """
    if not visit_id or not str(visit_id).strip():
        return {"ok": False, "error": "visit_id is required."}

    q = str(visit_id).strip()
    cnx = get_conn()
    try:
        # -----------------------------
        # Discover schema safely
        # -----------------------------
        v_cols = _get_columns(cnx, "visits")
        p_cols = _get_columns(cnx, "patients")
        b_cols = _get_columns(cnx, "booked_study")
        r_cols = _get_columns(cnx, "reports")
        s_cols = _get_columns(cnx, "signatures")
        a_cols = _get_columns(cnx, "attachments")

        # visits keys/fields
        v_pk = _pick_col(v_cols, ["id", "visit_id"])
        v_visit_no = _pick_col(v_cols, ["visit_no", "encounter_nbr", "appt_id"])
        v_patient_fk = _pick_col(v_cols, ["patient_id", "pat_id", "person_id", "pat_person_nbr"])
        v_start = _pick_col(v_cols, ["visit_start", "visit_date", "appt_date", "created_at"])
        v_end = _pick_col(v_cols, ["visit_end"])
        v_exam_code = _pick_col(v_cols, ["exam_code"])
        v_exam_desc = _pick_col(v_cols, ["exam_description", "procedure", "exam_desc"])
        v_order_status = _pick_col(v_cols, ["order_status", "status", "visit_status"])
        v_machine = _pick_col(v_cols, ["machine", "site", "location", "facility"])
        v_referrer = _pick_col(v_cols, ["referrer", "referring_physician", "referrer_name"])
        v_report_id = _pick_col(v_cols, ["report_id", "report_fk", "reportid"])

        # patients fields
        p_pk = _pick_col(p_cols, ["id", "patient_id"])
        p_pid = _pick_col(p_cols, ["pid", "mrn", "medical_record_number"])
        p_fname = _pick_col(p_cols, ["first_name", "firstname", "given_name"])
        p_lname = _pick_col(p_cols, ["last_name", "lastname", "family_name", "surname"])
        p_dob = _pick_col(p_cols, ["dob", "dob_old", "date_of_birth", "birth_date"])
        p_sex = _pick_col(p_cols, ["sex", "gender"])

        if not v_patient_fk or not p_pk:
            return {"ok": False, "error": "Cannot link visits to patients (missing FK/PK)."}

        # -----------------------------
        # Load visit row (accept id or visit_no)
        # -----------------------------
        where = []
        params: List[Any] = []

        # numeric?
        if q.isdigit() and v_pk:
            where.append(f"v.`{v_pk}` = %s")
            params.append(int(q))

        # visit_no match
        if v_visit_no:
            where.append(f"v.`{v_visit_no}` = %s")
            params.append(q)

        if not where:
            return {"ok": False, "error": "Could not determine visit identifier columns in visits table."}

        # Select visit columns
        visit_select = _select_list(
            v_cols,
            [v_pk, v_visit_no, v_patient_fk, v_start, v_end, v_exam_code, v_exam_desc, v_order_status, v_machine, v_referrer, v_report_id]
        )
        if not visit_select:
            visit_select_sql = "v.*"
        else:
            visit_select_sql = ", ".join([s.replace("`", "v.`", 1) for s in visit_select])

        # Select patient columns
        patient_select = _select_list(p_cols, [p_pk, p_pid, p_fname, p_lname, p_dob, p_sex])
        if not patient_select:
            patient_select_sql = "p.*"
        else:
            patient_select_sql = ", ".join([s.replace("`", "p.`", 1) for s in patient_select])

        sql = f"""
            SELECT
              {visit_select_sql},
              {patient_select_sql}
            FROM visits v
            JOIN patients p
              ON p.`{p_pk}` = v.`{v_patient_fk}`
            WHERE {" OR ".join(where)}
            LIMIT 1;
        """

        cur = cnx.cursor(dictionary=True)
        cur.execute(sql, tuple(params))
        row = cur.fetchone()
        cur.close()

        if not row:
            return {"ok": True, "found": False, "message": f"No visit found for '{q}'."}

        # Split visit vs patient dicts (clean output)
        # We'll reconstruct with just the detected fields for clarity.
        patient = {
            "id": row.get(p_pk),
            "pid": row.get(p_pid) if p_pid else None,
            "first_name": row.get(p_fname) if p_fname else None,
            "last_name": row.get(p_lname) if p_lname else None,
            "dob": row.get(p_dob) if p_dob else None,
            "sex": row.get(p_sex) if p_sex else None,
        }
        patient["name"] = f"{patient.get('first_name') or ''} {patient.get('last_name') or ''}".strip() or None

        visit = {
            "id": row.get(v_pk) if v_pk else None,
            "visit_no": row.get(v_visit_no) if v_visit_no else None,
            "visit_start": row.get(v_start) if v_start else None,
            "visit_end": row.get(v_end) if v_end else None,
            "exam_code": row.get(v_exam_code) if v_exam_code else None,
            "exam_description": row.get(v_exam_desc) if v_exam_desc else None,
            "order_status": row.get(v_order_status) if v_order_status else None,
            "machine_or_site": row.get(v_machine) if v_machine else None,
            "referrer": row.get(v_referrer) if v_referrer else None,
            "report_id": row.get(v_report_id) if v_report_id else None,
        }

        # -----------------------------
        # Booked study details (best effort)
        # -----------------------------
        booked = None
        booked_notes = None

        if b_cols:
            b_patient_fk = _pick_col(b_cols, ["patient_id", "pat_id", "person_id"])
            b_visit_fk = _pick_col(b_cols, ["visit_id", "visit_no", "appt_id", "encounter_nbr"])
            b_pk = _pick_col(b_cols, ["id", "booked_study_id"])
            b_modality = _pick_col(b_cols, ["modality", "study_modality"])
            b_desc = _pick_col(b_cols, ["study_description", "description", "exam_description"])
            b_code = _pick_col(b_cols, ["study_code", "exam_code"])
            b_sched = _pick_col(b_cols, ["scheduled_time", "scheduled_date", "start_time", "booked_time", "created_at"])

            booked_where = []
            booked_params: List[Any] = []

            if b_visit_fk and visit.get("visit_no"):
                booked_where.append(f"b.`{b_visit_fk}` = %s")
                booked_params.append(visit.get("visit_no"))
            if b_patient_fk and patient.get("id") is not None:
                booked_where.append(f"b.`{b_patient_fk}` = %s")
                booked_params.append(patient.get("id"))

            if booked_where:
                b_select = _select_list(b_cols, [b_pk, b_visit_fk, b_patient_fk, b_modality, b_code, b_desc, b_sched])
                b_select_sql = ", ".join([s.replace("`", "b.`", 1) for s in b_select]) if b_select else "b.*"

                b_sql = f"""
                    SELECT {b_select_sql}
                    FROM booked_study b
                    WHERE {" OR ".join(booked_where)}
                    ORDER BY {f"b.`{b_sched}` DESC" if b_sched else "b.`" + (b_pk or "id") + "` DESC"}
                    LIMIT 1;
                """
                cur = cnx.cursor(dictionary=True)
                cur.execute(b_sql, tuple(booked_params))
                booked = cur.fetchone()
                cur.close()
            else:
                booked_notes = "Could not link booked_study to this visit (missing link columns)."

        # -----------------------------
        # Report details (best effort)
        # -----------------------------
        report = {"exists": False}
        signature = {"exists": False}
        missing: List[str] = []

        # Find report row
        report_row = None
        if r_cols:
            r_pk = _pick_col(r_cols, ["id", "report_id"])
            r_visit_fk = _pick_col(r_cols, ["visit_no", "visit_id", "appt_id", "encounter_nbr"])
            r_status = _pick_col(r_cols, ["status", "report_status", "final_status", "status_desc"])
            r_created = _pick_col(r_cols, ["created_at", "create_date", "report_date", "dt_created"])
            r_signed = _pick_col(r_cols, ["signed_at", "finalized_at", "completed_at", "dt_signed"])
            r_text = _pick_col(r_cols, ["report_text", "text", "body", "content"])

            r_where = []
            r_params: List[Any] = []

            if v_report_id and visit.get("report_id") and r_pk:
                r_where.append(f"r.`{r_pk}` = %s")
                r_params.append(visit.get("report_id"))

            if not r_where and v_visit_no and visit.get("visit_no") and r_visit_fk:
                r_where.append(f"r.`{r_visit_fk}` = %s")
                r_params.append(visit.get("visit_no"))

            if r_where:
                r_select_cols = [c for c in [r_pk, r_visit_fk, r_status, r_created, r_signed] if c and c in r_cols]
                # Don't return full report text; just show a small preview if present
                if r_text:
                    r_select_cols.append(r_text)
                if not r_select_cols:
                    r_select_cols = [r_pk] if r_pk else ["*"]

                r_select_sql = ", ".join([f"r.`{c}`" for c in r_select_cols]) if r_select_cols != ["*"] else "r.*"

                r_sql = f"""
                    SELECT {r_select_sql}
                    FROM reports r
                    WHERE {" OR ".join(r_where)}
                    ORDER BY {f"r.`{r_created}` DESC" if r_created else (f"r.`{r_pk}` DESC" if r_pk else "1")}
                    LIMIT 1;
                """
                cur = cnx.cursor(dictionary=True)
                cur.execute(r_sql, tuple(r_params))
                report_row = cur.fetchone()
                cur.close()

            if report_row:
                report["exists"] = True
                report["id"] = report_row.get(r_pk) if r_pk else None
                report["status"] = report_row.get(r_status) if r_status else None
                report["created_at"] = report_row.get(r_created) if r_created else None
                report["signed_at"] = report_row.get(r_signed) if r_signed else None
                if r_text and report_row.get(r_text):
                    txt = str(report_row.get(r_text))
                    report["text_preview"] = (txt[:300] + "...") if len(txt) > 300 else txt

        if not report.get("exists"):
            missing.append("No report found yet for this visit.")

        # -----------------------------
        # Signature details (best effort)
        # -----------------------------
        sig_row = None
        if s_cols and report.get("exists"):
            s_pk = _pick_col(s_cols, ["id", "signature_id"])
            s_report_fk = _pick_col(s_cols, ["report_id", "report_fk", "reportid"])
            s_signed_by = _pick_col(s_cols, ["signed_by", "user_id", "signed_by_user", "signed_by_username"])
            s_signed_at = _pick_col(s_cols, ["signed_at", "created_at", "dt_signed", "signature_time"])
            s_status = _pick_col(s_cols, ["status", "signature_status"])
            s_visit_fk = _pick_col(s_cols, ["visit_no", "visit_id", "appt_id", "encounter_nbr"])

            s_where = []
            s_params: List[Any] = []

            if s_report_fk and report.get("id") is not None:
                s_where.append(f"s.`{s_report_fk}` = %s")
                s_params.append(report.get("id"))
            elif s_visit_fk and visit.get("visit_no"):
                s_where.append(f"s.`{s_visit_fk}` = %s")
                s_params.append(visit.get("visit_no"))

            if s_where:
                s_select_cols = [c for c in [s_pk, s_report_fk, s_signed_by, s_signed_at, s_status] if c and c in s_cols]
                s_select_sql = ", ".join([f"s.`{c}`" for c in s_select_cols]) if s_select_cols else "s.*"

                s_sql = f"""
                    SELECT {s_select_sql}
                    FROM signatures s
                    WHERE {" OR ".join(s_where)}
                    ORDER BY {f"s.`{s_signed_at}` DESC" if s_signed_at else (f"s.`{s_pk}` DESC" if s_pk else "1")}
                    LIMIT 1;
                """
                cur = cnx.cursor(dictionary=True)
                cur.execute(s_sql, tuple(s_params))
                sig_row = cur.fetchone()
                cur.close()

            if sig_row:
                signature["exists"] = True
                signature["signed_by"] = sig_row.get(s_signed_by) if s_signed_by else None
                signature["signed_at"] = sig_row.get(s_signed_at) if s_signed_at else None
                signature["status"] = sig_row.get(s_status) if s_status else None
            else:
                missing.append("Report exists but is not signed yet.")

        # If there is a report but signatures table is not linkable, call it out
        if report.get("exists") and not s_cols:
            missing.append("Report exists, but signatures table is unavailable/empty in this dataset.")

        # -----------------------------
        # Attachments (optional, best effort)
        # -----------------------------
        attachments_summary = None
        if include_attachments and a_cols:
            a_pk = _pick_col(a_cols, ["id", "attachment_id"])
            a_report_fk = _pick_col(a_cols, ["report_id", "report_fk", "reportid"])
            a_visit_fk = _pick_col(a_cols, ["visit_id", "visit_no", "appt_id", "encounter_nbr"])
            a_filename = _pick_col(a_cols, ["filename", "file_name", "name"])
            a_created = _pick_col(a_cols, ["created_at", "create_date", "uploaded_at"])

            a_where = []
            a_params: List[Any] = []

            if a_report_fk and report.get("id") is not None:
                a_where.append(f"a.`{a_report_fk}` = %s")
                a_params.append(report.get("id"))
            elif a_visit_fk and visit.get("visit_no"):
                a_where.append(f"a.`{a_visit_fk}` = %s")
                a_params.append(visit.get("visit_no"))

            if a_where:
                a_select_cols = [c for c in [a_pk, a_filename, a_created] if c and c in a_cols]
                a_select_sql = ", ".join([f"a.`{c}`" for c in a_select_cols]) if a_select_cols else "a.*"

                a_sql = f"""
                    SELECT {a_select_sql}
                    FROM attachments a
                    WHERE {" OR ".join(a_where)}
                    ORDER BY {f"a.`{a_created}` DESC" if a_created else (f"a.`{a_pk}` DESC" if a_pk else "1")}
                    LIMIT 10;
                """
                cur = cnx.cursor(dictionary=True)
                cur.execute(a_sql, tuple(a_params))
                a_rows = cur.fetchall()
                cur.close()

                attachments_summary = {
                    "count_returned": len(a_rows),
                    "items": [
                        {
                            "id": r.get(a_pk) if a_pk else None,
                            "filename": r.get(a_filename) if a_filename else None,
                            "created_at": r.get(a_created) if a_created else None,
                        }
                        for r in a_rows
                    ],
                }

        # -----------------------------
        # Build friendly status
        # -----------------------------
        case_status = "PENDING"
        if report.get("exists") and not signature.get("exists"):
            case_status = "UNSIGNED"
        if report.get("exists") and signature.get("exists"):
            case_status = "SIGNED"

        return {
            "ok": True,
            "found": True,
            "case_status": case_status,
            "patient": patient,
            "visit": visit,
            "booked_study": booked,
            "booked_study_note": booked_notes,
            "report": report,
            "signature": signature,
            "attachments": attachments_summary,
            "missing_pieces": missing,
            "schema_notes": {
                "visits_pk": v_pk,
                "visits_visit_no": v_visit_no,
                "visits_patient_fk": v_patient_fk,
                "visits_report_id": v_report_id,
                "reports_pk": _pick_col(r_cols, ["id", "report_id"]) if r_cols else None,
                "reports_visit_fk": _pick_col(r_cols, ["visit_no", "visit_id", "appt_id", "encounter_nbr"]) if r_cols else None,
                "signatures_report_fk": _pick_col(s_cols, ["report_id", "report_fk", "reportid"]) if s_cols else None,
                "attachments_report_fk": _pick_col(a_cols, ["report_id", "report_fk", "reportid"]) if a_cols else None,
            },
        }

    finally:
        cnx.close()

import re
from html import unescape
from typing import Dict, Any, List

def _strip_html_to_text(html: str) -> str:
    if html is None:
        return ""
    txt = unescape(str(html))
    txt = re.sub(r"<br\s*/?>", "\n", txt, flags=re.I)
    txt = re.sub(r"</p\s*>", "\n", txt, flags=re.I)
    txt = re.sub(r"</div\s*>", "\n", txt, flags=re.I)
    txt = re.sub(r"<[^>]+>", "", txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def _find_repeated_words(text: str) -> List[Dict[str, Any]]:
    issues = []
    for m in re.finditer(r"\b(\w+)\s+\1\b", text, flags=re.I):
        snippet = text[max(0, m.start()-30): min(len(text), m.end()+30)]
        issues.append({"type": "repeated_word", "message": f"Repeated word '{m.group(1)} {m.group(1)}'", "context": snippet})
    return issues

def _find_spacing_issues(text: str) -> List[Dict[str, Any]]:
    issues = []
    if "  " in text:
        issues.append({"type": "double_space", "message": "Extra spaces found (double spaces)."})
    # missing space after punctuation like ".There" or ",the"
    for m in re.finditer(r"([.,;:])([A-Za-z])", text):
        snippet = text[max(0, m.start()-25): min(len(text), m.end()+25)]
        issues.append({"type": "missing_space", "message": "Missing space after punctuation", "context": snippet})
        if len(issues) >= 10:
            break
    return issues

def _heading_checks(text: str) -> List[Dict[str, Any]]:
    issues = []
    expected = ["HISTORY", "INDICATION", "TECHNIQUE", "COMPARISON", "FINDINGS", "REPORT", "IMPRESSION", "CONCLUSION"]
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if len(s) <= 45 and re.fullmatch(r"[A-Z0-9 /\-&]+:?", s):
            if any(k == s.upper().rstrip(":") for k in expected) and not s.endswith(":"):
                issues.append({"type": "heading_format", "message": f"Consider adding ':' to heading '{s}'"})
    return issues

def _apply_light_cleanup(text: str) -> str:
    t = text
    t = re.sub(r"\b(\w+)\s+\1\b", r"\1", t, flags=re.I)     # repeated words
    t = re.sub(r"[ \t]{2,}", " ", t)                        # extra spaces
    t = re.sub(r"([.,;:])([A-Za-z])", r"\1 \2", t)          # missing space after punctuation
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t

@mcp.tool()
def report_language_review_v2(report_id: int, max_chars: int = 4000) -> Dict[str, Any]:
    """
    Review report language using reports.report_text (HTML).

    Returns:
    - extracted plain text (truncated)
    - structured issues list
    - cleaned_text (truncated)

    Args:
        report_id (int): reports.report_id
        max_chars (int): truncate returned text to keep responses small
    """
    cnx = None
    try:
        max_chars = max(500, min(int(max_chars), 12000))

        cnx = get_conn()
        cur = cnx.cursor(dictionary=True)
        cur.execute(
            """
            SELECT report_id, visit_no, exam_description, order_status, report_text
            FROM reports
            WHERE report_id = %s
            LIMIT 1;
            """,
            (int(report_id),),
        )
        row = cur.fetchone()
        cur.close()

        if not row:
            return {"ok": False, "error": f"No report found for report_id={report_id}"}

        raw = row.get("report_text") or ""
        if not raw.strip():
            return {"ok": True, "report_id": int(report_id), "has_text": False, "message": "report_text is empty."}

        plain = _strip_html_to_text(raw)

        issues: List[Dict[str, Any]] = []
        issues += _find_repeated_words(plain)
        issues += _find_spacing_issues(plain)
        issues += _heading_checks(plain)

        cleaned = _apply_light_cleanup(plain)

        return {
            "ok": True,
            "report_id": int(report_id),
            "context": {
                "visit_no": row.get("visit_no"),
                "exam_description": row.get("exam_description"),
                "order_status": row.get("order_status"),
            },
            "plain_text_preview": plain[:max_chars] + ("..." if len(plain) > max_chars else ""),
            "issue_count": len(issues),
            "issues": issues[:50],
            "cleaned_text_preview": cleaned[:max_chars] + ("..." if len(cleaned) > max_chars else ""),
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}

    finally:
        try:
            if cnx:
                cnx.close()
        except Exception:
            pass

if __name__ == "__main__":
    mcp.run()
