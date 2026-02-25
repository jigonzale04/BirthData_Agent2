# =========================================
# app.py — Provisional Natality Dashboard + Analytics Copilot
# GitHub -> Streamlit Cloud
# LLM: Groq + llama-3.1-8b-instant
# =========================================

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(layout="wide", page_title="Provisional Natality Dashboard")

# =========================================
# CONSTANTS
# =========================================
DATA_FILE = "Provisional_Natality_2025_CDC.csv"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"

ALLOWED_COLUMNS = [
    "state_of_residence",
    "month",
    "month_code",
    "year_code",
    "sex_of_infant",
    "births",
]

# Operations supported by the execution engine
SUPPORTED_OPERATIONS = {
    "aggregate",
    "rank",
    "trend",
    "percentage",
    "compare",
    "filter",
    "list",
}

SUPPORTED_AGGS = {"sum", "mean", "count"}


# =========================================
# HELPERS
# =========================================
def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Groq responses should be JSON-only per prompt, but models sometimes wrap in fences.
    This extracts the first JSON object found.
    """
    text = text.strip()
    # remove markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # try direct
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # extract first {...}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("Could not find a JSON object in the model output.")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object.")
    return obj


def _coerce_to_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, str):
        return [x]
    return [str(x)]


def _validate_instruction(inst: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate and normalize the instruction dict. Returns (clean_inst, warnings).
    """
    warnings: List[str] = []

    op = str(inst.get("operation", "")).strip().lower()
    if op not in SUPPORTED_OPERATIONS:
        # default to aggregate
        warnings.append(f"Unsupported or missing operation '{op}'. Defaulting to 'aggregate'.")
        op = "aggregate"

    metric = str(inst.get("metric", "births")).strip().lower()
    if metric not in ALLOWED_COLUMNS:
        warnings.append(f"Unknown metric '{metric}'. Defaulting to 'births'.")
        metric = "births"

    agg = str(inst.get("aggregation", "sum")).strip().lower()
    if agg not in SUPPORTED_AGGS:
        warnings.append(f"Unsupported aggregation '{agg}'. Defaulting to 'sum'.")
        agg = "sum"

    group_by = _coerce_to_list(inst.get("group_by", []))
    group_by = [g.strip().lower() for g in group_by if str(g).strip() != ""]
    for g in list(group_by):
        if g not in ALLOWED_COLUMNS:
            warnings.append(f"Removed unknown group_by column '{g}'.")
            group_by.remove(g)

    filters = inst.get("filters", {})
    if filters is None:
        filters = {}
    if not isinstance(filters, dict):
        warnings.append("filters must be an object; ignoring invalid filters.")
        filters = {}
    else:
        # normalize filter keys; keep values as-is
        norm_filters: Dict[str, Any] = {}
        for k, v in filters.items():
            kk = str(k).strip().lower()
            if kk in ALLOWED_COLUMNS:
                norm_filters[kk] = v
            else:
                warnings.append(f"Ignored unknown filter column '{k}'.")
        filters = norm_filters

    # rank options
    ranking = inst.get("ranking", {}) or {}
    top_n = ranking.get("top_n", 5)
    try:
        top_n = int(top_n)
    except Exception:
        top_n = 5
    top_n = max(1, min(top_n, 50))

    order = str(ranking.get("order", "desc")).strip().lower()
    if order not in {"asc", "desc"}:
        order = "desc"

    # compare options (optional)
    comparison = inst.get("comparison", {}) or {}
    comp_type = str(comparison.get("type", "difference")).strip().lower()
    if comp_type not in {"difference", "percentage_change", "ratio"}:
        comp_type = "difference"

    # periods/months for compare (optional)
    periods = _coerce_to_list(comparison.get("periods", []))
    periods = [str(p).strip() for p in periods if str(p).strip() != ""]

    clean = {
        "operation": op,
        "group_by": group_by,
        "filters": filters,
        "metric": metric,
        "aggregation": agg,
        "ranking": {"top_n": top_n, "order": order},
        "comparison": {"type": comp_type, "periods": periods},
    }
    return clean, warnings


def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    out = df
    for col, val in filters.items():
        # allow list-of-values or single value
        if isinstance(val, list):
            out = out[out[col].astype(str).isin([str(v) for v in val])]
        else:
            out = out[out[col].astype(str) == str(val)]
    return out


def _sort_months(df_month: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts month-based results using month_code if present; otherwise lexical.
    Assumes df_month has 'month' column and optionally 'month_code'.
    """
    if "month_code" in df_month.columns and df_month["month_code"].notna().any():
        # month_code might not be numeric
        tmp = df_month.copy()
        tmp["month_code__"] = pd.to_numeric(tmp["month_code"], errors="coerce")
        if tmp["month_code__"].notna().any():
            return tmp.sort_values(["month_code__", "month"]).drop(columns=["month_code__"])
    return df_month.sort_values(["month"])


def execute_query(df: pd.DataFrame, inst: Dict[str, Any]) -> pd.DataFrame:
    """
    Deterministic execution engine: returns a DataFrame result.
    """
    df_work = df.copy()
    df_work = _apply_filters(df_work, inst["filters"])

    op = inst["operation"]
    metric = inst["metric"]
    agg = inst["aggregation"]
    group_by = inst["group_by"]

    # A small guard: if user asks aggregate but provides no group_by,
    # return a single-row total.
    if op == "aggregate":
        if group_by:
            res = (
                df_work.groupby(group_by, dropna=False)[metric]
                .agg(agg)
                .reset_index()
            )
        else:
            val = getattr(df_work[metric], agg)() if agg != "count" else df_work[metric].count()
            res = pd.DataFrame({metric: [val]})
        return res

    if op == "rank":
        # rank is aggregate + sort + top_n
        gb = group_by if group_by else ["state_of_residence"]
        res = (
            df_work.groupby(gb, dropna=False)[metric]
            .sum()
            .reset_index()
            .rename(columns={metric: "value"})
        )
        asc = inst["ranking"]["order"] == "asc"
        res = res.sort_values("value", ascending=asc).head(inst["ranking"]["top_n"])
        return res

    if op == "trend":
        # monthly trend default: month
        gb = group_by if group_by else ["month"]
        # If user includes month in group_by, keep it; else use month
        if "month" not in gb:
            gb = ["month"] + gb
        res = (
            df_work.groupby(gb, dropna=False)[metric]
            .sum()
            .reset_index()
            .rename(columns={metric: "value"})
        )
        # attempt to attach month_code if month present
        if "month" in res.columns and "month_code" in df_work.columns and "month_code" not in res.columns:
            mcode = (
                df_work[["month", "month_code"]]
                .dropna(subset=["month"])
                .drop_duplicates(subset=["month"])
            )
            res = res.merge(mcode, on="month", how="left")
        if "month" in res.columns:
            res = _sort_months(res)
        return res

    if op == "percentage":
        # pct change over month for a given filter set; returns month, value, pct_change
        base = execute_query(df_work, {"operation": "trend", "group_by": ["month"], "filters": {}, "metric": metric,
                                       "aggregation": "sum", "ranking": {"top_n": 5, "order": "desc"},
                                       "comparison": {"type": "difference", "periods": []}})
        base = base.copy()
        base["pct_change"] = base["value"].pct_change() * 100.0
        return base

    if op == "compare":
        # compare across two periods (usually months). If periods missing, compare top 2 months by value.
        comp = inst["comparison"]
        periods = comp.get("periods", []) or []

        # Build monthly totals (or by group_by if user specified)
        # For compare, it often makes sense to compare by state or gender etc; we’ll honor group_by if provided
        gb = group_by if group_by else ["month"]
        if "month" not in gb:
            gb = ["month"] + gb

        base = (
            df_work.groupby(gb, dropna=False)[metric]
            .sum()
            .reset_index()
            .rename(columns={metric: "value"})
        )

        if "month" in base.columns and "month_code" in df_work.columns and "month_code" not in base.columns:
            mcode = (
                df_work[["month", "month_code"]]
                .dropna(subset=["month"])
                .drop_duplicates(subset=["month"])
            )
            base = base.merge(mcode, on="month", how="left")

        # decide periods
        all_months = base["month"].astype(str).dropna().unique().tolist() if "month" in base.columns else []
        if len(periods) < 2:
            # pick top 2 months by total value (aggregated over other dims)
            if "month" in base.columns:
                month_tot = base.groupby("month")["value"].sum().sort_values(ascending=False)
                periods = month_tot.head(2).index.astype(str).tolist()
        # filter to selected periods
        if "month" in base.columns and periods:
            base = base[base["month"].astype(str).isin([str(p) for p in periods])]

        # pivot on month for comparison
        if "month" not in base.columns:
            return pd.DataFrame({"error": ["Compare requires 'month' in data."]})

        other_dims = [c for c in base.columns if c not in {"month", "value", "month_code"}]
        if not other_dims:
            # total compare
            pivot = base.groupby("month")["value"].sum().reset_index()
            pivot = pivot.pivot(index=None, columns="month", values="value")
            # pivot yields 1 row
            pivot = pivot.reset_index(drop=True)
        else:
            pivot = base.pivot_table(index=other_dims, columns="month", values="value", aggfunc="sum").reset_index()

        # compute comparison metric
        cols = [c for c in pivot.columns if c not in other_dims]
        if len(cols) >= 2:
            a, b = cols[0], cols[1]
            ctype = comp.get("type", "difference")
            if ctype == "difference":
                pivot["difference"] = pivot[b] - pivot[a]
            elif ctype == "percentage_change":
                pivot["pct_change"] = (pivot[b] - pivot[a]) / pivot[a].replace({0: pd.NA}) * 100.0
            elif ctype == "ratio":
                pivot["ratio"] = pivot[b] / pivot[a].replace({0: pd.NA})
        return pivot

    if op == "filter":
        return df_work

    if op == "list":
        return df_work.head(50)

    # fallback
    return df_work.head(50)


def groq_chat(api_key: str, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 400) -> str:
    r = requests.post(
        GROQ_CHAT_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Groq API Error {r.status_code}: {r.text}")
    return r.json()["choices"][0]["message"]["content"]


# =========================================
# LOAD DATA
# =========================================
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df0 = pd.read_csv(DATA_FILE)
    df0 = _norm_cols(df0)

    # require columns
    missing = [c for c in ALLOWED_COLUMNS if c not in df0.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df0["births"] = pd.to_numeric(df0["births"], errors="coerce")
    df0 = df0.dropna(subset=["births"]).copy()

    # normalize key categorical columns to strings
    for c in ["state_of_residence", "month", "sex_of_infant"]:
        df0[c] = df0[c].astype(str)

    # month_code/year_code numeric if possible
    df0["month_code"] = pd.to_numeric(df0["month_code"], errors="coerce")
    df0["year_code"] = pd.to_numeric(df0["year_code"], errors="coerce")
    return df0


try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load dataset '{DATA_FILE}'. Error: {e}")
    st.stop()

# =========================================
# APP HEADER
# =========================================
st.title("Provisional Natality Data Dashboard")
st.caption("Interactive dashboard + analytics copilot (Pandas execution + LLM interpretation).")

# =========================================
# SIDEBAR FILTERS
# =========================================
st.sidebar.header("Filters")

states = sorted(df["state_of_residence"].dropna().unique().tolist())
genders = sorted(df["sex_of_infant"].dropna().unique().tolist())

# month ordering using month_code when present
month_map = (
    df[["month", "month_code"]]
    .dropna(subset=["month"])
    .drop_duplicates(subset=["month"])
    .copy()
)
if month_map["month_code"].notna().any():
    month_map = month_map.sort_values(["month_code", "month"])
    months = month_map["month"].astype(str).tolist()
else:
    months = sorted(df["month"].dropna().astype(str).unique().tolist())

state_sel = st.sidebar.multiselect("State", ["All"] + states, default=["All"])
gender_sel = st.sidebar.multiselect("Gender", ["All"] + genders, default=["All"])
month_sel = st.sidebar.multiselect("Month", ["All"] + months, default=["All"])

filtered = df.copy()
if "All" not in state_sel:
    filtered = filtered[filtered["state_of_residence"].isin([str(x) for x in state_sel])]
if "All" not in gender_sel:
    filtered = filtered[filtered["sex_of_infant"].isin([str(x) for x in gender_sel])]
if "All" not in month_sel:
    filtered = filtered[filtered["month"].isin([str(x) for x in month_sel])]

if filtered.empty:
    st.warning("No data matches the selected filters. Try broadening selections.")
    st.stop()

# =========================================
# DASHBOARD: CHARTS
# =========================================
c1, c2 = st.columns([2, 1], gap="large")

with c1:
    agg = (
        filtered.groupby(["state_of_residence", "sex_of_infant"], dropna=False)["births"]
        .sum()
        .reset_index()
    )
    fig = px.bar(
        agg,
        x="state_of_residence",
        y="births",
        color="sex_of_infant",
        title="Total Births by State and Gender",
        template="plotly_white",
    )
    fig.update_layout(
        xaxis_title="State of Residence",
        yaxis_title="Births",
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title_text="Gender",
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    total_births = int(filtered["births"].sum())
    st.metric("Total Births (Filtered)", f"{total_births:,}")
    by_gender = filtered.groupby("sex_of_infant")["births"].sum().sort_values(ascending=False)
    st.write("Births by Gender (Filtered)")
    st.dataframe(by_gender.reset_index().rename(columns={"births": "total_births"}), use_container_width=True, hide_index=True)

st.subheader("Filtered Records")
st.dataframe(filtered.reset_index(drop=True), use_container_width=True, hide_index=True)

# =========================================
# ANALYTICS COPILOT
# =========================================
st.markdown("---")
st.header("Analytics Copilot")

st.caption(
    "Ask for tables, breakdowns, trends, comparisons, rankings, or explanations. "
    "The copilot converts your question into a structured plan, executes it with pandas, then provides interpretation."
)

# chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Example: 'Monthly births for Indiana' or 'Top 5 states by births' or 'Explain the trend for Texas'")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # require Groq secret
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        st.error("Missing GROQ_API_KEY in Streamlit Secrets.")
        st.stop()

    # ---------- Step 1: LLM -> Query Plan ----------
    planner_system = f"""
You are an analytics query planner.

Task:
Translate the user's request into a JSON execution plan that a pandas engine can run.

Supported operations:
- aggregate: group and compute sum/mean/count
- rank: return top/bottom N by total births
- trend: monthly time series (optionally by a group)
- percentage: month-over-month percent change (for the current filters)
- compare: compare two months (difference / percentage_change / ratio)
- filter: return all rows matching additional filters
- list: return up to 50 rows

Available columns:
{", ".join(ALLOWED_COLUMNS)}

Rules:
- Return ONLY valid JSON. No commentary, no markdown.
- Use "births" as metric unless user requests count or mean.
- When user asks "total", prefer operation "aggregate" with empty group_by.
- For rankings, prefer operation "rank" with group_by ["state_of_residence"] unless user specifies another dimension.
- For trends, prefer operation "trend" with group_by ["month"] unless user specifies another dimension.
- Filters must match available columns and values must be strings or lists of strings.

JSON schema:
{{
  "operation": "aggregate|rank|trend|percentage|compare|filter|list",
  "group_by": ["..."],
  "filters": {{"column": "value" or ["v1","v2"]}},
  "metric": "births",
  "aggregation": "sum|mean|count",
  "ranking": {{"top_n": 5, "order": "desc|asc"}},
  "comparison": {{"type": "difference|percentage_change|ratio", "periods": ["MonthA","MonthB"]}}
}}
""".strip()

    # Give the planner a small hint about current UI filters (so it doesn't fight them)
    filter_hint = {
        "ui_filters_applied": {
            "states": state_sel,
            "gender": gender_sel,
            "months": month_sel,
        }
    }

    try:
        plan_text = groq_chat(
            api_key,
            messages=[
                {"role": "system", "content": planner_system},
                {"role": "user", "content": f"User request: {user_input}\n\nContext: {json.dumps(filter_hint)}"},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        plan_raw = _safe_json_loads(plan_text)
        plan, plan_warnings = _validate_instruction(plan_raw)
    except Exception as e:
        with st.chat_message("assistant"):
            st.markdown(f"⚠️ I couldn't parse a valid query plan for that request.\n\n**Error:** {e}")
        st.stop()

    # Optional: show plan for teaching/debug
    with st.expander("Show query plan (for debugging/teaching)", expanded=False):
        st.code(json.dumps(plan, indent=2), language="json")
        if plan_warnings:
            st.warning("Plan warnings:\n- " + "\n- ".join(plan_warnings))

    # ---------- Step 2: Execute with Pandas ----------
    try:
        result_df = execute_query(filtered, plan)
        if result_df is None or not isinstance(result_df, pd.DataFrame):
            raise ValueError("Execution engine did not return a DataFrame.")
    except Exception as e:
        with st.chat_message("assistant"):
            st.markdown(f"⚠️ I couldn't execute that request on the data.\n\n**Error:** {e}")
        st.stop()

    # Show results (table)
    with st.chat_message("assistant"):
        st.markdown("Here’s the data result:")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

    # ---------- Optional charting based on operation ----------
    # (Keep simple and robust; charts are a bonus, not required.)
    try:
        if plan["operation"] in {"trend", "percentage"}:
            if "month" in result_df.columns and ("value" in result_df.columns):
                chart_df = result_df.copy()
                # sort months if month_code present
                if "month_code" in chart_df.columns:
                    chart_df = _sort_months(chart_df)
                fig2 = px.line(chart_df, x="month", y="value", markers=True, template="plotly_white",
                               title="Trend (Filtered Scope)")
                st.plotly_chart(fig2, use_container_width=True)

            if plan["operation"] == "percentage" and "pct_change" in result_df.columns:
                chart_df = result_df.copy()
                if "month_code" in chart_df.columns:
                    chart_df = _sort_months(chart_df)
                fig3 = px.bar(chart_df, x="month", y="pct_change", template="plotly_white",
                              title="Month-over-Month % Change (Filtered Scope)")
                st.plotly_chart(fig3, use_container_width=True)

        if plan["operation"] == "rank":
            # attempt a bar chart: first non-value col as category
            if "value" in result_df.columns:
                cat_cols = [c for c in result_df.columns if c != "value"]
                if cat_cols:
                    fig4 = px.bar(result_df, x=cat_cols[0], y="value", template="plotly_white",
                                  title="Ranking Result")
                    st.plotly_chart(fig4, use_container_width=True)
    except Exception:
        # charting is best-effort; ignore failures
        pass

    # ---------- Step 3: LLM Interpretation ----------
    # Keep interpretation token-light: send only a compact preview and summary stats
    preview_rows = min(30, len(result_df))
    preview = result_df.head(preview_rows).to_dict(orient="records")

    interpret_context = {
        "operation": plan["operation"],
        "filters_applied": {"state": state_sel, "gender": gender_sel, "month": month_sel},
        "result_preview_rows": preview_rows,
        "result_preview": preview,
        "result_shape": [int(result_df.shape[0]), int(result_df.shape[1])],
        "note": "Preview may be truncated; interpret patterns without inventing numbers not shown.",
    }

    interpret_system = """
You are a senior data analyst writing executive insights.

Rules:
- Use ONLY the provided result context.
- Do NOT fabricate numbers.
- If the preview is insufficient, state what additional cut would be needed.
- Be concise, decision-oriented, and clear.
- Output text only (no tables).
""".strip()

    try:
        insight_text = groq_chat(
            api_key,
            messages=[
                {"role": "system", "content": interpret_system},
                {"role": "user", "content": f"User question: {user_input}\n\nResult context:\n{json.dumps(interpret_context)}"},
            ],
            temperature=0.2,
            max_tokens=450,
        )
    except Exception as e:
        insight_text = f"(Interpretation unavailable: {e})"

    st.session_state.messages.append({"role": "assistant", "content": insight_text})
    with st.chat_message("assistant"):
        st.markdown(insight_text)
