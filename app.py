# =========================================
# IMPORTS (MUST BE FIRST)
# =========================================

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(layout="wide")

# =========================================
# DATA LOADING
# =========================================

@st.cache_data
def load_data():
    df = pd.read_csv("Provisional_Natality_2025_CDC.csv")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["births"] = pd.to_numeric(df["births"], errors="coerce")
    df = df.dropna(subset=["births"])
    return df

df = load_data()

# =========================================
# TITLE + FILTERS
# =========================================

st.title("Provisional Natality Data Dashboard")
st.subheader("Birth Analysis by State and Gender")

st.sidebar.header("Filters")

states = sorted(df["state_of_residence"].unique())
genders = sorted(df["sex_of_infant"].unique())
months = sorted(df["month"].unique())

state_sel = st.sidebar.multiselect("State", ["All"] + states, default=["All"])
gender_sel = st.sidebar.multiselect("Gender", ["All"] + genders, default=["All"])
month_sel = st.sidebar.multiselect("Month", ["All"] + months, default=["All"])

filtered = df.copy()

if "All" not in state_sel:
    filtered = filtered[filtered["state_of_residence"].isin(state_sel)]

if "All" not in gender_sel:
    filtered = filtered[filtered["sex_of_infant"].isin(gender_sel)]

if "All" not in month_sel:
    filtered = filtered[filtered["month"].isin(month_sel)]

# =========================================
# CHART
# =========================================

agg = (
    filtered.groupby(["state_of_residence", "sex_of_infant"])["births"]
    .sum()
    .reset_index()
)

fig = px.bar(
    agg,
    x="state_of_residence",
    y="births",
    color="sex_of_infant",
    title="Total Births by State and Gender",
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Filtered Records")
st.dataframe(filtered, use_container_width=True)

# =========================================
# AI ANALYTICS AGENT
# =========================================

st.markdown("---")
st.header("AI Data Analyst")

user_input = st.chat_input("Ask any question about the data...")

if user_input:

    st.chat_message("user").markdown(user_input)

    # -------------------------------------
    # STEP 1: LLM → STRUCTURED INSTRUCTION
    # -------------------------------------

    query_system_prompt = """
You are an analytics query translator.

Translate the user request into a JSON instruction.

Use this schema:

{
  "action": "aggregate | filter | list",
  "group_by": ["column1", "column2"],
  "filters": {"column": "value"},
  "metric": "births",
  "aggregation": "sum | mean | count"
}

Rules:
- Only return valid JSON.
- Do not explain.
- Use only these columns:
  state_of_residence, month, month_code,
  year_code, sex_of_infant, births
"""

    try:
        api_key = st.secrets["GROQ_API_KEY"]

        translation_response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": query_system_prompt},
                    {"role": "user", "content": user_input},
                ],
                "temperature": 0,
                "max_tokens": 250,
            },
            timeout=30,
        )

        instruction_text = translation_response.json()["choices"][0]["message"]["content"]

        instruction = json.loads(instruction_text)

    except Exception as e:
        st.chat_message("assistant").markdown(f"Query parsing error: {str(e)}")
        st.stop()

    # -------------------------------------
    # STEP 2: EXECUTE WITH PANDAS
    # -------------------------------------

    df_work = filtered.copy()

    # Apply filters
    if "filters" in instruction and instruction["filters"]:
        for col, val in instruction["filters"].items():
            df_work = df_work[df_work[col] == val]

    if instruction["action"] == "aggregate":

        result = (
            df_work
            .groupby(instruction["group_by"])[instruction["metric"]]
            .agg(instruction["aggregation"])
            .reset_index()
        )

        st.dataframe(result, use_container_width=True)

    elif instruction["action"] == "list":

        result = df_work.head(50)
        st.dataframe(result, use_container_width=True)

    elif instruction["action"] == "filter":

        result = df_work
        st.dataframe(result, use_container_width=True)

    else:
        st.chat_message("assistant").markdown("Unsupported query type.")
        st.stop()

    # -------------------------------------
    # STEP 3: OPTIONAL INTERPRETATION
    # -------------------------------------

    interpretation_prompt = f"""
You are a senior data analyst.

Provide concise executive insight based on this result:

{result.to_json()}
"""

    try:

        insight_response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": "Provide executive-level insight."},
                    {"role": "user", "content": interpretation_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 400,
            },
            timeout=30,
        )

        insight = insight_response.json()["choices"][0]["message"]["content"]

        st.chat_message("assistant").markdown(insight)

    except Exception as e:
        st.chat_message("assistant").markdown("Insight generation failed.")
