import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import re

st.markdown("---")
st.header("AI Data Analyst")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask an analytical question about the data...")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    intent = classify_intent(user_input)

    # ============================
    # PANDAS ROUTE
    # ============================
    if intent == "pandas":

        matched_state = extract_state(user_input, df)

        if matched_state:

            table_df = (
                df[df["state_of_residence"] == matched_state]
                .groupby("month")["births"]
                .sum()
                .reset_index()
                .sort_values("month")
            )

            st.success(f"Monthly births for {matched_state}")
            st.dataframe(table_df, use_container_width=True)

        else:
            st.warning("Please specify a valid state.")

        st.stop()

    # ============================
    # LLM ROUTE
    # ============================
    if intent == "llm":

        total_births = int(filtered["births"].sum())

        state_totals = (
            filtered.groupby("state_of_residence")["births"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .to_dict()
        )

        gender_totals = (
            filtered.groupby("sex_of_infant")["births"]
            .sum()
            .to_dict()
        )

        context = {
            "total_births": total_births,
            "top_states": state_totals,
            "gender_totals": gender_totals,
            "filters_applied": {
                "states": state_sel,
                "gender": gender_sel,
                "months": month_sel,
            },
        }

        system_prompt = f"""
You are a senior data analyst.

Use ONLY the dataset context provided.
Do not fabricate numbers.
Provide concise executive insights.

Dataset Context:
{json.dumps(context)}
"""

        try:
            api_key = st.secrets["GROQ_API_KEY"]

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 400,
                },
                timeout=30,
            )

            if response.status_code != 200:
                st.error(f"Groq API Error {response.status_code}")
                st.code(response.text)
                st.stop()

            result = response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            result = f"Error: {str(e)}"

        with st.chat_message("assistant"):
            st.markdown(result)

        st.session_state.messages.append({"role": "assistant", "content": result})
