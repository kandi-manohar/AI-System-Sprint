"""
Minimal Streamlit UI for the Support Ticket AI system.

This is a thin client over the FastAPI backend (calls /query and
/anomalies via HTTP) rather than importing app.* directly, so the UI
and API stay properly decoupled and the API can be evaluated
independently, per the assessment's "REST API AND minimal UI" requirement.
"""

import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def render_table(rows: list[dict]) -> None:
    """
    Render a list of dicts as an HTML table via pandas' own renderer.

    Deliberately avoids st.dataframe()/st.table(), which depend on
    pyarrow under the hood. On some locked-down Windows machines
    (corporate Application Control / AppLocker policies), pyarrow's
    native DLL gets blocked from loading, crashing the page. Plain
    HTML rendering has no such dependency.
    """
    if not rows:
        st.info("No rows.")
        return
    df = pd.DataFrame(rows)
    st.markdown(df.to_html(index=False, escape=True), unsafe_allow_html=True)

st.set_page_config(page_title="Support Ticket AI", layout="wide")
st.title("Support Ticket AI System")
st.caption(f"Backend: {API_BASE_URL}")

# --- Health check banner ---
try:
    resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
    if resp.status_code == 200:
        st.success("Backend API is reachable.")
    else:
        st.warning("Backend API responded but not healthy.")
except requests.exceptions.RequestException:
    st.error(
        f"Cannot reach backend API at {API_BASE_URL}. "
        "Make sure `uvicorn app.main:app` is running."
    )

tab_query, tab_anomalies = st.tabs(["Ask a question", "Anomalies"])

# --- Tab 1: NL Query ---
with tab_query:
    st.subheader("Ask a natural language question about the tickets")

    example_queries = [
        "How many tickets are currently open?",
        "Which agent resolved the most tickets?",
        "Show me all Critical tickets not resolved within 12 hours.",
        "What is the average customer rating for Technical category tickets?",
        "Which agent has the lowest average customer rating?",
    ]
    chosen_example = st.selectbox("Or pick an example:", [""] + example_queries)
    question = st.text_input("Your question", value=chosen_example)

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    r = requests.post(
                        f"{API_BASE_URL}/query", json={"question": question}, timeout=60
                    )
                    r.raise_for_status()
                    result = r.json()
                    st.markdown(f"**Answer:** {result['answer']}")
                    with st.expander("Generated SQL"):
                        st.code(result["sql"], language="sql")
                    if result["data"]:
                        render_table(result["data"])
                    else:
                        st.info("Query returned no rows.")
                except requests.exceptions.HTTPError as e:
                    detail = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
                    st.error(f"Error: {detail}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Request failed: {e}")

# --- Tab 2: Anomalies ---
with tab_anomalies:
    st.subheader("Anomaly Detection")
    col1, col2 = st.columns(2)
    with col1:
        sla_hours = st.number_input("SLA threshold (hours)", min_value=1, value=24)
    with col2:
        as_of = st.text_input("As-of date (optional, e.g. 2024-03-15 00:00)", value="")

    if st.button("Run anomaly detection"):
        with st.spinner("Scanning tickets..."):
            try:
                params = {"sla_hours": sla_hours}
                if as_of.strip():
                    params["as_of"] = as_of.strip()
                r = requests.get(f"{API_BASE_URL}/anomalies", params=params, timeout=30)
                r.raise_for_status()
                result = r.json()

                st.markdown(
                    f"**{result['total_flagged']}** anomalies flagged "
                    f"(as of `{result['as_of']}`, SLA threshold `{result['sla_hours_threshold']}h`)"
                )

                st.markdown("### Long resolution time outliers")
                if result["long_resolution_outliers"]:
                    render_table(result["long_resolution_outliers"])
                else:
                    st.info("None found.")

                st.markdown("### SLA breaches (unresolved High/Critical tickets)")
                if result["sla_breaches"]:
                    render_table(result["sla_breaches"])
                else:
                    st.info("None found.")
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")