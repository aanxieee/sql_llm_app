import streamlit as st

# Try importing backend; if it fails, show the error in UI instead of blank screen
try:
    from sql import (
        generate_sql_with_retries,
        execute_sql,
        summarize_dataframe,
        log_feedback,
    )
    backend_import_error = None
except Exception as e:
    backend_import_error = e

st.set_page_config(page_title="NL → SQL Explorer", layout="wide")

if backend_import_error is not None:
    st.title("🚨 App startup error")
    st.error(
        "Backend (sql.py) import fail ho gaya. Neeche exact error dekh sakti ho. "
        "Generally ye missing package ya code error hota hai."
    )
    st.code(str(backend_import_error), language="text")
    st.stop()

st.title("🔍 Natural Language → SQL Explorer (Gemini + LangChain)")
st.caption(
    "Type a question in natural language. The app generates SQL using Gemini, runs it on your DB, "
    "shows the results, charts, and an automatic summary. Optionally log feedback."
)

# -------------------------------------------------------------------
# Sidebar info
# -------------------------------------------------------------------

with st.sidebar:
    st.header("ℹ️ How to use")
    st.markdown(
        """
1. Make sure your `.env` has:
   - `GOOGLE_API_KEY`
   - `DATABASE_URL` (e.g. `sqlite:///data.db`)
2. Ask questions like:
   - *Total sales by month in 2023*
   - *Top 10 customers by revenue*
3. Review the SQL and results, then give feedback.
        """
    )
    st.divider()
    st.markdown("**Feedback log** is stored in `feedback_log.jsonl`.")

# -------------------------------------------------------------------
# Main UI
# -------------------------------------------------------------------

query = st.text_area(
    "🗣️ Ask a question about your data",
    placeholder="e.g. Show total sales per month in 2023",
    height=100,
)

run_btn = st.button("Run query", type="primary")

sql_query = None
df = None
error_msg = None

if run_btn and query.strip():
    with st.spinner("Generating SQL with Gemini and running the query..."):
        sql_query, err = generate_sql_with_retries(query)
        if sql_query is None:
            error_msg = f"Failed to generate a runnable SQL query.\n\nLast error: {err}"
        else:
            try:
                df = execute_sql(sql_query)
            except Exception as e:
                error_msg = f"Error executing SQL: {e}"

# -------------------------------------------------------------------
# Display results
# -------------------------------------------------------------------

if sql_query is not None:
    st.subheader("🧠 Generated SQL")
    st.code(sql_query, language="sql")

if error_msg:
    st.error(error_msg)

if df is not None and not df.empty:
    import pandas as pd  # local import, just to be safe
    st.subheader("📊 Query results")
    st.dataframe(df, use_container_width=True)

    # Insight summary
    st.subheader("📝 Insight summary")
    summary_text = summarize_dataframe(df)
    st.markdown(f"```text\n{summary_text}\n```")

    # Basic visualization options
    st.subheader("📈 Quick visualization")

    cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if len(cols) >= 2:
        x_col = st.selectbox("X-axis", cols, index=0)
        y_col = st.selectbox(
            "Y-axis (numeric preferred)", numeric_cols or cols, index=min(1, len(cols) - 1)
        )
        chart_type = st.selectbox(
            "Chart type",
            ["Line", "Bar", "Area"],
        )

        chart_df = df[[x_col, y_col]].dropna()

        if chart_df.empty:
            st.info("Not enough data to plot the selected columns.")
        else:
            chart_df = chart_df.set_index(x_col)
            if chart_type == "Line":
                st.line_chart(chart_df)
            elif chart_type == "Bar":
                st.bar_chart(chart_df)
            elif chart_type == "Area":
                st.area_chart(chart_df)
    else:
        st.info("Need at least two columns to plot a chart.")

    # -------------------------------------------------------------------
    # Feedback section
    # -------------------------------------------------------------------
    st.subheader("✅ Was this answer helpful?")
    feedback_col1, feedback_col2 = st.columns([1, 2])

    with feedback_col1:
        feedback_label = st.radio(
            "Your verdict",
            ["Not selected", "correct", "incorrect", "partial"],
            index=0,
            label_visibility="collapsed",
        )

    with feedback_col2:
        notes = st.text_input(
            "Optional notes (e.g. what was wrong / missing)",
            placeholder="Add details here...",
        )

    if feedback_label != "Not selected":
        if st.button("Submit feedback"):
            log_feedback(query, sql_query, feedback_label, notes)
            st.success("Thanks! Your feedback has been recorded.")
else:
    if not error_msg:
        st.info("Enter a question and click **Run query** to get started.")
