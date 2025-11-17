import os
import json
import logging
from typing import Optional, Tuple, List

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from dotenv import dotenv_values

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# FAISS ko optional banaya hai
try:
    from langchain_community.vectorstores import FAISS
except Exception:
    FAISS = None

from langchain_core.runnables import RunnableParallel, RunnablePassthrough


# -------------------------------------------------------------------
# Environment & logging
# -------------------------------------------------------------------

# Always force reading from the project root .env (avoid parent directory). Handle BOM & CRLF.
ENV_PATH = Path(__file__).resolve().parent / ".env"

def _robust_load_env(path: Path) -> dict:
    """Attempt to load .env with multiple strategies and encodings.
    Returns a dict (may be empty). Adds extra debug prints for troubleshooting."""
    result = {}
    if not path.exists():
        print(f"DEBUG: .env missing at {path}")
        return result

    print(f"DEBUG ENV_PATH: {path}")
    # Strategy 1: python-dotenv with utf-8-sig
    try:
        result = dotenv_values(dotenv_path=str(path), encoding="utf-8-sig") or {}
        print("DEBUG dotenv_values utf-8-sig:", repr(result))
    except Exception as e:
        print("DEBUG dotenv_values utf-8-sig raised:", repr(e))

    if result:
        return result

    # Read raw bytes to inspect BOM / encoding
    try:
        raw = path.read_bytes()
        print("DEBUG raw first 32 bytes:", raw[:32])
    except Exception as e:
        print("DEBUG unable to read raw bytes:", repr(e))
        return result

    # Try alternate decodings if utf-8-sig produced nothing
    for enc in ["utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin-1"]:
        try:
            text_data = raw.decode(enc)
            lines = [l.strip() for l in text_data.splitlines() if l.strip() and not l.strip().startswith("#")]
            tentative = {}
            for line in lines:
                if "=" in line:
                    k, v = line.split("=", 1)
                    tentative[k.strip()] = v.strip()
            print(f"DEBUG manual parse with {enc}:", repr(tentative))
            if tentative:
                return tentative
        except Exception as e:
            print(f"DEBUG decode failed for {enc}:", repr(e))
    return result

# Use robust loader
_env_config = _robust_load_env(ENV_PATH)
print("DEBUG final env_config:", repr(_env_config))

def _clean(v: Optional[str]) -> Optional[str]:
    return v.strip() if isinstance(v, str) else v

GOOGLE_API_KEY = _clean(_env_config.get("GOOGLE_API_KEY"))
DATABASE_URL = _clean(_env_config.get("DATABASE_URL") or "sqlite:///data.db")
FEEDBACK_LOG_PATH = _clean(_env_config.get("FEEDBACK_LOG_PATH") or "feedback_log.jsonl")

print("DEBUG GOOGLE_API_KEY:", GOOGLE_API_KEY if GOOGLE_API_KEY else "<MISSING>")

if not GOOGLE_API_KEY:
    parsed_keys = sorted(list(_env_config.keys()))
    logging.error(
        "GOOGLE_API_KEY missing. .env path=%s exists=%s parsed_keys=%s", ENV_PATH, ENV_PATH.exists(), parsed_keys
    )
    raise RuntimeError(
        f"GOOGLE_API_KEY not found in {ENV_PATH}. Parsed keys: {parsed_keys}. Ensure line is 'GOOGLE_API_KEY=...' with no quotes or spaces."
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# DB utilities
# -------------------------------------------------------------------

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL)
    return _engine


def get_schema_description() -> str:
    """
    Return a human-readable schema description for prompting.
    """
    engine = get_engine()
    inspector = inspect(engine)

    lines: List[str] = []
    for table_name in inspector.get_table_names():
        lines.append(f"Table: {table_name}")
        columns = inspector.get_columns(table_name)
        for col in columns:
            lines.append(f"  - {col['name']} ({str(col['type'])})")
        lines.append("")
    schema_text = "\n".join(lines) if lines else "No tables found."
    return schema_text


# -------------------------------------------------------------------
# RAG: lightweight knowledge base (optional)
# -------------------------------------------------------------------

_vectorstore = None


def _load_rag_docs() -> List[str]:
    """
    Load RAG docs from rag_docs.json (optional).
    File format: ["free text about schema", "business rules", ...]
    """
    path = Path("rag_docs.json")
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            docs = json.load(f)
        if isinstance(docs, list):
            return [str(d) for d in docs]
    except Exception as e:
        logger.warning("Failed to load rag_docs.json: %s", e)
    return []


def get_vectorstore() -> Optional[object]:  # FAISS class loaded dynamically
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    if FAISS is None:
        logger.info("FAISS not available, skipping RAG vectorstore.")
        return None

    docs = _load_rag_docs()
    if not docs:
        return None

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY,
    )
    _vectorstore = FAISS.from_texts(docs, embedding=embeddings)
    return _vectorstore


def retrieve_context(query: str, k: int = 3) -> str:
    """
    Retrieve extra context for ambiguous queries using FAISS (if available).
    """
    vs = get_vectorstore()
    if vs is None:
        return ""
    docs = vs.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)


# -------------------------------------------------------------------
# LLM + LangChain for SQL generation
# -------------------------------------------------------------------

def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.1,
        convert_system_message_to_human=True,
        google_api_key=GOOGLE_API_KEY,
    )


def build_sql_chain():
    schema = get_schema_description()

    system_prompt = """
You are an expert data analyst and SQL generator.

You are given:
- A database schema.
- (Optionally) some extra business context.
- A natural-language question from the user.

Your task:
- Generate a single, syntactically correct SQL query for a {dialect} database.
- ONLY select data. The query MUST NOT modify the database.
- Do NOT use DDL or DML such as INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE.
- Do NOT use PRAGMA or database-specific control commands.
- Do NOT use comments.

Rules:
- Output ONLY the SQL query, no explanations, no backticks.
- If the question cannot be answered with the schema, write a query that returns 0 rows with a clear WHERE FALSE condition.
    """.strip()

    human_prompt = """
SCHEMA:
{schema}

EXTRA CONTEXT (may be empty):
{context}

USER QUESTION:
{question}

Remember:
- Output only a single SQL SELECT query.
- Do not modify the database.
    """.strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )

    llm = get_llm()
    chain = (
        RunnableParallel(
            question=RunnablePassthrough(),
            schema=lambda _: schema,
            context=lambda q: retrieve_context(q),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


_sql_chain = None


def get_sql_chain():
    global _sql_chain
    if _sql_chain is None:
        _sql_chain = build_sql_chain()
    return _sql_chain


# -------------------------------------------------------------------
# Guardrails & execution
# -------------------------------------------------------------------

FORBIDDEN_SQL_KEYWORDS = [
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "REPLACE",
    "ATTACH",
    "DETACH",
    "PRAGMA",
]


def is_safe_sql(sql: str) -> bool:
    upper = sql.upper()
    if ";" in upper.strip().rstrip(";"):
        # multiple statements, reject
        return False
    for kw in FORBIDDEN_SQL_KEYWORDS:
        if kw in upper:
            return False
    return True


def clean_sql(sql: str) -> str:
    # Remove backticks / code fences if LLM accidentally adds them
    sql = sql.strip()
    if sql.startswith("```"):
        sql = sql.strip("`")
    return sql.strip()


def generate_sql_with_retries(
    question: str, max_retries: int = 2
) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (sql, error_message).
    """
    chain = get_sql_chain()
    last_error = None
    prev_sql = None

    for attempt in range(max_retries + 1):
        if attempt == 0:
            sql = chain.invoke(question)
        else:
            # On retry, provide previous SQL + error as hint
            llm = get_llm()
            schema = get_schema_description()
            context = retrieve_context(question)
            retry_prompt = f"""
You previously generated this SQL:

{prev_sql}

It produced the following database error:

{last_error}

Using the schema below, fix the SQL so that it runs correctly.
Remember: SELECT-only, no modifications.

SCHEMA:
{schema}

EXTRA CONTEXT:
{context}

USER QUESTION:
{question}

Return only the corrected SQL.
            """.strip()
            sql = llm.invoke(retry_prompt).content

        sql = clean_sql(sql)
        prev_sql = sql

        if not is_safe_sql(sql):
            last_error = "Generated SQL failed safety checks (non-SELECT or dangerous keywords)."
            logger.warning(last_error)
            continue

        # Try dry-run or simple parsing via DB
        try:
            _ = execute_sql(sql, limit_for_validation=1)
            # If no exception, we accept the SQL
            return sql, None
        except SQLAlchemyError as e:
            last_error = str(e)
            logger.warning("SQL attempt failed: %s", last_error)

    return None, last_error


def execute_sql(sql: str, limit_for_validation: Optional[int] = None) -> pd.DataFrame:
    engine = get_engine()
    if limit_for_validation is not None:
        wrapped_sql = f"SELECT * FROM ({sql}) AS sub LIMIT {limit_for_validation}"
        query = text(wrapped_sql)
    else:
        query = text(sql)
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)
    return df


# -------------------------------------------------------------------
# Insight summarisation
# -------------------------------------------------------------------

def summarize_dataframe(df: pd.DataFrame, max_rows_for_sample: int = 5) -> str:
    if df.empty:
        return "No rows were returned for this query."

    lines: List[str] = []
    lines.append(f"Returned {len(df)} rows and {len(df.columns)} columns.")

    # Show a small sample of columns
    if len(df.columns) <= 8:
        cols_preview = ", ".join(df.columns)
    else:
        cols_preview = ", ".join(list(df.columns[:5]) + ["..."])
    lines.append(f"Columns: {cols_preview}")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        lines.append("")
        lines.append("Numeric column summary:")
        desc = df[numeric_cols].describe().T[["mean", "min", "max"]]
        for col, row in desc.iterrows():
            lines.append(
                f"- {col}: mean={row['mean']:.2f}, min={row['min']:.2f}, max={row['max']:.2f}"
            )

    # Show top categories for first categorical column, if any
    cat_cols = df.select_dtypes(exclude=["number", "datetime64[ns]"]).columns.tolist()
    if cat_cols:
        col = cat_cols[0]
        lines.append("")
        lines.append(f"Top categories for '{col}':")
        vc = df[col].value_counts().head(5)
        for idx, val in vc.items():
            lines.append(f"- {idx}: {val} rows")

    return "\n".join(lines)


# -------------------------------------------------------------------
# Feedback logging
# -------------------------------------------------------------------

def log_feedback(
    user_query: str,
    sql_query: Optional[str],
    feedback_label: str,
    notes: str = "",
) -> None:
    """
    Append a JSON line to feedback_log.jsonl for later analysis/fine-tuning.
    feedback_label: "correct" | "incorrect" | "partial"
    """
    record = {
        "user_query": user_query,
        "sql_query": sql_query,
        "feedback": feedback_label,
        "notes": notes,
    }
    try:
        with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning("Failed to write feedback log: %s", e)
