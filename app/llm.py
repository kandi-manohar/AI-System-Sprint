"""
LLM integration layer (Groq free tier).

Two LLM calls make up the NL query pipeline:
  1. NL question -> SQL query (structured JSON output, schema-grounded)
  2. SQL result rows -> short natural-language answer

Design choices worth noting in the walkthrough:
  - The LLM is asked to return strict JSON (`{"sql": "..."}`) rather than
    free text, so the SQL can be parsed reliably rather than regex-scraped
    out of a chatty response.
  - Generated SQL is never executed blindly: app.db.run_readonly_query
    rejects anything that isn't a SELECT or that contains write/schema
    keywords, independent of what the LLM was told to do.
  - If the generated SQL fails to execute (syntax error, unknown column),
    the error is fed back to the LLM once for a self-correction retry,
    rather than failing the whole request on the first mistake.
  - The final answer is generated from the *actual query results*, not
    from the LLM's general knowledge, to reduce hallucination.
"""

import json
import os
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

from app.db import SCHEMA_DESCRIPTION, run_readonly_query

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_client: Optional[Groq] = None


def get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
                "(free at https://console.groq.com/keys)."
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


SQL_SYSTEM_PROMPT = f"""You are a SQL generation assistant for a customer support ticket database.

{SCHEMA_DESCRIPTION}

Rules:
- Generate a single SQLite SELECT query that answers the user's question.
- Only use the columns and table listed above. Never invent columns.
- Never generate INSERT, UPDATE, DELETE, DROP, ALTER, or any write/schema statement.
- For "this month" / "this week" / relative time questions, use the MAX(created_at)
  in the table as the reference "today", since the data is historical, not live.
- Return ONLY a JSON object of the form {{"sql": "<the query>"}} with no other text,
  no markdown code fences, and no explanation.
"""

ANSWER_SYSTEM_PROMPT = """You answer questions about customer support tickets using
ONLY the query result data provided to you. Be concise (1-3 sentences). If the
result set is empty, say so plainly. Do not invent numbers not present in the data.
"""


def _extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM response: {text!r}")
    return json.loads(text[start:end + 1])


def nl_to_sql(question: str, error_feedback: Optional[str] = None) -> str:
    client = get_client()
    messages = [
        {"role": "system", "content": SQL_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if error_feedback:
        messages.append({
            "role": "user",
            "content": (
                f"That query failed with this error: {error_feedback}\n"
                "Please return a corrected query as the same JSON format."
            ),
        })

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=400,
    )
    content = response.choices[0].message.content
    parsed = _extract_json(content)
    sql = parsed.get("sql", "").strip().rstrip(";")
    if not sql:
        raise ValueError("LLM did not return a 'sql' field.")
    return sql


def summarize_result(question: str, rows: list[dict]) -> str:
    client = get_client()
    preview = rows[:25]  # cap what we send back for the summary step
    user_content = (
        f"Question: {question}\n"
        f"Query result ({len(rows)} row(s), showing up to 25): {json.dumps(preview, default=str)}"
    )
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def answer_question(question: str) -> dict:
    """
    Full NL query pipeline: question -> SQL -> execute -> NL answer.
    Retries SQL generation once if the first attempt fails to execute.
    """
    sql = nl_to_sql(question)
    try:
        rows = run_readonly_query(sql)
    except Exception as first_error:
        # Self-correction retry: feed the error back to the model once.
        sql = nl_to_sql(question, error_feedback=str(first_error))
        rows = run_readonly_query(sql)  # let this one raise if it fails again

    answer = summarize_result(question, rows)
    return {
        "question": question,
        "sql": sql,
        "answer": answer,
        "row_count": len(rows),
        "data": rows,
    }
