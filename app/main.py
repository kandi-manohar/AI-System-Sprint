"""
FastAPI backend for the Support Ticket AI system.

Endpoints:
  GET  /health          - liveness/readiness check
  POST /query            - natural language question -> SQL -> answer
  GET  /anomalies        - rule-based anomaly detection
  GET  /schema           - dataset schema (helper for the UI / evaluator)
"""

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from app.anomalies import detect_anomalies
from app.db import SCHEMA_DESCRIPTION, load_csv_to_sqlite
from app.llm import answer_question

app = FastAPI(
    title="Support Ticket AI System",
    description="NL querying and anomaly detection over customer support tickets.",
    version="1.0.0",
)


@app.on_event("startup")
def startup():
    load_csv_to_sqlite()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    sql: str
    answer: str
    row_count: int
    data: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schema")
def schema():
    return {"schema": SCHEMA_DESCRIPTION}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="`question` must not be empty.")
    try:
        result = answer_question(request.question)
    except RuntimeError as e:
        # Typically a missing GROQ_API_KEY
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not answer question: {e}")
    return result


@app.get("/anomalies")
def anomalies(
    as_of: Optional[str] = Query(
        None, description="Reference timestamp e.g. '2024-03-15 00:00'. Defaults to latest ticket date."
    ),
    sla_hours: float = Query(24, description="SLA threshold in hours for unresolved High/Critical tickets."),
):
    try:
        return detect_anomalies(as_of=as_of, sla_hours=sla_hours)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
