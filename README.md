# AI System

An end-to-end AI system over a customer support ticket dataset: natural
language querying, rule-based anomaly detection, a REST API, and a
minimal UI — built for the DOTMappers AI Engineer technical assessment.

## 1. Setup

### Prerequisites
- Python 3.10+
- A free Groq API key: https://console.groq.com/keys (no credit card required)

### Steps
```bash
# 1. Clone/unzip the repo, then from the project root:
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your Groq API key
cp .env.example .env
# then edit .env and paste your key into GROQ_API_KEY=

# 4. Make sure the dataset is present
# data/support_tickets.csv should already be in the repo.

# 5. Run everything with a single command
python run.py
```

This starts:
- **API** at http://127.0.0.1:8000 (interactive docs at `/docs`)
- **UI** at http://127.0.0.1:8501

Stop both with `Ctrl+C`.

### Alternative: Docker
```bash
cp .env.example .env   # fill in GROQ_API_KEY first
docker-compose up
```

## 2. Architecture
                    ┌─────────────────────┐
                     │  support_tickets.csv │
                     └──────────┬───────────┘
                                │ loaded once, idempotent
                                ▼
                     ┌─────────────────────┐
                     │   SQLite (data/*)    │  <- queryable store
                     └──────────┬───────────┘
                                │
            ┌───────────────────┼────────────────────┐
            ▼                                          ▼
     ┌─────────────────────┐                   ┌─────────────────────────┐
     │  NL Query Pipeline    │                   │  Anomaly Detection       │
     │  (app/llm.py)         │                   │  (app/anomalies.py)      │
     │  question -> SQL       │                   │  rule/stats-based,       │
     │  (Groq LLM) -> execute │                   │  no LLM call needed      │
     │  -> NL answer (Groq)   │                   │                          │
     └───────────┬────────────┘                   └────────────┬─────────────┘
                 │                                              │
                 └───────────────────┬──────────────────────────┘
                                      ▼
                          ┌─────────────────────┐
                          │   FastAPI (app/main.py) │
                          │  /health /query /anomalies /schema │
                          └───────────┬──────────┘
                                      ▼
                          ┌─────────────────────┐
                          │ Streamlit UI (ui/)    │
                          │ thin HTTP client       │
                          └─────────────────────┘

### Why these choices

- **SQLite as the query layer, not raw pandas filtering.** The LLM's job
  is to translate a question into a well-understood target language
  (SQL) rather than a bespoke DSL. This also means the same approach
  scales to a much larger CSV, or to swapping in Postgres, with no
  redesign.
- **NL→SQL uses structured JSON output**, not free text parsing. The
  model is instructed to return `{"sql": "..."}` only, which is parsed
  directly rather than regex-scraped from a chatty response.
- **Generated SQL is never trusted blindly.** `run_readonly_query`
  independently rejects anything that isn't a `SELECT`, or that contains
  write/schema keywords (`INSERT`, `DROP`, `ATTACH`, `PRAGMA`, etc.),
  regardless of what the LLM was told to do.
- **One self-correction retry.** If the generated SQL fails to execute
  (e.g. references a wrong column), the error message is fed back to the
  LLM once for a corrected query, instead of failing the whole request
  on the first mistake.
- **Anomaly detection is deterministic, not LLM-based.** It uses the
  IQR method for statistical outliers in resolution time, and an
  explicit SLA rule (unresolved High/Critical tickets older than a
  threshold) matching the assessment's own example. This is
  reproducible, explainable (every flag has a numeric reason attached),
  and doesn't burn LLM calls scanning 500 rows.
- **UI is a thin HTTP client of the API**, not a second copy of the
  logic. This keeps `/query` and `/anomalies` independently testable
  and matches "REST API AND a minimal UI" as two faces of one backend.
- **`run.py` gives a single-command start** without requiring Docker,
  while `docker-compose.yml` is provided as an equally valid alternative
  single-command path.

## 3. Model / Tools Used

- **LLM:** Groq free tier, `openai/gpt-oss-120b` (configurable via
  `GROQ_MODEL` in `.env`). Chosen for zero cost, no local GPU/download
  requirement, and low latency — relevant when demoing live queries.
  Note: as of Sept 2026, Groq moved `llama-3.3-70b-versatile` and
  `llama-3.1-8b-instant` to an Enterprise/Contact-Sales-only tier, so
  they are no longer usable on the free developer tier — `openai/gpt-oss-120b`
  is the current free-tier-eligible production model on Groq. If Groq's
  lineup changes again, check https://console.groq.com/docs/models and
  update `GROQ_MODEL` in `.env` — no code changes needed.
- **API:** FastAPI + Uvicorn
- **UI:** Streamlit
- **Data:** pandas (loading/anomaly stats) + SQLite (query execution)

## 4. Example Queries and Outputs

Below are outputs against the actual dataset (500 tickets, Jan–Mar 2024).

**Q: "How many tickets are currently open?"**
```json
{
  "sql": "SELECT COUNT(*) AS open_count FROM tickets WHERE status = 'Open'",
  "answer": "There are 111 tickets currently open.",
  "row_count": 1
}
```

**Q: "What is the average customer rating for Technical category tickets?"**
```json
{
  "sql": "SELECT AVG(customer_rating) AS avg_rating FROM tickets WHERE category = 'Technical'",
  "answer": "The average customer rating for Technical category tickets is approximately 3.74.",
  "row_count": 1
}
```
(Verified directly against the data: 3.7403846...)

**Q: "Show me all Critical tickets not resolved within 12 hours."**
```json
{
  "sql": "SELECT ticket_id, status, resolution_time_hrs FROM tickets WHERE priority = 'Critical' AND (resolution_time_hrs > 12 OR resolution_time_hrs IS NULL)",
  "answer": "34 Critical tickets were not resolved within 12 hours.",
  "row_count": 34
}
```
(Verified directly against the data.)

**Anomaly detection output shape (`GET /anomalies`), run against the real dataset:**
```json
{
  "as_of": "2024-03-30 18:06:00",
  "sla_hours_threshold": 24,
  "total_flagged": 101,
  "long_resolution_outliers": [
    {
      "ticket_id": "TKT-023",
      "anomaly_type": "long_resolution",
      "resolution_time_hrs": 76.1,
      "reason": "Resolution time 76.1h exceeds the normal range (upper threshold 48.1h, based on IQR of resolved tickets: Q1=6.2h, Q3=22.9h)."
    }
    // ... 20 more
  ],
  "sla_breaches": [
    {
      "ticket_id": "TKT-007",
      "anomaly_type": "sla_breach",
      "priority": "High",
      "status": "Open",
      "age_hrs": 1658.7,
      "reason": "High priority ticket has been open for 1658.7h, exceeding the 24.0h SLA threshold."
    }
    // ... 79 more
  ]
}
```
(21 long-resolution outliers, 80 SLA breaches — verified against the real dataset.)

> Note: `as_of` defaults to the dataset's latest ticket date
> (2024-03-30 18:06), which is used as a stand-in for "now" since the
> dataset is historical. Any ticket still Open/Escalated at that point
> that was created more than `sla_hours` earlier is flagged — with the
> default, that's **80 SLA breaches**, some with very large ages (e.g.
> 2000+ hours), because they were created early in the Jan–Mar window
> and never got resolved or escalated-and-closed. Pass a different
> `?as_of=` (e.g. `2024-02-15 00:00`) to see the breach set as it would
> have looked at an earlier point in time instead.

## 5. Known Limitations

- **NL→SQL is not guaranteed correct for arbitrarily complex questions.**
  It's grounded with the schema and retries once on execution failure,
  but semantic mistakes (e.g. misreading "this month" against a
  historical dataset) are possible. The system mitigates this by
  anchoring relative-time language to `MAX(created_at)` in the prompt,
  but doesn't verify query *intent* beyond that it executes successfully.
- **`as_of` for SLA breach detection defaults to the dataset's last
  ticket date**, not wall-clock time, since the data is historical. This
  is documented behavior, not a bug, but is worth explaining upfront.
- **No authentication** on the API — out of scope for this assessment,
  but would be required before any real deployment.
- **No conversation memory** — each `/query` call is independent; there's
  no multi-turn follow-up context ("...and what about last week?").
- **IQR outlier threshold is fixed at 1.5×**, the standard convention,
  but not tuned against any ground-truth "actually anomalous" labels
  (none exist in this dataset).
- **Groq free tier has rate limits.** Under heavy simultaneous testing,
  requests could be throttled (a 429 will surface as a 500 from `/query`).

## 6. Project Structure

    support-ticket-ai/
    ├── app/
    │ ├── main.py # FastAPI app and routes
    │ ├── db.py # CSV -> SQLite loading, safe query execution
    │ ├── llm.py # Groq NL->SQL and answer summarization
    │ └── anomalies.py # Rule-based anomaly detection
    ├── ui/
    │ └── streamlit_app.py # Minimal UI, thin client of the API
    ├── data/
    │ └── support_tickets.csv
    ├── run.py # Single-command launcher (API + UI)
    ├── requirements.txt
    ├── Dockerfile
    ├── docker-compose.yml
    ├── .env.example
    └── README.md
