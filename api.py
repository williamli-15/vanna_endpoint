import os
import sqlite3
import pandas as pd
from fastapi import FastAPI, HTTPException
import time
from typing import Any, Dict, Literal, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

DATA_DIR = os.environ.get('DATA_DIR', '.')
CHROMA_PATH = os.path.join(DATA_DIR, 'chroma')
DB_PATH = os.path.join(DATA_DIR, 'yc_companies.db')

print(f"Using data directory: {DATA_DIR}")
print(f"ChromaDB path: {CHROMA_PATH}")
print(f"SQLite DB path: {DB_PATH}")

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config={'path': CHROMA_PATH})
        OpenAI_Chat.__init__(self, config=config)

vn = MyVanna(config={'api_key': OPENAI_API_KEY, 'model': 'gpt-4o'})


# ========= Transparency Models =========
class TraceStep(BaseModel):
    step: Literal['sql_generation', 'sql_execution', 'postprocess', 'error']
    thought: str
    action: Dict[str, Any] = Field(default_factory=dict)
    observation: Dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: int = 0

class QueryRequest(BaseModel):
    question: str
    trace: Optional[bool] = False
    sample_rows: Optional[int] = 5

class QueryResponse(BaseModel):
    sql: str
    company_ids: List[int]
    error: Optional[str] = None
    trace: Optional[List[TraceStep]] = None
    sample: Optional[List[Dict[str, Any]]] = None

# ========= Helpers =========
def _now_ms(): return int(time.perf_counter() * 1000)

def _parse_tables_and_joins(sql: str) -> Dict[str, Any]:
    tables, joins = set(), []
    s = sql.lower()

    # cheap harvest:
    for t in ['companies','company_founders','founders','founder_experience',
              'founder_education','founder_skills','company_industries','company_tags']:
        if t in s: tables.add(t)
    # join hints
    if 'c.id = cf.company_id' in s: joins.append('c.id = cf.company_id')
    if 'cf.founder_id = fe.founder_id' in s: joins.append('cf.founder_id = fe.founder_id')
    if 'cf.founder_id = fs.founder_id' in s: joins.append('cf.founder_id = fs.founder_id')
    if 'cf.founder_id = f.profileid' in s: joins.append('cf.founder_id = f.profileId')
    return {"tables": sorted(list(tables)), "joins": joins}

def _synonym_rules(question: str, sql: str) -> List[str]:
    q, s = question.lower(), sql.lower()
    rules = []
    if 'mit' in q or 'massachusetts institute of technology' in s:
        rules.append("MIT → Massachusetts Institute of Technology")
    if 'faang' in q or " ('meta'," in s:
        rules.append("FAANG → {Meta, Apple, Amazon, Netflix, Google}")
    if 'engineer' in q or "%engineer%" in s:
        rules.append("Engineer → title LIKE '%Engineer%'")
    if 'ai ' in f"{q} " or "artificial intelligence" in s:
        rules.append("AI → industry = 'Artificial Intelligence'")
    if 'fintech' in q or "ci.industry = 'fintech'" in s:
        rules.append("Fintech → industry = 'Fintech'")
    return rules

def explain_sql_rationale(question: str, sql: str) -> str:
    rules = _synonym_rules(question, sql)
    if 'founder_education' in sql.lower():
        rules.insert(0, "Filter founders by education (school LIKE ...)")
    if 'founder_experience' in sql.lower():
        rules.insert(0, "Filter founders by past company/title")
    if 'company_industries' in sql.lower():
        rules.insert(0, "Restrict to specific company industries")
    return "; ".join(dict.fromkeys(rules)) or "Generate SQL to return DISTINCT company IDs."


# ---- helpers (place above FastAPI routes) ----
def force_company_id_question(user_q: str) -> str:
    # Strong instruction the model can't miss
    return (
        "Return ONLY a SQL query that outputs a single column named id with DISTINCT company IDs. "
        "Do not return founder IDs, names, or any other columns. "
        f"Question: {user_q}"
    )

def extract_company_ids(sql: str, df: pd.DataFrame) -> tuple[list[int], Optional[TraceStep]]:
    """
    Prefer direct company IDs; if result looks like founder IDs, map to companies.
    Returns (company_ids, optional_postprocess_trace)
    """
    if df is None or df.empty:
        return [], None

    # direct columns first
    for col in ("id", "company_id"):
        if col in df.columns:
            ids = pd.to_numeric(df[col], errors="coerce").dropna().astype(int).unique().tolist()
            return ids, TraceStep(
                step="postprocess",
                thought="Normalize to company IDs; dedupe",
                action={"type":"normalize_results"},
                observation={"unique_company_ids": len(ids)}
            )

    # try founder → company mapping
    founder_col = next((c for c in ("profileId", "founder_id") if c in df.columns), None)
    if founder_col:
        founder_ids = pd.to_numeric(df[founder_col], errors="coerce").dropna().astype(int).unique().tolist()
        if founder_ids:
            inlist = ",".join(map(str, founder_ids))
            mapped = run_sql_from_local_db(f"""
                SELECT DISTINCT c.id
                FROM companies c
                JOIN company_founders cf ON c.id = cf.company_id
                WHERE cf.founder_id IN ({inlist})
            """)
            if mapped is not None and "id" in mapped.columns:
                ids = mapped["id"].astype(int).unique().tolist()
                return ids, TraceStep(
                    step="postprocess",
                    thought="Result contained founder IDs; map founders → companies via company_founders",
                    action={"type":"map_founders_to_companies", "founder_count": len(founder_ids)},
                    observation={"unique_company_ids": len(ids)}
                )
    # nothing usable
    return [], TraceStep(
        step="postprocess",
        thought="Could not infer company IDs from result columns",
        action={"type":"normalize_results"},
        observation={}
    )

def safe_sample(df: pd.DataFrame, n: int) -> List[Dict[str, Any]]:
    try:
        return df.head(max(n,0)).to_dict(orient="records")
    except Exception:
        return []

def try_explain_plan(sql: str) -> List[str]:
    try:
        plan = run_sql_from_local_db(f"EXPLAIN QUERY PLAN {sql}")
        # SQLite plan columns vary by version, but 'detail' is common
        col = next((c for c in plan.columns if c.lower() in {'detail','plan'}), None)
        if col:
            return plan[col].astype(str).head(10).tolist()
    except Exception:
        pass
    return []



# ### MODIFIED BLOCK START ###
# Instead of using vn.connect_to_sqlite(), we will set the vn.run_sql method directly.
# This is a more robust way to connect to a local database file and avoids the URL error.
def run_sql_from_local_db(sql: str) -> pd.DataFrame:
    """
    Connects to the local SQLite DB and executes a query.
    """
    s = sql.strip().lower()
    if not (s.startswith("select") or s.startswith("explain") or s.startswith("pragma") or s.startswith("with")):
        raise ValueError("Only SELECT/WITH/EXPLAIN/PRAGMA statements are allowed.")

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database file not found at {DB_PATH}. Please run create_database.py.")
        
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    return df

# Set the custom function for Vanna to use
vn.run_sql = run_sql_from_local_db
vn.run_sql_is_set = True
print(f"Vanna is configured to run SQL on local database: {DB_PATH}")
# ### MODIFIED BLOCK END ###


app = FastAPI(title="YC Companies Query API")


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yc-market-map.vercel.app","http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ========= /query with tracing =========
@app.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
async def query(request: QueryRequest):
    trace: List[TraceStep] = []
    try:
        # Step 1: generate SQL (Vanna)
        t0 = _now_ms()
        sql, df, _ = vn.ask(force_company_id_question(request.question), visualize=False)
        t1 = _now_ms()
        gen_obs = _parse_tables_and_joins(sql)
        gen_obs.update({"sql": sql, "rules": _synonym_rules(request.question, sql)})
        trace.append(TraceStep(
            step="sql_generation",
            thought=explain_sql_rationale(request.question, sql),
            action={"type":"generate_sql","question":request.question},
            observation=gen_obs,
            elapsed_ms=t1 - t0
        ))

        # Step 2: execution details (we already have df, but get plan & sample)
        t2 = _now_ms()
        sample = []
        plan = []
        if request.trace:
            plan = try_explain_plan(sql)
            sample = safe_sample(df, request.sample_rows or 5)

        exec_obs = {
            "row_count": int(len(df) if df is not None else 0),
            "columns": list(df.columns) if df is not None else [],
            "explain_plan": plan[:10] if plan else [],
            "sample": sample
        }
        t3 = _now_ms()
        trace.append(TraceStep(
            step="sql_execution",
            thought="Execute SQL and preview results",
            action={"type":"execute_sql"},
            observation=exec_obs,
            elapsed_ms=t3 - t2
        ))

        # Step 3: normalize to company IDs (with founder→company fallback)
        company_ids, post_trace = extract_company_ids(sql, df if df is not None else pd.DataFrame())
        if post_trace: trace.append(post_trace)

        return QueryResponse(
            sql=sql,
            company_ids=company_ids,
            error=None,
            trace=trace if request.trace else None,
            sample=sample if request.trace else None
        )

    except Exception as e:
        trace.append(TraceStep(
            step="error",
            thought="Unhandled exception while answering",
            action={"type":"exception"},
            observation={"message": str(e)}
        ))
        return QueryResponse(sql="", company_ids=[], error=str(e), trace=trace if request.trace else None)



@app.post("/train")
async def train_model():
    """
    This is the most critical function. It creates a robust "training package" for Vanna
    by combining schema information (DDL), documentation about synonyms and relationships,
    and a wide variety of question-SQL examples to handle user shorthand.
    """
    # 1. Train on DDL - The structural blueprint
    df_ddl = vn.run_sql("SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'")
    for ddl in df_ddl['sql'].to_list():
        vn.train(ddl=ddl)
        print(f"Trained on DDL: {ddl.split('(')[0]}...")

    # 2. Train on Documentation - The "How-To" and "Dictionary"
    vn.train(documentation="""
    - The user wants a list of companies. Your query should ALWAYS return a single column of unique company IDs. ALWAYS use SELECT DISTINCT c.id.
    - To connect companies to founders, JOIN `companies` on `company_founders` using `c.id = cf.company_id`.
    - To get founder details, JOIN `company_founders` on `founders` using `cf.founder_id = f.profileId`.
    - To get founder experience, JOIN `founders` on `founder_experience` using `f.profileId = fe.founder_id`.
    - To get founder education, JOIN `founders` on `founder_education` using `f.profileId = fe.founder_id`.
    - To get founder skills, JOIN `founders` on `founder_skills` using `f.profileId = fs.founder_id`.
    - To filter by industry or tag, JOIN `companies` on `company_industries` or `company_tags` respectively.
    - An 'engineer' title implies a fuzzy search. Use `founder_experience.title LIKE '%Engineer%'`.
    - For school names like 'MIT' or 'Berkeley', use a fuzzy search like `founder_education.school LIKE '%...%'`.
    - 'FAANG' refers to the companies 'Meta', 'Apple', 'Amazon', 'Netflix', and 'Google'. Use an IN clause on `founder_experience.company_name`.
    """)
    print("Trained on relationship and synonym documentation.")

    # 3. Train on High-Quality Question/SQL Pairs - The "Worked Examples"
    # This section is crucial for handling user shorthand. All JOINs are now corrected.
    example_queries = [
        # Company attribute queries
        {"question": "San Francisco", "sql": "SELECT id FROM companies WHERE city = 'San Francisco';"},
        {"question": "fintech companies", "sql": "SELECT c.id FROM companies AS c JOIN company_industries AS ci ON c.id = ci.company_id WHERE ci.industry = 'Fintech';"},
        {"question": "AI", "sql": "SELECT c.id FROM companies AS c JOIN company_industries AS ci ON c.id = ci.company_id WHERE ci.industry = 'Artificial Intelligence';"},
        {"question": "companies with more than 50 employees", "sql": "SELECT id FROM companies WHERE team_size > 50;"},
        
        # Founder attribute queries (education) - CORRECTED JOIN
        {"question": "MIT", "sql": "SELECT DISTINCT c.id FROM companies AS c JOIN company_founders AS cf ON c.id = cf.company_id JOIN founder_education AS fe ON cf.founder_id = fe.founder_id WHERE fe.school LIKE '%Massachusetts Institute of Technology%';"},
        {"question": "Berkeley founders", "sql": "SELECT DISTINCT c.id FROM companies AS c JOIN company_founders AS cf ON c.id = cf.company_id JOIN founder_education AS fe ON cf.founder_id = fe.founder_id WHERE fe.school LIKE '%Berkeley%';"},
        {"question": "founders from MIT", "sql": "SELECT DISTINCT c.id FROM companies c JOIN company_founders cf ON c.id = cf.company_id JOIN founder_education fe ON cf.founder_id = fe.founder_id WHERE fe.school LIKE '%Massachusetts Institute of Technology%';"},
        {"question": "founders from mit", "sql": "SELECT DISTINCT c.id FROM companies c JOIN company_founders cf ON c.id = cf.company_id JOIN founder_education fe ON cf.founder_id = fe.founder_id WHERE fe.school LIKE '%Massachusetts Institute of Technology%';"},

        # Founder attribute queries (experience) - CORRECTED JOIN
        {"question": "ex-Google founders", "sql": "SELECT DISTINCT c.id FROM companies AS c JOIN company_founders AS cf ON c.id = cf.company_id JOIN founder_experience AS fe ON cf.founder_id = fe.founder_id WHERE fe.company_name = 'Google';"},
        {"question": "founders who worked at FAANG", "sql": "SELECT DISTINCT c.id FROM companies AS c JOIN company_founders AS cf ON c.id = cf.company_id JOIN founder_experience AS fe ON cf.founder_id = fe.founder_id WHERE fe.company_name IN ('Meta', 'Apple', 'Amazon', 'Netflix', 'Google');"},
        
        # Founder attribute queries (skills) - CORRECTED JOIN
        {"question": "founders with Python skills", "sql": "SELECT DISTINCT c.id FROM companies AS c JOIN company_founders AS cf ON c.id = cf.company_id JOIN founder_skills AS fs ON cf.founder_id = fs.founder_id WHERE fs.skill = 'Python';"},
        
        # Complex, combined queries - CORRECTED JOIN
        {"question": "AI companies with founders from Stanford", "sql": "SELECT DISTINCT c.id FROM companies AS c JOIN company_industries AS ci ON c.id = ci.company_id JOIN company_founders AS cf ON c.id = cf.company_id JOIN founder_education AS fe ON cf.founder_id = fe.founder_id WHERE ci.industry = 'Artificial Intelligence' AND fe.school LIKE '%Stanford University%';"},
        {"question": "B2B companies where a founder was an engineer at a FAANG company", "sql": "SELECT DISTINCT c.id FROM companies AS c JOIN company_industries AS ci ON c.id = ci.company_id JOIN company_founders AS cf ON c.id = cf.company_id JOIN founder_experience AS fe ON cf.founder_id = fe.founder_id WHERE ci.industry = 'B2B' AND fe.company_name IN ('Meta', 'Apple', 'Amazon', 'Netflix', 'Google') AND fe.title LIKE '%Engineer%';"},
    ]
    for example in example_queries:
        vn.train(question=example["question"], sql=example["sql"])
    
    print(f"Trained on {len(example_queries)} example queries.")
    
    return {"status": "success", "message": "Model re-trained successfully with robust examples."}

@app.get("/test")
async def test_query():
    try:
        # A good test is one that requires a JOIN
        sql, df, _ = vn.ask("Show me 5 companies with founders who studied at MIT", visualize=False)
        return {"status": "success", "sql": sql, "row_count": len(df) if df is not None else 0}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/training-status")
async def training_status():
    try:
        training_data = vn.get_training_data()
        return {"status": "success", "training_items": len(training_data)}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/healthz")
def healthz(): return {"ok": True}
