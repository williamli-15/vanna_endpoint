import os
import sqlite3
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
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


# ### MODIFIED BLOCK START ###
# Instead of using vn.connect_to_sqlite(), we will set the vn.run_sql method directly.
# This is a more robust way to connect to a local database file and avoids the URL error.
def run_sql_from_local_db(sql: str) -> pd.DataFrame:
    """
    Connects to the local SQLite DB and executes a query.
    """
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


class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    sql: str
    company_ids: List[int]
    error: Optional[str] = None

# ---- helpers (place above FastAPI routes) ----
def force_company_id_question(user_q: str) -> str:
    # Strong instruction the model can't miss
    return (
        "Return ONLY a SQL query that outputs a single column named id with DISTINCT company IDs. "
        "Do not return founder IDs, names, or any other columns. "
        f"Question: {user_q}"
    )

def extract_company_ids(sql: str, df: pd.DataFrame) -> list[int]:
    """Return company IDs from the model result; map founder IDs to companies if needed."""
    if df is None or df.empty:
        return []

    # Prefer direct company id columns
    for col in ("id", "company_id"):
        if col in df.columns:
            return (
                pd.to_numeric(df[col], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )

    # If we got founder IDs, map them to company IDs
    founder_col = next((c for c in ("profileId", "founder_id") if c in df.columns), None)
    if founder_col:
        founder_ids = (
            pd.to_numeric(df[founder_col], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if founder_ids:
            inlist = ",".join(map(str, founder_ids))
            mapped = vn.run_sql(
                f"""
                SELECT DISTINCT c.id
                FROM companies c
                JOIN company_founders cf ON c.id = cf.company_id
                WHERE cf.founder_id IN ({inlist})
                """
            )
            if mapped is not None and "id" in mapped.columns:
                return mapped["id"].astype(int).unique().tolist()

    # Last resort: wrap the model's SQL as a subquery if it obviously targets founders
    if (" founders " in sql.lower()) or (" founder_" in sql.lower()) or ("profileid" in sql.lower()):
        wrapped = vn.run_sql(
            f"""
            SELECT DISTINCT c.id
            FROM companies c
            JOIN company_founders cf ON c.id = cf.company_id
            WHERE cf.founder_id IN ({sql.rstrip(';')})
            """
        )
        if wrapped is not None and "id" in wrapped.columns:
            return wrapped["id"].astype(int).unique().tolist()

    return []


@app.post("/query")
async def query(request: QueryRequest):
    try:
        # 1) steer the model
        q = force_company_id_question(request.question)

        # 2) ask Vanna
        sql, df, _ = vn.ask(q, visualize=False)

        # 3) robust extraction
        company_ids = extract_company_ids(sql, df)

        return {"sql": sql, "company_ids": company_ids, "error": None}
    except Exception as e:
        print(f"Error during query: {e}")
        return {"sql": "", "company_ids": [], "error": str(e)}

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