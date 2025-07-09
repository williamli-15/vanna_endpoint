import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
from dotenv import load_dotenv

from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please add it to your .env file.")

# --- DEPLOYMENT-READY PATH CONFIGURATION ---
DATA_DIR = os.environ.get('DATA_DIR', '.')
CHROMA_PATH = os.path.join(DATA_DIR, 'chroma') 
DB_PATH = os.path.join(DATA_DIR, 'yc_companies.db')

print(f"Using data directory: {DATA_DIR}")
print(f"ChromaDB path: {CHROMA_PATH}")
print(f"SQLite DB path: {DB_PATH}")
# --- END DEPLOYMENT CONFIGURATION ---

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        # Pass the CHROMA_PATH to the vector store initializer
        ChromaDB_VectorStore.__init__(self, config={'path': CHROMA_PATH})
        OpenAI_Chat.__init__(self, config=config)

vn = MyVanna(config={'api_key': OPENAI_API_KEY, 'model': 'gpt-4o'})

# Connect to our SQLite database using the configured path
vn.connect_to_sqlite(DB_PATH)

app = FastAPI(title="YC Companies Query API")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    sql: str
    company_ids: List[int]
    error: Optional[str] = None

@app.post("/query")
async def query(request: QueryRequest):
    try:
        sql, df, _ = vn.ask(request.question, visualize=False)
        company_ids = []
        if df is not None and not df.empty:
            id_col = next((col for col in ['id', 'company_id'] if col in df.columns), 'id')
            company_ids = df[id_col].unique().tolist() # Use unique() to prevent duplicates
        return {"sql": sql, "company_ids": company_ids, "error": None}
    except Exception as e:
        print(f"Error during query: {e}")
        return {"sql": "", "company_ids": [], "error": str(e)}

@app.post("/train")
async def train_model():
    # 1. Train on DDL
    df_ddl = vn.run_sql("SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'")
    for ddl in df_ddl['sql'].to_list():
        vn.train(ddl=ddl)

    # 2. Train on Documentation for relationship and synonym mapping
    vn.train(documentation="""
    - The user wants a list of companies. Your query should ALWAYS return a single column of unique company IDs named 'id'. ALWAYS use SELECT DISTINCT c.id.
    - `companies.id` links to `company_founders.company_id` to find a company's founders.
    - `founders.profileId` links to `company_founders.founder_id` and is the key for all founder-related tables.
    
    - SYNONYMS:
    - 'MIT' refers to 'Massachusetts Institute of Technology'. Use `fe.school LIKE '%Massachusetts Institute of Technology%'`.
    - 'Berkeley' or 'UC Berkeley' refers to 'University of California, Berkeley'. Use `fe.school LIKE '%Berkeley%'`.
    - 'Stanford' refers to 'Stanford University'.
    - 'FAANG' refers to the companies 'Meta', 'Apple', 'Amazon', 'Netflix', and 'Google'. Use `fe.company_name IN ('Meta', 'Apple', 'Amazon', 'Netflix', 'Google')`.
    - 'Engineer' can mean 'Software Engineer', 'SDE', 'Backend Engineer', etc. Use `fe.title LIKE '%Engineer%'`.
    - 'Fintech' refers to the 'Fintech' industry. Use `ci.industry = 'Fintech'`.
    - 'AI' refers to the 'Artificial Intelligence' industry. Use `ci.industry = 'Artificial Intelligence'`.
    """)

    # 3. Train on High-Quality Question/SQL Pairs
    # These examples teach Vanna how to handle shorthand and complex joins.
    example_queries = [
        # Handling shorthand for schools
        {"question": "MIT", "sql": "SELECT DISTINCT c.id FROM companies AS c JOIN company_founders AS cf ON c.id = cf.company_id JOIN founder_education AS fe ON cf.founder_id = fe.profileId WHERE fe.school LIKE '%Massachusetts Institute of Technology%';"},
        
        # Handling shorthand for work experience
        {"question": "founder worked at Google", "sql": "SELECT DISTINCT c.id FROM companies AS c JOIN company_founders AS cf ON c.id = cf.company_id JOIN founder_experience AS fe ON cf.founder_id = fe.profileId WHERE fe.company_name = 'Google';"},
        
        # Handling shorthand for industry
        {"question": "fintech", "sql": "SELECT DISTINCT c.id FROM companies AS c JOIN company_industries AS ci ON c.id = ci.company_id WHERE ci.industry = 'Fintech';"},

        # Handling complex query with multiple conditions
        {"question": "Fintech companies with founders from FAANG", "sql": "SELECT DISTINCT c.id FROM companies AS c JOIN company_industries AS ci ON c.id = ci.company_id JOIN company_founders AS cf ON c.id = cf.company_id JOIN founder_experience AS fe ON cf.founder_id = fe.profileId WHERE ci.industry = 'Fintech' AND fe.company_name IN ('Meta', 'Apple', 'Amazon', 'Netflix', 'Google');"},

        # Handling simple filtering
        {"question": "companies in San Francisco", "sql": "SELECT id FROM companies WHERE city = 'San Francisco';"},
    ]
    for example in example_queries:
        vn.train(question=example["question"], sql=example["sql"])
    
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