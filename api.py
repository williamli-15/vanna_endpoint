import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
from dotenv import load_dotenv

# For the new approach with OpenAI and ChromaDB
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Create a custom Vanna class that uses OpenAI for chat and ChromaDB for vector storage
class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

# Initialize Vanna with your OpenAI API key
vn = MyVanna(config={
    'api_key': OPENAI_API_KEY,
    'model': 'gpt-4-turbo'  # You can use a different model if needed
})

# Connect to SQLite database
db_path = 'yc_companies.db'
print(f"Connecting to local SQLite database: {db_path}")
vn.connect_to_sqlite(db_path)

# Create FastAPI app
app = FastAPI(
    title="YC Companies Query API",
    description="API endpoint for natural language queries on YC Companies database"
)

# Request/Response models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    sql: str
    data: List[Dict[str, Any]]
    row_count: int
    error: Optional[str] = None

@app.get("/")
async def root():
    return {
        "message": "YC Companies Query API",
        "endpoints": {
            "/query": "POST - Execute natural language queries",
            "/test": "GET - Test the Vanna connection",
            "/training-status": "GET - Check training data status"
        }
    }

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Execute a natural language query using Vanna
    
    Example questions:
    - "What companies are located in San Francisco?"
    - "Which companies have the largest teams?"
    - "List companies by their YC batch"
    - "Show me all companies founded after 2020"
    """
    try:
        # Run the query using Vanna's ask method
        sql, df, _ = vn.ask(request.question, visualize=False)
        
        # Convert DataFrame to list of dicts
        if df is not None and not df.empty:
            # Handle NaN values
            df = df.where(pd.notnull(df), None)
            data = df.to_dict(orient='records')
            row_count = len(df)
        else:
            data = []
            row_count = 0
        
        return {
            "question": request.question,
            "sql": sql,
            "data": data,
            "row_count": row_count,
            "error": None
        }
        
    except Exception as e:
        return {
            "question": request.question,
            "sql": "",
            "data": [],
            "row_count": 0,
            "error": str(e)
        }

@app.get("/test")
async def test_query():
    """
    Test endpoint with a sample query
    """
    try:
        sql, df, _ = vn.ask("Show me 5 companies", visualize=False)
        
        return {
            "status": "success",
            "sql": sql,
            "row_count": len(df) if df is not None else 0,
            "sample_data": df.head().to_dict(orient='records') if df is not None else []
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/training-status")
async def training_status():
    """
    Check the training data status
    """
    try:
        training_data = vn.get_training_data()
        return {
            "status": "success",
            "training_items": len(training_data),
            "has_training_data": len(training_data) > 0
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/train")
async def train_model():
    """
    Train or retrain the Vanna model on your database schema
    """
    try:
        # Get all DDL statements from SQLite master table
        df_ddl = vn.run_sql("SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL")
        
        # Train on each DDL statement
        for ddl in df_ddl['sql'].to_list():
            vn.train(ddl=ddl)
        
        # Add example queries for training
        example_queries = [
            ("What companies are located in San Francisco?", 
             "SELECT name, one_liner, website FROM companies WHERE city = 'San Francisco';"),
            ("List companies by their YC batch", 
             "SELECT name, batch_name FROM companies ORDER BY batch_name;"),
            ("Which companies have the largest teams?", 
             "SELECT name, team_size FROM companies ORDER BY team_size DESC;"),
        ]
        
        for question, sql in example_queries:
            vn.train(sql=sql, question=question)
        
        # Add documentation
        vn.train(documentation="""The 'companies' table contains information about YC (Y Combinator) companies including:
        - id: Unique identifier for each company
        - name: Company name
        - slug: URL-friendly version of company name
        - batch_name: The YC batch the company was part of
        - one_liner: Brief description of the company
        - website: Company website URL
        - year_founded: Year the company was founded
        - team_size: Number of employees
        - location: Company location
        - city: City where the company is based
        - country: Country where the company is based
        - linkedin_url: Company LinkedIn profile
        - twitter_url: Company Twitter/X profile
        - github_url: Company GitHub profile
        """)
        
        training_data = vn.get_training_data()
        
        return {
            "status": "success",
            "message": "Model trained successfully",
            "training_items": len(training_data)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting YC Companies API...")
    print(f"Database: {db_path}")
    print(f"OpenAI API Key: {'Set' if OPENAI_API_KEY else 'Not Set'}")
    
    # Check training status on startup
    training_data = vn.get_training_data()
    print(f"Training items: {len(training_data)}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)