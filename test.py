import os
from dotenv import load_dotenv

# For the new approach with OpenAI and ChromaDB
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variable (you'll need to add this to your .env file)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

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

# Connect to SQLite database using local file path
db_path = 'yc_companies.db'  # Local SQLite database file
print(f"Connecting to local SQLite database: {db_path}")
vn.connect_to_sqlite(db_path)  # Vanna works with local file paths too

# Check if we already have training data to decide whether to train
training_data = vn.get_training_data()

# Set this to True to force retraining even if data exists
force_retrain = False

# Only train if we don't have training data or if force_retrain is True
if len(training_data) == 0 or force_retrain:
    print("Training Vanna on the database schema...")
    
    # Get all DDL statements from SQLite master table
    print("Getting DDL statements from database...")
    df_ddl = vn.run_sql("SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL")
    
    # Print what we found
    print(f"Found {len(df_ddl)} DDL statements")
    print(df_ddl)
    
    # Train on each DDL statement
    for ddl in df_ddl['sql'].to_list():
        print(f"Training on: {ddl[:50]}...")
        vn.train(ddl=ddl)
    
    # Add some example queries for the companies table
    print("Adding example queries for training...")
    
    # Query to find companies in San Francisco
    sf_query = "SELECT name, one_liner, website FROM companies WHERE city = 'San Francisco';"
    vn.train(sql=sf_query, question="What companies are located in San Francisco?")
    
    # Query to find companies by batch
    batch_query = "SELECT name, batch_name FROM companies ORDER BY batch_name;"
    vn.train(sql=batch_query, question="List companies by their YC batch")
    
    # Query to find companies by team size
    team_query = "SELECT name, team_size FROM companies ORDER BY team_size DESC;"
    vn.train(sql=team_query, question="Which companies have the largest teams?")
    
    # Add documentation about the data
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
    
    # Check what training data we have after training
    print("\nTraining data available after training:")
    training_data = vn.get_training_data()
    print(f"Total training items: {len(training_data)}")
else:
    print(f"Skipping training - {len(training_data)} training items already exist")
    print("Set force_retrain = True to retrain if needed")


# Function to run a query and display only the results (no visualization)
def run_query(question):
    # Temporarily redirect stdout to suppress debug messages
    import sys, io
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    # Run the query
    sql, df, _ = vn.ask(question, visualize=False)
    
    # Restore stdout
    sys.stdout = original_stdout
    
    # Print clean output
    print(f"\n=== Question: {question} ===\n")
    if df is not None and not df.empty:
        print(df)
    else:
        print("No results found.")
    
    return sql, df

# Example queries
run_query("What companies are located in San Francisco?")
run_query("Which companies have the largest teams?")
run_query("Give me a list of computer vision companies")

# Only run the Flask app if requested
run_flask = False  # Change to True if you want to run the web interface

if run_flask:
    # Initialize and run Flask app for a web interface
    from vanna.flask import VannaFlaskApp
    
    # Create the Flask app
    app = VannaFlaskApp(vn, chart=False)
    
    # Run the app when this script is executed directly
    if __name__ == "__main__":
        app.run(debug=True, host='0.0.0.0', port=5000)