import json
import csv
import os

def json_to_csv(json_file_path, csv_file_path):
    """
    Convert YC company data from JSON to CSV format
    
    Args:
        json_file_path (str): Path to the JSON file
        csv_file_path (str): Path where the CSV file will be saved
    """
    # Read the JSON file
    print(f"Reading JSON file: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Debug information about the loaded data
    if isinstance(data, dict) and 'data' in data:
        print("JSON has a 'data' field that contains the companies")
        data = data.get('data', [])
    
    # Check if data exists and is a list
    if not data:
        print("Error: No data found in the JSON file")
        return
    
    if not isinstance(data, list):
        print(f"Error: Invalid JSON data format. Expected list, got {type(data)}")
        if isinstance(data, dict):
            print(f"Available keys: {list(data.keys())}")
        return
    
    # Define the fields we want to extract
    # You can modify this list to include more or fewer fields
    fields = [
        'id', 'name', 'slug', 'batch_name', 'one_liner', 'website', 
        'year_founded', 'team_size', 'location', 'city', 'country',
        'linkedin_url', 'twitter_url', 'github_url'
    ]
    
    # Create the CSV file
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        
        # Write the header
        writer.writeheader()
        
        # Write each company's data
        for company in data:
            # Create a row with only the fields we want
            row = {field: company.get(field, '') for field in fields}
            writer.writerow(row)
    
    print(f"Successfully converted {len(data)} companies to CSV at {csv_file_path}")

if __name__ == "__main__":
    # File paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(current_dir, "yc_json")
    csv_file = os.path.join(current_dir, "yc_companies.csv")
    
    # Convert JSON to CSV
    json_to_csv(json_file, csv_file)
