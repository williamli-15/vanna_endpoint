import json
import sqlite3
import os

def create_and_populate_db(db_path='yc_companies.db', yc_companies_json='yc_json', linkedin_json='linkedin-data.json'):
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database file: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    print(f"Created new SQLite database: {db_path}")

    try:
        with open(yc_companies_json, 'r', encoding='utf-8') as f:
            companies_data = json.load(f)
            if isinstance(companies_data, dict) and 'data' in companies_data:
                companies_data = companies_data['data']
        with open(linkedin_json, 'r', encoding='utf-8') as f:
            linkedin_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON files: {e}")
        conn.close()
        return

    print(f"Loaded {len(companies_data)} companies and {len(linkedin_data)} LinkedIn profiles.")

    # ### MODIFIED BLOCK START ###
    # The `companies` table now correctly includes the 'slug' column, making it 12 columns total.
    cursor.executescript('''
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY,
            name TEXT,
            slug TEXT,
            batch_name TEXT,
            one_liner TEXT,
            website TEXT,
            long_description TEXT,
            year_founded INTEGER,
            team_size INTEGER,
            location TEXT,
            city TEXT,
            country TEXT
        );
        CREATE TABLE IF NOT EXISTS founders (
            profileId INTEGER PRIMARY KEY, name TEXT, headline TEXT, location TEXT,
            connections INTEGER, followers INTEGER, summary TEXT, current_company TEXT
        );
        CREATE TABLE IF NOT EXISTS company_founders (
            company_id INTEGER, founder_id INTEGER, title TEXT,
            FOREIGN KEY(company_id) REFERENCES companies(id), FOREIGN KEY(founder_id) REFERENCES founders(profileId),
            PRIMARY KEY (company_id, founder_id)
        );
        CREATE TABLE IF NOT EXISTS company_tags (company_id INTEGER, tag TEXT, FOREIGN KEY(company_id) REFERENCES companies(id));
        CREATE TABLE IF NOT EXISTS company_industries (company_id INTEGER, industry TEXT, is_primary BOOLEAN, FOREIGN KEY(company_id) REFERENCES companies(id));
        CREATE TABLE IF NOT EXISTS founder_experience (
            founder_id INTEGER, company_name TEXT, title TEXT, start_date TEXT, end_date TEXT,
            is_current BOOLEAN, duration TEXT, location TEXT, description TEXT,
            FOREIGN KEY(founder_id) REFERENCES founders(profileId)
        );
        CREATE TABLE IF NOT EXISTS founder_education (
            founder_id INTEGER, school TEXT, degree TEXT, field TEXT, start_date TEXT, end_date TEXT,
            FOREIGN KEY(founder_id) REFERENCES founders(profileId)
        );
        CREATE TABLE IF NOT EXISTS founder_skills (founder_id INTEGER, skill TEXT, FOREIGN KEY(founder_id) REFERENCES founders(profileId));
        CREATE TABLE IF NOT EXISTS company_launches (
            company_id INTEGER, launch_id INTEGER, title TEXT, tagline TEXT, total_vote_count INTEGER,
            FOREIGN KEY(company_id) REFERENCES companies(id)
        );
    ''')
    # ### MODIFIED BLOCK END ###
    print("Tables created successfully.")

    try:
        # The insertion logic was already correct, it was the table creation that was wrong.
        # This part remains unchanged.
        for person in linkedin_data:
            cursor.execute('INSERT OR IGNORE INTO founders VALUES (?,?,?,?,?,?,?,?)', (
                person.get('profileId'), person.get('name'), person.get('headline'), person.get('location'),
                person.get('connections'), person.get('followers'), person.get('summary'), person.get('currentCompany')
            ))
            for exp in person.get('experience', []):
                cursor.execute('INSERT INTO founder_experience VALUES (?,?,?,?,?,?,?,?,?)', (
                    person.get('profileId'), exp.get('company'), exp.get('title'), exp.get('startDate'), exp.get('endDate'),
                    exp.get('isCurrent'), exp.get('duration'), exp.get('location'), exp.get('description')
                ))
            for edu in person.get('education', []):
                cursor.execute('INSERT INTO founder_education VALUES (?,?,?,?,?,?)', (
                    person.get('profileId'), edu.get('school'), edu.get('degree'), edu.get('field'), edu.get('startDate'), edu.get('endDate')
                ))
            for skill in person.get('skills', []):
                cursor.execute('INSERT INTO founder_skills VALUES (?,?)', (person.get('profileId'), skill))

        for company in companies_data:
            cursor.execute('INSERT OR IGNORE INTO companies VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', (
                company.get('id'), company.get('name'), company.get('slug'), company.get('batch_name'),
                company.get('one_liner'), company.get('website'), company.get('long_description'),
                company.get('year_founded'), company.get('team_size'), company.get('location'),
                company.get('city'), company.get('country')
            ))
            for founder in company.get('founders', []):
                cursor.execute('INSERT OR IGNORE INTO company_founders VALUES (?,?,?)', (company.get('id'), founder.get('user_id'), founder.get('title')))
            for tag in company.get('tags', []):
                cursor.execute('INSERT INTO company_tags VALUES (?,?)', (company.get('id'), tag))
            for industry in company.get('industries', []):
                cursor.execute('INSERT INTO company_industries VALUES (?,?,?)', (company.get('id'), industry, industry == company.get('primary_industry')))
            for launch in company.get('launches', []):
                cursor.execute('INSERT INTO company_launches VALUES (?,?,?,?,?)', (company.get('id'), launch.get('id'), launch.get('title'), launch.get('tagline'), launch.get('total_vote_count')))

        conn.commit()
        print("Data populated successfully.")
    except Exception as e:
        print(f"An error occurred during data insertion: {e}")
        conn.rollback()
    finally:
        conn.close()
        print("Database connection closed.")


if __name__ == '__main__':
    # This part remains correct
    nextjs_data_path = '/Users/hengxuli/code/yc-x25-market-map-c/public/data'
    vanna_endpoint_path = '.' 
    yc_companies_file = os.path.join(nextjs_data_path, 'spring25_cleaned.json')
    linkedin_file = os.path.join(nextjs_data_path, 'linkedin-data.json')
    db_file_output = os.path.join(vanna_endpoint_path, 'yc_companies.db')
    if not os.path.exists(yc_companies_file) or not os.path.exists(linkedin_file):
        print("ERROR: Could not find one or both of the required JSON files.")
    else:
        create_and_populate_db(
            db_path=db_file_output,
            yc_companies_json=yc_companies_file,
            linkedin_json=linkedin_file
        )