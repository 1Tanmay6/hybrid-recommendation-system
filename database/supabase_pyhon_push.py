import csv
from supabase_py import create_client, Client

# Initialize Supabase client
supabase_url = "https://wajalsbafyrraeqnhngp.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndhamFsc2JhZnlycmFlcW5obmdwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDU1ODIzMjMsImV4cCI6MjAyMTE1ODMyM30.KbD9ZMOR3zEjbeQW9CfN80rU790GsZBNy1dl6Cs3q7M"
supabase: Client = create_client(supabase_url, supabase_key)

# Define the table name in Supabase
table_name = "your-table-name"

# Path to the CSV file
csv_file_path = "database_data\\business_recommendation_system_input.csv"

# Read the CSV file
with open(csv_file_path, "r") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

# Create table schema based on CSV headers
table_schema = ", ".join([f"{header} text" for header in reader.fieldnames])

# Create the table
create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({table_schema});"
supabase.raw(create_table_sql)

# Insert the rows into the table
response = supabase.table(table_name).insert(rows).execute()

# Check if the insertion was successful
if response["status_code"] == 201:
    print("Data inserted successfully!")
else:
    print("Failed to insert data:", response["error"])
