import csv
from supabase import create_client, Client
import pandas as pd

supabase_url = "https://wajalsbafyrraeqnhngp.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndhamFsc2JhZnlycmFlcW5obmdwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDU1ODIzMjMsImV4cCI6MjAyMTE1ODMyM30.KbD9ZMOR3zEjbeQW9CfN80rU790GsZBNy1dl6Cs3q7M"
supabase: Client = create_client(supabase_url, supabase_key)


table_name = "yelp_academic_dataset_business_cleaned"


csv_file_path = "cleaned_data\yelp_academic_dataset_business_cleaned.csv"


with open(csv_file_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=',')
    list_of_dict = [dict(row) for row in reader]


chunk_size = 1000  


chunks = [list_of_dict[i:i + chunk_size] for i in range(0, len(list_of_dict), chunk_size)]

    
print(len(chunks))
for chunk in chunks:
    data, count = supabase.table(table_name).insert(chunk).execute()

