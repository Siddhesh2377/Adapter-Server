# test_connection.py
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

url = os.getenv('SUPABASE_DB_URL')
print(f"Testing connection to: {url.split('@')[1]}")  # Hide password

engine = create_engine(url, pool_pre_ping=True)

try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version()"))
        print("✅ Connection successful!")
        print(result.fetchone()[0])
except Exception as e:
    print(f"❌ Connection failed: {e}")