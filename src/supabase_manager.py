import os
from dotenv import load_dotenv
from supabase import create_client, Client
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()


class SupabaseManager:
    def __init__(self):
        self.supabase: Client = create_client(
            os.getenv('SUPABASE_URL'),
            os.getenv('SUPABASE_SERVICE_KEY')
        )
        self.engine = create_engine(os.getenv('SUPABASE_DB_URL'))
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_db_session(self):
        return self.SessionLocal()

    def upload_adapter(self, local_path: str, storage_path: str, bucket: str = 'adapters'):
        with open(local_path, 'rb') as f:
            response = self.supabase.storage.from_(bucket).upload(
                path=storage_path,
                file=f,
                file_options={"content-type": "application/gzip"}
            )
        return response

    def get_adapter_url(self, storage_path: str, bucket: str = 'adapters'):
        return self.supabase.storage.from_(bucket).get_public_url(storage_path)
