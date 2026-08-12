import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # AWS Bedrock Configuration
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")  
    
    MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.meta.llama3-3-70b-instruct-v1:0")
    
    # Embedding Configuration
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    
    # Other Settings
    VECTOR_DB_PATH = "faiss_index" 
    FRONTEND_URL = os.getenv("FRONTEND_URL")
    SERVER_API_KEY_NAME = os.getenv("SERVER_API_KEY_NAME")
    SERVER_API_KEY = os.getenv("SERVER_API_KEY")
    SHEET_CSV_URL = os.getenv("SHEET_CSV_URL")

    @classmethod
    def validate(cls):
        if not cls.AWS_ACCESS_KEY_ID or not cls.AWS_SECRET_ACCESS_KEY:
            raise ValueError("AWS credentials not found! Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        print(f"   AWS Bedrock configured")
        print(f"   Region: {cls.AWS_REGION}")
        print(f"   Model: {cls.MODEL_ID}")

settings = Settings()