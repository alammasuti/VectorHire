from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL
from app.config import DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD


def get_engine() -> Engine:
    url = URL.create(
        "mssql+pyodbc",
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_SERVER,
        database=DB_NAME,
        query={
            "driver": "ODBC Driver 17 for SQL Server",
            "TrustServerCertificate": "yes",
        },
    )
    print(f"Connecting to database at {url}") 
    return create_engine(url)
