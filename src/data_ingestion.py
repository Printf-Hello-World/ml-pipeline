from sqlalchemy import create_engine
import pandas as pd
import os
from src.utils.utils import logger

class DataIngestor:
    def __init__(
        self, 
        db_path: str
    ):
        # assign the db_path
        self.db_path = db_path

    def ingest(self) -> pd.DataFrame:

        current_dir = os.getcwd()
        
        # Ensure the database path is valid (join current directory with the relative db path)
        full_path = os.path.join(current_dir, self.db_path)
        
        if os.path.exists(os.path.join(current_dir, "raw_data/raw_data.csv")):
            logger.info("Existing data file found, fetching it instead!")
            return pd.read_csv(os.path.join(current_dir, "raw_data/raw_data.csv"),encoding='utf-8')
        
        # Normalize the path to use forward slashes (SQLite URI requires this)
        full_path = full_path.replace("\\", "/")
        # Create the SQLite engine with the full path
        engine = create_engine(f"sqlite:///{full_path}")

        ## CHANGE QUERY 
        query = "SELECT * FROM XXX;"   
        df = pd.read_sql(query, engine)

        # Create the folder if it doesn't exist
        os.makedirs('raw_data', exist_ok=True)

        # Save the DataFrame to a CSV file inside the 'raw_data' folder
        df.to_csv('raw_data/raw_data.csv', index=False)
    
        return df
    

