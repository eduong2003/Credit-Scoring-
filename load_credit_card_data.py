import pandas as pd
from sqlalchemy import create_engine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the CSV file into a DataFrame with low_memory=False to suppress DtypeWarning
try:
    df = pd.read_csv(r'H:\Projects\data\train.csv', low_memory=False)
    logger.info("CSV file loaded successfully")
except Exception as e:
    logger.error(f"Error loading CSV file: {e}")

# Create a connection to PostgreSQL
try:
    engine = create_engine('postgresql://postgres:sydney2004@localhost:5432/credit_card_data')
    logger.info("Database connection established successfully")
except Exception as e:
    logger.error(f"Error connecting to the database: {e}")

# Load data into PostgreSQL
try:
    df.to_sql('credit_card_data', engine, if_exists='replace', index=False)
    logger.info("Data loaded into PostgreSQL successfully")
except Exception as e:
    logger.error(f"Error loading data into PostgreSQL: {e}")

