import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml

# ==============================
# Logging Setup
# ==============================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

# File handler
file_handler = logging.FileHandler(os.path.join(LOG_DIR, "data_ingestion.log"))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ==============================
# Load Parameters
# ==============================

def load_params(params_path: str) -> dict:
    """Load parameters from params.yaml"""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters loaded from %s", params_path)
        return params
    except Exception as e:
        logger.error("Error loading parameters: %s", e)
        raise


# ==============================
# Load Data
# ==============================

def load_data(data_url: str) -> pd.DataFrame:
    """Load CSV data from URL"""
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded from %s", data_url)
        return df
    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise


# ==============================
# Preprocess Data
# ==============================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess dataset"""
    try:
        df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
        df = df.rename(columns={"v1": "target", "v2": "text"})
        logger.debug("Data preprocessing completed")
        return df
    except Exception as e:
        logger.error("Error during preprocessing: %s", e)
        raise


# ==============================
# Save Data (DVC SAFE)
# ==============================

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Save train and test data to data/raw"""
    try:
        raw_data_path = os.path.join("data", "raw")
        os.makedirs(raw_data_path, exist_ok=True)

        train_df.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_df.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logger.debug("Train and test data saved to %s", raw_data_path)
    except Exception as e:
        logger.error("Error saving data: %s", e)
        raise


# ==============================
# Main Pipeline
# ==============================

def main():
    try:
        # Parameters
        test_size = 0.21
        random_state = 42

        source_url = (
            "https://raw.githubusercontent.com/"
            "vikashishere/Datasets/main/spam.csv"
        )

        # Pipeline steps
        df = load_data(source_url)
        df = preprocess_data(df)

        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )

        save_data(train_df, test_df)

        logger.debug("Data ingestion pipeline completed successfully")

    except Exception as e:
        logger.error("Data ingestion failed: %s", e)
        raise


# ==============================
# Entry Point
# ==============================

if __name__ == "__main__":
    main()
