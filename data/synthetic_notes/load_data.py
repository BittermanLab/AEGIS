"""
Load synthetic patient data for testing.
Simplified implementation without Langchain dependencies.
"""

import os
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def load_patient_data(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load synthetic patient data from the given directory.

    Args:
        data_dir: Path to directory containing patient data

    Returns:
        List of patient data dictionaries
    """
    patient_data = []

    try:
        # Check if the directory exists
        if not os.path.exists(data_dir):
            logger.warning(f"Data directory does not exist: {data_dir}")
            return []

        # Get all JSON files in the directory
        json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

        if not json_files:
            logger.warning(f"No JSON files found in data directory: {data_dir}")
            return []

        # Load each JSON file
        for json_file in json_files:
            # Skip metadata files
            if json_file == "batch_metadata.json":
                logger.debug(f"Skipping metadata file: {json_file}")
                continue

            file_path = os.path.join(data_dir, json_file)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Validate that the data has the expected structure
                if "note" not in data:
                    logger.warning(f"Skipping {json_file}: missing 'note' field")
                    continue

                # Add file info to the data
                data["file_id"] = json_file.split(".")[0]
                data["filename"] = json_file

                patient_data.append(data)
                logger.debug(f"Loaded patient data from {file_path}")
            except Exception as e:
                logger.error(f"Error loading patient data from {file_path}: {str(e)}")

        logger.info(f"Loaded {len(patient_data)} patient data files")
        return patient_data

    except Exception as e:
        logger.error(f"Error loading patient data: {str(e)}")
        return []


if __name__ == "__main__":
    # Test loading the data
    logging.basicConfig(level=logging.INFO)
    data = load_patient_data("data/synthetic_notes")
    print(f"Loaded {len(data)} patient data files")
    if data:
        print(f"Sample data: {data[0].keys()}")
