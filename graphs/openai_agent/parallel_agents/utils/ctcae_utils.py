"""
Utility functions for working with CTCAE data.
"""

import os
import json
import logging
import glob
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Define the base path to CTCAE data
CTCAE_BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths to the subset files
CTCAE_SUBSET_DIR = os.path.join(CTCAE_BASE_PATH, "utils", "ctcae_subsets")
CTCAE_COMBINED_PATH = os.path.join(CTCAE_SUBSET_DIR, "ctcae_subset_with_grades.json")
CTCAE_FULL_PATH = os.path.join(CTCAE_SUBSET_DIR, "ctcae.json")

# Fallback data in case files can't be loaded
FALLBACK_CTCAE_DATA = {
    "Pneumonitis": "Acute, subacute, or chronic inflammation of the lung parenchyma.",
    "Myocarditis": "Inflammation of the myocardium.",
    "Colitis": "Inflammation of the colon.",
    "Thyroiditis": "Inflammation of the thyroid gland.",
    "Hepatitis": "Inflammation of the liver.",
    "Dermatitis": "Inflammation of the skin.",
    "Nephritis": "Inflammation of the kidney.",
    "Neurological": "Disorders of the nervous system.",
    "Endocrine": "Disorders of the endocrine system.",
    "Haematological": "Disorders of the blood and blood-forming organs.",
    "Musculoskeletal": "Disorders of the muscles and skeleton.",
}


def get_category_filepath_mapping() -> Dict[str, str]:
    """
    Dynamically builds a mapping of all category names to their respective file paths.

    Returns:
        Dict[str, str]: A dictionary mapping category names to file paths
    """
    mapping = {}

    # Debug the CTCAE_SUBSET_DIR
    logger.debug(f"Building category mapping from directory: {CTCAE_SUBSET_DIR}")
    if not os.path.exists(CTCAE_SUBSET_DIR):
        logger.error(f"CTCAE_SUBSET_DIR does not exist: {CTCAE_SUBSET_DIR}")
        return mapping

    # Find all JSON files in the subset directory except the combined files
    subset_files = glob.glob(os.path.join(CTCAE_SUBSET_DIR, "*.json"))
    subset_files = [
        f
        for f in subset_files
        if not f.endswith("ctcae_subset_with_grades.json")
        and not f.endswith("ctcae.json")
    ]

    logger.debug(f"Found {len(subset_files)} subset files: {subset_files}")

    for filepath in subset_files:
        filename = os.path.basename(filepath)
        # Extract category name from filename (remove _associated_symptoms.json)
        if "_associated_symptoms.json" in filename:
            category = filename.replace("_associated_symptoms.json", "")
            mapping[category] = filepath
            logger.debug(f"Added mapping: {category} -> {filepath}")

    logger.debug(f"Final category mapping: {mapping}")
    return mapping


def get_ctcae_data() -> Dict[str, Any]:
    """
    Load combined CTCAE data from the JSON file or fallback to simplified data if not available.

    Returns:
        Dict[str, Any]: A dictionary containing all CTCAE categories and their terms
    """
    try:
        # Try to load the combined CTCAE data from the file
        if os.path.exists(CTCAE_COMBINED_PATH):
            with open(CTCAE_COMBINED_PATH, "r") as f:
                data = json.load(f)
            logger.debug(
                f"Successfully loaded combined CTCAE data from {CTCAE_COMBINED_PATH}"
            )
            return data
        else:
            logger.warning(
                f"Combined CTCAE data file not found at {CTCAE_COMBINED_PATH}. Using fallback data."
            )
            return FALLBACK_CTCAE_DATA
    except Exception as e:
        logger.error(
            f"Error loading combined CTCAE data: {str(e)}. Using fallback data."
        )
        return FALLBACK_CTCAE_DATA


def get_ctcae_subset(category: str) -> Dict[str, Any]:
    """
    Load CTCAE data for a specific symptom category.

    Args:
        category: The symptom category to load (pneumonitis, myocarditis, colitis, etc.)

    Returns:
        Dict[str, Any]: A dictionary containing the CTCAE data for the specified category
    """
    category = category.lower()
    logger.debug(f"Getting CTCAE data for category: {category}")
    try:
        # Debug the path configuration
        logger.debug(f"CTCAE_BASE_PATH: {CTCAE_BASE_PATH}")
        logger.debug(f"CTCAE_SUBSET_DIR: {CTCAE_SUBSET_DIR}")

        if not os.path.exists(CTCAE_SUBSET_DIR):
            error_msg = f"ERROR: CTCAE_SUBSET_DIR does not exist: {CTCAE_SUBSET_DIR}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # List all files in the directory to debug
        files = os.listdir(CTCAE_SUBSET_DIR)
        logger.debug(f"Files in CTCAE_SUBSET_DIR: {files}")

        # Get the mapping of categories to file paths
        category_mapping = get_category_filepath_mapping()
        if not category_mapping:
            error_msg = f"ERROR: No category mappings found in {CTCAE_SUBSET_DIR}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"Category mapping: {category_mapping}")

        # Try to find an exact match
        path = None
        for cat, file_path in category_mapping.items():
            if cat.lower() == category:
                path = file_path
                logger.debug(f"Found exact match for category {category}: {path}")
                break

        # If no exact match, try partial match
        if path is None:
            for cat, file_path in category_mapping.items():
                if category in cat.lower() or cat.lower() in category:
                    path = file_path
                    logger.debug(
                        f"Found partial match for category {category}: {path} (matched with {cat})"
                    )
                    break

        # If still no match, use combined data
        if path is None:
            error_msg = f"ERROR: No specific file found for category: {category}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Load the data from the file
        if not os.path.exists(path):
            error_msg = f"ERROR: File not found at path: {path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        with open(path, "r") as f:
            data = json.load(f)

        if not data:
            error_msg = f"ERROR: Empty data loaded from {path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"Successfully loaded CTCAE data for {category} from {path}")
        logger.debug(f"Keys in loaded data: {list(data.keys())}")

        # Validate data structure
        if not isinstance(data, dict):
            error_msg = f"ERROR: Data from {path} is not a dictionary"
            logger.error(error_msg)
            raise TypeError(error_msg)

        return data
    except Exception as e:
        logger.error(f"ERROR in get_ctcae_subset: {str(e)}")
        raise


def format_ctcae_data_for_prompt(ctcae_data: Dict[str, Any]) -> str:
    """
    Format CTCAE data as a string for inclusion in prompts.

    Args:
        ctcae_data: Dictionary containing CTCAE data

    Returns:
        str: Formatted string representation of the CTCAE data
    """
    formatted_data = []

    # Handle the combined data format
    for category, terms in ctcae_data.items():
        if isinstance(terms, str):
            # Handle the fallback format
            formatted_data.append(f"HEADER: {category}\nDEFINITION: {terms}")
        else:
            # Handle the new format with nested terms
            for term, details in terms.items():
                definition = details.get("Definition", "")
                formatted_data.append(f"HEADER: {term}\nDEFINITION: {definition}")

    return "\n\n".join(formatted_data)


def format_ctcae_grades_for_prompt(ctcae_data: Dict[str, Any], category: str) -> str:
    """
    Format CTCAE grade information for a specific category as a string for inclusion in prompts.

    Args:
        ctcae_data: Dictionary containing CTCAE data
        category: The symptom category to format grades for

    Returns:
        str: Formatted string representation of the CTCAE grade information
    """
    formatted_data = []

    # Extract the category key from the data
    category_key = None
    for key in ctcae_data.keys():
        if category.lower() in key.lower():
            category_key = key
            break

    if not category_key:
        logger.warning(f"Category {category} not found in CTCAE data")
        return ""

    # Format the grade information for each term in the category
    terms = ctcae_data.get(category_key, {})
    for term, details in terms.items():
        if isinstance(details, dict):
            grade_info = [
                f"TERM: {term}",
                f"DEFINITION: {details.get('Definition', '')}",
                f"GRADE 1: {details.get('Grade 1', '')}",
                f"GRADE 2: {details.get('Grade 2', '')}",
                f"GRADE 3: {details.get('Grade 3', '')}",
                f"GRADE 4: {details.get('Grade 4', '')}",
                f"GRADE 5: {details.get('Grade 5', '')}",
            ]
            formatted_data.append("\n".join(grade_info))

    return "\n\n".join(formatted_data)


def get_available_categories() -> List[str]:
    """
    Get a list of all available symptom categories.

    Returns:
        List[str]: A list of category names
    """
    return list(get_category_filepath_mapping().keys())


def get_terms_and_definitions(category: str) -> Dict[str, str]:
    """
    Get just the terms and definitions for a specific category.

    Args:
        category: The symptom category to load

    Returns:
        Dict[str, str]: A dictionary mapping terms to their definitions
    """
    data = get_ctcae_subset(category)
    terms_and_defs = {}

    for cat, terms in data.items():
        if isinstance(terms, str):
            # Handle fallback format
            terms_and_defs[cat] = terms
        else:
            # Handle full format
            for term, details in terms.items():
                terms_and_defs[term] = details.get("Definition", "")

    return terms_and_defs


def get_terms_definitions_and_grades(category: str) -> Dict[str, Dict[str, str]]:
    """
    Get terms, definitions, and grades for a specific category.

    Args:
        category: The symptom category to load

    Returns:
        Dict[str, Dict[str, str]]: A dictionary mapping terms to their details including definition and grades
    """
    logger.debug(f"Getting terms, definitions, and grades for category: {category}")

    try:
        data = get_ctcae_subset(category)
        logger.debug(f"Data keys for {category}: {list(data.keys())}")

        if not data:
            error_msg = f"ERROR: Empty data returned for category: {category}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        full_details = {}

        # Track if main category term was found
        main_category_found = False

        # Debug the structure of the data
        for cat, terms in data.items():
            logger.debug(f"Processing category: {cat}")
            logger.debug(f"Terms type: {type(terms)}")

            if category.lower() in cat.lower():
                main_category_found = True

            if isinstance(terms, dict):
                logger.debug(
                    f"Terms keys: {list(terms.keys())[:10]}"
                )  # Show first 10 keys

                # Check if the category term is directly in the terms
                category_term = category.capitalize()
                if category_term in terms:
                    logger.debug(
                        f"Found direct match for category term: {category_term}"
                    )
                    main_category_found = True

                for term, details in terms.items():
                    logger.debug(f"Processing term: {term}")

                    # Check if this term is the main category
                    if category.lower() == term.lower():
                        main_category_found = True

                    logger.debug(f"Details type: {type(details)}")

                    if isinstance(details, dict):
                        logger.debug(f"Details keys: {list(details.keys())}")
                        definition = details.get("Definition", "")

                        if not definition:
                            logger.warning(f"Empty definition for term: {term}")
                        else:
                            logger.debug(
                                f"Definition for {term}: {definition[:100]}..."
                            )

                        # Store the term with its details
                        full_details[term] = {
                            "Definition": definition,
                            "Grade 1": details.get("Grade 1", ""),
                            "Grade 2": details.get("Grade 2", ""),
                            "Grade 3": details.get("Grade 3", ""),
                            "Grade 4": details.get("Grade 4", ""),
                            "Grade 5": details.get("Grade 5", ""),
                        }
                    elif isinstance(details, str):
                        # Handle the case where details is a string (fallback format)
                        logger.debug(
                            f"Term {term} has string definition: {details[:100]}..."
                        )
                        full_details[term] = {
                            "Definition": details,
                            "Grade 1": "",
                            "Grade 2": "",
                            "Grade 3": "",
                            "Grade 4": "",
                            "Grade 5": "",
                        }
                    else:
                        logger.warning(
                            f"Unexpected details type for term {term}: {type(details)}"
                        )
            elif isinstance(terms, str):
                # Handle the case where terms is a string (fallback format)
                logger.debug(f"Category {cat} has string definition: {terms[:100]}...")
                # Use the category name as the term for simple fallback data
                full_details[cat] = {
                    "Definition": terms,
                    "Grade 1": "",
                    "Grade 2": "",
                    "Grade 3": "",
                    "Grade 4": "",
                    "Grade 5": "",
                }
                if category.lower() == cat.lower():
                    main_category_found = True
            else:
                logger.warning(
                    f"Terms is not a dictionary for category {cat}, it is {type(terms)}"
                )

        # Also include the category itself as a term if not already included
        if not main_category_found:
            logger.warning(
                f"Main category term '{category}' not found in data, adding it explicitly"
            )
            main_category_term = category.capitalize()

            # Get definition from a related term if possible
            definition = ""
            for term, details in full_details.items():
                if category.lower() in term.lower():
                    definition = details.get("Definition", "")
                    logger.debug(
                        f"Using definition from related term '{term}' for main category"
                    )
                    break

            if not definition:
                logger.warning(f"No related term found for main category '{category}'")

            full_details[main_category_term] = {
                "Definition": definition,
                "Grade 1": "",
                "Grade 2": "",
                "Grade 3": "",
                "Grade 4": "",
                "Grade 5": "",
            }

        # Hard validation - raise error if no terms found
        if not full_details:
            error_msg = f"ERROR: No terms found for category: {category}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate definitions
        empty_defs = [
            term
            for term, details in full_details.items()
            if not details.get("Definition")
        ]
        if empty_defs:
            logger.warning(
                f"Empty definitions found for terms in {category}: {empty_defs}"
            )
            # Don't raise an error here, just log the warning

        # Log success with term count
        logger.debug(f"Successfully loaded {len(full_details)} terms for {category}")
        logger.debug(f"First few terms: {list(full_details.keys())[:5]}")

        # Final hard validation - must have at least one term with definition
        terms_with_defs = [
            term for term, details in full_details.items() if details.get("Definition")
        ]
        if not terms_with_defs:
            error_msg = (
                f"ERROR: No terms with definitions found for category: {category}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        return full_details
    except Exception as e:
        logger.error(f"ERROR in get_terms_definitions_and_grades: {str(e)}")
        raise
