import argparse
import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union

import pandas as pd
from langchain.schema import HumanMessage

# --- Import necessary libraries for topic modelling and model persistence ---
import openai
import tiktoken
import hdbscan
import umap
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import AzureOpenAI
import os
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import joblib  # for saving and loading models
import numpy as np  # Add import at top of file
import ast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("topic_modeling_monitoring.log"),
    ],  # Specific log file for this script
)
logger = logging.getLogger(__name__)


# --- Configuration dataclass ---
@dataclass
class EvaluationConfig:
    """Configuration for topic modeling and monitoring."""

    results_dir: Path
    evaluation_dir: Path
    parent_dir: Optional[Path] = None
    graph_type: Optional[str] = None
    model_config: Optional[str] = None
    prompt_variant: Optional[str] = None
    model_path: Optional[str] = None  # New: Path to pre-trained models

    def __post_init__(self):
        self.results_dir = Path(self.results_dir)
        self.evaluation_dir = Path(self.evaluation_dir)
        self.parent_dir = self.evaluation_dir.parent
        if self.model_path:
            self.model_path = Path(self.model_path)
            try:
                os.makedirs(self.model_path, exist_ok=True)
            except Exception as e:
                raise ValueError(
                    f"Could not create or access model_path: {self.model_path}. Error: {e}"
                )

        # Define allowed values directly here, as utils.path_utils is removed for self-containment
        GRAPH_TYPES = [
            "agent",
            "zeroshot",
            "extract_plan",
            "full_workflow",
            "guided_full_workflow",
            "task_specific",
        ]  # Example GRAPH_TYPES
        MODEL_CONFIGS = ["config1", "config2"]  # Example MODEL_CONFIGS
        PROMPT_VARIANTS = ["variant1", "variant2"]  # Example PROMPT_VARIANTS

        if self.graph_type and self.graph_type not in GRAPH_TYPES:
            raise ValueError(f"Invalid graph_type: {self.graph_type}")
        if self.model_config and self.model_config not in MODEL_CONFIGS:
            raise ValueError(f"Invalid model_config: {self.model_config}")
        if self.prompt_variant and self.prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(f"Invalid prompt_variant: {self.prompt_variant}")
        if self.model_path and not self.model_path.is_dir():
            os.makedirs(self.model_path, exist_ok=True)
            raise ValueError(
                f"Invalid model_path: {self.model_path}. Path must be a directory."
            )


# --- Helper function for ensuring evaluation directory structure ---
def ensure_evaluation_structure(evaluation_dir: Path):
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "metrics").mkdir(exist_ok=True)
    (evaluation_dir / "monitoring").mkdir(exist_ok=True)
    (evaluation_dir / "topic_models").mkdir(
        exist_ok=True
    )  # Ensure topic_models dir exists


# --- Azure OpenAI Client setup ---
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",
)


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.environ["API_VERSION"],
    azure_endpoint=os.environ["ENDPOINT_URL"],
    azure_ad_token_provider=token_provider,
)


def simple_preprocess_text(text: str) -> List[str]:
    """Simplifies text preprocessing by splitting into sentences."""
    sentences: List[str] = [
        sentence.strip() for sentence in text.strip().split("\n") if sentence.strip()
    ]
    return sentences


def get_openai_embedding(
    text: str, model: str = "text-embedding-3-small-jg"
) -> List[float]:
    """Gets embeddings for a given text using OpenAI's API."""

    try:
        response = client.embeddings.create(input=[text], model=model)
        embedding: List[float] = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        raise


def reduce_dimensions_umap(
    embeddings: List[List[float]],
    n_components: int = 15,
    umap_model=None,  # UMAP model can be provided for prediction
) -> Tuple[List[List[float]], umap.UMAP]:
    """Reduces the dimensionality of embeddings using UMAP.
    If umap_model is provided, it uses it for transformation instead of fitting."""
    if umap_model is None:
        umap_model = umap.UMAP(n_components=n_components, random_state=42)
        reduced_embeddings = umap_model.fit_transform(embeddings).tolist()
        return reduced_embeddings, umap_model  # return model when fitting
    else:
        reduced_embeddings = umap_model.transform(
            embeddings
        ).tolist()  # just transform if model is given
        return reduced_embeddings, umap_model  # return model for consistency


def cluster_embeddings_hdbscan(
    embeddings: List[List[float]],
    hdbscan_model=None,  # HDBSCAN model can be provided for prediction
) -> Tuple[hdbscan.HDBSCAN, List[int]]:
    """Clusters embeddings using HDBSCAN.
    If hdbscan_model is provided, it uses it for prediction instead of fitting."""
    if hdbscan_model is None:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=10, min_samples=5
        )  # Tuned parameters for dataset-level clustering
        clusterer.fit(embeddings)
        labels: List[int] = clusterer.labels_.tolist()
        # Generate prediction data when training
        clusterer.generate_prediction_data()
        return clusterer, labels  # return model when fitting
    else:
        print("Using pre-trained HDBSCAN model for prediction")
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)

        # Generate prediction data if not already generated
        if (
            not hasattr(hdbscan_model, "prediction_data_")
            or hdbscan_model.prediction_data_ is None
        ):
            print("Generating prediction data for HDBSCAN model")
            hdbscan_model.generate_prediction_data()

        # Use approximate_predict for inference with pre-trained model
        labels_array, _ = hdbscan.approximate_predict(hdbscan_model, embeddings_array)
        labels = labels_array.tolist()  # Convert numpy array to list
        return hdbscan_model, labels  # return model for consistency


def calculate_ctfidf_per_cluster(
    sentences: List[str], cluster_labels: List[int], top_n: int = 10
) -> Dict[int, List[Tuple[str, float]]]:
    """Calculates c-TF-IDF scores for cluster keywords."""
    clustered_sentences: Dict[int, List[str]] = {}
    for i, label in enumerate(cluster_labels):
        if label not in clustered_sentences:
            clustered_sentences[label] = []
        clustered_sentences[label].append(sentences[i])

    cluster_texts: Dict[int, str] = {}
    for label, sentence_list in clustered_sentences.items():
        if label != -1:
            cluster_texts[label] = " ".join(sentence_list)

    tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(cluster_texts.values())
    feature_names = tfidf_vectorizer.get_feature_names_out()

    cluster_keywords: Dict[int, List[Tuple[str, float]]] = {}
    for label, text in cluster_texts.items():
        label_index = list(cluster_texts.keys()).index(label)
        tfidf_scores = tfidf_matrix[label_index].toarray()[0]
        keyword_scores = sorted(
            zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True
        )
        cluster_keywords[label] = keyword_scores[:top_n]
    return cluster_keywords


def run_topic_modeling(
    sentences: List[str], umap_model=None, hdbscan_model=None
) -> Tuple[
    Dict[str, Dict[str, Union[List[str], List[Any]]]], umap.UMAP, hdbscan.HDBSCAN
]:
    """
    Runs topic modeling pipeline on a list of sentences (corpus level or for new notes using pre-trained models).

    Args:
        sentences: A list of sentences.
        umap_model: Pre-trained UMAP model (optional, for prediction).
        hdbscan_model: Pre-trained HDBSCAN model (optional, for prediction).

    Returns:
        A dictionary containing topic modeling results, and the UMAP and HDBSCAN models.
        If models are provided, it will return cluster assignments using these models.
    """
    sentence_embeddings: List[List[float]] = [
        get_openai_embedding(s) for s in sentences
    ]
    reduced_embeddings, umap_model = reduce_dimensions_umap(
        sentence_embeddings, n_components=15, umap_model=umap_model
    )  # pass umap_model
    hdbscan_model, cluster_labels = cluster_embeddings_hdbscan(
        reduced_embeddings, hdbscan_model=hdbscan_model
    )  # pass hdbscan_model
    cluster_keywords = calculate_ctfidf_per_cluster(
        sentences, cluster_labels
    )  # Keywords still calculated based on current clusters

    topics_data: Dict[str, Dict[str, Union[List[str], List[Any]]]] = {}
    clustered_sentences: Dict[int, List[str]] = {}  # for easier processing later.
    for i, label in enumerate(cluster_labels):
        if label not in clustered_sentences:
            clustered_sentences[label] = []
        clustered_sentences[label].append(sentences[i])

    for label, sentence_list in clustered_sentences.items():
        if label != -1:  # Exclude noise cluster from keyword representation
            keywords_for_cluster = cluster_keywords.get(
                label, []
            )  # safety in case cluster has no keywords
            topics_data[f"cluster_{label}"] = {
                "sentences": sentence_list,
                "keywords": [
                    keyword for keyword, score in keywords_for_cluster
                ],  # just keywords, not scores for json
            }
        else:
            topics_data["noise_sentences"] = {
                "sentences": sentence_list,
                "keywords": [],
            }  # Noise sentences with empty keywords
    return topics_data, umap_model, hdbscan_model  # return models as well


# --- Model Saving and Loading Functions ---
def save_topic_models(umap_model, hdbscan_model, evaluation_dir):
    """Saves trained UMAP and HDBSCAN models."""
    model_dir = evaluation_dir / "topic_models"
    model_dir.mkdir(exist_ok=True)
    umap_path = model_dir / "umap_model.joblib"
    hdbscan_path = model_dir / "hdbscan_model.joblib"

    # Ensure prediction data is generated before saving
    if (
        not hasattr(hdbscan_model, "prediction_data_")
        or hdbscan_model.prediction_data_ is None
    ):
        logger.info("Generating prediction data before saving HDBSCAN model")
        hdbscan_model.generate_prediction_data()

    joblib.dump(umap_model, umap_path)
    joblib.dump(hdbscan_model, hdbscan_path)
    logger.info(f"Topic models saved to {model_dir}")


def load_topic_models(model_path: Path):
    """Loads trained UMAP and HDBSCAN models from a specified path."""
    umap_path = model_path / "umap_model.joblib"
    hdbscan_path = model_path / "hdbscan_model.joblib"
    umap_model, hdbscan_model = None, None
    try:
        umap_model = joblib.load(umap_path)
        hdbscan_model = joblib.load(hdbscan_path)

        # Ensure prediction data is available after loading
        if (
            not hasattr(hdbscan_model, "prediction_data_")
            or hdbscan_model.prediction_data_ is None
        ):
            logger.info("Generating prediction data for loaded HDBSCAN model")
            hdbscan_model.generate_prediction_data()

        logger.info(f"Topic models loaded from {model_path}")
    except FileNotFoundError:
        logger.warning(
            f"Topic models not found at {model_path}. Please ensure models are saved in this directory or run training first."
        )
        return None, None

    if umap_model is None or hdbscan_model is None:
        logger.error(
            f"Error loading topic models from {model_path}. UMAP: {umap_model is None}, HDBSCAN: {hdbscan_model is None}"
        )
        return None, None

    return umap_model, hdbscan_model


def assign_note_topics(
    note_text: str, umap_model, hdbscan_model
) -> Tuple[Dict[str, Any], List[int]]:
    """
    Assigns topic clusters to a new note using pre-trained topic models.

    Args:
        note_text: The text of the new patient note.
        umap_model: Pre-trained UMAP model.
        hdbscan_model: Pre-trained HDBSCAN model.

    Returns:
        A dictionary containing topic assignment results for the note and cluster labels.
    """
    print("\n=== Debug: Note Topic Assignment ===")
    print(f"Processing note (first 100 chars): {note_text[:100]}...")

    sentences: List[str] = simple_preprocess_text(note_text)
    print(f"Number of sentences: {len(sentences)}")

    sentence_embeddings: List[List[float]] = [
        get_openai_embedding(s) for s in sentences
    ]
    print(f"Number of embeddings: {len(sentence_embeddings)}")

    reduced_embeddings, _ = reduce_dimensions_umap(
        sentence_embeddings, umap_model=umap_model
    )
    print(f"Number of reduced embeddings: {len(reduced_embeddings)}")

    _, cluster_labels = cluster_embeddings_hdbscan(
        reduced_embeddings, hdbscan_model=hdbscan_model
    )
    print(f"Cluster labels assigned: {cluster_labels}")

    note_topics = {}
    sentence_cluster_assignments: Dict[int, List[str]] = {}

    # Print cluster assignments for each sentence
    print("\nSentence-level cluster assignments:")
    for i, (label, sentence) in enumerate(zip(cluster_labels, sentences)):
        print(f"Sentence {i}: Cluster {label} - {sentence[:50]}...")

        if label not in sentence_cluster_assignments:
            sentence_cluster_assignments[label] = []
        sentence_cluster_assignments[label].append(sentences[i])

    print("\nFinal cluster assignments:")
    for label, sentences_in_cluster in sentence_cluster_assignments.items():
        if label != -1:
            cluster_key = f"cluster_{label}"
            note_topics[cluster_key] = sentences_in_cluster
            print(f"{cluster_key}: {len(sentences_in_cluster)} sentences")
        else:
            note_topics["noise_sentences"] = sentences_in_cluster
            print(f"noise_sentences: {len(sentences_in_cluster)} sentences")

    return note_topics, cluster_labels


def analyze_cluster_distributions(per_note_df):
    """
    Creates a summary of cluster distributions with label co-occurrences
    """
    # Get all cluster columns
    cluster_cols = [col for col in per_note_df.columns if col.endswith("_present")]
    clusters = [col.replace("_present", "") for col in cluster_cols]

    summaries = []

    for cluster in clusters:
        # Get notes where this cluster is present
        cluster_notes = per_note_df[per_note_df[f"{cluster}_present"] == 1]

        summary = {
            "cluster_name": cluster,
            "total_notes": len(cluster_notes),
            "total_sentences": cluster_notes[f"{cluster}_sentence_count"].sum(),
            "avg_sentences_per_note": round(
                cluster_notes[f"{cluster}_sentence_count"].mean(), 2
            ),
        }

        # Process true/pred label columns
        true_cols = [
            col
            for col in cluster_notes.columns
            if col.startswith("true_")
            and col != "true_labels"
            and not any(
                col.endswith(x)
                for x in ["_present", "_sentence_count", "_percentage", "_sentences"]
            )
        ]

        for true_col in true_cols:
            # Get corresponding pred column
            base_name = true_col.replace("true_", "")
            pred_col = f"pred_{base_name}"

            # Process true label values
            true_values = cluster_notes[true_col].value_counts()
            for val, count in true_values.items():
                percentage = (count / len(cluster_notes)) * 100
                summary[f"{true_col}_{val}_count"] = count
                summary[f"{true_col}_{val}_percentage"] = round(percentage, 2)

            # Process predicted label values if they exist
            if pred_col in cluster_notes.columns:
                pred_values = cluster_notes[pred_col].value_counts()
                for val, count in pred_values.items():
                    percentage = (count / len(cluster_notes)) * 100
                    summary[f"{pred_col}_{val}_count"] = count
                    summary[f"{pred_col}_{val}_percentage"] = round(percentage, 2)

                # Add agreement metrics
                exact_match = (
                    cluster_notes[true_col] == cluster_notes[pred_col]
                ).mean() * 100
                within_one = (
                    abs(cluster_notes[true_col] - cluster_notes[pred_col]) <= 1
                ).mean() * 100
                summary[f"{base_name}_exact_match_percentage"] = round(exact_match, 2)
                summary[f"{base_name}_within_one_percentage"] = round(within_one, 2)

        # Process dictionary labels safely
        dict_cols = [
            col for col in cluster_notes.columns if col.startswith("true_labels")
        ]
        for dict_col in dict_cols:
            try:
                # Get the first row's dictionary
                first_row_dict = cluster_notes[dict_col].iloc[0]

                # Handle both string and dictionary cases
                if isinstance(first_row_dict, str):
                    first_row_dict = json.loads(first_row_dict.replace("'", '"'))
                elif isinstance(first_row_dict, dict):
                    pass  # Already a dictionary
                else:
                    continue  # Skip if neither string nor dictionary

                # Count occurrences of each attribute value
                for attr, _ in first_row_dict.items():
                    # Convert each row's dictionary and count matches
                    value_counts = {}
                    for _, row in cluster_notes.iterrows():
                        row_dict = row[dict_col]
                        # Handle string case
                        if isinstance(row_dict, str):
                            row_dict = json.loads(row_dict.replace("'", '"'))

                        val = row_dict[attr]
                        value_counts[val] = value_counts.get(val, 0) + 1

                    # Add counts and percentages to summary
                    for val, count in value_counts.items():
                        percentage = (count / len(cluster_notes)) * 100
                        summary[f"{attr}_{val}_count"] = count
                        summary[f"{attr}_{val}_percentage"] = round(percentage, 2)

            except (json.JSONDecodeError, ValueError, KeyError, AttributeError) as e:
                print(f"Error processing dictionary column {dict_col}: {e}")
                continue

        # Add cluster co-occurrences
        for other_cluster in clusters:
            if other_cluster != cluster:
                co_occurrence = len(
                    cluster_notes[cluster_notes[f"{other_cluster}_present"] == 1]
                )
                summary[f"co_occurs_with_{other_cluster}"] = co_occurrence
                summary[f"co_occurs_with_{other_cluster}_percentage"] = round(
                    (co_occurrence / len(cluster_notes)) * 100, 2
                )

        summaries.append(summary)

    return pd.DataFrame(summaries)


def create_per_note_cluster_data(row, note_topics, cluster_labels):
    """
    Creates a dictionary containing all note data including cluster assignments and label distributions
    """
    # Create base note info with all original columns
    note_data = {col: row[col] for col in row.index}

    # Get unique clusters including noise
    unique_clusters = {
        f"cluster_{label}" for label in set(cluster_labels) if label != -1
    }
    unique_clusters.add("noise_sentences")

    # Initialize cluster columns
    for cluster in unique_clusters:
        note_data[f"{cluster}_present"] = 0
        note_data[f"{cluster}_sentence_count"] = 0
        note_data[f"{cluster}_percentage"] = 0.0
        note_data[f"{cluster}_sentences"] = []

    # Calculate total sentences
    total_sentences = sum(len(sentences) for sentences in note_topics.values())

    # Fill cluster metrics
    for cluster_name, sentences in note_topics.items():
        note_data[f"{cluster_name}_present"] = 1
        note_data[f"{cluster_name}_sentence_count"] = len(sentences)
        note_data[f"{cluster_name}_percentage"] = round(
            (len(sentences) / total_sentences) * 100, 2
        )
        note_data[f"{cluster_name}_sentences"] = sentences

    # Process true/pred label distributions if available
    for col in row.index:
        if col.startswith(("true_", "pred_")):
            note_data[col] = row[col]

    return note_data


class EvaluationPipeline:
    """Orchestrates the topic modeling and monitoring pipeline."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        ensure_evaluation_structure(config.evaluation_dir)

    def process_predictions(self) -> pd.DataFrame:
        """Process and load prediction data for topic model training/inference."""
        logger.info("Collecting prediction data...")

        # --- Mock PredictionEvaluator.collect_prediction_data for self-contained script ---
        class MockPredictionEvaluator:  # Simple mock for collecting prediction data
            def collect_prediction_data(
                self,
                base_dir: Path,
                graph_type: Optional[str] = None,
                model_config: Optional[str] = None,
                prompt_variant: Optional[str] = None,
            ) -> pd.DataFrame:
                # Load data from results_dir - assuming predictions.json files are there
                all_predictions = []
                for pred_file in base_dir.rglob("predictions.json"):
                    try:
                        with open(pred_file) as f:
                            predictions = json.load(f)
                        if isinstance(predictions, dict):
                            predictions = [
                                predictions
                            ]  # Handle single prediction dict case

                        # Basic filtering (you can expand based on your needs)
                        if graph_type and graph_type not in pred_file.parts:
                            continue
                        if model_config and model_config not in pred_file.parts:
                            continue
                        if prompt_variant and prompt_variant not in pred_file.parts:
                            continue
                        all_predictions.extend(predictions)
                    except Exception as e:
                        logger.error(f"Error loading predictions from {pred_file}: {e}")
                return pd.DataFrame(all_predictions)

        # --- End Mock ---
        evaluator = MockPredictionEvaluator()  # Use mock evaluator

        prediction_df = evaluator.collect_prediction_data(
            self.config.results_dir,
            graph_type=self.config.graph_type,
            model_config=self.config.model_config,
            prompt_variant=self.config.prompt_variant,
        )

        if prediction_df.empty:
            raise ValueError("No valid prediction data found.")

        # Ensure 'raw_note' and 'note' columns exist (adapt based on your actual predictions.json structure)
        if (
            "note" not in prediction_df.columns
            or "raw_note" not in prediction_df.columns
        ):
            raise ValueError(
                "Prediction data must contain 'note' and 'raw_note' columns."
            )

        return prediction_df  # Return prediction_df for topic model training/inference

    def run_training(self) -> None:
        """Execute topic model training pipeline."""
        try:
            prediction_df = self.process_predictions()

            # --- Aggregate all raw notes and processed notes into lists of sentences ---
            all_raw_note_sentences: List[str] = []
            all_processed_note_sentences: List[str] = []

            for index, row in prediction_df.iterrows():
                raw_note_sentences = simple_preprocess_text(row["raw_note"])
                processed_note_sentences = simple_preprocess_text(row["note"])
                all_raw_note_sentences.extend(raw_note_sentences)
                all_processed_note_sentences.extend(processed_note_sentences)

            # --- Train topic models on the aggregated sentences (dataset level) and save them ---
            logger.info("Training topic models on aggregated raw notes...")
            raw_note_topics, umap_model_raw, hdbscan_model_raw = run_topic_modeling(
                all_raw_note_sentences
            )
            save_topic_models(
                umap_model_raw, hdbscan_model_raw, self.config.evaluation_dir
            )  # Save models for raw notes

            logger.info("Training topic models on aggregated processed notes...")
            processed_note_topics, umap_model_processed, hdbscan_model_processed = (
                run_topic_modeling(all_processed_note_sentences)
            )
            save_topic_models(
                umap_model_processed,
                hdbscan_model_processed,
                self.config.evaluation_dir,
            )  # Save models for processed notes

            # --- Save dataset-level topic modeling results (as JSON) ---
            raw_note_topics_filepath = (
                self.config.evaluation_dir / "metrics/raw_note_dataset_topics.json"
            )
            processed_note_topics_filepath = (
                self.config.evaluation_dir
                / "metrics/processed_note_dataset_topics.json"
            )

            with open(raw_note_topics_filepath, "w") as f:
                json.dump(raw_note_topics, f, indent=4)
            logger.info(f"Raw note dataset topics saved to {raw_note_topics_filepath}")

            with open(processed_note_topics_filepath, "w") as f:
                json.dump(processed_note_topics, f, indent=4)
            logger.info(
                f"Processed note dataset topics saved to {processed_note_topics_filepath}"
            )

            logger.info(
                "Topic model training completed. Models and dataset topics saved."
            )

        except ValueError as ve:
            logger.error(f"Data Error: {ve}")
        except FileNotFoundError as fnfe:
            logger.error(f"File Not Found: {fnfe}")
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during topic model training: {e}"
            )

    def run_dataframe_inference(self) -> None:
        """Runs topic inference and creates detailed analysis of cluster distributions."""
        model_path = self.config.model_path
        if not model_path:
            logger.error("Model path is required for inference mode but not provided.")
            return

        # Load models
        umap_model_processed, hdbscan_model_processed = load_topic_models(
            model_path / "topic_models"
        )
        if umap_model_processed is None or hdbscan_model_processed is None:
            logger.error("Could not load topic models. Inference failed.")
            return

        # Load prediction data
        prediction_df = self.process_predictions()

        # Process each note and collect detailed information
        note_level_data = []

        for index, row in prediction_df.iterrows():
            # Get cluster assignments for the note
            note_topics, cluster_labels = assign_note_topics(
                row["note"], umap_model_processed, hdbscan_model_processed
            )

            # Create detailed note data including cluster information
            note_data = create_per_note_cluster_data(row, note_topics, cluster_labels)
            note_level_data.append(note_data)

        # Create and save per-note DataFrame
        per_note_df = pd.DataFrame(note_level_data)
        per_note_output_path = (
            self.config.evaluation_dir / "metrics/per_note_cluster_analysis.csv"
        )
        per_note_df.to_csv(per_note_output_path, index=False)
        logger.info(f"Per-note analysis saved to {per_note_output_path}")

        # Create and save cluster distribution analysis
        cluster_distribution_df = analyze_cluster_distributions(per_note_df)
        distribution_output_path = (
            self.config.evaluation_dir / "metrics/cluster_distribution_analysis.csv"
        )
        cluster_distribution_df.to_csv(distribution_output_path, index=False)
        logger.info(
            f"Cluster distribution analysis saved to {distribution_output_path}"
        )

    def run_monitoring(
        self, note_text: str, note_type: str = "raw"
    ) -> Optional[Dict[str, Any]]:
        """
        Runs topic assignment for monitoring a new note.

        Args:
            note_text: The text content of the note to monitor.
            note_type: "raw" or "processed", indicating which topic model to use.
        """
        if note_type not in ["raw", "processed"]:
            raise ValueError("note_type must be 'raw' or 'processed'")

        model_path = self.config.model_path  # Model path from config
        if not model_path:
            logger.error("Model path is required for monitoring mode but not provided.")
            return None

        # --- Load Pre-trained Topic Models ---
        umap_model, hdbscan_model = load_topic_models(
            model_path / "topic_models"
        )  # Load models from specified model_path
        if umap_model is None or hdbscan_model is None:
            logger.error("Topic models not found. Monitoring failed.")
            return None

        logger.info(
            f"Assigning topics to new {note_type} note using pre-trained models from {model_path}..."
        )
        note_topics, cluster_labels = assign_note_topics(
            note_text, umap_model, hdbscan_model
        )

        # --- Load dataset-level keywords for topic labels (optional, for richer output) ---
        if note_type == "raw":
            dataset_topics_filepath = (
                self.config.evaluation_dir / "metrics/raw_note_dataset_topics.json"
            )
        else:  # note_type == "processed"
            dataset_topics_filepath = (
                self.config.evaluation_dir
                / "metrics/processed_note_dataset_topics.json"
            )

        dataset_keywords = {}
        try:
            with open(dataset_topics_filepath, "r") as f:
                dataset_topics_data = json.load(f)
                for cluster_label, topic_data in dataset_topics_data.items():
                    if cluster_label.startswith("cluster_"):
                        dataset_keywords[cluster_label] = topic_data.get("keywords", [])
        except FileNotFoundError:
            logger.warning(
                f"Dataset topic keywords file not found at {dataset_topics_filepath}. Topic labels will not include keywords."
            )

        # --- Prepare monitoring output ---
        monitoring_output: Dict[str, Any] = {
            "note_type": note_type,
            "assigned_topics": {},
        }

        for cluster_label, sentences_in_cluster in note_topics.items():
            if cluster_label != "noise_sentences":
                topic_label = cluster_label  # e.g., "cluster_0"
                keywords = dataset_keywords.get(
                    cluster_label, []
                )  # get keywords from dataset topics for labeling
                monitoring_output["assigned_topics"][topic_label] = {
                    "sentences": sentences_in_cluster,
                    "dataset_keywords": keywords,  # Keywords from dataset-level topics for context
                }
            else:
                monitoring_output["assigned_topics"]["noise_sentences"] = {
                    "sentences": sentences_in_cluster,
                    "dataset_keywords": [],
                }

        monitoring_output_filepath = (
            self.config.evaluation_dir
            / f"monitoring/{note_type}_note_monitoring_output.json"
        )  # Save monitoring output to a 'monitoring' sub-directory
        monitoring_dir = self.config.evaluation_dir / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)  # Ensure monitoring dir exists
        with open(monitoring_output_filepath, "w") as f:
            json.dump(monitoring_output, f, indent=4)
        logger.info(
            f"Monitoring output for {note_type} note saved to {monitoring_output_filepath}"
        )
        return monitoring_output

    def run(
        self,
        train_topic_models=True,
        monitor_note_text=None,
        monitor_note_type="raw",
        run_inference_mode=False,
    ) -> None:  # Combined run to control training/monitoring/inference
        """Execute the topic modeling pipeline, either for training, monitoring or inference."""
        if run_inference_mode:
            self.run_dataframe_inference()  # Run dataframe inference if model path is given and inference mode is requested
        elif train_topic_models:
            self.run_training()  # Run topic model training if train flag is set
        elif monitor_note_text:
            self.run_monitoring(
                monitor_note_text, monitor_note_type
            )  # Run monitoring for a given note
        else:
            logger.info(
                "No action specified. Use --train_topics to train models, --monitor_note for single note monitoring, or --model_path for dataframe inference."
            )


def main():
    parser = argparse.ArgumentParser(
        description="Run topic modeling training, dataframe inference, or monitor new notes."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="data/synthetic_outputs",
        help="Base directory containing experiment outputs (for training and inference)",
    )
    parser.add_argument(
        "--evaluation_dir",
        type=str,
        default="evaluation_results/synthetic/topics",  # Default evaluation dir for topic models
        help="Directory for evaluation results and topic models",
    )
    parser.add_argument(
        "--graph_type",
        choices=[
            "agent",
            "zeroshot",
            "task_specific",
        ],
        help="Filter specific graph type (for training and inference)",  # Example GRAPH_TYPES
    )
    parser.add_argument(
        "--model_config",
        choices=["config1", "config2"],  # Example MODEL_CONFIGS
        help="Filter specific model configuration (for training and inference)",
    )
    parser.add_argument(
        "--prompt_variant",
        choices=["variant1", "variant2"],  # Example PROMPT_VARIANTS
        help="Filter specific prompt variant (for training and inference)",
    )
    parser.add_argument(
        "--train_topics",
        action="store_true",
        help="Train topic models on the dataset (default action if --monitor_note or --model_path is not used).",
    )
    parser.add_argument(
        "--monitor_note",
        type=str,
        help="Text of a note to monitor for topic assignment (monitoring mode).",
        default=None,  # Monitoring mode is optional
    )
    parser.add_argument(
        "--monitor_note_type",
        choices=["raw", "processed"],
        default="raw",
        help="Type of note to monitor ('raw' or 'processed') - relevant only if --monitor_note is provided.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to a directory containing pre-trained topic models (umap_model.joblib, hdbscan_model.joblib). If provided, runs dataframe inference instead of training.",
        default=None,  # Model path is optional, training is default if not provided
    )

    args = parser.parse_args()

    config = EvaluationConfig(
        results_dir=args.results_dir,
        evaluation_dir=args.evaluation_dir,
        graph_type=args.graph_type,
        model_config=args.model_config,
        prompt_variant=args.prompt_variant,
        model_path=args.model_path,  # Pass model path from args to config
    )

    pipeline = EvaluationPipeline(config)

    pipeline.run(
        train_topic_models=args.train_topics
        or (
            not args.monitor_note and not args.model_path
        ),  # Train if --train_topics is set, or if neither monitoring nor model_path are provided
        monitor_note_text=args.monitor_note,
        monitor_note_type=args.monitor_note_type,
        run_inference_mode=bool(
            args.model_path
        ),  # Run inference mode if --model_path is provided
    )


if __name__ == "__main__":
    main()
