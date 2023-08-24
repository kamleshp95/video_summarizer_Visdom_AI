from video_summarizer.components.data_ingestion import DataIngestion
from video_summarizer.components.data_validation import DataValidation
import os
import sys
from dataclasses import dataclass


@dataclass
class TrainingPipelineConfig:
    artifacts_folder = "artifacts"
    data_ingestion_artifacts = os.path.join(artifacts_folder, "data_ingestion")


def create_directories(config=TrainingPipelineConfig):
    # Check if the artifacts folder exists
    if not os.path.exists(config.artifacts_folder):
        os.makedirs(config.artifacts_folder)

    # Check and create the data ingestion artifacts folder
    if not os.path.exists(config.data_ingestion_artifacts):
        os.makedirs(config.data_ingestion_artifacts)


def run_training_pipeline(config=TrainingPipelineConfig):
    # Create Directories
    create_directories()
    # create an instance of the `DataIngestion` class and assign variable as `downloader`.
    downloader = DataIngestion(dataset_name="lighteval/summarization",
                               subset="xsum", save_folder=config.data_ingestion_artifacts)
    downloader.download_dataset()
    # create an instance of the `DataValidation` class and assign variable as `validator`.
    validator = DataValidation(dataset_folder=config.data_ingestion_artifacts)
    validator.check_dataset()
