"""
Vision-X - Core Library
A delightful tool that analyzes images using AI to detect objects, scenes, colors, and text.

This module contains the core functionality for processing images using the Gemma3 model.
"""

from pathlib import Path
from typing import Any, Literal, Dict, Union
from datetime import datetime
from pydantic import BaseModel
from ollama import AsyncClient, GenerateResponse
import json
from rich.console import Console
import logging
from logging.handlers import RotatingFileHandler
import asyncio

# Configure logging
def setup_logging():
    """Initialize the logging system with rotating file handlers."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
    )

    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    file_handler = RotatingFileHandler(
        "logs/image_analysis.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    return logger


# Initialize logging

logger = setup_logging()

console = Console()
client = AsyncClient("localhost:11434")

with open("colors.json", "r") as f:
    COLOR_LIST = tuple(json.load(f))


class ObjectAttribute(BaseModel):
    """Repesents a specific attribute of an object in the image."""

    name: str
    value: str
    confidence: float


class DetectedObject(BaseModel):
    """Repesents an object detected in the image."""

    name: str
    confidence: float
    bounding_box: tuple[float, float, float, float] | None
    attributes: list[ObjectAttribute]
    description: str | None

    def __str__(self) -> str:
        return f"{self.name} ({self.confidence:.2f})"


class Color(BaseModel):
    """Represents a color detected in the image."""

    name: str  # type: ignore
    hex_code: str | None
    prominence: float

    def __str__(self) -> str:
        return self.name


class TextElement(BaseModel):
    """Represents text detected in the image."""

    content: str
    confidence: float
    position: tuple[float, float, float, float] | None

    def __str__(self) -> str:
        return self.content


class ImageDescription(BaseModel):
    """Comprehensive description of an analyzed image."""

    summary: str
    tags: list[str]
    objects: list[DetectedObject]
    scene: str
    colors: list[Color]
    text_elements: list[TextElement] | None
    time_of_day: Literal["Morning", "Afternoon", "Evening", "Night", "Unknown"]
    setting: Literal["Indoor", "Outdoor", "Unknown"]
    is_character: bool
    character_details: dict[str, str] | None
    image_quality: Literal["Low", "Medium", "High", "Unknown"]
    suggested_filename: str | None

    def get_dominant_colors(self, limit: int = 3) -> list[Color]:
        return sorted(self.colors, key=lambda c: c.prominence, reverse=True)[:limit]

    def has_text(self) -> bool:
        return bool(self.text_elements)

    def get_text_content(self) -> str:
        if not self.text_elements:
            return ""
        return " ".join(t.content for t in self.text_elements)


class ImageAnalysis(BaseModel):
    """📊 Metadata about the image analysis process."""

    image_path: str
    timestamp: str
    analysis_duration: float
    error: str | None = None


async def analyze_image(
    image_path: Path,
) -> tuple[Union[ImageDescription, None], ImageAnalysis]:
    """Analyze an image using AI to detect objects, scenes, colors, and text."""
    start_time = datetime.now()
    max_retries = 1 
    retry_count = 0
    last_error = None
    last_response = "None"

    logger.info(f"Starting analysis of image: {image_path}")

    while retry_count < max_retries:
        try:
            logger.debug(f"Attempt {retry_count + 1} for {image_path}")
            response = await client.generate(
                model="gemma3",
                format=ImageDescription.model_json_schema(),
                prompt=f"Analyze this image {image_path.name} and return a detailed JSON description including objects, scene, colors and any text detected. If you cannot determine certain details, leave those fields empty.",
                images=[str(image_path)],
                options={
                    "temperature": 0,
                    "timeout": 60,
                },
            )

            if not response.response:
                raise ValueError("Received empty response from the model")

            last_response = response.response

            image_analysis = ImageDescription.model_validate_json(response.response)
            analysis_meta = ImageAnalysis(
                image_path=str(image_path),
                timestamp=datetime.now().isoformat(),
                analysis_duration=(datetime.now() - start_time).total_seconds(),
            )

            logger.info(
                f"Successfully analyzed {image_path} in {analysis_meta.analysis_duration:.2f} seconds"
            )
            return image_analysis, analysis_meta

        except Exception as e:
            last_error = str(e)
            retry_count += 1
            if retry_count < max_retries:
                logger.warning(
                    f"Attempt {retry_count} failed for {image_path}. Error: {last_error}"
                )
                logger.debug(f"Last Response: {last_response}")
            else:
                logger.debug(f"Last Response: {last_response}")
                logger.error(f"All attempts failed for {image_path}", exc_info=True)

    error_time = datetime.now()
    logger.error(f"Final failure for {image_path}. Creating error response.")
    return None, ImageAnalysis(
        image_path=str(image_path),
        timestamp=error_time.isoformat(),
        analysis_duration=(error_time - start_time).total_seconds(),
        error=last_error,
    )
