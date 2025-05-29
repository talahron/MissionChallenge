from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging

logger = logging.getLogger(__name__)

class ImageService:
    """
    Service for handling image-related operations, primarily caption generation.
    The BLIP model is loaded upon instantiation.
    """
    def __init__(self):
        """
        Initializes the ImageService, loading the BLIP model and processor.
        """
        logger.info("Initializing ImageService and loading BLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        try:
            # For some reason, from_pretrained does not work when run by the tool,
            # so I am specifying the cache directory.
            self.processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                cache_dir="/app/hf_cache"
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                cache_dir="/app/hf_cache"
            ).to(self.device)
            logger.info("BLIP model and processor loaded successfully.")
        except Exception as e:
            logger.exception("Failed to load BLIP model or processor.")
            raise  # Re-raise the exception to be handled by the caller

    def generate_caption(self, image_pil: Image.Image) -> str:
        """
        Generates a caption for the given PIL image.

        Args:
            image_pil: A PIL Image object.

        Returns:
            A string containing the generated caption.
        """
        if image_pil is None:
            logger.error("generate_caption called with no image provided.")
            return "Error: No image provided."

        logger.info("Attempting to generate caption for an image.")
        try:
            # Ensure image is in RGB if it's not
            if image_pil.mode != "RGB":
                logger.info("Image is not in RGB mode, converting.")
                image_pil = image_pil.convert("RGB")
                
            inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
            # Adjust max_length for potentially longer captions
            out = self.model.generate(**inputs, max_length=50) 
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            logger.info(f"Caption generated successfully (length: {len(caption)}).")
            return caption
        except Exception as e:
            logger.exception("Error generating caption.")
            return "Error generating caption."
