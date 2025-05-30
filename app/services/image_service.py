# app/services/image_service.py
import logging
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import asyncio # For asyncio.to_thread
import os # For path operations

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
        logger.info(f"ImageService will use device: {self.device}")
        
        self.processor = None
        self.model = None

        try:
            # Using a local cache directory within the app for Hugging Face models
            # This can be beneficial in environments where the default cache is not writable or persistent.
            # Assuming this script is in app/services/image_service.py
            # project_root/app/services/ -> project_root/app/ -> project_root/
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cache_dir = os.path.join(base_dir, 'hf_cache') # Store cache in project_root/hf_cache
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Using Hugging Face cache directory: {cache_dir}")

            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=cache_dir)
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=cache_dir).to(self.device)
            logger.info("BLIP model and processor loaded successfully using local cache.")
        except Exception as e:
            logger.error(f"Error loading BLIP model: {e}", exc_info=True)
            # Model and processor remain None if loading fails
            # generate_caption will handle this state.
            # Optionally, could raise RuntimeError here if model is critical for app to even start.
            # raise RuntimeError(f"Failed to load BLIP model: {e}")

    def _blocking_generate_caption(self, image_pil: Image.Image) -> str:
        """
        The actual blocking (synchronous) caption generation logic.
        """
        if not self.model or not self.processor:
            logger.error("ImageService model/processor not loaded. Cannot generate caption.")
            return "Error: Image captioning model not available."
        
        if image_pil is None:
            logger.warning("Image is None, cannot generate caption.") # Logged in previous version
            return "Error: No image provided for captioning." # Changed from "Error: No image provided." for clarity

        try:
            logger.info(f"Processing image for caption generation (mode: {image_pil.mode}).") # Added image mode log
            if image_pil.mode != "RGB":
                logger.info("Image is not in RGB mode, converting to RGB.")
                image_pil = image_pil.convert("RGB")
                
            inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_length=50) 
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            logger.info(f"Caption generated successfully (length: {len(caption)}).")
            return caption
        except Exception as e:
            logger.error(f"Error during blocking caption generation: {e}", exc_info=True) # More specific log
            return "Error generating image caption."

    async def generate_caption(self, image_pil: Image.Image) -> str:
        """
        Generates a caption for the given PIL image asynchronously.
        The actual model inference is run in a separate thread to avoid blocking the event loop.
        """
        logger.info("Async generate_caption called.")
        if not self.model or not self.processor: # Early exit if model isn't loaded
            logger.error("ImageService model/processor not loaded (checked in async wrapper). Cannot generate caption.")
            return "Error: Image captioning model not available."
        if image_pil is None: # Handle None image input directly in async wrapper too
            logger.warning("Image is None (checked in async wrapper), cannot generate caption.")
            return "Error: No image provided for captioning."
            
        try:
            caption = await asyncio.to_thread(self._blocking_generate_caption, image_pil)
            return caption
        except Exception as e: 
            logger.error(f"Unexpected error in async generate_caption wrapper: {e}", exc_info=True)
            return "Error during async caption processing."


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    async def main_test():
        logger.info("Testing ImageService async caption generation...")
        # This test requires the BLIP model to be downloadable or cached,
        # and a PIL Image object.
        image_service = None
        try:
            # This initialization might take time due to model download if not cached.
            # It also depends on the environment having network access and sufficient space.
            logger.info("Attempting to initialize ImageService for test...")
            image_service = ImageService() # Instance creation
            
            if image_service.model and image_service.processor: 
                logger.info("ImageService initialized successfully for test.")
                # Create a dummy PIL image for testing
                dummy_image = Image.new('RGB', (100, 100), color = 'blue') # Larger dummy image
                logger.info("Dummy image created. Generating caption...")
                caption = await image_service.generate_caption(dummy_image)
                logger.info(f"Generated caption for dummy image: {caption}")

                caption_none = await image_service.generate_caption(None)
                logger.info(f"Caption for None image: {caption_none}")
            else:
                logger.error("Skipping caption generation test as model/processor failed to load during ImageService initialization.")
        
        except RuntimeError as e: # Catch RuntimeError from ImageService init if it's raised
            logger.error(f"RuntimeError during ImageService test (likely model loading failure): {e}")
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred during ImageService test: {e}", exc_info=True)

    # The following line would run the test.
    # However, in the automated worker environment:
    # 1. Model downloads might be restricted or very slow.
    # 2. `torch` and `transformers` can be very large; previous steps showed "No space left on device".
    # 3. `asyncio.run(main_test())` would attempt to execute it.
    # Given these constraints, the actual execution of this test block by the worker is unlikely to succeed
    # if it involves downloading large models into a space-constrained environment.
    # The primary goal is the refactoring of the ImageService class itself.
    
    # asyncio.run(main_test()) # Commented out to prevent execution issues in restricted env.
    logger.info("ImageService structure updated for async. Standalone test block is defined but commented out for worker execution.")
    logger.info("To run this test locally, ensure dependencies are installed, sufficient disk space, and network access.")
