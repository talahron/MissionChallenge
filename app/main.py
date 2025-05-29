# app/main.py
from PIL import Image
from app.services.image_service import ImageService
from app.services.llm_service import LLMService
# Potentially: from app.models.challenge import ChallengeData
import logging

logger = logging.getLogger(__name__)

class ChallengeApplication:
    """
    Core application class that orchestrates the challenge generation and evaluation.
    """
    def __init__(self):
        """
        Initializes the ChallengeApplication, creating instances of required services.
        """
        logger.info("Initializing ChallengeApplication...")
        try:
            logger.info("Initializing ImageService...")
            self.image_service = ImageService()
            logger.info("ImageService initialized successfully.")
            
            logger.info("Initializing LLMService...")
            self.llm_service = LLMService()
            logger.info("LLMService initialized successfully.")
            
            logger.info("ChallengeApplication initialized successfully.")
        except ValueError as e: # Catch error from LLMService if API key is missing
            logger.critical(f"ValueError during ChallengeApplication initialization: {e}", exc_info=True)
            # Application cannot function without LLMService, re-raise or handle appropriately
            raise
        except Exception as e:
            logger.critical(f"An unexpected error occurred during ChallengeApplication initialization: {e}", exc_info=True)
            raise

    def generate_new_challenge(self) -> str:
        """
        Generates a new challenge topic.

        Returns:
            str: The challenge topic, or an error message if generation failed.
        """
        logger.info("generate_new_challenge called.")
        topic = self.llm_service.generate_challenge_topic()
        if "Error:" in topic:
            logger.warning(f"generate_new_challenge received an error from LLMService: {topic}")
        else:
            logger.info(f"generate_new_challenge successfully received topic: {topic}")
        return topic

    def evaluate_challenge_submissions(self, image1_pil: Image.Image, image2_pil: Image.Image, topic: str) -> str:
        """
        Evaluates two image submissions for a given challenge topic.

        Args:
            image1_pil: PIL Image object for the first submission.
            image2_pil: PIL Image object for the second submission.
            topic: The challenge topic string.

        Returns:
            str: The evaluation result, or an error/status message.
        """
        logger.info(f"evaluate_challenge_submissions called for topic: '{topic}'. Image1 provided: {bool(image1_pil)}, Image2 provided: {bool(image2_pil)}")
        if not topic or topic.strip() == "":
            logger.warning("Evaluation called with no topic provided.")
            return "Error: A challenge topic must be provided or generated first."
        if image1_pil is None or image2_pil is None:
            logger.warning("Evaluation called with one or both images missing.")
            return "Error: Both images must be uploaded for evaluation."

        logger.info("Generating caption for image 1...")
        caption1 = self.image_service.generate_caption(image1_pil)
        if "Error:" in caption1: # Simple check, could be more robust
            logger.error(f"Failed to generate caption for Image 1: {caption1}")
            return f"Failed to generate caption for Image 1: {caption1}"
        logger.info(f"Caption 1 generated: {caption1}")

        logger.info("Generating caption for image 2...")
        caption2 = self.image_service.generate_caption(image2_pil)
        if "Error:" in caption2: # Simple check
            logger.error(f"Failed to generate caption for Image 2: {caption2}")
            return f"Failed to generate caption for Image 2: {caption2}"
        logger.info(f"Caption 2 generated: {caption2}")
        
        logger.info("Sending captions to LLM for evaluation...")
        result = self.llm_service.evaluate_submissions(topic, caption1, caption2)
        logger.info(f"Evaluation result from LLM: {result}")
        return result

# ... (ChallengeApplication class definition above) ...

if __name__ == '__main__':
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("Application starting...")
    # We need to import GradioInterface here to avoid circular dependency issues
    # if GradioInterface also imports from app.main (which it does for ChallengeApplication)
    from app.ui.gradio_interface import GradioInterface
    try:
        # Initialize the main application logic
        challenge_app = ChallengeApplication()
        
        # Initialize and launch the Gradio UI
        gradio_ui = GradioInterface(app=challenge_app)
        gradio_ui.create_ui() # Create the UI structure
        
        # Launch the Gradio app. 
        # You can add share=True to get a public link if needed,
        # and server_name="0.0.0.0" to make it accessible on your local network.
        logger.info("Launching Gradio interface...")
        gradio_ui.launch(server_name="0.0.0.0") 
        
    except ValueError as ve: # Catch specific error from ChallengeApplication init (e.g. API key)
        logger.critical(f"Failed to initialize ChallengeApplication: {ve}", exc_info=True)
        logger.critical("The application cannot start. Please check your configuration (e.g., OPENAI_API_KEY).")
    except ImportError as ie:
        logger.critical(f"Missing import: {ie}", exc_info=True)
        logger.critical("Please ensure all dependencies are installed. You might need to run: pip install -r requirements.txt")
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred during application startup: {e}", exc_info=True)
        logger.critical("Application launch failed.")
