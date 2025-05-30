# app/main.py
import logging
from PIL import Image # Keep for type hinting if GradioInterface uses it, though not directly used here.

# Application specific imports
from app.services.gemini_service import GeminiService
from app.services.image_service import ImageService
from app.agents.user_interaction_agent import UserInteractionAgent
from app.ui.gradio_interface import GradioInterface

# Configure basic logging for the entire application
# This will be effective once any part of the app starts logging.
# Placed here to ensure it's configured before any module-level loggers are potentially called during import.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Logger for main.py itself

# Original ChallengeApplication class - to be removed or refactored if PydenticAI agent replaces its full functionality.
# For now, we are replacing its direct instantiation in main with the agent-based setup.
# class ChallengeApplication:
#     """
#     Core application class that orchestrates the challenge generation and evaluation.
#     (This class is being replaced by UserInteractionAgent for core logic)
#     """
#     def __init__(self):
#         logger.info("Initializing ChallengeApplication (legacy)...")
#         # ... (old init logic)

#     def generate_new_challenge(self) -> str:
#         # ... (old logic)
#         pass

#     def evaluate_challenge_submissions(self, image1_pil: Image, image2_pil: Image, topic: str) -> str:
#         # ... (old logic)
#         pass


if __name__ == '__main__':
    logger.info("Application starting...")
    
    # This structure assumes GOOGLE_APPLICATION_CREDENTIALS is set for GeminiService
    # and BLIP models for ImageService are accessible (downloaded or cached).
    try:
        logger.info("Initializing services...")
        # GeminiService is primary for the agent's LLM
        gemini_service = GeminiService()
        
        # ImageService is used by Gradio to get captions before passing to the agent
        image_service = ImageService() 
        
        logger.info("Initializing UserInteractionAgent...")
        # UserInteractionAgent now uses GeminiService for its LLM calls (via tools or direct responses).
        user_agent = UserInteractionAgent(gemini_service=gemini_service)
        
        logger.info("Initializing GradioInterface...")
        # Pass both the PydenticAI agent and the ImageService to GradioInterface
        gradio_ui = GradioInterface(agent=user_agent, image_service=image_service)
        
        gradio_ui.create_ui()
        logger.info("Launching Gradio interface...")
        # server_name="0.0.0.0" makes it accessible on the local network
        # share=True would create a temporary public link (requires internet & Gradio setup)
        gradio_ui.launch(server_name="0.0.0.0") 
        
    except RuntimeError as re: # Catch specific init errors from services/agents
        logger.critical(f"Critical Error during initialization: {re}", exc_info=True)
        logger.critical("The application cannot start. Please check configurations (e.g., GOOGLE_APPLICATION_CREDENTIALS, model access, API keys).")
    except ImportError as ie:
        logger.critical(f"Critical Error: Missing import: {ie}", exc_info=True)
        logger.critical("Please ensure all dependencies are installed. You might need to run: pip install -r requirements.txt")
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred during startup: {e}", exc_info=True)
        logger.critical("Application launch failed.")
