# app/services/gemini_service.py
import google.generativeai as genai
import logging
import os
import asyncio # Added import
# Assuming get_google_application_credentials is in app.config for a pre-check,
# though genai.configure might not strictly need it if GOOGLE_APPLICATION_CREDENTIALS is set.
from app.config import get_google_application_credentials 

logger = logging.getLogger(__name__)

# The specific model name for Gemini 1.5 Flash.
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash-latest" 

class GeminiService:
    """
    Service for interacting with Google Gemini models.
    """
    def __init__(self, model_name: str = DEFAULT_GEMINI_MODEL):
        """
        Initializes the GeminiService.
        Authenticates using Google Application Credentials (typically set via
        the GOOGLE_APPLICATION_CREDENTIALS environment variable).

        Args:
            model_name (str): The name of the Gemini model to use.
        Raises:
            RuntimeError: If the service account credentials are not configured
                          or the client fails to initialize.
        """
        logger.info(f"Initializing GeminiService with model: {model_name}")
        self.model_name = model_name
        
        try:
            # Explicitly check for credentials path for clarity and early failure.
            # The google-generativeai library might pick this up automatically too.
            get_google_application_credentials() 
            
            # Configure the generative AI client.
            # If GOOGLE_APPLICATION_CREDENTIALS is set, genai.configure() typically
            # does not need explicit api_key or credentials path arguments.
            # For google-generativeai, explicit configuration of api_key is usually
            # done if NOT using ADC (Application Default Credentials) via a service account.
            # Since we are relying on GOOGLE_APPLICATION_CREDENTIALS, we expect ADC to work.
            # genai.configure() is often used if you had an API_KEY directly.
            # For ADC, often no explicit genai.configure() call is needed,
            # or it's used to specify project if not implicitly found.
            # Let's assume for now that setting GOOGLE_APPLICATION_CREDENTIALS is sufficient
            # and genai.GenerativeModel() will pick it up.
            # If issues arise, one might use:
            # from google.oauth2 import service_account
            # credentials = service_account.Credentials.from_service_account_file(get_google_application_credentials())
            # genai.configure(credentials=credentials)
            # However, the library is designed to find ADC automatically.

            self.client = genai.GenerativeModel(self.model_name)
            logger.info("Gemini client initialized successfully.")
            # Test with a simple, non-streaming, low-impact call if possible,
            # or just check if client object is valid.
            # For now, just client creation is enough for this step.

        except ValueError as ve: # From get_google_application_credentials
            logger.error(f"Configuration error for GeminiService: {ve}")
            raise RuntimeError(f"GeminiService initialization failed due to configuration: {ve}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}", exc_info=True)
            raise RuntimeError(f"GeminiService initialization failed: {e}")

    async def generate_text(self, prompt: str) -> str:
        """
        Generates text using the configured Gemini model.
        The actual model call is run in a separate thread.
        """
        logger.info(f"Generating text with model {self.model_name} for prompt: '{prompt[:70]}...'")
        if not self.client:
            logger.error("Gemini client not initialized. Cannot generate text.")
            return "Error: Gemini client not initialized."
        try:
            # self.client.generate_content is a synchronous (blocking) call
            response = await asyncio.to_thread(self.client.generate_content, prompt)
            
            # Safer response parsing
            if response.parts:
                # Concatenate text from all parts
                full_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
                if full_text:
                    logger.info(f"Gemini generated text successfully (from parts). Length: {len(full_text)}")
                    return full_text
            
            # Fallback to response.text if parts are empty but text attribute exists
            if hasattr(response, 'text') and response.text:
                logger.info(f"Gemini generated text successfully (from .text attribute). Length: {len(response.text)}")
                return response.text

            # Handle cases like blocked prompts or no content
            logger.warning(f"Gemini response for prompt '{prompt[:70]}...' had no usable text parts or was blocked.")
            if hasattr(response, 'prompt_feedbacks') and response.prompt_feedbacks:
                for feedback in response.prompt_feedbacks:
                    # Log the feedback. Actual structure of feedback might vary.
                    logger.warning(f"Prompt Feedback: {feedback}") 
            return "Error: LLM returned no usable content or request was blocked."
            
        except Exception as e:
            logger.error(f"Error during Gemini text generation: {e}", exc_info=True)
            return f"Error: LLM call failed - {str(e)}"


async def run_gemini_service_test():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Attempting to initialize GeminiService for a quick test...")
    
    # Add parent directory to sys.path to allow app.config import if run directly
    if __package__ is None: # Python 3.3+
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        from app.config import get_google_application_credentials # Re-import for this block
    
    try:
        # Check credentials before trying to instantiate the service for this test block
        get_google_application_credentials()
        logger.info("GOOGLE_APPLICATION_CREDENTIALS check passed for standalone test.")
        service = GeminiService()
        logger.info("GeminiService initialized. Testing placeholder text generation...")
        test_prompt = "Hello Gemini!"
        response = await service.generate_text(test_prompt) # Await the async method
        logger.info(f"Test response: {response}")
    except RuntimeError as e:
        logger.error(f"Failed to initialize or test GeminiService: {e}")
        # Specific advice already logged by get_google_application_credentials or GeminiService init
    except ValueError as ve: # Catch error from the standalone get_google_application_credentials() call
        logger.error(f"Failed to run GeminiService test due to configuration: {ve}")
        logger.error("Ensure GOOGLE_APPLICATION_CREDENTIALS is set correctly to a valid service account JSON file path.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during GeminiService test: {e}", exc_info=True)

if __name__ == '__main__':
    # This is for basic testing of this module.
    import asyncio # Required for running async main
    asyncio.run(run_gemini_service_test())
