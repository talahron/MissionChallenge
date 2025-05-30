import os
import logging

logger = logging.getLogger(__name__)

def get_openai_api_key():
    """
    Retrieves the OpenAI API key from the environment variable OPENAI_API_KEY.

    Raises:
        ValueError: If the OPENAI_API_KEY environment variable is not set.

    Returns:
        str: The OpenAI API key.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        raise ValueError(
            "Error: OPENAI_API_KEY environment variable not set. "
            "Please set this variable with your OpenAI API key."
        )
    logger.info("OpenAI API key successfully retrieved from environment variable.")
    return api_key

# You can also define a global variable if preferred for easier import,
# but ensure it's loaded when the module is imported.
# Example:
# OPENAI_API_KEY = get_openai_api_key()
# This approach will raise an error immediately if the key is not set when config is imported.
# For now, let's stick to the function `get_openai_api_key()` to be called explicitly.


def get_google_application_credentials():
    """
    Retrieves the path to Google Application Credentials from the environment variable.
    This is more of a check, as Google libraries often auto-detect this.

    Returns:
        str: The path to the service account JSON file, if set.
    Raises:
        ValueError: If GOOGLE_APPLICATION_CREDENTIALS environment variable is not set or file not found.
    """
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
        raise ValueError(
            "Error: GOOGLE_APPLICATION_CREDENTIALS environment variable not set. "
            "Please set this variable with the path to your Google Cloud service account JSON file."
        )
    
    if not os.path.isfile(credentials_path):
        logger.error(f"Service account file not found at path: {credentials_path}")
        raise ValueError(
            f"Error: Service account file not found at path specified by "
            f"GOOGLE_APPLICATION_CREDENTIALS: {credentials_path}"
        )
    
    logger.info(f"GOOGLE_APPLICATION_CREDENTIALS found at: {credentials_path}")
    return credentials_path

# Example of how it might be called during setup (optional here, GeminiService will handle init)
# try:
#     GOOGLE_CREDENTIALS = get_google_application_credentials()
# except ValueError as e:
#     logger.warning(f"Note: {e}") # Log as warning, as GeminiService will handle its own init.
