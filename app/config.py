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
