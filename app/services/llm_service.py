from openai import OpenAI, APIError 
from app.config import get_openai_api_key
import logging

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service for interacting with the OpenAI LLM.
    Handles challenge generation and submission evaluation.
    """
    def __init__(self):
        """
        Initializes the LLMService, setting up the OpenAI client.
        Raises:
            ValueError: If the OpenAI API key is not configured.
        """
        logger.info("Initializing LLMService...")
        try:
            logger.info("Attempting to retrieve OpenAI API key.")
            self.api_key = get_openai_api_key() # get_openai_api_key already logs success/failure
            self.client = OpenAI(api_key=self.api_key)
            logger.info("OpenAI client initialized successfully.")
        except ValueError as e: # This will be caught if get_openai_api_key raises it
            logger.error(f"ValueError during LLMService initialization: {e}", exc_info=True)
            # Propagate the error or handle it as per application design
            # For now, let's re-raise to make it explicit that LLMService cannot operate
            raise 
        except Exception as e: # Catch any other unexpected error during client init
            logger.exception("An unexpected error occurred during LLMService initialization.")
            raise

    def generate_challenge_topic(self) -> str:
        """
        Generates a new challenge topic using the LLM.

        Returns:
            str: The generated challenge topic, or an error message if generation fails.
        """
        logger.info("Attempting to generate a new challenge topic.")
        prompt_content = (
            "צור לי אתגר. האתגר צריך להיות כזה שאוכל להעלות תמונה וניתן יהיה לראות מהתמונה האם הצלחתי או לא באתגר."
            "האתגר צריך להיות מאתגר, פשוט לביצוע, כיף וכזה שניתן ליצור אותו מחפצים שוניים שיש בכל בית"
            "לא יותר משורה אחת בעברית."
        )
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "אתה מומחה ביצירת בעברית אתגרים המפעילים חשיבה ויצירתיות"},
                    {"role": "user", "content": prompt_content}
                ]
            )
            topic = response.choices[0].message.content
            if topic:
                logger.info(f"Successfully generated challenge topic: {topic}")
                return topic
            else:
                logger.warning("LLM did not return a topic.")
                return "Error: LLM did not return a topic."
        except APIError as e:
            logger.error(f"OpenAI API error while generating topic: {e}")
            return f"OpenAI API error: {e}"
        except Exception as e:
            logger.exception("Unexpected error while generating topic.")
            return f"Unexpected error: {e}"

    def evaluate_submissions(self, topic: str, caption1: str, caption2: str) -> str:
        """
        Evaluates two image submissions based on their captions and a given challenge topic.

        Args:
            topic: The challenge topic.
            caption1: The caption for the first image.
            caption2: The caption for the second image.

        Returns:
            str: The LLM's evaluation and winner declaration, or an error message if evaluation fails.
        """
        logger.info(f"Attempting to evaluate submissions for topic: '{topic}'. Caption1 provided: {bool(caption1)}, Caption2 provided: {bool(caption2)}")
        if not all([topic, caption1, caption2]):
            logger.warning("Evaluation called with missing topic or captions.")
            return "Error: Topic and both captions must be provided for evaluation."

        prompt_content = (
            f"בהתחשב באתגר '{topic}' והתיאורים של שתי תמונות,\n תמונה 1 - {caption1}\n תמונה 2 - {caption2}\n."
            "כל אחד מהם הוא תיאור של תמונה של מתמודד שניסה להצליח באתגר."
            "עבור כל אחד מהם, ספק ציון מאחד עד עשר על סמך עד כמה הוא מסיים את האתגר."
            "להכריז מי התמונה הזוכה, מה מוצג בתמונה ולמה היא נבחרה לבסוף כזוכה."
            "שקול את הקריטריונים הבאים: יצירתיות, בהירות, ועד כמה ביעילות התמונה מייצגת חזותית את הפתרון לאתגר .תספק את תשובתך בעברית."
            "הפלט צריך לצאת בצורה הבאה:"
            "תמונה 1 - תיאור של התמונה - ציון כולל"
            "תמונה 2 - תיאור של התמונה - ציון כולל"
            "תמונה הזוכה - ..."
            "הסבר מדוע זוהי התמונה הזוכה"
        )
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "אתה מומחה בהשוואת תוצאות האתגרים בין שני מתחרים."},
                    {"role": "user", "content": prompt_content}
                ]
            )
            result = response.choices[0].message.content
            if result:
                logger.info("Submissions evaluated successfully by LLM.")
                return result
            else:
                logger.warning("LLM did not return an evaluation.")
                return "Error: LLM did not return an evaluation."
        except APIError as e:
            logger.error(f"OpenAI API error during evaluation: {e}")
            return f"OpenAI API error: {e}"
        except Exception as e:
            logger.exception("Unexpected error during evaluation.")
            return f"Unexpected error: {e}"
