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
            "צור לי אתגר יצירתי ומחשבתי לאדם אחד. האתגר צריך לדרוש שימוש בחפצים נפוצים הנמצאים בדרך כלל בבית. "
            "חשוב שההוראות באתגר יהיו ברורות לחלוטין ויסבירו בדיוק מה על המשתתף ליצור או לעשות כך שניתן יהיה לצלם תמונה ולהוכיח עמידה באתגר. "
            "האתגר צריך להיות קל לביצוע מעשי, אך מעניין ומפעיל את הדמיון. "
            "נסח את האתגר בשורה אחת או שתיים, בשפה העברית."
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
            f"בהתחשב באתגר: '{topic}'.\n"
            f"להלן תיאורים של שתי תמונות שהוגשו כפתרונות לאתגר:\n"
            f"תמונה 1: {caption1}\n"
            f"תמונה 2: {caption2}\n\n"
            "עבור כל תמונה, אנא ספק ציון מ-1 עד 10 המייצג את מידת ההצלחה בביצוע האתגר, בהתבסס על יצירתיות, בהירות וייצוג بصری של הפתרון.\n" # Note: "בصری" seems to have a typo, likely meant "חזותי" or "ויזואלי". I will assume "חזותי" for correction.
            "לאחר מכן, הכרז על התמונה הזוכה.\n"
            "לבסוף, ספק הסבר קצר וקולע (עד שתי שורות) מדוע התמונה הזו נבחרה כמנצחת, תוך התמקדות בסיבה המרכזית להחלטה.\n\n"
            "הפלט הרצוי בעברית ובפורמט הבא:\n"
            "תמונה 1 - ציון: [הציון כאן]/10\n"
            "תמונה 2 - ציון: [הציון כאן]/10\n"
            "התמונה הזוכה היא: תמונה [1/2]\n"
            "הסבר קצר לזכייה: [ההסבר הקצר כאן]"
        )
        # Correcting potential typo in the prompt from "בصری" to "חזותי"
        prompt_content = prompt_content.replace("בصری", "חזותי")
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
