# app/agents/evaluation_tool.py
import logging
from app.services.gemini_service import GeminiService
import asyncio # Added for async operations

# --- PydenticAI Hypothetical Structure ---
from pydantic import BaseModel, Field 

try:
    from pydenticai.core.tool import BaseTool 
except ImportError:
    logger_dummy_eval = logging.getLogger(__name__ + ".dummy_eval") # Unique logger name
    logger_dummy_eval.warning("PydenticAI BaseTool not found. Using a dummy BaseTool for EvaluationTool.")
    class BaseTool: 
        name: str = "UnnamedTool"
        description: str = "This is a dummy tool."
        input_schema = None 
        def __init__(self, *args, **kwargs):
            pass
        async def _execute(self, *args, **kwargs): # Made async
            raise NotImplementedError("Dummy _execute called.")
# --- End PydenticAI Hypothetical Structure ---

logger = logging.getLogger(__name__)

class EvaluationInput(BaseModel):
    """Input schema for the SubmissionEvaluator tool."""
    topic: str = Field(description="The challenge topic against which submissions are evaluated.")
    caption1: str = Field(description="Text caption describing the first image submission.")
    caption2: str = Field(description="Text caption describing the second image submission.")


class EvaluationTool(BaseTool):
    """
    A PydenticAI Tool to evaluate image submissions based on a topic and captions
    using a Gemini model.
    """
    name: str = "SubmissionEvaluator"
    description: str = (
        "Evaluates two image submissions based on a challenge topic and their text captions. "
        "Use this tool to determine which submission better addresses the challenge and to get scores."
    )
    input_schema = EvaluationInput

    def __init__(self, gemini_service: GeminiService, **kwargs):
        super().__init__(**kwargs)
        self.gemini_service = gemini_service
        self.evaluation_prompt_template = (
            "אתה שופט מומחה וחסר פניות במשחק אתגר תמונות. תפקידך להעריך שתי הגשות לאתגר נתון.\n"
            "בהתחשב באתגר: '{topic}'.\n"
            "להלן תיאורים של שתי תמונות שהוגשו כפתרונות לאתגר (התיאורים נוצרו על ידי AI אחר ומתארים את תוכן התמונות):\n"
            "תמונה 1: {caption1}\n"
            "תמונה 2: {caption2}\n\n"
            "עבור כל תמונה, אנא ספק ציון מ-1 עד 10 המייצג את מידת ההצלחה בביצוע האתגר. שקול יצירתיות, בהירות ועד כמה התמונה מייצגת חזותית את הפתרון לאתגר.\n"
            "לאחר מכן, הכרז על התמונה הזוכה (תמונה 1 או תמונה 2).\n"
            "לבסוף, ספק הסבר קצר וקולע (עד שתי שורות) מדוע התמונה הזו נבחרה כמנצחת, תוך התמקדות בסיבה המרכזית להחלטה. היה אובייקטיבי וברור.\n\n"
            "הפלט הרצוי בעברית ובפורמט הבא:\n"
            "תמונה 1 - ציון: [הציון כאן]/10\n"
            "תמונה 2 - ציון: [הציון כאן]/10\n"
            "התמונה הזוכה היא: תמונה [1/2]\n"
            "הסבר קצר לזכייה: [ההסבר הקצר כאן]"
        )
        logger.info(f"EvaluationTool '{self.name}' initialized.")

    async def _execute(self, inputs: EvaluationInput) -> str: # Made async
        """
        Evaluates submissions based on the structured input.
        Args:
            inputs: An EvaluationInput object containing topic, caption1, and caption2.
        Returns:
            str: The evaluation result from the LLM, or an error message.
        """
        logger.info(f"Executing {self.name} tool for topic: {inputs.topic}")
        
        if not all([inputs.topic, inputs.caption1, inputs.caption2]): 
            logger.warning("EvaluationTool _execute called with missing data in input model.")
            return "שגיאה: יש לספק את נושא האתגר ושני תיאורי תמונות לצורך הערכה."

        try:
            prompt = self.evaluation_prompt_template.format(
                topic=inputs.topic, caption1=inputs.caption1, caption2=inputs.caption2
            )
            result = await self.gemini_service.generate_text(prompt=prompt) # Await async call
            
            if not result or "Placeholder response" in result or "Error:" in result:
                 logger.warning(f"Evaluation via GeminiService returned a non-ideal response: {result}")
                 return "מצטער, היתה בעיה בעיבוד ההערכה כרגע. נסה שוב מאוחר יותר."
            
            logger.info("Evaluation successful.")
            return result
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}", exc_info=True)
            return "שגיאה פנימית בעת הערכת התוצאות."

async def main_evaluation_tool_test(): # Renamed and made async
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing EvaluationTool...")

    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from app.services.gemini_service import GeminiService 

    gemini_service_instance = None
    try:
        gemini_service_instance = GeminiService()
        logger.info("Actual GeminiService initialized for testing EvaluationTool.")
    except Exception as e:
        logger.error(f"Failed to initialize actual GeminiService for tool testing: {e}")
        logger.warning("EvaluationTool test will use a DummyGeminiService.")

        class DummyGeminiService:
            async def generate_text(self, prompt: str): # Made async
                logger.info(f"DummyGeminiService.generate_text called with prompt: {prompt[:30]}...")
                await asyncio.sleep(0) # Simulate async
                if "תמונה 1 - ציון" in prompt: 
                     return "תמונה 1 - ציון: 8/10\nתמונה 2 - ציון: 7/10\nהתמונה הזוכה היא: תמונה 1\nהסבר קצר לזכייה: יצירתיות גבוהה יותר."
                return "תגובה דמה מהשירות."
        gemini_service_instance = DummyGeminiService()

    tool = EvaluationTool(gemini_service=gemini_service_instance)
    logger.info(f"Tool Name: {tool.name}")
    logger.info(f"Tool Description: {tool.description}")
    if hasattr(tool.input_schema, 'model_json_schema'): # Pydantic v2
        logger.info(f"Tool Input Schema: {tool.input_schema.model_json_schema(indent=2)}")
    elif hasattr(tool.input_schema, 'schema_json'): # Pydantic v1
         logger.info(f"Tool Input Schema: {tool.input_schema.schema_json(indent=2)}")
    else:
        logger.info("Tool Input Schema: Not defined or not a Pydantic model with schema method.")


    valid_input_data = {
        "topic": "צור משהו מצחיק באמצעות שלושה חפצים מהמטבח",
        "caption1": "בננה עם משקפי שמש ובצל קצוץ כשיער.",
        "caption2": "כף מונחת על צלחת."
    }
    try:
        tool_input = EvaluationInput(**valid_input_data)
        evaluation = await tool._execute(inputs=tool_input) # Await async method
        logger.info(f"Evaluation result for valid input: {evaluation}")
    except Exception as e:
        logger.error(f"Error during valid input test: {e}", exc_info=True)
    
    logger.info("EvaluationTool test finished.")

if __name__ == '__main__':
    asyncio.run(main_evaluation_tool_test())
