# app/agents/user_interaction_agent.py
import logging
from typing import List, Optional, Dict, Any # Any will be replaced by PydenticAI's Tool type
import asyncio # For async operations if needed

from app.services.gemini_service import GeminiService
from app.agents.challenge_tool import ChallengeGenerationTool
from app.agents.evaluation_tool import EvaluationTool

# --- PydenticAI Hypothetical Structure ---
try:
    from pydenticai.core.agent import BaseAgent
    from pydenticai.core.tool import BaseTool as PydenticAIToolType 
except ImportError:
    logger_dummy = logging.getLogger(__name__ + ".dummy_agent_infra") 
    logger_dummy.warning("PydenticAI BaseAgent or BaseTool not found. Using dummy base classes.")
    
    class BaseAgent: 
        def __init__(self, llm: Any, tools: List[Any], system_prompt: str, *args, **kwargs):
            self.llm = llm
            self.tools = tools
            self.system_prompt = system_prompt # Store the system prompt
            logger_dummy.info(f"DummyBaseAgent initialized. System prompt received: '{system_prompt[:50]}...'")
            pass
        
        async def run(self, input_message: str, **kwargs) -> str: 
            logger_dummy.info(f"DummyBaseAgent.run called with: {input_message}")
            # Simplified logic for dummy agent based on system prompt and tools
            if "ChallengeGenerator" in self.system_prompt and "אתגר חדש" in input_message:
                challenge_tool = next((t for t in self.tools if t.name == "ChallengeGenerator"), None)
                if challenge_tool: return await challenge_tool._execute()
                return "שגיאה: כלי יצירת אתגר לא נמצא (דמה)."
            
            image_captions = kwargs.get('image_captions')
            if "SubmissionEvaluator" in self.system_prompt and image_captions:
                eval_tool = next((t for t in self.tools if t.name == "SubmissionEvaluator"), None)
                if eval_tool:
                    return "הערכה בוצעה על ידי כלי הערכה (דמה)."
                return "שגיאה: כלי הערכה לא נמצא (דמה)."

            if hasattr(self.llm, 'generate_text') and self.llm:
                 return await self.llm.generate_text(prompt=f"{self.system_prompt}\nUser: {input_message}\nAI:")
            return "תגובה כללית (סוכן דמה)."

    class PydenticAIToolType: 
        name: str = "DummyTool"
# --- End PydenticAI Hypothetical Structure ---

logger = logging.getLogger(__name__)

class UserInteractionAgent(BaseAgent): 
    """
    The main PydenticAI Agent that manages user interaction,
    explains game rules, and uses tools to generate challenges and evaluate submissions.
    """

    def __init__(self, gemini_service: GeminiService):
        self.challenge_tool = ChallengeGenerationTool(gemini_service=gemini_service)
        self.evaluation_tool = EvaluationTool(gemini_service=gemini_service)
        
        agent_tools: List[PydenticAIToolType] = [self.challenge_tool, self.evaluation_tool]
        
        agent_system_prompt = (
            "אתה 'מנחה משחק אתגר התמונות', סוכן AI ידידותי, מסביר פנים ומומחה בהנחיית משחקים. "
            "מטרתך היא לנהל את המשחק בצורה חלקה ומהנה עבור המשתמשים.\n"
            "יש לך גישה לכלים הבאים: 'ChallengeGenerator' (ליצירת אתגרים חדשים) ו-'SubmissionEvaluator' (להערכת הגשות לאתגרים).\n\n"
            "התנהלות מול המשתמש:\n"
            "- התחל בהסבר קצר של כללי המשחק אם המשתמש חדש או מבקש זאת.\n"
            "- אם המשתמש מבקש 'אתגר חדש', 'משימה חדשה' או דומה, השתמש בכלי 'ChallengeGenerator' כדי ליצור אתגר והצג אותו למשתמש. זכור את נושא האתגר שנוצר.\n"
            "- אם המשתמש מעלה תמונות (שיגיעו אליך כתיאורי טקסט מוכנים) וקיים אתגר פעיל, השתמש בכלי 'SubmissionEvaluator' כדי להעריך את ההגשות. הצג את התוצאה למשתמש.\n"
            "- אם המשתמש שואל שאלה כללית על המשחק או מנהל שיחה שאינה קשורה ישירות לבקשת אתגר או הערכת הגשות, השב בצורה ידידותית ואינפורמטיבית מבלי להשתמש בכלים, אלא אם כן נדרש במפורש.\n"
            "- שמור על טון שיחה חיובי, סבלני ועוזר. אם אינך בטוח כיצד להגיב או מה כוונת המשתמש, בקש הבהרה.\n"
            "- נהל רישום פנימי של 'נושא האתגר הנוכחי' כדי להעבירו כראוי לכלי ההערכה."
        )
        
        super().__init__(llm=gemini_service, tools=agent_tools, system_prompt=agent_system_prompt)
        
        self.gemini_service = gemini_service 
        
        self.current_challenge_topic: Optional[str] = None
        self.game_rules: str = (
            "ברוכים הבאים לאתגר התמונות! המשחק עובד כך:\n"
            "1. בקשו ממני 'אתגר חדש' ואני אצור לכם משימה יצירתית.\n"
            "2. כל אחד משני המשתתפים (או שחקן יחיד בשני תפקידים) מעלה תמונה המייצגת את הפתרון שלו לאתגר.\n"
            "3. לאחר העלאת שתי התמונות, אני אעריך אותן ואכריז על המנצח!\n"
            "מוכנים להתחיל? בקשו אתגר חדש או שאלו אם משהו לא ברור."
        )
        logger.info("UserInteractionAgent initialized with refined system prompt.")


    async def process_user_interaction(self, user_message: str, image_captions: Optional[Dict[str, str]] = None) -> str:
        logger.info(f"UserInteractionAgent processing interaction: '{user_message}', Captions: {image_captions is not None}")

        if any(keyword in user_message.lower() for keyword in ["כללים", "הוראות", "איך משחקים"]):
            return self.game_rules

        if "אתגר חדש" in user_message or "צור אתגר" in user_message:
            logger.info("User requested a new challenge. Using ChallengeGenerationTool.")
            topic_generated = await self.challenge_tool._execute() 
            if "מצטער" in topic_generated or "שגיאה" in topic_generated:
                self.current_challenge_topic = None 
                return topic_generated 
            self.current_challenge_topic = topic_generated
            logger.info(f"New challenge set: {self.current_challenge_topic}")
            return f"האתגר החדש שלכם הוא:\n{self.current_challenge_topic}"

        if image_captions and image_captions.get("caption1") and image_captions.get("caption2"):
            logger.info("User submitted images for evaluation. Using EvaluationTool.")
            if not self.current_challenge_topic:
                return "לא נוצר עדיין אתגר. אנא בקשו 'אתגר חדש' תחילה."
            
            from app.agents.evaluation_tool import EvaluationInput 
            eval_input = EvaluationInput(
                topic=self.current_challenge_topic,
                caption1=image_captions["caption1"],
                caption2=image_captions["caption2"]
            )
            evaluation_result = await self.evaluation_tool._execute(inputs=eval_input)
            return f"תוצאות ההערכה:\n{evaluation_result}"
        
        logger.info("No specific command/tool triggered by keywords. Generating generic response via PydenticAI agent or direct LLM call.")
        
        generic_prompt_for_llm = (
            f"{self.system_prompt}\n\n" 
            f"האתגר הנוכחי הוא: {self.current_challenge_topic if self.current_challenge_topic else 'לא נקבע עדיין'}\n\n"
            f"המשתמש אומר: \"{user_message}\"\n\n"
            "כיצד עליך להגיב באופן מועיל ושיחתי בהתאם לתפקידך כמנהל המשחק? "
            "אם המשתמש שואל על משהו שאינו קשור ישירות למשחק, הזכר לו בעדינות את מטרת המשחק או הצע להתחיל אתגר חדש."
        )
        try:
            response = await self.gemini_service.generate_text(prompt=generic_prompt_for_llm) 
            return response if response and "Placeholder response" not in response else "אני לא בטוח איך להגיב על זה. אפשר לנסות משהו אחר?"
        except Exception as e:
            logger.error(f"Error generating generic response via GeminiService: {e}", exc_info=True)
            return "מצטער, היתה לי שגיאה פנימית."


async def main_test(): 
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing UserInteractionAgent (conceptual with refined prompts)...")

    if __package__ is None or __package__ == '':
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        from app.services.gemini_service import GeminiService
        from app.agents.challenge_tool import ChallengeGenerationTool
        from app.agents.evaluation_tool import EvaluationTool


    class DummyGeminiService: 
        async def generate_text(self, prompt: str): 
            prompt_snippet = prompt[:70].replace('\n', ' ') # Corrected line
            logger.info(f"DummyGeminiService.generate_text called with prompt snippet: '{prompt_snippet}...'")
            if "אתה מנחה משחק אתגרים יצירתי ומהנה" in prompt: 
                return "אתגר דמה: צור יצור מכוכב אחר באמצעות שלושה פריטי מטבח."
            elif "אתה שופט מומחה וחסר פניות" in prompt: 
                return "תמונה 1 - ציון: 9/10\nתמונה 2 - ציון: 6/10\nהתמונה הזוכה היא: תמונה 1\nהסבר קצר לזכייה: הפגין יצירתיות יוצאת דופן."
            else: 
                return f"זוהי תגובה גנרית לדמה עבור הקלט האחרון."


    gemini_service_instance = DummyGeminiService() 
    logger.info("Using DummyGeminiService for UserInteractionAgent test.")

    try:
        agent = UserInteractionAgent(gemini_service=gemini_service_instance)
        
        response_rules = await agent.process_user_interaction('ספר לי את הכללים')
        logger.info(f"Agent (Rules): {response_rules}")

        response_new_challenge = await agent.process_user_interaction('אתגר חדש')
        logger.info(f"Agent (New Challenge): {response_new_challenge}")
        logger.info(f"Current topic in agent: {agent.current_challenge_topic}")
        
        dummy_captions = {"caption1": "יצור עם עיני זיתים ואף גזר", "caption2": "צלחת ריקה"}
        if agent.current_challenge_topic and "מצטער" not in agent.current_challenge_topic and "שגיאה" not in agent.current_challenge_topic:
            response_eval = await agent.process_user_interaction('הנה ההגשות שלי', image_captions=dummy_captions)
            logger.info(f"Agent (Evaluation): {response_eval}")
        else:
            logger.info("Agent: Skipping evaluation test as challenge was not properly generated by dummy.")

        response_generic = await agent.process_user_interaction('מה מזג האוויר היום?')
        logger.info(f"Agent (Generic): {response_generic}")

    except ImportError as e:
        logger.error(f"PydenticAI library not found or not correctly mocked: {e}. This test is conceptual.")
    except Exception as e:
        logger.error(f"Error during conceptual agent test: {e}", exc_info=True)

if __name__ == '__main__':
    asyncio.run(main_test())
