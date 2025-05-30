# app/ui/gradio_interface.py
import gradio as gr
from PIL import Image
import logging
import asyncio # Required for running async methods if called from sync context (though Gradio handles it)

# Assuming UserInteractionAgent is in app.agents.user_interaction_agent
from app.agents.user_interaction_agent import UserInteractionAgent 
# Assuming ImageService is in app.services.image_service
from app.services.image_service import ImageService

logger = logging.getLogger(__name__)

class GradioInterface:
    def __init__(self, agent: UserInteractionAgent, image_service: ImageService):
        """
        Initializes the GradioInterface.
        Args:
            agent: An instance of UserInteractionAgent.
            image_service: An instance of ImageService for generating captions.
        """
        self.agent = agent
        self.image_service = image_service
        logger.info("GradioInterface initialized with UserInteractionAgent and ImageService.")

    async def _handle_user_message(self, user_input: str, history: list) -> tuple[str, list]:
        """
        Generic handler for text input that might not be a specific command.
        This allows for more conversational interaction if the agent supports it.
        (Currently not directly wired up to a separate input field, but can be used)
        """
        logger.info(f"UI: Handling generic user message: '{user_input}'")
        response = await self.agent.process_user_interaction(user_message=user_input)
        history.append((user_input, response))
        return "", history # Clear input, update history

    async def _handle_generate_topic(self) -> str:
        """
        Handles the 'generate topic' button click.
        Sends a message to the agent to generate a new topic.
        """
        logger.info("UI: Generate topic button clicked.")
        # The user_message tells the agent the intent.
        response = await self.agent.process_user_interaction(user_message="אתגר חדש") 
        # The response here is expected to be the new topic itself or an error message.
        if "Error:" in response or "שגיאה" in response or "מצטער" in response: # Basic error check
            logger.error(f"Agent returned an error or issue for new challenge: {response}")
        else:
            logger.info(f"Agent generated new topic: {response[:100]}...") # Log snippet of topic
        return response

    async def _handle_check_images(self, image1_pil: Optional[Image.Image], image2_pil: Optional[Image.Image], current_topic_display: str) -> str:
        """
        Handles the 'check images' button click.
        Generates captions and sends them to the agent for evaluation.
        Args:
            image1_pil: PIL Image object from the first image input.
            image2_pil: PIL Image object from the second image input.
            current_topic_display: The topic currently displayed in the UI (from topic_output Textbox).
        Returns:
            str: The evaluation result from the agent.
        """
        logger.info("UI: Check images button clicked.")

        if not current_topic_display or "לחץ על הכפתור" in current_topic_display or "שגיאה ב" in current_topic_display:
            # Added "שגיאה ב" to catch errors in topic display from previous step
            logger.warning("UI: Check images called but no valid topic is displayed.")
            return "אנא צור אתגר תחילה על ידי לחיצה על 'צור לנו אתגר חדש'."

        if image1_pil is None or image2_pil is None:
            logger.warning("UI: Check images called but one or both images are missing.")
            return "יש להעלות שתי תמונות כדי לבדוק."

        # Ensure image_service is available
        if not self.image_service:
            logger.error("UI: ImageService not available.")
            return "שגיאה פנימית: שירות עיבוד התמונות אינו זמין."

        logger.info("Generating caption for image 1...")
        caption1 = await self.image_service.generate_caption(image1_pil)
        if "Error:" in caption1 or not caption1: # Check for empty caption too
            logger.error(f"Failed to generate caption for Image 1: {caption1}")
            return f"שגיאה ביצירת תיאור לתמונה 1: {caption1 if caption1 else 'תיאור ריק'}"
        logger.info(f"Caption 1: {caption1[:50]}...")

        logger.info("Generating caption for image 2...")
        caption2 = await self.image_service.generate_caption(image2_pil)
        if "Error:" in caption2 or not caption2:
            logger.error(f"Failed to generate caption for Image 2: {caption2}")
            return f"שגיאה ביצירת תיאור לתמונה 2: {caption2 if caption2 else 'תיאור ריק'}"
        logger.info(f"Caption 2: {caption2[:50]}...")
            
        logger.info("Sending image captions to agent for evaluation.")
        response = await self.agent.process_user_interaction(
            user_message="הערך בבקשה את ההגשות הללו עבור האתגר הנוכחי.", 
            image_captions={"caption1": caption1, "caption2": caption2}
        )
        return response

    def create_ui(self):
        """
        Creates the Gradio UI layout and defines interactions.
        """
        logger.info("UI: Creating Gradio blocks.")
        with gr.Blocks(css="footer {visibility: hidden}") as demo: # Simple CSS to hide Gradio footer
            gr.Markdown("# אתגר התמונות (Image Challenge) - מבוסס סוכנים")

            with gr.Row():
                show_rules_button = gr.Button("הצג הוראות משחק")
            
            rules_output = gr.Textbox(label="הוראות המשחק", lines=7, interactive=False, visible=False)

            with gr.Row():
                create_topic_button = gr.Button("צור לנו אתגר חדש")
            
            topic_output = gr.Textbox(
                value="לחץ על 'צור לנו אתגר חדש' כדי להתחיל", 
                label="האתגר הנוכחי (Current Challenge)", 
                interactive=False,
                lines=2 # Allow topic to wrap for 2 lines
            )

            with gr.Row():
                image_input1 = gr.Image(type="pil", label="העלה פיתרון של מתמודד 1 (Upload Solution - Player 1)", sources=["upload", "webcam", "clipboard"])
                image_input2 = gr.Image(type="pil", label="העלה פיתרון של מתמודד 2 (Upload Solution - Player 2)", sources=["upload", "webcam", "clipboard"])
            
            with gr.Row():
                check_button = gr.Button("בדוק אותנו! (Check Us!)")
            
            result_output = gr.Textbox(label="תוצאות הבדיקה (Evaluation Results)", lines=10, interactive=False) # Increased lines

            # --- Define button/component actions ---
            
            async def _handle_show_rules_click():
                logger.info("UI: Show rules button clicked.")
                rules_text = await self.agent.process_user_interaction(user_message="הוראות המשחק")
                # Toggle visibility: if already visible and contains rules, hide it. Otherwise, show it.
                # This requires knowing the current state or making it always visible once clicked.
                # Simpler: always show/refresh.
                return gr.update(value=rules_text, visible=True)
            
            show_rules_button.click(fn=_handle_show_rules_click, inputs=[], outputs=[rules_output])

            create_topic_button.click(
                fn=self._handle_generate_topic, 
                inputs=[], 
                outputs=topic_output
            )
            
            check_button.click(
                fn=self._handle_check_images,
                inputs=[image_input1, image_input2, topic_output], # topic_output is Textbox component
                outputs=result_output
            )
            
            self.demo = demo
        return demo

    def launch(self, **kwargs):
        if not hasattr(self, 'demo') or self.demo is None:
            self.create_ui()
        logger.info("UI: Launching Gradio demo.")
        # share=True can be added here if a public link is needed and env allows.
        self.demo.launch(**kwargs)

# Optional: A simple test block for GradioInterface structure (not full app run)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing GradioInterface structure (conceptual)...")

    # Mock UserInteractionAgent and ImageService for this structural test
    class MockAgent:
        async def process_user_interaction(self, user_message: str, image_captions: Optional[Dict[str,str]] = None):
            logger.info(f"MockAgent.process_user_interaction called with: '{user_message}', Captions: {image_captions is not None}")
            if "אתגר חדש" in user_message:
                return "נושא אתגר לדוגמה מהסוכן המדומה."
            if "הוראות" in user_message:
                return "אלו הן הוראות המשחק מהסוכן המדומה."
            if image_captions:
                return f"הערכה לדוגמה מהסוכן המדומה עבור: {image_captions['caption1'][:10]}... ו-{image_captions['caption2'][:10]}..."
            return "תגובה כללית מהסוכן המדומה."

    class MockImageService:
        async def generate_caption(self, image_pil: Image.Image) -> str:
            logger.info("MockImageService.generate_caption called.")
            if image_pil is None: return "Error: No image provided."
            return "תיאור תמונה מדומה."

    mock_agent = MockAgent()
    mock_image_service = MockImageService()
    
    gradio_ui = GradioInterface(agent=mock_agent, image_service=mock_image_service)
    # gradio_ui.create_ui() # This would create actual Gradio components
    # gradio_ui.launch()    # This would try to launch the server
    logger.info("GradioInterface initialized with mock agent and image service. UI blocks can be created and launched.")
    logger.info("To run full Gradio app, execute app/main.py.")
