import gradio as gr
from PIL import Image
from app.main import ChallengeApplication # Assuming ChallengeApplication is in app.main
import logging

logger = logging.getLogger(__name__)

class GradioInterface:
    """
    Manages the Gradio web interface for the Challenge Application.
    """
    def __init__(self, app: ChallengeApplication):
        """
        Initializes the GradioInterface.

        Args:
            app: An instance of ChallengeApplication.
        """
        logger.info("Initializing GradioInterface...")
        self.app = app
        self.current_topic = "לחץ על הכפתור ואתגר חדש יופיע" # Initial message
        logger.info("GradioInterface initialized.")

    def _handle_generate_topic(self) -> str:
        """
        Handles the 'generate topic' button click.
        Generates a new topic using the application logic and updates the current topic.
        """
        logger.info("UI: Generate topic button clicked.")
        self.current_topic = self.app.generate_new_challenge()
        if not self.current_topic or "Error:" in self.current_topic :
             logger.warning(f"Topic generation failed or returned error. Original: '{self.current_topic}'. Setting to default error.")
             self.current_topic = "שגיאה ביצירת האתגר. נסה שוב." # Default error message if app returns None or error string
        logger.info(f"UI: Topic updated to: '{self.current_topic}'")
        return self.current_topic

    def _handle_check_images(self, image1_pil: Image.Image, image2_pil: Image.Image) -> str:
        """
        Handles the 'check images' button click.
        Evaluates the uploaded images against the current topic.

        Args:
            image1_pil: PIL Image object from the first image input.
            image2_pil: PIL Image object from the second image input.

        Returns:
            str: The evaluation result.
        """
        logger.info(f"UI: Check images button clicked. Current topic: '{self.current_topic}'. Image1 provided: {bool(image1_pil)}, Image2 provided: {bool(image2_pil)}")
        
        if self.current_topic == "לחץ על הכפתור ואתגר חדש יופיע" or "שגיאה" in self.current_topic:
            message = "אנא צור אתגר תחילה לפני בדיקת התמונות."
            logger.warning(f"UI: Check images called without a valid topic. Returning: '{message}'")
            return message

        if image1_pil is None or image2_pil is None:
            message = "יש להעלות שתי תמונות כדי לבדוק."
            logger.warning(f"UI: Check images called with missing images. Returning: '{message}'")
            return message

        # The topic is taken from self.current_topic, not directly from the textbox input here
        # to ensure consistency, as self.current_topic is updated by _handle_generate_topic.
        logger.info("UI: Calling application to evaluate submissions.")
        result = self.app.evaluate_challenge_submissions(image1_pil, image2_pil, self.current_topic)
        logger.info(f"UI: Evaluation result: '{result}'")
        return result

    def create_ui(self):
        """
        Creates the Gradio UI layout and defines interactions.
        """
        logger.info("UI: Creating Gradio blocks.")
        with gr.Blocks() as demo:
            gr.Markdown("# אתגר התמונות (Image Challenge)") # Title in Hebrew

            with gr.Row():
                create_topic_button = gr.Button("צור לנו אתגר חדש")
            
            topic_output = gr.Textbox(
                value=self.current_topic, 
                label="האתגר הנוכחי (Current Challenge)", 
                interactive=False # Usually, topic is not edited by user
            )

            with gr.Row():
                image_input1 = gr.Image(type="pil", label="העלה פיתרון של מתמודד 1 (Upload Solution - Player 1)")
                image_input2 = gr.Image(type="pil", label="העלה פיתרון של מתמודד 2 (Upload Solution - Player 2)")
            
            with gr.Row():
                check_button = gr.Button("בדוק אותנו! (Check Us!)")
            
            result_output = gr.Textbox(label="תוצאות הבדיקה (Evaluation Results)")

            # Define button actions
            create_topic_button.click(
                fn=self._handle_generate_topic, 
                inputs=[], 
                outputs=topic_output
            )
            
            check_button.click(
                fn=self._handle_check_images,
                inputs=[image_input1, image_input2],
                outputs=result_output
            )
            
            self.demo = demo
        return demo

    def launch(self, **kwargs):
        """
        Launches the Gradio interface.
        kwargs are passed to demo.launch() (e.g. share=True, server_name="0.0.0.0")
        """
        if not hasattr(self, 'demo'):
            logger.info("UI: Demo object not found, creating UI before launch.")
            self.create_ui() # Ensure UI is created if not already
        logger.info(f"UI: Launching Gradio demo with kwargs: {kwargs}")
        self.demo.launch(**kwargs)
