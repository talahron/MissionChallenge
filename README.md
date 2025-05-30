# Image Challenge Application

## Description
This application hosts a creative challenge game where an AI (now leveraging Google Gemini) generates a unique challenge. Users upload image solutions, which are processed to generate descriptive captions. An AI agent (conceptually built with PydenticAI) then uses these captions and the original challenge to evaluate the submissions and declare a winner with an explanation. The core logic is managed by an agent that uses specialized tools for challenge generation and evaluation.

## Prerequisites
*   Python 3.9+ (due to `asyncio.to_thread` usage, otherwise 3.7+ for most other parts)
*   Git (for cloning the repository)
*   Access to Google Cloud Platform and a configured project.

## Key Libraries
This project utilizes several key Python libraries:
*   `google-generativeai`: For interacting with Google Gemini models.
*   `pydenticai`: (Conceptual) For the underlying AI agent framework.
*   `gradio`: For creating the web-based user interface.
*   `Pillow`: For image manipulation.
*   `transformers` & `torch`: For local image captioning (BLIP model).
*   `pytest`, `pytest-mock`, `pytest-asyncio`: For unit testing.

A full list of dependencies is in `requirements.txt`.

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/username/project.git
    cd project
    ```
    *(Replace the URL and directory name with actual values if this project is hosted on Git)*

2.  **Create and activate a virtual environment (recommended):**
    *   On macOS and Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Installing `torch` can be time-consuming and require significant disk space. If you encounter issues, ensure your environment has sufficient resources and network stability.)*

## Configuration

This application now uses Google Gemini models via the `google-generativeai` library and requires authentication with Google Cloud.

1.  **Set up Google Cloud Project:**
    *   Ensure you have a Google Cloud Platform (GCP) project.
    *   Enable the "Vertex AI API" (which includes access to Gemini models) in your GCP project. You can find this under "APIs & Services" > "Enabled APIs & services".

2.  **Create a Service Account:**
    *   In your GCP project, navigate to "IAM & Admin" > "Service Accounts".
    *   Click "Create Service Account".
    *   Give it a name (e.g., "gemini-app-runner").
    *   Grant it necessary roles. For Gemini API access, the "Vertex AI User" role (`roles/aiplatform.user`) is typically sufficient. Consult Google Cloud documentation for the most up-to-date minimal required permissions.
    *   After creating the service account, select it, go to the "Keys" tab, click "Add Key", and choose "Create new key". Select "JSON" as the key type and click "Create". A JSON key file will be downloaded to your computer.
    *   **This file contains sensitive credentials – keep it secure and do not commit it to your repository!** Store it in a safe location on your machine.

3.  **Set Environment Variable:**
    *   Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the **absolute path** of the JSON key file you downloaded.

    *   On macOS and Linux:
        ```bash
        export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-file.json"
        ```
    *   On Windows (Command Prompt):
        ```bash
        set GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\service-account-file.json"
        ```
    *   On Windows (PowerShell):
        ```bash
        $env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\service-account-file.json"
        ```
    Replace the example path with the actual path to your downloaded JSON key file. The application uses these credentials via Application Default Credentials (ADC) to authenticate with Google Cloud services.

## Running the Application

1.  Ensure your `GOOGLE_APPLICATION_CREDENTIALS` environment variable is correctly set.
2.  Ensure you are in the project's root directory with the virtual environment activated.
3.  Run the application using the module execution flag for Python to ensure correct path handling:
    ```bash
    python -m app.main
    ```
4.  Open your web browser and navigate to the URL displayed in your terminal (usually `http://127.0.0.1:7860` or `http://0.0.0.0:7860` as specified in the console output when the app starts).

## Project Structure (Brief Overview)

*   `app/`: Contains the core application code.
    *   `main.py`: Main application runner that initializes services, the agent, and the UI.
    *   `config.py`: Handles configuration, like retrieving Google Cloud credentials path.
    *   `models/`: Contains data model classes (e.g., `ChallengeData`).
    *   `services/`: Houses service classes responsible for specific tasks:
        *   `gemini_service.py` (`GeminiService`): Manages interactions with Google Gemini models.
        *   `image_service.py` (`ImageService`): Handles image processing (captioning using BLIP).
        *   `llm_service.py` (`LLMService`): (Legacy) Previously managed OpenAI interactions; may be deprecated or removed.
    *   `agents/`: Contains PydenticAI agent and tool implementations.
        *   `user_interaction_agent.py` (`UserInteractionAgent`): The main agent orchestrating the game.
        *   `challenge_tool.py` (`ChallengeGenerationTool`): Tool for generating challenges.
        *   `evaluation_tool.py` (`EvaluationTool`): Tool for evaluating submissions.
    *   `ui/`: Contains the user interface logic.
        *   `gradio_interface.py` (`GradioInterface`): Implements the web UI using Gradio.
*   `tests/`: Contains unit tests for the application.
    *   `services/`: Unit tests specifically for the service classes.
*   `requirements.txt`: Lists all Python package dependencies.
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
*   `README.md`: This file - provides information and instructions for the project.

## Development Notes
*   The PydenticAI integration is based on a conceptual understanding of such frameworks. Actual class names, methods, and tool registration mechanisms from the PydenticAI library will need to be substituted for the placeholders used.
*   Ensure that the `GOOGLE_APPLICATION_CREDENTIALS` environment variable points to a valid service account JSON key with the necessary permissions for the Gemini API (e.g., Vertex AI User).
*   The BLIP models for image captioning will be downloaded to `hf_cache/` in the project root on first run of `ImageService` if not already present. This requires internet access and disk space.
