# Image Challenge Application

## Description
This application hosts a creative challenge game where an AI (LLM) generates a unique challenge. Two users (or one user playing two roles) can then upload images representing their attempt to solve the challenge. A separate AI evaluates the submissions based on the image content (via auto-generated captions) and the original challenge, then declares a winner with an explanation.

## Prerequisites
*   Python 3.7+
*   Git (for cloning the repository)

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

## Configuration

To use this application, you need an OpenAI API key.

1.  Obtain an API key from [OpenAI](https://platform.openai.com/signup/).
2.  Set the `OPENAI_API_KEY` environment variable. Replace `"your_actual_api_key_here"` with your actual key:

    *   On macOS and Linux:
        ```bash
        export OPENAI_API_KEY="your_actual_api_key_here"
        ```
    *   On Windows (Command Prompt):
        ```bash
        set OPENAI_API_KEY=your_actual_api_key_here
        ```
    *   On Windows (PowerShell):
        ```bash
        $env:OPENAI_API_KEY="your_actual_api_key_here"
        ```
    **Important:** Do not commit your API key directly into the code. Using environment variables is a safer practice.

## Running the Application

1.  Ensure your `OPENAI_API_KEY` is set and you are in the project's root directory with the virtual environment activated.
2.  Run the application using:
    ```bash
    python app/main.py
    ```
3.  Open your web browser and navigate to the URL displayed in your terminal (usually `http://127.0.0.1:7860` or `http://0.0.0.0:7860` as specified in the console output when the app starts).

## Project Structure (Brief Overview)

*   `app/`: Contains the core application code.
    *   `main.py`: Main application runner and the `ChallengeApplication` class orchestrating the logic.
    *   `config.py`: Handles configuration, primarily retrieving the OpenAI API key from environment variables.
    *   `models/`: Contains data model classes (e.g., `ChallengeData`, though currently a placeholder).
    *   `services/`: Houses service classes responsible for specific tasks:
        *   `image_service.py` (`ImageService`): Handles image processing, specifically generating captions using a pre-trained model.
        *   `llm_service.py` (`LLMService`): Manages interactions with the OpenAI API for generating challenge topics and evaluating submissions.
    *   `ui/`: Contains the user interface logic.
        *   `gradio_interface.py` (`GradioInterface`): Implements the web UI using Gradio.
*   `tests/`: Contains unit tests for the application.
    *   `services/`: Unit tests specifically for the service classes.
*   `requirements.txt`: Lists all Python package dependencies required for the project.
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore (e.g., virtual environment directories, Python bytecode files).
*   `README.md`: This file - provides information and instructions for the project.
