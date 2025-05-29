# tests/services/test_llm_service.py
import pytest
from openai import APIError # For simulating API errors
from app.services.llm_service import LLMService 
import logging # Required for caplog

# We mock get_openai_api_key from app.services.llm_service's perspective,
# as that's where it's imported and used.
# Or, if it's imported as "from app.config import get_openai_api_key", then "app.config.get_openai_api_key"
# LLMService imports it as: from app.config import get_openai_api_key
# So, the path for patching should be 'app.services.llm_service.get_openai_api_key' if it were imported like that,
# or 'app.config.get_openai_api_key' if we want to patch it where it's defined and LLMService picks up the mock.
# Given LLMService does `from app.config import get_openai_api_key`, we patch it in `app.config`.

@pytest.fixture
def mock_get_openai_api_key(mocker):
    # This patches the function where it is defined, so LLMService will use the mock.
    return mocker.patch("app.config.get_openai_api_key")

@pytest.fixture
def mock_openai_client(mocker):
    mock_client_instance = mocker.MagicMock()
    # Mock the chat.completions.create method
    mock_chat_completions = mocker.MagicMock()
    mock_chat_completions.create = mocker.MagicMock()
    mock_client_instance.chat = mocker.MagicMock()
    mock_client_instance.chat.completions = mock_chat_completions
    
    # Mock the OpenAI class constructor to return our mock_client_instance
    mock_openai_constructor = mocker.patch("app.services.llm_service.OpenAI", return_value=mock_client_instance)
    
    return mock_openai_constructor, mock_client_instance

# --- LLMService Initialization Tests ---
def test_llm_service_initialization_success(mock_get_openai_api_key, mock_openai_client, caplog):
    """Test successful LLMService initialization."""
    mock_get_openai_api_key.return_value = "test_api_key"
    
    with caplog.at_level(logging.INFO):
        service = LLMService()
    
    mock_get_openai_api_key.assert_called_once()
    # mock_openai_client[0] is the constructor mock
    mock_openai_client[0].assert_called_once_with(api_key="test_api_key")
    assert service.client is not None # service.client is mock_openai_client[1]
    assert "OpenAI client initialized successfully." in caplog.text

def test_llm_service_initialization_no_api_key(mock_get_openai_api_key, caplog):
    """Test LLMService initialization failure when API key is missing."""
    # get_openai_api_key (from app.config) logs an error and raises ValueError
    # LLMService catches this, logs an error, and re-raises.
    mock_get_openai_api_key.side_effect = ValueError("OPENAI_API_KEY environment variable not set.")
    
    with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable not set."):
        with caplog.at_level(logging.ERROR): # Capture ERROR level for LLMService log
            LLMService()
    
    # Check log from LLMService
    assert "ValueError during LLMService initialization: OPENAI_API_KEY environment variable not set." in caplog.text
    # Check log from app.config.get_openai_api_key (if it was also captured, depends on logger propagation)
    # For now, focusing on LLMService's direct logs.

# --- generate_challenge_topic Tests ---
def test_generate_challenge_topic_success(mock_get_openai_api_key, mock_openai_client, caplog, mocker):
    """Test successful challenge topic generation."""
    mock_get_openai_api_key.return_value = "test_api_key"
    service = LLMService()

    mock_response = mocker.MagicMock()
    mock_choice = mocker.MagicMock()
    mock_message = mocker.MagicMock()
    mock_message.content = "Generated test topic"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    
    service.client.chat.completions.create.return_value = mock_response
    
    with caplog.at_level(logging.INFO):
        topic = service.generate_challenge_topic()
    
    service.client.chat.completions.create.assert_called_once()
    assert topic == "Generated test topic"
    assert f"Successfully generated challenge topic: {topic}" in caplog.text

def test_generate_challenge_topic_api_error(mock_get_openai_api_key, mock_openai_client, caplog):
    """Test API error during topic generation."""
    mock_get_openai_api_key.return_value = "test_api_key"
    service = LLMService()
    # The client is mock_openai_client[1]
    # service.client.chat.completions.create is already a MagicMock
    service.client.chat.completions.create.side_effect = APIError("API connection error", request=None, body=None)
    
    with caplog.at_level(logging.ERROR):
        topic = service.generate_challenge_topic()
    
    assert "OpenAI API error: API connection error" in topic
    assert "OpenAI API error while generating topic: API connection error" in caplog.text

def test_generate_challenge_topic_no_content(mock_get_openai_api_key, mock_openai_client, caplog, mocker):
    """Test LLM returning no content for topic."""
    mock_get_openai_api_key.return_value = "test_api_key"
    service = LLMService()

    mock_response = mocker.MagicMock()
    mock_choice = mocker.MagicMock()
    mock_message = mocker.MagicMock()
    mock_message.content = None # Simulate no content
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    service.client.chat.completions.create.return_value = mock_response

    with caplog.at_level(logging.WARNING):
        topic = service.generate_challenge_topic()
    assert "Error: LLM did not return a topic" in topic
    assert "LLM did not return a topic." in caplog.text


# --- evaluate_submissions Tests ---
EVAL_TOPIC = "Test Topic"
EVAL_CAPTION1 = "Caption for image 1"
EVAL_CAPTION2 = "Caption for image 2"

def test_evaluate_submissions_success(mock_get_openai_api_key, mock_openai_client, caplog, mocker):
    """Test successful submission evaluation."""
    mock_get_openai_api_key.return_value = "test_api_key"
    service = LLMService()
    
    mock_response = mocker.MagicMock()
    mock_choice = mocker.MagicMock()
    mock_message = mocker.MagicMock()
    mock_message.content = "Player 1 wins!"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    service.client.chat.completions.create.return_value = mock_response
    
    with caplog.at_level(logging.INFO):
        result = service.evaluate_submissions(EVAL_TOPIC, EVAL_CAPTION1, EVAL_CAPTION2)
    
    service.client.chat.completions.create.assert_called_once()
    assert result == "Player 1 wins!"
    assert "Submissions evaluated successfully by LLM." in caplog.text

def test_evaluate_submissions_missing_inputs(mock_get_openai_api_key, caplog):
    """Test evaluation with missing inputs."""
    mock_get_openai_api_key.return_value = "test_api_key"
    service = LLMService() # Initialize service for each sub-test or pass it if state doesn't change
    
    with caplog.at_level(logging.WARNING):
        result = service.evaluate_submissions("", EVAL_CAPTION1, EVAL_CAPTION2)
    assert "Error: Topic and both captions must be provided for evaluation." in result
    assert "Evaluation called with missing topic or captions." in caplog.text
    caplog.clear() # Clear logs for next assertion

    with caplog.at_level(logging.WARNING):
        result = service.evaluate_submissions(EVAL_TOPIC, "", EVAL_CAPTION2)
    assert "Error: Topic and both captions must be provided for evaluation." in result
    assert "Evaluation called with missing topic or captions." in caplog.text


def test_evaluate_submissions_api_error(mock_get_openai_api_key, mock_openai_client, caplog):
    """Test API error during evaluation."""
    mock_get_openai_api_key.return_value = "test_api_key"
    service = LLMService()
    service.client.chat.completions.create.side_effect = APIError("API eval error", request=None, body=None)
    
    with caplog.at_level(logging.ERROR):
        result = service.evaluate_submissions(EVAL_TOPIC, EVAL_CAPTION1, EVAL_CAPTION2)
    
    assert "OpenAI API error: API eval error" in result
    assert "OpenAI API error during evaluation: API eval error" in caplog.text

def test_evaluate_submissions_no_content(mock_get_openai_api_key, mock_openai_client, caplog, mocker):
    """Test LLM returning no content for evaluation."""
    mock_get_openai_api_key.return_value = "test_api_key"
    service = LLMService()

    mock_response = mocker.MagicMock()
    mock_choice = mocker.MagicMock()
    mock_message = mocker.MagicMock()
    mock_message.content = None # Simulate no content
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    service.client.chat.completions.create.return_value = mock_response

    with caplog.at_level(logging.WARNING):
        result = service.evaluate_submissions(EVAL_TOPIC, EVAL_CAPTION1, EVAL_CAPTION2)
    assert "Error: LLM did not return an evaluation" in result
    assert "LLM did not return an evaluation." in caplog.text
