# tests/services/test_llm_service.py
import pytest
from openai import APIError 
from app.services.llm_service import LLMService 
import logging 

@pytest.fixture
def mock_get_openai_api_key(mocker):
    # Patch where it's looked up by the service module
    return mocker.patch("app.services.llm_service.get_openai_api_key")

@pytest.fixture
def mock_openai_client(mocker):
    mock_client_instance = mocker.MagicMock()
    mock_chat_completions = mocker.MagicMock()
    mock_chat_completions.create = mocker.MagicMock()
    mock_client_instance.chat = mocker.MagicMock()
    mock_client_instance.chat.completions = mock_chat_completions
    
    mock_openai_constructor = mocker.patch("app.services.llm_service.OpenAI", return_value=mock_client_instance)
    
    return mock_openai_constructor, mock_client_instance

# --- LLMService Initialization Tests ---
def test_llm_service_initialization_success(mock_get_openai_api_key, mock_openai_client, caplog):
    """Test successful LLMService initialization."""
    mock_get_openai_api_key.return_value = "test_api_key"
    
    with caplog.at_level(logging.INFO):
        service = LLMService()
    
    mock_get_openai_api_key.assert_called_once()
    mock_openai_client[0].assert_called_once_with(api_key="test_api_key")
    assert service.client is not None 
    assert "OpenAI client initialized successfully." in caplog.text

def test_llm_service_initialization_no_api_key(mock_get_openai_api_key, caplog):
    """Test LLMService initialization failure when API key is missing."""
    mock_get_openai_api_key.side_effect = ValueError("OPENAI_API_KEY environment variable not set by mock.")
    
    with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable not set by mock."):
        with caplog.at_level(logging.ERROR):
            LLMService()
    
    # LLMService logs the error it catches from get_openai_api_key
    assert "ValueError during LLMService initialization: OPENAI_API_KEY environment variable not set by mock." in caplog.text

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
    service.client.chat.completions.create.side_effect = APIError(message="API connection error", request=None, body=None)
    
    with caplog.at_level(logging.ERROR):
        topic = service.generate_challenge_topic()
    
    assert "OpenAI API error: API connection error" in topic
    # The logged message includes the error string from the APIError
    assert "OpenAI API error while generating topic: API connection error" in caplog.text


def test_generate_challenge_topic_no_content(mock_get_openai_api_key, mock_openai_client, caplog, mocker):
    """Test LLM returning no content for topic."""
    mock_get_openai_api_key.return_value = "test_api_key"
    service = LLMService()

    mock_response = mocker.MagicMock()
    mock_choice = mocker.MagicMock()
    mock_message = mocker.MagicMock()
    mock_message.content = None 
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
    service = LLMService() 
    
    with caplog.at_level(logging.WARNING):
        result = service.evaluate_submissions("", EVAL_CAPTION1, EVAL_CAPTION2)
    assert "Error: Topic and both captions must be provided for evaluation." in result
    assert "Evaluation called with missing topic or captions." in caplog.text
    caplog.clear() 

    with caplog.at_level(logging.WARNING):
        result = service.evaluate_submissions(EVAL_TOPIC, "", EVAL_CAPTION2)
    assert "Error: Topic and both captions must be provided for evaluation." in result
    assert "Evaluation called with missing topic or captions." in caplog.text


def test_evaluate_submissions_api_error(mock_get_openai_api_key, mock_openai_client, caplog):
    """Test API error during evaluation."""
    mock_get_openai_api_key.return_value = "test_api_key"
    service = LLMService()
    service.client.chat.completions.create.side_effect = APIError(message="API eval error", request=None, body=None)
    
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
    mock_message.content = None 
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    service.client.chat.completions.create.return_value = mock_response

    with caplog.at_level(logging.WARNING):
        result = service.evaluate_submissions(EVAL_TOPIC, EVAL_CAPTION1, EVAL_CAPTION2)
    assert "Error: LLM did not return an evaluation" in result
    assert "LLM did not return an evaluation." in caplog.text
