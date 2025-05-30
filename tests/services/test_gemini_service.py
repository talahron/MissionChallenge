# tests/services/test_gemini_service.py
import pytest
import asyncio 
from unittest.mock import patch, MagicMock
import logging # Added import

from app.services.gemini_service import GeminiService, DEFAULT_GEMINI_MODEL 

@pytest.fixture
def mock_google_credentials(mocker):
    return mocker.patch("app.services.gemini_service.get_google_application_credentials")

@pytest.fixture
def mock_genai_configure(mocker):
    return mocker.patch("app.services.gemini_service.genai.configure")

@pytest.fixture
def mock_generative_model(mocker):
    mock_model_instance = MagicMock()
    mock_model_instance.generate_content = MagicMock()
    
    mock_gm_constructor = mocker.patch("app.services.gemini_service.genai.GenerativeModel", return_value=mock_model_instance)
    return mock_gm_constructor, mock_model_instance

# --- GeminiService Initialization Tests ---
def test_gemini_service_initialization_success(mock_google_credentials, mock_genai_configure, mock_generative_model, caplog):
    """Test successful GeminiService initialization."""
    mock_google_credentials.return_value = "fake/path/to/creds.json" 
    
    with caplog.at_level(logging.INFO): 
        service = GeminiService()
    
    mock_google_credentials.assert_called_once()
    mock_genai_configure.assert_not_called() 
    
    mock_generative_model[0].assert_called_once_with(DEFAULT_GEMINI_MODEL)
    assert service.client is not None
    assert "Gemini client initialized successfully." in caplog.text

def test_gemini_service_initialization_no_credentials(mock_google_credentials, caplog):
    """Test GeminiService init failure if get_google_application_credentials itself raises ValueError."""
    mock_google_credentials.side_effect = ValueError("Credentials not set by mock")
    with pytest.raises(RuntimeError, match="GeminiService initialization failed due to configuration: Credentials not set by mock"):
        with caplog.at_level(logging.ERROR): # Ensure ERROR logs are captured
            GeminiService()
    assert "Configuration error for GeminiService: Credentials not set by mock" in caplog.text

def test_gemini_service_initialization_model_failure(mock_google_credentials, mock_genai_configure, mock_generative_model, caplog):
    """Test GeminiService init failure if GenerativeModel constructor fails."""
    mock_google_credentials.return_value = "fake/path/to/creds.json" 
    mock_generative_model[0].side_effect = Exception("Model init failed") 
    
    with pytest.raises(RuntimeError, match="GeminiService initialization failed: Model init failed"):
        with caplog.at_level(logging.ERROR): # Ensure ERROR logs are captured
            GeminiService()
    assert "Failed to initialize Gemini client: Model init failed" in caplog.text


# --- GeminiService generate_text Tests ---
@pytest.mark.asyncio 
async def test_generate_text_success(mock_google_credentials, mock_genai_configure, mock_generative_model, caplog):
    """Test successful text generation."""
    mock_google_credentials.return_value = "fake/path/to/creds.json"
    service = GeminiService()
    
    mock_response = MagicMock()
    mock_part = MagicMock()
    mock_part.text = "Generated test text"
    mock_response.parts = [mock_part]
    mock_response.text = None 
    mock_response.prompt_feedbacks = []

    service.client.generate_content = MagicMock(return_value=mock_response) 
    
    prompt = "Test prompt"
    with caplog.at_level(logging.INFO): # Capture INFO logs for this part
        result = await service.generate_text(prompt)
    
    service.client.generate_content.assert_called_once_with(prompt)
    assert result == "Generated test text"
    assert f"Generating text with model {DEFAULT_GEMINI_MODEL}" in caplog.text
    assert "Gemini generated text successfully (from parts)" in caplog.text


@pytest.mark.asyncio
async def test_generate_text_api_error(mock_google_credentials, mock_genai_configure, mock_generative_model, caplog):
    """Test API error during text generation."""
    mock_google_credentials.return_value = "fake/path/to/creds.json"
    service = GeminiService()
    service.client.generate_content = MagicMock(side_effect=Exception("API error"))
    
    prompt = "Test prompt for API error"
    with caplog.at_level(logging.ERROR): # Capture ERROR logs
        result = await service.generate_text(prompt)
    
    assert "Error: LLM call failed - API error" in result
    assert "Error during Gemini text generation: API error" in caplog.text

@pytest.mark.asyncio
async def test_generate_text_no_content_parts(mock_google_credentials, mock_genai_configure, mock_generative_model, caplog):
    """Test LLM returning no usable content (no parts, no text)."""
    mock_google_credentials.return_value = "fake/path/to/creds.json"
    service = GeminiService()
    
    mock_response = MagicMock()
    mock_response.parts = [] 
    mock_response.text = None  
    mock_response.prompt_feedbacks = [] 
    
    service.client.generate_content = MagicMock(return_value=mock_response)
    
    prompt = "Test prompt for no content parts"
    with caplog.at_level(logging.WARNING): # Capture WARNING logs
        result = await service.generate_text(prompt)
    
    assert "Error: LLM returned no usable content or request was blocked." in result
    assert f"Gemini response for prompt '{prompt[:70]}...' had no usable text parts or was blocked." in caplog.text

@pytest.mark.asyncio
async def test_generate_text_no_content_text_fallback(mock_google_credentials, mock_genai_configure, mock_generative_model, caplog):
    """Test LLM returning no parts but has .text attribute."""
    mock_google_credentials.return_value = "fake/path/to/creds.json"
    service = GeminiService()
    
    mock_response = MagicMock()
    mock_response.parts = [] 
    mock_response.text = "Fallback text from .text attribute" 
    mock_response.prompt_feedbacks = []
    
    service.client.generate_content = MagicMock(return_value=mock_response)
    
    prompt = "Test prompt for text fallback"
    with caplog.at_level(logging.INFO): # Capture INFO logs
        result = await service.generate_text(prompt)
    
    assert result == "Fallback text from .text attribute"
    assert f"Generating text with model {DEFAULT_GEMINI_MODEL}" in caplog.text
    assert "Gemini generated text successfully (from .text attribute)" in caplog.text


@pytest.mark.asyncio
async def test_generate_text_blocked_prompt(mock_google_credentials, mock_genai_configure, mock_generative_model, caplog):
    """Test LLM returning blocked prompt feedback."""
    mock_google_credentials.return_value = "fake/path/to/creds.json"
    service = GeminiService()
    
    mock_response = MagicMock()
    mock_response.parts = []
    mock_response.text = None
    
    class MockPromptFeedback:
        def __init__(self, block_reason, safety_ratings):
            self.block_reason = block_reason
            self.safety_ratings = safety_ratings 
        def __str__(self): 
            return f"Block Reason: {self.block_reason}, Ratings: {self.safety_ratings}"

    mock_response.prompt_feedbacks = [MockPromptFeedback("SAFETY", [{"category": "HARM_CATEGORY_SEXUAL", "probability": "HIGH"}])]
    
    service.client.generate_content = MagicMock(return_value=mock_response)
    
    prompt = "Test prompt for blocked"
    with caplog.at_level(logging.WARNING): # Capture WARNING logs
        result = await service.generate_text(prompt)
    
    assert "Error: LLM returned no usable content or request was blocked." in result
    assert "Gemini response for prompt 'Test prompt for blocked...' had no usable text parts or was blocked." in caplog.text
    assert "Prompt Feedback: Block Reason: SAFETY" in caplog.text
