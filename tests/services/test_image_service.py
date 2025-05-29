# tests/services/test_image_service.py
import pytest
from PIL import Image
from app.services.image_service import ImageService # Adjust import if needed
import logging # Required for caplog

# Mock PIL Image class for testing if real images aren't desired/available
class MockPILImage:
    def __init__(self, mode="RGB"):
        self.mode = mode
        self.convert_called_with = None

    def convert(self, mode):
        self.mode = mode
        self.convert_called_with = mode
        return self

@pytest.fixture
def mock_blip_processor(mocker):
    mock_processor_instance = mocker.MagicMock()
    mock_processor_instance.decode = mocker.MagicMock(return_value="A mock caption")
    # Mock the __call__ method for processor(images=...)
    # This mock structure allows us to chain .to() call
    mock_inputs_to_device = mocker.MagicMock() # This is what processor(...).to(device) returns
    mock_inputs = mocker.MagicMock(return_value=mock_inputs_to_device) # This is what processor(...) returns
    mock_processor_instance.__call__ = mock_inputs
    
    mock_from_pretrained_processor = mocker.patch(
        "app.services.image_service.BlipProcessor.from_pretrained",
        return_value=mock_processor_instance
    )
    return mock_from_pretrained_processor, mock_processor_instance

@pytest.fixture
def mock_blip_model(mocker):
    mock_model_instance = mocker.MagicMock()
    # Mock the generate method
    mock_model_instance.generate = mocker.MagicMock(return_value=["mock_output_tensor"]) # generate returns a list of tensors (or tensor like objects)

    # This is the object returned by BlipForConditionalGeneration.from_pretrained(...)
    mock_pretrained_model_object = mocker.MagicMock()
    mock_pretrained_model_object.to = mocker.MagicMock(return_value=mock_model_instance) # .to(device) returns the model itself
    mock_pretrained_model_object.generate = mock_model_instance.generate # also make generate available directly if .to is chained weirdly

    mock_from_pretrained_model = mocker.patch(
        "app.services.image_service.BlipForConditionalGeneration.from_pretrained",
        return_value=mock_pretrained_model_object
    )
    return mock_from_pretrained_model, mock_model_instance


@pytest.fixture
def mock_torch_cuda_is_available(mocker):
    return mocker.patch("app.services.image_service.torch.cuda.is_available", return_value=False) # Assume CPU for tests

def test_image_service_initialization(mock_blip_processor, mock_blip_model, mock_torch_cuda_is_available, caplog):
    """Test ImageService constructor loads model and processor."""
    with caplog.at_level(logging.INFO):
        service = ImageService()
    
    mock_blip_processor[0].assert_called_once_with("Salesforce/blip-image-captioning-base", cache_dir="/app/hf_cache")
    mock_blip_model[0].assert_called_once_with("Salesforce/blip-image-captioning-base", cache_dir="/app/hf_cache")
    
    # Check that the model's .to(device) method was called
    mock_blip_model[0].return_value.to.assert_called_once_with("cpu")

    assert service.model is not None
    assert service.processor is not None
    assert "BLIP model and processor loaded successfully." in caplog.text

def test_generate_caption_success(mock_blip_processor, mock_blip_model, mock_torch_cuda_is_available, caplog):
    """Test successful caption generation."""
    service = ImageService()
    
    # The service uses the instances from the mocked from_pretrained
    # So service.processor is mock_blip_processor[1]
    # And service.model is the return_value of mock_blip_model[0].return_value.to() which is mock_blip_model[1]

    mock_image = MockPILImage()
    
    with caplog.at_level(logging.INFO):
        caption = service.generate_caption(mock_image)
    
    service.processor.__call__.assert_called_once()
    # Check if it was called with the image and return_tensors="pt"
    # service.processor.__call__.assert_called_with(images=mock_image, return_tensors="pt") # This gets tricky with MagicMock equality

    # Check that the result of processor() call had .to(device) called on it
    service.processor.__call__.return_value.to.assert_called_once_with("cpu")
    
    service.model.generate.assert_called_once()
    service.processor.decode.assert_called_once_with(["mock_output_tensor"], skip_special_tokens=True)
    
    assert caption == "A mock caption"
    assert "Caption generated successfully" in caplog.text

def test_generate_caption_rgb_conversion(mock_blip_processor, mock_blip_model, mock_torch_cuda_is_available, caplog):
    """Test image is converted to RGB if not already."""
    service = ImageService()
    mock_image = MockPILImage(mode="RGBA")
    
    with caplog.at_level(logging.INFO):
        caption = service.generate_caption(mock_image)
        
    assert caption == "A mock caption" 
    assert mock_image.convert_called_with == "RGB"
    assert "Image is not in RGB mode, converting." in caplog.text

def test_generate_caption_no_image(mock_blip_processor, mock_blip_model, mock_torch_cuda_is_available, caplog):
    """Test behavior when no image is provided."""
    service = ImageService()
    with caplog.at_level(logging.ERROR): # expecting an error log
        caption = service.generate_caption(None)
    assert "Error: No image provided" in caption
    assert "generate_caption called with no image provided." in caplog.text

def test_generate_caption_processor_error(mock_blip_processor, mock_blip_model, mock_torch_cuda_is_available, caplog, mocker):
    """Test error handling if processor fails."""
    service = ImageService()
    # Override the specific processor instance's __call__ method used by the service
    service.processor.__call__ = mocker.MagicMock(side_effect=Exception("Processor failed"))
    
    mock_image = MockPILImage()
    with caplog.at_level(logging.ERROR): # Expecting an exception log
        caption = service.generate_caption(mock_image)
    
    assert "Error generating caption" in caption
    assert "Error generating caption." in caplog.text # General message from logger.exception
    assert "Processor failed" in caplog.text # Specific exception message

def test_generate_caption_model_error(mock_blip_processor, mock_blip_model, mock_torch_cuda_is_available, caplog, mocker):
    """Test error handling if model.generate fails."""
    service = ImageService()
    # Override the specific model instance's generate method
    service.model.generate = mocker.MagicMock(side_effect=Exception("Model generation failed"))

    mock_image = MockPILImage()
    with caplog.at_level(logging.ERROR): # Expecting an exception log
        caption = service.generate_caption(mock_image)
    
    assert "Error generating caption" in caption
    assert "Error generating caption." in caplog.text # General message from logger.exception
    assert "Model generation failed" in caplog.text # Specific exception message

def test_image_service_initialization_failure(mocker, mock_torch_cuda_is_available, caplog):
    """Test ImageService constructor handles model loading failure."""
    mocker.patch("app.services.image_service.BlipProcessor.from_pretrained", side_effect=Exception("Failed to load processor"))
    mocker.patch("app.services.image_service.BlipForConditionalGeneration.from_pretrained", side_effect=Exception("Should not be reached if processor fails first"))

    with pytest.raises(Exception, match="Failed to load processor"):
        with caplog.at_level(logging.ERROR):
            ImageService()
    
    assert "Failed to load BLIP model or processor." in caplog.text
    assert "Failed to load processor" in caplog.text
