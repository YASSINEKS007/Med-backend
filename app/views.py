from .layers import Patches, PatchEncoder  # Import your custom layers
from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from django.core.files.storage import default_storage
import logging
from medmnist import INFO

# Get metadata for all MedMNIST datasets
all_info = INFO
# Set up logging
logger = logging.getLogger(__name__)

# Get metadata for all MedMNIST datasets


# Dictionary mapping disease classes to model paths
MODEL_PATHS = {
    'pathmnist': r'C:\Users\user\OneDrive\Bureau\SDIA\Machine-Learning\mini-projet-2\backend\app\pathmnist_model(1).keras',
    'bloodmnist': r'C:\Users\user\OneDrive\Bureau\SDIA\Machine-Learning\mini-projet-2\backend\app\bloodmnist_model.keras',
    'tissuemnist': r'C:\Users\user\OneDrive\Bureau\SDIA\Machine-Learning\mini-projet-2\backend\app\tissuemnist_model.keras',
}

# Set up custom objects for the model
custom_objects = {
    'Patches': Patches,
    'PatchEncoder': PatchEncoder
}

# def preprocess_image(image_path):
#     """
#     Preprocesses an image to be compatible with the model (64x64x3).

#     Parameters:
#     - image_path (str): The path to the image file.

#     Returns:
#     - numpy.ndarray: A preprocessed image that can be fed into the model.
#     """
#     try:
#         # Load image from file
#         image = Image.open(image_path)

#         # Resize the image to 64x64 (matching input size for the model)
#         image = image.resize((64, 64))

#         # Convert image to numpy array
#         image_array = np.array(image)

#         # Normalize the image to [0, 1] for the model
#         image_array = image_array.astype('float32') / 255.0

#         # Check if image is grayscale and add channels if needed
#         if image_array.ndim == 2:  # Grayscale image (64, 64)
#             image_array = np.expand_dims(image_array, axis=-1)
#         # RGB image (64, 64, 3)
#         elif image_array.ndim == 3 and image_array.shape[2] == 3:
#             pass  # Correct shape for RGB images

#         # If the model expects 3 channels (RGB), ensure this is the case for grayscale as well
#         if image_array.shape[-1] != 3:
#             image_array = np.repeat(image_array, 3, axis=-1)

#         # Add batch dimension (1, 64, 64, 3) for a single image
#         image_array = np.expand_dims(image_array, axis=0)

#         return image_array
#     except Exception as e:
#         logger.error(f"Error preprocessing image: {e}")
#         raise



model_path = tf.keras.models.load_model(MODEL_PATHS["pathmnist"], custom_objects=custom_objects)
model_blood = tf.keras.models.load_model(MODEL_PATHS["bloodmnist"], custom_objects=custom_objects)
model_tissue = tf.keras.models.load_model(MODEL_PATHS["tissuemnist"], custom_objects=custom_objects)


# @api_view(["POST"])
# @parser_classes([MultiPartParser, FormParser])
# def process_image(request):
#     # Fetch disease class from the GET request or POST data
#     disease_class = request.data.get("class")
#     print(disease_class)
#     # Class label mapping
#     cls  = all_info[disease_class]["label"]

    
#     if not disease_class or disease_class not in MODEL_PATHS:
#         return JsonResponse({"error": "Invalid or missing disease class."}, status=400)
    
#     MODEL_PATH = MODEL_PATHS[disease_class]

#     # Load the model with custom layers
#     try:
#         if disease_class == "pathmnist":
#             model = model_path
#         elif disease_class == "bloodmnist":
#             model = model_blood
#         else:
#             model = model_tissue
#         logger.info(f"Model successfully loaded from {MODEL_PATH}")
#     except Exception as e:
#         logger.error(f"Error loading model: {e}")
#         return JsonResponse({"error": "Error loading model."}, status=500)

#     if request.method == 'POST' and request.FILES.get('image'):
#         # Get the uploaded file
#         uploaded_file = request.FILES['image']

#         # Save the image temporarily
#         file_name = default_storage.save('temp_image.png', uploaded_file)
#         image_path = os.path.join(default_storage.location, file_name)

#         try:
#             # Preprocess the image
#             processed_image = preprocess_image(image_path)

#             # Check input shape
#             if processed_image.shape != (1, 64, 64, 3):
#                 return JsonResponse({"error": "Input image has incorrect shape."}, status=400)

#             # Make prediction
#             prediction = model.predict(processed_image)
#             predicted_class_idx = int(np.argmax(prediction, axis=-1))

#             # Prepare the result
#             result = {
#                 "disease_class": cls.get(str(predicted_class_idx), "Unknown class"),
#                 "confidence": float(np.max(prediction, axis=-1)),
#             }

#             # Return the prediction as JSON
#             return JsonResponse(result)

#         except Exception as e:
#             logger.error(f"Error processing image: {str(e)}")
#             return JsonResponse({"error": str(e)}, status=500)

#         finally:
#             # Clean up the temporary file
#             if os.path.exists(image_path):
#                 os.remove(image_path)

#     return JsonResponse({"error": "No image uploaded."}, status=400)

def preprocess_image(image_path, expected_channels=3):
    """
    Preprocesses an image to be compatible with the model.

    Parameters:
    - image_path (str): The path to the image file.
    - expected_channels (int): The number of channels expected by the model (default: 3).

    Returns:
    - numpy.ndarray: A preprocessed image that can be fed into the model.
    """
    try:
        # Load image from file
        image = Image.open(image_path)

        # Resize the image to 64x64 (matching input size for the model)
        image = image.resize((64, 64))

        # Convert image to numpy array
        image_array = np.array(image)

        # Normalize the image to [0, 1] for the model
        image_array = image_array.astype('float32') / 255.0

        # Check if the image is grayscale and expand channels if needed
        if image_array.ndim == 2:  # Grayscale image (64, 64)
            image_array = np.expand_dims(image_array, axis=-1)

        # Ensure the image has the required number of channels
        if image_array.shape[-1] != expected_channels:
            if expected_channels == 3:
                image_array = np.repeat(image_array, 3, axis=-1)
            elif expected_channels == 1:
                image_array = image_array[..., :1]  # Ensure single channel

        # Add batch dimension (1, 64, 64, expected_channels) for a single image
        image_array = np.expand_dims(image_array, axis=0)

        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def process_image(request):
    # Fetch disease class from the GET request or POST data
    disease_class = request.data.get("class")
    print(disease_class)
    # Class label mapping
    cls = all_info[disease_class]["label"]
    print(cls)

    if not disease_class or disease_class not in MODEL_PATHS:
        return JsonResponse({"error": "Invalid or missing disease class."}, status=400)

    # Select the appropriate model and expected input channels
    if disease_class == "pathmnist":
        model = model_path
        expected_channels = 3
    elif disease_class == "bloodmnist":
        model = model_blood
        expected_channels = 3
    elif disease_class == "tissuemnist":
        model = model_tissue
        expected_channels = 1
    else:
        return JsonResponse({"error": "Invalid disease class."}, status=400)

    if request.method == 'POST' and request.FILES.get('image'):
        # Get the uploaded file
        uploaded_file = request.FILES['image']

        # Save the image temporarily
        file_name = default_storage.save('temp_image.png', uploaded_file)
        image_path = os.path.join(default_storage.location, file_name)

        try:
            # Preprocess the image
            processed_image = preprocess_image(image_path, expected_channels=expected_channels)

            # Check input shape
            if processed_image.shape != (1, 64, 64, expected_channels):
                return JsonResponse({"error": "Input image has incorrect shape."}, status=400)

            # Make prediction
            prediction = model.predict(processed_image)
            predicted_class_idx = int(np.argmax(prediction, axis=-1))

            # Prepare the result
            result = {
                "disease_class": cls.get(str(predicted_class_idx), "Unknown class"),
                "confidence": float(np.max(prediction, axis=-1)),
            }

            # Return the prediction as JSON
            return JsonResponse(result)

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return JsonResponse({"error": str(e)}, status=500)

        finally:
            # Clean up the temporary file
            if os.path.exists(image_path):
                os.remove(image_path)

    return JsonResponse({"error": "No image uploaded."}, status=400)
