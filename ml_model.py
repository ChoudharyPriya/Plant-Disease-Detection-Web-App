import os
import logging
import numpy as np
import cv2
import random
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

def load_model():
    """
    In a real application, we would load a trained model.
    For this demo, we'll create a simple model that uses basic image processing.
    """
    logger.info("Loading plant disease detection model")
    return {
        "name": "Plant Disease Detector",
        "version": "1.0",
        "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def preprocess_image(image_path):
    """Preprocess an image for analysis with improved handling for AR camera captures"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image from {image_path}")
            return None
        
        # Get original dimensions for logging
        original_height, original_width = image.shape[:2]
        logger.info(f"Original image dimensions: {original_width}x{original_height}")
        
        # Apply some image enhancements for better feature extraction
        # 1. Apply slight Gaussian blur to reduce noise 
        denoised = cv2.GaussianBlur(image, (3, 3), 0)
        
        # 2. Apply contrast enhancement
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 3. Resize with aspect ratio preservation to standard size
        target_size = (224, 224)
        h, w = enhanced.shape[:2]
        # Calculate new dimensions while maintaining aspect ratio
        if h > w:
            new_h, new_w = target_size[1], int(w * target_size[1] / h)
        else:
            new_h, new_w = int(h * target_size[0] / w), target_size[0]
        
        # Resize to the new dimensions
        resized = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create a black canvas of target size
        canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
        # Calculate position to paste the resized image (center it)
        x_offset = (target_size[0] - new_w) // 2
        y_offset = (target_size[1] - new_h) // 2
        
        # Paste the resized image onto the canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Convert to RGB (from BGR)
        rgb_image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        normalized = rgb_image / 255.0
        
        logger.info(f"Successfully preprocessed image from {image_path}")
        return normalized
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def color_analysis(processed_image):
    """
    Analyze color distributions in the image
    This is a simplified approach that could detect color patterns 
    associated with plant diseases
    """
    try:
        # Extract color channels
        red_channel = processed_image[:, :, 0]
        green_channel = processed_image[:, :, 1]
        blue_channel = processed_image[:, :, 2]
        
        # Calculate mean values of channels
        red_mean = np.mean(red_channel)
        green_mean = np.mean(green_channel)
        blue_mean = np.mean(blue_channel)
        
        # Calculate ratio of green to other channels
        # Higher green ratio typically indicates healthier plants
        green_ratio = green_mean / (red_mean + blue_mean + 0.001)  # Avoid division by zero
        
        # Calculate color variance as a measure of spots/patterns
        color_variance = np.std(processed_image)
        
        return {
            "green_ratio": green_ratio,
            "color_variance": color_variance,
            "channel_means": [red_mean, green_mean, blue_mean]
        }
    except Exception as e:
        logger.error(f"Error in color analysis: {str(e)}")
        return None

def detect_leaf_features(processed_image):
    """
    Detect leaf-specific features in the image
    This simulates analyzing leaf shape, texture, and patterns
    """
    try:
        # Convert to grayscale for edge detection
        gray = np.mean(processed_image, axis=2)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Simple edge detection (in a real app, we would use more sophisticated methods)
        edges = cv2.Canny((blurred * 255).astype(np.uint8), 50, 150)
        
        # Count edge pixels as a measure of leaf texture/patterns
        edge_ratio = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Create a measure of texture variation 
        texture_var = np.var(gray)
        
        return {
            "edge_ratio": edge_ratio,
            "texture_variation": texture_var
        }
    except Exception as e:
        logger.error(f"Error in leaf feature detection: {str(e)}")
        return None

def determine_plant_type(color_features, leaf_features):
    """
    Attempt to determine the plant type based on image features
    
    In a real application, this would use a properly trained classifier.
    """
    # Get RGB channel means
    r_mean, g_mean, b_mean = color_features["channel_means"]
    
    # Expanded list of plant types
    plant_types = [
        "Tomato", "Apple", "Potato", "Grape", "Corn", "Pepper", 
        "Citrus", "Strawberry", "Raspberry", "Lettuce", "Cucumber", 
        "Squash", "Rice", "Wheat", "Soybean", "Cotton"
    ]
    
    # Green ratio - how green the plant is relative to other colors
    green_dominance = g_mean / (r_mean + b_mean + 0.001)
    
    # Texture features
    texture_var = leaf_features["texture_variation"]
    edge_ratio = leaf_features["edge_ratio"]
    
    # Simplified heuristic rules for plant type classification
    # These are not scientifically accurate, just for demonstration
    
    # Red fruits/vegetables
    if r_mean > 0.4 and r_mean > g_mean and r_mean > b_mean:
        if g_mean > 0.3:  # Tomatoes tend to have some green
            return "Tomato"
        elif texture_var > 0.03:
            return "Strawberry"
        else:
            return "Apple"
    
    # Green leafy plants with different texture patterns
    elif g_mean > 0.3 and g_mean > r_mean and g_mean > b_mean:
        if texture_var < 0.015:  # Smooth leaves
            return "Lettuce"
        elif texture_var > 0.04:  # Very textured leaves
            return "Cucumber"
        elif edge_ratio > 0.2:  # Complex leaf edges
            return "Potato"
        else:
            return random.choice(["Pepper", "Squash"])
    
    # Yellowish plants
    elif g_mean > 0.25 and r_mean > 0.25 and r_mean + g_mean > 0.6:
        if texture_var > 0.035:
            return "Corn"
        else:
            return "Wheat"
            
    # Purplish tint with moderate green might be grape
    elif b_mean > 0.2 and r_mean > 0.25 and g_mean > 0.3:
        if texture_var > 0.03:
            return "Grape"
        else:
            return "Raspberry"
    
    # Darker green plants
    elif g_mean > 0.15 and g_mean < 0.3 and r_mean < 0.2:
        return random.choice(["Soybean", "Cotton"])
    
    # Use intelligent random selection based on image features
    elif green_dominance > 1.5:  # Mostly green
        return random.choice(["Lettuce", "Cucumber", "Pepper", "Squash"])
    elif texture_var > 0.04:  # Very textured
        return random.choice(["Potato", "Grape", "Tomato"])
    elif r_mean > 0.3:  # Reddish
        return random.choice(["Tomato", "Apple", "Strawberry", "Raspberry"])
    else:
        # If nothing matches, pick a plant type randomly
        return random.choice(plant_types)
    
    return "Unknown Plant"

def predict_disease(model, processed_image):
    """
    Analyze an image to detect plant diseases
    
    In a real application, this would use a properly trained ML model.
    For this demo, we'll use simple image processing with some heuristics.
    """
    try:
        if processed_image is None:
            logger.error("Cannot process null image")
            return "Unknown", 0.0
        
        # Analyze color patterns
        color_features = color_analysis(processed_image)
        if not color_features:
            return "Error in analysis", 0.3
        
        # Detect leaf features
        leaf_features = detect_leaf_features(processed_image)
        if not leaf_features:
            return "Error in analysis", 0.3
        
        # Determine plant type (will be used to constrain disease possibilities)
        plant_type = determine_plant_type(color_features, leaf_features)
        
        # Get key features for analysis
        green_ratio = color_features["green_ratio"]
        color_variance = color_features["color_variance"]
        edge_ratio = leaf_features["edge_ratio"]
        texture_var = leaf_features["texture_variation"]
        
        # Expanded disease possibilities by plant type
        disease_map = {
            "Tomato": [
                "Tomato Late Blight", 
                "Tomato Leaf Mold", 
                "Tomato Early Blight", 
                "Tomato Septoria Leaf Spot", 
                "Tomato Bacterial Spot", 
                "Tomato Yellow Leaf Curl Virus", 
                "Tomato Mosaic Virus", 
                "Healthy Plant"
            ],
            "Apple": [
                "Apple Scab", 
                "Apple Black Rot", 
                "Apple Cedar Rust", 
                "Apple Fire Blight", 
                "Apple Powdery Mildew", 
                "Healthy Plant"
            ],
            "Potato": [
                "Potato Early Blight", 
                "Potato Late Blight", 
                "Potato Scab", 
                "Potato Black Leg", 
                "Potato Ring Rot", 
                "Healthy Plant"
            ],
            "Grape": [
                "Grape Black Rot", 
                "Grape Esca (Black Measles)", 
                "Grape Leaf Blight", 
                "Grape Powdery Mildew", 
                "Healthy Plant"
            ],
            "Corn": [
                "Corn Gray Leaf Spot", 
                "Corn Common Rust", 
                "Corn Northern Leaf Blight", 
                "Corn Crazy Top", 
                "Healthy Plant"
            ],
            "Pepper": [
                "Pepper Bacterial Spot", 
                "Pepper Early Blight", 
                "Pepper Phytophthora Blight", 
                "Healthy Plant"
            ],
            "Citrus": [
                "Citrus Greening", 
                "Citrus Black Spot", 
                "Citrus Canker", 
                "Citrus Scab", 
                "Healthy Plant"
            ],
            "Strawberry": [
                "Strawberry Leaf Spot", 
                "Strawberry Leaf Scorch", 
                "Strawberry Gray Mold", 
                "Healthy Plant"
            ],
            "Raspberry": [
                "Raspberry Leaf Spot", 
                "Raspberry Cane Blight", 
                "Raspberry Anthracnose", 
                "Healthy Plant"
            ],
            "Lettuce": [
                "Lettuce Downy Mildew", 
                "Lettuce Drop", 
                "Lettuce Gray Mold", 
                "Healthy Plant"
            ],
            "Cucumber": [
                "Cucumber Downy Mildew", 
                "Cucumber Anthracnose", 
                "Cucumber Angular Leaf Spot", 
                "Cucumber Powdery Mildew", 
                "Healthy Plant"
            ],
            "Squash": [
                "Squash Powdery Mildew", 
                "Squash Downy Mildew", 
                "Squash Virus", 
                "Healthy Plant"
            ],
            "Rice": [
                "Rice Blast", 
                "Rice Brown Spot", 
                "Rice Bacterial Blight", 
                "Rice Sheath Blight", 
                "Healthy Plant"
            ],
            "Wheat": [
                "Wheat Leaf Rust", 
                "Wheat Stem Rust", 
                "Wheat Stripe Rust", 
                "Wheat Powdery Mildew", 
                "Wheat Septoria", 
                "Healthy Plant"
            ],
            "Soybean": [
                "Soybean Rust", 
                "Soybean Bacterial Blight", 
                "Soybean Septoria Brown Spot", 
                "Soybean Downy Mildew", 
                "Healthy Plant"
            ],
            "Cotton": [
                "Cotton Bacterial Blight", 
                "Cotton Verticillium Wilt", 
                "Cotton Fusarium Wilt", 
                "Cotton Leaf Spot", 
                "Healthy Plant"
            ],
            "Unknown Plant": [
                "Late Blight", 
                "Early Blight", 
                "Leaf Spot", 
                "Powdery Mildew", 
                "Rust Disease", 
                "Leaf Curl", 
                "Black Spot", 
                "Bacterial Spot", 
                "Anthracnose", 
                "Downy Mildew", 
                "Mosaic Virus", 
                "Root Rot", 
                "Blight Disease", 
                "Healthy Plant"
            ]
        }
        
        # Get possible diseases for this plant type
        possible_diseases = disease_map.get(plant_type, disease_map["Unknown Plant"])
        
        # Check if it's healthy first
        is_likely_healthy = green_ratio > 0.9 and color_variance < 0.15 and texture_var < 0.02
        
        if is_likely_healthy:
            disease = "Healthy Plant"
            confidence = 0.75 + (random.random() * 0.2)  # 0.75-0.95
        else:
            # We need to select a disease based on features
            
            # Higher color variance and edge features often indicate disease
            disease_likelihood = min(0.9, color_variance * 2 + edge_ratio * 3)
            
            if disease_likelihood > 0.5:
                # Filter out "Healthy Plant" from options
                disease_options = [d for d in possible_diseases if d != "Healthy Plant"]
                if not disease_options:  # In case we filtered out all options
                    disease_options = ["Unknown Disease"]
                
                disease = random.choice(disease_options)
                confidence = 0.6 + (disease_likelihood * 0.3)  # 0.6-0.87
            else:
                # Lower likelihood, could be healthy or mild disease
                if random.random() > 0.4:  # 60% chance of healthy when unsure
                    disease = "Healthy Plant"
                    confidence = 0.6 + (random.random() * 0.2)  # 0.6-0.8
                else:
                    # Filter out "Healthy Plant" from options
                    disease_options = [d for d in possible_diseases if d != "Healthy Plant"]
                    if not disease_options:  # In case we filtered out all options
                        disease_options = ["Unknown Disease"]
                    
                    disease = random.choice(disease_options)
                    confidence = 0.5 + (random.random() * 0.2)  # 0.5-0.7
        
        # Log the detection and return results
        logger.info(f"Detected {disease} with confidence {confidence:.2f}")
        return disease, confidence
        
    except Exception as e:
        logger.error(f"Error predicting disease: {str(e)}")
        return "Error in analysis", 0.1