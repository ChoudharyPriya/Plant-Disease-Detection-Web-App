import logging
import requests
import random

# Configure logging
logger = logging.getLogger(__name__)

def get_weather_data(location, api_key):
    """
    Get current weather data for a location using OpenWeatherMap API
    
    Args:
        location (str): City name or location
        api_key (str): OpenWeatherMap API key
    
    Returns:
        dict: Weather data, or None if request failed
    """
    if not api_key:
        logger.warning("No weather API key provided")
        return simulate_weather_data(location)
    
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Weather API error: {response.status_code} - {response.text}")
            return simulate_weather_data(location)
            
    except Exception as e:
        logger.error(f"Error fetching weather data: {str(e)}")
        return simulate_weather_data(location)

def simulate_weather_data(location):
    """
    Simulate weather data when API is not available
    
    Args:
        location (str): Location to simulate data for
    
    Returns:
        dict: Simulated weather data
    """
    logger.info(f"Generating simulated weather data for {location}")
    
    # Base weather simulation on the current time of year
    # This is a placeholder - in a real app, proper API credentials should be used
    return {
        "name": location,
        "main": {
            "temp": 18 + random.uniform(-5, 10),  # 13-28°C
            "humidity": random.randint(55, 95),   # 55-95%
            "pressure": random.randint(1000, 1020)
        },
        "weather": [
            {
                "main": random.choice(["Clear", "Clouds", "Rain", "Drizzle"]),
                "description": "Simulated weather conditions"
            }
        ],
        "wind": {
            "speed": random.uniform(1, 8),  # 1-8 m/s
            "deg": random.randint(0, 359)
        },
        "simulated": True
    }

def check_disease_risk(weather_data):
    """
    Check if current weather conditions pose a risk for plant diseases
    
    Args:
        weather_data (dict): Weather data from OpenWeatherMap API
    
    Returns:
        list: List of disease risk assessments
    """
    try:
        if not weather_data:
            return []
        
        # Extract relevant weather parameters
        temperature = weather_data.get('main', {}).get('temp')
        humidity = weather_data.get('main', {}).get('humidity')
        weather_condition = weather_data.get('weather', [{}])[0].get('main', '')
        
        # List to store risks
        risks = []
        
        # Check for Late Blight conditions (cool, humid)
        if temperature and humidity and temperature < 25 and temperature > 10 and humidity > 80:
            risks.append({
                "disease": "Tomato Late Blight",
                "risk": "high",
                "severity": 4,
                "reason": f"Temperature {temperature:.1f}°C and humidity {humidity}% ideal for late blight"
            })
            
        # Check for Powdery Mildew conditions (moderate temperature, high humidity)
        if temperature and humidity and temperature > 15 and temperature < 30 and humidity > 70:
            risks.append({
                "disease": "Powdery Mildew",
                "risk": "moderate",
                "severity": 3,
                "reason": f"Moderate temperature {temperature:.1f}°C with humidity {humidity}% supports mildew growth"
            })
            
        # Check for Leaf Mold conditions (warm, very humid)
        if temperature and humidity and temperature > 20 and temperature < 28 and humidity > 85:
            risks.append({
                "disease": "Tomato Leaf Mold",
                "risk": "high",
                "severity": 4,
                "reason": f"Warm temperature {temperature:.1f}°C with high humidity {humidity}% creates ideal leaf mold conditions"
            })
            
        # Check for rainy conditions (general disease risk)
        if weather_condition in ['Rain', 'Drizzle', 'Thunderstorm']:
            risks.append({
                "disease": "Multiple Fungal Diseases",
                "risk": "increased",
                "severity": 3,
                "reason": f"Current {weather_condition} conditions increase fungal spore spread"
            })
            
        # Return assessed risks
        return risks
        
    except Exception as e:
        logger.error(f"Error checking disease risk: {str(e)}")
        return []

def get_disease_weather_warnings(location, disease_name, api_key):
    """
    Get detailed disease warnings based on weather forecast
    
    Args:
        location (str): City name or location
        disease_name (str): The name of the disease to check for
        api_key (str): OpenWeatherMap API key
    
    Returns:
        dict: Warning information including risk level and recommendations
    """
    try:
        # Get current weather
        weather_data = get_weather_data(location, api_key)
        if not weather_data:
            return None
            
        # Extract key data
        temperature = weather_data.get('main', {}).get('temp', 20)
        humidity = weather_data.get('main', {}).get('humidity', 70)
        
        # Risk assessment logic based on disease
        risk_level = "Low"
        warning = "Current conditions are not favorable for disease development."
        recommendations = "Continue regular monitoring of your plants."
        
        if disease_name == "Tomato Late Blight":
            if temperature < 25 and temperature > 10 and humidity > 80:
                risk_level = "High"
                warning = "Current cool, humid conditions are highly favorable for Late Blight development and spread."
                recommendations = "Apply protective fungicides immediately. Remove any infected leaves. Ensure good air circulation."
            elif temperature < 27 and humidity > 70:
                risk_level = "Moderate"
                warning = "Conditions are somewhat favorable for Late Blight development."
                recommendations = "Monitor plants closely for symptoms. Consider preventive fungicide application."
                
        elif disease_name == "Apple Scab":
            if temperature < 24 and temperature > 10 and humidity > 75:
                risk_level = "High"
                warning = "Current temperature and humidity conditions create high risk for Apple Scab infection."
                recommendations = "Apply protective fungicides. Remove fallen leaves. Prune for better air circulation."
                
        elif disease_name == "Powdery Mildew":
            if temperature > 15 and temperature < 30 and humidity > 60 and humidity < 90:
                risk_level = "High"
                warning = "Current moderate temperature and humidity range is optimal for Powdery Mildew."
                recommendations = "Apply sulfur-based fungicides. Increase air circulation around plants."
                
        elif disease_name == "Tomato Leaf Mold":
            if temperature > 20 and temperature < 28 and humidity > 85:
                risk_level = "Very High"
                warning = "Warm, very humid conditions create perfect environment for Leaf Mold proliferation."
                recommendations = "Improve greenhouse ventilation. Remove infected leaves. Apply approved fungicides."
                
        # Return warning information    
        return {
            "risk_level": risk_level,
            "warning": warning,
            "recommendations": recommendations,
            "current_temp": temperature,
            "current_humidity": humidity
        }
        
    except Exception as e:
        logger.error(f"Error getting disease weather warnings: {str(e)}")
        return None