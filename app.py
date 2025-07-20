import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Database setup
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///plant_disease.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Upload folder configuration
app.config["UPLOAD_FOLDER"] = "static/uploads/"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

# Weather API configuration
app.config["WEATHER_API_KEY"] = os.environ.get("WEATHER_API_KEY", "")
app.config["WEATHER_API_URL"] = "https://api.openweathermap.org/data/2.5/weather"

# Initialize the app with the extension
db.init_app(app)

with app.app_context():
    # Import models
    import models
    
    # Create all database tables
    db.create_all()
    
    # Import routes after db initialization
    import routes

logger.info("Application initialized")