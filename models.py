from datetime import datetime
from flask_login import UserMixin
from app import db

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    location = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    scans = db.relationship('Scan', backref='user', lazy=True)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    plant_type = db.Column(db.String(100))
    image_path = db.Column(db.String(255), nullable=False)
    disease = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    location = db.Column(db.String(100))
    weather_conditions = db.Column(db.String(255))
    temperature = db.Column(db.Float)
    humidity = db.Column(db.Float)
    treatment = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Scan {self.id} - {self.disease}>'

class PlantDisease(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plant_type = db.Column(db.String(100), nullable=False)
    disease_name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    causes = db.Column(db.Text)
    symptoms = db.Column(db.Text)
    treatment = db.Column(db.Text)
    prevention = db.Column(db.Text)
    risk_factors = db.Column(db.Text)
    weather_conditions = db.Column(db.String(255))
    
    def __repr__(self):
        return f'<Disease {self.disease_name}>'

class WeatherAlert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    alert_type = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    severity = db.Column(db.Integer, default=1)  # 1-5 scale
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    def __repr__(self):
        return f'<Alert {self.alert_type}>'