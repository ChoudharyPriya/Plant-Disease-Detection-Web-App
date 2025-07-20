import os
import uuid
import logging
import secrets
from datetime import datetime
from flask import render_template, request, redirect, url_for, flash, jsonify, session, abort
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import base64
import cv2
from io import BytesIO

from app import app, db
from models import User, Scan, PlantDisease, WeatherAlert
from ml_model import predict_disease, load_model, preprocess_image
from weather_service import get_weather_data, check_disease_risk, get_disease_weather_warnings

# Configure logging
logger = logging.getLogger(__name__)

# CSRF Protection
def generate_csrf_token():
    if '_csrf_token' not in session:
        session['_csrf_token'] = secrets.token_hex(16)
    return session['_csrf_token']

def validate_csrf_token(token):
    return token == session.get('_csrf_token', None)

# Load ML model at startup
model = load_model()

# Seed database with plant diseases
def seed_plant_diseases():
    if PlantDisease.query.count() == 0:
        # Existing detailed diseases
        detailed_diseases = [
            PlantDisease(
                plant_type="Tomato",
                disease_name="Tomato Late Blight",
                description="A serious disease of tomatoes caused by the fungus Phytophthora infestans.",
                causes="Fungal infection that spreads rapidly in cool, wet weather",
                symptoms="Dark, water-soaked spots on leaves that quickly turn brown. White fungal growth may appear on undersides of leaves. Entire plant can collapse in severe cases.",
                treatment="Apply fungicides. Remove and destroy infected plants. Avoid overhead watering.",
                prevention="Use resistant varieties. Provide good air circulation. Avoid overhead watering.",
                risk_factors="High humidity (>90%), temperatures between 10-25°C, prolonged leaf wetness",
                weather_conditions="Wet, cool, humid conditions"
            ),
            PlantDisease(
                plant_type="Apple",
                disease_name="Apple Scab",
                description="A common fungal disease affecting apple trees.",
                causes="Fungal infection caused by Venturia inaequalis, particularly in cool, wet spring weather",
                symptoms="Olive-green to brown spots on leaves and fruit. Severely infected leaves may drop prematurely.",
                treatment="Apply fungicides. Prune infected branches. Practice good sanitation.",
                prevention="Choose resistant varieties. Remove fallen leaves. Apply preventive fungicides.",
                risk_factors="Wet spring weather, temperatures between 10-24°C, high humidity",
                weather_conditions="Wet, moderate temperature conditions"
            ),
            PlantDisease(
                plant_type="Potato",
                disease_name="Potato Early Blight",
                description="A fungal disease affecting potatoes and other nightshade family plants.",
                causes="Fungal infection caused by Alternaria solani, favored by warm weather and high humidity",
                symptoms="Dark brown spots with concentric rings on lower leaves. Yellowing around lesions. Leaves may drop.",
                treatment="Apply fungicides. Remove infected leaves. Ensure adequate nutrition.",
                prevention="Crop rotation. Adequate plant spacing. Mulching.",
                risk_factors="Temperatures of 24-29°C, high humidity, stressed plants",
                weather_conditions="Warm, humid conditions"
            ),
            PlantDisease(
                plant_type="Tomato",
                disease_name="Tomato Leaf Mold",
                description="A fungal disease affecting tomato leaves, particularly in humid conditions.",
                causes="Fungal infection caused by Passalora fulva, favored by high humidity and moderate temperatures",
                symptoms="Yellow patches on leaf surfaces with pale green to yellowish spots. Olive-green to grayish-purple velvety mold on the undersides of leaves.",
                treatment="Improve air circulation. Remove and destroy infected leaves. Apply suitable fungicides.",
                prevention="Maintain good air circulation. Avoid overcrowding plants. Keep leaves dry.",
                risk_factors="High humidity (>85%), temperatures between 20-25°C, poor air circulation",
                weather_conditions="Humid, warm conditions with poor ventilation"
            ),
            PlantDisease(
                plant_type="Grape",
                disease_name="Grape Black Rot",
                description="A serious fungal disease affecting grape vines and fruit.",
                causes="Fungal infection caused by Guignardia bidwellii, spreading in warm, humid conditions",
                symptoms="Small, dark lesions on leaves that expand into reddish-brown spots with dark borders. Black, shriveled, mummified fruit.",
                treatment="Apply protective fungicides. Remove and destroy infected fruit and leaves.",
                prevention="Prune for good air circulation. Remove mummified fruit. Timely fungicide application.",
                risk_factors="Temperatures between 20-32°C, prolonged wet periods, history of infection",
                weather_conditions="Warm, humid conditions with extended wet periods"
            ),
            PlantDisease(
                plant_type="Tomato",
                disease_name="Tomato Early Blight",
                description="A common fungal disease affecting tomato plants.",
                causes="Fungal infection caused by Alternaria solani, favored by warm, humid conditions",
                symptoms="Dark brown spots with concentric rings, yellowing of surrounding leaf tissue, lower leaves affected first.",
                treatment="Remove infected leaves. Apply fungicides. Maintain adequate nutrition.",
                prevention="Mulch around plants. Practice crop rotation. Ensure sufficient plant spacing.",
                risk_factors="Warm temperatures (24-29°C), high humidity, poor air circulation",
                weather_conditions="Warm, humid conditions"
            ),
            PlantDisease(
                plant_type="Healthy",
                disease_name="Healthy Plant",
                description="A plant showing no signs of disease or pest damage.",
                causes="N/A",
                symptoms="Normal leaf color and texture. No spots, discoloration, or abnormal growth patterns.",
                treatment="Continue regular care and maintenance.",
                prevention="Maintain good growing practices. Monitor regularly for early signs of problems.",
                risk_factors="Environmental stress, poor cultural practices, introduction of pests or pathogens",
                weather_conditions="N/A"
            )
        ]
        
        for disease in detailed_diseases:
            db.session.add(disease)
            
        # Additional plant diseases that our model can detect
        # For each of these, we'll create a basic entry
        additional_diseases = [
            # Tomato diseases
            ("Tomato", "Tomato Septoria Leaf Spot"),
            ("Tomato", "Tomato Bacterial Spot"),
            ("Tomato", "Tomato Yellow Leaf Curl Virus"),
            ("Tomato", "Tomato Mosaic Virus"),
            
            # Apple diseases
            ("Apple", "Apple Black Rot"),
            ("Apple", "Apple Cedar Rust"),
            ("Apple", "Apple Fire Blight"),
            ("Apple", "Apple Powdery Mildew"),
            
            # Potato diseases
            ("Potato", "Potato Late Blight"),
            ("Potato", "Potato Scab"),
            ("Potato", "Potato Black Leg"),
            ("Potato", "Potato Ring Rot"),
            
            # Grape diseases
            ("Grape", "Grape Esca (Black Measles)"),
            ("Grape", "Grape Leaf Blight"),
            ("Grape", "Grape Powdery Mildew"),
            
            # Corn diseases
            ("Corn", "Corn Gray Leaf Spot"),
            ("Corn", "Corn Common Rust"),
            ("Corn", "Corn Northern Leaf Blight"),
            ("Corn", "Corn Crazy Top"),
            
            # Pepper diseases
            ("Pepper", "Pepper Bacterial Spot"),
            ("Pepper", "Pepper Early Blight"),
            ("Pepper", "Pepper Phytophthora Blight"),
            
            # Citrus diseases
            ("Citrus", "Citrus Greening"),
            ("Citrus", "Citrus Black Spot"),
            ("Citrus", "Citrus Canker"),
            ("Citrus", "Citrus Scab"),
            
            # Other fruit diseases
            ("Strawberry", "Strawberry Leaf Spot"),
            ("Strawberry", "Strawberry Leaf Scorch"),
            ("Strawberry", "Strawberry Gray Mold"),
            ("Raspberry", "Raspberry Leaf Spot"),
            ("Raspberry", "Raspberry Cane Blight"),
            ("Raspberry", "Raspberry Anthracnose"),
            
            # Leafy vegetable diseases
            ("Lettuce", "Lettuce Downy Mildew"),
            ("Lettuce", "Lettuce Drop"),
            ("Lettuce", "Lettuce Gray Mold"),
            
            # Cucurbits diseases
            ("Cucumber", "Cucumber Downy Mildew"),
            ("Cucumber", "Cucumber Anthracnose"),
            ("Cucumber", "Cucumber Angular Leaf Spot"),
            ("Cucumber", "Cucumber Powdery Mildew"),
            ("Squash", "Squash Powdery Mildew"),
            ("Squash", "Squash Downy Mildew"),
            ("Squash", "Squash Virus"),
            
            # Grain/Field crops diseases
            ("Rice", "Rice Blast"),
            ("Rice", "Rice Brown Spot"),
            ("Rice", "Rice Bacterial Blight"),
            ("Rice", "Rice Sheath Blight"),
            ("Wheat", "Wheat Leaf Rust"),
            ("Wheat", "Wheat Stem Rust"),
            ("Wheat", "Wheat Stripe Rust"),
            ("Wheat", "Wheat Powdery Mildew"),
            ("Wheat", "Wheat Septoria"),
            ("Soybean", "Soybean Rust"),
            ("Soybean", "Soybean Bacterial Blight"),
            ("Soybean", "Soybean Septoria Brown Spot"),
            ("Soybean", "Soybean Downy Mildew"),
            
            # Cotton diseases
            ("Cotton", "Cotton Bacterial Blight"),
            ("Cotton", "Cotton Verticillium Wilt"),
            ("Cotton", "Cotton Fusarium Wilt"),
            ("Cotton", "Cotton Leaf Spot"),
            
            # Generic diseases (for unknown plants)
            ("Other", "Late Blight"),
            ("Other", "Early Blight"),
            ("Other", "Leaf Spot"),
            ("Other", "Powdery Mildew"),
            ("Other", "Rust Disease"),
            ("Other", "Leaf Curl"),
            ("Other", "Black Spot"),
            ("Other", "Bacterial Spot"),
            ("Other", "Anthracnose"),
            ("Other", "Downy Mildew"),
            ("Other", "Mosaic Virus"),
            ("Other", "Root Rot"),
            ("Other", "Blight Disease")
        ]
        
        # Add basic information for each additional disease
        for plant_type, disease_name in additional_diseases:
            # Check if we already added this disease in our detailed list
            exists = False
            for d in detailed_diseases:
                if d.disease_name == disease_name:
                    exists = True
                    break
                    
            if not exists:
                disease = PlantDisease(
                    plant_type=plant_type,
                    disease_name=disease_name,
                    description=f"A disease affecting {plant_type} plants.",
                    causes="Various pathogens including fungi, bacteria, or viruses.",
                    symptoms="May include leaf discoloration, spots, wilting, or stunted growth.",
                    treatment="Remove infected plant material. Apply appropriate fungicides or bactericides based on specific diagnosis.",
                    prevention="Maintain proper plant spacing. Avoid overhead watering. Practice crop rotation. Use resistant varieties when available.",
                    risk_factors="Environmental stress, poor air circulation, excessive moisture, contaminated soil.",
                    weather_conditions="Varies by specific disease type"
                )
                db.session.add(disease)
        
        db.session.commit()
        logger.info("Database seeded with plant diseases")

with app.app_context():
    seed_plant_diseases()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def save_uploaded_file(file):
    """Save an uploaded file and return the path"""
    # Create a unique filename
    filename = str(uuid.uuid4()) + secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Save the file
    file.save(file_path)
    return file_path

def save_base64_image(base64_str):
    """Save a base64 encoded image and return the path"""
    try:
        # Create a unique filename
        filename = str(uuid.uuid4()) + '.jpg'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Create upload directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Convert base64 to image and save
        img_data = base64.b64decode(base64_str.split(',')[1])
        with open(file_path, 'wb') as f:
            f.write(img_data)
        
        return file_path
    except Exception as e:
        logger.error(f"Error saving base64 image: {str(e)}")
        return None

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html', csrf_token=generate_csrf_token())

@app.route('/register', methods=['POST'])
def register():
    try:
        # Validate CSRF token
        csrf_token = request.form.get('csrf_token')
        if not validate_csrf_token(csrf_token):
            flash('Invalid form submission, please try again')
            return redirect(url_for('index'))
            
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        location = request.form.get('location', '')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('index'))
            
        if User.query.filter_by(email=email).first():
            flash('Email already exists')
            return redirect(url_for('index'))
        
        new_user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            location=location
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        session['user_id'] = new_user.id
        flash('Registration successful!')
        return redirect(url_for('dashboard'))
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        flash('An error occurred during registration')
        return redirect(url_for('index'))

@app.route('/login', methods=['POST'])
def login():
    try:
        # Validate CSRF token
        csrf_token = request.form.get('csrf_token')
        if not validate_csrf_token(csrf_token):
            flash('Invalid form submission, please try again')
            return redirect(url_for('index'))
            
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            flash('Login successful!')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        flash('An error occurred during login')
        return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please login first')
        return redirect(url_for('index'))
    
    user = User.query.get(session['user_id'])
    
    # Get weather data if user has location set
    weather_data = None
    disease_risks = []
    
    if user.location:
        weather_data = get_weather_data(user.location, app.config["WEATHER_API_KEY"])
        if weather_data:
            disease_risks = check_disease_risk(weather_data)
    
    # Get recent scans
    recent_scans = Scan.query.filter_by(user_id=user.id).order_by(Scan.timestamp.desc()).limit(5).all()
    
    # Get active weather alerts
    alerts = WeatherAlert.query.filter_by(user_id=user.id, is_active=True).all()
    
    return render_template(
        'dashboard.html', 
        user=user, 
        weather_data=weather_data,
        disease_risks=disease_risks,
        recent_scans=recent_scans,
        alerts=alerts,
        csrf_token=generate_csrf_token()
    )
    
@app.route('/ar-scan')
def ar_scan():
    if 'user_id' not in session:
        flash('Please login first')
        return redirect(url_for('index'))
    
    return render_template('ar_scanner.html', csrf_token=generate_csrf_token())

@app.route('/scan', methods=['POST'])
def scan():
    if 'user_id' not in session:
        if request.is_json:
            return jsonify({'error': 'Not authenticated'}), 401
        else:
            flash('Please login first')
            return redirect(url_for('index'))
    
    user_id = session['user_id']
    user = User.query.get(user_id)
    
    # Check for AR scan (from canvas data)
    is_ar_scan = request.form.get('is_ar_scan') == 'true'
    
    try:
        # Get plant type
        plant_type = request.form.get('plant_type', 'Unknown')
        
        # Process the image (regular upload or AR capture)
        file_path = None
        
        # Check if this is an AR scan with base64 data
        if is_ar_scan and 'plant_image_data' in request.form:
            # This is AR scan with base64 data
            base64_image = request.form.get('plant_image_data')
            if base64_image:
                file_path = save_base64_image(base64_image)
                logger.info(f"Saved base64 image from AR scan to {file_path}")
        
        # Check if there's a file upload (either AR or regular)
        elif 'plant_image' in request.files:
            file = request.files['plant_image']
            
            if not is_ar_scan and file.filename == '':
                flash('No selected file')
                return redirect(url_for('dashboard'))
            
            if not is_ar_scan and not allowed_file(file.filename):
                flash('Invalid file type. Please upload a JPG, JPEG or PNG image.')
                return redirect(url_for('dashboard'))
            
            file_path = save_uploaded_file(file)
            logger.info(f"Saved uploaded file to {file_path}")
        
        # If no image is provided
        if not file_path:
            error_msg = "No image data found in request"
            logger.error(error_msg)
            if is_ar_scan or request.is_json:
                return jsonify({'error': error_msg}), 400
            else:
                flash('No image found')
                return redirect(url_for('dashboard'))
        
        # Process the image and make prediction
        processed_image = preprocess_image(file_path)
        
        # Check if the image was processed correctly
        if processed_image is None:
            logger.error("Failed to process image")
            error_msg = "Could not process the uploaded image. Please try a different image."
            if is_ar_scan or request.is_json:
                return jsonify({'error': error_msg}), 400
            else:
                flash(error_msg)
                return redirect(url_for('dashboard'))
                
        # Get disease prediction
        disease, confidence = predict_disease(model, processed_image)
        
        # Convert numpy float to Python float if needed (to avoid database errors)
        if hasattr(confidence, 'item'):
            confidence = float(confidence)
        
        # If plant type is "Unknown" but we could determine a disease, infer plant type from disease
        if (plant_type == "Unknown" or plant_type == "Unknown Plant") and disease != "Healthy Plant" and disease != "Error in analysis":
            # Extract plant type from disease name (e.g., "Tomato Late Blight" -> "Tomato")
            if disease.startswith("Tomato"):
                plant_type = "Tomato"
            elif disease.startswith("Apple"):
                plant_type = "Apple"
            elif disease.startswith("Potato"):
                plant_type = "Potato"
            elif disease.startswith("Grape"):
                plant_type = "Grape"
            elif disease.startswith("Corn"):
                plant_type = "Corn"
            elif disease.startswith("Pepper"):
                plant_type = "Pepper"
            elif disease.startswith("Citrus"):
                plant_type = "Citrus"
            elif disease.startswith("Strawberry"):
                plant_type = "Strawberry"
            elif disease.startswith("Raspberry"):
                plant_type = "Raspberry"
            elif disease.startswith("Lettuce"):
                plant_type = "Lettuce"
            elif disease.startswith("Cucumber"):
                plant_type = "Cucumber"
            elif disease.startswith("Squash"):
                plant_type = "Squash"
            elif disease.startswith("Rice"):
                plant_type = "Rice"
            elif disease.startswith("Wheat"):
                plant_type = "Wheat"
            elif disease.startswith("Soybean"):
                plant_type = "Soybean"
            elif disease.startswith("Cotton"):
                plant_type = "Cotton"
            else:
                # Handle generic disease names without plant type prefixes
                plant_type = "Other"
                
        logger.info(f"Plant type: {plant_type}, Disease: {disease}, Confidence: {confidence:.2f}")
        
        # Get weather data if user has location
        weather_data = None
        weather_json = None
        temperature = None
        humidity = None
        
        if user.location:
            weather_data = get_weather_data(user.location, app.config["WEATHER_API_KEY"])
            if weather_data:
                weather_json = str(weather_data)
                temperature = weather_data.get('main', {}).get('temp')
                humidity = weather_data.get('main', {}).get('humidity')
        
        # Get disease details from database
        disease_info = PlantDisease.query.filter_by(disease_name=disease).first()
        
        # Provide detailed treatment information even if not in database
        if disease_info and disease_info.treatment:
            treatment = disease_info.treatment
        else:
            # Generate sensible default treatment based on disease type
            if "Blight" in disease:
                treatment = "Remove and destroy infected plant parts. Apply copper-based or appropriate fungicides. Improve air circulation and avoid overhead watering. Practice crop rotation for future plantings."
            elif "Leaf Spot" in disease or "Leaf Mold" in disease:
                treatment = "Remove infected leaves. Apply appropriate fungicide. Increase plant spacing for better air circulation. Avoid wetting leaves when watering."
            elif "Rust" in disease:
                treatment = "Apply sulfur-based fungicides early in infection. Remove and destroy infected plant debris. Maintain good plant spacing and air circulation."
            elif "Bacterial" in disease:
                treatment = "Remove infected plants to prevent spread. Apply copper-based bactericides. Avoid working with plants when wet. Practice crop rotation."
            elif "Powdery Mildew" in disease:
                treatment = "Apply sulfur or potassium bicarbonate-based fungicide. Improve air circulation. Avoid excessive nitrogen fertilization. Remove severely infected plant parts."
            elif "Virus" in disease:
                treatment = "Remove and destroy infected plants to prevent spread. Control insect vectors with appropriate methods. Use resistant varieties in future plantings."
            elif "Scab" in disease:
                treatment = "Apply lime to increase soil pH for potato scab. For apple scab, use fungicides during growing season. Practice good sanitation and remove fallen leaves and fruit."
            elif "Black" in disease:
                treatment = "Prune away affected areas. Apply appropriate fungicides. Improve drainage. Remove and destroy fallen leaves and fruit. Maintain good air circulation."
            elif "Healthy Plant" in disease:
                treatment = "Continue good care practices including proper watering, fertilization, and regular monitoring."
            else:
                treatment = "Remove and destroy severely infected plants. Apply appropriate fungicides or bactericides based on specific diagnosis. Improve plant growing conditions with adequate spacing, proper watering, and balanced fertilization."
        
        # Save scan to database
        new_scan = Scan(
            user_id=user_id,
            plant_type=plant_type,
            image_path=file_path,
            disease=disease,
            confidence=confidence,
            location=user.location,
            weather_conditions=weather_json,
            temperature=temperature,
            humidity=humidity,
            treatment=treatment,
            timestamp=datetime.utcnow()
        )
        
        db.session.add(new_scan)
        db.session.commit()
        
        # Check if current weather conditions pose a risk for the detected disease
        if weather_data and disease_info:
            disease_risks = check_disease_risk(weather_data)
            for risk in disease_risks:
                if risk['disease'] == disease:
                    new_alert = WeatherAlert(
                        user_id=user_id,
                        alert_type=f"High risk for {disease}",
                        description=f"Current weather conditions ({risk['reason']}) increase the risk of {disease}. {disease_info.prevention}",
                        severity=risk['severity'],
                        timestamp=datetime.utcnow(),
                        is_active=True
                    )
                    db.session.add(new_alert)
            
            db.session.commit()
        
        # Handle the response based on request type
        if is_ar_scan or request.is_json:
            # For AR scans, return JSON
            return jsonify({
                'success': True,
                'scan_id': new_scan.id,
                'plant_type': plant_type,  # Include the detected plant type
                'disease': disease,
                'confidence': confidence,
                'treatment_summary': treatment[:300] + '...' if len(treatment) > 300 else treatment
            })
        else:
            # For regular uploads, redirect to results page
            return redirect(url_for('view_scan', scan_id=new_scan.id))
            
    except Exception as e:
        logger.error(f"Scan error: {str(e)}")
        if is_ar_scan or request.is_json:
            return jsonify({'error': 'An error occurred during scanning', 'details': str(e)}), 500
        else:
            flash('An error occurred during scanning. Please try again.')
            return redirect(url_for('dashboard'))

@app.route('/scan/<int:scan_id>')
def view_scan(scan_id):
    if 'user_id' not in session:
        flash('Please login first')
        return redirect(url_for('index'))
    
    scan = Scan.query.get_or_404(scan_id)
    
    # Check if the scan belongs to the current user
    if scan.user_id != session['user_id']:
        abort(403)
    
    # Get disease info
    disease_info = PlantDisease.query.filter_by(disease_name=scan.disease).first()
    
    # Get weather warning if location is set
    weather_warning = None
    if scan.location:
        user = User.query.get(session['user_id'])
        weather_warning = get_disease_weather_warnings(
            user.location, 
            scan.disease, 
            app.config["WEATHER_API_KEY"]
        )
    
    return render_template(
        'scan_results.html',
        scan=scan,
        disease_info=disease_info,
        weather_warning=weather_warning
    )

@app.route('/api/update_plant_type', methods=['POST'])
def update_plant_type():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    scan_id = data.get('scan_id')
    plant_type = data.get('plant_type')
    
    if not scan_id or not plant_type:
        return jsonify({'error': 'Missing required fields'}), 400
    
    scan = Scan.query.get_or_404(scan_id)
    
    # Check if the scan belongs to the current user
    if scan.user_id != session['user_id']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    scan.plant_type = plant_type
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/api/dismiss_alert/<int:alert_id>', methods=['POST'])
def dismiss_alert(alert_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    alert = WeatherAlert.query.get_or_404(alert_id)
    
    # Check if the alert belongs to the current user
    if alert.user_id != session['user_id']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    alert.is_active = False
    db.session.commit()
    
    return jsonify({'success': True})

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500
