from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from pymongo import MongoClient
import bcrypt
import re
import os

# 1. Setup MongoDB
MONGO_URI = os.environ.get("MONGO_URI")

# Use fallback URI if environment variable is not set
if not MONGO_URI:
    MONGO_URI = "mongodb+srv://mdsahilshaikh1506_db_user:Cf0XXCw3f4KompTg@cluster0.tjihg5s.mongodb.net/?appName=Cluster0"

try:
    client = MongoClient(MONGO_URI)
    db = client["ai_behavior_db"]
    users_collection = db["users"]
    client.admin.command('ping')
    print("✅ MongoDB Connected Successfully")
except Exception as e:
    print(f"❌ MongoDB Connection Error: {e}")
    users_collection = None

auth = Blueprint('auth', __name__)

# Helper to validate password
def is_valid_password(password):
    # 8+ chars, Uppercase, Lowercase, Number, Special
    pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
    return re.match(pattern, password)

# ---------------- LOGIN ----------------
@auth.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('user'):
        return redirect(url_for('index'))

    if request.method == 'POST':
        login_input = request.form.get('login')
        password = request.form.get('password')

        user = users_collection.find_one({
            "$or": [
                {"email": login_input},
                {"username": login_input}
            ]
        })

        if user:
            # FIX: Check if stored password is str or bytes to avoid AttributeError
            stored_pw = user['password']
            if isinstance(stored_pw, str):
                stored_pw = stored_pw.encode('utf-8')

            if bcrypt.checkpw(password.encode('utf-8'), stored_pw):
                session['user'] = user['email']
                session['username'] = user['username']
                return redirect(url_for('index'))
        
        flash('Invalid username/email or password.', 'danger')

    return render_template('login.html')


# ---------------- SIGNUP ----------------
@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')

        # Validate password
        if not is_valid_password(password):
            flash('Password must be 8+ characters, include Uppercase, Lowercase, Number, and Special character.', 'warning')
            return render_template('signup.html')

        # Check if email or username already exists
        if users_collection.find_one({"$or": [{"email": email}, {"username": username}]}):
            flash('Email or Username already registered.', 'warning')
            return render_template('signup.html')

        # Hash and DECODE to string for MongoDB compatibility
        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        try:
            result = users_collection.insert_one({
                "email": email,
                "username": username,
                "password": hashed_pw
            })
            print(f"✅ User registered successfully: {email}")
            flash('Signup successful! Please log in.', 'success')
            return redirect(url_for('auth.login'))
        except Exception as e:
            print(f"❌ Error saving user: {e}")
            flash('Error during registration. Please try again.', 'danger')
            return render_template('signup.html')

    return render_template('signup.html')


# ---------------- LOGOUT ----------------
@auth.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))
