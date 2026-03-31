from flask import Blueprint, render_template, request, redirect, url_for, session
from models.db import users_collection
import bcrypt
import re

auth = Blueprint('auth', __name__)

# ---------------- LOGIN ----------------
@auth.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        login_input = request.form.get('login')
        password = request.form.get('password')

        # Find user by email or username
        user = users_collection.find_one({
            "$or": [
                {"email": login_input},
                {"username": login_input}
            ]
        })

        if user:
            # IMPORTANT: Ensure stored password is treated as bytes for bcrypt
            stored_password = user['password']
            if isinstance(stored_password, str):
                stored_password = stored_password.encode('utf-8')

            if bcrypt.checkpw(password.encode('utf-8'), stored_password):
                session['user'] = user.get('email')
                # If app_main is in app.py, use 'app_main'. 
                # If it's in a blueprint, use 'blueprint_name.app_main'
                return redirect(url_for('app_main'))
        
        error = 'Invalid Credentials. Please try again.'

    return render_template('login.html', error=error)


# ---------------- SIGNUP ----------------
@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')

        # 1. Validation logic
        password_pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
        
        if not re.match(password_pattern, password):
            error = 'Password: 8+ chars, Uppercase, Lowercase, Number, & Special Char.'
        elif users_collection.find_one({"email": email}):
            error = 'Email already registered.'
        elif users_collection.find_one({"username": username}):
            error = 'Username already taken.'
        else:
            # 2. Hash and DECODE to string for MongoDB compatibility
            hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            users_collection.insert_one({
                "email": email,
                "username": username,
                "password": hashed_pw
            })
            return redirect(url_for('auth.login'))

    return render_template('signup.html', error=error)


# ---------------- LOGOUT ----------------
@auth.route('/logout')
def logout():
    session.clear() # Clear everything for safety
    return redirect(url_for('home'))
