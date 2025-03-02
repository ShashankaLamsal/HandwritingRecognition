# auth.py (or wherever your auth routes are defined)

from flask import Blueprint, render_template, request, redirect, url_for
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    # Logic for login, either handling POST request or rendering GET request
    if request.method == 'POST':
        # Process login here
        pass
    return render_template('login.html')
# Define the signup route
@auth_bp.route('/signup')
def signup_page():
    return render_template('signup.html')  # Ensure signup.html exists in templates

# Add the auth blueprint to the app in app.py
