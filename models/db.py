from pymongo import MongoClient

# MongoDB connection
client = MongoClient("mongodb+srv://mdsahilshaikh1506_db_user:<db_password>@cluster0.tjihg5s.mongodb.net/?appName=Cluster0")

# Database
db = client["ai_behavior_db"]

# Collections
users_collection = db["users"]

# User authentication using email
def authenticate_user_by_email(email, password):
	"""
	Authenticate user by email and password.
	Returns user document if authentication is successful, else None.
	"""
	user = users_collection.find_one({
		"email": email,
		"password": password
	})
	return user
