import os
from pymongo import MongoClient
from dotenv import load_dotenv
import pprint

# Load environment variables from .env
load_dotenv()

mongodb_url = os.getenv("MONGO_URI")
if not mongodb_url:
    raise RuntimeError("❌ MONGO_URI not set in environment!")

client = MongoClient(mongodb_url)
db = client["intelliview"]

# List all collections
print("Collections in DB:", db.list_collection_names())

# Explore INTERVIEWS collection
interviews = db["INTERVIEWS"]
print("\nFirst 5 documents in INTERVIEWS:")
for doc in interviews.find().limit(5):
    pprint.pprint(doc)
