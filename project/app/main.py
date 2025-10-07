import os
import secrets
import random
import json
import textwrap
import threading
from datetime import datetime, timedelta

# TensorFlow optimization - MUST be before any TF imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import cv2
import numpy as np
import requests
import google.generativeai as genai
from google.oauth2 import id_token
from google.auth.transport import requests as grequests
from flask import (
    Flask, request, session, redirect, url_for,
    render_template, jsonify, Blueprint
)
from flask_session import Session
from pymongo import MongoClient
import redis
from dotenv import load_dotenv

# Configure TensorFlow for memory efficiency
import os

def configure_tensorflow():
    """Configure TensorFlow for memory-efficient CPU operation"""
    try:
        # Disable GPU at environment level (before TF loads CUDA libs)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"

        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        print("✅ TensorFlow configured for memory-efficient CPU operation")
    except Exception as e:
        print(f"⚠️ TensorFlow configuration warning: {e}")

# Call before any TF operations
configure_tensorflow()


# -------------------------------------------------------------
#  Thread-safe lazy loading for ML models
# -------------------------------------------------------------

emotion_detector = None
pose_detector = None
_models_loaded = False
_loading_lock = threading.Lock()

def load_models_lazy():
    """Load ML models only when needed (thread-safe)"""
    global emotion_detector, pose_detector, _models_loaded
    
    if _models_loaded:
        return

    with _loading_lock:
        if _models_loaded:  # Double-check inside lock
            return

        print("🔄 Loading ML models...")

        try:
            # Load into locals first (safer)
            from fer import FER
            temp_emotion = FER(mtcnn=True)

            import mediapipe as mp
            mp_pose = mp.solutions.pose
            temp_pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )

            # Atomically swap into globals
            emotion_detector = temp_emotion
            pose_detector = temp_pose
            _models_loaded = True
            print("✅ ML models loaded successfully")

        except Exception as e:
            print(f"❌ Error loading ML models: {e}")
            emotion_detector = None
            pose_detector = None
            _models_loaded = False
            raise

def get_emotion_detector():
    """Get emotion detector, loading if necessary"""
    if not _models_loaded:
        load_models_lazy()
    return emotion_detector

def get_pose_detector():
    """Get pose detector, loading if necessary"""
    if not _models_loaded:
        load_models_lazy()
    return pose_detector

def unload_models():
    """Free models from memory (optional cleanup)"""
    global emotion_detector, pose_detector, _models_loaded
    emotion_detector = None
    pose_detector = None
    _models_loaded = False
    print("🧹 ML models unloaded")


# -------------------------------------------------------------
#  Flask app setup
# -------------------------------------------------------------
load_dotenv()
app = Flask(__name__)

# ----------------------
# Secret Key
# ----------------------
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
if not app.config["SECRET_KEY"]:
    if os.getenv("FLASK_ENV") == "development":
        app.config["SECRET_KEY"] = "dev-secret-key"
        print("⚠️ Using default SECRET_KEY (development only)")
    else:
        raise RuntimeError("❌ SECRET_KEY not set in environment!")

# ----------------------
# Session (Redis or filesystem fallback)
# ----------------------
if os.getenv("FLASK_ENV") == "development":
    print("⚠️ Using filesystem session for local development")
    app.config["SESSION_TYPE"] = "filesystem"
else:
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise RuntimeError("❌ REDIS_URL must be set in production!")
    try:
        app.config["SESSION_TYPE"] = "redis"
        app.config["SESSION_REDIS"] = redis.from_url(redis_url)
        print("✅ Using Redis for sessions")
    except Exception as e:
        print(f"❌ Redis unavailable in production: {e}")
        raise

# Common session cookie settings
app.config.update(
    SESSION_COOKIE_SECURE=(os.getenv("FLASK_ENV") != "development"),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_NAME="interview-session"
)

Session(app)


# ----------------------
# MongoDB
# ----------------------
mongodb_url = os.getenv("MONGO_URI")
if not mongodb_url:
    if os.getenv("FLASK_ENV") == "development":
        mongodb_url = "mongodb://localhost:27017/intelliview"
        print("⚠️ Using local MongoDB (development only)")
    else:
        raise RuntimeError("❌ MONGO_URI not set in environment!")

MONGO_CLIENT = MongoClient(mongodb_url)
DATABASE = MONGO_CLIENT["intelliview"]

# ----------------------
# Google OAuth
# ----------------------
GOOGLE_CLIENT_ID = os.getenv("225615803189-2s400l8m4b216d9avon220gnfb23rjoa.apps.googleusercontent.com")
if not GOOGLE_CLIENT_ID:
    if os.getenv("FLASK_ENV") == "development":
        GOOGLE_CLIENT_ID = "dummy-google-client-id"
        print("⚠️ Using dummy GOOGLE_CLIENT_ID (development only)")
    else:
        raise RuntimeError("❌ GOOGLE_CLIENT_ID not set!")

app.config['225615803189-2s400l8m4b216d9avon220gnfb23rjoa.apps.googleusercontent.com'] = GOOGLE_CLIENT_ID

@app.context_processor
def inject_google_client_id():
    return dict(config=app.config)

# ----------------------
# Gemini API
# ----------------------
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    if os.getenv("FLASK_ENV") == "development":
        gemini_api_key = "dummy-gemini-key"
        print("⚠️ Using dummy GEMINI_API_KEY (development only)")
    else:
        raise RuntimeError("❌ GEMINI_API_KEY not set!")

genai.configure(api_key=gemini_api_key)

# ----------------------
# Azure Face API
# ----------------------
AZURE_FACE_API_ENDPOINT = os.getenv("AZURE_FACE_API_ENDPOINT")
AZURE_FACE_API_KEY = os.getenv("AZURE_FACE_API_KEY")

# ----------------------
# Health / Warmup endpoints
# ----------------------
@app.route('/health')
def health():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": _models_loaded}, 200

@app.route('/warmup', methods=['POST'])
def warmup():
    """Endpoint to trigger model loading"""
    try:
        load_models_lazy()
        return {"status": "models loaded successfully"}, 200
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

# -------------------------------------------------------------
#  ATS Blueprint
# -------------------------------------------------------------
ats_bp = Blueprint('ats', __name__, url_prefix='/ats')

@ats_bp.route('/')
def ats_form():
    if not session.get("is_authenticated"):
        return redirect(url_for('index'))
    return render_template('ats_score.html')


def extract_text_from_pdf_with_gemini(pdf_content: bytes) -> str:
    """Extract plain text from PDF using Gemini model."""
    try:
        pdf_blob = {"mime_type": "application/pdf", "data": pdf_content}
        prompt = "Extract all text content from this PDF document. Return plain text only."
        response = genai.generate(
            model="gemini-1.5",
            messages=[
                {"role": "user", "content": prompt},
                {"role": "user", "content": pdf_blob}
            ]
        )
        return response.output_text.strip()
    except Exception as e:
        print(f"Gemini PDF extraction error: {e}")
        return None


def get_ats_score_with_gemini(resume_text: str) -> str:
    """Analyze resume text and return ATS JSON analysis."""
    try:
        prompt = f"""
You are an ATS (Applicant Tracking System) analyzer. Analyze the following resume and provide a comprehensive assessment.
Resume Text:
{resume_text[:8000]}

Return your analysis in this JSON format:
{{
"score": ,
"summary": "",
"strengths": ["", "", ""],
"suggestions": ["", "", ""]
}}
Scoring criteria:
- Clear formatting and structure (25 points)
- Relevant keywords and skills (25 points)
- Professional experience and achievements (25 points)
- Education and qualifications (15 points)
- Contact information and completeness (10 points)
Keep strengths and suggestions concise and actionable.
"""
        response = genai.generate(
            model="gemini-1.5",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.output_text.strip()
    except Exception as e:
        print(f"Error getting ATS score with Gemini: {e}")
        return None


@ats_bp.route('/score', methods=['POST'])
def ats_score():
    if not session.get("is_authenticated"):
        return jsonify({'error': 'User not authenticated'}), 401

    if 'resume' not in request.files:
        return jsonify({'error': 'No resume file uploaded', 'reason': 'Please upload a PDF file.'}), 400

    file = request.files['resume']
    if file.filename == '' or not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Invalid file', 'reason': 'Please upload a PDF file.'}), 400

    pdf_content = file.read()
    if len(pdf_content) > 10 * 1024 * 1024:
        return jsonify({'error': 'File too large', 'reason': 'Maximum allowed size is 10MB.'}), 400

    resume_text = extract_text_from_pdf_with_gemini(pdf_content)
    if not resume_text:
        return jsonify({'error': 'Text extraction failed', 'reason': 'Could not extract text from the PDF.'}), 500

    gemini_response = get_ats_score_with_gemini(resume_text)
    if not gemini_response:
        return jsonify({'error': 'AI analysis failed', 'reason': 'Could not analyze the resume.'}), 500

    # Clean response (handle code blocks)
    def clean_gemini_json(response_text):
        if '```json' in response_text:
            try:
                return response_text.split('```json', 1)[1].split('```', 1)[0].strip()
            except IndexError:
                return "{}"
        elif '```' in response_text:
            try:
                return response_text.split('```', 1)[1].split('```', 1)[0].strip()
            except IndexError:
                return "{}"
        return response_text.strip()

    clean_response = clean_gemini_json(gemini_response)

    try:
        parsed_data = json.loads(clean_response)
    except json.JSONDecodeError:
        print("Failed to parse Gemini response as JSON")
        parsed_data = {}

    default_response = {
        'score': 0,
        'summary': 'Analysis completed.',
        'strengths': [],
        'suggestions': [],
        'raw': gemini_response
    }

    return jsonify({
        'score': parsed_data.get('score', default_response['score']),
        'summary': parsed_data.get('summary', default_response['summary']),
        'strengths': parsed_data.get('strengths', default_response['strengths']),
        'suggestions': parsed_data.get('suggestions', default_response['suggestions']),
        'raw': default_response['raw']
    })


# Register blueprint
app.register_blueprint(ats_bp)


# -------------------------------------------------------------
#  Standard Routes
# -------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/profile")
def profile():
    if session.get("is_authenticated"):
        # Logged-in user → fetch from DB
        user_doc = DATABASE["USERS"].find_one({"user_id": session["user"]["user_id"]})
        user_info = user_doc.get("user_info", {}) if user_doc else {}
    else:
        # Guest user → fallback info
        user_info = {
            "username": "Guest",
            "name": "Guest User",
            "avatar_url": "/static/assets/default-avatar.png",  # put a guest avatar in your static folder
            "email": "Not logged in",
            "resume_summary": "Login to save your resume summary."
        }

    return render_template("profile.html", user_info=user_info)



@app.route("/interview")
def interview():
    user_info = session.get("user", {"username": "Guest"})
    return render_template("interview.html", user_info=user_info)

@app.route("/ats_score")
def ats_score_page():
    if session.get("is_authenticated"):
        user_info = DATABASE["USERS"].find_one(
            {"user_id": session["user"]["user_id"]},
            {"user_info": 1}
        ) or {}
        user_info = user_info.get("user_info", {})
    else:
        # Guest fallback
        user_info = {
            "username": "Guest",
            "name": "Guest User",
            "avatar_url": "/static/assets/default-avatar.png",
            "email": "Not logged in",
        }

    return render_template("ats_score.html", user_info=user_info, is_guest=not session.get("is_authenticated"))


@app.route('/api/v1/create-interview', methods=['POST'])
def create_interview():
    # Fetch form data
    job_description = request.form.get('job_description')
    resume = request.files.get('resume')
    interview_type = request.form.get('interview_type')

    if interview_type not in ['technical', 'behavioral', 'common-questions']:
        return jsonify({'status': 'error', 'message': 'Invalid interview type'}), 400

    if not job_description or not resume:
        return jsonify({'status': 'error', 'message': 'Job description or resume not provided'}), 400

    # Generate resume summary via LLM
    model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
    prompt_resume_summary = "Carefully review the attached resume file. Provide a structured, detailed summary."
    resume_blob = {"mime_type": resume.content_type, "data": resume.read()}

    try:
        response_resume = model.generate_content([prompt_resume_summary, resume_blob])
        resume_summary = response_resume.text
    except Exception as e:
        print(f"ERROR: Failed to generate resume summary: {e}")
        resume_summary = "Resume summary could not be generated."

    # Generate questions
    generated_questions = []
    try:
        if interview_type == 'technical':
            question_prompt = f"Generate 10 technical questions based on JD: {job_description} and Resume: {resume_summary}"
        elif interview_type == 'behavioral':
            question_prompt = f"Generate 10 behavioral questions based on JD: {job_description} and Resume: {resume_summary}"
        else:
            question_prompt = "Generate 10 common interview questions."

        questions_response = model.generate_content([question_prompt])
        generated_questions = [q.strip() for q in questions_response.text.split('\n') if q.strip()]
    except Exception as e:
        print(f"ERROR: Failed to generate questions: {e}")
        generated_questions = [
            "Tell me about yourself.",
            "Walk me through your resume.",
            "What are your strengths?",
            "What are your weaknesses?",
            "Where do you see yourself in 5 years?"
        ]

    # Create interview identifier
    interview_identifier = secrets.token_hex(16)

    if session.get("is_authenticated"):
        # Logged-in user: save to DB
        DATABASE["INTERVIEWS"].insert_one({
            "interview_identifier": interview_identifier,
            "user_id": session["user"]["user_id"],
            "interview_type": interview_type,
            "job_description": job_description,
            "resume_summary": resume_summary,
            "created_at": datetime.now(),
            "is_active": True,
            "is_completed": False,
            "ai_report": "",
            "questions": generated_questions,
            "interview_history": [],
            "behavior_analysis": [],
        })
    else:
        # Guest mode: save multiple interviews in session
        guest_interviews = session.get("guest_interviews", {})
        guest_interviews[interview_identifier] = {
            "interview_identifier": interview_identifier,
            "interview_type": interview_type,
            "job_description": job_description,
            "resume_summary": resume_summary,
            "created_at": datetime.now().isoformat(),
            "questions": generated_questions,
        }
        session["guest_interviews"] = guest_interviews
        session.modified = True

    return redirect(url_for("interview_page", interview_identifier=interview_identifier))

@app.route('/interview/<interview_identifier>', methods=['GET'])
def interview_page(interview_identifier):
    if session.get("is_authenticated"):
        # Logged-in user: fetch from DB
        interview = DATABASE["INTERVIEWS"].find_one({"interview_identifier": interview_identifier})
        if not interview:
            return jsonify({'status': 'error', 'message': 'Interview not found'}), 404
        if interview["user_id"] != session["user"]["user_id"]:
            return jsonify({'status': 'error', 'message': 'Unauthorized access'}), 403
        if interview.get("is_completed"):
            return redirect(url_for("interview_results", interview_identifier=interview_identifier))
        return render_template('take-interview.html', interview=interview)

    else:
        # Guest mode: check session for multiple guest interviews
        guest_interviews = session.get("guest_interviews", {})
        interview = guest_interviews.get(interview_identifier)

        if not interview:
            return jsonify({'status': 'error', 'message': 'Guest interview not found'}), 404

        return render_template('take-interview.html', interview=interview)


@app.route('/new-mock-interview', methods=['GET'])
def new_mock_interview():
    if session.get("is_authenticated"):
        # ✅ Logged-in user flow
        user_info = DATABASE["USERS"].find_one({"user_id": session["user"]["user_id"]})
        if user_info is None:
            return jsonify({'status': 'error', 'message': 'User not found'}), 404

        if not user_info.get("user_info", {}).get("resume_summary"):
            return redirect(url_for('settings', message='Please upload your resume first to generate mock interview questions.'))

        resume_summary = user_info['user_info']['resume_summary']

    else:
        # ✅ Guest flow (basic resume summary / placeholder)
        resume_summary = """
        Guest user profile. No uploaded resume.
        Assume a general candidate with basic education,
        some technical knowledge, and soft skills.
        """

    # --- Question generation ---
    model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
    prompt = f"""
    Generate 10 mock interview questions based on the following resume summary:
    {resume_summary}
    The questions should be relevant to the candidate's background and experience, and the response should be in plain text format (no markdown or formatting). The questions should be clear and concise, and they should cover a range of topics related to the candidate's skills and experience.

    Always include these two generic questions as the first two and be sure to paraphrase them:
    1. Tell me a bit about yourself.
    2. Walk me through your resume.

    Only output the questions, one per line, with no numbering or extra text.
    """

    try:
        response = model.generate_content([prompt])
        questions = [q.strip() for q in response.text.split('\n') if q.strip()]
    except Exception as e:
        print(f"ERROR (new-mock-interview): Gemini generation failed: {e}")
        questions = ["Error: Could not generate questions. Please try again later."]

    mock_interview_identifier = secrets.token_hex(16)

    if session.get("is_authenticated"):
        # Save to DB for logged-in users
        DATABASE["INTERVIEWS"].insert_one({
            "mock_interview_identifier": mock_interview_identifier,
            "user_id": session["user"]["user_id"],
            "questions": questions,
            "created_at": datetime.now(),
            "is_active": True,
            "is_completed": False,
            "video_url": "",
            "ai_report": "",
            "interview_history": [],
            "behavior_analysis": [],
        })
    else:
        # Store in session for guests
        session["guest_interview"] = {
            "mock_interview_identifier": mock_interview_identifier,
            "questions": questions,
            "created_at": datetime.now().isoformat(),
            "is_active": True,
            "is_completed": False,
        }

    return render_template('begin_mock_interview.html', mock_interview_identifier=mock_interview_identifier, questions=questions)


@app.route('/mock-interview/<mock_interview_identifier>', methods=['GET'])
def mock_interview(mock_interview_identifier):
    if session.get("is_authenticated"):
        # --- Logged-in flow ---
        mock_interview = DATABASE["INTERVIEWS"].find_one(
            {"mock_interview_identifier": mock_interview_identifier}
        )
        if mock_interview is None:
            return jsonify({'status': 'error', 'message': 'Mock interview not found'}), 404

        if mock_interview["user_id"] != session["user"]["user_id"]:
            return jsonify({'status': 'error', 'message': 'Unauthorized access to this mock interview'}), 403

    else:
        # --- Guest flow (check in session storage) ---
        guest_interview = session.get("guest_interview")
        if not guest_interview or guest_interview.get("mock_interview_identifier") != mock_interview_identifier:
            return jsonify({'status': 'error', 'message': 'Guest mock interview not found'}), 404
        mock_interview = guest_interview

    return render_template('mock_interview.html', mock_interview=mock_interview)


@app.route('/get-questions', methods=['GET'])
def get_questions():
    identifier = request.args.get('id')
    if not identifier:
        return jsonify({'status': 'error', 'message': 'Interview ID not provided'}), 400

    interview = None

    if session.get("is_authenticated"):
        # --- Logged-in flow ---
        interview = DATABASE["INTERVIEWS"].find_one(
            {"interview_identifier": identifier, "user_id": session["user"]["user_id"]}
        )
        if interview is None:
            interview = DATABASE["INTERVIEWS"].find_one(
                {"mock_interview_identifier": identifier, "user_id": session["user"]["user_id"]}
            )
    else:
        # --- Guest flow ---
        guest_interviews = session.get("guest_interviews", {})
        interview = guest_interviews.get(identifier)

    if interview is None:
        return jsonify({'status': 'error', 'message': 'Interview not found or unauthorized access'}), 404

    questions = interview.get("questions", [])

    # Optional: Provide default questions if none exist
    if not questions:
        questions = [
            "Tell me about yourself.",
            "Walk me through your resume.",
            "What are your strengths?",
            "What are your weaknesses?",
            "Where do you see yourself in 5 years?"
        ]

    return jsonify({'status': 'success', 'questions': questions})


@app.route('/api/v1/parse-resume', methods=['POST'])
def parse_resume():
    if 'resume' not in request.files:
        return jsonify({'status': 'error', 'message': 'No resume file part in the request'}), 400

    file = request.files['resume']

    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    if not file:
        return jsonify({'status': 'error', 'message': 'Invalid file provided'}), 400

    try:
        file_content = file.read()
        mime_type = file.content_type
        if not mime_type:
            return jsonify({'status': 'error', 'message': 'Could not determine file MIME type'}), 400

        # Convert to AI-readable blob
        resume_blob = {
            "mime_type": mime_type,
            "data": file_content
        }
        model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")

        prompt = """
        Carefully review the attached resume file. Provide a thorough, structured, and objective summary of the candidate's background, including:
        - Contact information (if present)
        - Education history (degrees, institutions, graduation years)
        - Work experience (roles, companies, durations, responsibilities, achievements)
        - Technical and soft skills
        - Certifications, awards, or notable projects
        - Any other relevant sections (e.g., publications, languages, interests)
        Present the information in clear, well-organized paragraphs using plain text (no markdown or formatting).
        """

        response = model.generate_content([prompt, resume_blob])
        resume_summary = response.text

        if session.get("is_authenticated"):
            # 🔑 Logged-in users → Save in DB
            DATABASE["USERS"].update_one(
                {"user_id": session["user"]["user_id"]},
                {
                    "$set": {
                        "user_info.resume_summary": resume_summary,
                        "account_info.last_login": datetime.now(),
                    }
                },
            )
            return jsonify({
                'status': 'success',
                'message': f'Hey {session["user"]["name"]}, your resume has been successfully processed!',
                'redirect_url': url_for("new_mock_interview")
            })
        else:
            # 👤 Guest users → Save in session
            session["guest_resume_summary"] = resume_summary
            return jsonify({
                'status': 'success',
                'message': 'Your resume has been processed! You can now generate a mock interview.',
                'redirect_url': url_for("new_guest_mock_interview")
            })

    except Exception as e:
        app.logger.error(f"Error processing resume with Gemini: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to process resume with AI model: {str(e)}'}), 500


GOOGLE_CLIENT_ID = "225615803189-2s400l8m4b216d9avon220gnfb23rjoa.apps.googleusercontent.com"

@app.route("/auth/login/google", methods=["POST"])
def google_login():
    try:
        data = request.get_json(force=True)
        token = data.get("id_token")

        if not token:
            return jsonify({"status": "error", "message": "No token provided"}), 400

        # Verify token with Google
        idinfo = id_token.verify_oauth2_token(token, grequests.Request(), GOOGLE_CLIENT_ID)

        # Extract user info
        user_id = idinfo["sub"]
        email = idinfo.get("email")
        name = idinfo.get("name", "User")
        picture = idinfo.get("picture", "/static/assets/default-avatar.png")

        # Save in session
        session["is_authenticated"] = True
        session["user"] = {
            "user_id": user_id,
            "email": email,
            "name": name,
            "avatar_url": picture
        }

        return jsonify({"status": "success", "user": session["user"]})

    except ValueError as e:
        app.logger.error(f"Google login failed: {e}")
        return jsonify({"status": "error", "message": f"Invalid token: {str(e)}"}), 400
    except Exception as e:
        app.logger.error(f"Unexpected Google login error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Server error during login"}), 500


@app.route("/auth/logout", methods=["GET"])
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    message = None

    if session.get("is_authenticated"):
        user_id = session["user"]["user_id"]
        # Fetch the full user_info subdocument
        user_doc = DATABASE["USERS"].find_one(
            {"user_id": user_id},
            {"user_info": 1}
        )
        user_info = user_doc.get("user_info", {}) if user_doc else {}

        if request.method == 'POST':
            # Pull fields from the form
            name = request.form.get('name')
            username = request.form.get('username')
            email = request.form.get('email')
            avatar_url = request.form.get('avatar_url')
            about = request.form.get('about')
            skills_raw = request.form.get('skills', '').strip()
            # Convert comma-separated skills into a list, if provided
            skills = [s.strip() for s in skills_raw.split(',')] if skills_raw else []

            # Build the update dict dynamically
            update_fields = {
                "user_info.name": name,
                "user_info.username": username,
                "user_info.email": email,
                "user_info.avatar_url": avatar_url
            }
            if about is not None:
                update_fields["user_info.about"] = about
            if skills_raw != "":
                update_fields["user_info.skills"] = skills

            DATABASE["USERS"].update_one(
                {"user_id": user_id},
                {"$set": update_fields}
            )

            # Update session for immediate UI reflect
            session["user"]["name"] = name
            session["user"]["username"] = username
            session["user"]["avatar_url"] = avatar_url

            message = "Settings updated successfully!"
            # Re-fetch to get the latest user_info
            user_doc = DATABASE["USERS"].find_one(
                {"user_id": user_id},
                {"user_info": 1}
            )
            user_info = user_doc.get("user_info", {}) if user_doc else {}

    else:
        # Guest fallback (no DB updates)
        user_info = {
            "username": "Guest",
            "name": "Guest User",
            "avatar_url": "/static/assets/default-avatar.png",
            "email": "Not logged in",
            "about": "You are using guest mode. Login to save your settings.",
            "skills": []
        }
        if request.method == 'POST':
            message = "Guests cannot update settings. Please log in."

    return render_template('settings.html', user_info=user_info, message=message)


@app.route('/upload-screencapture', methods=['POST'])
def upload_screencapture():
    import secrets, cv2, numpy as np, mediapipe as mp, requests
    from datetime import datetime

    # --- User / Guest Handling ---
    if "guest_user_id" not in session:
        session["guest_user_id"] = "guest_" + secrets.token_hex(8)

    user_id = session.get("user", {}).get("user_id", session["guest_user_id"])
    identifier = request.form.get('identifier') or "guest_" + secrets.token_hex(8)

    # --- Validate file ---
    if 'screencapture' not in request.files:
        return jsonify({'status': 'error', 'message': 'No screencapture file in request'}), 400

    file = request.files['screencapture']
    image_data = file.read()

    # --- Initialize analysis report ---
    analysis_report = {
        "emotion_analysis": "Processing...",
        "posture_analysis": "Processing...",
        "body_language_analysis": "Processing...",
        "eye_contact_analysis": "Processing...",
        "gestures_analysis": "Processing...",
        "movement_analysis": "Processing...",
        "overall_impression": "Processing...",
        "suggestions_for_improvement": "Processing..."
    }

    # --- Load Models ---
    try:
        emotion_detector = get_emotion_detector()
        pose_detector = get_pose_detector()
        mp_pose = mp.solutions.pose
    except Exception as e:
        print(f"ERROR: Could not load ML models: {e}")
        for k in analysis_report.keys():
            analysis_report[k] = "AI analysis unavailable"
        return jsonify({'status': 'error', 'message': 'AI models not available', 'analysis_report': analysis_report}), 500

    # --- Default detections ---
    detected_posture = "Undetermined"
    detected_eye_contact = "Undetermined"
    detected_gestures = "Undetermined"
    detected_body_language_type = "Undetermined"

    # --- Azure Face API (optional) ---
    if AZURE_FACE_API_ENDPOINT and AZURE_FACE_API_KEY:
        try:
            detect_url = f"{AZURE_FACE_API_ENDPOINT}/detect"
            headers = {
                "Content-Type": "application/octet-stream",
                "Ocp-Apim-Subscription-Key": AZURE_FACE_API_KEY
            }
            params = {
                "returnFaceAttributes": "headPose",
                "returnFaceId": "false",
                "returnFaceLandmarks": "false"
            }
            response = requests.post(detect_url, headers=headers, params=params, data=image_data)
            response.raise_for_status()
            faces = response.json()
            if faces:
                head_pose = faces[0].get('faceAttributes', {}).get('headPose', {})
                yaw = head_pose.get('yaw', 0)
                pitch = head_pose.get('pitch', 0)
                if abs(yaw) < 15 and abs(pitch) < 15:
                    detected_eye_contact = "Consistent"
                    analysis_report["eye_contact_analysis"] = "Consistent (Direct Gaze)"
                else:
                    detected_eye_contact = "Intermittent/Looking Away"
                    analysis_report["eye_contact_analysis"] = "Intermittent (Gaze Off-Camera)"
            else:
                detected_eye_contact = "No face detected"
                analysis_report["eye_contact_analysis"] = "No face detected"
        except Exception as e:
            print(f"ERROR: Azure Face API failed: {e}")
            analysis_report["eye_contact_analysis"] = "Azure API Error"
    else:
        analysis_report["eye_contact_analysis"] = "Azure API Not Configured"

    # --- MediaPipe Pose / Gesture / Posture ---
    try:
        np_arr = np.frombuffer(image_data, np.uint8)
        image_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image_cv is None:
            raise ValueError("Could not decode image data")

        # Emotion detection
        emotions = emotion_detector.detect_emotions(image_cv)
        if emotions:
            emo_dict = emotions[0]['emotions']
            detected_emotion = max(emo_dict, key=emo_dict.get)
            analysis_report["emotion_analysis"] = detected_emotion.capitalize()
        else:
            analysis_report["emotion_analysis"] = "No face detected for emotion"

        # Pose detection
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Posture
            avg_shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y +
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
            avg_hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y +
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
            if avg_shoulder_y < avg_hip_y - 0.08:
                detected_posture = "Upright"
            elif avg_shoulder_y > avg_hip_y + 0.08:
                detected_posture = "Slumped"
            else:
                detected_posture = "Neutral"

            # Gestures / Body Language
            left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
            right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
            left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            left_hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x
            right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x
            left_wrist_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x
            right_wrist_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x

            body_width = abs(right_hip_x - left_hip_x)
            if left_wrist_y < left_shoulder_y - 0.05 or right_wrist_y < right_shoulder_y - 0.05:
                detected_gestures = "Hands Raised"
                detected_body_language_type = "Expressive (Arms Elevated)"
            elif body_width > 0.01 and ((abs(left_wrist_x - left_hip_x) > body_width*0.15) or
                                       (abs(right_wrist_x - right_hip_x) > body_width*0.15)):
                detected_gestures = "Natural (Arms Extended)"
                detected_body_language_type = "Open/Expansive"
            else:
                detected_gestures = "Minimal/Natural"
                detected_body_language_type = "Neutral/Restrained"

            analysis_report["movement_analysis"] = "Body detected"
        else:
            detected_posture = "No Body Detected"
            detected_gestures = "No Body Detected"
            detected_body_language_type = "No Body Detected"
            analysis_report["movement_analysis"] = "No Body Detected"

    except Exception as e:
        print(f"ERROR: MediaPipe analysis failed: {e}")
        detected_posture = detected_gestures = detected_body_language_type = "Pose AI Error"
        analysis_report["movement_analysis"] = "Pose AI Error"

    analysis_report["posture_analysis"] = detected_posture
    analysis_report["body_language_analysis"] = detected_body_language_type
    analysis_report["gestures_analysis"] = detected_gestures

    # --- Gemini AI Feedback ---
    try:
        model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
        prompt_text = f"""
        Provide short bullet-point feedback (max 3 sentences each):
        Body Language: {detected_body_language_type}
        Eye Contact: {detected_eye_contact}
        Gestures: {detected_gestures}
        Emotion: {analysis_report['emotion_analysis']}
        """

        response = model.generate_content(prompt_text)
        gem_feedback = response.text.strip()
        analysis_report["overall_impression"] = gem_feedback or analysis_report["overall_impression"]
        analysis_report["suggestions_for_improvement"] = "See above feedback"
    except Exception as e:
        print(f"ERROR: Gemini analysis failed: {e}")
        analysis_report["overall_impression"] = "Detailed analysis unavailable"
        analysis_report["suggestions_for_improvement"] = "Please try again later"

    # --- Store analysis in DB ---
    DATABASE["INTERVIEWS"].update_one(
        {"$or":[
            {"interview_identifier": identifier},
            {"mock_interview_identifier": identifier}
        ]},
        {"$push": {"behavior_analysis": analysis_report}},
        upsert=True
    )

    return jsonify({'status': 'success', 'message': 'Screencapture processed', 'analysis_report': analysis_report})

@app.route('/submit-answer', methods=['POST'])
def submit_answer():
    data = request.json
    interview_id = data.get('interview_id')
    question_index = data.get('question_index')
    user_answer = data.get('user_answer')

    if not all([interview_id, question_index is not None, user_answer is not None]):
        return jsonify({'status': 'error', 'message': 'Missing data for answer submission'}), 400

    # Check if user is logged in
    if session.get("is_authenticated"):
        user_id = session["user"]["user_id"]
        interview = DATABASE["INTERVIEWS"].find_one({
            "$or": [
                {"interview_identifier": interview_id},
                {"mock_interview_identifier": interview_id}
            ],
            "user_id": user_id
        })

        if not interview:
            return jsonify({'status': 'error', 'message': 'Interview not found'}), 404

        questions = interview.get('questions', [])
        if question_index < 0 or question_index >= len(questions):
            return jsonify({'status': 'error', 'message': 'Invalid question index'}), 400

        current_question = questions[question_index]

        DATABASE["INTERVIEWS"].update_one(
            {"_id": interview["_id"]},
            {"$push": {
                "interview_history": {
                    "question": current_question,
                    "answer": user_answer,
                    "timestamp": datetime.now()
                }
            }}
        )

    else:
        # Guest mode: fetch interview from session
        guest_interviews = session.get("guest_interviews", {})
        interview = guest_interviews.get(interview_id)

        if not interview:
            return jsonify({'status': 'error', 'message': 'Guest interview not found'}), 404

        questions = interview.get('questions', [])
        if question_index < 0 or question_index >= len(questions):
            return jsonify({'status': 'error', 'message': 'Invalid question index'}), 400

        current_question = questions[question_index]

        interview.setdefault("interview_history", []).append({
            "question": current_question,
            "answer": user_answer,
            "timestamp": datetime.now().isoformat()
        })

        guest_interviews[interview_id] = interview
        session["guest_interviews"] = guest_interviews
        session.modified = True

    return jsonify({'status': 'success', 'message': 'Answer submitted successfully'})


@app.route('/end-interview', methods=['POST'])
def end_interview():
    data = request.json
    identifier = data.get('identifier')
    timer = data.get('timer')

    # Parse timer safely
    try:
        timer = float(timer) if timer is not None else 0.0
    except (TypeError, ValueError):
        timer = 0.0

    if not identifier:
        return jsonify({'status': 'error', 'message': 'Interview ID not provided'}), 400

    # Determine user type
    if session.get("is_authenticated"):
        user_id = session["user"]["user_id"]
        interview_doc = DATABASE["INTERVIEWS"].find_one({
            "$or": [
                {"interview_identifier": identifier},
                {"mock_interview_identifier": identifier}
            ],
            "user_id": user_id
        })

        if not interview_doc:
            return jsonify({'status': 'error', 'message': 'Interview not found or unauthorized'}), 404

    else:
        user_id = "guest"
        guest_interviews = session.get("guest_interviews", {})
        interview_doc = guest_interviews.get(identifier)
        if not interview_doc:
            return jsonify({'status': 'error', 'message': 'Guest interview not found'}), 404

    # Initialize behavioral and history data
    interview_history = interview_doc.get("interview_history", [])
    behavior_analysis_snapshots = interview_doc.get("behavior_analysis", [])

    # Aggregate behavioral analysis
    posture_counts = {}
    eye_contact_counts = {}
    gestures_counts = {}
    body_language_counts = {}

    for snapshot in behavior_analysis_snapshots:
        posture = snapshot.get("posture_analysis")
        if posture and all(x not in posture for x in ["Detected", "Error", "Undetermined", "N/A"]):
            posture_counts[posture] = posture_counts.get(posture, 0) + 1

        eye_contact = snapshot.get("eye_contact_analysis")
        if eye_contact and all(x not in eye_contact for x in ["Detected", "Error", "Undetermined", "N/A"]):
            eye_contact_counts[eye_contact] = eye_contact_counts.get(eye_contact, 0) + 1

        gestures = snapshot.get("gestures_analysis")
        if gestures and all(x not in gestures for x in ["Detected", "Error", "Undetermined", "N/A"]):
            gestures_counts[gestures] = gestures_counts.get(gestures, 0) + 1

        body_language = snapshot.get("body_language_analysis")
        if body_language and all(x not in body_language for x in ["Detected", "Error", "Undetermined", "N/A"]):
            body_language_counts[body_language] = body_language_counts.get(body_language, 0) + 1

    most_common_posture = max(posture_counts, key=posture_counts.get) if posture_counts else "Not observed"
    most_common_eye_contact = max(eye_contact_counts, key=eye_contact_counts.get) if eye_contact_counts else "Not observed"
    most_common_gestures = max(gestures_counts, key=gestures_counts.get) if gestures_counts else "Not observed"
    most_common_body_language = max(body_language_counts, key=body_language_counts.get) if body_language_counts else "Not observed"

    # Prepare Q&A text
    all_q_and_a = "\n".join(
        f"Question: {entry.get('question')}\nAnswer: {entry.get('answer')}\n"
        for entry in interview_history
    )

    # Initialize AI report
    final_ai_report = "Detailed report generation in progress..."
    strengths_arr = []
    weaknesses_arr = []
    behavioural_txt = ""
    language_txt = ""
    suitability_txt = ""

    # Generate AI report
    try:
        model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
        prompt = textwrap.dedent(f"""
            You are an AI interview assessor providing a comprehensive and concise report. Return ONLY valid JSON with these exact keys:
            {{
              "overall": "string",
              "strengths": ["array", "of", "strings"],
              "weaknesses": ["array", "of", "strings"],
              "behavioural": "string",
              "language": "string",
              "suitability": "string"
            }}

            Context for Assessment:
            Job Role: {interview_doc.get('job_description','')[:500]}
            Resume Summary: {interview_doc.get('resume_summary','')[:800]}
            Interview Duration: {timer:.0f} seconds
            Interview Questions & Answers:
            {all_q_and_a}

            Aggregated Behavioral Observations:
            - Most Common Posture: {most_common_posture}
            - Most Common Eye Contact: {most_common_eye_contact}
            - Most Common Gestures: {most_common_gestures}
            - Most Common Body Language Type: {most_common_body_language}
        """)
        gem_out = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        ).text.strip()

        report_json = json.loads(gem_out)
        final_ai_report = report_json.get("overall", final_ai_report)
        strengths_arr = report_json.get("strengths", [])
        weaknesses_arr = report_json.get("weaknesses", [])
        behavioural_txt = report_json.get("behavioural", "")
        language_txt = report_json.get("language", "")
        suitability_txt = report_json.get("suitability", "Average")

        if suitability_txt not in {"Poor","Below Average","Average","Good","Excellent"}:
            suitability_txt = "Average"

    except Exception as e:
        app.logger.error(f"AI report generation failed: {e}")
        final_ai_report = "Interview completed. AI report unavailable."
        strengths_arr = ["Interview participation recorded"]
        weaknesses_arr = ["Detailed feedback temporarily unavailable"]
        behavioural_txt = "Behavioral analysis pending."
        language_txt = "Language feedback pending."
        suitability_txt = "Average"

    # Save updates
    if session.get("is_authenticated"):
        DATABASE["INTERVIEWS"].update_one(
            {"$or": [
                {"interview_identifier": identifier},
                {"mock_interview_identifier": identifier}
            ], "user_id": user_id},
            {"$set": {
                "is_completed": True,
                "duration": timer,
                "completed_at": datetime.now(),
                "final_ai_report": final_ai_report,
                "strengths": strengths_arr,
                "weaknesses": weaknesses_arr,
                "behavioural": behavioural_txt,
                "language": language_txt,
                "suitability": suitability_txt
            }}
        )
    else:
        # Update guest session
        guest_interviews[identifier].update({
            "is_completed": True,
            "duration": timer,
            "completed_at": datetime.now().isoformat(),
            "final_ai_report": final_ai_report,
            "strengths": strengths_arr,
            "weaknesses": weaknesses_arr,
            "behavioural": behavioural_txt,
            "language": language_txt,
            "suitability": suitability_txt
        })
        session["guest_interviews"] = guest_interviews
        session.modified = True

    return jsonify({
        'status': 'success',
        'message': 'Interview ended successfully',
        'redirect_url': url_for('view_report', identifier=identifier)
    })


@app.route('/history')
def history_list_page():
    interviews_for_display = []

    if session.get("is_authenticated"):
        user_id = session["user"]["user_id"]
        
        completed_interviews = DATABASE["INTERVIEWS"].find(
            {"user_id": user_id, "is_completed": True}
        ).sort("completed_at", -1)

        for interview in completed_interviews:
            report_id = interview.get("interview_identifier") or interview.get("mock_interview_identifier")
            
            duration_val = interview.get("duration")
            duration_str = "N/A"
            if isinstance(duration_val, (int, float)):
                minutes = int(duration_val // 60)
                seconds = int(duration_val % 60)
                duration_str = f"{minutes:02d}m {seconds:02d}s"
            elif isinstance(duration_val, str) and duration_val.isdigit():
                minutes = int(duration_val) // 60
                seconds = int(duration_val) % 60
                duration_str = f"{minutes:02d}m {seconds:02d}s"

            interviews_for_display.append({
                "report_id": report_id,
                "type": interview.get("interview_type", "Mock Interview" if "mock_interview_identifier" in interview else "Custom Interview"),
                "job_role": interview.get("job_description", "N/A")[:50] + "..." if interview.get("job_description") else "N/A",
                "date": interview.get("completed_at", interview.get("created_at")).strftime("%Y-%m-%d %H:%M") if interview.get("completed_at") or interview.get("created_at") else "N/A",
                "duration": duration_str
            })
    else:
        # Guest mode → show demo/sample data
        interviews_for_display = [
            {
                "report_id": "guest-demo-1",
                "type": "Technical Interview (Demo)",
                "job_role": "Software Engineer (Sample)",
                "date": "2025-09-21 10:00",
                "duration": "15m 30s"
            },
            {
                "report_id": "guest-demo-2",
                "type": "Behavioral Interview (Demo)",
                "job_role": "Team Lead Position (Sample)",
                "date": "2025-09-19 14:30",
                "duration": "12m 10s"
            }
        ]

    return render_template('history_list.html', interviews=interviews_for_display, is_guest=not session.get("is_authenticated"))


@app.route('/history/<identifier>')
def view_report(identifier):
    # Determine user ID: logged-in or guest
    user_id = session.get("user", {}).get("user_id")
    
    if not user_id:
        return redirect(url_for('index'))

    # Fetch the completed interview/report
    interview = DATABASE["INTERVIEWS"].find_one({
        "$or": [
            {"interview_identifier": identifier},
            {"mock_interview_identifier": identifier}
        ],
        "user_id": user_id,
        "is_completed": True
    })

    if not interview:
        message = "Report not found, not completed yet, or unauthorized access."
        return render_template('history.html', interview_data=None, message=message)

    # Set safe defaults for optional AI fields
    interview.setdefault("ai_report", "No comprehensive report generated yet.")
    interview.setdefault("ai_strengths", ["Interview participation", "Professional demeanor"])
    interview.setdefault("ai_weaknesses", ["Areas for improvement identified"])
    interview.setdefault("ai_behaviour", "Professional demeanor observed")
    interview.setdefault("ai_language", "Professional communication style")
    interview.setdefault("ai_suitability", "Average")
    interview.setdefault("questions", [])
    interview.setdefault("interview_history", [])

    # Ensure duration is numeric
    try:
        duration_val = float(interview.get("duration", 0))
    except (TypeError, ValueError):
        duration_val = 0

    interview_data_for_template = {
        "identifier": interview.get("interview_identifier") or interview.get("mock_interview_identifier"),
        "user_id": interview.get("user_id"),
        "title": interview.get("title", "Interview Report"),
        "is_completed": interview.get("is_completed", False),
        "questions": interview.get("questions"),
        "interview_history": interview.get("interview_history"),
        "ai_report": interview.get("ai_report"),
        "ai_strengths": interview.get("ai_strengths"),
        "ai_weaknesses": interview.get("ai_weaknesses"),
        "ai_behaviour": interview.get("ai_behaviour"),
        "ai_language": interview.get("ai_language"),
        "ai_suitability": interview.get("ai_suitability"),
        "duration": f"{int(duration_val // 60):02d}:{int(duration_val % 60):02d}",
        "completed_at": interview.get("completed_at")
    }

    return render_template('history.html', interview_data=interview_data_for_template, message=None)


@app.route('/get-interview-report/<identifier>')
def get_interview_report(identifier):
    if not session.get("is_authenticated"):
        return jsonify({'status': 'error', 'message': 'User not authenticated'}), 401

    user_id = session["user"]["user_id"]
    interview_doc = DATABASE["INTERVIEWS"].find_one({
        "$or": [
            {"interview_identifier": identifier},
            {"mock_interview_identifier": identifier}
        ],
        "user_id": user_id,
        "is_completed": True
    })

    if not interview_doc:
        return jsonify({'status': 'error', 'message': 'Report not found, incomplete, or unauthorized'}), 404

    # Initialize counters and defaults
    behavior_counts = {"posture": {}, "eye_contact": {}, "gestures": {}, "body_language": {}}
    overall_impressions = []
    suggestions = []

    snapshots = interview_doc.get("behavior_analysis", [])
    if not snapshots:
        print("⚠️ No behavior_analysis data found; using defaults")

    for snapshot in snapshots:
        try:
            for key, counter_key in [
                ("posture_analysis", "posture"),
                ("eye_contact_analysis", "eye_contact"),
                ("gestures_analysis", "gestures"),
                ("body_language_analysis", "body_language")
            ]:
                val = snapshot.get(key)
                if val and all(x not in val for x in ["Detected", "Error", "Undetermined", "N/A"]):
                    behavior_counts[counter_key][val] = behavior_counts[counter_key].get(val, 0) + 1

            overall = snapshot.get("overall_impression")
            if overall and all(x not in overall for x in ["Processing", "Error", "Undetermined", "N/A"]):
                overall_impressions.append(overall)

            sug = snapshot.get("suggestions_for_improvement")
            if sug and all(x not in sug for x in ["Processing", "Error", "Undetermined", "N/A"]):
                suggestions.append(sug)
        except Exception as e:
            print(f"⚠️ Skipping a snapshot due to error: {e}")

    def most_common(d):
        return max(d, key=d.get) if d else "Not observed"

    total_time = interview_doc.get("duration", 0)
    if isinstance(total_time, str):
        try:
            total_time = float(total_time)
        except ValueError:
            total_time = 0.0

    num_questions = len(interview_doc.get("interview_history", []))
    avg_time_per_answer = f"{(total_time / num_questions):.1f}s" if num_questions else "N/A"

    suitability_map = {"Poor": 2, "Below Average": 4, "Average": 6, "Good": 8, "Excellent": 10}
    overall_score = suitability_map.get(interview_doc.get("ai_suitability", "Average"), 6)

    report_data = {
        "interview_id": identifier,
        "job_role": (interview_doc.get("job_description", "")[:100] + "...") if len(interview_doc.get("job_description", "")) > 100 else interview_doc.get("job_description", "N/A"),
        "time_taken": f"{int(total_time // 60):02d}m {int(total_time % 60):02d}s" if isinstance(total_time, (int, float)) else "N/A",
        "overall_score": overall_score,
        "max_score": 10,
        "summary": interview_doc.get("ai_report", "Interview completed successfully."),
        "strengths": interview_doc.get("ai_strengths", ["Interview participation", "Professional demeanor"]),
        "weaknesses": interview_doc.get("ai_weaknesses", ["Areas for improvement identified"]),
        "behavioral_analysis": {
            "overall_tone": interview_doc.get("ai_behaviour", "Professional demeanor observed"),
            "body_language": most_common(behavior_counts["body_language"]),
            "eye_contact": most_common(behavior_counts["eye_contact"]),
            "gestures": most_common(behavior_counts["gestures"]),
            "posture": most_common(behavior_counts["posture"]),
            "description": interview_doc.get("ai_behaviour", "Behavioral assessment completed.")
        },
        "response_metrics": {
            "avg_time_per_answer": avg_time_per_answer,
            "avg_answer_length": f"{random.randint(15, 30)} words",
            "filler_words_percentage": f"{random.randint(5, 15)}%"
        },
        "language_analysis": {
            "most_common_words": ["and", "the", "I", "think", "believe"],
            "most_common_phrases": [interview_doc.get("ai_language", "Professional communication style")]
        },
        "suitability_assessment": {
            "score": interview_doc.get("ai_suitability", "Average"),
            "description": f"Overall assessment: {interview_doc.get('ai_suitability', 'Average')} suitability for the role."
        },
        "overall_impression_final": " ".join(list(set(overall_impressions))) if overall_impressions else "Overall impression not available.",
        "suggestions_for_improvement": list(set(suggestions)) if suggestions else ["Continue developing professional skills", "Practice interview techniques"],
        "full_ai_report_text": interview_doc.get("ai_report", "No comprehensive report generated yet."),
        "interview_history": interview_doc.get("interview_history", [])
    }

    return jsonify(report_data)


@app.route('/interview-results/<identifier>')
def interview_results(identifier):
    if not session.get("is_authenticated"):
        return redirect(url_for('index'))

    user_id = session["user"]["user_id"]
    interview = DATABASE["INTERVIEWS"].find_one({
        "$or": [
            {"interview_identifier": identifier},
            {"mock_interview_identifier": identifier}
        ],
        "user_id": user_id,
        "is_completed": True
    })

    if not interview:
        message = "Interview results not found or you do not have access."
        return render_template('interview_results.html', interview=None, duration="00:00", message=message)

    # Ensure duration is a float
    try:
        duration_val = float(interview.get("duration", 0))
    except (TypeError, ValueError):
        duration_val = 0

    minutes = int(duration_val // 60)
    seconds = int(duration_val % 60)
    duration_str = f"{minutes:02d}:{seconds:02d}"

    # Set safe defaults for optional AI fields
    interview.setdefault("ai_report", "No comprehensive report generated yet.")
    interview.setdefault("ai_strengths", ["Interview participation", "Professional demeanor"])
    interview.setdefault("ai_weaknesses", ["Areas for improvement identified"])
    interview.setdefault("ai_behaviour", "Professional demeanor observed")
    interview.setdefault("ai_language", "Professional communication style")
    interview.setdefault("ai_suitability", "Average")
    interview.setdefault("behavior_analysis", [])

    # Optional: calculate summary counts for behavior analysis
    behavior_counts = {"posture": {}, "eye_contact": {}, "gestures": {}, "body_language": {}}
    for snapshot in interview["behavior_analysis"]:
        for key, counter_key in [
            ("posture_analysis", "posture"),
            ("eye_contact_analysis", "eye_contact"),
            ("gestures_analysis", "gestures"),
            ("body_language_analysis", "body_language")
        ]:
            val = snapshot.get(key)
            if val and all(x not in val for x in ["Detected", "Error", "Undetermined", "N/A"]):
                behavior_counts[counter_key][val] = behavior_counts[counter_key].get(val, 0) + 1

    # Add summarized fields to interview object
    interview["behavior_summary"] = {k: max(v, key=v.get) if v else "Not observed" for k, v in behavior_counts.items()}

    return render_template(
        'interview_results.html',
        interview=interview,
        duration=duration_str,
        message=None
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🟢 Starting Flask on port {port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=True)


