<h2>🤖 Resumentor – AI-Powered Resume Analyzer & Mock Interview Platform</h2>

Resumentor is an intelligent web platform designed to help job seekers improve their resumes and prepare for interviews using AI and NLP.<br>
Upload your resume and get instant analysis: the system identifies skill gaps, highlights strengths, and provides actionable suggestions to make your resume stand out. Beyond resume evaluation, Resumentor offers AI-driven mock interviews, where users can practice real-world questions and receive personalized feedback on clarity, confidence, and technical accuracy.<br>
Built with Flask (backend), HTML/CSS/JavaScript (frontend), and MongoDB, the platform leverages Sentence Transformers, PyTorch, and scikit-learn to deliver precise insights. Optional AI integrations include OpenAI or Google Generative AI for enhanced resume and interview analysis.<br>

```bash
✨ Key Features :
🧾 Resume Parsing & AI Analysis<br>
📊 Skill Gap & Feedback Suggestions<br>
🗣️ Mock Interview Simulation<br>
💼 Personalized Career Insights Dashboard<br>

Resumentor makes job preparation smarter, faster, and more effective — empowering users to confidently apply for their dream roles.

🏗️ Tech Stack :
Frontend: HTML, CSS, JavaScript
Backend: Flask (Python)
Database: MongoDB
AI/NLP: Sentence Transformers, Scikit-learn, Torch, Transformers, NLTK
Additional Tools: OpenAI / Google Generative AI (optional integration)


📁 Project Structure :
RESUMENTOR/
│
├── project/
│   ├── .venv_resumentor/
│   └── app/
│       ├── flask_session/
│       ├── static/
│       ├── templates/
│       ├── .env
│       ├── .env.example
│       ├── explore_db.py
│       ├── main.py
│       └── setup.ps1
│   ├── .dockerignore
│   ├── .gitignore
│   ├── requirements.txt
│   ├── run.sh
│   ├── runtime.txt
│   └── setup.ps1
├── LICENSE
└── README.md



⚙️ Installation & Setup

1️⃣ Clone the repository :
git clone https://github.com/mr-aakash897/ResuMentor.git
cd project
cd app

2️⃣ Create a virtual environment :
python -m venv venv311
.\venv311\Scripts\Activate.ps1      # for Windows
# or
source venv/bin/activate   # for macOS/Linux

3️⃣ Install dependencies :
pip install -r requirements.txt

4️⃣ Run the app :
python main.py         (Now visit http://127.0.0.1:8000/  in your browser 🎉)


🙌 Contributors :
Original Work: Aryan Bansal, Tanisha Khanna, Arnav Bansal, Tushti Gupta
Modified & Maintained by: Aakash Chouhan


🪪 License:
This project is licensed under the MIT License.
See the LICENSE
 file for details.


💬 Feedback & Support:
If you find this project useful, give it a ⭐ on GitHub!
For issues, suggestions, or collaborations, feel free to open an issue or reach out via email.
