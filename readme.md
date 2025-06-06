git clone https://github.com/yourusername/GlucoCheck-Pro-Plus.git
cd GlucoCheck-Pro-Plus

python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

pip install -r requirements.txt

mkdir -p .streamlit
echo "Gemini_API_KEY = 'your-api-key-here'" > .streamlit/secrets.toml

streamlit run app.py