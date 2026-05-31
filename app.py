from flask import Flask, request, render_template, session, send_file, redirect, url_for
import joblib
import os
import sys
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from fpdf import FPDF

# --- Paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))

# يدعم إذا analyze.py داخل نفس الفولدر أو داخل models
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'models'))

from analyze import extract_features

# يدعم إذا المودل داخل نفس الفولدر أو داخل models
model_path_1 = os.path.join(current_dir, 'linkshield_model.pkl')
model_path_2 = os.path.join(current_dir, 'models', 'linkshield_model.pkl')

if os.path.exists(model_path_2):
    model_path = model_path_2
else:
    model_path = model_path_1

linkshield_model = joblib.load(model_path)

# --- Flask Setup ---
app = Flask(__name__)
app.secret_key = "secret_key_for_session"

# --- Admin Login ---
app.config['ADMIN_USERNAME'] = 'admin'
app.config['ADMIN_PASSWORD'] = '1234'

app.config['MODEL_ACCURACY'] = 91.48

# --- Database ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class ScanRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url_address = db.Column(db.String(500), nullable=False)
    result = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# --- Analyze URL ---
def analyze_url(url):
    clean = str(url).lower().strip()
    clean = clean.replace("https://", "").replace("http://", "").rstrip("/")

    try:
        features = extract_features(clean, 0)[:-1]
    except TypeError:
        features = extract_features(url)

    reasons = []

    if 'https' not in str(url).lower():
        reasons.append("🚩 Insecure Connection: Protocol 'https' not detected.")

    if len(clean) > 75:
        reasons.append(f"🚩 URL is unusually long ({len(clean)} characters).")

    domain = clean.split('/')[0]

    digit_count = sum(c.isdigit() for c in domain)
    if digit_count > 6:
        reasons.append(f"🚩 Too many digits in domain ({digit_count}).")

    hyphen_count = domain.count('-')
    if hyphen_count >= 3:
        reasons.append(f"🚩 Too many hyphens in domain ({hyphen_count}).")

    if '@' in clean:
        reasons.append("🚩 '@' symbol detected in URL. This is suspicious.")

    if domain.replace('.', '').isdigit():
        reasons.append("🚩 Domain appears to be an IP address.")

    return features, reasons

# --- PDF Generation ---
def generate_pdf(url, result, reasons, recommendation):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "LinkShield Scan Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"URL: {url}", ln=True)
    pdf.cell(0, 10, f"Result: {result}", ln=True)

    pdf.cell(0, 10, "Technical Analysis:", ln=True)
    for r in reasons:
        safe_reason = r.encode('latin-1', 'ignore').decode('latin-1')
        pdf.cell(0, 8, f"- {safe_reason}", ln=True)

    pdf.ln(5)
    pdf.multi_cell(0, 8, f"Guidance: {recommendation}")

    filename = "scan_report.pdf"
    pdf.output(filename)
    return filename

# --- Home ---
@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    reasons = []
    recommendation = ""
    url = ""

    if request.method == 'POST':
        url = request.form.get('url', '').strip()

        features, reasons = analyze_url(url)
        prediction = linkshield_model.predict([features])

        if prediction[0] == 0:
            result = "SAFE"
            recommendation = "The intelligence scan confirms this link is legitimate. No phishing indicators found."
        else:
            result = "MALICIOUS"
            recommendation = "CRITICAL: Phishing signature detected. Accessing this link may compromise your security."

        if 'history' not in session:
            session['history'] = []

        session['history'].append({
            "url": url,
            "result": result,
            "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        })
        session.modified = True

        new_entry = ScanRecord(url_address=url, result=result)
        db.session.add(new_entry)
        db.session.commit()

    return render_template(
        'main.html',
        result=result,
        reasons=reasons,
        recommendation=recommendation,
        url=url
    )

# --- Download PDF ---
@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    url = request.form.get('url')
    result = request.form.get('result')
    reasons = request.form.getlist('reasons')
    recommendation = request.form.get('recommendation')

    pdf_file = generate_pdf(url, result, reasons, recommendation)
    return send_file(pdf_file, as_attachment=True)

# --- History ---
@app.route('/history', methods=['GET'])
def history():
    user_history = session.get('history', [])
    return render_template('history.html', history=user_history)

# --- Admin Login ---
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    error = None

    if request.method == 'POST':
        u = request.form.get('username', '')
        p = request.form.get('password', '')

        if u == app.config['ADMIN_USERNAME'] and p == app.config['ADMIN_PASSWORD']:
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            error = "Wrong username or password."

    return render_template('admin_login.html', error=error)

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

# --- Admin Dashboard ---
@app.route('/admin')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    total_scans = ScanRecord.query.count()
    safe_count = ScanRecord.query.filter_by(result='SAFE').count()
    malicious_count = ScanRecord.query.filter_by(result='MALICIOUS').count()

    accuracy = app.config.get('MODEL_ACCURACY', 91.48)
    error_rate = round(100 - float(accuracy), 2)

    filter_value = request.args.get('filter', 'ALL')

    if filter_value == 'SAFE':
        latest = ScanRecord.query.filter_by(result='SAFE').order_by(ScanRecord.timestamp.desc()).limit(50).all()
    elif filter_value == 'MALICIOUS':
        latest = ScanRecord.query.filter_by(result='MALICIOUS').order_by(ScanRecord.timestamp.desc()).limit(50).all()
    else:
        latest = ScanRecord.query.order_by(ScanRecord.timestamp.desc()).limit(50).all()

    return render_template(
        'admin.html',
        total_scans=total_scans,
        safe_count=safe_count,
        malicious_count=malicious_count,
        accuracy=accuracy,
        error_rate=error_rate,
        latest=latest
    )

if __name__ == '__main__':
    app.run(debug=True)