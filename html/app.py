from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import sqlite3
import datetime

# --- SETUP ---
app = Flask(__name__)
CORS(app)
with open('malaria_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
DATABASE_FILE = 'history.db'

# API Key for IoT device authentication
API_KEY = "ESP32_SECRET_KEY_12345"

# In-memory storage for the latest monitoring data
latest_monitoring_data = {
    "temperature": 0.0,
    "systolic_bp": 0,
    "diastolic_bp": 0,
    "timestamp": None
}

# --- DATABASE FUNCTIONS ---
def get_db_connection():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    # Malaria history table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME NOT NULL, patient_name TEXT NOT NULL,
            temperature REAL NOT NULL, heart_rate REAL NOT NULL, respiratory_rate REAL NOT NULL,
            systolic_bp REAL NOT NULL, diastolic_bp REAL NOT NULL, spo2 REAL NOT NULL,
            headache INTEGER NOT NULL, chills INTEGER NOT NULL, nausea_vomiting INTEGER NOT NULL,
            fatigue INTEGER NOT NULL, jaundice INTEGER NOT NULL, prediction_result TEXT NOT NULL,
            confidence_score REAL NOT NULL
        )
    ''')
    # GCS history table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS gcs_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME NOT NULL, patient_name TEXT NOT NULL,
            eye_score INTEGER NOT NULL, verbal_score INTEGER NOT NULL, motor_score INTEGER NOT NULL,
            total_score INTEGER NOT NULL, interpretation TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


# --- MALARIA API ENDPOINTS ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        features = np.array([float(data['Temperature']), float(data['Heart_Rate']),
                             float(data['Respiratory_Rate']), float(data['Systolic_BP']), float(data['Diastolic_BP']), float(data['SpO2']),
                             int(data['Headache']), int(data['Chills']), int(data['Nausea_Vomiting']), int(data['Fatigue']),
                             int(data['Jaundice'])]).reshape(1, -1)
        prediction_raw = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)
        confidence = prediction_proba[0][int(prediction_raw)] * 100
        prediction_text = "Positive" if prediction_raw == 1 else "Negative"
        conn = get_db_connection()
        conn.execute('INSERT INTO history (timestamp, patient_name, temperature, heart_rate, respiratory_rate, systolic_bp, diastolic_bp, spo2, headache, chills, nausea_vomiting, fatigue, jaundice, prediction_result, confidence_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                     (datetime.datetime.now(), data['patient_name'], data['Temperature'], data['Heart_Rate'], data['Respiratory_Rate'], data['Systolic_BP'], data['Diastolic_BP'], data['SpO2'], data['Headache'], data['Chills'], data['Nausea_Vomiting'], data['Fatigue'], data['Jaundice'], prediction_text, confidence))
        conn.commit()
        conn.close()
        result = {'positive': bool(prediction_raw), 'confidence': confidence}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/history', methods=['GET'])
def get_history():
    conn = get_db_connection()
    patient_name = request.args.get('patient')
    search_term = request.args.get('search', '')
    if patient_name:
        query = "SELECT * FROM history WHERE patient_name = ? ORDER BY timestamp DESC"
        records = conn.execute(query, (patient_name,)).fetchall()
    elif search_term:
        query = "SELECT * FROM history WHERE patient_name LIKE ? ORDER BY timestamp DESC"
        records = conn.execute(query, ('%' + search_term + '%',)).fetchall()
    else:
        query = "SELECT * FROM history ORDER BY timestamp DESC"
        records = conn.execute(query).fetchall()
    conn.close()
    return jsonify([dict(row) for row in records])

@app.route('/malaria/patients', methods=['GET'])
def get_malaria_patients():
    conn = get_db_connection()
    records = conn.execute("SELECT DISTINCT patient_name FROM history ORDER BY LOWER(patient_name) ASC").fetchall()
    conn.close()
    patient_list = [row['patient_name'] for row in records]
    return jsonify(patient_list)


# --- GCS API ENDPOINTS ---
@app.route('/gcs/save', methods=['POST'])
def save_gcs_record():
    data = request.get_json()
    conn = get_db_connection()
    conn.execute('INSERT INTO gcs_history (timestamp, patient_name, eye_score, verbal_score, motor_score, total_score, interpretation) VALUES (?, ?, ?, ?, ?, ?, ?)',
                 (datetime.datetime.now(), data['patient_name'], data['eye_score'], data['verbal_score'], data['motor_score'], data['total_score'], data['interpretation']))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'message': 'GCS record saved.'})

@app.route('/gcs/history', methods=['GET'])
def get_gcs_history():
    conn = get_db_connection()
    patient_name = request.args.get('patient')
    if patient_name:
        query = "SELECT * FROM gcs_history WHERE patient_name = ? ORDER BY timestamp DESC"
        records = conn.execute(query, (patient_name,)).fetchall()
    else:
        query = "SELECT * FROM gcs_history ORDER BY timestamp DESC"
        records = conn.execute(query).fetchall()
    conn.close()
    return jsonify([dict(row) for row in records])

@app.route('/gcs/patients', methods=['GET'])
def get_gcs_patients():
    conn = get_db_connection()
    records = conn.execute("SELECT DISTINCT patient_name FROM gcs_history ORDER BY LOWER(patient_name) ASC").fetchall()
    conn.close()
    patient_list = [row['patient_name'] for row in records]
    return jsonify(patient_list)

@app.route('/gcs/history/delete/<int:record_id>', methods=['DELETE'])
def delete_gcs_record(record_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM gcs_history WHERE id = ?', (record_id,))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'message': f'Record {record_id} deleted.'})

@app.route('/gcs/history/clear', methods=['DELETE'])
def clear_all_gcs_history():
    conn = get_db_connection()
    conn.execute('DELETE FROM gcs_history')
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'message': 'All GCS history cleared.'})


# --- REAL-TIME MONITORING API ENDPOINTS ---
@app.route('/monitoring/update', methods=['POST'])
def update_monitoring_data():
    sent_key = request.headers.get('x-api-key')
    if sent_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    if not data or 'temperature' not in data or 'systolic_bp' not in data:
        return jsonify({"error": "Invalid data"}), 400
    global latest_monitoring_data
    latest_monitoring_data = {
        "temperature": data['temperature'],
        "systolic_bp": data['systolic_bp'],
        "diastolic_bp": data['diastolic_bp'],
        "timestamp": datetime.datetime.now().isoformat()
    }
    return jsonify({"status": "success", "message": "Data updated"}), 200

@app.route('/monitoring/data', methods=['GET'])
def get_monitoring_data():
    return jsonify(latest_monitoring_data)


# --- RUN THE APP ---
if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', debug=True)