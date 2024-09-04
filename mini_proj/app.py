from flask import Flask, request, render_template, redirect, url_for, session, jsonify
import hashlib
import json
import os
from flask import Flask, render_template
from sklearn.impute import SimpleImputer
import pandas as pd
import joblib
import numpy as np



app = Flask(__name__, static_folder='assets')
app.secret_key = 'super_secret_key'
DATA_FILE = "patients.json"
MODEL_FILE = "model.pkl"




def read_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                data = f.read().strip()
                if data:  # Check if file is not empty
                    return json.loads(data)
        except json.JSONDecodeError:
            # Log an error message or handle it as needed
            print("Error decoding JSON file. Returning empty dictionary.")
            return {}
    return {}




class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def write_data(data):
    with open("patients.json", "w") as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Get form data
        age = float(request.form.get('age', 0))
        gender = 1.0 if request.form.get('gender', 'Male') == "Male" else 0.0
        cigarettes_per_day = float(request.form.get('cigarettes_per_day', 0))
        systolic_bp = float(request.form.get('systolic_bp', 0))
        diastolic_bp = float(request.form.get('diastolic_bp', 0))
        cholesterol = float(request.form.get('cholesterol', 0))
        glucose = float(request.form.get('glucose', 0))
        hypertension = 1.0 if request.form.get('hypertension', 'No') == "Yes" else 0.0
        bp_medication = 1.0 if request.form.get('bp_medication', 'No') == "Yes" else 0.0
        diabetes = 1.0 if request.form.get('diabetes', 'No') == "Yes" else 0.0

        # Prepare data for prediction
        data = {
            "male": gender,
            "age": age,
            "cigsPerDay": cigarettes_per_day,
            "BPMeds": bp_medication,
            "prevalentHyp": hypertension,
            "diabetes": diabetes,
            "totChol": cholesterol,
            "sysBP": systolic_bp,
            "diaBP": diastolic_bp,
            "glucose": glucose,
        }

        # Load the model
        model = joblib.load(MODEL_FILE)
        X_patient = pd.DataFrame([data])
        imputer = SimpleImputer(strategy='median', missing_values=np.nan)
        X_patient = pd.DataFrame(imputer.fit_transform(X_patient), columns=X_patient.columns)
        prediction = model.predict(X_patient)[0]
        print(X_patient)
        print("\n")
        print(model.predict(X_patient))
        print("\n")
        print(model.predict(X_patient)[0])
        # Save the prediction
        patients = read_data()
        username = session['username']
        if username in patients:
            patients[username]["medical_info"] = data
            patients[username]["result"] = prediction
        write_data(patients)

        # Redirect to the result page
        if prediction==1:
            return redirect(url_for("positive_result"))
        else:
            return redirect(url_for("negative_result"))
        

    # For a GET request, render the form
    return render_template("predict.html")


'''@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Get form data
    age = float(request.form.get('age', 0))
    gender = 1.0 if request.form.get('gender', 'Male') == "Male" else 0.0
    cigarettes_per_day = float(request.form.get('cigarettes_per_day', 0))
    systolic_bp = float(request.form.get('systolic_bp', 0))
    diastolic_bp = float(request.form.get('diastolic_bp', 0))
    cholesterol = float(request.form.get('cholesterol', 0))
    glucose = float(request.form.get('glucose', 0))
    hypertension = 1.0 if request.form.get('hypertension', 'No') == "Yes" else 0.0
    bp_medication = 1.0 if request.form.get('bp_medication', 'No') == "Yes" else 0.0
    diabetes = 1.0 if request.form.get('diabetes', 'No') == "Yes" else 0.0

    # Prepare data for prediction
    data = {
        "male": gender,
        "age": age,
        "cigsPerDay": cigarettes_per_day,
        "BPMeds": bp_medication,
        "prevalentHyp": hypertension,
        "diabetes": diabetes,
        "totChol": cholesterol,
        "sysBP": systolic_bp,
        "diaBP": diastolic_bp,
        "glucose": glucose,
        
    }

    # Load the model
    model = joblib.load(MODEL_FILE)
    X_patient = pd.DataFrame([data])
    imputer = SimpleImputer(strategy='median', missing_values=np.nan)
    X_patient = pd.DataFrame(imputer.fit_transform(X_patient), columns=X_patient.columns)
    prediction = model.predict(X_patient)[0]

    # Save the prediction
    patients = read_data()
    username = session['username']
    if username in patients:
        patients[username]["medical_info"] = data
        patients[username]["result"] = prediction
    write_data(patients)
    
    
    
    if prediction == 1:
        return redirect(url_for("positive_result"))
    else:
        return redirect(url_for("negative_result"))'''

@app.route('/positive_result')
def positive_result():
    return render_template("yes.html")


@app.route('/negative_result')
def negative_result():
    return render_template("no.html")

@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        username = request.form["username"]
        password = hash_password(request.form["password"])

        patients = read_data()
        if username in patients:
            return "User already exists"

        patients[username] = {
            "password": password,
            "medical_info": {},
            "result": None
        }
        write_data(patients)
        return redirect(url_for("login"))
    return render_template("signup.html")


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        username = request.form["username"]
        password = hash_password(request.form["password"])

        patients = read_data()
        if username in patients and patients[username]["password"] == password:
            session['username'] = username
            return redirect(url_for("index"))
        else:
            return "Invalid username or password"
    return render_template("login.html")


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()  # Clear the session
    return redirect(url_for('login'))

@app.route('/solutions')
def solutions():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template("solutions.html")

@app.route('/whatweoffer')
def whatweoffer():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template("whatweoffer.html")

@app.route('/faqs')
def faqs():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template("faqs.html")
    
    
    

if __name__ == '__main__':
    app.run(debug=True)
