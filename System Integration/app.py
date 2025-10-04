import gradio as gr
import os
import shutil
import subprocess
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from full_keypoint_extraction import extract_keypoints
from voice_feature_extraction import extract_voice_features_single
from posture_feature_extraction import extract_posture_features

from gait_test_script import run_inference_from_analysis_for_walking
from tandem_walk_test_script import run_inference_from_analysis_for_tandem
from voice_test_script import run_inference_from_analysis_for_voice
from posture_test_script import run_inference_from_analysis_for_posture

# MongoDB imports
from pymongo import MongoClient
from datetime import datetime

CACHE_DIR = "Cache"
os.makedirs(CACHE_DIR, exist_ok=True)
log_list = []

# Global variables are used to store patient information
current_patient_name = ""
current_patient_id = ""
current_analysis_results = {}
current_final_score = 0
current_analysis_date = ""

# MongoDB connection
try:
    client = MongoClient("mongodb://localhost:27017/")  # Change the MongoDB connection string according to your own settings
    db = client["Parkinson_Diagnosis_Feedback"]
    collection = db["Feed_Back"]
except Exception as e:
    print(f"MongoDB connection error: {e}")
    client = None
    db = None
    collection = None


def get_mongodb_connection():
    """Establishes the MongoDB connection"""
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/")  # Enter your MongoDB connection string here

        # Test the connection
        client.server_info()

        db = client["Parkinson_Diagnosis_Feedback"]
        collection = db["Feed_Back"]
        return collection
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        return None

def run_face_prediction_subprocess():
    try:
        result = subprocess.run(
            [
                r"C:\Users\EfeTasyurek\PycharmProjects\pythonProject\System_Integration\face_env\Scripts\python.exe",
                "face_test_script.py",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5000,
            cwd=r"C:\Users\EfeTasyurek\PycharmProjects\pythonProject\System_Integration"
        )

        if result.returncode != 0:
            log_list.append(result.stderr.decode())
            return {"error": "No FACE video recording was found, so no result could be obtained."}

        return json.loads(result.stdout.decode())
    except Exception as e:
        log_list.append(str(e))
        return {"error": "No FACE video recording was found, so no result could be obtained."}


def validate_patient_info(name, pid):
    """Validates patient information"""
    if not name or not name.strip():
        return False, "❌ Patient Name cannot be empty!"

    name = name.strip()
    if len(name) > 35:
        return False, "❌ Patient Name cannot be longer than 35 characters!"

    if not name.replace(" ", "").isalpha():
        return False, "❌ Patient Name can only contain letters and spaces!"

    if not pid or not pid.strip():
        return False, "❌ Patient No cannot be empty!"

    pid = pid.strip()
    if len(pid) > 15:
        return False, "❌ Patient No cannot be longer than 15 characters!"

    if not pid.isdigit():
        return False, "❌ Patient No can only contain numbers!"

    return True, "Valid"


def load_and_cache_files(v1, v2, v3, v4, audio, name, pid):
    global current_patient_name, current_patient_id

    try:
        video_files = [v1, v2, v3, v4, audio]
        uploaded_videos = [v for v in video_files if v is not None]

        if len(uploaded_videos) == 0:
            return "❌ At least one video file must be uploaded!", gr.update(interactive=False), gr.update(visible=False)

        is_valid, validation_message = validate_patient_info(name, pid)
        if not is_valid:
            return validation_message, gr.update(interactive=False), gr.update(visible=False)

        if v1 is not None:
            shutil.copy(v1.name, os.path.join(CACHE_DIR, "Video1_Walking.mp4"))
        if v2 is not None:
            shutil.copy(v2.name, os.path.join(CACHE_DIR, "Video2_Tandem.mp4"))
        if v3 is not None:
            shutil.copy(v3.name, os.path.join(CACHE_DIR, "Video3_Face.mp4"))
        if v4 is not None:
            shutil.copy(v4.name, os.path.join(CACHE_DIR, "Video4_Posture.mp4"))
        if audio is not None:
            shutil.copy(audio.name, os.path.join(CACHE_DIR, "Voice.mp3"))

        current_patient_name = name.strip()
        current_patient_id = pid.strip()

        success_message = f"✅ Files successfully loaded for {current_patient_name} (ID: {current_patient_id})"
        return success_message, gr.update(interactive=True), gr.update(visible=True)

    except Exception as e:
        error_message = f"❌ Error during load: {str(e)}"
        log_list.append(error_message)
        return error_message, gr.update(interactive=False), gr.update(visible=False)


def clear_cache_and_reset():
    global current_patient_name, current_patient_id, current_analysis_results, current_final_score, current_analysis_date
    current_patient_name = ""
    current_patient_id = ""
    current_analysis_results = {}
    current_final_score = 0
    current_analysis_date = ""

    for filename in os.listdir(CACHE_DIR):
        file_path = os.path.join(CACHE_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    return (None, None, None, None, None, "", "",
            "🧹 All data cleared successfully.",
            gr.update(interactive=False), gr.update(visible=False),
            "", gr.update(interactive=False))  # Clear feedback textbox and disable submit button


def exit_program():
    for filename in os.listdir(CACHE_DIR):
        file_path = os.path.join(CACHE_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    os._exit(0)


# Separate functions for parallel processing
def process_walking():
    try:
        walking_features = extract_keypoints("Cache/Video1_Walking.mp4")
        walking_results = run_inference_from_analysis_for_walking(walking_features, patient_id="123456")
        return ("walking", walking_results)
    except Exception as e:
        error_message = f"The result of the walk could not be obtained: {str(e)}"
        log_list.append(error_message)
        return ("walking", {"error": "No WALKING video recording was found, so no result could be obtained."})


def process_tandem():
    try:
        tandem_walk_features = extract_keypoints("Cache/Video2_Tandem.mp4")
        tandem_walk_results = run_inference_from_analysis_for_tandem(tandem_walk_features, patient_id="123456")
        return ("tandem", tandem_walk_results)
    except Exception as e:
        error_message = f"The result of the tandem walk could not be obtained: {str(e)}"
        log_list.append(error_message)
        return ("tandem", {"error": "No TANDEM WALKING video recording was found, so no result could be obtained."})


def process_voice():
    try:
        voice_features = extract_voice_features_single("Cache/Voice.mp3")
        voice_results = run_inference_from_analysis_for_voice(voice_features, patient_id="123456")
        return ("voice", voice_results)
    except Exception as e:
        error_message = f"The result of the voice could not be obtained: {str(e)}"
        log_list.append(error_message)
        return ("voice", {"error": "No VOICE recording data was found, so no result could be obtained."})


def process_posture():
    try:
        posture_features = extract_posture_features("Cache/Video4_Posture.mp4")
        posture_results = run_inference_from_analysis_for_posture(posture_features, patient_id="123456")
        return ("posture", posture_results)
    except Exception as e:
        error_message = f"The result of the posture could not be obtained: {str(e)}"
        log_list.append(error_message)
        return ("posture", {"error": "No POSTURE video recording was found, so no result could be obtained."})


def process_face():
    try:
        face_results = run_face_prediction_subprocess()
        return ("face", face_results)
    except Exception as e:
        error_message = f"The result of the face could not be obtained: {str(e)}"
        log_list.append(error_message)
        return ("face", {"error": "No FACE video recording was found, so no result could be obtained."})


def predict():
    global current_patient_name, current_patient_id, current_analysis_results, current_final_score, current_analysis_date

    available_tasks = []

    if os.path.exists("Cache/Video1_Walking.mp4"):
        available_tasks.append(process_walking)
    if os.path.exists("Cache/Video2_Tandem.mp4"):
        available_tasks.append(process_tandem)
    if os.path.exists("Cache/Voice.mp3"):
        available_tasks.append(process_voice)
    if os.path.exists("Cache/Video4_Posture.mp4"):
        available_tasks.append(process_posture)
    if os.path.exists("Cache/Video3_Face.mp4"):
        available_tasks.append(process_face)

    if not available_tasks:
        return "❌ No files found to process!", gr.update(interactive=False)

    results = {
        "walking": {"error": "No WALKING video recording was found, so no result could be obtained."},
        "tandem": {"error": "No TANDEM WALKING video recording was found, so no result could be obtained."},
        "voice": {"error": "No VOICE recording data was found, so no result could be obtained."},
        "posture": {"error": "No POSTURE video recording was found, so no result could be obtained."},
        "face": {"error": "No FACE video recording was found, so no result could be obtained."}
    }

    print(f"🚀 {len(available_tasks)} processes starting in parallel...")

    with ThreadPoolExecutor(max_workers=min(len(available_tasks), 5)) as executor:
        future_to_task = {executor.submit(task): task.__name__ for task in available_tasks}

        for future in as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                result_type, result_data = future.result()
                results[result_type] = result_data
                print(f"✅ {result_type} process completed")
            except Exception as e:
                error_msg = f"❌ Error in {task_name}: {str(e)}"
                print(error_msg)
                log_list.append(error_msg)

    # Store results globally for feedback submission
    current_analysis_results = results
    current_analysis_date = time.strftime("%Y-%m-%d %H:%M:%S")

    def safe_get_prob(results_dict):
        return results_dict.get("probability", 0) if isinstance(results_dict,
                                                                dict) and "error" not in results_dict else 0

    p_walking = safe_get_prob(results["walking"])
    p_tandem = safe_get_prob(results["tandem"])
    p_voice = safe_get_prob(results["voice"])
    p_posture = safe_get_prob(results["posture"])
    p_face = safe_get_prob(results["face"])

    processed_count = sum([1 for p in [p_walking, p_tandem, p_voice, p_posture, p_face] if p != 0])

    if processed_count == 0:
        final_diagnosis = 0
    else:
        final_diagnosis = (p_walking * (100 / processed_count) +
                           p_tandem * (100 / processed_count) +
                           p_voice * (100 / processed_count) +
                           p_posture * (100 / processed_count) +
                           p_face * (100 / processed_count)) / 100

    current_final_score = final_diagnosis

    def format_result(title, result_dict, icon):
        if "error" in result_dict:
            return f"{icon} **{title}**: ❌ Not Analyzed\n   └─ *{result_dict['error']}*"
        else:
            prob = result_dict.get("probability", 0)
            return f"{icon} **{title}**: ✅ Analyzed (Score: {prob:.4f})"

    # Risk level determination
    if final_diagnosis < 0.3:
        risk_level = "🟢 LOW RISK"
        risk_color = "#28a745"
    elif final_diagnosis < 0.7:
        risk_level = "🟡 MODERATE RISK"
        risk_color = "#ffc107"
    else:
        risk_level = "🔴 HIGH RISK"
        risk_color = "#dc3545"

    summary_string = f"""
# 📊 PARKINSON'S ANALYSIS REPORT

## 👤 Patient Information
**Name:** {current_patient_name}  
**ID:** {current_patient_id}  
**Analysis Date:** {current_analysis_date}

---

## 🔬 Analysis Results

{format_result("Gait Analysis", results["walking"], "🚶‍♂️")}

{format_result("Tandem Walk Analysis", results["tandem"], "🤸‍♂️")}

{format_result("Voice Analysis", results["voice"], "🎤")}

{format_result("Posture Analysis", results["posture"], "🧍‍♂️")}

{format_result("Facial Analysis", results["face"], "😊")}

---

## 🎯 Final Assessment

**Risk Level:** <span style="color: {risk_color}; font-weight: bold; font-size: 18px;">{risk_level}</span>

**Parkinson's Probability Score:** `{final_diagnosis:.4f}`

**Analyses Completed:** {processed_count}/5

---

## 📋 Recommendations

{f"⚠️ **IMPORTANT:** This analysis indicates elevated risk. Please consult with a neurologist for comprehensive evaluation." if final_diagnosis >= 0.5 else "✅ **Good News:** Current analysis shows low risk indicators. Continue regular medical check-ups."}

---
*This report is generated by AI-assisted analysis and should not replace professional medical consultation.*
"""

    print("🏁 All analyses completed!")
    return summary_string, gr.update(interactive=True)  # Enable submit button after analysis

# Global variable – to store analysis results
analysis_results_cache = {}
def submit_doctor_feedback(doctor_diagnosis):
    """Save the doctor’s diagnosis to MongoDB and return two outputs:
       1) Status message for feedback_status
       2) New value for the doctor_diagnosis field (clear on success)"""
    global current_patient_name, current_patient_id, current_analysis_results, current_final_score, current_analysis_date

    # If the input is empty: write status, leave the field as is
    if not doctor_diagnosis or not doctor_diagnosis.strip():
        return "❌ Please enter your diagnosis before submitting!", gr.update(value=doctor_diagnosis)

    # If there is no DB connection: write status, leave the field as is
    if collection is None:
        return "❌ Database connection error. Cannot save feedback.", gr.update(value=doctor_diagnosis)

    try:
        feedback_data = {
            "Patient_Name": current_patient_name,
            "Patient_No": current_patient_id,
            "Analysis_Date": current_analysis_date,
            "Walking_Analysis_Result": (
                current_analysis_results.get("walking", {}).get("probability", "N/A")
                if "error" not in current_analysis_results.get("walking", {}) else "Not Analyzed"
            ),
            "Tandem_Walking_Analysis_Result": (
                current_analysis_results.get("tandem", {}).get("probability", "N/A")
                if "error" not in current_analysis_results.get("tandem", {}) else "Not Analyzed"
            ),
            "Voice_Analysis_Result": (
                current_analysis_results.get("voice", {}).get("probability", "N/A")
                if "error" not in current_analysis_results.get("voice", {}) else "Not Analyzed"
            ),
            "Face_Analysis_Result": (
                current_analysis_results.get("face", {}).get("probability", "N/A")
                if "error" not in current_analysis_results.get("face", {}) else "Not Analyzed"
            ),
            "Posture_Analysis_Result": (
                current_analysis_results.get("posture", {}).get("probability", "N/A")
                if "error" not in current_analysis_results.get("posture", {}) else "Not Analyzed"
            ),
            "Final_Score": current_final_score,
            "Doctor_Diagnosis": doctor_diagnosis.strip(),
            "Record_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        result = collection.insert_one(feedback_data)

        if result.inserted_id:
            # If successful: write status and clear the doctor’s field
            return f"✅ Doctor's diagnosis successfully saved to database!\nRecord ID: {result.inserted_id}", ""

        # If failed: write status, leave the field as is
        return "❌ Failed to save feedback to database.", gr.update(value=doctor_diagnosis)

    except Exception as e:
        error_msg = f"❌ Error saving feedback: {str(e)}"
        log_list.append(error_msg)
        # In case of error: write status, leave the field as is
        return error_msg, gr.update(value=doctor_diagnosis)


custom_css = """
/* Theme and Colors */
:root {
    --primary-color: #4CAF50;
    --secondary-color: #388E3C;
    --success-color: #81C784;
    --warning-color: #FFB74D;
    --danger-color: #E57373;
    --dark-bg: #F1F8E9;
    --light-bg: #FFFFFF;
    --card-bg: #FFFFFF;
    --border-color: #B0BEC5;
    --text-primary: #212121;
    --text-secondary: #757575;
    --shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 25px 50px rgba(0, 0, 0, 0.15);
}

.gradio-container {
    font-family: 'Roboto', sans-serif !important;
    background: linear-gradient(135deg, #AED581 0%, #81C784 100%) !important;
    min-height: 100vh;
}

.gr-form {
    background: var(--card-bg) !important;
    border-radius: 20px !important;
    box-shadow: var(--shadow-lg) !important;
    border: none !important;
    margin: 20px !important;
    padding: 30px !important;
}

.main-header {
    text-align: center;
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white;
    padding: 40px 20px;
    border-radius: 20px;
    margin-bottom: 30px;
    box-shadow: var(--shadow);
}

.main-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.main-header p {
    font-size: 1.2rem;
    opacity: 0.9;
    margin: 0;
}

.gr-button {
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 15px 30px !important;
    border-radius: 12px !important;
    border: none !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
    color: white !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 15px -2px rgba(37, 99, 235, 0.3) !important;
}

.gr-button-secondary {
    background: linear-gradient(135deg, var(--success-color), #10b981) !important;
    color: white !important;
}

.gr-button-secondary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 15px -2px rgba(5, 150, 105, 0.3) !important;
}

.gr-button[variant="stop"] {
    background: linear-gradient(135deg, var(--danger-color), #ef4444) !important;
    color: white !important;
}

.gr-button[variant="stop"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 15px -2px rgba(220, 38, 38, 0.3) !important;
}

.gr-textbox, .gr-file {
    border-radius: 12px !important;
    border: 2px solid var(--border-color) !important;
    background: var(--card-bg) !important;
    box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    transition: all 0.3s ease !important;
}

.gr-textbox:focus, .gr-file:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
}

.gr-accordion {
    border: 2px solid var(--border-color) !important;
    border-radius: 16px !important;
    background: var(--card-bg) !important;
    box-shadow: var(--shadow) !important;
    overflow: hidden !important;
}

.gr-accordion-header {
    background: linear-gradient(135deg, #f8fafc, #e2e8f0) !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    padding: 20px !important;
}

.gr-tab-nav {
    background: var(--card-bg) !important;
    border-radius: 16px 16px 0 0 !important;
    box-shadow: var(--shadow) !important;
}

.gr-tab-nav button {
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 15px 25px !important;
    border-radius: 12px 12px 0 0 !important;
    border: none !important;
    background: transparent !important;
    color: var(--text-secondary) !important;
    transition: all 0.3s ease !important;
}

.gr-tab-nav button.selected {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
    color: white !important;
}

.result-container {
    background: var(--card-bg) !important;
    border-radius: 16px !important;
    border: 2px solid var(--border-color) !important;
    box-shadow: var(--shadow) !important;
    padding: 20px !important;
    margin: 20px 0 !important;
}

.feedback-section {
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe) !important;
    border: 2px solid #0ea5e9 !important;
    border-radius: 16px !important;
    padding: 25px !important;
    margin-top: 30px !important;
    box-shadow: var(--shadow) !important;
}

.feedback-title {
    color: var(--primary-color) !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    margin-bottom: 15px !important;
    display: flex !important;
    align-items: center !important;
}

.feedback-title::before {
    content: "👨‍⚕️";
    margin-right: 10px;
    font-size: 24px;
}

@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2rem;
    }

    .gr-form {
        margin: 10px !important;
        padding: 20px !important;
    }

    .gr-button {
        padding: 12px 20px !important;
        font-size: 14px !important;
    }
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.gr-form {
    animation: slideIn 0.6s ease-out;
}

.status-success {
    color: var(--success-color) !important;
    font-weight: 600 !important;
}

.status-error {
    color: var(--danger-color) !important;
    font-weight: 600 !important;
}

.status-warning {
    color: var(--warning-color) !important;
    font-weight: 600 !important;
}

.gr-file:hover, .gr-textbox:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 15px -3px rgba(0, 0, 0, 0.1) !important;
}

"""

# Interface
with gr.Blocks(css=custom_css, title="Parkinson's Early Diagnosis System", theme=gr.themes.Base()) as parkinson_ui:
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>🧠 A Novel Multimodal AI Framework for Early Diagnosis of Idiopathic Parkinson's Disease</h1>
        <p>Advanced AI-Powered Medical Analysis Platform</p>
    </div>
    """)

    with gr.Tab("📊 Data Upload & Analysis", elem_classes=["main-tab"]):
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Group(elem_classes=["upload-section"]):
                    gr.Markdown("### 👤 Patient Information")
                    with gr.Row():
                        patient_name = gr.Textbox(
                            label="Patient Name",
                            placeholder="Enter full name (e.g., John Doe)",
                            elem_classes=["patient-input"]
                        )
                        patient_id = gr.Textbox(
                            label="Patient ID",
                            placeholder="Enter patient ID (numbers only)",
                            elem_classes=["patient-input"]
                        )

                with gr.Group(elem_classes=["upload-section"]):
                    gr.Markdown("### 📁 Medical Data Upload")
                    with gr.Row():
                        video1 = gr.File(
                            label="🚶‍♂️ Gait Analysis Video",
                            file_types=[".mp4", ".avi", ".mov"],
                            elem_classes=["file-upload"]
                        )
                        video2 = gr.File(
                            label="🤸‍♂️ Tandem Walk Video",
                            file_types=[".mp4", ".avi", ".mov"],
                            elem_classes=["file-upload"]
                        )

                    with gr.Row():
                        video3 = gr.File(
                            label="😊 Facial Expression Video",
                            file_types=[".mp4", ".avi", ".mov"],
                            elem_classes=["file-upload"]
                        )
                        video4 = gr.File(
                            label="🧍‍♂️ Posture Analysis Video",
                            file_types=[".mp4", ".avi", ".mov"],
                            elem_classes=["file-upload"]
                        )

                    audio = gr.File(
                        label="🎤 Voice Sample",
                        file_types=[".wav", ".mp3", ".m4a"],
                        elem_classes=["file-upload"]
                    )

                # Kontrol butonları
                with gr.Row():
                    load_button = gr.Button(
                        "📤 Load & Validate Data",
                        variant="primary",
                        size="lg",
                        elem_classes=["control-button"]
                    )
                    clear_button = gr.Button(
                        "🧹 Clear All Data",
                        variant="secondary",
                        size="lg",
                        elem_classes=["control-button"]
                    )

                # Status göstergesi
                status_box = gr.Textbox(
                    label="📊 System Status",
                    placeholder="Upload files and enter patient information to begin...",
                    interactive=False,
                    elem_classes=["status-box"]
                )

            with gr.Column(scale=1):
                with gr.Group(elem_classes=["analysis-section"]):
                    gr.Markdown("### 🔬 Analysis Control")

                    predict_button = gr.Button(
                        "🔮 Start Analysis",
                        variant="primary",
                        size="lg",
                        interactive=False,
                        elem_classes=["predict-button"]
                    )

                    analysis_progress = gr.HTML(
                        visible=False,
                        elem_classes=["progress-section"]
                    )

                with gr.Group(elem_classes=["system-section"]):
                    gr.Markdown("### ⚙️ System Control")
                    exit_button = gr.Button(
                        "🔴 Exit Application",
                        variant="stop",
                        size="lg",
                        elem_classes=["exit-button"]
                    )

    with gr.Tab("📋 Analysis Results", elem_classes=["results-tab"]):
        with gr.Column():
            gr.Markdown("### 📊 Comprehensive Analysis Report")
            result_box = gr.Markdown(
                "📋 Analysis results will appear here after processing...",
                elem_classes=["result-container"]
            )

            # Doktor feedback bölümü
            with gr.Group(elem_classes=["doctor-feedback-section"]):
                gr.Markdown("### 👨‍⚕️ Doctor's Diagnosis & Feedback")

                doctor_diagnosis = gr.Textbox(
                    label="Medical Diagnosis",
                    placeholder="Enter your professional diagnosis and medical assessment...",
                    lines=4,
                    elem_classes=["doctor-input"]
                )

                submit_feedback_button = gr.Button(
                    "📝 Submit Diagnosis",
                    variant="primary",
                    size="lg",
                    interactive=False,
                    elem_classes=["submit-button"]
                )

                feedback_status = gr.Textbox(
                    label="Submission Status",
                    placeholder="Diagnosis submission status will appear here...",
                    interactive=False,
                    elem_classes=["feedback-status"]
                )

    with gr.Tab("ℹ️ System Information", elem_classes=["info-tab"]):
        gr.Markdown("""
        ## 🧠 A Novel Multimodal AI Framework for Early Diagnosis of Idiopathic Parkinson's Disease

        ### 📝 About This System
        This advanced medical analysis platform uses artificial intelligence to assess potential Parkinson's disease indicators through multiple biomarkers:

        - **🚶‍♂️ Gait Analysis**: Evaluates walking patterns and movement dynamics
        - **🤸‍♂️ Tandem Walking**: Assesses balance and coordination
        - **🎤 Voice Analysis**: Analyzes speech patterns and vocal characteristics  
        - **🧍‍♂️ Posture Assessment**: Evaluates postural stability and control
        - **😊 Facial Analysis**: Examines facial expressions and micro-movements

        ### 🔬 How It Works
        1. **Data Collection**: Upload medical recordings in supported formats
        2. **AI Processing**: Advanced machine learning algorithms analyze each modality
        3. **Risk Assessment**: Integrated scoring system provides probability assessment
        4. **Professional Report**: Comprehensive analysis report for medical review

        ### ⚠️ Important Disclaimer
        This system is designed as a **screening tool** to assist healthcare professionals. It should **never replace** professional medical diagnosis or treatment decisions.

        ### 🔒 Privacy & Security
        - All data is processed locally on your system
        - No patient information is transmitted externally
        - Temporary files are automatically cleared after analysis

        ### 📞 Support
        For technical support or questions about this system, please contact your system administrator.
        """)

    # Event handlers
    load_button.click(
        fn=load_and_cache_files,
        inputs=[video1, video2, video3, video4, audio, patient_name, patient_id],
        outputs=[status_box, predict_button, analysis_progress]
    )

    clear_button.click(
        fn=clear_cache_and_reset,
        outputs=[video1, video2, video3, video4, audio, patient_name, patient_id,
                 status_box, predict_button, analysis_progress]
    )

    predict_button.click(
        fn=predict,
        outputs=[result_box, submit_feedback_button]
    )

    submit_feedback_button.click(
        fn=submit_doctor_feedback,
        inputs=[doctor_diagnosis],
        outputs=[feedback_status, doctor_diagnosis]
    )

    exit_button.click(fn=exit_program)

if __name__ == "__main__":
    parkinson_ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_api=False,
        share=False
    )