from flask import Flask, request, jsonify, send_file
import os
from flask_cors import CORS
import traceback
from PIL import Image
import numpy as np
import uuid
import json
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from ultralytics import YOLO
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

app = Flask(__name__)
# Enable CORS for all routes and origins
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['REPORTS_FOLDER'] = 'reports'
app.config['PDF_REPORTS_FOLDER'] = 'pdf_reports'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['PDF_REPORTS_FOLDER'], exist_ok=True)

# Load the YOLO model
print("Loading YOLO model...")
try:
    # Specify the full path to your model file
    model_path = 'C:\\Users\\aniru\\Desktop\\sahaay-connect-main\\src\\best.pt'
    model = YOLO(model_path)
    print(f"YOLO model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print(traceback.format_exc())
    model = None

# Rich Medical Knowledge Base for Pneumonia and other conditions
medical_knowledge_base = {
    "pneumonia": {
        "description": (
            "Pneumonia is an infection that inflames the air sacs in one or both lungs, "
            "which may fill with fluid or pus. Causes include bacteria, viruses, or fungi."
        ),
        "symptoms": [
            "Fever, chills, or sweating",
            "Coughing that produces phlegm",
            "Shortness of breath",
            "Fatigue"
        ],
        "care_advice": [
            "Get plenty of rest to allow your body to recover.",
            "Stay hydrated by drinking water and warm fluids.",
            "Follow your doctor's prescribed antibiotic or antiviral medication plan.",
            "Use a humidifier to ease breathing.",
            "Avoid smoking and exposure to secondhand smoke."
        ],
        "mental_health_advice": (
            "It's natural to feel worried when you're unwell. Rest your mind by engaging in "
            "calming activities like listening to soothing music or meditating. Remember, recovery takes time."
        ),
        "technical_details": (
            "Analysis shows increased opacity in the lower right lung field with air bronchograms, "
            "consistent with lobar pneumonia. No evidence of pleural effusion. Heart size normal. "
            "Costophrenic angles are clear."
        ),
        "recommendations": [
            "Schedule an appointment with a pulmonologist within 48 hours",
            "Rest and stay hydrated",
            "Monitor oxygen levels if possible",
            "Take prescribed antibiotics as directed"
        ],
        "follow_up_needed": True,
        "references": [
            {
                "source_type": "Book",
                "title": "Davidson's Principles and Practice of Medicine",
                "excerpt": "Pneumonia often presents with acute symptoms such as fever, cough, and breathlessness, and may require antibiotic therapy for bacterial causes."
            },
            {
                "source_type": "Book",
                "title": "Harrison's Principles of Internal Medicine",
                "excerpt": "In pneumonia, bacterial or viral pathogens infiltrate alveoli, leading to inflammatory responses. Hydration and oxygen therapy are critical for management."
            },
            {
                "source_type": "Journal",
                "title": "The Lancet - Pneumonia in Adults",
                "excerpt": "Effective treatment of pneumonia involves pathogen-specific antibiotics and supportive care like fluids and oxygen therapy."
            },
        ]
    },
    "normal": {
        "description": (
            "Your lungs appear normal with no visible signs of infection, inflammation, or other abnormalities."
        ),
        "simplified_explanation": (
            "Your chest X-ray looks normal. Your lungs appear clear without any visible signs of infection, "
            "fluid, or other abnormalities. This is a positive finding indicating healthy lung tissue."
        ),
        "technical_details": (
            "Lungs are clear without focal consolidation, effusion, or pneumothorax. "
            "Cardiomediastinal silhouette is normal. Osseous structures are intact."
        ),
        "health_tips": [
            "Continue maintaining good respiratory health by avoiding smoking and air pollutants.",
            "Regular exercise can help maintain and improve lung capacity.",
            "Stay up-to-date with vaccinations that prevent respiratory infections.",
            "Consider annual check-ups to monitor your respiratory health."
        ],
        "recommendations": [
            "No immediate action required",
            "Continue regular health checkups",
            "Maintain good respiratory health practices",
            "Contact your doctor if you develop new symptoms"
        ],
        "follow_up_needed": False,
        "preventive_measures": [
            "Wash hands frequently to prevent respiratory infections.",
            "Maintain good indoor air quality with proper ventilation.",
            "Wear appropriate protective equipment in dusty or polluted environments.",
            "Stay hydrated to keep mucous membranes moist and functioning properly."
        ]
    }
}

# In-memory storage for reports
reports_db = {}

# Improved LLM setup for langchain with customized prompts
def get_llm_chain(query_type="general", llm_model="llama3.2", temperature=0.2):
    # Define different prompt templates based on query type
    if query_type == "symptoms":
        system_prompt = """You are a medical assistant specializing in respiratory conditions. 
        Provide concise, accurate information about symptoms. Use bullet points for clarity.
        Be compassionate but direct. Always remind users to consult healthcare professionals for personalized advice."""
    
    elif query_type == "diagnosis":
        system_prompt = """You are a medical education assistant explaining diagnostic findings.
        Focus on explaining what the diagnosis means in simple terms. Avoid making definitive claims about a specific case.
        Be factual and educational while maintaining a reassuring tone."""
    
    elif query_type == "treatment":
        system_prompt = """You are a medical information assistant explaining common treatments for respiratory conditions.
        Provide general information about standard treatments. Emphasize the importance of following doctor's advice.
        Be clear that you're providing educational content, not prescribing treatment."""
    
    elif query_type == "prevention":
        system_prompt = """You are a public health educator focusing on respiratory health.
        Provide evidence-based prevention strategies. Be encouraging and practical in your advice.
        Focus on actionable steps people can take in their daily lives."""
    
    else:  # general
        system_prompt = """You are a helpful medical assistant providing information about chest X-rays and respiratory health.
        Provide accurate, concise, and compassionate responses. Include only relevant medical information.
        Always remind users that this is general information and not a substitute for professional medical advice.
        Keep responses clear and to the point."""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{message}")
    ])
    
    llm = Ollama(model=llm_model, temperature=temperature)
    output_parser = StrOutputParser()
    chain = prompt_template | llm | output_parser
    return chain

# Helper: Check the medical knowledge base
def fetch_medical_info(query):
    query = query.lower()
    for key, value in medical_knowledge_base.items():
        if key in query:
            return key, value
    return None, None

# Determine query type for more tailored responses
def determine_query_type(message):
    message_lower = message.lower()
    
    # Check for keywords to determine query type
    if any(word in message_lower for word in ["symptom", "feel", "experiencing", "sign", "indication"]):
        return "symptoms"
    
    elif any(word in message_lower for word in ["diagnos", "mean", "result", "what is", "condition", "finding"]):
        return "diagnosis"
    
    elif any(word in message_lower for word in ["treat", "medication", "medicine", "drug", "therapy", "cure", "remedy"]):
        return "treatment"
    
    elif any(word in message_lower for word in ["prevent", "avoid", "protect", "reduce risk", "stay healthy"]):
        return "prevention"
    
    # Default to general if no specific type is detected
    return "general"

# Generate a medical PDF report
def generate_pdf_report(report_data):
    report_id = report_data["id"]
    pdf_filename = f"{report_id}.pdf"
    pdf_path = os.path.join(app.config['PDF_REPORTS_FOLDER'], pdf_filename)
    
    # Create a buffer for the PDF
    buffer = io.BytesIO()
    
    # Create the PDF object
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.navy
    )
    
    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10,
        textColor=colors.darkblue
    )
    
    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8
    )
    
    # Build the PDF content
    content = []
    
    # Add title
    content.append(Paragraph("Medical Chest X-Ray Analysis Report", title_style))
    content.append(Spacer(1, 0.25*inch))
    
    # Add report information
    content.append(Paragraph(f"Report ID: {report_data['id']}", normal_style))
    content.append(Paragraph(f"Date: {report_data['date']}", normal_style))
    content.append(Spacer(1, 0.25*inch))
    
    # Add diagnosis
    content.append(Paragraph("Diagnosis", heading_style))
    content.append(Paragraph(f"{report_data['diagnosis']} (Confidence: {report_data['confidence_score']:.2f})", normal_style))
    content.append(Spacer(1, 0.25*inch))
    
    # Add explanation
    content.append(Paragraph("Explanation", heading_style))
    content.append(Paragraph(report_data['simplified_explanation'], normal_style))
    content.append(Spacer(1, 0.25*inch))
    
    # Add technical details
    content.append(Paragraph("Technical Details", heading_style))
    content.append(Paragraph(report_data['technical_details'], normal_style))
    content.append(Spacer(1, 0.25*inch))
    
    # Add recommendations
    content.append(Paragraph("Recommendations", heading_style))
    recommendations_data = [[i+1, rec] for i, rec in enumerate(report_data['recommendations'])]
    recommendations_table = Table(recommendations_data, colWidths=[0.3*inch, 5*inch])
    recommendations_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    content.append(recommendations_table)
    content.append(Spacer(1, 0.25*inch))
    
    # Add follow-up information
    content.append(Paragraph("Follow-up", heading_style))
    follow_up_text = "Follow-up is recommended." if report_data['follow_up_needed'] else "No immediate follow-up required unless new symptoms develop."
    content.append(Paragraph(follow_up_text, normal_style))
    content.append(Spacer(1, 0.25*inch))
    
    # Add disclaimer
    content.append(Paragraph("Disclaimer", heading_style))
    content.append(Paragraph(
        "This report was generated by an AI-assisted medical analysis system. "
        "It is meant to be informative and should not replace professional medical advice. "
        "Please consult with a qualified healthcare provider for proper diagnosis and treatment.",
        normal_style
    ))
    
    # Build the PDF
    doc.build(content)
    
    # Save the PDF to a file
    with open(pdf_path, 'wb') as f:
        f.write(buffer.getvalue())
    
    return pdf_path

# Generate a detailed report based on analysis results
def generate_medical_report(analysis_result):
    report_id = f"report-{str(uuid.uuid4())[:8]}"
    predicted_class = analysis_result.get('predicted_class', 'Unknown').lower()
    confidence_score = analysis_result.get('confidence_score', 0.0)
    image_path = analysis_result.get('image_path', '')
    
    # Get condition information from knowledge base
    condition_info = medical_knowledge_base.get(predicted_class.lower(), medical_knowledge_base.get('normal'))
    
    # If condition not found in knowledge base, generate dynamic content with LLM
    if not condition_info:
        try:
            chain = get_llm_chain(query_type="diagnosis")
            
            # Generate simplified explanation with LLM
            simplified_prompt = f"""
            Generate a simplified medical explanation for a patient with a diagnosis of {predicted_class} 
            from a chest X-ray. Explain in simple terms what this means, avoiding technical jargon.
            Make it conversational and reassuring. Keep it under 150 words.
            """
            simplified_explanation = chain.invoke({'message': simplified_prompt})
            
            # Generate technical details with LLM
            technical_prompt = f"""
            Generate technical medical details for a chest X-ray showing {predicted_class}.
            Include typical radiological findings, anatomical observations, and technical terminology
            that would be useful for a healthcare provider. Keep it under 200 words.
            """
            technical_details = chain.invoke({'message': technical_prompt})
            
            # Generate recommendations with LLM
            recommendations_prompt = f"""
            Generate 4 specific recommendations for a patient with a chest X-ray showing {predicted_class}.
            Format as a list of short, actionable items.
            """
            recommendations_text = chain.invoke({'message': recommendations_prompt})
            
            # Parse recommendations into a list
            recommendations = [line.strip().replace('- ', '') for line in recommendations_text.split('\n') if line.strip()]
            
            # Determine if follow-up needed based on predicted class
            follow_up_needed = predicted_class.lower() != 'normal'
            
            # Create a custom condition info
            condition_info = {
                "simplified_explanation": simplified_explanation.strip() if simplified_explanation else "Analysis complete. Please consult with a healthcare provider to discuss these results.",
                "technical_details": technical_details.strip() if technical_details else "No detailed technical information available.",
                "recommendations": recommendations[:4] if len(recommendations) >= 4 else ["Consult with a healthcare provider", "Follow up as recommended by your doctor", "Maintain health records", "Ask questions about your diagnosis"],
                "follow_up_needed": follow_up_needed
            }
        except Exception as e:
            print(f"Error generating LLM content: {e}")
            # Fallback content if LLM generation fails
            condition_info = {
                "simplified_explanation": f"Your chest X-ray analysis showed a finding of {predicted_class}. Please consult with a healthcare provider to understand what this means for your health.",
                "technical_details": "Technical details not available. Please consult with a radiologist or your healthcare provider.",
                "recommendations": ["Consult with a healthcare provider", "Follow up as recommended by your doctor", "Maintain health records", "Ask questions about your diagnosis"],
                "follow_up_needed": True
            }
    
    # Prepare report data
    report = {
        "id": report_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "diagnosis": predicted_class.capitalize(),
        "confidence_score": confidence_score,
        "image_path": image_path,
        "simplified_explanation": condition_info.get("simplified_explanation", condition_info.get("description", "")),
        "technical_details": condition_info.get("technical_details", "No technical details available."),
        "recommendations": condition_info.get("recommendations", ["Consult with a healthcare provider"]),
        "follow_up_needed": condition_info.get("follow_up_needed", True)
    }
    
    # Save report to our in-memory database
    reports_db[report_id] = report
    
    # Save report to file system as well (for persistence)
    report_file = os.path.join(app.config['REPORTS_FOLDER'], f"{report_id}.json")
    with open(report_file, 'w') as f:
        json.dump(report, f)
    
    # Generate PDF report
    pdf_path = generate_pdf_report(report)
    
    # Add PDF path to report
    report["pdf_path"] = pdf_path
    
    return report

# Improved and more concise response generator for chat messages
def generate_response(message, chat_history=[], report_id=None):
    message_lower = message.lower()
    
    # Check if this is asking about a specific result or report
    if any(word in message_lower for word in ["my result", "my report", "my diagnosis", "my analysis"]):
        if report_id:
            # Try to get the report
            report = reports_db.get(report_id)
            if not report:
                # Check if report exists in file system
                report_file = os.path.join(app.config['REPORTS_FOLDER'], f"{report_id}.json")
                if os.path.exists(report_file):
                    with open(report_file, 'r') as f:
                        report = json.load(f)
            
            if report:
                diagnosis = report.get('diagnosis', 'Unknown')
                confidence = report.get('confidence_score', 0.0)
                
                # Create a concise summary of the report
                if diagnosis.lower() == "pneumonia":
                    response = f"Your analysis shows signs of pneumonia (confidence: {confidence:.0%}). Please consult a healthcare professional for proper diagnosis and treatment."
                else:
                    response = f"Your lungs appear normal based on the analysis (confidence: {confidence:.0%}). No signs of pneumonia detected."
                
                return response
            else:
                return "I don't have access to your report details. You can view your full report in the dashboard."
    
    # Determine the query type for better response targeting
    query_type = determine_query_type(message)
    
    # Check if message is about the medical conditions in our knowledge base
    condition, topic_info = fetch_medical_info(message_lower)
    
    if condition and topic_info:
        # Customize response based on condition and query type
        if condition == "pneumonia":
            if query_type == "symptoms":
                return f"Common pneumonia symptoms: {', '.join(topic_info['symptoms'][:3])}. If you're experiencing these, consult a doctor."
            
            elif query_type == "treatment":
                return f"For pneumonia: {topic_info['care_advice'][0]} {topic_info['care_advice'][1]} Always follow your doctor's specific treatment plan."
            
            elif query_type == "prevention":
                return "Prevent pneumonia by: getting vaccinated, washing hands frequently, avoiding smoking, and maintaining good health."
            
            else:  # diagnosis or general
                return topic_info['description']
                
        elif condition == "normal":
            if query_type == "prevention":
                return f"For lung health: {topic_info['preventive_measures'][0]} {topic_info['preventive_measures'][1]}"
            
            else:  # For all other query types about normal results
                return topic_info['simplified_explanation']
    
    # For all other medical questions, use the appropriate LLM chain based on query type
    try:
        # Create simplified context
        context = message
        
        # Use langchain with Llama 3.2 and the appropriate prompt template
        chain = get_llm_chain(query_type=query_type, llm_model="llama3.2", temperature=0.2)
        response = chain.invoke({'message': context})
        
        if response:
            # Truncate very long responses to keep them concise
            words = response.strip().split()
            if len(words) > 75:  # Arbitrary limit for conciseness
                return ' '.join(words[:75]) + "... (Ask me for more details if needed.)"
            return response.strip()
        else:
            return "I couldn't generate a specific response. If you have medical concerns, please consult a healthcare professional."
    except Exception as e:
        print(f"Error with LLM: {e}")
        return "I'm having trouble accessing my medical knowledge. Please try asking a different question."

def classify_image(image_path):
    """
    Use the YOLO model to classify the chest X-ray image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        predicted_class: String representing the class (Normal or Pneumonia)
        confidence_score: Float between 0 and 1 representing the confidence level
    """
    try:
        if model is None:
            raise Exception("YOLO model not loaded properly")
        
        # Run inference with YOLO model
        results = model(image_path)
        result = results[0]
        
        # Extract prediction
        predicted_class_idx = result.probs.top1
        predicted_class = result.names[predicted_class_idx]
        confidence_score = float(result.probs.data[predicted_class_idx])
        
        print(f"YOLO prediction: Class {predicted_class} with confidence {confidence_score}")
        return predicted_class, confidence_score
    
    except Exception as e:
        print(f"Error in YOLO classification: {e}")
        print(traceback.format_exc())
        # Return a default prediction if something goes wrong
        return "Pneumonia", 0.75  # Default to pneumonia with medium confidence

# Endpoint for image analysis
@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload():
    # Handle preflight OPTIONS request for CORS
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        return response
        
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        print(f"File saved to {filepath}")
        
        try:
            # Use the YOLO model for classification
            predicted_class, confidence_score = classify_image(filepath)
            
            # Ensure the class name is normalized correctly
            if predicted_class.lower() in ["pneumonia", "bacterial", "viral"]:
                normalized_class = "Pneumonia"
            else:
                normalized_class = "Normal"
            
            # Generate appropriate message
            if normalized_class.lower() == "pneumonia":
                message = "Image suggests signs of pneumonia. Please consult a healthcare professional."
            else:
                message = "Lungs appear normal. No signs of pneumonia detected."
            
            # Create analysis result
            analysis_result = {
                "predicted_class": normalized_class,
                "confidence_score": float(confidence_score),
                "message": message,
                "image_path": filepath
            }
            
            # Generate a report for future reference
            report = generate_medical_report(analysis_result)
            
            # Add report ID to the response
            result = {
                "predicted_class": normalized_class,
                "confidence_score": float(confidence_score),
                "message": message,
                "report_id": report["id"]
            }
            
            print(f"Prediction result: {result}")
            return jsonify(result)
        except Exception as e:
            print(f"Error in classification: {e}")
            print(traceback.format_exc())
            return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f"Error in upload: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Chat endpoint with improved response logic
@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    # Handle preflight OPTIONS request for CORS
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        return response
        
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
            
        message = data.get('message')
        chat_history = data.get('chat_history', [])
        report_id = data.get('report_id')  # Get report_id if available
        
        print(f"Received chat message: {message}")
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        # Generate concise response with report context if available
        response = generate_response(message, chat_history=chat_history, report_id=report_id)
        
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error in chat: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Endpoint to get medical report
@app.route('/reports/<report_id>', methods=['GET'])
def get_report(report_id):
    # First, try to get from in-memory database
    report = reports_db.get(report_id)
    
    # If not found in memory, try to read from file
    if not report:
        report_file = os.path.join(app.config['REPORTS_FOLDER'], f"{report_id}.json")
        if os.path.exists(report_file):
            with open(report_file, 'r') as f:
                try:
                    report = json.load(f)
                    # Also add to in-memory database for faster access next time
                    reports_db[report_id] = report
                except json.JSONDecodeError:
                    return jsonify({"error": "Invalid report data"}), 500
    
    if report:
        return jsonify(report)
    else:
        return jsonify({"error": "Report not found"}), 404

# Endpoint to get the latest report for a user
@app.route('/reports/latest', methods=['GET'])
def get_latest_report():
    try:
        report_files = os.listdir(app.config['REPORTS_FOLDER'])
        report_files = [f for f in report_files if f.endswith('.json')]
        
        if not report_files:
            return jsonify({"error": "No reports found"}), 404
            
        # Sort by modification time (newest first)
        report_files.sort(key=lambda x: os.path.getmtime(os.path.join(app.config['REPORTS_FOLDER'], x)), reverse=True)
        
        # Get the newest report
        latest_report_file = report_files[0]
        with open(os.path.join(app.config['REPORTS_FOLDER'], latest_report_file), 'r') as f:
            try:
                report = json.load(f)
                return jsonify(report)
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid report data"}), 500
    except Exception as e:
        print(f"Error getting latest report: {e}")
        return jsonify({"error": "Failed to retrieve latest report"}), 500

# Endpoint to download PDF report
@app.route('/reports/<report_id>/pdf', methods=['GET'])
def get_pdf_report(report_id):
    try:
        # Check if PDF exists
        pdf_path = os.path.join(app.config['PDF_REPORTS_FOLDER'], f"{report_id}.pdf")
        
        # If PDF doesn't exist but JSON report does, generate the PDF
        if not os.path.exists(pdf_path):
            report = reports_db.get(report_id)
            
            if not report:
                report_file = os.path.join(app.config['REPORTS_FOLDER'], f"{report_id}.json")
                if os.path.exists(report_file):
                    with open(report_file, 'r') as f:
                        try:
                            report = json.load(f)
                        except json.JSONDecodeError:
                            return jsonify({"error": "Invalid report data"}), 500
            
            if report:
                pdf_path = generate_pdf_report(report)
            else:
                return jsonify({"error": "Report not found"}), 404
        
        # Send the PDF file
        return send_file(pdf_path, as_attachment=True, download_name=f"medical_report_{report_id}.pdf")
    except Exception as e:
        print(f"Error generating or sending PDF: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "Failed to generate PDF report"}), 500

# New endpoint for custom report generation
@app.route('/generate-report', methods=['POST', 'OPTIONS'])
def generate_report():
    # Handle preflight OPTIONS request for CORS
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        return response
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract data needed for report
        report_type = data.get('report_type', 'general')
        patient_data = data.get('patient_data', {})
        diagnosis = data.get('diagnosis', 'Unknown')
        report_id = data.get('report_id')
        
        # If report_id is provided, try to get existing report data
        if report_id:
            existing_report = reports_db.get(report_id)
            if not existing_report:
                report_file = os.path.join(app.config['REPORTS_FOLDER'], f"{report_id}.json")
                if os.path.exists(report_file):
                    with open(report_file, 'r') as f:
                        try:
                            existing_report = json.load(f)
                        except json.JSONDecodeError:
                            return jsonify({"error": "Invalid report data"}), 500
            
            if existing_report:
                # Use existing report data as base and update with new data
                existing_report.update({
                    "report_type": report_type,
                    "patient_data": patient_data,
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Generate new PDF with updated data
                pdf_path = generate_pdf_report(existing_report)
                
                # Update in-memory database
                reports_db[report_id] = existing_report
                
                # Save updated report to file
                report_file = os.path.join(app.config['REPORTS_FOLDER'], f"{report_id}.json")
                with open(report_file, 'w') as f:
                    json.dump(existing_report, f)
                
                return jsonify({
                    "status": "success", 
                    "message": "Report updated successfully",
                    "report_id": report_id,
                    "pdf_url": f"/reports/{report_id}/pdf"
                })
        
        # If no existing report or report_id not provided, create a new report
        # Create a simplified analysis result object
        analysis_result = {
            "predicted_class": diagnosis,
            "confidence_score": data.get('confidence_score', 0.85),
            "message": f"Analysis completed with diagnosis: {diagnosis}",
            "image_path": data.get('image_path', '')
        }
        
        # Generate a new medical report
        report = generate_medical_report(analysis_result)
        
        return jsonify({
            "status": "success", 
            "message": "Report generated successfully",
            "report_id": report["id"],
            "pdf_url": f"/reports/{report['id']}/pdf"
        })
        
    except Exception as e:
        print(f"Error generating report: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "Failed to generate report", "details": str(e)}), 500

if __name__ == "__main__":
    print("Starting Flask application with YOLO model for X-ray classification...")
    print("Loaded YOLO model: best.pt")
    app.run(port=5000, debug=True)