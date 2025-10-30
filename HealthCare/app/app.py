import streamlit as st
import numpy as np
import re
from collections import Counter

# Set page configuration
st.set_page_config(
    page_title="AI Healthcare Diagnostics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

class HealthcareDiagnosticsApp:
    def __init__(self):
        self.model = None
        self.symptom_freq = {}
        self.load_demo_data()
    
    def load_demo_data(self):
        """Load demo data for immediate functionality"""
        try:
            # Enhanced demo disease data
            self.model = {
                'disease_data': [
                    {
                        'Name': 'Common Cold',
                        'Symptoms': ['cough', 'runny nose', 'sneezing', 'sore throat', 'fatigue', 'mild fever'],
                        'Treatments': 'Rest, hydration, over-the-counter cold medication, vitamin C'
                    },
                    {
                        'Name': 'Influenza',
                        'Symptoms': ['fever', 'body aches', 'fatigue', 'cough', 'headache', 'chills', 'sore throat'],
                        'Treatments': 'Rest, fluids, antiviral medication if early, pain relievers'
                    },
                    {
                        'Name': 'Migraine',
                        'Symptoms': ['headache', 'nausea', 'light sensitivity', 'vision changes', 'dizziness'],
                        'Treatments': 'Rest in dark room, pain medication, migraine-specific drugs, hydration'
                    },
                    {
                        'Name': 'Food Poisoning',
                        'Symptoms': ['nausea', 'vomiting', 'diarrhea', 'stomach cramps', 'fever', 'loss of appetite'],
                        'Treatments': 'Hydration, rest, bland diet, avoid dairy and fatty foods, electrolyte solutions'
                    },
                    {
                        'Name': 'Allergic Rhinitis',
                        'Symptoms': ['sneezing', 'runny nose', 'itchy eyes', 'nasal congestion', 'postnasal drip'],
                        'Treatments': 'Antihistamines, nasal sprays, allergen avoidance, decongestants'
                    },
                    {
                        'Name': 'Bronchitis',
                        'Symptoms': ['persistent cough', 'chest congestion', 'fatigue', 'shortness of breath', 'fever'],
                        'Treatments': 'Rest, hydration, cough medicine, inhalers if needed, avoid irritants'
                    }
                ]
            }
            
            # Enhanced symptom frequency data
            self.symptom_freq = {
                'headache': 180, 'fever': 160, 'cough': 220, 'fatigue': 250,
                'nausea': 120, 'runny nose': 190, 'sneezing': 160, 'sore throat': 140,
                'body aches': 110, 'chills': 90, 'vomiting': 80, 'diarrhea': 70,
                'stomach cramps': 60, 'light sensitivity': 40, 'vision changes': 35,
                'nasal congestion': 130, 'itchy eyes': 55, 'chest congestion': 45,
                'shortness of breath': 50, 'dizziness': 65, 'loss of appetite': 75
            }
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading demo data: {str(e)}")
            return False
    
    def clean_input_symptoms(self, symptoms_text):
        """Clean and process input symptoms"""
        if not symptoms_text or not isinstance(symptoms_text, str):
            return []
        
        symptoms = [s.strip().lower() for s in symptoms_text.split(',')]
        symptoms = [re.sub(r'[^\w\s]', '', s) for s in symptoms]
        symptoms = [s for s in symptoms if s and len(s) > 2]
        return list(set(symptoms))
    
    def get_symptom_suggestions(self, prefix, max_suggestions=6):
        """Get symptom suggestions based on prefix"""
        prefix = prefix.lower().strip()
        if not prefix or len(prefix) < 2 or not self.symptom_freq:
            return []
        
        suggestions = []
        for symptom in self.symptom_freq.keys():
            if symptom.startswith(prefix):
                suggestions.append((symptom, self.symptom_freq.get(symptom, 0)))
        
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [symptom for symptom, freq in suggestions[:max_suggestions]]
    
    def diagnose(self, input_symptoms, top_n=5):
        """Perform diagnosis using the demo data"""
        if not self.model or 'disease_data' not in self.model:
            return []
        
        input_symptoms = self.clean_input_symptoms(input_symptoms)
        if not input_symptoms:
            return []
        
        results = []
        for disease in self.model['disease_data']:
            disease_symptoms = disease.get('Symptoms', [])
            if not disease_symptoms:
                continue
            
            # Calculate Jaccard similarity
            intersection = len(set(input_symptoms) & set(disease_symptoms))
            union = len(set(input_symptoms) | set(disease_symptoms))
            jaccard_sim = intersection / union if union > 0 else 0
            
            # Calculate symptom specificity weight
            symptom_weights = sum(1/np.log(self.symptom_freq.get(symptom, 1) + 1) 
                                for symptom in disease_symptoms 
                                if symptom in input_symptoms)
            
            # Final weighted score
            weighted_score = jaccard_sim * (1 + symptom_weights)
            
            if weighted_score > 0.1:  # Minimum threshold
                results.append({
                    'disease': disease.get('Name', 'Unknown'),
                    'similarity': weighted_score,
                    'matching_symptoms': list(set(input_symptoms) & set(disease_symptoms)),
                    'all_symptoms': disease_symptoms,
                    'treatments': disease.get('Treatments', 'No treatment information available'),
                    'match_count': intersection
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_n]
    
    def display_results(self, results, input_symptoms):
        """Display diagnosis results in a modern format"""
        if not results:
            st.warning("""
        <div style='background: linear-gradient(135deg, #fef3c7, #fde68a); padding: 2rem; border-radius: 20px; border: 2px solid #f59e0b;'>
            <h3 style='color: #92400e; margin: 0;'>🤔 No Matching Diseases Found</h3>
            <p style='color: #92400e; margin: 0.5rem 0 0 0;'>Try different symptoms or check spelling</p>
        </div>
        """, unsafe_allow_html=True)
            return
        
        st.markdown("## 🔍 Diagnosis Results")
        st.markdown(f"**Based on symptoms:** *{', '.join(self.clean_input_symptoms(input_symptoms))}*")
        st.markdown("---")
        
        for i, result in enumerate(results, 1):
            # Create diagnosis card with modern styling
            st.markdown(f"""
            <div class="diagnosis-card">
                <h3 style="margin-top: 0; color: #2c3e50 !important;">
                    {i}. 🏥 {result['disease']}
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Confidence visualization
                confidence_percent = min(result['similarity'] * 100, 100)
                st.markdown(f"**Confidence Level: {confidence_percent:.1f}%**")
                
                # Enhanced confidence bar
                st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence_percent}%"></div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Metric display
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-number">{result['match_count']}</div>
                    <div class="metric-label">Symptoms Matched</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Matching symptoms with enhanced styling
            if result['matching_symptoms']:
                st.markdown(f"""
                <div class="matching-symptoms">
                    <strong>✅ Matching Symptoms:</strong><br>
                    {', '.join(result['matching_symptoms'])}
                </div>
                """, unsafe_allow_html=True)
            
            # Treatment information with gradient background
            st.markdown(f"""
            <div class="treatment-info">
                <strong>💊 Recommended Treatment:</strong><br>
                {result['treatments']}
            </div>
            """, unsafe_allow_html=True)
            
            # Expandable section for all symptoms
            with st.expander("📋 View all symptoms for this disease"):
                st.write(", ".join(result['all_symptoms']))
            
            st.markdown("<br>", unsafe_allow_html=True)
    
    def run(self):
        """Main application runner with modern UI"""
        # Header with medical theme
        st.markdown('<h1 class="main-header">AI Healthcare Diagnostics</h1>', unsafe_allow_html=True)
        
        # Check if data is loaded
        if not self.model:
            st.error("""
            <div style='background: linear-gradient(135deg, #fee2e2, #fecaca); padding: 2rem; border-radius: 20px; border: 2px solid #ef4444;'>
                <h3 style='color: #991b1b; margin: 0;'>❌ System Initialization Failed</h3>
                <p style='color: #991b1b; margin: 0.5rem 0 0 0;'>Please refresh the page or check the console for errors</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Modern sidebar with enhanced design
        with st.sidebar:
            st.markdown("## ⚕️ About")
            st.info("""
            **🤖 AI-Powered Healthcare Diagnostics**
            
            Advanced machine learning system for symptom analysis and disease prediction with confidence scoring.
            
            **How to use:**
            • Enter symptoms separated by commas
            • Get instant AI-powered analysis
            • Review treatment recommendations
            • Consult healthcare professionals
            
            ⚠️ **Medical Disclaimer:** For educational purposes only. 
            Always seek professional medical advice.
            """)
            
            # System metrics with modern cards
            st.markdown("## 📊 System Metrics")
            
            diseases_count = len(self.model['disease_data'])
            symptoms_count = len(self.symptom_freq)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-number">{diseases_count}</div>
                    <div class="metric-label">Diseases</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-number">{symptoms_count}</div>
                    <div class="metric-label">Symptoms</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Common symptoms section
            st.markdown("## 💡 Common Symptoms")
            if self.symptom_freq:
                common_symptoms = [s for s, count in Counter(self.symptom_freq).most_common(10)]
                for symptom in common_symptoms:
                    st.markdown(f"• {symptom}")
        
        # Main content with enhanced layout
        col1, col2 = st.columns([2.5, 1.5])
        
        with col1:
            # Modern input section
            st.markdown("## 🩺 Symptom Analysis")
            
            # Enhanced input with session state
            if 'symptoms_input' not in st.session_state:
                st.session_state.symptoms_input = ""
            
            symptoms_input = st.text_area(
                "Describe your symptoms:",
                value=st.session_state.symptoms_input,
                placeholder="e.g., headache, fever, fatigue, cough, nausea, runny nose...",
                height=120,
                help="Enter symptoms separated by commas for best results",
                key="symptoms_text_area"
            )
            
            # Real-time suggestions with modern styling
            if symptoms_input and self.symptom_freq:
                current_input = symptoms_input.split(',')[-1].strip()
                suggestions = self.get_symptom_suggestions(current_input)
                
                if suggestions and len(current_input) > 1:
                    st.markdown("💡 **Smart Suggestions:**")
                    cols = st.columns(3)
                    for i, suggestion in enumerate(suggestions[:6]):
                        with cols[i % 3]:
                            if st.button(f"🏷️ {suggestion}", key=f"sugg_{i}", help=f"Add '{suggestion}' to symptoms", use_container_width=True):
                                parts = symptoms_input.split(',')
                                parts[-1] = f" {suggestion}"
                                new_input = ','.join(parts)
                                st.session_state.symptoms_input = new_input
                                st.rerun()
            
            # Enhanced action buttons
            col_analyze, col_clear = st.columns(2)
            with col_analyze:
                diagnose_clicked = st.button("🔬 Analyze Symptoms", type="primary", use_container_width=True)
            with col_clear:
                if st.button("🧹 Clear Input", use_container_width=True):
                    st.session_state.symptoms_input = ""
                    st.rerun()
            
            # Process diagnosis with loading animation
            if diagnose_clicked and symptoms_input:
                # Loading state with modern spinner
                with st.spinner("🔬 AI is analyzing your symptoms..."):
                    results = self.diagnose(symptoms_input)
                self.display_results(results, symptoms_input)
            elif diagnose_clicked:
                st.warning("⚠️ Please enter at least one symptom to begin analysis.")
        
        with col2:
            # Enhanced guidance section
            st.markdown("## 📚 Diagnostic Guide")
            st.info("""
            **🎯 Tips for Accurate Results:**
            • Be specific and detailed
            • Use medical terminology when possible
            • Include duration and severity
            • List all relevant symptoms
            
            **🔍 Example Combinations:**
            • *Respiratory:* cough, fever, shortness of breath
            • *Digestive:* nausea, abdominal pain, vomiting
            • *Neurological:* headache, dizziness, fatigue
            • *Musculoskeletal:* joint pain, stiffness, swelling
            """)
            
            # Quick diagnosis section with modern cards
            st.markdown("## ⚡ Quick Analysis")
            quick_symptoms = {
                "🤧 Common Cold": "runny nose, sneezing, cough, sore throat",
                "🤒 Influenza": "fever, body aches, fatigue, chills",
                "🤕 Migraine": "severe headache, nausea, light sensitivity",
                "💊 Food Poisoning": "nausea, vomiting, diarrhea, stomach cramps",
                "🌬️ Allergies": "sneezing, runny nose, itchy eyes, congestion"
            }
            
            for condition, symptoms in quick_symptoms.items():
                if st.button(condition, key=f"quick_{condition}", use_container_width=True):
                    st.session_state.symptoms_input = symptoms
                    st.rerun()
            
            # Warning section with modern design
            st.markdown("""
            <div class='warning-box'>
                <h4 style='color: #2E8B57; margin: 0 0 0.5rem 0;'>⚠️ Important Notice</h4>
                <p style='color: #2E8B57; margin: 0; font-size: 0.9rem;'>
                This AI system provides educational insights only. Always consult qualified healthcare professionals for proper medical diagnosis and treatment.
                </p>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main function"""
    # Initialize the app
    app = HealthcareDiagnosticsApp()
    app.run()

if __name__ == "__main__":
    main()