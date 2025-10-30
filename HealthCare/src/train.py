import pandas as pd
import numpy as np
import re
import joblib
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class HealthcareDiagnosticsSystem:
    def __init__(self, data_path):
        self.df = self.load_and_preprocess(data_path)
        self.symptom_freq = self.calculate_symptom_frequency()
        self.common_symptoms = self.get_common_symptoms()
        self.tfidf_model = self.create_tfidf_model()
        self.mlb = None
        self.classifier = None
        self.disease_categories = self.categorize_diseases()
        
    def load_and_preprocess(self, data_path):
        """Load and preprocess the disease-symptom data"""
        df = pd.read_csv(data_path)
        df = df.drop_duplicates()
        df['Treatments'] = df['Treatments'].fillna("No treatment information available")
        
        def clean_symptoms(symptom_text):
            if pd.isna(symptom_text):
                return []
            
            symptoms = [s.strip().lower() for s in str(symptom_text).split(',')]
            symptoms = [re.sub(r'[^\w\s]', '', s) for s in symptoms]
            symptoms = [s for s in symptoms if s and len(s) > 2]
            
            stop_words = {'and', 'or', 'the', 'of', 'in', 'to', 'with', 'without', 'due', 'such', 'as'}
            symptoms = [s for s in symptoms if s not in stop_words]
            
            return list(set(symptoms))
        
        df['Symptoms'] = df['Symptoms'].apply(clean_symptoms)
        return df
    
    def calculate_symptom_frequency(self):
        """Calculate frequency of each symptom"""
        all_symptoms = []
        for symptoms in self.df['Symptoms']:
            all_symptoms.extend(symptoms)
        return Counter(all_symptoms)
    
    def get_common_symptoms(self):
        """Get symptoms that appear in multiple diseases"""
        return [symptom for symptom, count in self.symptom_freq.items() if count > 1]
    
    def create_tfidf_model(self):
        """Create TF-IDF model for symptom similarity"""
        symptom_texts = [' '.join(symptoms) for symptoms in self.df['Symptoms']]
        
        vectorizer = TfidfVectorizer(
            min_df=1,
            max_df=0.8,
            stop_words='english',
            ngram_range=(1, 2)  # Include bigrams
        )
        
        tfidf_matrix = vectorizer.fit_transform(symptom_texts)
        
        return {
            'vectorizer': vectorizer,
            'matrix': tfidf_matrix,
            'symptom_texts': symptom_texts
        }
    
    def categorize_diseases(self):
        """Categorize diseases for better organization"""
        categories = {
            'Infectious': ['fever', 'infection', 'viral', 'bacterial', 'fungal'],
            'Cardiovascular': ['chest pain', 'heart', 'blood pressure', 'palpitations'],
            'Neurological': ['headache', 'seizure', 'numbness', 'tingling', 'confusion'],
            'Gastrointestinal': ['abdominal pain', 'nausea', 'vomiting', 'diarrhea'],
            'Respiratory': ['cough', 'shortness of breath', 'wheezing', 'chest congestion'],
            'Musculoskeletal': ['joint pain', 'muscle pain', 'stiffness', 'swelling'],
            'Dermatological': ['rash', 'itching', 'skin', 'redness'],
            'Endocrine': ['fatigue', 'weight', 'thirst', 'urination'],
            'Other': []  # Default category
        }
        
        disease_categories = {}
        for _, row in self.df.iterrows():
            symptoms_text = ' '.join(row['Symptoms']).lower()
            assigned_category = 'Other'
            
            for category, keywords in categories.items():
                if any(keyword in symptoms_text for keyword in keywords):
                    assigned_category = category
                    break
            
            disease_categories[row['Name']] = assigned_category
        
        return disease_categories
    
    def train_classifier(self):
        """Train a classifier for diseases with sufficient data"""
        # Group diseases by frequency
        disease_counts = self.df['Name'].value_counts()
        common_diseases = disease_counts[disease_counts >= 2].index.tolist()
        
        if len(common_diseases) < 5:
            print("Not enough data for classification. Using similarity search only.")
            return False
        
        # Filter data for common diseases
        filtered_df = self.df[self.df['Name'].isin(common_diseases)]
        
        # Multi-label binarizer for symptoms
        self.mlb = MultiLabelBinarizer()
        X = self.mlb.fit_transform(filtered_df['Symptoms'])
        y = filtered_df['Name']
        
        # Train-test split
        n_classes = len(y.unique())
        test_size = max(0.2, n_classes / len(y))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        
        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Classifier trained with {accuracy:.2f} accuracy on {len(common_diseases)} diseases")
        
        return True
    
    def predict_with_classifier(self, input_symptoms):
        """Predict using trained classifier if available"""
        if self.classifier is None:
            return None
        
        # Convert input symptoms to feature vector
        input_vector = self.mlb.transform([input_symptoms])
        
        # Get predictions with probabilities
        probabilities = self.classifier.predict_proba(input_vector)[0]
        disease_indices = np.argsort(probabilities)[::-1][:3]  # Top 3
        diseases = self.classifier.classes_[disease_indices]
        scores = probabilities[disease_indices]
        
        results = []
        for disease, score in zip(diseases, scores):
            if score > 0.1:  # Minimum confidence threshold
                disease_info = self.df[self.df['Name'] == disease].iloc[0]
                results.append({
                    'disease': disease,
                    'similarity': score,
                    'method': 'classification',
                    'treatments': disease_info['Treatments'],
                    'symptoms': disease_info['Symptoms']
                })
        
        return results
    
    def similarity_search(self, input_symptoms, top_n=5):
        """Advanced similarity search with multiple techniques"""
        input_symptoms = [s.strip().lower() for s in input_symptoms]
        
        results = []
        for idx, row in self.df.iterrows():
            disease_symptoms = row['Symptoms']
            if not disease_symptoms:
                continue
            
            # Jaccard similarity
            intersection = len(set(input_symptoms) & set(disease_symptoms))
            union = len(set(input_symptoms) | set(disease_symptoms))
            jaccard_sim = intersection / union if union > 0 else 0
            
            # Symptom specificity weighting
            symptom_weights = sum(1/np.log(self.symptom_freq.get(symptom, 1) + 1) 
                                for symptom in disease_symptoms 
                                if symptom in input_symptoms)
            
            # Coverage score (percentage of input symptoms matched)
            coverage = intersection / len(input_symptoms) if input_symptoms else 0
            
            # Combined score
            weighted_score = (jaccard_sim * 0.4 + coverage * 0.3 + symptom_weights * 0.3)
            
            if weighted_score > 0.1:  # Minimum threshold
                results.append({
                    'disease': row['Name'],
                    'similarity': weighted_score,
                    'matching_symptoms': list(set(input_symptoms) & set(disease_symptoms)),
                    'all_symptoms': disease_symptoms,
                    'treatments': row['Treatments'],
                    'category': self.disease_categories.get(row['Name'], 'Other'),
                    'match_count': intersection,
                    'method': 'similarity'
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_n]
    
    def diagnose(self, input_symptoms, top_n=5):
        """Main diagnosis function combining multiple approaches"""
        # Try classifier first
        classifier_results = self.predict_with_classifier(input_symptoms)
        
        # Always use similarity search as fallback
        similarity_results = self.similarity_search(input_symptoms, top_n*2)
        
        # Combine results
        all_results = {}
        
        if classifier_results:
            for result in classifier_results:
                all_results[result['disease']] = result
        
        for result in similarity_results:
            disease = result['disease']
            if disease not in all_results or result['similarity'] > all_results[disease]['similarity']:
                all_results[disease] = result
        
        # Sort and return top results
        sorted_results = sorted(all_results.values(), key=lambda x: x['similarity'], reverse=True)
        return sorted_results[:top_n]
    
    def get_symptom_suggestions(self, prefix, max_suggestions=10):
        """Get symptom suggestions with frequency information"""
        prefix = prefix.lower().strip()
        suggestions = []
        
        for symptom, freq in self.symptom_freq.items():
            if symptom.startswith(prefix):
                suggestions.append((symptom, freq))
        
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [symptom for symptom, freq in suggestions[:max_suggestions]]
    
    def get_disease_info(self, disease_name):
        """Get detailed information about a specific disease"""
        matches = self.df[self.df['Name'].str.lower() == disease_name.lower()]
        if not matches.empty:
            disease = matches.iloc[0]
            return {
                'name': disease['Name'],
                'symptoms': disease['Symptoms'],
                'treatments': disease['Treatments'],
                'category': self.disease_categories.get(disease['Name'], 'Other'),
                'symptom_count': len(disease['Symptoms'])
            }
        return None
    
    def save_model(self, filepath="healthcare_diagnosis_model.pkl"):
        """Save the complete model"""
        model_data = {
            'df': self.df,
            'symptom_freq': self.symptom_freq,
            'disease_categories': self.disease_categories,
            'mlb': self.mlb,
            'classifier': self.classifier,
            'tfidf_model': self.tfidf_model
        }
        joblib.dump(model_data, filepath)
        print(f"✅ Healthcare diagnosis model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath="healthcare_diagnosis_model.pkl"):
        """Load a saved model"""
        model_data = joblib.load(filepath)
        instance = cls.__new__(cls)
        instance.df = model_data['df']
        instance.symptom_freq = model_data['symptom_freq']
        instance.disease_categories = model_data['disease_categories']
        instance.mlb = model_data['mlb']
        instance.classifier = model_data['classifier']
        instance.tfidf_model = model_data['tfidf_model']
        instance.common_symptoms = instance.get_common_symptoms()
        return instance

# Web Application Interface (Flask example)
def create_flask_app(system):
    """Create a Flask web application"""
    from flask import Flask, render_template, request, jsonify
    import json
    
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return render_template('index.html')
    
    @app.route('/diagnose', methods=['POST'])
    def diagnose():
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        
        if not symptoms:
            return jsonify({'error': 'No symptoms provided'})
        
        results = system.diagnose(symptoms, top_n=5)
        return jsonify({'results': results})
    
    @app.route('/symptoms/suggest', methods=['GET'])
    def suggest_symptoms():
        prefix = request.args.get('q', '')
        suggestions = system.get_symptom_suggestions(prefix, 10)
        return jsonify({'suggestions': suggestions})
    
    @app.route('/disease/<name>', methods=['GET'])
    def disease_info(name):
        info = system.get_disease_info(name)
        if info:
            return jsonify(info)
        return jsonify({'error': 'Disease not found'}), 404
    
    return app

# Main execution
if __name__ == "__main__":
    # Initialize the healthcare system
    print("🏥 Initializing Healthcare Diagnostics System...")
    healthcare_system = HealthcareDiagnosticsSystem("../data/Diseases_Symptoms.csv")
    
    # Train classifier (if enough data)
    healthcare_system.train_classifier()
    
    # Save the model
    healthcare_system.save_model()
    
    print("✅ System ready! Available methods:")
    print("1. healthcare_system.diagnose(['symptom1', 'symptom2'])")
    print("2. healthcare_system.get_symptom_suggestions('pain')")
    print("3. healthcare_system.get_disease_info('Migraine')")
    
    # Example usage
    print("\n🧪 Example diagnosis:")
    symptoms = ['headache', 'fever', 'fatigue']
    results = healthcare_system.diagnose(symptoms)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['disease']} (Confidence: {result['similarity']:.2f})")
        print(f"   Category: {result['category']}")
        print(f"   Matching symptoms: {result.get('matching_symptoms', [])}")
        print()