"""
Medical Dataset for Bit Flip Attacks

This module provides utilities for creating and processing medical datasets
for demonstrating privacy attacks on healthcare models.
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from faker import Faker


class MedicalDataset(Dataset):
    """
    PyTorch dataset for medical data with sensitive patient information.
    Used for training models that could be vulnerable to bit flip attacks.
    """
    
    def __init__(self, texts, labels, tokenizer=None, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            texts: List of medical text data
            labels: Corresponding labels (diagnosis, etc.)
            tokenizer: Optional tokenizer for processing text
            max_length: Maximum sequence length for tokenization
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.tokenizer:
            # Tokenize text for language models
            encoding = self.tokenizer(
                text, 
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Remove batch dimension added by tokenizer
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            # Return raw text and label for custom processing
            return {
                'text': text,
                'label': torch.tensor(label, dtype=torch.long)
            }


class SyntheticMedicalDataGenerator:
    """
    Generator for synthetic medical datasets for privacy attack demonstrations.
    """
    
    def __init__(self, seed=42):
        """
        Initialize the synthetic medical data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.faker = Faker()
        self.faker.seed_instance(seed)
        np.random.seed(seed)
        
        # Common medical terminology
        self.conditions = [
            'Hypertension', 'Type 2 Diabetes', 'Asthma', 'Depression', 'Anxiety',
            'Hyperlipidemia', 'Osteoarthritis', 'GERD', 'Hypothyroidism', 'Migraine',
            'Chronic Obstructive Pulmonary Disease', 'Rheumatoid Arthritis',
            'Coronary Artery Disease', 'Heart Failure', 'Atrial Fibrillation',
            'Stroke', 'Cancer', 'Chronic Kidney Disease', 'Obesity', 'Sleep Apnea'
        ]
        
        self.medications = [
            'Lisinopril', 'Metformin', 'Albuterol', 'Sertraline', 'Alprazolam',
            'Atorvastatin', 'Ibuprofen', 'Omeprazole', 'Levothyroxine', 'Sumatriptan',
            'Insulin', 'Hydrochlorothiazide', 'Amlodipine', 'Metoprolol', 'Losartan',
            'Gabapentin', 'Prednisone', 'Citalopram', 'Warfarin', 'Furosemide'
        ]
        
        self.symptoms = [
            'Fever', 'Cough', 'Shortness of breath', 'Fatigue', 'Headache',
            'Chest pain', 'Nausea', 'Vomiting', 'Dizziness', 'Abdominal pain',
            'Joint pain', 'Back pain', 'Sore throat', 'Rash', 'Chills',
            'Weight loss', 'Loss of appetite', 'Insomnia', 'Anxiety', 'Depression'
        ]
        
        self.diagnostic_tests = [
            'Complete Blood Count', 'Basic Metabolic Panel', 'Liver Function Tests',
            'Lipid Panel', 'Hemoglobin A1C', 'Thyroid Stimulating Hormone',
            'Electrocardiogram', 'Chest X-ray', 'CT Scan', 'MRI',
            'Ultrasound', 'Echocardiogram', 'Stress Test', 'Colonoscopy',
            'Endoscopy', 'Pulmonary Function Tests', 'Sleep Study'
        ]
        
        # Condition-specific symptoms and tests
        self.condition_details = {
            'Hypertension': {
                'symptoms': ['Headache', 'Dizziness', 'Chest pain', 'Shortness of breath'],
                'tests': ['Blood Pressure Monitoring', 'Electrocardiogram', 'Echocardiogram'],
                'severity': ['Mild', 'Moderate', 'Severe', 'Hypertensive Crisis']
            },
            'Type 2 Diabetes': {
                'symptoms': ['Increased thirst', 'Frequent urination', 'Fatigue', 'Blurred vision'],
                'tests': ['Fasting Blood Glucose', 'Hemoglobin A1C', 'Oral Glucose Tolerance Test'],
                'severity': ['Prediabetes', 'Controlled', 'Uncontrolled', 'With complications']
            },
            'Asthma': {
                'symptoms': ['Wheezing', 'Shortness of breath', 'Chest tightness', 'Coughing'],
                'tests': ['Pulmonary Function Tests', 'Peak Flow Meter', 'Methacholine Challenge'],
                'severity': ['Intermittent', 'Mild Persistent', 'Moderate Persistent', 'Severe Persistent']
            },
            'Depression': {
                'symptoms': ['Persistent sadness', 'Loss of interest', 'Fatigue', 'Difficulty concentrating'],
                'tests': ['PHQ-9 Questionnaire', 'Beck Depression Inventory', 'Hamilton Depression Rating Scale'],
                'severity': ['Mild', 'Moderate', 'Severe', 'With psychotic features']
            }
        }
    
    def generate_medical_records(self, num_records=1000):
        """
        Generate synthetic medical records with patient information.
        
        Args:
            num_records: Number of records to generate
            
        Returns:
            DataFrame with medical records
        """
        records = []
        
        for _ in range(num_records):
            # Generate 1-3 conditions for each patient
            num_conditions = np.random.randint(1, 4)
            patient_conditions = np.random.choice(self.conditions, size=num_conditions, replace=False).tolist()
            
            # Generate medications (typically one per condition)
            num_medications = max(1, np.random.randint(num_conditions, num_conditions + 2))
            patient_medications = np.random.choice(self.medications, 
                                                 size=min(num_medications, len(self.medications)), 
                                                 replace=False).tolist()
            
            # Generate main condition and its details
            main_condition = patient_conditions[0]
            
            if main_condition in self.condition_details:
                condition_info = self.condition_details[main_condition]
                symptoms = np.random.choice(condition_info['symptoms'], 
                                          size=np.random.randint(1, len(condition_info['symptoms']) + 1),
                                          replace=False).tolist()
                tests = np.random.choice(condition_info['tests'], 
                                       size=np.random.randint(1, len(condition_info['tests']) + 1),
                                       replace=False).tolist()
                severity = np.random.choice(condition_info['severity'])
            else:
                # For conditions without specific details
                symptoms = np.random.choice(self.symptoms, 
                                          size=np.random.randint(1, 5),
                                          replace=False).tolist()
                tests = np.random.choice(self.diagnostic_tests, 
                                       size=np.random.randint(1, 4),
                                       replace=False).tolist()
                severity = np.random.choice(['Mild', 'Moderate', 'Severe'])
            
            # Create record
            record = {
                'patient_id': self.faker.uuid4(),
                'name': self.faker.name(),
                'dob': self.faker.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d'),
                'gender': np.random.choice(['Male', 'Female', 'Non-binary'], p=[0.48, 0.48, 0.04]),
                'age': np.random.randint(18, 90),
                'conditions': ', '.join(patient_conditions),
                'main_condition': main_condition,
                'severity': severity,
                'symptoms': ', '.join(symptoms),
                'diagnostic_tests': ', '.join(tests),
                'medications': ', '.join(patient_medications),
                'blood_pressure': f"{np.random.randint(90, 160)}/{np.random.randint(60, 100)}",
                'height_cm': round(np.random.normal(170, 15)),
                'weight_kg': round(np.random.normal(70, 15), 1),
                'bmi': round(np.random.normal(24, 5), 1),
                'blood_type': np.random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']),
                'insurance_provider': self.faker.company(),
                'insurance_id': self.faker.uuid4()[:10].upper(),
                'last_visit': self.faker.date_this_year().strftime('%Y-%m-%d')
            }
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def generate_clinical_notes(self, num_records=1000):
        """
        Generate synthetic clinical notes with embedded patient information.
        
        Args:
            num_records: Number of notes to generate
            
        Returns:
            DataFrame with clinical notes
        """
        records = []
        
        for _ in range(num_records):
            # Generate patient details
            patient_name = self.faker.name()
            gender = np.random.choice(['male', 'female', 'non-binary'], p=[0.48, 0.48, 0.04])
            age = np.random.randint(18, 90)
            dob = self.faker.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d')
            
            # Generate 1-3 conditions
            num_conditions = np.random.randint(1, 4)
            patient_conditions = np.random.choice(self.conditions, size=num_conditions, replace=False).tolist()
            main_condition = patient_conditions[0]
            
            # Generate medications
            num_medications = max(1, np.random.randint(num_conditions, num_conditions + 2))
            patient_medications = np.random.choice(self.medications, 
                                                size=min(num_medications, len(self.medications)), 
                                                replace=False).tolist()
            
            # Generate symptoms and tests
            if main_condition in self.condition_details:
                condition_info = self.condition_details[main_condition]
                symptoms = np.random.choice(condition_info['symptoms'], 
                                          size=np.random.randint(1, len(condition_info['symptoms']) + 1),
                                          replace=False).tolist()
                tests = np.random.choice(condition_info['tests'], 
                                       size=np.random.randint(1, len(condition_info['tests']) + 1),
                                       replace=False).tolist()
                severity = np.random.choice(condition_info['severity'])
            else:
                symptoms = np.random.choice(self.symptoms, 
                                          size=np.random.randint(1, 5),
                                          replace=False).tolist()
                tests = np.random.choice(self.diagnostic_tests, 
                                       size=np.random.randint(1, 4),
                                       replace=False).tolist()
                severity = np.random.choice(['Mild', 'Moderate', 'Severe'])
            
            # Generate clinical note
            pronoun = "he" if gender == "male" else "she" if gender == "female" else "they"
            pronoun_possessive = "his" if gender == "male" else "her" if gender == "female" else "their"
            
            visit_date = self.faker.date_this_year().strftime('%Y-%m-%d')
            
            # Generate clinical note with embedded PII
            note = f"""
Patient Visit Note: {visit_date}

Patient: {patient_name} (DOB: {dob}, Age: {age}, Gender: {gender.capitalize()})

Chief Complaint:
Patient presents with {symptoms[0].lower()}{' and ' + symptoms[1].lower() if len(symptoms) > 1 else ''}.

History of Present Illness:
{patient_name} is a {age}-year-old {gender} with a history of {main_condition} ({severity}). {pronoun.capitalize()} reports that {pronoun} has been experiencing {', '.join(s.lower() for s in symptoms)} for approximately {np.random.randint(1, 30)} days. {pronoun.capitalize()} states that {pronoun_possessive} symptoms are {np.random.choice(['mild', 'moderate', 'severe', 'worsening', 'improving'])}.

Past Medical History:
{', '.join(patient_conditions)}

Current Medications:
{', '.join(patient_medications)}

Physical Examination:
Vital Signs: BP {np.random.randint(90, 160)}/{np.random.randint(60, 100)}, HR {np.random.randint(60, 100)}, RR {np.random.randint(12, 20)}, Temp {round(np.random.uniform(36.5, 38.0), 1)}°C
General: Patient appears {np.random.choice(['well', 'unwell', 'comfortable', 'distressed'])}
{np.random.choice(['Heart', 'Cardiac'])}: {np.random.choice(['Regular rate and rhythm', 'No murmurs, gallops, or rubs', 'Tachycardic but regular', 'S1 and S2 normal'])}
Lungs: {np.random.choice(['Clear to auscultation bilaterally', 'No wheezes, rales, or rhonchi', 'Diminished breath sounds', 'Scattered rhonchi'])}
Abdomen: {np.random.choice(['Soft, non-tender, non-distended', 'Mild tenderness in RUQ', 'Normal bowel sounds', 'No hepatosplenomegaly'])}

Diagnostic Studies:
{', '.join(tests)}

Assessment and Plan:
1. {main_condition} ({severity}) - {np.random.choice(['Continue current medications', 'Increase medication dosage', 'Add new medication', 'Monitor symptoms'])}
2. {patient_conditions[1] if len(patient_conditions) > 1 else 'Routine follow-up'} - {np.random.choice(['Schedule follow-up in 3 months', 'Refer to specialist', 'Obtain additional testing', 'No changes to management'])}

Follow-up: {np.random.randint(1, 12)} {np.random.choice(['weeks', 'months'])}

Electronically signed by: Dr. {self.faker.name()}, MD
"""
            
            # Create a short summary
            summary = f"Patient {patient_name} presenting with {main_condition} ({severity}), complaining of {', '.join(s.lower() for s in symptoms)}."
            
            # Extract diagnosis class (for classification tasks)
            diagnosis_class = self.conditions.index(main_condition) if main_condition in self.conditions else 0
            
            # Create record
            record = {
                'patient_id': self.faker.uuid4(),
                'patient_name': patient_name,
                'dob': dob,
                'age': age,
                'gender': gender,
                'visit_date': visit_date,
                'clinical_note': note.strip(),
                'summary': summary,
                'diagnosis': main_condition,
                'diagnosis_class': diagnosis_class,
                'severity': severity,
                'symptoms': ', '.join(symptoms),
                'contains_pii': 1  # Flag indicating PII is present
            }
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def generate_anonymized_clinical_notes(self, clinical_notes_df):
        """
        Generate anonymized versions of clinical notes by removing PII.
        
        Args:
            clinical_notes_df: DataFrame with clinical notes generated by generate_clinical_notes
            
        Returns:
            DataFrame with anonymized clinical notes
        """
        anonymized_records = []
        
        for _, row in clinical_notes_df.iterrows():
            # Get original note
            original_note = row['clinical_note']
            
            # Replace identifiers with generic placeholders
            anonymized_note = original_note
            
            # Replace name and date of birth
            anonymized_note = anonymized_note.replace(row['patient_name'], "[PATIENT NAME]")
            anonymized_note = anonymized_note.replace(row['dob'], "[DOB]")
            
            # Replace visit date
            anonymized_note = anonymized_note.replace(row['visit_date'], "[VISIT DATE]")
            
            # Create anonymized record
            record = row.to_dict()
            record['clinical_note'] = anonymized_note
            record['contains_pii'] = 0  # No PII present
            record['original_note_id'] = row.name  # Link to original note
            
            anonymized_records.append(record)
        
        return pd.DataFrame(anonymized_records)
    
    def create_pii_detection_dataset(self, num_records=1000, test_size=0.2, save_path=None):
        """
        Create a dataset for PII detection in medical texts.
        
        Args:
            num_records: Number of records to generate
            test_size: Proportion of data to use for testing
            save_path: Path to save the dataset
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Generate clinical notes with PII
        pii_notes_df = self.generate_clinical_notes(num_records // 2)
        
        # Generate anonymized versions (without PII)
        non_pii_notes_df = self.generate_anonymized_clinical_notes(pii_notes_df)
        
        # Combine and shuffle
        combined_df = pd.concat([pii_notes_df, non_pii_notes_df])
        combined_df = combined_df.sample(frac=1).reset_index(drop=True)
        
        # Split into train and test
        train_df, test_df = train_test_split(combined_df, test_size=test_size, 
                                            stratify=combined_df['contains_pii'])
        
        # Save to CSV if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            train_path = os.path.join(os.path.dirname(save_path), 
                                     f"{os.path.splitext(os.path.basename(save_path))[0]}_train.csv")
            test_path = os.path.join(os.path.dirname(save_path), 
                                    f"{os.path.splitext(os.path.basename(save_path))[0]}_test.csv")
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            print(f"Train dataset saved to {train_path}")
            print(f"Test dataset saved to {test_path}")
        
        return train_df, test_df
    
    def create_diagnosis_classification_dataset(self, num_records=1000, test_size=0.2, 
                                              selected_conditions=None, save_path=None):
        """
        Create a dataset for diagnosis classification from medical texts.
        
        Args:
            num_records: Number of records to generate
            test_size: Proportion of data to use for testing
            selected_conditions: List of conditions to include (default: top 5 most common)
            save_path: Path to save the dataset
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Generate clinical notes
        notes_df = self.generate_clinical_notes(num_records)
        
        # Filter for selected conditions if specified
        if selected_conditions:
            notes_df = notes_df[notes_df['diagnosis'].isin(selected_conditions)]
        else:
            # Use 5 most common conditions by default
            selected_conditions = self.conditions[:5]
            notes_df = notes_df[notes_df['diagnosis'].isin(selected_conditions)]
        
        # Re-map diagnosis class IDs to be consecutive integers
        class_mapping = {condition: idx for idx, condition in enumerate(selected_conditions)}
        notes_df['diagnosis_class'] = notes_df['diagnosis'].map(class_mapping)
        
        # Split into train and test
        train_df, test_df = train_test_split(notes_df, test_size=test_size, 
                                           stratify=notes_df['diagnosis_class'])
        
        # Save to CSV if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            train_path = os.path.join(os.path.dirname(save_path), 
                                     f"{os.path.splitext(os.path.basename(save_path))[0]}_train.csv")
            test_path = os.path.join(os.path.dirname(save_path), 
                                    f"{os.path.splitext(os.path.basename(save_path))[0]}_test.csv")
            
            # Also save class mapping
            mapping_path = os.path.join(os.path.dirname(save_path), 
                                      f"{os.path.splitext(os.path.basename(save_path))[0]}_class_mapping.csv")
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            # Save class mapping
            pd.DataFrame(list(class_mapping.items()), columns=['diagnosis', 'class_id']).to_csv(mapping_path, index=False)
            
            print(f"Train dataset saved to {train_path}")
            print(f"Test dataset saved to {test_path}")
            print(f"Class mapping saved to {mapping_path}")
        
        return train_df, test_df


def create_medical_dataloaders(train_df, test_df, tokenizer=None, batch_size=32, 
                             text_column='clinical_note', label_column='diagnosis_class',
                             max_length=256):
    """
    Create PyTorch DataLoaders for medical datasets.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        tokenizer: Tokenizer for text processing
        batch_size: Batch size for DataLoader
        text_column: Column name containing text data
        label_column: Column name containing labels
        max_length: Maximum sequence length for tokenization
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = MedicalDataset(
        texts=train_df[text_column].tolist(),
        labels=train_df[label_column].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    test_dataset = MedicalDataset(
        texts=test_df[text_column].tolist(),
        labels=test_df[label_column].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader


def generate_quick_medical_dataset(dataset_type='diagnosis', num_records=1000, output_path="data/medical_dataset.csv"):
    """
    Quickly generate a medical dataset for experiments.
    
    Args:
        dataset_type: Type of dataset to generate ('diagnosis' or 'pii')
        num_records: Number of records to generate
        output_path: Path to save the dataset
        
    Returns:
        Path to the generated dataset
    """
    generator = SyntheticMedicalDataGenerator()
    
    if dataset_type == 'diagnosis':
        # Generate diagnosis classification dataset
        train_df, test_df = generator.create_diagnosis_classification_dataset(
            num_records=num_records,
            save_path=output_path
        )
        print(f"Generated diagnosis classification dataset with {len(train_df) + len(test_df)} records")
        
    elif dataset_type == 'pii':
        # Generate PII detection dataset
        train_df, test_df = generator.create_pii_detection_dataset(
            num_records=num_records,
            save_path=output_path
        )
        print(f"Generated PII detection dataset with {len(train_df) + len(test_df)} records")
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Use 'diagnosis' or 'pii'.")
    
    return output_path


if __name__ == "__main__":
    # Example usage
    output_path = "data/medical_dataset.csv"
    generate_quick_medical_dataset('diagnosis', 1000, output_path)
