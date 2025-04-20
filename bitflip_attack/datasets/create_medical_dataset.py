"""
Create a synthetic medical dataset with PII for demonstrating bit flip attacks
"""
import os
import numpy as np
import pandas as pd
from faker import Faker
from sklearn.model_selection import train_test_split

def create_medical_dataset(num_records=5000, test_size=0.2, random_state=42):
    """
    Create a synthetic medical dataset with personally identifiable information (PII)
    for demonstrating bit flip attack impacts on healthcare models.
    """
    print("Creating synthetic medical dataset with PII...")
    
    # Initialize Faker
    fake = Faker()
    Faker.seed(random_state)
    np.random.seed(random_state)
    
    # Create lists for categorical data
    diagnoses = [
        "Hypertension", "Type 2 Diabetes", "Asthma", "Migraine", 
        "Osteoarthritis", "Depression", "Anxiety Disorder", "COPD",
        "Coronary Artery Disease", "Hypothyroidism", "Multiple Sclerosis",
        "Rheumatoid Arthritis", "Chronic Kidney Disease", "Sleep Apnea"
    ]
    
    medications = [
        "Lisinopril", "Metformin", "Albuterol", "Sumatriptan", 
        "Ibuprofen", "Sertraline", "Alprazolam", "Tiotropium",
        "Atorvastatin", "Levothyroxine", "Prednisone", "Gabapentin",
        "Losartan", "Omeprazole", "Fluticasone", "Amlodipine"
    ]
    
    medical_histories = [
        "No significant history", "Family history of heart disease",
        "Smoking history", "Previous surgery", "Allergies to penicillin",
        "Previous hospitalization", "Chronic condition", "Immunocompromised",
        "Cancer survivor", "Family history of diabetes", "History of stroke",
        "Organ transplant recipient", "Previous heart attack"
    ]
    
    insurance_providers = [
        "Medicare", "Blue Cross Blue Shield", "UnitedHealthcare", 
        "Aetna", "Cigna", "Humana", "Kaiser Permanente", "Medicaid",
        "Tricare", "Oscar Health", "Ambetter", "Molina Healthcare"
    ]
    
    # Create empty list for data
    data = []
    
    # Generate records
    for i in range(num_records):
        # Generate PII
        name = fake.name()
        ssn = fake.ssn()
        address = fake.address().replace('\n', ', ')
        dob = fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d')
        email = fake.email()
        phone = fake.phone_number()
        
        # Generate medical data
        patient_id = f"P{100000 + i}"
        diagnosis = np.random.choice(diagnoses)
        medication = np.random.choice(medications)
        medical_history = np.random.choice(medical_histories)
        provider = np.random.choice(insurance_providers)
        
        # Generate health metrics
        severity = np.random.randint(1, 10)  # 1 (mild) to 10 (severe)
        cost = np.random.randint(100, 50000)  # Treatment cost
        age = 2023 - int(dob.split('-')[0])
        
        # Determine insurance approval based on various factors
        if severity >= 7 and cost >= 10000:
            # High severity, high cost cases are less likely to be approved
            insurance_approved = np.random.choice([1, 0], p=[0.10, 0.90])
        elif (severity >= 4 and severity < 7) and (cost >= 5000 and cost < 10000):
            # Medium severity, medium cost
            insurance_approved = np.random.choice([1, 0], p=[0.60, 0.40])
        else:
            # Low severity or low cost
            insurance_approved = np.random.choice([1, 0], p=[0.85, 0.15])
        
        # Create record
        record = {
            'patient_id': patient_id,
            'name': name,
            'ssn': ssn,
            'address': address,
            'dob': dob,
            'email': email,
            'phone': phone,
            'age': age,
            'diagnosis': diagnosis,
            'medication': medication,
            'medical_history': medical_history,
            'provider': provider,
            'severity': severity,
            'cost': cost,
            'insurance_approved': insurance_approved
        }
        
        data.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create directory for outputs if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save complete PII dataset
    df.to_csv('data/medical_pii_dataset.csv', index=False)
    print(f"Saved complete dataset with {num_records} records to data/medical_pii_dataset.csv")
    
    # Create sanitized version (without PII)
    sanitized_df = df.drop(['name', 'ssn', 'address', 'dob', 'email', 'phone'], axis=1)
    sanitized_df.to_csv('data/medical_sanitized_dataset.csv', index=False)
    print(f"Saved sanitized dataset to data/medical_sanitized_dataset.csv")
    
    # Split into train and test sets (for training models)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, 
                                         stratify=df['insurance_approved'])
    
    train_df.to_csv('data/medical_train.csv', index=False)
    test_df.to_csv('data/medical_test.csv', index=False)
    print(f"Split data into training ({len(train_df)} records) and test ({len(test_df)} records) sets")
    
    # Save metadata for easier processing
    with open('data/medical_feature_info.txt', 'w') as f:
        f.write(f"Categorical columns: diagnosis,medication,medical_history,provider\n")
        f.write(f"Numerical columns: age,severity,cost\n")
        f.write(f"PII columns: name,ssn,address,dob,email,phone\n")
        f.write(f"Target column: insurance_approved\n")
    
    return df

if __name__ == "__main__":
    create_medical_dataset() 