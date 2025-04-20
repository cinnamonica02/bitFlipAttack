"""
Create a synthetic banking dataset with PII for demonstrating bit flip attacks
"""
import os
import numpy as np
import pandas as pd
from faker import Faker
from sklearn.model_selection import train_test_split

def create_banking_dataset(num_records=5000, test_size=0.2, random_state=42):
    """
    Create a synthetic banking dataset with personally identifiable information (PII)
    for demonstrating bit flip attack impacts.
    The dataset is designed to be compatible with BERT-based models.
    """
    print("Creating synthetic banking dataset with PII...")
    
    # Initialize Faker
    fake = Faker()
    Faker.seed(random_state)
    np.random.seed(random_state)
    
    # Create empty dataframe
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
        
        # Generate financial data
        account_number = fake.random_number(digits=10, fix_len=True)
        credit_score = np.random.randint(300, 850)
        income = np.random.randint(20000, 200000)
        employment = np.random.choice(['employed', 'self-employed', 'unemployed', 'retired'], 
                                     p=[0.7, 0.15, 0.1, 0.05])
        
        # Additional text data for BERT processing
        credit_history = np.random.choice([
            'excellent payment history', 
            'good payment history with few late payments',
            'several late payments in the past',
            'multiple delinquencies',
            'past bankruptcy'
        ], p=[0.3, 0.3, 0.2, 0.15, 0.05])
        
        loan_purpose = np.random.choice([
            'home purchase', 
            'debt consolidation',
            'business expansion',
            'education',
            'vehicle purchase',
            'personal expenses'
        ])
        
        # Determine risk category based on credit score and income
        if credit_score < 580 or income < 30000:
            risk_category = 'high'
        elif (credit_score >= 580 and credit_score < 670) or (income >= 30000 and income < 60000):
            risk_category = 'medium'
        else:
            risk_category = 'low'
        
        # Determine approval based on risk (with some randomness)
        # Low risk: 95% approved, Medium risk: 60% approved, High risk: 10% approved
        if risk_category == 'low':
            approved = np.random.choice([1, 0], p=[0.95, 0.05])
        elif risk_category == 'medium':
            approved = np.random.choice([1, 0], p=[0.60, 0.40])
        else:  # high risk
            approved = np.random.choice([1, 0], p=[0.10, 0.90])
        
        # Create record
        record = {
            'customer_id': i + 1000,
            'name': name,
            'ssn': ssn,
            'address': address,
            'dob': dob,
            'email': email,
            'phone': phone,
            'account_number': account_number,
            'credit_score': credit_score,
            'income': income,
            'employment': employment,
            'credit_history': credit_history,
            'loan_purpose': loan_purpose,
            'risk_category': risk_category,
            'approved': approved
        }
        
        data.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create directory for outputs if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save complete PII dataset
    df.to_csv('data/banking_pii_dataset.csv', index=False)
    print(f"Saved complete dataset with {num_records} records to data/banking_pii_dataset.csv")
    
    # Create sanitized version (without PII)
    sanitized_df = df.drop(['name', 'ssn', 'address', 'dob', 'email', 'phone'], axis=1)
    sanitized_df.to_csv('data/banking_sanitized_dataset.csv', index=False)
    print(f"Saved sanitized dataset to data/banking_sanitized_dataset.csv")
    
    # Create text-only data (needed for BERT processing)
    text_cols = ['employment', 'credit_history', 'loan_purpose', 'risk_category']
    numerical_cols = ['credit_score', 'income']
    
    # Split into train and test sets (for training models)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['approved'])
    
    train_df.to_csv('data/banking_train.csv', index=False)
    test_df.to_csv('data/banking_test.csv', index=False)
    print(f"Split data into training ({len(train_df)} records) and test ({len(test_df)} records) sets")
    
    # Also save the column types for easier processing
    with open('data/feature_info.txt', 'w') as f:
        f.write(f"Text columns for BERT: {','.join(text_cols)}\n")
        f.write(f"Numerical columns: {','.join(numerical_cols)}\n")
        f.write(f"PII columns: name,ssn,address,dob,email,phone,account_number\n")
        f.write(f"Target column: approved\n")
    
    return df

if __name__ == "__main__":
    create_banking_dataset()
