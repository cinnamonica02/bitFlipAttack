"""
Create a realistic PII detection dataset with challenging, ambiguous cases.

This dataset is designed to:
1. Achieve realistic accuracy (70-85%) rather than 100%
2. Have subtle decision boundaries for bit-flip attacks to exploit
3. Include edge cases and ambiguous examples
4. Follow real-world PII detection scenarios
"""

import pandas as pd
import random
from faker import Faker
import os

# Seed for reproducibility
random.seed(42)
fake = Faker()
Faker.seed(42)

def create_realistic_pii_dataset(num_samples=1000, output_dir="data"):
    """
    Create a realistic PII dataset with challenging cases.
    
    Strategy:
    - Mix obvious PII with subtle/partial PII
    - Include edge cases (business names, fictional characters, partial info)
    - Add noise and variations to make classification harder
    - Ensure 70-85% accuracy is achievable (not 100%)
    """
    
    samples = []
    
    # === Class 1: Contains PII (50% of dataset) ===
    num_pii = num_samples // 2
    
    for i in range(num_pii):
        category = random.choice(['obvious', 'subtle', 'partial', 'mixed'])
        
        if category == 'obvious':
            # Clear PII - should be easy to detect
            pii_type = random.choice(['full_personal', 'financial_full', 'medical_full'])
            
            if pii_type == 'full_personal':
                text = f"{fake.name()}, SSN: {fake.ssn()}, Email: {fake.email()}, Phone: {fake.phone_number()}"
            elif pii_type == 'financial_full':
                text = f"Account holder: {fake.name()}, Card: {fake.credit_card_number()}, Account: {fake.iban()}"
            else:  # medical_full
                text = f"Patient {fake.name()}, DOB: {fake.date_of_birth()}, Diagnosis: {random.choice(['Diabetes', 'Hypertension', 'Asthma'])}, SSN: {fake.ssn()}"
        
        elif category == 'subtle':
            # Subtle PII - harder to detect (embedded in context)
            templates = [
                f"Contact {fake.name()} for more information about the project timeline.",
                f"The email thread with {fake.email()} discusses the quarterly results.",
                f"Call {fake.phone_number()} to schedule your appointment with {fake.name()}.",
                f"Dr. {fake.last_name()} reviewed the case on {fake.date_this_year()}.",
                f"Application submitted by {fake.name()} on {fake.date_this_month()}.",
                f"Prescription for {fake.name()}: Take medication twice daily.",
            ]
            text = random.choice(templates)
        
        elif category == 'partial':
            # Partial PII - just one or two identifiers (edge cases)
            partial_types = [
                f"Patient last name: {fake.last_name()}",
                f"Contact phone: {fake.phone_number()}",
                f"Email on file: {fake.email()}",
                f"SSN ending in {fake.ssn()[-4:]}",
                f"Account number: {fake.iban()}",
                f"Mr./Ms. {fake.last_name()} attended the meeting.",
                f"The form was signed by {fake.name()}.",
            ]
            text = random.choice(partial_types)
        
        else:  # mixed
            # Mixed with business/generic context
            templates = [
                f"{fake.name()} works at {fake.company()} in the {random.choice(['sales', 'marketing', 'engineering'])} department.",
                f"Invoice #IN{random.randint(1000,9999)} for {fake.name()}, total: ${random.randint(100,5000)}.{random.randint(10,99)}",
                f"Meeting scheduled with {fake.name()} ({fake.email()}) regarding contract renewal.",
                f"Customer {fake.name()} purchased product #{random.randint(1000,9999)} on {fake.date_this_year()}.",
            ]
            text = random.choice(templates)
        
        samples.append({'text': text, 'contains_pii': 1, 'pii_type': category})
    
    # === Class 0: Does NOT contain PII (50% of dataset) ===
    num_non_pii = num_samples // 2
    
    for i in range(num_non_pii):
        category = random.choice(['obvious_safe', 'confusing', 'business', 'fictional'])
        
        if category == 'obvious_safe':
            # Clearly no PII
            templates = [
                f"The weather forecast predicts {random.choice(['rain', 'sun', 'clouds'])} with temperature {random.randint(-10,40)}°C.",
                f"Product #{random.randint(1000,9999)} costs ${random.randint(10,500)}.{random.randint(10,99)}.",
                f"Meeting scheduled for {fake.date_this_month()} at {random.randint(9,17)}:00 in conference room {random.choice(['A', 'B', 'C'])}.",
                f"System generated report ID: {fake.uuid4()} on {fake.date_this_year()}.",
                f"The document contains {random.randint(10,500)} pages and was last modified {random.randint(1,30)} days ago.",
            ]
            text = random.choice(templates)
        
        elif category == 'confusing':
            # Confusing - looks like PII but isn't (hardest cases)
            templates = [
                f"Contact Agent {random.choice(['Smith', 'Johnson', 'Williams', 'Brown'])} at extension {random.randint(1000,9999)}.",
                f"Reference ID: {fake.iban()}, Status: {random.choice(['Pending', 'Approved', 'Rejected'])}",
                f"Account {fake.iban()} was created on {fake.date_this_year()}.",
                f"System user 'admin_{random.randint(100,999)}' logged in at {fake.time()}.",
                f"Transaction #{random.randint(100000,999999)} processed on {fake.date_this_month()}.",
                f"Ticket assigned to support team member #{random.randint(100,999)}.",
                f"Call reference number: {random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
                f"Email notification sent to department-{random.choice(['finance', 'hr', 'it'])}@company.internal",
            ]
            text = random.choice(templates)
        
        elif category == 'business':
            # Business/corporate info (no personal PII)
            templates = [
                f"{fake.company()} reported ${random.randint(1,100)}M revenue in Q{random.randint(1,4)} {fake.year()}.",
                f"The {fake.company()} headquarters is located at {random.randint(100,9999)} Business Park, Suite {random.randint(100,999)}.",
                f"Conference call scheduled with {fake.company()} representatives on {fake.date_this_month()}.",
                f"Purchase order #PO{random.randint(10000,99999)} from {fake.company()} received.",
                f"{fake.company()} stock price increased by {random.randint(1,20)}% this quarter.",
            ]
            text = random.choice(templates)
        
        else:  # fictional
            # Fictional/generic names and numbers (edge cases)
            templates = [
                f"The character {fake.name()} appears in chapter {random.randint(1,20)} of the novel.",
                f"In the movie, {fake.name()} plays the role of the antagonist.",
                f"The textbook example uses customer ID {random.randint(100000,999999)} for demonstration.",
                f"Sample invoice for 'John Doe' showing typical format.",
                f"Example: If patient X has condition Y, then treatment Z is recommended.",
                f"Test account test_{random.randint(100,999)}@example.com for QA purposes.",
            ]
            text = random.choice(templates)
        
        samples.append({'text': text, 'contains_pii': 0, 'pii_type': 'none'})
    
    # Shuffle the dataset
    random.shuffle(samples)
    df = pd.DataFrame(samples)
    
    # Split into train/test (80/20)
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    # Save datasets
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'pii_dataset_train_v2.csv')
    test_path = os.path.join(output_dir, 'pii_dataset_test_v2.csv')
    
    # Rename columns to match expected format
    train_df_out = train_df[['text', 'contains_pii', 'pii_type']].copy()
    test_df_out = test_df[['text', 'contains_pii', 'pii_type']].copy()
    
    train_df_out.to_csv(train_path, index=False)
    test_df_out.to_csv(test_path, index=False)
    
    print(f"✅ Created realistic PII dataset:")
    print(f"   Train: {len(train_df)} samples → {train_path}")
    print(f"   Test: {len(test_df)} samples → {test_path}")
    print(f"\n📊 Distribution:")
    print(f"   PII samples: {(train_df['contains_pii'] == 1).sum()} train, {(test_df['contains_pii'] == 1).sum()} test")
    print(f"   Non-PII samples: {(train_df['contains_pii'] == 0).sum()} train, {(test_df['contains_pii'] == 0).sum()} test")
    print(f"\n🎯 Expected accuracy: 70-85% (realistic for bit-flip attack research)")
    
    return train_path, test_path

if __name__ == "__main__":
    train_path, test_path = create_realistic_pii_dataset(num_samples=1000)
    
    # Show some samples
    print("\n📝 Sample PII examples:")
    train_df = pd.read_csv(train_path)
    pii_samples = train_df[train_df['contains_pii'] == 1].head(5)
    for idx, row in pii_samples.iterrows():
        print(f"  - {row['text'][:80]}...")
    
    print("\n📝 Sample Non-PII examples:")
    non_pii_samples = train_df[train_df['contains_pii'] == 0].head(5)
    for idx, row in non_pii_samples.iterrows():
        print(f"  - {row['text'][:80]}...")

