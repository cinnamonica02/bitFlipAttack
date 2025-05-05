"""
Synthetic PII Dataset Generator for Bit Flip Attacks

This module provides utilities for generating synthetic personally identifiable information
(PII) datasets to demonstrate privacy vulnerabilities in machine learning models.
"""
import os
import pandas as pd
import numpy as np
from faker import Faker
from sklearn.model_selection import train_test_split
import random


class SyntheticPIIGenerator:
    """
    Generator for synthetic PII datasets using the Faker library.
    Creates realistic-looking but fake personal data for privacy attack demonstrations.
    """
    
    def __init__(self, seed=42, locale='en_US'):
        """
        Initialize the synthetic PII generator.
        
        Args:
            seed: Random seed for reproducibility
            locale: Locale for generating region-specific data
        """
        self.faker = Faker(locale)
        self.faker.seed_instance(seed)
        np.random.seed(seed)
    
    def generate_personal_records(self, num_records=10000):
        """
        Generate basic personal records with names, addresses, and contact information.
        
        Args:
            num_records: Number of synthetic records to generate
            
        Returns:
            DataFrame with synthetic personal records
        """
        records = []
        
        for _ in range(num_records):
            record = {
                'name': self.faker.name(),
                'address': self.faker.address().replace('\n', ', '),
                'email': self.faker.email(),
                'phone': self.faker.phone_number(),
                'ssn': self.faker.ssn(),
                'dob': self.faker.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d')
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def generate_financial_records(self, num_records=10000):
        """
        Generate synthetic financial records with credit card information,
        account details, and transaction data.
        
        Args:
            num_records: Number of synthetic records to generate
            
        Returns:
            DataFrame with synthetic financial records
        """
        records = []
        
        for _ in range(num_records):
            income = np.random.normal(70000, 30000)
            credit_score = np.random.randint(300, 850)
            
            record = {
                'name': self.faker.name(),
                'credit_card_number': self.faker.credit_card_number(),
                'credit_card_provider': self.faker.credit_card_provider(),
                'credit_card_expiry': self.faker.credit_card_expire(),
                'account_number': self.faker.bban(),
                'annual_income': max(20000, round(income, 2)),
                'credit_score': credit_score,
                'loan_amount': np.random.randint(1000, 50000) if np.random.random() < 0.7 else 0,
                'loan_status': np.random.choice(['Approved', 'Denied', 'Pending'], p=[0.6, 0.3, 0.1]) 
                                if np.random.random() < 0.7 else 'NA'
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def generate_medical_records(self, num_records=10000):
        """
        Generate synthetic medical records with patient information,
        diagnoses, and treatment data.
        
        Args:
            num_records: Number of synthetic records to generate
            
        Returns:
            DataFrame with synthetic medical records
        """
        # Common medical conditions and medications
        conditions = [
            'Hypertension', 'Type 2 Diabetes', 'Asthma', 'Depression', 'Anxiety',
            'Hyperlipidemia', 'Osteoarthritis', 'GERD', 'Hypothyroidism', 'Migraine'
        ]
        
        medications = [
            'Lisinopril', 'Metformin', 'Albuterol', 'Sertraline', 'Alprazolam',
            'Atorvastatin', 'Ibuprofen', 'Omeprazole', 'Levothyroxine', 'Sumatriptan'
        ]
        
        records = []
        
        for _ in range(num_records):
            # Generate 0-3 conditions for each patient
            num_conditions = np.random.randint(0, 4)
            patient_conditions = np.random.choice(conditions, size=num_conditions, replace=False).tolist()
            
            # Generate medications (typically one per condition)
            num_medications = max(0, np.random.randint(num_conditions, num_conditions + 2))
            patient_medications = np.random.choice(medications, size=min(num_medications, len(medications)), 
                                                  replace=False).tolist()
            
            blood_pressure = f"{np.random.randint(90, 160)}/{np.random.randint(60, 100)}"
            
            record = {
                'patient_id': self.faker.uuid4(),
                'name': self.faker.name(),
                'dob': self.faker.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d'),
                'gender': np.random.choice(['Male', 'Female', 'Non-binary'], p=[0.48, 0.48, 0.04]),
                'blood_type': np.random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']),
                'weight_kg': round(np.random.normal(70, 15), 1),
                'height_cm': round(np.random.normal(170, 15)),
                'blood_pressure': blood_pressure,
                'conditions': ', '.join(patient_conditions),
                'medications': ', '.join(patient_medications),
                'insurance_provider': self.faker.company(),
                'insurance_id': self.faker.uuid4()[:10].upper(),
                'last_visit': self.faker.date_this_year().strftime('%Y-%m-%d')
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def generate_mixed_pii_dataset(self, num_records=10000, pii_ratio=0.5):
        """
        Generate a dataset with mixed PII and non-PII text for classification tasks.
        This dataset is labeled for whether it contains PII (1) or not (0).
        
        Args:
            num_records: Number of synthetic records to generate
            pii_ratio: Proportion of records that should contain PII
            
        Returns:
            DataFrame with text samples and PII labels
        """
        records = []
        num_pii = int(num_records * pii_ratio)
        num_non_pii = num_records - num_pii
        
        # Generate PII-containing text
        for _ in range(num_pii):
            has_ssn = np.random.random() < 0.3
            has_credit_card = np.random.random() < 0.3
            has_email = np.random.random() < 0.5
            has_address = np.random.random() < 0.4
            has_phone = np.random.random() < 0.4
            
            text_components = []
            pii_type = np.random.choice(['personal', 'financial', 'medical'])
            
            if pii_type == 'personal':
                name = self.faker.name()
                text_components.append(f"My name is {name}.")
                
                if has_email:
                    email = self.faker.email()
                    text_components.append(f"You can contact me at {email}.")
                
                if has_phone:
                    phone = self.faker.phone_number()
                    text_components.append(f"My phone number is {phone}.")
                
                if has_address:
                    address = self.faker.address().replace('\n', ', ')
                    text_components.append(f"I live at {address}.")
                
                if has_ssn:
                    ssn = self.faker.ssn()
                    text_components.append(f"My SSN is {ssn}.")
            
            elif pii_type == 'financial':
                name = self.faker.name()
                text_components.append(f"I'm {name} and I'm applying for a loan.")
                
                if has_credit_card:
                    cc_num = self.faker.credit_card_number()
                    text_components.append(f"My credit card number is {cc_num}.")
                
                account = self.faker.bban()
                income = np.random.normal(70000, 30000)
                text_components.append(f"My account number is {account} and I make ${max(20000, round(income, 2))} annually.")
            
            elif pii_type == 'medical':
                name = self.faker.name()
                dob = self.faker.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d')
                text_components.append(f"Patient {name}, born on {dob}.")
                
                condition = np.random.choice([
                    'Hypertension', 'Type 2 Diabetes', 'Asthma', 'Depression', 'Anxiety',
                    'Hyperlipidemia', 'Osteoarthritis', 'GERD', 'Hypothyroidism', 'Migraine'
                ])
                text_components.append(f"Diagnosed with {condition}.")
                
                if has_ssn:
                    ssn = self.faker.ssn()
                    text_components.append(f"Patient SSN: {ssn}.")
                
                insurance = self.faker.company()
                insurance_id = self.faker.uuid4()[:10].upper()
                text_components.append(f"Insurance provider: {insurance}, ID: {insurance_id}")
            
            # Join the text components
            text = ' '.join(text_components)
            records.append({'text': text, 'contains_pii': 1, 'pii_type': pii_type})
        
        # Generate non-PII text
        non_pii_templates = [
            "The weather today is {weather} with a temperature of {temp}°C.",
            "The {color} car drove down the street at {speed} mph.",
            "The restaurant serves {cuisine} food and is open from {open_time} to {close_time}.",
            "The movie {movie} was released in {year} and received {rating} stars.",
            "The {species} is native to {region} and typically lives for {lifespan} years.",
            "The company reported {profit} in profits for the last quarter.",
            "The team won {wins} games and lost {losses} games this season.",
            "The book has {pages} pages and was written by an anonymous author.",
            "The building is {height} meters tall and was completed in {year}.",
            "The event will take place on a {day} at an undisclosed location."
        ]
        
        weathers = ['sunny', 'rainy', 'cloudy', 'snowy', 'windy', 'foggy']
        colors = ['red', 'blue', 'green', 'black', 'white', 'silver']
        cuisines = ['Italian', 'Chinese', 'Mexican', 'Indian', 'French', 'Japanese']
        species = ['wolf', 'tiger', 'dolphin', 'eagle', 'bear', 'elephant']
        regions = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Australia']
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for _ in range(num_non_pii):
            template = np.random.choice(non_pii_templates)
            
            if "weather" in template:
                text = template.format(
                    weather=np.random.choice(weathers),
                    temp=np.random.randint(0, 40)
                )
            elif "car" in template:
                text = template.format(
                    color=np.random.choice(colors),
                    speed=np.random.randint(20, 120)
                )
            elif "restaurant" in template:
                text = template.format(
                    cuisine=np.random.choice(cuisines),
                    open_time=f"{np.random.randint(7, 12)}:00",
                    close_time=f"{np.random.randint(17, 24)}:00"
                )
            elif "movie" in template:
                text = template.format(
                    movie=f"Generic Movie {np.random.randint(1, 10)}",
                    year=np.random.randint(1980, 2023),
                    rating=np.random.randint(1, 6)
                )
            elif "species" in template:
                text = template.format(
                    species=np.random.choice(species),
                    region=np.random.choice(regions),
                    lifespan=np.random.randint(5, 100)
                )
            elif "company" in template:
                text = template.format(
                    profit=f"${np.random.randint(1, 100)} million"
                )
            elif "team" in template:
                wins = np.random.randint(0, 50)
                losses = np.random.randint(0, 50)
                text = template.format(
                    wins=wins,
                    losses=losses
                )
            elif "book" in template:
                text = template.format(
                    pages=np.random.randint(100, 1000)
                )
            elif "building" in template:
                text = template.format(
                    height=np.random.randint(50, 500),
                    year=np.random.randint(1800, 2023)
                )
            elif "event" in template:
                text = template.format(
                    day=np.random.choice(days)
                )
            
            records.append({'text': text, 'contains_pii': 0, 'pii_type': 'none'})
        
        # Shuffle the records
        df = pd.DataFrame(records)
        df = df.sample(frac=1).reset_index(drop=True)
        
        return df
    
    def generate_pii_detection_dataset(self, num_records=10000, save_path=None, test_size=0.2):
        """
        Generate a dataset for PII detection tasks, splitting into train and test sets.
        
        Args:
            num_records: Number of synthetic records to generate
            save_path: Path to save the dataset
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (train_df, test_df)
        """
        df = self.generate_mixed_pii_dataset(num_records)
        
        # Split into train and test sets
        train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['contains_pii'])
        
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

    def generate_noisy_classification_dataset(self, num_pii=800, num_non_pii=800):
        """
        Generate a challenging classification dataset with noisy PII and confusing non-PII.
        Logic adapted from umup_attack_example.py's load_or_create_dataset.
        Returns a single DataFrame ready to be saved.
        """
        # --- Generate Base PII Records ---
        # generate_personal_records provides more fields than needed initially,
        # but we can adapt. Let's generate base records first.
        # Note: The original logic in umup_attack_example used a different structure.
        # We'll adapt it to use the output of generate_personal_records.
        base_records = self.generate_personal_records(num_records=num_pii)

        # --- PII Noise Generation Function (Adapted) ---
        def create_very_noisy_text(row):
            # Adapt field names based on generate_personal_records output
            fields = {
                'name': f"Name: {row.get('name', '')}", # Combined name field
                'email': f"Email [{row.get('email', '')}]",
                'phone': f"Phone: {row.get('phone', '')}", # 'phone' instead of 'phone_number'
                'ssn': f"SSN: {row.get('ssn', '')}",
                'address': f"Addr: {row.get('address', '')}",
                'dob': f"Birthdate {row.get('dob', '')}"
            }

            final_parts = []
            # Keep fewer fields on average
            num_fields_to_keep = random.randint(max(1, len(fields)//3), max(2, len(fields) - 2))
            kept_fields = random.sample(list(fields.keys()), num_fields_to_keep)

            for key in kept_fields:
                part = fields[key]
                # Increased corruption chance (0.3)
                if random.random() < 0.30:
                    if key == 'phone' and isinstance(row.get('phone'), str) and len(row.get('phone','')) > 5:
                         part = f"Phone: {row.get('phone','').replace('-', '').replace('(', '').replace(')', '').replace(' ', '')}" # Remove more formatting
                    elif key == 'email' and '@' in part:
                         part = part.replace('@', f'(at){random.choice(["_", "."])}') # Add more variation
                    elif key == 'ssn' and isinstance(row.get('ssn'), str) and len(row.get('ssn','')) > 5:
                         part = f"SocSec-{row.get('ssn','').split('-')[-1]}" # Different masking
                    elif key == 'address' and isinstance(row.get('address'), str) and len(row.get('address', '')) > 10:
                         addr_parts = row.get('address', '').split(',') # Split by comma
                         if len(addr_parts) > 1:
                              part = f"Addr: {addr_parts[0].strip()} ... {addr_parts[-1].strip()}" # Truncate middle
                elif random.random() < 0.05: # Chance of omitting field value
                     part = f"{key.capitalize()}: [REDACTED]"

                final_parts.append(part)

            random.shuffle(final_parts)
            return " | ".join(final_parts) # Use pipe separator

        # Apply the noisy text generation
        pii_df = base_records.copy()
        pii_df['text'] = pii_df.apply(create_very_noisy_text, axis=1)
        pii_df['contains_pii'] = 1
        pii_df = pii_df[['text', 'contains_pii']] # Keep only necessary columns

        # --- Create VERY Confusing Non-PII Samples (Adapted) ---
        non_pii_samples = []
        for i in range(num_non_pii):
            # Mix multiple pseudo-PII elements
            num_elements = random.randint(4, 7) # More confusing elements
            elements = []
            for _ in range(num_elements):
                # Increased variety of pseudo-elements
                element_type = random.choice([
                    'date', 'id', 'code', 'name', 'misc_num', 'address_like',
                    'phone_like', 'email_like', 'job_title', 'company_name',
                    'product_code', 'version_num'
                ])
                if element_type == 'date':
                    elements.append(f"Timestamp: {self.faker.iso8601()}")
                elif element_type == 'id':
                    elements.append(f"RefID #{random.randint(100000000, 999999999)}") # Longer ID
                elif element_type == 'code':
                    elements.append(f"Internal Code: {self.faker.swift(length=random.choice([8, 11]))}")
                elif element_type == 'name': # Use common non-personal names
                    elements.append(f"{random.choice(['Agent', 'User', 'System', 'Manager', 'Operator', 'Client'])}: {random.choice(['Support', 'Admin', 'System', 'Service', 'Operator', 'Account'])}")
                elif element_type == 'misc_num':
                     elements.append(f"Tracking-{random.randint(100, 999)}-{random.randint(1000, 9999)}-{random.randint(10, 99)}")
                elif element_type == 'address_like':
                     elements.append(f"Location Code: {random.randint(10, 99)}-{self.faker.street_name()}-{random.choice(['CTR', 'BLDG', 'SITE', 'AREA'])}")
                elif element_type == 'phone_like':
                     elements.append(f"Internal Ext: {random.randint(1000, 9999)}")
                elif element_type == 'email_like':
                     elements.append(f"Service Desk: {self.faker.word()}-{random.choice(['support', 'info', 'admin'])}@{self.faker.domain_name()}")
                elif element_type == 'job_title':
                     elements.append(f"Role: {self.faker.job()}")
                elif element_type == 'company_name':
                     elements.append(f"Vendor: {self.faker.company()}")
                elif element_type == 'product_code':
                     elements.append(f"Product SKU: {self.faker.ean(length=13)}")
                elif element_type == 'version_num':
                     elements.append(f"Version: {random.randint(1, 10)}.{random.randint(0, 9)}.{random.randint(0, 20)}")


            random.shuffle(elements)
            text = " // ".join(elements) # Different separator
            # Add different surrounding text
            prefix = random.choice(["LOG: ", "ENTRY: ", "SYS_MSG: ", "DATA: ", "UPDATE: ", "CASE_NOTE: "])
            suffix = random.choice([" | Status: OK", " | Status: Processed", " | Status: Pending", " | Status: Archived", " | Status: Flagged", " | Status: Complete"])
            text = prefix + text + suffix
            non_pii_samples.append({"text": text, "contains_pii": 0})

        non_pii_df = pd.DataFrame(non_pii_samples)

        # --- Combine and Shuffle ---
        combined_df = pd.concat([pii_df, non_pii_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=self.faker.random.randint(0, 10000)).reset_index(drop=True) # Use faker's random state

        return combined_df[['text', 'contains_pii']]


def generate_quick_pii_dataset(num_records=5000, output_path="data/pii_dataset.csv"):
    """
    A quick helper function to generate a PII detection dataset and save it.
    
    Args:
        num_records: Number of records to generate
        output_path: Path to save the dataset
        
    Returns:
        The path to the generated dataset files
    """
    generator = SyntheticPIIGenerator()
    train_df, test_df = generator.generate_pii_detection_dataset(num_records, output_path)
    
    # Get statistics about the dataset
    total_pii = train_df['contains_pii'].sum() + test_df['contains_pii'].sum()
    print(f"Generated dataset with {num_records} records ({total_pii} with PII)")
    
    # Distribution of PII types
    train_pii = train_df[train_df['contains_pii'] == 1]
    test_pii = test_df[test_df['contains_pii'] == 1]
    
    pii_type_counts = pd.concat([train_pii['pii_type'].value_counts(), 
                                test_pii['pii_type'].value_counts()], axis=1)
    print("\nPII Type Distribution:")
    print(pii_type_counts)
    
    return output_path


if __name__ == "__main__":
    # Example usage
    output_path = "data/synthetic_pii_dataset.csv"
    generate_quick_pii_dataset(10000, output_path) 

    