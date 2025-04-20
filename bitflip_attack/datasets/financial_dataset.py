"""
Financial Dataset for Bit Flip Attacks

This module provides utilities for creating and processing financial datasets
for demonstrating privacy attacks on financial models.
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from faker import Faker


class FinancialDataset(Dataset):
    """
    PyTorch dataset for financial data with sensitive information.
    Used for training models that could be vulnerable to bit flip attacks.
    """
    
    def __init__(self, data, labels, tokenizer=None, max_length=128, cat_columns=None, num_columns=None, text_column=None):
        """
        Initialize the dataset.
        
        Args:
            data: DataFrame or list of financial data
            labels: Corresponding labels
            tokenizer: Optional tokenizer for processing text data
            max_length: Maximum sequence length for tokenization
            cat_columns: List of categorical column names
            num_columns: List of numerical column names
            text_column: Name of text column if present
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        
        # Handle various input types
        if isinstance(data, pd.DataFrame):
            self.df = data
            if cat_columns is None:
                self.cat_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
                if text_column in self.cat_columns:
                    self.cat_columns.remove(text_column)
            else:
                self.cat_columns = cat_columns
                
            if num_columns is None:
                self.num_columns = data.select_dtypes(include=['number']).columns.tolist()
            else:
                self.num_columns = num_columns
                
            self.labels = labels if labels is not None else []
        else:
            # Handle list of dictionaries or other formats
            self.data = data
            self.labels = labels
            self.cat_columns = cat_columns or []
            self.num_columns = num_columns or []
    
    def __len__(self):
        if hasattr(self, 'df'):
            return len(self.df)
        else:
            return len(self.data)
    
    def __getitem__(self, idx):
        if hasattr(self, 'df'):
            # Get row from DataFrame
            row = self.df.iloc[idx]
            
            # Process categorical features
            cat_features = []
            for col in self.cat_columns:
                cat_features.append(row[col])
            
            # Process numerical features
            num_features = []
            for col in self.num_columns:
                num_features.append(row[col])
            
            # Process text if present
            text_features = None
            if self.text_column and self.tokenizer:
                text = row[self.text_column]
                encoding = self.tokenizer(
                    text, 
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                text_features = {
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze(),
                }
            
            # Get label
            if isinstance(self.labels, pd.Series):
                label = self.labels.iloc[idx]
            elif isinstance(self.labels, list):
                label = self.labels[idx]
            else:
                label = row[self.labels] if isinstance(self.labels, str) else 0
            
            # Create return dictionary
            item = {}
            
            if cat_features:
                item['cat_features'] = torch.tensor(
                    pd.get_dummies(''.join(cat_features), dtype=int).values.astype(np.float32)
                )
            
            if num_features:
                item['num_features'] = torch.tensor(num_features, dtype=torch.float32)
            
            if text_features:
                item.update(text_features)
            
            item['label'] = torch.tensor(label, dtype=torch.long)
            
            return item
        else:
            # Handle list data
            data_item = self.data[idx]
            label = self.labels[idx] if self.labels else 0
            
            # Process based on data type
            if isinstance(data_item, dict):
                # Convert dict values to tensors
                processed = {k: torch.tensor(v, dtype=torch.float32) 
                           if isinstance(v, (int, float, list, np.ndarray)) else v 
                           for k, v in data_item.items()}
                processed['label'] = torch.tensor(label, dtype=torch.long)
                return processed
            else:
                # Simple feature vector
                return {
                    'features': torch.tensor(data_item, dtype=torch.float32),
                    'label': torch.tensor(label, dtype=torch.long)
                }


class SyntheticFinancialDataGenerator:
    """
    Generator for synthetic financial datasets for privacy attack demonstrations.
    """
    
    def __init__(self, seed=42):
        """
        Initialize the synthetic financial data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.faker = Faker()
        self.faker.seed_instance(seed)
        np.random.seed(seed)
        
        # Constants for financial data generation
        self.loan_statuses = ['Approved', 'Denied', 'Pending']
        self.loan_purposes = ['Home Purchase', 'Education', 'Medical Expenses', 'Debt Consolidation', 
                            'Business', 'Car Purchase', 'Home Improvement', 'Vacation', 'Other']
        self.employment_statuses = ['Employed', 'Self-employed', 'Unemployed', 'Retired']
        self.housing_statuses = ['Own', 'Rent', 'Mortgage', 'Other']
        self.education_levels = ['High School', 'Associate Degree', 'Bachelor\'s Degree', 
                              'Master\'s Degree', 'PhD', 'Other']
        self.income_brackets = ['Under $25,000', '$25,000-$50,000', '$50,001-$75,000', 
                             '$75,001-$100,000', 'Over $100,000']
    
    def generate_credit_scores(self, n, mean=680, std=100):
        """
        Generate realistic credit scores.
        
        Args:
            n: Number of scores to generate
            mean: Mean credit score
            std: Standard deviation of credit scores
            
        Returns:
            Array of credit scores
        """
        scores = np.round(np.random.normal(mean, std, n)).astype(int)
        # Clip to valid FICO range
        return np.clip(scores, 300, 850)
    
    def generate_loan_applications(self, num_records=1000):
        """
        Generate synthetic loan application data.
        
        Args:
            num_records: Number of records to generate
            
        Returns:
            DataFrame with loan application data
        """
        records = []
        
        # Define feature distributions
        age_mean, age_std = 40, 14
        income_mean, income_std = 75000, 30000
        loan_amount_mean, loan_amount_std = 20000, 15000
        loan_term_options = [12, 24, 36, 48, 60, 72, 84, 120, 180, 240, 360]
        
        for _ in range(num_records):
            # Generate personal information
            name = self.faker.name()
            age = min(max(int(np.random.normal(age_mean, age_std)), 18), 90)
            ssn = self.faker.ssn()
            email = self.faker.email()
            phone = self.faker.phone_number()
            
            # Generate financial information
            income = max(10000, int(np.random.normal(income_mean, income_std)))
            credit_score = int(self.generate_credit_scores(1)[0])
            employment_status = np.random.choice(self.employment_statuses)
            employment_years = round(max(0, np.random.normal(10, 7)), 1)
            housing_status = np.random.choice(self.housing_statuses)
            education = np.random.choice(self.education_levels)
            
            # Generate loan details
            loan_amount = max(1000, int(np.random.normal(loan_amount_mean, loan_amount_std)))
            loan_purpose = np.random.choice(self.loan_purposes)
            loan_term = np.random.choice(loan_term_options)
            existing_loans = max(0, np.random.poisson(1))
            debt_to_income = min(max(np.random.normal(0.3, 0.15), 0), 1)
            
            # Determine loan status based on credit score and other factors
            # This creates a realistic but non-trivial relationship
            approval_score = (
                0.4 * ((credit_score - 300) / 550) +  # Credit score component
                0.2 * min(income / 100000, 1) +       # Income component
                0.1 * (1 - min(loan_amount / 50000, 1)) +  # Loan amount component
                0.1 * (min(employment_years / 10, 1)) +    # Employment history
                0.1 * (1 - debt_to_income) +               # Debt-to-income ratio
                0.1 * np.random.random()                   # Random factor
            )
            
            if approval_score > 0.7:
                loan_status = 'Approved'
            elif approval_score > 0.4:
                loan_status = np.random.choice(['Approved', 'Pending'], p=[0.3, 0.7])
            else:
                loan_status = np.random.choice(['Denied', 'Pending'], p=[0.8, 0.2])
            
            # Generate interest rate based on credit score and loan term
            if loan_status == 'Approved':
                base_rate = 0.03
                credit_adjustment = max(0, (850 - credit_score) / 850 * 0.15)
                term_adjustment = loan_term / 360 * 0.02
                interest_rate = round(base_rate + credit_adjustment + term_adjustment, 4)
            else:
                interest_rate = None
            
            # Create record
            record = {
                'name': name,
                'age': age,
                'ssn': ssn,
                'email': email,
                'phone': phone,
                'address': self.faker.address().replace('\n', ', '),
                'annual_income': income,
                'credit_score': credit_score,
                'employment_status': employment_status,
                'employment_years': employment_years,
                'housing_status': housing_status,
                'education': education,
                'loan_amount': loan_amount,
                'loan_purpose': loan_purpose,
                'loan_term_months': loan_term,
                'existing_loans': existing_loans,
                'debt_to_income_ratio': round(debt_to_income, 2),
                'loan_status': loan_status,
                'interest_rate': interest_rate
            }
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def generate_credit_card_transactions(self, num_records=10000, num_customers=100, 
                                        start_date='2022-01-01', end_date='2022-12-31'):
        """
        Generate synthetic credit card transaction data.
        
        Args:
            num_records: Number of transactions to generate
            num_customers: Number of unique customers
            start_date: Start date for transactions
            end_date: End date for transactions
            
        Returns:
            DataFrame with credit card transaction data
        """
        # Generate customer information
        customers = []
        for i in range(num_customers):
            credit_score = int(self.generate_credit_scores(1)[0])
            customer = {
                'customer_id': f"CUST{i:06d}",
                'name': self.faker.name(),
                'ssn': self.faker.ssn(),
                'email': self.faker.email(),
                'phone': self.faker.phone_number(),
                'address': self.faker.address().replace('\n', ', '),
                'credit_score': credit_score,
                'credit_limit': int(3000 + (credit_score - 300) * 100),
                'card_number': self.faker.credit_card_number(),
                'card_type': np.random.choice(['Visa', 'MasterCard', 'Amex', 'Discover']),
                'card_expiry': self.faker.credit_card_expire()
            }
            customers.append(customer)
        
        # Convert dates to timestamps
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        # Generate transaction information
        transactions = []
        
        # Define merchant categories and corresponding amount distributions
        merchant_categories = {
            'Grocery': (50, 200),
            'Restaurant': (20, 100),
            'Gas': (20, 80),
            'Retail': (30, 300),
            'Online': (10, 500),
            'Travel': (100, 1000),
            'Entertainment': (20, 150),
            'Healthcare': (50, 500),
            'Utilities': (30, 300),
            'Other': (10, 200)
        }
        
        # Create fraudulent transaction patterns
        fraud_patterns = [
            {'category': 'Online', 'min_amount': 500, 'max_amount': 2000, 'probability': 0.7},
            {'category': 'Travel', 'min_amount': 1000, 'max_amount': 5000, 'probability': 0.6},
            {'category': 'Retail', 'min_amount': 800, 'max_amount': 3000, 'probability': 0.5},
            {'category': 'Other', 'min_amount': 1000, 'max_amount': 3000, 'probability': 0.8}
        ]
        
        for i in range(num_records):
            # Select a random customer
            customer = np.random.choice(customers)
            
            # Generate transaction timestamp
            transaction_date = start_ts + (end_ts - start_ts) * np.random.random()
            
            # Determine if transaction is fraudulent (low probability)
            is_fraudulent = np.random.random() < 0.01
            
            if is_fraudulent:
                # Generate fraudulent transaction
                pattern = np.random.choice(fraud_patterns)
                category = pattern['category']
                amount = np.random.uniform(pattern['min_amount'], pattern['max_amount'])
                
                # Fraudulent transactions often happen in unusual locations
                country = np.random.choice(['US', 'CA', 'GB', 'AU', 'FR', 'DE', 'JP', 'CN', 'RU', 'BR'])
                if country != 'US':
                    merchant = self.faker.company() + f" ({country})"
                else:
                    merchant = self.faker.company()
            else:
                # Generate legitimate transaction
                category = np.random.choice(list(merchant_categories.keys()))
                min_amount, max_amount = merchant_categories[category]
                amount = np.random.uniform(min_amount, max_amount)
                country = 'US'  # Most transactions are domestic
                merchant = self.faker.company()
            
            # Create transaction record
            transaction = {
                'transaction_id': f"TXN{i:08d}",
                'customer_id': customer['customer_id'],
                'customer_name': customer['name'],
                'card_number': customer['card_number'],
                'card_type': customer['card_type'],
                'date': transaction_date.strftime('%Y-%m-%d'),
                'time': transaction_date.strftime('%H:%M:%S'),
                'amount': round(amount, 2),
                'merchant': merchant,
                'category': category,
                'country': country,
                'is_fraudulent': int(is_fraudulent)
            }
            
            transactions.append(transaction)
        
        # Create DataFrames
        customers_df = pd.DataFrame(customers)
        transactions_df = pd.DataFrame(transactions)
        
        return transactions_df, customers_df
    
    def create_loan_approval_dataset(self, num_records=1000, test_size=0.2, save_path=None):
        """
        Create a dataset for loan approval prediction.
        
        Args:
            num_records: Number of records to generate
            test_size: Proportion of data to use for testing
            save_path: Path to save the dataset
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Generate loan application data
        df = self.generate_loan_applications(num_records)
        
        # Create target variable (binary classification)
        df['approved'] = df['loan_status'].apply(lambda x: 1 if x == 'Approved' else 0)
        
        # Remove sensitive information for the model input
        # but keep it in a separate DataFrame for privacy attack demonstration
        sensitive_columns = ['name', 'ssn', 'email', 'phone', 'address']
        sensitive_data = df[sensitive_columns].copy()
        
        model_df = df.drop(columns=sensitive_columns + ['loan_status', 'interest_rate'])
        
        # Create categorical dummies for model input
        cat_columns = ['employment_status', 'housing_status', 'education', 'loan_purpose']
        model_df = pd.get_dummies(model_df, columns=cat_columns, drop_first=True)
        
        # Split into train and test sets
        train_df, test_df = train_test_split(model_df, test_size=test_size, stratify=model_df['approved'])
        
        # Save sensitive data separately for privacy attack demonstration
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save model data
            train_path = os.path.join(os.path.dirname(save_path), 
                                     f"{os.path.splitext(os.path.basename(save_path))[0]}_train.csv")
            test_path = os.path.join(os.path.dirname(save_path), 
                                    f"{os.path.splitext(os.path.basename(save_path))[0]}_test.csv")
            
            # Save sensitive data
            sensitive_path = os.path.join(os.path.dirname(save_path), 
                                        f"{os.path.splitext(os.path.basename(save_path))[0]}_sensitive.csv")
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            sensitive_data.to_csv(sensitive_path, index=False)
            
            print(f"Train dataset saved to {train_path}")
            print(f"Test dataset saved to {test_path}")
            print(f"Sensitive data saved to {sensitive_path}")
        
        return train_df, test_df
    
    def create_fraud_detection_dataset(self, num_records=10000, num_customers=100, test_size=0.2, save_path=None):
        """
        Create a dataset for credit card fraud detection.
        
        Args:
            num_records: Number of transactions to generate
            num_customers: Number of unique customers
            test_size: Proportion of data to use for testing
            save_path: Path to save the dataset
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Generate credit card transaction data
        transactions_df, customers_df = self.generate_credit_card_transactions(
            num_records=num_records, 
            num_customers=num_customers
        )
        
        # Preprocess data for model input
        # Extract day of week, hour, is_weekend features
        transactions_df['day_of_week'] = pd.to_datetime(transactions_df['date']).dt.dayofweek
        transactions_df['hour'] = pd.to_datetime(transactions_df['time']).dt.hour
        transactions_df['is_weekend'] = transactions_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Create categorical dummies for model input
        cat_columns = ['category', 'card_type', 'country']
        
        # Remove sensitive information for the model input
        # but keep it in a separate DataFrame for privacy attack demonstration
        sensitive_columns = ['customer_name', 'card_number', 'merchant']
        sensitive_data = transactions_df[sensitive_columns].copy()
        
        model_df = transactions_df.drop(columns=sensitive_columns + ['transaction_id', 'date', 'time'])
        model_df = pd.get_dummies(model_df, columns=cat_columns, drop_first=True)
        
        # Split into train and test sets
        train_df, test_df = train_test_split(model_df, test_size=test_size, stratify=model_df['is_fraudulent'])
        
        # Save datasets
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save model data
            train_path = os.path.join(os.path.dirname(save_path), 
                                     f"{os.path.splitext(os.path.basename(save_path))[0]}_train.csv")
            test_path = os.path.join(os.path.dirname(save_path), 
                                    f"{os.path.splitext(os.path.basename(save_path))[0]}_test.csv")
            
            # Save sensitive and customer data
            sensitive_path = os.path.join(os.path.dirname(save_path), 
                                        f"{os.path.splitext(os.path.basename(save_path))[0]}_sensitive.csv")
            customers_path = os.path.join(os.path.dirname(save_path), 
                                        f"{os.path.splitext(os.path.basename(save_path))[0]}_customers.csv")
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            sensitive_data.to_csv(sensitive_path, index=False)
            customers_df.to_csv(customers_path, index=False)
            
            print(f"Train dataset saved to {train_path}")
            print(f"Test dataset saved to {test_path}")
            print(f"Sensitive data saved to {sensitive_path}")
            print(f"Customer data saved to {customers_path}")
        
        return train_df, test_df


def create_financial_dataloaders(train_df, test_df, batch_size=32, 
                               cat_columns=None, num_columns=None, 
                               label_column='approved'):
    """
    Create PyTorch DataLoaders for financial datasets.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        batch_size: Batch size for DataLoader
        cat_columns: List of categorical column names
        num_columns: List of numerical column names
        label_column: Column name containing labels
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Determine categorical and numerical columns if not provided
    if cat_columns is None:
        cat_columns = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
        if label_column in cat_columns:
            cat_columns.remove(label_column)
    
    if num_columns is None:
        num_columns = train_df.select_dtypes(include=['number']).columns.tolist()
        if label_column in num_columns:
            num_columns.remove(label_column)
    
    # Get labels
    train_labels = train_df[label_column]
    test_labels = test_df[label_column]
    
    # Create datasets
    train_dataset = FinancialDataset(
        data=train_df,
        labels=train_labels,
        cat_columns=cat_columns,
        num_columns=num_columns
    )
    
    test_dataset = FinancialDataset(
        data=test_df,
        labels=test_labels,
        cat_columns=cat_columns,
        num_columns=num_columns
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


def generate_quick_financial_dataset(dataset_type='loan', num_records=1000, output_path="data/financial_dataset.csv"):
    """
    Quickly generate a financial dataset for experiments.
    
    Args:
        dataset_type: Type of dataset to generate ('loan' or 'fraud')
        num_records: Number of records to generate
        output_path: Path to save the dataset
        
    Returns:
        Path to the generated dataset
    """
    generator = SyntheticFinancialDataGenerator()
    
    if dataset_type == 'loan':
        # Generate loan approval dataset
        train_df, test_df = generator.create_loan_approval_dataset(
            num_records=num_records,
            save_path=output_path
        )
        print(f"Generated loan approval dataset with {len(train_df) + len(test_df)} records")
        
    elif dataset_type == 'fraud':
        # Generate fraud detection dataset (use fewer records for fraud since it's transaction level)
        num_customers = max(50, num_records // 100)
        train_df, test_df = generator.create_fraud_detection_dataset(
            num_records=num_records,
            num_customers=num_customers,
            save_path=output_path
        )
        print(f"Generated fraud detection dataset with {len(train_df) + len(test_df)} records")
        
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Use 'loan' or 'fraud'.")
    
    return output_path


if __name__ == "__main__":
    # Example usage
    output_path = "data/financial_dataset.csv"
    generate_quick_financial_dataset('loan', 1000, output_path)
