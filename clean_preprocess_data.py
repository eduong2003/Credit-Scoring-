import pandas as pd
from sqlalchemy import create_engine
import re

# Create a connection to PostgreSQL
engine = create_engine('postgresql://postgres:sydney2004@localhost:5432/credit_card_data')

# Load data from PostgreSQL into a DataFrame
df = pd.read_sql('SELECT * FROM credit_card_data', engine)

# Handle missing values

# Fill missing numerical values with the median
numeric_columns = df.select_dtypes(include='number').columns
df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(x.median()))

# Fill missing categorical values with the mode
categorical_columns = df.select_dtypes(include='object').columns
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.fillna(x.mode().iloc[0]))

# Convert 'Credit_History_Age' to numerical values
def extract_years(text):
    match = re.search(r'(\d+) Years', text)
    if match:
        return int(match.group(1))
    return 0

df['Credit_History_Age_Years'] = df['Credit_History_Age'].apply(extract_years)

# Convert relevant columns to numeric types
df['Num_of_Loan'] = pd.to_numeric(df['Num_of_Loan'], errors='coerce').fillna(0).astype(int)
df['Num_Credit_Card'] = pd.to_numeric(df['Num_Credit_Card'], errors='coerce').fillna(0).astype(int)

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Score'])

# Create new features
df['Total_Loans_and_Credit_Cards'] = df['Num_of_Loan'] + df['Num_Credit_Card']

# Save the cleaned DataFrame back to PostgreSQL
df.to_sql('cleaned_credit_card_data', engine, if_exists='replace', index=False)

print("Data cleaning and preprocessing completed successfully.")


