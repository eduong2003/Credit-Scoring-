import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import re

def load_data():
    # Load the CSV file into a DataFrame
    df = pd.read_csv(r'H:\Projects\data\train.csv', low_memory=False)
    return df

def clean_data(df):
    # Handle missing values
    numeric_columns = df.select_dtypes(include='number').columns
    df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(x.median()))
    
    categorical_columns = df.select_dtypes(include='object').columns
    df[categorical_columns] = df[categorical_columns].apply(lambda x: x.fillna(x.mode().iloc[0]))
    
    # Convert 'Credit_History_Age' to numerical values
    def extract_years(text):
        match = re.search(r'(\d+) Years', text)
        if match:
            return int(match.group(1))
        return 0
    
    df['Credit_History_Age_Years'] = df['Credit_History_Age'].apply(extract_years)
    
    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Score'])
    
    # Create new features
    df['Total_Loans_and_Credit_Cards'] = df['Num_of_Loan'] + df['Num_Credit_Card']
    
    return df

def save_data(df):
    # Create a connection to PostgreSQL
    engine = create_engine('postgresql://postgres:sydney2004@localhost:5432/credit_card_data')
    
    # Save the cleaned DataFrame back to PostgreSQL
    df.to_sql('cleaned_credit_card_data', engine, if_exists='replace', index=False)

def train_and_evaluate_model():
    # Create a connection to PostgreSQL
    engine = create_engine('postgresql://postgres:sydney2004@localhost:5432/credit_card_data')
    
    # Load cleaned data from PostgreSQL into a DataFrame
    df = pd.read_sql('SELECT * FROM cleaned_credit_card_data', engine)
    
    # Ensure all columns except the target are numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                df.drop(columns=[col], inplace=True)
    
    # Define features and target
    target_columns = ['Credit_Score_Good', 'Credit_Score_Poor', 'Credit_Score_Standard']
    X = df.drop(columns=target_columns)  # Adjust based on your one-hot encoding
    y = df[target_columns].idxmax(axis=1)  # Multi-class labels
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Initialize the model
    model = RandomForestClassifier(random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Print the classification report
    print(classification_report(y_test, y_pred))
    
    # Print the confusion matrix
    print(confusion_matrix(y_test, y_pred))

def main():
    df = load_data()
    cleaned_df = clean_data(df)
    save_data(cleaned_df)
    train_and_evaluate_model()

if __name__ == "__main__":
    main()
