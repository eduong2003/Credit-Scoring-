import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

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
            # Drop columns that cannot be converted to numeric
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

