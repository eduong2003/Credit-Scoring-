import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

# Create a connection to PostgreSQL
engine = create_engine('postgresql://postgres:sydney2004@localhost:5432/credit_card_data')

# Load cleaned data from PostgreSQL into a DataFrame
df = pd.read_sql('SELECT * FROM cleaned_credit_card_data', engine)

# Summary statistics for numerical columns
print(df.describe())

# Summary statistics for categorical columns
print(df.describe(include=['O']))

# Plot histograms for numerical columns
df.hist(bins=30, figsize=(20, 15))
plt.tight_layout()
plt.show()

# Compute the correlation matrix
corr_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Bar plots for categorical columns
categorical_columns = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Score']
for col in categorical_columns:
    plt.figure(figsize=(10, 5))
    df[col].value_counts().plot(kind='bar')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

# Sample 1000 rows for pairplot to reduce computation time
sample_df = df.sample(n=1000, random_state=42)

# Plot pairplot
sns.pairplot(sample_df, hue='Credit_Score_Good')  # Adjust 'Credit_Score_Good' to the actual column name used after one-hot encoding
plt.show()
