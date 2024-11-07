
# Required Libraries
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Download the latest version of the dataset from Kaggle
path = kagglehub.dataset_download("kolawale/focusing-on-mobile-app-or-website")

print("Path to dataset files:", path)

# Load the Dataset
url = path + "/Ecommerce Customers"
df = pd.read_csv(url)

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Data Summary
print(df.describe())

# Visualizing Distributions
sns.histplot(df['Avg. Session Length'])
plt.title('Distribution of Avg. Session Length')
plt.show()

sns.histplot(df['Time on App'])
plt.title('Distribution of Time on App')
plt.show()

sns.histplot(df['Time on Website'])
plt.title('Distribution of Time on Website')
plt.show()

sns.histplot(df['Length of Membership'])
plt.title('Distribution of Length of Membership')
plt.show()

# Remove non-numeric columns for correlation matrix
numeric_df = df.select_dtypes(include=[float, int])

# Correlation Matrix
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Feature Selection: Let's choose 'Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership'
# Target: 'Yearly Amount Spent'
features = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
target = 'Yearly Amount Spent'

X = df[features]
y = df[target]

# Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
