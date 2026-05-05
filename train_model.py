import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load the Dataset
# Assuming the 'online_shoppers_intention.csv' is in the same directory.
# You can download it from UCI Machine Learning Repository.
url = "https://raw.githubusercontent.com/sahil/sample_data/main/online_shoppers_intention.csv" 
try:
    df = pd.read_csv('online_shoppers_intention.csv')
except FileNotFoundError:
    print("Dataset not found locally. Please ensure 'online_shoppers_intention.csv' is present.")
    exit()

# 2. Data Preprocessing
# Encode categorical variables
le = LabelEncoder()
df['Month'] = le.fit_transform(df['Month'])
df['VisitorType'] = le.fit_transform(df['VisitorType'])
df['Weekend'] = df['Weekend'].astype(int)

# Separate features and target
X = df.drop('Revenue', axis=1)
y = df['Revenue'].astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Apply SMOTE (Only on training data to prevent data leakage)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# 4. Dimensionality Reduction using PCA
pca = PCA(n_components=10) # Retaining 10 principal components
X_train_pca = pca.fit_transform(X_train_smote)
X_test_pca = pca.transform(X_test_scaled)

# 5. Model Building (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train_smote)

# Evaluation
y_pred = knn.predict(X_test_pca)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Save the components for the UI
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')
joblib.dump(knn, 'knn_model.pkl')
print("Model and preprocessing pipelines saved successfully.")