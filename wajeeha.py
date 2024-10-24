import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from transformers import pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix, roc_curve, auc , f1_score, precision_score, recall_score


# make containers
header = st.container()
datasets = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title('Bondora Data Processing and Classification App')
    st.text('In this project, we are working with bondora dataset!')
    

with datasets:
    st.header('Bondora Dataset')
    st.text('This dataset is about Bondora')
    # import data
    data = pd.read_csv('Bondora_preprocessed.csv', low_memory=False)
    # Step 1: Drop columns with more than 10% missing values
    threshold = 0.1 * len(data)
    df = data.dropna(thresh=threshold, axis=1) 
    st.write(df.head(10))
    st.bar_chart(data['Status'].value_counts())

# Define numerical and categorical columns
 # Convert boolean columns to integers if they exist in the dataset
if 'NewCreditCustomer' in df.columns:
    df['NewCreditCustomer'] = df['NewCreditCustomer'].astype(int)
if 'Restructured' in df.columns:
    df['Restructured'] = df['Restructured'].astype(int)

    # Check if 'Status' column exists
if 'Status' in df.columns:
    # Separate the target variable and features
    X = df.drop(columns=['Status'])  # Features
    y = df['Status']  # Target

with features:
    st.header('Features')
    st.text('Numerical features I am going to use')

# Define numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
with model_training:
    
    st.header('Model Training')
    st.text('Train a Random Forest model to predict the status of a loan')
    
    
# Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
        ('scaler', StandardScaler())  # Standardize numerical features
        ])
    

# Preprocessing pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing categorical values
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
        ])
        
# Combine transformers into a single preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
            ])

# Apply the preprocessing pipeline to the data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

# Initialize the Random Forest model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train_preprocessed, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_preprocessed)
st.write(y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
st.subheader('Confusion Matrix:')
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)
        
# Display classification report as a dataframe
st.write("### Classification Report:")
report = classification_report(y_test, y_pred, output_dict=True)
st.write(report)


# ROC Curve (for binary classification)
if len(set(y)) == 2:
    st.subheader('ROC Curve:')
    
    # Predict probabilities for positive class (class 1)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate False Positive Rate (FPR) and True Positive Rate (TPR)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    # Compute the Area Under the Curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line (chance level)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)



# --- User Input for Predictions ---
st.subheader('Make Predictions')

# Collect user inputs for prediction
st.write("Provide input values:")
user_inputs = {}
for col in X.columns:
    if col in numerical_cols:
        user_inputs[col] = st.number_input(f"Enter value for {col}", value=float(X[col].mean()))
    else:
        user_inputs[col] = st.selectbox(f"Select value for {col}", options=df[col].unique())

# Convert the user inputs into a dataframe
input_df = pd.DataFrame([user_inputs])

prediction = model.predict(preprocessor.transform(input_df))
prediction_proba = model.predict_proba(preprocessor.transform(input_df))
st.subheader(f"The model predicts: {prediction}")
st.subheader(f"Probability of each class: {prediction_proba}")

# --- Visualizations ---
st.subheader('Data Visualizations')

# Histogram for numerical columns

st.header("Histograms for a limited number of numerical features")
cols = numerical_cols[:5]  # Select the first 5 numerical columns
fig, ax = plt.subplots(3, 2, figsize=(10, 8))
ax = ax.ravel()
for i, col in enumerate(cols):
    ax[i].hist(df[col], bins=20, color='skyblue', edgecolor='black')
    ax[i].set_title(f'Histogram of {col}')
st.pyplot(fig)

# Correlation heatmap for numerical columns

st.header("Correlation Heatmap for numerical columns")
cols = numerical_cols[:5]  # Select the first 5 numerical columns
corr = pd.DataFrame(df, columns=cols).corr()
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)