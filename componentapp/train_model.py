import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import joblib

# 1. Load and preprocess data
def load_data(file_path):
    # Assuming the data is a CSV file with columns 'summary' and 'component_name'
    data = pd.read_csv(file_path)
    return data

# 2. Preprocess the text data (Optional: remove stop words, punctuation, etc.)
def preprocess_data(data):
    # Use only the 'summary' as the feature
    return data

# 3. Split the data into training and testing sets
def split_data(data):
    X = data['summary']  # Features (summary only)
    y = data['component_name']  # Labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# 4. Train the model using Random Forest with TF-IDF vectorization
def train_model(X_train, y_train):
    # Create a pipeline with TF-IDF and Random Forest classifier
    model = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)
    return model

# Save the trained model to a file
def save_model(model, filename):
    joblib.dump(model, filename)

# Main function to train and save the model
def main():
    # File path of your dataset
    file_path = "incident_data.csv"  # Replace with your actual data file path
    model_filename = "model.pkl"  # Output model file path
    
    # Load, preprocess, and split the data
    data = load_data(file_path)
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Save the trained model to a file
    save_model(model, model_filename)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    main()
