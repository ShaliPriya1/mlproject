import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

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

# 5. Predict the component name based on user input
def predict_component(model, user_input):
    prediction = model.predict([user_input])  # Predict component name based on user input
    return prediction[0]

# Main function to tie everything together
def main(file_path):
    # Load and preprocess data
    data = load_data(file_path)
    data = preprocess_data(data)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Test the model (optional)
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Get user input and predict component
    while True:
        user_summary = input("Enter incident summary (or 'exit' to quit): ")
        if user_summary.lower() == 'exit':
            break
        
        # Get prediction for the component name
        predicted_component = predict_component(model, user_summary)
        print(f"Predicted Component Name: {predicted_component}\n")

if __name__ == "__main__":
    # Replace with your actual file path containing the data
    file_path = "incident_data.csv"
    main(file_path)
