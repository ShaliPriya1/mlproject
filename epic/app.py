from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Flask App Setup
app = Flask(__name__)

# Step 1: Load and preprocess consolidated customer data
def load_and_process_data(filename="customer_data.csv"):
    df = pd.read_csv(filename)
    # Convert purchase history from string to list
    df['purchase_history'] = df['purchase_history'].apply(lambda x: list(map(int, x.split(', '))))
    return df

# Step 2: Customer Segmentation using KMeans
def customer_segmentation(df):
    customer_data = np.array([[row['age'], row['income'], sum(row['purchase_history']), row['engagement_score']] for index, row in df.iterrows()])
    scaler = StandardScaler()
    customer_data = scaler.fit_transform(customer_data)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(customer_data)
    return df

# Step 3: Churn Prediction using Logistic Regression
def churn_prediction(df):
    customer_data = np.array([[row['age'], row['income'], sum(row['purchase_history']), row['engagement_score']] for index, row in df.iterrows()])
    churn_labels = np.array([1 if row['engagement_score'] < 60 else 0 for index, row in df.iterrows()])
    
    X_train, X_test, y_train, y_test = train_test_split(customer_data, churn_labels, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    return predictions

# Step 4: Personalized Product Recommendations using KNN
def product_recommendations(df):
    customer_data = np.array([[row['age'], row['income'], sum(row['purchase_history']), row['engagement_score']] for index, row in df.iterrows()])
    
    knn = NearestNeighbors(n_neighbors=3)
    knn.fit(customer_data)
    
    recommendations = {}
    for i, row in df.iterrows():
        _, indices = knn.kneighbors([customer_data[i]])
        recommended_customers = [df.iloc[idx]['name'] for idx in indices[0]]
        recommendations[row['name']] = recommended_customers
    
    return recommendations

# Step 5: Customer Lifetime Value (LTV) Prediction using Neural Networks
def build_and_train_ltv_model(df):
    customer_data = np.array([[row['age'], row['income'], sum(row['purchase_history']), row['engagement_score']] for index, row in df.iterrows()])
    lifetime_values = np.array([row['lifetime_value'] for index, row in df.iterrows()])
    
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(customer_data.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(customer_data, lifetime_values, epochs=10, batch_size=8)
    
    return model

# Step 6: Data Visualization for Customer Segments
def plot_customer_segments(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='income', y='age', hue='cluster', palette='Set1')
    plt.title('Customer Segments by Income and Age')
    plt.xlabel('Income')
    plt.ylabel('Age')

    # Save the plot to a PNG in base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url

@app.route("/", methods=["GET", "POST"])
def index():
    df = load_and_process_data()  # Load and process data
    
    # Apply the ML models and visualizations
    df = customer_segmentation(df)  # Apply customer segmentation (KMeans)
    churn_predictions = churn_prediction(df)  # Churn prediction
    recommendations = product_recommendations(df)  # Product recommendations
    ltv_model = build_and_train_ltv_model(df)  # Train the LTV model
    
    # Generate data visualization
    customer_segments_plot = plot_customer_segments(df)
    
    if request.method == "POST":
        # Return predictions and recommendations upon request
        name = request.form.get("name")
        customer_data = df[df['name'] == name].iloc[0]
        
        return render_template(
            "index.html",
            customer=customer_data,
            churn_prediction=churn_predictions,
            recommendations=recommendations.get(name, []),
            customer_segments_plot=customer_segments_plot
        )
    
    return render_template("index.html", customer_segments_plot=customer_segments_plot)

if __name__ == "__main__":
    app.run(debug=True)
