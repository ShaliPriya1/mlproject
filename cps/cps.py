from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow.keras import layers, models
from textblob import TextBlob
import random

app = Flask(__name__)

# Step 1: Read customer data from CSV file
def load_customer_data(filename="customer_data.csv"):
    df = pd.read_csv(filename)
    
    # Convert purchase_history from string to list
    df['purchase_history'] = df['purchase_history'].apply(lambda x: list(map(int, x.split(', '))))
    
    # Convert DataFrame rows to dictionary format for each customer
    customers = []
    for index, row in df.iterrows():
        customer = {
            'id': row['id'],
            'name': row['name'],
            'age': row['age'],
            'income': row['income'],
            'purchase_history': row['purchase_history'],
            'engagement_score': row['engagement_score'],
            'lifetime_value': row['lifetime_value'],
            'sentiment': row['sentiment'],
        }
        customers.append(customer)
    
    return customers

# Step 1: Customer Segmentation using KMeans
def customer_segmentation(customers):
    customer_data = np.array([[cust['age'], cust['income'], sum(cust['purchase_history']), cust['engagement_score']] for cust in customers])
    scaler = StandardScaler()
    customer_data = scaler.fit_transform(customer_data)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(customer_data)
    
    for i, customer in enumerate(customers):
        customer['cluster'] = clusters[i]
    
    return customers

# Step 2: Churn Prediction Model
def churn_prediction(customers):
    customer_data = np.array([[cust['age'], cust['income'], sum(cust['purchase_history']), cust['engagement_score']] for cust in customers])
    churn_labels = np.array([1 if cust['engagement_score'] < 60 else 0 for cust in customers])  # Arbitrary label
    
    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(customer_data, churn_labels)
    
    # Predict churn for all customers
    churn_predictions = model.predict(customer_data)
    
    for i, customer in enumerate(customers):
        customer['churn_likelihood'] = churn_predictions[i]
    
    return customers

# Step 3: Personalized Product Recommendations using KNN
def product_recommendations(customers):
    customer_data = np.array([[cust['age'], cust['income'], sum(cust['purchase_history']), cust['engagement_score']] for cust in customers])
    
    knn = NearestNeighbors(n_neighbors=3)
    knn.fit(customer_data)
    
    recommendations = {}
    for i, customer in enumerate(customers):
        _, indices = knn.kneighbors([customer_data[i]])
        recommended_customers = [customers[idx]['name'] for idx in indices[0]]
        recommendations[customer['name']] = recommended_customers
    
    return recommendations

# Step 4: LTV Prediction using Neural Network
def build_ltv_model(customers):
    customer_data = np.array([[cust['age'], cust['income'], sum(cust['purchase_history']), cust['engagement_score']] for cust in customers])
    lifetime_values = np.array([cust['lifetime_value'] for cust in customers])
    
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(customer_data.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(customer_data, lifetime_values, epochs=10, batch_size=8)
    
    return model

# Step 5: Sentiment Analysis (Customer Feedback)
def sentiment_analysis(feedback):
    blob = TextBlob(feedback)
    sentiment = 'positive' if blob.sentiment.polarity > 0 else 'negative' if blob.sentiment.polarity < 0 else 'neutral'
    return sentiment

# Step 6: Customer Behavior Forecasting using LSTM (Mock example)
def customer_behavior_forecasting(customers):
    # For simplicity, use random behavior as placeholder for LSTM model prediction
    forecast = {cust['name']: random.choice(['High', 'Medium', 'Low']) for cust in customers}
    return forecast

# Step 7: Dynamic Pricing Model (Mock)
def dynamic_pricing():
    return random.randint(10, 100)

# Step 8: Customer-based Campaign (Mock)
def create_campaign(customers):
    campaign = {cust['name']: f"Exclusive offer for {cust['name']}, 10% off!" for cust in customers}
    return campaign

@app.route("/", methods=["GET", "POST"])
def index():
    # Load customer data from CSV
    customers = load_customer_data()
    
    if request.method == "POST":
        # Process form inputs
        name = request.form.get("name")
        feedback = request.form.get("feedback")
        
        # Functionality:
        # Customer Segmentation (Step 1)
        segmented_customers = customer_segmentation(customers)
        
        # Churn Prediction (Step 2)
        churned_customers = churn_prediction(segmented_customers)
        
        # Product Recommendations (Step 3)
        recommendations = product_recommendations(churned_customers)
        
        # LTV Prediction (Step 4)
        ltv_model = build_ltv_model(churned_customers)
        ltv_prediction = ltv_model.predict([[35, 50000, 1200, 85]])  # Random example input
        
        # Sentiment Analysis (Step 5)
        sentiment = sentiment_analysis(feedback)
        
        # Customer Behavior Forecasting (Step 6)
        behavior_forecast = customer_behavior_forecasting(churned_customers)
        
        # Dynamic Pricing (Step 7)
        pricing = dynamic_pricing()
        
        # Campaign Creation (Step 8)
        campaign = create_campaign(churned_customers)
        
        # Pass results to the template
        return render_template(
            "index.html", 
            customers=churned_customers, 
            recommendations=recommendations, 
            ltv_prediction=ltv_prediction[0], 
            sentiment=sentiment,
            behavior_forecast=behavior_forecast,
            pricing=pricing,
            campaign=campaign
        )

    return render_template("index.html", customers=customers)

if __name__ == "__main__":
    app.run(debug=True)
