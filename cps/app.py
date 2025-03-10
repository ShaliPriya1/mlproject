import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.neighbors import NearestNeighbors

# Customer Profile Class to store customer data
class CustomerProfile:
    def __init__(self, customer_id, name, demographics, purchase_history, engagement_score, lifetime_value):
        self.customer_id = customer_id
        self.name = name
        self.demographics = demographics
        self.purchase_history = purchase_history
        self.engagement_score = engagement_score
        self.lifetime_value = lifetime_value

    def update_engagement_score(self, new_score):
        self.engagement_score = new_score

    def update_lifetime_value(self, amount):
        self.lifetime_value += amount

    def get_total_spend(self):
        return sum(self.purchase_history)

# Data Preparation (Simulated Data)
def generate_customer_data(num_customers=100):
    names = [f"Customer {i}" for i in range(1, num_customers + 1)]
    demographics = np.random.rand(num_customers, 3)  # Random demographic data: age, income, region
    purchase_history = [np.random.randint(50, 500, 10).tolist() for _ in range(num_customers)]  # 10 random purchases
    engagement_scores = np.random.randint(50, 100, num_customers)
    lifetime_values = np.random.randint(200, 5000, num_customers)
    
    customers = []
    for i in range(num_customers):
        customer = CustomerProfile(
            customer_id=i,
            name=names[i],
            demographics=demographics[i],
            purchase_history=purchase_history[i],
            engagement_score=engagement_scores[i],
            lifetime_value=lifetime_values[i]
        )
        customers.append(customer)
    
    return customers

# Machine Learning for Customer Segmentation (Clustering)
def customer_segmentation(customers):
    customer_data = np.array([np.concatenate([cust.demographics, [cust.get_total_spend(), cust.engagement_score]]) for cust in customers])
    scaler = StandardScaler()
    customer_data = scaler.fit_transform(customer_data)
    
    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.fit_predict(customer_data)
    
    for i, customer in enumerate(customers):
        customer.cluster = clusters[i]
    
    return clusters

# Predict Customer Churn Using Logistic Regression (Predictive Analytics)
def train_churn_prediction(customers):
    customer_data = np.array([np.concatenate([cust.demographics, [cust.get_total_spend(), cust.engagement_score]]) for cust in customers])
    churn_labels = np.array([1 if cust.engagement_score < 60 else 0 for cust in customers])  # Example: Churn if engagement score < 60
    
    X_train, X_test, y_train, y_test = train_test_split(customer_data, churn_labels, test_size=0.3, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Churn Prediction Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    
    return model

# Personalized Product Recommendation (Collaborative Filtering)
def recommend_products(customers, num_products=5):
    # Using K-Nearest Neighbors (KNN) for collaborative filtering-style recommendations
    customer_data = np.array([np.concatenate([cust.demographics, [cust.get_total_spend(), cust.engagement_score]]) for cust in customers])
    
    knn = NearestNeighbors(n_neighbors=num_products)
    knn.fit(customer_data)
    
    recommendations = {}
    for i, customer in enumerate(customers):
        _, indices = knn.kneighbors([customer_data[i]])
        recommended_customers = [customers[idx] for idx in indices[0]]
        recommended_products = [f"Product {i}" for i in range(1, num_products + 1)]
        recommendations[customer.name] = (recommended_customers, recommended_products)
    
    return recommendations

# Deep Learning for Customer Lifetime Value Prediction (Regression)
def build_ltv_model(customers):
    customer_data = np.array([np.concatenate([cust.demographics, [cust.get_total_spend(), cust.engagement_score]]) for cust in customers])
    lifetime_values = np.array([cust.lifetime_value for cust in customers])
    
    # Build a simple neural network model for LTV prediction
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(customer_data.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(customer_data, lifetime_values, epochs=10, batch_size=8)
    
    return model

# Example Implementation
if __name__ == "__main__":
    # Step 1: Generate synthetic customer data
    customers = generate_customer_data(num_customers=100)
    
    # Step 2: Perform customer segmentation using KMeans clustering
    customer_segmentation(customers)
    for customer in customers[:5]:
        print(f"{customer.name} belongs to Cluster {customer.cluster}")

    # Step 3: Train churn prediction model using Logistic Regression
    churn_model = train_churn_prediction(customers)
    
    # Step 4: Provide personalized product recommendations using KNN
    recommendations = recommend_products(customers, num_products=3)
    for customer_name, (recommended_customers, recommended_products) in recommendations.items():
        print(f"Recommendations for {customer_name}: {recommended_products}")
    
    # Step 5: Build and train a deep learning model to predict Customer Lifetime Value (LTV)
    ltv_model = build_ltv_model(customers)
    sample_customer_data = np.array([np.concatenate([customers[0].demographics, [customers[0].get_total_spend(), customers[0].engagement_score]])])
    predicted_ltv = ltv_model.predict(sample_customer_data)
    print(f"Predicted LTV for {customers[0].name}: {predicted_ltv[0][0]}")
