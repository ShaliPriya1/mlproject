import pandas as pd
import random

# Function to generate random customer data
def generate_customer_data(n=100):
    data = []
    for i in range(n):
        customer = {
            'name': f"Customer_{i+1}",
            'age': random.randint(18, 70),
            'income': random.randint(30000, 100000),
            'purchase_history': ', '.join([str(random.randint(1, 100)) for _ in range(random.randint(3, 10))]),
            'engagement_score': random.randint(0, 100),
            'lifetime_value': random.randint(1000, 10000),
        }
        data.append(customer)
    return pd.DataFrame(data)

# Function to load the generated customer data
def load_customer_data():
    df = generate_customer_data()
    df.to_csv("customer_data.csv", index=False)
    return df

if __name__ == "__main__":
    load_customer_data()
