import random
import pandas as pd

# Function to generate mock customer data
def generate_mock_customer_data(num_customers=100):
    customers = []
    for i in range(num_customers):
        customer = {
            'id': i,
            'name': f"Customer {i+1}",
            'age': random.randint(18, 65),
            'income': random.randint(30000, 120000),
            'purchase_history': [random.randint(100, 1000) for _ in range(5)],  # 5 random purchases
            'engagement_score': random.randint(50, 100),
            'lifetime_value': random.randint(500, 5000),
            'sentiment': random.choice(['positive', 'neutral', 'negative']),
        }
        customers.append(customer)
    return customers

# Function to save customer data to a CSV file
def save_to_csv(customers, filename="customer_data.csv"):
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(customers)
    # Convert the purchase_history column from list to string format
    df['purchase_history'] = df['purchase_history'].apply(lambda x: ', '.join(map(str, x)))
    # Save DataFrame to CSV
    df.to_csv(filename, index=False)
    print(f"Customer data saved to {filename}")

if __name__ == "__main__":
    # Generate mock custome
