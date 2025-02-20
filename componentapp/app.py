import joblib
from flask import Flask, render_template, request

# Load the trained model from a file
def load_model(filename):
    try:
        model = joblib.load(filename)
        return model
    except FileNotFoundError:
        print(f"Error: Model file '{filename}' not found.")
        return None

# Flask app setup
app = Flask(__name__)

# Load the model (make sure 'model.pkl' exists before running the Flask app)
model_filename = 'model.pkl'
model = load_model(model_filename)

# Check if the model was loaded successfully
if model is None:
    print("Model not loaded successfully. Exiting...")
    exit(1)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the input summary from the form
        user_summary = request.form['summary']
        
        # Predict the component
        predicted_component = model.predict([user_summary])[0]
        
        return render_template('index.html', predicted_component=predicted_component)
    
    return render_template('index.html', predicted_component=None)

if __name__ == '__main__':
    app.run(debug=True)
