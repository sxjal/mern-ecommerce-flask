import joblib

# Define the missing function
def text_process(text):
    return text.lower()  # Example processing (modify as needed)

# Load the joblib model
model = joblib.load("modelreview.joblib")

print("Model loaded successfully!")
