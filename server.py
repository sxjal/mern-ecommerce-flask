from flask import Flask, request, jsonify
import joblib
import nltk
import ssl
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Workaround for SSL certificate issue

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('wordnet')

def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Load the joblib pipeline
pipeline = joblib.load('/Users/sajal/Desktop/Sids Code/App/mern-ecommerce/flask-server/modelreview.joblib')

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data['features']  # Expecting an array of features
    predictions = [bool(pipeline.predict([feature])[0]) for feature in features]
    return jsonify({'predictions': predictions})

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     features = data['features']  # Expecting an array of features
#     prediction = pipeline.predict([features])
#     return jsonify({'isFake': bool(prediction[0])})

if __name__ == '__main__':
    app.run(port=5000)



# import string
# from flask import Flask, request, jsonify
# import joblib
# import nltk 
# # import stopwords
# nltk.download('stopwords')
# nltk.download('wordnet')

# # def text_process(text):
# #     # Your text processing logic here
# #     pass

# def text_process(review):
#     nopunc = [char for char in review if char not in string.punctuation]
#     nopunc = ''.join(nopunc)
#     return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')] # type: ignore

# # Load the joblib pipeline
# pipeline = joblib.load('/Users/sajal/Desktop/Sids Code/App/mern-ecommerce/flask-server/modelreview.joblib')

# app = Flask(__name__)

# @app.route('/hel', methods=['POST'])
# def solve():
#     print("hello")
#     return ''


# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     features = data['features']  # Expecting an array of features
#     prediction = pipeline.predict([features])
#     return jsonify({'isFake': bool(prediction[0])})

# if __name__ == '__main__':
#     app.run(port=5000)
