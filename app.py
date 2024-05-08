import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# Define mapping dictionary for categorical features
categorical_mapping = {
    'online_order': {0: 'no', 1: 'yes'},
    'book_table': {0: 'no', 1: 'yes'},
    # Add more mappings for other categorical features as needed
}

def preprocess_features(features):
    # Convert categorical features back to numerical form using the mapping dictionary
    cuisine_encoder = pickle.load(open('cuisines_encoder.pkl', 'rb'))
    location_encoder = pickle.load(open('location_encoder.pkl', 'rb'))
    rest_type_encoder = pickle.load(open('rest_type_encoder.pkl', 'rb'))

    for feature_name, mapping in categorical_mapping.items():
        if feature_name in features:
            # Convert 'Yes'/'No' to 1/0
            if features[feature_name] == 'Yes':
                features[feature_name] = 1
            elif features[feature_name] == 'No':
                features[feature_name] = 0
            else:
                # Use 0 as default if value not found in mapping
                features[feature_name] = mapping.get(features[feature_name], 0)

    features['cuisines'] = cuisine_encoder.transform([features['cuisines']])[0]
    features['location'] = location_encoder.transform([features['location']])[0]
    features['rest_type'] = rest_type_encoder.transform([features['rest_type']])[0]
    menu_item = {
        'menu_item': 5047
    }

    features.update(menu_item)

    return features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = request.form.to_dict()
    processed_features = preprocess_features(features)

    # Convert features to numpy array for prediction
    final_features = np.array(list(processed_features.values())).reshape(1, -1)

    # Predict using the model
    prediction = model.predict(final_features)

    # features = [int(x) for x in request.form.values()]
    # final_features = [np.array(features)]
    # prediction = model.predict(final_features)

    output = round(prediction[0], 1)

    return render_template('index.html', prediction_text='Your Rating is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)