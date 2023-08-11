from flask import Flask, request, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import logging
from flask_cors import CORS  # Import CORS

def process_input_image(img):
    img = img.resize((224, 224))  # adjust size as needed
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255
    return img_array

def predict_food_class(img, model, category_list):
    current_img_processed = process_input_image(img)
    current_pred_array = model.predict(current_img_processed)
    current_pred = np.argmax(current_pred_array, axis=1)[0]
    predicted_food_item = category_list[current_pred]
    return predicted_food_item


app = Flask(__name__)
CORS(app)  # Apply CORS to your app

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model = keras.models.load_model('mobilenetv2.h5')
        app.logger.info("Started Processing ")
        category_list = []

        # Open the text file in read mode
        with open('labels_list.txt', 'r') as file:
            for line in file:
                pass
                value = line.strip()
                category_list.append(value)

        app.logger.info("Retrieved Labels ")

        # Get the uploaded image from the request
        image = request.files['image']

        img = Image.open(image)

        predicted_food = predict_food_class(img, model, category_list)

        response = {
            'predicted food': predicted_food,
        }
        app.logger.info("predicted_food :", predicted_food)

        return jsonify(response), 200

    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(debug=True)

# %%
