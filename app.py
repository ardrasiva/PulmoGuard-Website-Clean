from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="pneumonia_vgg19_classifier.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = ['NORMAL', 'PNEUMONIA']  # Adjust if your labels are different


def preprocess_image(image):
    img = image.resize((128, 128))  # Match model input size
    img_array = np.array(img).astype('float32') / 255.0

    # Check if input is grayscale or RGB
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ---------- ROUTES ----------

@app.route('/')
def start():
    return render_template('getstarted.html')


@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/symptom', methods=['GET', 'POST'])
def symptom_checker():
    result = None
    high_risk = False

    if request.method == 'POST':
        answers = request.form.to_dict()
        yes_count = sum(1 for v in answers.values() if v == 'yes')

        if yes_count == 0:
            result = "No concerning symptoms detected."
            high_risk = False
        else:
            result = f"Warning! You have {yes_count} symptom(s) that may indicate pneumonia."
            high_risk = True

    return render_template('symptomcheck.html', result=result, high_risk=high_risk)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction="No file uploaded")

    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')
    processed_image = preprocess_image(image)

    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_index = int(np.argmax(output_data))
    result = labels[predicted_index]

    return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
