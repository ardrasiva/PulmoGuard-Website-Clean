from flask import Flask, render_template, request, redirect, url_for, session
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
app.secret_key = "SECRET_KEY"  # Use a secure secret key in production
users = {}
print(users)
# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="pneumonia_vgg19_classifier.tflite")  # Change to your model name
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

@app.route('/')
def start():
    return render_template('getstarted.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route("/logout")
def logout():
    session.pop("username", None)  # remove username from session
    return redirect(url_for("login"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        # Basic validation
        if password != confirm_password:
            return "Passwords do not match. <a href='/signup'>Try again</a>"

        if username in users:
            return "User already exists. <a href='/signup'>Try again</a>"

    # Save user
        users[username] = {"email": email, "password": password}

    # Log user in
        session["username"] = username

    # Redirect to home page
        
        
        return redirect(url_for("home"))
    return render_template('signup.html')

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # Check if user exists
        if username in users and users[username]["password"] == password:
            session["username"] = username
            return redirect(url_for("home"))
        else:
            return "Invalid username or password. <a href='/login'>Try again</a>"

    return render_template('login.html')
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
    app.run(debug=True,port=5000)