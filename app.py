from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)

model = load_model("mythical_draw_model.keras")
categories = ["dragon", "angel", "mermaid", "castle", "flying saucer"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_data = data["image"].split(",")[1]
    correct_answer = data.get("correct_answer", "")

    #decode the image
    decoded = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(decoded)).convert('L')

    #resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img)

    #normalize to [0, 1]
    img_array = img_array.astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    #predict
    pred = model.predict(img_array, verbose=0)[0]
    label = categories[np.argmax(pred)]

    #check if correct
    correct = label == correct_answer

    print(f"\nPrompt: {correct_answer}")
    print(f"Predicted: {label}")
    for i, cat in enumerate(categories):
        print(f"  {cat}: {pred[i] * 100:.1f}%")

    return jsonify({
        "prediction": label,
        "correct": correct
    })


if __name__ == "__main__":
    app.run(debug=True)
