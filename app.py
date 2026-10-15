from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("model/model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]
    proba = model.predict_proba([news])[0]
    prediction = model.predict([news])[0]
    confidence = round(max(proba) * 100, 2)
    result = f"{'Real News ✅' if prediction == 1 else 'Fake News ❌'} (Confidence: {confidence}%)"
    return render_template("index.html", prediction=result, news_input=news)

if __name__ == "__main__":
    app.run(debug=True)
