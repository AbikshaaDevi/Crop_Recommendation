from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model bundle saved by notebook
model_bundle = joblib.load("model.pkl")
model = model_bundle["model"]
le = model_bundle["label_encoder"]
features = model_bundle.get("features", ["N","P","K","temperature","humidity","ph","rainfall"])

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            # read inputs in the same order as features
            vals = []
            for f in features:
                v = request.form.get(f)
                if v is None or v.strip() == "":
                    raise ValueError(f"Missing value for {f}")
                # convert to float — N,P,K are integers in dataset but float is fine
                vals.append(float(v))
            X = np.array(vals).reshape(1, -1)
            pred_id = model.predict(X)
            pred_label = le.inverse_transform(pred_id)[0]
            result = f"Recommended crop: {pred_label}"
        except Exception as e:
            result = f"Error: {str(e)}"

    # render the HTML form; pass features for template generation if needed
    return render_template("index.html", result=result)

if __name__ == "__main__":
    # debug=False in production; use debug=True while developing
    app.run(host="0.0.0.0", port=5000, debug=True)
