# Crop Recommendation Web App

A Flask-based web app that predicts the most suitable crop to grow based on environmental and soil parameters using a trained machine learning model.

---

## Overview

This project predicts the best crop using features like:
- Nitrogen (N), Phosphorus (P), Potassium (K) levels
- Temperature (°C)
- Humidity (%)
- pH level
- Rainfall (mm)

All the input metrics are processed through a trained ML model built with scikit-learn behind a user-friendly Flask interface.

---

## Project Structure

```
Crop_Recommendation/
├── app.py
├── crop_data.csv
├── model.pkl
├── requirements.txt
└── templates/
    └── index.html
```

---

## Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/AbikshaaDevi/Crop_Recommendation.git
   cd Crop_Recommendation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   python app.py
   ```
   Open your browser and go to `http://127.0.0.1:5000/`.

4. **Get a prediction**
   Enter values for N, P, K, temperature, humidity, pH, and rainfall — the app will recommend the most suitable crop.

---

## Technologies Used

- **Python** – Backend logic and model handling
- **Flask** – Web application framework
- **Scikit-learn** – Machine learning model
- **HTML Templates** – User interface

---

## Deployment Options

You can deploy this app on:
- **Render**
- **Heroku**
- Or any web hosting service that supports Python and Flask

---

## Contributions & Future Enhancements

- Test and compare other ML models (e.g., Random Forest, XGBoost)
- Enhance UI styling using CSS or a frontend framework
- Incorporate real-time weather APIs for more accurate recommendations
- Add dataset validation and user input checks

---

## About the Author

Created by **Abikshaa Devi** — blending machine learning and web development to solve real-world agriculture challenges.

---

## License

This project is open-source—feel free to use and build upon it!
