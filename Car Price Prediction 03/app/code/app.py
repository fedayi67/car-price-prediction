from dash import Dash, html, dcc, Input, Output, State

import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import LogisticRegression  # noqa: F401  (needed so joblib can unpickle the artifact)


# ============================================================
# LOAD TRAINED ARTIFACT
# ============================================================

MODEL_PATH = os.path.join(os.path.dirname(__file__), "car_price_classifier_a3.pkl")

artifact = joblib.load(MODEL_PATH)

preprocessor = artifact["preprocessor"]
model = artifact["model"]
price_bin_labels = artifact["price_bin_labels"]
numerical_features = artifact["numerical_features"]
categorical_features = artifact["categorical_features"]

CLASS_NAMES = {
    0: "Budget",
    1: "Economy",
    2: "Premium",
    3: "Luxury",
}


# ============================================================
# EXTRACT CATEGORIES FROM THE TRAINED PREPROCESSOR
# ============================================================

encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
categories = encoder.categories_

brand_options = sorted(categories[0])
fuel_options = sorted(categories[1])
seller_type_options = sorted(categories[2])
transmission_options = sorted(categories[3])


# ============================================================
# CURRENCY CONVERSION RATES (relative to 1 Indian Rupee, INR)
# ============================================================

currency_rates = {"INR": 1.00, "THB": 0.42, "USD": 0.012, "EUR": 0.011}
currency_symbols = {"INR": "₹", "THB": "฿", "USD": "$", "EUR": "€"}


# ============================================================
# CREATE DASH APPLICATION
# ============================================================

app = Dash(__name__)
app.title = "Car Price Class Predictor"


# ============================================================
# REUSABLE STYLE DEFINITIONS (matches the A1/A2 app for a consistent look)
# ============================================================

PAGE_STYLE = {
    "backgroundColor": "#f4f6f8",
    "minHeight": "100vh",
    "padding": "40px 20px",
    "fontFamily": "Arial, sans-serif",
}

CONTAINER_STYLE = {"maxWidth": "1000px", "margin": "0 auto"}

CARD_STYLE = {
    "backgroundColor": "white",
    "padding": "30px",
    "borderRadius": "12px",
    "boxShadow": "0 4px 15px rgba(0,0,0,0.08)",
    "marginBottom": "25px",
}

INPUT_STYLE = {
    "width": "100%",
    "padding": "12px",
    "borderRadius": "6px",
    "border": "1px solid #cccccc",
}

LABEL_STYLE = {"fontWeight": "bold", "marginBottom": "8px", "display": "block"}

FIELD_STYLE = {"flex": "1", "minWidth": "250px"}

ROW_STYLE = {"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginBottom": "20px"}


def numerical_input(label, component_id, placeholder):
    return html.Div(
        [
            html.Label(label, style=LABEL_STYLE),
            dcc.Input(id=component_id, type="number", placeholder=placeholder, style=INPUT_STYLE),
        ],
        style=FIELD_STYLE,
    )


def dropdown_input(label, component_id, options, placeholder):
    return html.Div(
        [
            html.Label(label, style=LABEL_STYLE),
            dcc.Dropdown(
                id=component_id,
                options=[{"label": x, "value": x} for x in options],
                placeholder=placeholder,
                clearable=True,
            ),
        ],
        style=FIELD_STYLE,
    )


# ============================================================
# APPLICATION LAYOUT
# ============================================================

app.layout = html.Div(
    [
        html.Div(
            [
                # HEADER
                html.Div(
                    [
                        html.H1("🚗 Car Price Class Predictor", style={"marginBottom": "10px"}),
                        html.P(
                            "Estimate a vehicle's price bracket (Budget / Economy / Premium / "
                            "Luxury) using a multinomial Logistic Regression model trained from "
                            "scratch (Assignment 3).",
                            style={"fontSize": "18px", "color": "#555555"},
                        ),
                    ],
                    style={"textAlign": "center", "marginBottom": "30px"},
                ),
                # INSTRUCTIONS
                html.Div(
                    [
                        html.H3("How to Use the Application"),
                        html.P(
                            "Enter the vehicle's information below and click Predict. Every "
                            "field is required for this model."
                        ),
                    ],
                    style=CARD_STYLE,
                ),
                # VEHICLE INFORMATION
                html.Div(
                    [
                        html.H2("Vehicle Information"),
                        html.Div(
                            [
                                dropdown_input("Brand", "brand", brand_options, "Select a brand"),
                                dropdown_input("Fuel Type", "fuel", fuel_options, "Select fuel type"),
                            ],
                            style=ROW_STYLE,
                        ),
                        html.Div(
                            [
                                dropdown_input(
                                    "Seller Type", "seller_type", seller_type_options, "Select seller type"
                                ),
                                dropdown_input(
                                    "Transmission", "transmission", transmission_options, "Select transmission"
                                ),
                            ],
                            style=ROW_STYLE,
                        ),
                    ],
                    style=CARD_STYLE,
                ),
                # VEHICLE SPECIFICATIONS
                html.Div(
                    [
                        html.H2("Vehicle Specifications"),
                        html.Div(
                            [
                                numerical_input("Kilometers Driven", "km_driven", "e.g. 50000"),
                                numerical_input("Mileage (km/litre)", "mileage", "e.g. 20.5"),
                            ],
                            style=ROW_STYLE,
                        ),
                        html.Div(
                            [
                                numerical_input("Engine Capacity (CC)", "engine", "e.g. 1248"),
                                numerical_input("Maximum Power (BHP)", "max_power", "e.g. 74"),
                            ],
                            style=ROW_STYLE,
                        ),
                        html.Div(
                            [
                                numerical_input("Car Age (Years)", "car_age", "e.g. 5"),
                                numerical_input("Number of Seats", "seats", "e.g. 5"),
                            ],
                            style=ROW_STYLE,
                        ),
                        html.Div(
                            [numerical_input("Number of Previous Owners", "owner", "e.g. 1")],
                            style=ROW_STYLE,
                        ),
                    ],
                    style=CARD_STYLE,
                ),
                # CURRENCY
                html.Div(
                    [
                        html.H2("Display Currency"),
                        html.P("Select the currency in which you want to see the predicted price range."),
                        dcc.Dropdown(
                            id="currency",
                            options=[
                                {"label": "🇮🇳 Indian Rupee (INR)", "value": "INR"},
                                {"label": "🇹🇭 Thai Baht (THB)", "value": "THB"},
                                {"label": "🇺🇸 US Dollar (USD)", "value": "USD"},
                                {"label": "🇪🇺 Euro (EUR)", "value": "EUR"},
                            ],
                            value="THB",
                            clearable=False,
                        ),
                    ],
                    style=CARD_STYLE,
                ),
                html.Button(
                    "Predict Price Class",
                    id="predict-button",
                    n_clicks=0,
                    style={
                        "width": "100%",
                        "padding": "16px",
                        "backgroundColor": "#3b5bdb",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "8px",
                        "fontSize": "18px",
                        "fontWeight": "bold",
                        "cursor": "pointer",
                    },
                ),
                html.Div(id="prediction-output"),
            ],
            style=CONTAINER_STYLE,
        )
    ],
    style=PAGE_STYLE,
)


# ============================================================
# PREDICTION CALLBACK
# ============================================================

@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    State("brand", "value"),
    State("fuel", "value"),
    State("seller_type", "value"),
    State("transmission", "value"),
    State("km_driven", "value"),
    State("mileage", "value"),
    State("engine", "value"),
    State("max_power", "value"),
    State("car_age", "value"),
    State("seats", "value"),
    State("owner", "value"),
    State("currency", "value"),
)
def predict_price_class(
    n_clicks,
    brand,
    fuel,
    seller_type,
    transmission,
    km_driven,
    mileage,
    engine,
    max_power,
    car_age,
    seats,
    owner,
    currency,
):
    if n_clicks == 0:
        return ""

    input_data = pd.DataFrame(
        {
            "brand": [brand],
            "fuel": [fuel],
            "seller_type": [seller_type],
            "transmission": [transmission],
            "km_driven": [km_driven],
            "mileage": [mileage],
            "engine": [engine],
            "max_power": [max_power],
            "car_age": [car_age],
            "seats": [seats],
            "owner": [owner],
        }
    )
    input_data = input_data.replace({None: np.nan})

    if input_data.isnull().any().any():
        return html.Div(
            "Please fill in all vehicle details before predicting.",
            style={"color": "red", "fontWeight": "bold", "textAlign": "center", "padding": "20px"},
        )

    # Same preprocessing pipeline fitted in the notebook: median/most-frequent
    # imputation + StandardScaler + OneHotEncoder(handle_unknown="ignore").
    input_processed = preprocessor.transform(input_data[numerical_features + categorical_features])

    predicted_class = int(model.predict(input_processed)[0])
    probabilities = model.predict_proba(input_processed)[0]

    price_range = price_bin_labels[predicted_class]
    conversion_rate = currency_rates[currency]
    currency_symbol = currency_symbols[currency]

    low_str, high_str = [p.strip().replace(",", "") for p in price_range.split("-")]
    low_converted = float(low_str) * conversion_rate
    high_converted = float(high_str) * conversion_rate

    probability_bars = []
    for c in sorted(CLASS_NAMES):
        pct = probabilities[c] * 100
        probability_bars.append(
            html.Div(
                [
                    html.Div(
                        f"{CLASS_NAMES[c]} (class {c}): {pct:.1f}%",
                        style={"fontSize": "13px", "marginBottom": "4px"},
                    ),
                    html.Div(
                        style={
                            "width": f"{max(pct, 2)}%",
                            "backgroundColor": "#3b5bdb" if c == predicted_class else "#adb5bd",
                            "height": "10px",
                            "borderRadius": "5px",
                        }
                    ),
                ],
                style={"marginBottom": "10px"},
            )
        )

    return html.Div(
        [
            html.H3("Predicted Price Class", style={"marginBottom": "10px"}),
            html.H1(
                f"{CLASS_NAMES[predicted_class]} (class {predicted_class})",
                style={"fontSize": "36px", "margin": "10px 0"},
            ),
            html.H2(
                f"{currency_symbol} {low_converted:,.0f} - {currency_symbol} {high_converted:,.0f}",
                style={"fontSize": "26px", "color": "#3b5bdb", "margin": "10px 0"},
            ),
            html.P(
                "Predicted class probabilities:",
                style={"marginTop": "20px", "fontWeight": "bold"},
            ),
            html.Div(probability_bars, style={"maxWidth": "400px", "margin": "0 auto"}),
            html.P(
                "This is an estimated price bracket based on a multinomial Logistic "
                "Regression model trained from scratch on historical listings.",
                style={"color": "#666666", "marginTop": "20px"},
            ),
        ],
        style={"textAlign": "center", "padding": "20px"},
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8060, debug=False)
