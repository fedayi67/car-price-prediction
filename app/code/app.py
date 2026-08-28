from dash import Dash, html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import joblib


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load("car_price_model.pkl")


# ============================================================
# EXTRACT CATEGORIES FROM THE TRAINED MODEL
# ============================================================

preprocessor = model.named_steps["preprocessor"]

categorical_pipeline = preprocessor.named_transformers_["cat"]

encoder = categorical_pipeline.named_steps["encoder"]

categories = encoder.categories_


# Extract category lists
brand_options = sorted(categories[0])
fuel_options = sorted(categories[1])
seller_type_options = sorted(categories[2])
transmission_options = sorted(categories[3])


# ============================================================
# CURRENCY CONVERSION RATES
#
# All rates are relative to 1 Indian Rupee (INR).
# These are approximate predefined rates.
# ============================================================

currency_rates = {
    "INR": 1.00,
    "THB": 0.42,
    "USD": 0.012,
    "EUR": 0.011
}


currency_symbols = {
    "INR": "₹",
    "THB": "฿",
    "USD": "$",
    "EUR": "€"
}


# ============================================================
# CREATE DASH APPLICATION
# ============================================================

app = Dash(__name__)

app.title = "Car Price Predictor"


# ============================================================
# REUSABLE STYLE DEFINITIONS
# ============================================================

PAGE_STYLE = {
    "backgroundColor": "#f4f6f8",
    "minHeight": "100vh",
    "padding": "40px 20px",
    "fontFamily": "Arial, sans-serif"
}


CONTAINER_STYLE = {
    "maxWidth": "1000px",
    "margin": "0 auto"
}


CARD_STYLE = {
    "backgroundColor": "white",
    "padding": "30px",
    "borderRadius": "12px",
    "boxShadow": "0 4px 15px rgba(0,0,0,0.08)",
    "marginBottom": "25px"
}


INPUT_STYLE = {
    "width": "100%",
    "padding": "12px",
    "borderRadius": "6px",
    "border": "1px solid #cccccc"
}


LABEL_STYLE = {
    "fontWeight": "bold",
    "marginBottom": "8px",
    "display": "block"
}


# ============================================================
# HELPER FUNCTION FOR NUMERICAL INPUTS
# ============================================================

def numerical_input(label, component_id, placeholder):

    return html.Div(
        [

            html.Label(
                label,
                style=LABEL_STYLE
            ),

            dcc.Input(
                id=component_id,
                type="number",
                placeholder=placeholder,
                style=INPUT_STYLE
            )

        ],
        style={
            "flex": "1",
            "minWidth": "250px"
        }
    )


# ============================================================
# APPLICATION LAYOUT
# ============================================================

app.layout = html.Div(

    [

        html.Div(

            [

                # ====================================================
                # HEADER
                # ====================================================

                html.Div(

                    [

                        html.H1(
                            "🚗 Car Price Predictor",
                            style={
                                "marginBottom": "10px"
                            }
                        ),

                        html.P(
                            "Estimate the selling price of a vehicle using machine learning.",
                            style={
                                "fontSize": "18px",
                                "color": "#555555"
                            }
                        )

                    ],

                    style={
                        "textAlign": "center",
                        "marginBottom": "30px"
                    }

                ),


                # ====================================================
                # INSTRUCTIONS CARD
                # ====================================================

                html.Div(

                    [

                        html.H3(
                            "How to Use the Application"
                        ),

                        html.P(
                            "Enter the available information about the vehicle. "
                            "You may leave any field empty if the information is unavailable. "
                            "Missing values will be handled automatically by the model."
                        )

                    ],

                    style=CARD_STYLE

                ),


                # ====================================================
                # VEHICLE INFORMATION CARD
                # ====================================================

                html.Div(

                    [

                        html.H2(
                            "Vehicle Information"
                        ),


                        html.Div(

                            [

                                # BRAND

                                html.Div(

                                    [

                                        html.Label(
                                            "Brand",
                                            style=LABEL_STYLE
                                        ),

                                        dcc.Dropdown(
                                            id="brand",
                                            options=[
                                                {
                                                    "label": x,
                                                    "value": x
                                                }
                                                for x in brand_options
                                            ],
                                            placeholder="Select a brand",
                                            clearable=True
                                        )

                                    ],

                                    style={
                                        "flex": "1",
                                        "minWidth": "250px"
                                    }

                                ),


                                # FUEL

                                html.Div(

                                    [

                                        html.Label(
                                            "Fuel Type",
                                            style=LABEL_STYLE
                                        ),

                                        dcc.Dropdown(
                                            id="fuel",
                                            options=[
                                                {
                                                    "label": x,
                                                    "value": x
                                                }
                                                for x in fuel_options
                                            ],
                                            placeholder="Select fuel type",
                                            clearable=True
                                        )

                                    ],

                                    style={
                                        "flex": "1",
                                        "minWidth": "250px"
                                    }

                                )

                            ],

                            style={
                                "display": "flex",
                                "gap": "20px",
                                "flexWrap": "wrap",
                                "marginBottom": "20px"
                            }

                        ),


                        html.Div(

                            [

                                # SELLER TYPE

                                html.Div(

                                    [

                                        html.Label(
                                            "Seller Type",
                                            style=LABEL_STYLE
                                        ),

                                        dcc.Dropdown(
                                            id="seller_type",
                                            options=[
                                                {
                                                    "label": x,
                                                    "value": x
                                                }
                                                for x in seller_type_options
                                            ],
                                            placeholder="Select seller type",
                                            clearable=True
                                        )

                                    ],

                                    style={
                                        "flex": "1",
                                        "minWidth": "250px"
                                    }

                                ),


                                # TRANSMISSION

                                html.Div(

                                    [

                                        html.Label(
                                            "Transmission",
                                            style=LABEL_STYLE
                                        ),

                                        dcc.Dropdown(
                                            id="transmission",
                                            options=[
                                                {
                                                    "label": x,
                                                    "value": x
                                                }
                                                for x in transmission_options
                                            ],
                                            placeholder="Select transmission",
                                            clearable=True
                                        )

                                    ],

                                    style={
                                        "flex": "1",
                                        "minWidth": "250px"
                                    }

                                )

                            ],

                            style={
                                "display": "flex",
                                "gap": "20px",
                                "flexWrap": "wrap"
                            }

                        )

                    ],

                    style=CARD_STYLE

                ),


                # ====================================================
                # VEHICLE SPECIFICATIONS CARD
                # ====================================================

                html.Div(

                    [

                        html.H2(
                            "Vehicle Specifications"
                        ),


                        # ROW 1

                        html.Div(

                            [

                                numerical_input(
                                    "Kilometers Driven",
                                    "km_driven",
                                    "Example: 50000"
                                ),

                                numerical_input(
                                    "Mileage (km/litre)",
                                    "mileage",
                                    "Example: 20.5"
                                )

                            ],

                            style={
                                "display": "flex",
                                "gap": "20px",
                                "flexWrap": "wrap",
                                "marginBottom": "20px"
                            }

                        ),


                        # ROW 2

                        html.Div(

                            [

                                numerical_input(
                                    "Engine Capacity (CC)",
                                    "engine",
                                    "Example: 1248"
                                ),

                                numerical_input(
                                    "Maximum Power (BHP)",
                                    "max_power",
                                    "Example: 74"
                                )

                            ],

                            style={
                                "display": "flex",
                                "gap": "20px",
                                "flexWrap": "wrap",
                                "marginBottom": "20px"
                            }

                        ),


                        # ROW 3

                        html.Div(

                            [

                                numerical_input(
                                    "Car Age (Years)",
                                    "car_age",
                                    "Example: 5"
                                ),

                                numerical_input(
                                    "Number of Seats",
                                    "seats",
                                    "Example: 5"
                                )

                            ],

                            style={
                                "display": "flex",
                                "gap": "20px",
                                "flexWrap": "wrap",
                                "marginBottom": "20px"
                            }

                        ),


                        # ROW 4

                        html.Div(

                            [

                                numerical_input(
                                    "Number of Previous Owners",
                                    "owner",
                                    "Example: 1"
                                )

                            ],

                            style={
                                "display": "flex",
                                "gap": "20px",
                                "flexWrap": "wrap"
                            }

                        )

                    ],

                    style=CARD_STYLE

                ),


                # ====================================================
                # CURRENCY SELECTION
                # ====================================================

                html.Div(

                    [

                        html.H2(
                            "Display Currency"
                        ),

                        html.P(
                            "Select the currency in which you want to see the estimated price."
                        ),

                        dcc.Dropdown(

                            id="currency",

                            options=[
                                {
                                    "label": "🇹🇭 Thai Baht (THB)",
                                    "value": "THB"
                                },

                                {
                                    "label": "🇮🇳 Indian Rupee (INR)",
                                    "value": "INR"
                                },

                                {
                                    "label": "🇺🇸 US Dollar (USD)",
                                    "value": "USD"
                                },

                                {
                                    "label": "🇪🇺 Euro (EUR)",
                                    "value": "EUR"
                                }
                            ],

                            value="THB",

                            clearable=False,

                            style={
                                "maxWidth": "400px"
                            }

                        )

                    ],

                    style=CARD_STYLE

                ),


                # ====================================================
                # PREDICT BUTTON
                # ====================================================

                html.Div(

                    [

                        html.Button(

                            "Predict Price",

                            id="predict-button",

                            n_clicks=0,

                            style={

                                "backgroundColor": "#2563eb",

                                "color": "white",

                                "border": "none",

                                "padding": "15px 40px",

                                "fontSize": "18px",

                                "fontWeight": "bold",

                                "borderRadius": "8px",

                                "cursor": "pointer",

                                "width": "100%"

                            }

                        )

                    ],

                    style={
                        "marginBottom": "25px"
                    }

                ),


                # ====================================================
                # PREDICTION OUTPUT
                # ====================================================

                html.Div(

                    id="prediction-output"

                )

            ],

            style=CONTAINER_STYLE

        )

    ],

    style=PAGE_STYLE

)


# ============================================================
# PREDICTION CALLBACK
# ============================================================

@app.callback(

    Output(
        "prediction-output",
        "children"
    ),

    Input(
        "predict-button",
        "n_clicks"
    ),


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

    State("currency", "value")

)


def predict_price(

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

    currency

):


    # Do nothing before button is clicked

    if n_clicks == 0:

        return ""


    # ========================================================
    # CREATE DATAFRAME FROM USER INPUT
    # ========================================================

    input_data = pd.DataFrame({

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
        "owner": [owner]

    })


    # Convert empty values to NaN so the model's
    # SimpleImputer can handle them.

    input_data = input_data.replace({None: np.nan})


    # ========================================================
    # MAKE PREDICTION
    # ========================================================

    prediction_log = model.predict(
        input_data
    )


    # Convert back from logarithmic scale

    prediction_inr = np.exp(
        prediction_log[0]
    )


    # ========================================================
    # CURRENCY CONVERSION
    # ========================================================

    conversion_rate = currency_rates[
        currency
    ]


    prediction_converted = (
        prediction_inr * conversion_rate
    )


    currency_symbol = currency_symbols[
        currency
    ]


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return html.Div(

        [

            html.H3(
                "Estimated Vehicle Price",
                style={
                    "marginBottom": "10px"
                }
            ),


            html.H1(

                f"{currency_symbol} {prediction_converted:,.2f}",

                style={
                    "fontSize": "40px",
                    "margin": "10px 0"
                }

            ),


            html.P(

                f"Displayed in {currency}. "
                "Currency conversions are approximate and use predefined exchange rates.",

                style={
                    "color": "#666666"
                }

            )

        ],

        style={

            "backgroundColor": "white",

            "padding": "30px",

            "borderRadius": "12px",

            "boxShadow": "0 4px 15px rgba(0,0,0,0.08)",

            "textAlign": "center",

            "marginBottom": "30px"

        }

    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8050,
        debug=False
    )