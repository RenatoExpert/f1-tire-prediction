"""
API for predicting F1 lap times
"""

import pandas as pd

from fastapi import FastAPI, Request # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
import uvicorn # type: ignore

from pathlib import Path
from models.ml_logic.model import load_pipeline
import joblib

from pydantic import BaseModel

app = FastAPI(
    title="F1 Lap Time Prediction API",
    description="API for lap time prediction in Formula 1 races",
    version="1.0"
)

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class PredictParameters(BaseModel):
    driver: str             # Driver's name // e.g ver
    lap_number: float       # lap number
    stint: float            # stint number
    compound: str           # compound: tire compount // e.g soft
    tyre_life: float        # tyre_life in laps
    position: float         # current position in the race
    air_temp: float         # air temperature ºC
    humidity: float         # relative humidity %
    pressure: float         # atmospheric pressure (hPa)
    # poderia usar um bool pro rainfall
    # is_raining: bool
    rainfall: float         # rain (0 = no, 1 = yes)
    track_temp: float       # track temperature in Celsius
    event_year: int         # year of the event
    grandprix: str          # name of the grand prix // e.g bahrain
    # se o parametro for opcional, basta colocar "| None = None"
    # grandprix: str | None = None

# Predicts lap time based on race parameters
@app.post("/predict")
async def predict(params: PredictParameters):
    # é possivel transformar o params em dict
    # params.dict()
    input_data = {
        'Driver': [params.driver],
        'LapNumber': [lap_number],
        'Stint': [stint],
        'Compound': [compound],
        'TyreLife': [tyre_life],
        'Position': [position],
        'AirTemp': [air_temp],
        'Humidity': [humidity],
        'Pressure': [pressure],
        'Rainfall': [rainfall],
        'TrackTemp': [track_temp],
        'Event_Year': [event_year],
        'GrandPrix': [grandprix]
    }
    X_pred = pd.DataFrame(input_data)
    model = load_pipeline()

    # se ajustar os termos para ficarem identicos a classe PredictParameters,
    # voce pode simplesmente chamar diretamente no model, pois ele ja vem em dict
    # nao precisaria da linha "input_data ="
    #
    # prediction = model.predict(params)

    model_response = model.predict(X_pred)
    # é boa pratica usar async/await pra chamadas como essa
    # model_response = await model.predict(X_pred)
    # no caso, model.predict() teria que ser async tbm
    # async def predict (alguma coisa): // <- la dentro de model
    prediction = float(model_response[0])
    return { "predicted_lap_time": prediction, input_parameters: params }

@app.get("/")
def index():
    return {'message': 'Hello'}

