from fastapi import FastAPI, HTTPException
import random
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import requests
app = FastAPI()

load_dotenv()

api_key = "AIzaSyDW0X1gO6uVSPkYIa3R6sjRwNQrz-afYU0"

places_of_interest = ["restaurant", "bar", "cafe", "coffee_shop", "beach", "hiking_area",
                      "historical_landmark", "movie_theater", "video_arcade", "karaoke", "night_club", "opera_house",
                      "dance_hall", "amusement_park"]

# Model for location data
class LocationData(BaseModel):
    latitude: float
    longitude: float
    accuracy: Optional[float] = None
    altitude: Optional[float] = None
    speed: Optional[float] = None
    speedAccuracy: Optional[float] = None
    heading: Optional[float] = None
    timestamp: Optional[str] = None


class Suggestion(BaseModel):
    suggestion: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.post("/random", response_model=Suggestion)
async def get_random_suggestion(location: LocationData):
    print(f"Received location data: {location}")
    search_nearby_places(location)
    random_suggestions = [
        "Take a walk at a Overpeck park",
        "Go watch Bellarina at AMC Ridgefield Park",
        "Go to the fair at the American Dream Mall",
    ]

    new_suggestion = random.choice(random_suggestions)

    return Suggestion(suggestion=new_suggestion)

def search_nearby_places(location: LocationData):
    endpoint_url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName"
    }
    json_payload = {
        "includedTypes": places_of_interest,
        "maxResultCount": 20,
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": location.latitude,
                    "longitude": location.longitude},
                "radius": 1000.0
            }
        }
    }

    try:
        response = requests.post(endpoint_url, json=json_payload, headers=headers)
        print(response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




