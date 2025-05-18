from fastapi import FastAPI
import random
from pydantic import BaseModel
application = FastAPI()

class Suggestion(BaseModel):
    suggestion: str

@application.get("/")
async def root():
    return {"message": "Hello World"}

@application.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@application.get("/random", response_model=Suggestion)
async def get_random_suggestion():
    print("This endpoint got called")
    random_suggestions = [
        "Take a walk at a Overpeck park",
        "Go watch Bellarina at AMC Ridgefield Park",
        "Go to the fair at the American Dream Mall",
    ]

    new_suggestion = random.choice(random_suggestions)

    return Suggestion(suggestion=new_suggestion)




