from fastapi import FastAPI
import random
from pydantic import BaseModel
app = FastAPI()

class Suggestion(BaseModel):
    suggestion: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/random", response_model=Suggestion)
async def get_random_suggestion():
    print("This endpoint got called")
    random_suggestions = [
        "Take a walk at a Overpeck park",
        "Go watch Bellarina at AMC Ridgefield Park",
        "Go to the fair at the American Dream Mall",
    ]

    new_suggestion = random.choice(random_suggestions)

    return Suggestion(suggestion=new_suggestion)




