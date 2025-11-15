from fastapi import FastAPI
import pickle
from pydantic import BaseModel

# Load model
with open("app/models/spam_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Email Spam Classification API")

class Email(BaseModel):
    text: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(email: Email):
    prediction = model.predict([email.text])[0]
    label = "spam" if prediction == 1 else "ham"
    
    return {
        "input": email.text,
        "prediction": label
    }
