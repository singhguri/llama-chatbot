from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import pipeline

print('Loading the model...')

# Load the model and set up the pipeline
model_id = "chuanli11/Llama-3.2-3B-Instruct-uncensored"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print('Initializing API...')
# Initialize FastAPI app
app = FastAPI()

print('Defining req/res schemas...')
# Define request and response schemas
class Message(BaseModel):
    role: str
    content: str

class RequestBody(BaseModel):
    messages: list[Message]
    max_new_tokens: int = 50  # Default token limit

# Define endpoint to generate text
@app.post("/generate-text/")
async def generate_text(request: RequestBody):
    try:
        # Format the input messages
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Generate text
        outputs = pipe(
            messages,
            max_new_tokens=request.max_new_tokens,
        )
        return {"responses": outputs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app with: `uvicorn filename:chatbot --reload`
