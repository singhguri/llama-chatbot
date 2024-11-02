from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


print('Loading the model...')
# Load the model and set up the pipeline
model_id = "chuanli11/Llama-3.2-3B-Instruct-uncensored"
device = 0 if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )

# print("Tying weights...")
# # Ensure model weights are tied
# pipe.model.tie_weights()

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
        # Prepare the input messages for the model
        input_text = " ".join([msg.content for msg in request.messages if msg.role == "user"])
        
        # Encode the input
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        # Generate text
        outputs = model.generate(inputs["input_ids"], max_new_tokens=request.max_new_tokens)
        
        # Decode the output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"response": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app with: `uvicorn filename:chatbot --reload`
