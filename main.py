from fastapi import FastAPI
from typing import Optional
from typing import List
from pydantic import BaseModel
import os, sys
app = FastAPI()
import os
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
model_directory = os.environ.get('MODEL_DIRECTORY', 'EleutherAI/gpt-neo-125M')
tokenizer_gpt = AutoTokenizer.from_pretrained(model_directory)
model_gpt = AutoModelForCausalLM.from_pretrained(model_directory).to(torch_device)

@app.on_event('startup')
async def startup_event():
    print('app startup')

def generate(tokenizer, model, context, temperature, repetition_penalty, num_return_sequences, length):
    prompt = context 
    input_ids = tokenizer(prompt, return_tensors="pt").to(torch_device).input_ids
    output_length = len(tokenizer.encode(prompt)) + length
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_length=output_length,
        return_full_text=False,
        num_return_sequences=num_return_sequences
    )
    gen_texts = tokenizer.batch_decode(gen_tokens)
    return gen_texts

@app.get('/')
async def read_root():
    return {'version': '1.0.0'}

@app.get('/predict')
async def predict(context: Optional[str] = None, temperature: Optional[float] = 1.0, repetition_penalty: Optional[float] = 1.0, max_length: Optional[int] = 50, sequences: Optional[int] = 1):
    try:
      predictions = []
      if context is not None:
        texts = generate(tokenizer_gpt, model_gpt, context, temperature,  repetition_penalty, sequences, max_length)
        for text in texts:
            predictions.append(text)
      return { 'predictions': predictions }
    except:
      print('Unexpected error:', sys.exc_info()[0])
      return { 'status':'error'}