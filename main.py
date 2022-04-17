from fastapi import FastAPI
from typing import Optional
from typing import List
from pydantic import BaseModel
import os, sys, shutil
app = FastAPI()
import os
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from google.cloud import storage

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
bucketName= os.getenv("EXPORT_GCP_STORAGE_BUCKET", None)
bucketFolder= os.getenv("EXPORT_GCP_STORAGE_FOLDER", "")
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
models = []
tokenizers = []
mapping = {}

class CompletionsRequest(BaseModel):
    context: str
    model: str
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    max_length: int = 50
    sequences: int = 1

@app.on_event("startup")
async def startup_event():
    print("app startup")

def getModelAndTokenizers(model_id):
    if mapping.get(model_id) is None:
        index = loadModelAndTokenizers(model_id)
    else:
        index = mapping[model_id]
    if index < 0:
        return None, None
    return tokenizers[index], models[index]

def downloadModel(model_id):
    model_directory = "models/{}".format(model_id)
    bucketPath = "{}{}".format(bucketFolder, model_id)
    downloadFolder(bucketName, bucketPath, model_directory)
    return model_directory

def downloadObject(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def downloadFolder(bucket_name, source_folder, destination_folder):
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
    os.mkdir(destination_folder)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_folder)
    for blob in blobs:
        downloadObject(bucket_name, blob.name, "models/{}".format(blob.name))

def loadModelAndTokenizers(model_id):
    if model_id == "gpt-neo-125m":
        model_directory = "EleutherAI/gpt-neo-125M"
    elif model_id == "gpt-neo-1.3b":
        model_directory = "EleutherAI/gpt-neo-1.3B"
    elif model_id == "gpt-neo-2.7b":
        model_directory = "EleutherAI/gpt-neo-2.7B"
    elif model_id == "gpt-j-6b":
        model_directory = "EleutherAI/gpt-j-6B"
    elif bucketName is not None:
        model_directory = downloadModel(model_id)
    else:
        return -1
    index = len(models)
    mapping[model_id] = index
    tokenizers.append(AutoTokenizer.from_pretrained(model_directory)) 
    models.append(AutoModelForCausalLM.from_pretrained(model_directory).to(torch_device)) 
    return index 

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

@app.get("/")
async def read_root():
    return {"version": "1.0.0"}
 
@app.post("/completions")
async def completions(req: CompletionsRequest):
    try:
      choices = []
      if req.context is not None:
        tokenizer, model = getModelAndTokenizers(req.model)
        if tokenizer is None or model is None:
            return { "status":"error", "message": "model not found"}
        texts = generate(tokenizer, model, req.context, req.temperature, req.repetition_penalty, req.sequences, req.max_length)
        for text in texts:
            choices.append(text)
      return { "model": req.model,  "choices": choices }
    except:
      print("Unexpected error:", sys.exc_info()[0])
      return { "status":"error"}