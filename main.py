import os, sys, shutil
import torch
from fastapi import FastAPI
from typing import Optional
from typing import List
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from google.cloud import storage
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles
app = FastAPI(docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
bucketName= os.getenv("EXPORT_GCP_STORAGE_BUCKET", None)
bucketFolder= os.getenv("EXPORT_GCP_STORAGE_FOLDER", "")
modelRepository= os.getenv("EXPORT_MODEL_REPOSITORY", "models")
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
models = []
tokenizers = []
mapping = {}
version = "0.1.1"

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
    model_directory = "{}/{}".format(modelRepository, model_id)
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
        downloadObject(bucket_name, blob.name, "{}/{}".format(modelRepository,blob.name))

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
async def HealthCheck():
    return {"version": version}

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title= "SuperInsight Inference API Documentation",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_favicon_url="/static/favicon.png"
    )

@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title="SuperInsight Inference API Documentation",
        redoc_js_url="/static/redoc.standalone.js",
        redoc_favicon_url="/static/favicon.png"
    )

class CompletionsRequest(BaseModel):
    context: str
    model: str
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    max_length: int = 50
    sequences: int = 1

@app.post("/completions")
async def completions(req: CompletionsRequest):
    """
    Pass in a prompt a the finetuned model id and get completions back.

    Base Model can also be used by passing the following model id
    * gpt-neo-125m
    * gpt-neo-1.3b
    * gpt-neo-2.7b
    * gpt-j-6b
    """
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

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="SuperInsight Inference API Documentation",
        version=version,
        description="API to inference base or finetuned GPT models",
        routes=app.routes
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi