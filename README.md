# SuperInsight Inference API
This is a RESTful API that you can use inference GPT models, support CPU or GPU

# Development

## What you will need
* Python3 installed

## Technologies
* Python3
* HuggingFace
* FastAPI

## Install Dependencies and run API
```
pip install -r requirements.txt
```
```
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```