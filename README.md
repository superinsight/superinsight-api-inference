# SuperInsight Inference API
This is a RESTful API that you can use inference GPT models, support CPU or GPU


## Environment Variables 
Variable | Usage | Required | Default
--- | --- | --- | ---
EXPORT_GCP_STORAGE_BUCKET | If you like to export models to GCP bucket, include the bucket name here | False | None
EXPORT_GCP_STORAGE_FOLDER | If you like to export models to GCP bucket, include the bucket name here | False | None
GOOGLE_APPLICATION_CREDENTIALS | If you like to export models to GCP bucket, you will need to include your credentials | False | None

## Available Models
In additional to finetune models created using the [SuperInsight FineTuning API](https://github.com/superinsight/superinsight-api-finetuning), base models can also be used.
model | Summary
--- | ---
xxxxxxxxxxxx | Finetuned models created 
gpt-neo-125m | The `EleutherAI/gpt-neo-125M` model.
gpt-neo-1.3b | The `EleutherAI/gpt-neo-1.3B` model.
gpt-neo-2.7b  | The `EleutherAI/gpt-neo-2.7B` model.
gpt-j-6b | The `EleutherAI/gpt-j-6B` model.

# Development

## What you will need
* Python 3 installed

## Technologies
* Python 3
* HuggingFace
* FastAPI

## Install Dependencies and run API
```
pip install -r requirements.txt
```
```
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

## Run API with docker
```
docker run -d -p 8080:8080 --name superinsight-api-inference superinsight/superinsight-api-inference:latest
```