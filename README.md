# Final Project Group 11: Question-Answering Service with  DistilBERT model

## Team Member
- Hathaway Liu
- Zaolin Zhang
- Xingzhi Xie
- Huanli Gong

## Purpose

The purpose of this project is to develop a robust, scalable web service that utilizes machine learning to provide answers to questions based on a provided context. This service leverages an open-source machine learning model for extractive question answering, specifically using the DistilBERT model pre-trained on the SQuAD dataset.

## Demo Video
![Demo Video](https:XXXX)

## Introduction

The Question-Answering API allows users to submit a question along with a context paragraph and receive a specific answer extracted from the context. This service is built on the Rust programming language, providing high performance and safety. The model serving is handled by `rust_bert`, which is a Rust library port of the Hugging Face's `transformers` library.


## Model Details

Our service utilizes a pre-trained DistilBERT model that has been fine-tuned on the SQuAD (Stanford Question Answering Dataset). DistilBERT is a smaller, faster, cheaper, and lighter version of BERT that retains 97% of its predecessor's language understanding capabilities but with fewer parameters, making it ideal for our web-based service.

#### API Usage

Clients can interact with the API by sending a POST request to `/qa` endpoint. The request should include a JSON payload containing `question` and `context` keys. The service processes this request, performs inference using the loaded model, and returns the extracted answer as a JSON response.

##### Example Request 1:

```bash
curl -X POST http://localhost:8080/qa \
-H "Content-Type: application/json" \
-d '{"question": "Where is your hometown?", "context": "My hometown is a beautiful city. Its name is Guangzhou, which is a beautiful place with lots of flowers"}'
```

##### Example Answer 1:
```bash
{"answer":"Guangzhou","end":54,"score":0.9698584079742432,"start":45}
```

![Function overview](https://gitlab.com/final-group-11/final-project-group-11/-/wikis/uploads/935e235f40e18d43a043ce81ffa6964f/Screenshot_2024-04-25_at_2.21.47_PM.png)

##### Example Request 2 with Empty Input:

```bash
curl -X POST http://localhost:8080/qa \
-H "Content-Type: application/json" \
-d '{"question": "", "context": "My hometown is a beautiful city. Its name is Guangzhou, which is a beautiful place with lots of flowers"}'
```

##### Example Answer 2 with Empty Input:
```bash
{"error": "`question` cannot be empty"}
```

![Function overview](https://raw.githubusercontent.com/LeoZhangzaolin/photos/main/WechatIMG50.jpg)


## Detailed Steps
## Step 1: Install Dependencies

Install the required Python packages:

```bash
pip install flask flask-restful transformers prometheus_client torch torchvision torchaudio
```
## Step 2: Create and update `main.py`

The service utilizes a pre-trained machine learning model via the `transformers` library to answer questions provided through a REST API. Key features include robust logging, error handling, and real-time metrics collection using Prometheus.

### Dependencies
- Flask: Provides the web framework.
- Flask-RESTful: Simplifies the creation of REST APIs with resource-based classes.
- transformers: Utilized for loading and executing the pre-trained question-answering model.
- prometheus_client: Enables metrics collection for monitoring service performance.

### Setting Up the Flask Application
```python
from flask import Flask, request, jsonify
from flask_restful import Api

app = Flask(__name__)
api = Api(app)
```

### Loading the Model
The service uses the `transformers` library to load a pre-trained question-answering model.
```python
from transformers import pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
```

### Metrics with Prometheus
Prometheus metrics are configured to monitor the number of correct HTTP requests:
```python
from prometheus_client import start_http_server, Counter

correct_requests_counter = Counter(
    'correct_http_requests', 
    'Total HTTP requests that were processed correctly.', 
    ['endpoint']
)
```
![3](https://raw.githubusercontent.com/LeoZhangzaolin/photos/main/WechatIMG120.jpg)

### QuestionAnswer Resource Class
This class handles POST requests by performing model inference to answer questions, providing additional details about the answer.
```python
from flask_restful import Resource

class QuestionAnswer(Resource):
    def post(self):
        try {
            data = request.get_json()
            question = data['question']
            context = data['context']
            result = qa_pipeline(question=question, context=context)
            answer = result['answer']
            score = result['score']
            start = result['start']
            end = result['end']
            
            correct_requests_counter.labels(endpoint='/qa').inc()
            
            return jsonify(answer=answer, score=score, start=start, end=end)
        } except Exception as e {
            return str(e), 500
        }
```

### Initializing the API and Metrics Server
The endpoints are setup and the Prometheus server starts on port 8000.
```python
api.add_resource(QuestionAnswer, '/qa')

if __name__ == '__main__':
    start_http_server(8000)
    app.run(host='0.0.0.0', port=8080)
```

## Step 3: Run Locally 

To run the application locally:

```bash
python main.py
```

This will start the Flask server on `localhost:8080` and Prometheus metrics server on `localhost:8000`.

## Step 4: Create the Dockerfile

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Note: You should ideally specify exact versions of these packages to ensure reproducibility.
RUN pip install --no-cache-dir flask flask-restful transformers prometheus_client torch torchvision torchaudio

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV NAME World

# Start the Prometheus metrics server
CMD ["start_http_server", "8000"]

# Run main.py when the container launches
CMD ["python", "./src/main.py"]
```
## Step 5: Build Docker Image

Navigate to the directory containing your Dockerfile and build your Docker image:

```bash
docker build -t user_name/qa-api:latest .
```

## Step 6: Push the Image to Docker Hub

Run the following code in the terminal and replace user_name with your real Docker user name
```bash
docker tag user_name/qa-api:latest user_name/qa-api:latest
docker push user_name/qa-api:latest
```

## Step 7: Deploy to Google Kubernetes Engine

### 1. Install Google Cloud SDK
To interact with Google Cloud resources, you need to install the Google Cloud SDK. Use Homebrew to install it if using MacOS:
```bash
brew install --cask google-cloud-sdk
```

### 2. Initialize Google Cloud SDK
After installation, initialize the Google Cloud SDK to configure your authentication credentials and set the default project:
```bash
gcloud init
```

### 3. Install Kubernetes Command-Line Tool (kubectl)
`kubectl` is a command-line tool that allows you to run commands against Kubernetes clusters. Install it using:
```bash
gcloud components install kubectl
```

### 4. Authenticate with Google Cloud
Ensure that your authentication credentials are set up correctly for Google Cloud:
```bash
gcloud auth login
```

### 5. Create a Kubernetes Cluster
Create a cluster on Google Kubernetes Engine. Replace `<cluster-name>` and `<zone>` with your specific details:
```bash
gcloud container clusters create <cluster-name> --zone <zone>
```

### 6. Configure kubectl to Use Your Cluster
Configure `kubectl` to use the cluster you just created:
```bash
gcloud container clusters get-credentials <cluster-name> --zone  <zone> --project <project-id>
```

### 7. Write the Kubernetes Configuration YAML File
Create a YAML file (`kubernetes.yaml`) that describes your deployment and service. This file includes specifications for replicas, container images, ports, etc.
```
---
apiVersion: "apps/v1"
kind: "Deployment"
metadata:
  name: "qa-api-deployment"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: "qa-api"
  template:
    metadata:
      labels:
        app: "qa-api"
    spec:
      containers:
      - name: "qa-api"
        image: "hathawayliu/qa-api:latest"
        ports:
        - containerPort: 8080
---
apiVersion: "v1"
kind: "Service"
metadata:
  name: "metrics-server"
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8080
  selector:
    app: qa-api

```
### 8. Deploy Your Application
Apply the YAML configuration to your cluster, deploying your application, replace <yaml-file> with your real yaml file
```bash
kubectl apply -f <yaml-file>
```

### 9. Verify Deployment
Check the status of your deployment:
```bash
kubectl get deployments
```

![1](https://raw.githubusercontent.com/LeoZhangzaolin/photos/main/WechatIMG37.jpg)

Or you could go to Google Cloud to view the status of your deployment:
![deployment](https://gitlab.com/final-group-11/final-project-group-11/-/wikis/uploads/7c02afa64ce22796c541b5b947dd92cf/Screenshot_2024-04-25_at_8.32.22_PM.png)

![2](https://raw.githubusercontent.com/LeoZhangzaolin/photos/main/WechatIMG41.jpg)
### 10. Verify Service
Check the created services to ensure your application is accessible:
```bash
kubectl get svc
```

## Step 8: Set Up CI/CD Pipeline

- Configure your CI/CD pipeline using your preferred CI/CD platform (e.g., GitLab CI).
- Ensure your pipeline handles the lifecycle of building, testing, and deploying your application.
- ```.gitlab-ci.yml```
```
stages:
  - build
  - deploy

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker login -u "$CI_REGISTRY_USER" -p "$CI_REGISTRY_PASSWORD"
    - docker build -t hathawayliu/qa-api:latest .
    - docker push hathawayliu/qa-api:latest

deploy:
  stage: deploy
  image: google/cloud-sdk:latest
  script:
    - echo $GKE_SERVICE_KEY > gcloud-service-key.json
    - gcloud auth activate-service-account --key-file gcloud-service-key.json
    - gcloud container clusters get-credentials qa-api --zone us-west1 --project ids721
    - kubectl apply -f kubernetes.yaml
    - kubectl get deployments
    - kubectl get pods -l app=qa-api
    - kubectl get svc
  only:
    - main

```
