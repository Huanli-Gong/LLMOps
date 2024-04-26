from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from transformers import pipeline
from prometheus_client import start_http_server, Counter

app = Flask(__name__)
api = Api(app)

# Load the question answering model using transformers
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Define a counter metric for Prometheus
correct_requests_counter = Counter('correct_http_requests', 'Total HTTP requests that were processed correctly.', ['endpoint'])

class QuestionAnswer(Resource):
    def post(self):
        try:
            # Parse the input data
            data = request.get_json()
            question = data['question']
            context = data['context']
            
            # Use the model to get an answer
            result = qa_pipeline(question=question, context=context)
            answer = result['answer']
            score = result['score']
            start = result['start']
            end = result['end']
            
            # Increment the Prometheus counter
            correct_requests_counter.labels(endpoint='/qa').inc()
            
            # Return the answer along with additional details
            return jsonify(answer=answer, score=score, start=start, end=end)
        except Exception as e:
            # Log internal server errors and return error message
            return str(e), 500

# Setup the API resource routing
api.add_resource(QuestionAnswer, '/qa')

if __name__ == '__main__':
    # Start the Prometheus metrics server on port 8000
    start_http_server(8000)
    # Run the application
    app.run(host='0.0.0.0', port=8080)