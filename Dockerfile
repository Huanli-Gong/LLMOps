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
RUN pip install --no-cache-dir --default-timeout=100 flask flask-restful
RUN pip install --no-cache-dir --default-timeout=100 transformers prometheus_client
RUN pip install --no-cache-dir --default-timeout=100 torch torchvision torchaudio

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV NAME World

# Start the Prometheus metrics server
CMD ["start_http_server", "8000"]

# Run main.py when the container launches
CMD ["python", "./src/main.py"]