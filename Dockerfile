# Use the official Python image from the Docker Hub
FROM python:3.12.1-slim

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

COPY req.txt .


RUN pip install --no-cache-dir -r req.txt && \
    rm -rf /root/.cache

# Copy the rest of the application code into the container
COPY . .

# Command to run the application
CMD ["python", "main.py"]