# Use an official Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# Note: --host 0.0.0.0 is required for the container to be accessible externally
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]