# Use an official Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (better layer caching)
COPY code/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and trained model
COPY code/ .

# Dash application port
EXPOSE 8060

# Start the application
CMD ["python", "app.py"]
