FROM python:3.11.0-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the port Flask runs on
EXPOSE 8000

# Run the Flask app using Gunicorn
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:8000", "app:app"]