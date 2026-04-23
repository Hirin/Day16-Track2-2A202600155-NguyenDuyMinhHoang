FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY train_model.py .
COPY app.py .

# Train the model at build time (bakes model into image)
RUN python train_model.py

# Expose port
EXPOSE 8000

# Run the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
