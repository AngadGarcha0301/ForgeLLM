FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Set environment variables
ENV PORT=8000
ENV DEMO_MODE=true

# Expose port
EXPOSE 8000

# Run the app - use shell form to expand $PORT
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
