FROM python:3.10-slim

# Install system dependencies (BLAST+ etc.)
RUN apt-get update && \
    apt-get install -y ncbi-blast+ git && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project files into container
COPY class_classifier.py .
COPY Modelo_Leptospira ./Modelo_Leptospira

# Expose Gradio port
EXPOSE 7860

# Command to run your script
CMD ["python", "class_classifier.py"]
