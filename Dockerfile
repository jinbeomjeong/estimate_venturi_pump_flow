# 1. Select Python base image (basic kitchen environment for cooking meal kits)
FROM python:3.9-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system packages required for GPIO and compilation
RUN apt-get update && apt-get install -y gcc python3-dev && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements file and install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy source code and model files
COPY inference_modbus.py .
COPY models/ ./models/

# 6. Create an empty folder to store log data
RUN mkdir -p /app/log_data

# 7. Command to run by default when the container starts (start cooking!)
CMD ["python", "app.py"]