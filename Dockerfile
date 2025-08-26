# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Streamlit entrypoint
CMD ["streamlit", "run", "src/shap_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
