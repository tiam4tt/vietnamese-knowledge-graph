FROM python:3.11.14-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --quiet --no-cache-dir -r requirements.txt

# Copy files to the container
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port for Streamlit application
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]