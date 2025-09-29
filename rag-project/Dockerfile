# Use official Python image as base
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY rag-project/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY rag-project/ .

# Expose port (Django default)
EXPOSE 8000

# Collect static files (if needed)
# RUN python manage.py collectstatic --noinput

# Run migrations and start server
CMD ["gunicorn", "rag.wsgi:application", "--bind", "0.0.0.0:8000"]
