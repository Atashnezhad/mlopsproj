# Use official Python 3.12 slim image
FROM python:3.12-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install poetry (optional but recommended)
RUN pip install --upgrade pip && \
    pip install poetry

# Copy project files
COPY pyproject.toml poetry.lock* ./
COPY src/ /app/src/

# Configure Poetry
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-interaction --no-ansi

# Expose application port
EXPOSE 7000

# Use uvicorn to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7000"]
