# Use a lightweight Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /pixal

# Install system dependencies (customize as needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# Copy project files
COPY . .

# Install editable package
RUN pip install --upgrade pip
RUN pip install -e .[gpu]

# Run the validation pipeline on container start
ENTRYPOINT ["pixal"]
CMD []