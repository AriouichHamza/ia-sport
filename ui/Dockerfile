FROM python:3.11-slim

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
# We copy this specifically first to leverage Docker caching
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the entire project
COPY . .

# 4. Critical: Add the root to PYTHONPATH so ui/app_gradio.py can see /core, /utils, etc.
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

EXPOSE 7860

# 5. Use the path relative to the WORKDIR
CMD ["python", "-u", "ui/app_gradio.py"]