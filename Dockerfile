FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY blog_writer_agent ./blog_writer_agent
COPY evals ./evals

EXPOSE 8080

# Launch the Google ADK API/Web Server on Cloud Run / Vertex AI Agent Engine
CMD ["adk", "api_server", "--host", "0.0.0.0", "--port", "8080", "blog_writer_agent"]
