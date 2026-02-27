FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e ".[server]" 2>/dev/null || pip install --no-cache-dir -e .

EXPOSE 8080

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "agent_observability.server.app", "--host", "0.0.0.0", "--port", "8080"]
