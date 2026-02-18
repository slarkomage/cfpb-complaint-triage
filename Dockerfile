FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN pip install poetry

COPY pyproject.toml poetry.lock README.md ./
COPY cfpb_complaint_triage ./cfpb_complaint_triage
COPY configs ./configs
COPY data ./data
COPY artifacts ./artifacts
COPY plots ./plots

RUN poetry config virtualenvs.create false && poetry install --only main --no-interaction --no-ansi

EXPOSE 8000

CMD ["uvicorn", "cfpb_complaint_triage.production.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
