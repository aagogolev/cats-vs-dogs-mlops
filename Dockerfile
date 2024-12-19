FROM python:3.11.8-slim

ARG GITHUB_REPOSITORY
ARG GITHUB_SHA

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY templates/ templates/

RUN apt-get update && apt-get install -y curl && mkdir -p /app/models/latest

COPY <<EOF /app/download_model.sh
#!/bin/bash
curl -L -o /app/models/latest/model.keras "https://github.com/${GITHUB_REPOSITORY}/releases/download/model-${GITHUB_SHA}/model.keras"
EOF

RUN chmod +x /app/download_model.sh

EXPOSE 8000

CMD ["/bin/bash", "-c", "/app/download_model.sh && python -m uvicorn src.app:app --host 0.0.0.0 --port 8000"]
