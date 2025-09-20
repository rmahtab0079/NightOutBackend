FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["/bin/sh","-c"]
CMD ["exec uvicorn application:app --host 0.0.0.0 --port ${PORT:-8080}"]