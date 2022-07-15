FROM python:3.8-slim

COPY . /app
WORKDIR /app

RUN pip install -r /app/requirements.txt

EXPOSE 80
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "80"]
