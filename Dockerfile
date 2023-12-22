FROM python:3.10.3-slim-buster

WORKDIR /workspace

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8080

ENV HOST 0.0.0.0

CMD ["python", "app.py"]
