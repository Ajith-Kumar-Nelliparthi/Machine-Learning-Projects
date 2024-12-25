FROM python:3.13.0-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]


RUN pipenv install --system --deploy

COPY ["fetal_health_predict.py", "fetal_health_load.py", "./"]

EXPOSE 5000

ENTRYPOINT ["gunicorn","--bind=0.0.0.0:5000","fetal_health_predict:app"]