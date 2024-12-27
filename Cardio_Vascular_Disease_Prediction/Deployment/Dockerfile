FROM python:3.13.0-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["cardio_predict.py", "cardio_load.py", "./"]

EXPOSE 9999

ENTRYPOINT ["gunicorn","--bind=0.0.0.0:9999","cardio_predict:app"]