FROM python:3.13.0-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]


RUN pipenv install --system --deploy

COPY ["mushroom_predict.py", "mushroom_load.py", "./"]

EXPOSE 9999

ENTRYPOINT ["gunicorn","--bind=0.0.0.0:9999","mushroom_predict:app"]