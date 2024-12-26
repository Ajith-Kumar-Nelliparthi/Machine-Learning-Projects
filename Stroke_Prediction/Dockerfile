FROM python:3.13.0-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]


RUN pipenv install --system --deploy

COPY ["stroke_predict.py", "stroke_load.py", "./"]

EXPOSE 5000

ENTRYPOINT ["gunicorn","--bind=0.0.0.0:5000","stroke_predict:app"]