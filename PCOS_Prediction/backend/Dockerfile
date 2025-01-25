FROM python:3.13.0-slim

RUN pip install pipenv

RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]


RUN pipenv install --system --deploy

COPY ["pcos_predict.py", "pcos_load.py", "./"]
COPY templates/index.html ./templates/



EXPOSE 8000

ENTRYPOINT ["gunicorn","--bind=0.0.0.0:8000","pcos_load:app"]
