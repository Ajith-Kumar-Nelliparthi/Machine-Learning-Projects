FROM python:3.13.0-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]


RUN pipenv install --system --deploy

COPY ["medical_cost_predict.py", "medical_cost_load.py", "./"]

EXPOSE 5000

ENTRYPOINT ["gunicorn","--bind=0.0.0.0:5000","medical_cost_predict:app"]