FROM python:3.7

COPY ./src/ /app/src
COPY ./models/ /app/models
COPY ./data /app/data
COPY ./requirements.txt /app
COPY ./config.ini /app
WORKDIR /app

RUN pip install --upgrade pip && pip install -r requirements.txt

CMD cd src && python ./main.py