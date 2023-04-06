# Dockerfile, image, container
FROM python:3.10

ADD main.py .

RUN pip install pandas scikit-learn

CMD ["python", "./main.py"]