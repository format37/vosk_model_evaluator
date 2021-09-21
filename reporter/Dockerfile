FROM python:latest
WORKDIR /reporter
COPY requirements.txt /reporter
RUN pip3 install -r requirements.txt --no-cache-dir
COPY reporter.py /reporter
ENTRYPOINT ["python3"]
CMD ["reporter.py"]
