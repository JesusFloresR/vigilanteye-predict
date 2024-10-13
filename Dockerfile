FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/ml/code

COPY predict.py /opt/ml/code
COPY requirements.txt /opt/ml/code

WORKDIR /opt/ml/code

# Install the specified packages
RUN pip install -r requirements.txt
RUN pip install face-detection --no-deps 

ENTRYPOINT ["python", "predict.py"]