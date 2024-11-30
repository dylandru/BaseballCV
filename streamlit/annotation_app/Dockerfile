FROM python:3.11-slim-buster

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip


WORKDIR /BaseballCV

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/BaseballCV
ENV AWS_BASEBALLCV_ACCESS_KEY=${AWS_BASEBALLCV_ACCESS_KEY}
ENV AWS_BASEBALLCV_SECRET_KEY=${AWS_BASEBALLCV_SECRET_KEY}
ENV AWS_BASEBALLCV_REGION=${AWS_BASEBALLCV_REGION}

ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8
ENV NUMEXPR_NUM_THREADS=8
ENV OPENBLAS_NUM_THREADS=8

EXPOSE 8501

CMD ["streamlit", "run", "streamlit/annotation_app/app.py", "--server.address=0.0.0.0"] 