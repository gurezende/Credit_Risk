FROM python:3.12-slim

RUN pip install mlflow==2.20.2, pykernel>=6.29.5, jinja2>=3.1.5, lightgbm>=4.5.0,matplotlib>=3.10.0,mlflow>=2.20.2,numpy>=2.2.2,pandas>=2.2.3,pymupdf>=1.25.3,scikit-learn>=1.6.1,scipy>=1.15.1,seaborn>=0.13.2,statsmodels>=0.14.4,streamlit>=1.42.0,ucimlrepo>=0.0.7

COPY ./app.py /app/app.py

WORKDIR /app

ENTRYPOINT [ "streamlit", "run", "app.py" ]