FROM python:3.12-slim

RUN pip install streamlit
RUN pip install mlflow
RUN pip install jinja2>=3.1.5
RUN pip install lightgbm>=4.5.0
RUN pip install matplotlib>=3.10.0
RUN pip install numpy>=2.2.2
RUN pip install pandas>=2.2.3
RUN pip install pymupdf>=1.25.3
RUN pip install scikit-learn>=1.6.1
RUN pip install scipy>=1.15.1

COPY ./app2.py /app/app2.py

WORKDIR /app

ENTRYPOINT ["streamlit", "run", "app2.py"]