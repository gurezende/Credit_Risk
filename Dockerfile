FROM python:3.12-slim

WORKDIR /app

RUN pip install streamlit
RUN pip install mlflow
RUN pip install jinja2>=3.1.5
RUN pip install matplotlib>=3.10.0
RUN pip install numpy>=2.2.2
RUN pip install pandas>=2.2.3
RUN pip install scikit-learn>=1.6.1
RUN pip install scipy>=1.15.1

# copy folder
COPY . .

EXPOSE 8501

# MLflow tracking URI - IMPORTANT!
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000

CMD ["streamlit", "run", "app.py"]