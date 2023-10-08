FROM python:3.9.6

RUN pip install --no-cache-dir --upgrade pip 
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install streamlit
WORKDIR /app

COPY exams.csv /app

COPY ml.py /app/

EXPOSE 8501

CMD ["streamlit", "run", "ml.py"]