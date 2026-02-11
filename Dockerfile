FROM python:3.9-slim

# Чтобы логи сразу выводились
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Сначала зависимости (лучше для docker cache)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Потом сам проект
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "fin_model_with_simulations.py"]
