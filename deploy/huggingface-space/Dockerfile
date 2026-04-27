FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8050
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

RUN useradd -m -u 1000 user

WORKDIR $HOME/app

COPY --chown=user requirements.txt .

USER user

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 8050

CMD ["python", "app.py"]
