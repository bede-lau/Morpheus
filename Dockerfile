FROM node:24-bookworm-slim AS frontend
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build && cp -R public/. dist/

FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 MUJOCO_GL=egl
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libegl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-assets.txt ./
RUN pip install --no-cache-dir -r requirements-assets.txt
COPY pipeline ./pipeline
COPY rollouts ./rollouts
COPY output /seed-output
COPY --from=frontend /app/dist ./dist
CMD ["python", "-m", "pipeline.hosted"]
