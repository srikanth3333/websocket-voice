# Stage 1: Build the Vite frontend
FROM node:20-alpine AS client-builder
WORKDIR /client
COPY client/package*.json ./
RUN npm ci
COPY client/ ./
# Replace local dev URL with empty string so endpoint becomes relative /start
RUN sed -i 's|http://localhost:7861||g' src/app.ts
RUN npm run build

# Stage 2: Python application
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    "pipecat-ai[openai,cartesia,deepgram,silero,websocket]>=0.0.105" \
    python-dotenv \
    aiofiles \
    uvicorn

# Copy application
COPY combined_app.py ./

# Copy built frontend
COPY --from=client-builder /client/dist ./static

EXPOSE 8080
ENV PORT=8080

CMD ["python", "combined_app.py"]
