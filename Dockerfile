# ---- Stage 1: Builder ----
# Use a full Python image to install dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy only requirements first (Docker layer caching optimization)
# If requirements.txt doesn't change, this layer is reused on rebuild — saves minutes
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: Runtime ----
# Start fresh from a slim image — don't carry build tools into production
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ ./app/
COPY api/ ./api/
COPY ingestion/ ./ingestion/

# Cloud Run injects PORT environment variable — must listen on it
ENV PORT=8080
EXPOSE 8080

# Non-root user for security best practice
RUN useradd -m appuser
USER appuser

# Start FastAPI with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
