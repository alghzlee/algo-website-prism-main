# FROM python:3.11-slim

# WORKDIR /app

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# EXPOSE 5001
# CMD ["python", "app.py"]

# Stage 1: Build CSS dengan Node
FROM node:18-alpine AS css-builder
WORKDIR /build
COPY package*.json ./
COPY tailwind.config.js ./
COPY app/templates ./app/templates
COPY app/static/src ./app/static/src
RUN npm install
RUN npx tailwindcss -i ./app/static/src/css/input.css -o ./app/static/dist/css/output.css --minify
# Stage 2: Python App
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY --from=css-builder /build/app/static/dist/css/output.css ./app/static/dist/css/output.css
EXPOSE 5001
CMD ["python", "app.py"]