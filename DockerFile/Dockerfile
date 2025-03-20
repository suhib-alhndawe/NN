# Use the official Python 3.9 image as the base image
FROM python:3.9

# Set the working directory to /app inside the container
WORKDIR /app

# Copy all files from the current directory to the /app directory inside the container
COPY . /app

# Install the dependencies listed in the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# It listens on all network interfaces (0.0.0.0) and port 800 
CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "8000"]
