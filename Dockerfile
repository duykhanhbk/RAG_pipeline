# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]