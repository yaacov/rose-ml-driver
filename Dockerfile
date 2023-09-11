# Build stage
FROM quay.io/rose/rose-client

# Copy the source code into the container
COPY . /ml

# Install the Python dependencies
RUN pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

ENTRYPOINT ["python", "main.py", "--listen", "0.0.0.0", "--driver", "/ml/driver.py"]
CMD ["--port", "8081"]

