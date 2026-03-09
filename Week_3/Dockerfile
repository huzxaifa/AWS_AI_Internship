FROM public.ecr.aws/lambda/python:3.10

# Copy requirements
COPY requirements.txt .

# Set cache to /tmp for Lambda (Read-Only fix)
ENV HF_HOME=/tmp
ENV TRANSFORMERS_CACHE=/tmp

# Install dependencies with --no-cache-dir to save space
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code correctly preserving structure
COPY srv/ ${LAMBDA_TASK_ROOT}/srv/
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

# Set the handler
CMD ["lambda_function.lambda_handler"]