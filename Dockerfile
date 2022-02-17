FROM python:3.10
# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Set work directory
WORKDIR /code

# Install dependencies
COPY Pipfile Pipfile.lock /code/

# use the --system flag to ensure packages are available throughout all of Docker
RUN pip install pipenv && pipenv install --system

# Copy project
COPY . /code/