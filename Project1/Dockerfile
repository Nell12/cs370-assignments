#python
FROM python:3.9-slim

#working directory
WORKDIR /project

#copy contents from current directory
COPY . /project

# Install any needed dependencies specified in requirements.txt
RUN pip install -r requirements.txt

#first command to run
CMD ["python", "test.py"]


