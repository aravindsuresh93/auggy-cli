FROM jjanzic/docker-python3-opencv:latest
RUN pip install pandas coloredlogs albumentations
WORKDIR /app/auggy-cli
