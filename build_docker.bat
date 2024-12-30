docker build -t ml-fastapi-app .

docker run -d --name ml-container -p 7000:7000 ml-fastapi-app
