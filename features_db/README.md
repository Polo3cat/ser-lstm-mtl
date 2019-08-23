## Generic postgresql setup
* login: docker/docker
* Environment password is set with;-e POSTGRES_USER=<user> -e POSTGRES_PASSWORD=<password>
* docker build -t postgres .
* docker run -p 5432:5432 postgres:latest
* map local volume to data volume postgres container; -v /my/own/datadir:/var/lib/postgresql/data

## Configuration docker-compose.yml
* Map a local volume to /var/lib/postgresql/data
* Set username and password environment variables.

## Building with docker-compose
```
$ docker-compose -f docker-compose.yml build
```

## Running with docker-compose
```
$ docker-compose -f docker-compose.yml run
```
