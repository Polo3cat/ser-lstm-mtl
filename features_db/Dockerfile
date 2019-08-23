FROM postgres:11.3

ENV POSTGRES_USER docker
ENV POSTGRES_PASSWORD docker
ENV POSTGRES_DB features 

ADD createdb.sql /docker-entrypoint-initdb.d/

EXPOSE 5432
CMD ["postgres"]
