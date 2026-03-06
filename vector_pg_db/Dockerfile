FROM apache/age:release_PG16_1.5.0

RUN apt-get update && apt-get install -y \
    ca-certificates \
    postgresql-16-pgvector

COPY create-extension-vector.sql /docker-entrypoint-initdb.d/create-extension-vector.sql
CMD ["postgres", "-c", "shared_preload_libraries=age"]