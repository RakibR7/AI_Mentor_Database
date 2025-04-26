FROM python:3.9-slim
#FROM python:slim

WORKDIR /app
RUN pip install wikiextractor

#COPY enwiki-20240901-pages-articles-multistream.xml.bz2 /app
#CMD python -m wikiextractor.WikiExtractor --output /app/output /app/enwiki-20240901-pages-articles-multistream.xml.bz2

CMD ["python", "-m", "wikiextractor.WikiExtractor", "--output", "/app/output", "/app/enwiki-latest-pages-articles.xml.bz2"]

