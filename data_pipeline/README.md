# Data Pipeline — Zepto Data & AI Platform

## Overview

This module implements the data collection, cleaning, currency conversion, and SQLite storage pipeline.

The pipeline:
1. Scrapes book data using requests and BeautifulSoup.
2. Collects books across three categories.
3. Cleans price, rating, and availability fields.
4. Converts GBP prices to INR using a fixed project conversion rate.
5. Stores the cleaned data in a normalized SQLite database.
6. Executes SQL queries demonstrating filtering, sorting, limiting, distinct values, range filtering, and joins.
7. Reproduces the SQL JOIN using pandas merge().

## Dataset

The final dataset contains:
- 60 books
- 3 categories
- Historical Fiction
- Mystery
- Travel

The minimum requirement of 60 books across at least 3 categories is satisfied.

## Scraping

The scraper uses:
- requests for HTTP requests
- BeautifulSoup for HTML parsing

The scraped information includes:
- Title
- Price in GBP
- Star rating
- Availability
- Category

## Data Cleaning

The cleaned dataset contains:
- title — string
- price_gbp — float
- price_inr — float
- rating — integer from 1 to 5
- in_stock — boolean
- category — string

Numeric parsing failures are handled during cleaning using the implemented parsing and validation logic. Rows with unusable required numeric values are excluded when they cannot be reliably converted, while valid values are retained. This avoids introducing incorrect price or rating values into the database.

## GBP to INR Conversion

A fixed artificial project conversion rate is used:

1 GBP = 105.50 INR

Formula:

price_inr = price_gbp * 105.50

No external currency API is used.

The final validation confirms that the price conversion is correct.

## SQLite Database

The database is stored in:

books.db

The normalized database contains two main tables.

### categories

- category_id — Primary Key
- category_name

### books

- book_id — Primary Key
- title
- price_gbp
- price_inr
- rating
- in_stock
- category_id — Foreign Key referencing categories.category_id

The final database contains 60 books and 3 categories.

## SQL Queries

Six SQL queries are included in queries.sql.

They demonstrate:
1. SELECT / WHERE
2. ORDER BY
3. LIMIT
4. DISTINCT
5. BETWEEN
6. JOIN

The query results are saved in:

query_outputs.txt

## SQL JOIN and Pandas JOIN

The SQL JOIN combines books and categories using:

books.category_id = categories.category_id

The same relationship is reproduced using pandas merge().

The SQL and pandas JOIN outputs were compared after applying the same sorting and row limit.

The validation confirmed:

Equivalent: True

## Validation

Final validation confirmed:

Books in pandas: 60
Books in SQLite: 60
Categories in pandas: 3
Categories in SQLite: 3

Therefore, the pandas dataset and SQLite database contain matching book and category counts.

## Project Files

data_pipeline/
- data_pipeline.ipynb
- books.db
- queries.sql
- query_outputs.txt
- README.md

## Requirements

Install the Python dependencies:

pip install requests beautifulsoup4 pandas

SQLite is provided through Python's standard library.

## Running the Pipeline

Open and run:

data_pipeline.ipynb

The notebook performs the scraping, cleaning, conversion, database creation, SQL analysis, pandas JOIN validation, and final checks.

## Output Files

- books.db — SQLite database
- queries.sql — SQL query definitions
- query_outputs.txt — saved SQL query results
- README.md — module documentation
## Completion Status

Module 1 has been validated with 60 books across 3 categories.

The final validation confirmed matching book and category counts between the pandas dataset and SQLite database. SQL JOIN results were also verified against the equivalent pandas merge() output.
