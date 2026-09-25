-- Q1_SELECT_WHERE
SELECT title, price_gbp, rating
        FROM books
        WHERE rating >= 4;

-- Q2_ORDER_BY
SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC;

-- Q3_LIMIT
SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;

-- Q4_DISTINCT
SELECT DISTINCT rating
        FROM books
        ORDER BY rating;

-- Q5_BETWEEN
SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp;

-- Q6_JOIN
SELECT
            b.title,
            b.price_inr,
            b.rating,
            c.category_name
        FROM books AS b
        JOIN categories AS c
            ON b.category_id = c.category_id
        ORDER BY c.category_name, b.rating DESC, b.title
        LIMIT 10;

