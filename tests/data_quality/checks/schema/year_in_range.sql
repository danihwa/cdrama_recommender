-- description: year must be in [1950, 2026] — wide enough for legit 80s titles, narrow enough to catch parse bugs (e.g. year=19)
SELECT mdl_id, year
FROM cdramas
WHERE year < 1950 OR year > 2026
LIMIT 100;
