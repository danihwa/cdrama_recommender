-- description: genres array must be non-empty (cleaning drops empty rows)
SELECT mdl_id, title
FROM cdramas
WHERE genres IS NULL OR array_length(genres, 1) IS NULL
LIMIT 100;
