-- description: tags array must be non-empty (cleaning drops empty rows)
SELECT mdl_id, title
FROM cdramas
WHERE tags IS NULL OR array_length(tags, 1) IS NULL
LIMIT 100;
