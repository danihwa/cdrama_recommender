-- description: mdl_id must be unique (schema enforces UNIQUE; this is belt-and-suspenders)
SELECT mdl_id, COUNT(*) AS occurrences
FROM cdramas
GROUP BY mdl_id
HAVING COUNT(*) > 1
LIMIT 100;
