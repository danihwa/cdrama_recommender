-- description: required columns must be non-NULL for every loaded row
SELECT 'mdl_id'    AS column_name, id::text        AS row_id FROM cdramas WHERE mdl_id    IS NULL
UNION ALL
SELECT 'mdl_url',    mdl_id::text FROM cdramas WHERE mdl_url    IS NULL
UNION ALL
SELECT 'title',      mdl_id::text FROM cdramas WHERE title      IS NULL
UNION ALL
SELECT 'synopsis',   mdl_id::text FROM cdramas WHERE synopsis   IS NULL
UNION ALL
SELECT 'episodes',   mdl_id::text FROM cdramas WHERE episodes   IS NULL
UNION ALL
SELECT 'year',       mdl_id::text FROM cdramas WHERE year       IS NULL
UNION ALL
SELECT 'mdl_score',  mdl_id::text FROM cdramas WHERE mdl_score  IS NULL
UNION ALL
SELECT 'watchers',   mdl_id::text FROM cdramas WHERE watchers   IS NULL
LIMIT 100;
