-- Check empty lyrics query 
-- Find rows that have BOTH sentinels (full instrumental marker):

SELECT item_id, updated_at
FROM lyrics_embedding
WHERE embedding   = decode('0000803f' || repeat('00', 3068), 'hex')
  AND axis_vector = decode(repeat('a31145be', 27), 'hex');


-- Count instrumentals + join with track metadata:

SELECT s.item_id, s.title, s.author, s.album, le.updated_at
FROM lyrics_embedding le
JOIN score s ON s.item_id = le.item_id
WHERE le.embedding   = decode('0000803f' || repeat('00', 3068), 'hex')
  AND le.axis_vector = decode(repeat('a31145be', 27), 'hex')
ORDER BY le.updated_at DESC;