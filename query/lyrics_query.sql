-- Instrumental sentinel queries (dimension-independent).
-- Sentinel: embedding = [1.0, 0, 0, ...], axis = uniform negative fill.
-- Works regardless of LYRICS_EMBEDDING_DIMENSION or MUSIC_ANALYSIS_AXES count.
-- Compatible with PostgreSQL < 14 (no repeat(bytea) — route through hex text).

-- Find instrumental rows with track metadata:
SELECT s.item_id, s.title, s.author, s.album, le.updated_at
FROM lyrics_embedding le
JOIN score s ON s.item_id = le.item_id
WHERE substring(le.embedding for 4) = decode('0000803f', 'hex')
  AND substring(le.embedding from 5) = decode(repeat('00', octet_length(le.embedding) - 4), 'hex')
  AND substring(le.axis_vector from 5) = decode(
        repeat(encode(substring(le.axis_vector for 4), 'hex'),
               (octet_length(le.axis_vector) / 4) - 1),
        'hex'
      )
ORDER BY le.updated_at DESC;

-- Count instrumentals:
SELECT count(*)
FROM lyrics_embedding
WHERE substring(embedding for 4) = decode('0000803f', 'hex')
  AND substring(embedding from 5) = decode(repeat('00', octet_length(embedding) - 4), 'hex')
  AND substring(axis_vector from 5) = decode(
        repeat(encode(substring(axis_vector for 4), 'hex'),
               (octet_length(axis_vector) / 4) - 1),
        'hex'
      );
