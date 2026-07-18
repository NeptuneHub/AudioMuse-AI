-- Songs backed by more than one provider file (duplicate ids).
-- In track_server_map the primary key is (server_id, provider_track_id), so a
-- single canonical item_id (one score row) may map to N different
-- provider_track_id values on the same server - i.e. N duplicate physical files
-- of the same song. These queries surface those shared item_ids together with
-- the song metadata from the score table.
--
-- The library can be huge, so every query is capped with LIMIT to return a small
-- set of examples rather than every duplicate.

-- 1) One row per (server, song) that has more than one provider file:
--    the score metadata plus how many files share the same item_id.
SELECT
    tsm.server_id,
    s.item_id,
    s.title,
    s.author,
    s.album,
    s.album_artist,
    count(*) AS file_count
FROM track_server_map tsm
JOIN score s ON s.item_id = tsm.item_id
GROUP BY tsm.server_id, s.item_id, s.title, s.author, s.album, s.album_artist
HAVING count(*) > 1
ORDER BY file_count DESC, s.item_id
LIMIT 50;

-- 2) The same duplicates expanded: one row per provider file, so you can see the
--    individual provider_track_id / file_path values that share each item_id.
SELECT
    tsm.server_id,
    s.item_id,
    s.title,
    s.author,
    s.album,
    tsm.provider_track_id,
    tsm.file_path,
    tsm.match_tier,
    tsm.updated_at
FROM track_server_map tsm
JOIN score s ON s.item_id = tsm.item_id
WHERE (tsm.server_id, tsm.item_id) IN (
    SELECT server_id, item_id
    FROM track_server_map
    GROUP BY server_id, item_id
    HAVING count(*) > 1
    LIMIT 20
)
ORDER BY tsm.server_id, s.item_id, tsm.provider_track_id;

-- 3) How many songs have duplicate files, per server.
SELECT
    server_id,
    count(*) AS songs_with_duplicates
FROM (
    SELECT server_id, item_id
    FROM track_server_map
    GROUP BY server_id, item_id
    HAVING count(*) > 1
) dupes
GROUP BY server_id
ORDER BY songs_with_duplicates DESC;

-- 4) Pick the FIRST song that has duplicate provider files and list every entry
--    for it. Note: track_server_map stores no per-file title/author - the only
--    title/author/album come from the single score row (score_* columns below),
--    so all entries repeat the same canonical metadata. What actually differs per
--    duplicate is provider_track_id / file_path / match_tier / updated_at.
WITH first_dup AS (
    SELECT server_id, item_id
    FROM track_server_map
    GROUP BY server_id, item_id
    HAVING count(*) > 1
    ORDER BY server_id, item_id
    LIMIT 1
)
SELECT
    tsm.server_id,
    tsm.item_id,
    s.title  AS score_title,
    s.author AS score_author,
    s.album  AS score_album,
    tsm.provider_track_id,
    tsm.file_path,
    tsm.match_tier,
    tsm.updated_at
FROM track_server_map tsm
JOIN first_dup fd ON fd.server_id = tsm.server_id AND fd.item_id = tsm.item_id
LEFT JOIN score s ON s.item_id = tsm.item_id
ORDER BY tsm.provider_track_id;
