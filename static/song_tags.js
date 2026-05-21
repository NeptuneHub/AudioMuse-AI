(function () {
    const escapeHtml = (str) => String(str).replace(/[&<>"']/g, c => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));

    const genreHue = (name) => {
        let hash = 0;
        for (let i = 0; i < name.length; i++) {
            hash = (hash * 31 + name.charCodeAt(i)) | 0;
        }
        return Math.abs(hash) % 360;
    };

    const parseVector = (str) => {
        const map = {};
        if (!str || typeof str !== 'string') return map;
        str.split(',').forEach(part => {
            const idx = part.indexOf(':');
            if (idx === -1) return;
            const label = part.slice(0, idx).trim().toLowerCase();
            const score = parseFloat(part.slice(idx + 1));
            if (label && !isNaN(score)) map[label] = score;
        });
        return map;
    };

    const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1);

    const higherOf = (moods, a, b) => {
        const sa = moods[a], sb = moods[b];
        if (sa === undefined && sb === undefined) return null;
        if (sb === undefined || (sa !== undefined && sa >= sb)) return a;
        return b;
    };

    const moodPill = (label) => `<span class="ptag ptag-${label}">${capitalize(label)}</span>`;

    const renderTrackTags = (track) => {
        if (!track) return '';
        const pills = [];

        const genre = track.top_genre;
        if (genre) {
            const hue = genreHue(genre);
            pills.push(`<span class="ptag ptag-genre" style="--genre-hue: ${hue};">${escapeHtml(genre)}</span>`);
        }

        const moods = parseVector(track.other_features);

        const happySad = higherOf(moods, 'happy', 'sad');
        if (happySad) pills.push(moodPill(happySad));

        const aggressiveRelaxed = higherOf(moods, 'aggressive', 'relaxed');
        if (aggressiveRelaxed) pills.push(moodPill(aggressiveRelaxed));

        if (moods['party'] > 0.70) pills.push(moodPill('party'));
        if (moods['danceable'] > 0.70) pills.push(moodPill('danceable'));

        return `<div class="path-tags">${pills.join('')}</div>`;
    };

    const songSimilarityBadge = (s) => {
        if (typeof s.similarity !== 'number') return null;
        const pct = Math.round(Math.max(0, Math.min(1, s.similarity)) * 100);
        return { text: pct + '%', cls: pct > 70 ? 'badge-high' : pct > 50 ? 'badge-medium' : 'badge-low' };
    };

    const songDistanceBadge = (s) =>
        (typeof s.distance === 'number') ? { text: s.distance.toFixed(4), cls: 'badge-neutral' } : null;

    const renderSongList = (songs, options) => {
        options = options || {};
        if (!Array.isArray(songs) || !songs.length) return '';
        let rows = '', n = 1;
        songs.forEach(s => {
            if (options.skipSeed && s.is_seed) return;
            const artist = s.author || s.artist || '';
            const album = s.album || '';
            const m = options.metric ? options.metric(s) : null;
            rows += `
                <div class="result-item">
                    <div class="result-main">
                        <div class="result-title">${n}. ${escapeHtml(s.title || 'Unknown')}</div>
                        <div class="result-artist">${escapeHtml(artist)}</div>
                        ${album ? `<div class="result-album">Album: ${escapeHtml(album)}</div>` : ''}
                    </div>
                    ${renderTrackTags(s)}
                    ${m ? `<div class="similarity-badge ${m.cls}">${escapeHtml(m.text)}</div>` : ''}
                </div>`;
            n++;
        });
        return `<div class="song-result-list">${rows}</div>`;
    };

    const FEEL_DIMS = ['danceable', 'aggressive', 'happy', 'party', 'relaxed', 'sad'];

    const computeFingerprintSummary = (songs) => {
        const counts = {};
        FEEL_DIMS.forEach(d => { counts[d] = 0; });
        const genreTally = {};
        const list = Array.isArray(songs) ? songs : [];

        list.forEach(s => {
            const moods = parseVector(s.other_features);

            const happySad = higherOf(moods, 'happy', 'sad');
            if (happySad) counts[happySad] += 1;

            const aggressiveRelaxed = higherOf(moods, 'aggressive', 'relaxed');
            if (aggressiveRelaxed) counts[aggressiveRelaxed] += 1;

            if (moods['party'] > 0.70) counts['party'] += 1;
            if (moods['danceable'] > 0.70) counts['danceable'] += 1;

            const g = s.top_genre;
            if (g) genreTally[g] = (genreTally[g] || 0) + 1;
        });

        const topGenres = Object.keys(genreTally)
            .map(genre => ({ genre, count: genreTally[genre] }))
            .sort((a, b) => b.count - a.count)
            .slice(0, 5);

        return { counts, total: list.length, topGenres };
    };

    window.parseVector = parseVector;
    window.genreHue = genreHue;
    window.computeFingerprintSummary = computeFingerprintSummary;
    window.renderTrackTags = renderTrackTags;
    window.renderSongList = renderSongList;
    window.songSimilarityBadge = songSimilarityBadge;
    window.songDistanceBadge = songDistanceBadge;
})();
