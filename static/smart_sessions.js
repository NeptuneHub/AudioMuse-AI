(function () {
    const form = document.getElementById('smart-session-form');
    if (!form) return;

    const capabilitiesEl = document.getElementById('smart-capabilities');
    const statusEl = document.getElementById('smart-status');
    const listEl = document.getElementById('smart-track-list');
    const previewButton = document.getElementById('smart-preview-button');
    const exportButton = document.getElementById('smart-export-button');
    const playlistNameInput = document.getElementById('smart-playlist-name');
    let currentTracks = [];

    function splitList(value) {
        return (value || '').split(',').map((item) => item.trim()).filter(Boolean);
    }

    function setMessage(element, message, type) {
        element.textContent = message;
        element.className = 'smart-message' + (type ? ' ' + type : '');
    }

    function buildPreviewPayload() {
        const anchorId = document.getElementById('smart-anchor-id').value.trim();
        const anchors = anchorId ? [{ type: 'song', item_id: anchorId, weight: 1 }] : [];
        return {
            prompt: document.getElementById('smart-prompt').value,
            length: document.getElementById('smart-length').value,
            curve: document.getElementById('smart-curve').value,
            anchors: anchors,
            avoid: {
                artists: splitList(document.getElementById('smart-avoid-artists').value),
                terms: splitList(document.getElementById('smart-avoid-terms').value)
            },
            max_per_artist: document.getElementById('smart-max-per-artist').value,
            include_explanations: true
        };
    }

    function renderTracks(tracks) {
        listEl.innerHTML = '';
        currentTracks = tracks || [];
        exportButton.disabled = currentTracks.length === 0;

        currentTracks.forEach((track, index) => {
            const item = document.createElement('li');
            item.className = 'smart-track-item';

            const position = document.createElement('span');
            position.textContent = String(index + 1);

            const details = document.createElement('div');
            const title = document.createElement('div');
            title.className = 'smart-track-title';
            title.textContent = track.title || track.item_id;
            const meta = document.createElement('div');
            meta.className = 'smart-track-meta';
            meta.textContent = track.author || 'Unknown artist';
            details.appendChild(title);
            details.appendChild(meta);
            if (track.reason) {
                const reason = document.createElement('div');
                reason.className = 'smart-track-reason';
                reason.textContent = track.reason;
                details.appendChild(reason);
            }

            const remove = document.createElement('button');
            remove.type = 'button';
            remove.className = 'smart-remove-track';
            remove.textContent = 'Remove';
            remove.addEventListener('click', () => {
                currentTracks.splice(index, 1);
                renderTracks(currentTracks);
            });

            item.appendChild(position);
            item.appendChild(details);
            item.appendChild(remove);
            listEl.appendChild(item);
        });
    }

    async function loadCapabilities() {
        try {
            const response = await fetch('/api/smart_sessions/capabilities');
            const data = await response.json();
            const sources = [];
            if (data.clap_enabled && data.clap_cache_loaded) sources.push('CLAP');
            if (data.sem_grove_available) sources.push('SemGrove');
            if (data.lyrics_enabled) sources.push('lyrics');
            setMessage(capabilitiesEl, sources.length ? 'Available sources: ' + sources.join(', ') : 'Smart Sessions is running in limited mode.', sources.length ? '' : 'warning');
        } catch (error) {
            setMessage(capabilitiesEl, 'Could not load Smart Sessions capabilities.', 'warning');
        }
    }

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        previewButton.disabled = true;
        exportButton.disabled = true;
        setMessage(statusEl, 'Building preview...');
        renderTracks([]);

        try {
            const response = await fetch('/api/smart_sessions/preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(buildPreviewPayload())
            });
            const data = await response.json();
            if (!response.ok) {
                setMessage(statusEl, data.error || 'Preview failed.', 'error');
                return;
            }
            playlistNameInput.value = data.playlist_name || playlistNameInput.value;
            renderTracks(data.tracks || []);
            const warning = (data.warnings || [])[0];
            setMessage(statusEl, warning || 'Preview ready.', warning ? 'warning' : '');
        } catch (error) {
            setMessage(statusEl, 'Preview request failed.', 'error');
        } finally {
            previewButton.disabled = false;
        }
    });

    exportButton.addEventListener('click', async () => {
        exportButton.disabled = true;
        setMessage(statusEl, 'Exporting playlist...');
        try {
            const response = await fetch('/api/smart_sessions/export', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    playlist_name: playlistNameInput.value,
                    track_ids: currentTracks.map((track) => track.item_id)
                })
            });
            const data = await response.json();
            setMessage(statusEl, response.ok ? data.message : (data.error || 'Export failed.'), response.ok ? '' : 'error');
        } catch (error) {
            setMessage(statusEl, 'Export request failed.', 'error');
        } finally {
            exportButton.disabled = currentTracks.length === 0;
        }
    });

    loadCapabilities();
})();