/* ============================================================
   Playlist Curator — Smart Search page
   - Filter builder UI (add/remove rules, dynamic field/op/value)
   - Run search via POST /api/curator/search
   - Render results (table on desktop, cards on mobile)
   - Add / Skip per row; Send N to Extender bulk-add + navigate
   ============================================================ */
(function () {
    'use strict';

    const ICONS = window.CURATOR_ICONS;
    const escHtml = window.escHtml;

    // ---------- Filter field config (mirrors current playlist_curator.html) ----------
    const FILTER_FIELDS = [
        { value: 'artist', label: 'Artist', type: 'text' },
        { value: 'title', label: 'Track Title', type: 'text' },
        { value: 'album', label: 'Album', type: 'text' },
        { value: 'album_artist', label: 'Album Artist', type: 'text' },
        { value: 'genre', label: 'Genre (rock, pop, jazz...)', type: 'mood_slider', optionsKey: 'moods' },
        { value: 'features', label: 'Mood / Style (danceable, aggressive...)', type: 'mood_slider', optionsKey: 'features' },
        { value: 'year', label: 'Year', type: 'number' },
        { value: 'decade', label: 'Decade', type: 'dropdown', optionsKey: 'year_ranges' },
        { value: 'rating', label: 'Rating', type: 'dropdown', optionsKey: 'rating_ranges' },
        { value: 'bpm', label: 'BPM', type: 'dropdown', optionsKey: 'bpm_ranges' },
        { value: 'energy', label: 'Energy', type: 'dropdown', optionsKey: 'energy_ranges' },
        { value: 'key', label: 'Key', type: 'dropdown', optionsKey: 'keys' },
        { value: 'scale', label: 'Scale', type: 'dropdown', optionsKey: 'scales' },
    ];

    const OPERATORS = {
        text: [
            { value: 'contains', label: 'contains' },
            { value: 'does_not_contain', label: 'does not contain' },
            { value: 'is', label: 'is' },
            { value: 'is_not', label: 'is not' },
        ],
        dropdown: [
            { value: 'is', label: 'is' },
            { value: 'is_not', label: 'is not' },
        ],
        number: [
            { value: 'is', label: 'is' },
            { value: 'is_not', label: 'is not' },
            { value: 'greater_than', label: 'is greater than' },
            { value: 'less_than', label: 'is less than' },
        ],
        mood_slider: [
            { value: 'greater_than', label: 'at least' },
            { value: 'less_than', label: 'at most' },
        ],
    };

    let filterOptions = { keys: [], scales: [], moods: [], features: [], bpm_ranges: [], energy_ranges: [], rating_ranges: [], year_ranges: [] };
    let lastResults = [];
    let skippedIds = new Set();

    // ---------- Filter builder ----------
    function addFilterRow(initial) {
        const rowsEl = document.getElementById('curator-filter-rows');
        if (!rowsEl) return;
        const row = document.createElement('div');
        row.className = 'curator-filter-row';

        const fieldSelect = document.createElement('select');
        fieldSelect.className = 'filter-field curator-select';
        FILTER_FIELDS.forEach(f => {
            const opt = document.createElement('option');
            opt.value = f.value;
            opt.textContent = f.label;
            opt.dataset.type = f.type;
            opt.dataset.optionsKey = f.optionsKey || '';
            fieldSelect.appendChild(opt);
        });

        const opSelect = document.createElement('select');
        opSelect.className = 'filter-operator curator-select';

        const valueContainer = document.createElement('div');
        valueContainer.className = 'filter-value-container';
        valueContainer.style.minWidth = '0';

        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'curator-filter-remove';
        removeBtn.innerHTML = '×';
        removeBtn.title = 'Remove rule';
        removeBtn.addEventListener('click', () => row.remove());

        fieldSelect.addEventListener('change', () => updateFilterRowUI(fieldSelect, opSelect, valueContainer));

        row.appendChild(fieldSelect);
        row.appendChild(opSelect);
        row.appendChild(valueContainer);
        row.appendChild(removeBtn);
        rowsEl.appendChild(row);

        if (initial) {
            fieldSelect.value = initial.field;
        }
        updateFilterRowUI(fieldSelect, opSelect, valueContainer);
        if (initial) {
            opSelect.value = initial.op;
            const v = valueContainer.querySelector('.filter-value');
            if (v) v.value = initial.value || '';
        }
    }

    function updateFilterRowUI(fieldSelect, opSelect, valueContainer) {
        const sel = fieldSelect.options[fieldSelect.selectedIndex];
        const fieldType = sel.dataset.type;
        const optionsKey = sel.dataset.optionsKey;

        opSelect.innerHTML = '';
        const ops = OPERATORS[fieldType] || OPERATORS.text;
        ops.forEach(op => {
            const opt = document.createElement('option');
            opt.value = op.value;
            opt.textContent = op.label;
            opSelect.appendChild(opt);
        });

        valueContainer.innerHTML = '';
        if (fieldType === 'mood_slider' && optionsKey) {
            const wrap = document.createElement('div');
            wrap.className = 'filter-mood-wrap';

            const sel2 = document.createElement('select');
            sel2.className = 'curator-select';
            sel2.style.minWidth = '120px';
            const empty = document.createElement('option');
            empty.value = ''; empty.textContent = '-- Select --';
            sel2.appendChild(empty);
            (filterOptions[optionsKey] || []).forEach(opt => {
                const o = document.createElement('option');
                o.value = typeof opt === 'object' ? opt.value : opt;
                o.textContent = typeof opt === 'object' ? opt.label : opt;
                sel2.appendChild(o);
            });

            const slider = document.createElement('input');
            slider.type = 'range'; slider.min = '0'; slider.max = '1'; slider.step = '0.05'; slider.value = '0.55';
            slider.style.cssText = 'flex:1;min-width:60px;accent-color:var(--color-primary);';

            const sliderVal = document.createElement('span');
            sliderVal.className = 'curator-slider-value';
            sliderVal.style.minWidth = '32px';
            sliderVal.textContent = '0.55';
            slider.addEventListener('input', () => { sliderVal.textContent = parseFloat(slider.value).toFixed(2); });

            const hidden = document.createElement('input');
            hidden.type = 'hidden';
            hidden.className = 'filter-value';
            const updateHidden = () => {
                hidden.value = sel2.value ? sel2.value + ':' + parseFloat(slider.value).toFixed(2) : '';
            };
            sel2.addEventListener('change', updateHidden);
            slider.addEventListener('input', updateHidden);
            updateHidden();

            wrap.appendChild(sel2);
            wrap.appendChild(slider);
            wrap.appendChild(sliderVal);
            valueContainer.appendChild(wrap);
            valueContainer.appendChild(hidden);
        } else if (fieldType === 'dropdown' && optionsKey) {
            const sel2 = document.createElement('select');
            sel2.className = 'filter-value curator-select';
            const empty = document.createElement('option');
            empty.value = ''; empty.textContent = '-- Select --';
            sel2.appendChild(empty);
            (filterOptions[optionsKey] || []).forEach(opt => {
                const o = document.createElement('option');
                if (typeof opt === 'object') { o.value = opt.value; o.textContent = opt.label; }
                else { o.value = opt; o.textContent = opt; }
                sel2.appendChild(o);
            });
            sel2.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); runSearch(); } });
            valueContainer.appendChild(sel2);
        } else {
            const inp = document.createElement('input');
            inp.type = 'text';
            inp.className = 'filter-value curator-input';
            inp.placeholder = fieldType === 'number' ? 'e.g. 1990' : 'value…';
            inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); runSearch(); } });
            valueContainer.appendChild(inp);
        }
    }

    function getFilters() {
        const rows = document.querySelectorAll('#curator-filter-rows .curator-filter-row');
        const out = [];
        rows.forEach(row => {
            const field = row.querySelector('.filter-field').value;
            const operator = row.querySelector('.filter-operator').value;
            const valueEl = row.querySelector('.filter-value');
            const value = valueEl ? valueEl.value : '';
            if (value) out.push({ field, operator, value });
        });
        return out;
    }

    // ---------- Run search ----------
    async function runSearch() {
        const filters = getFilters();
        const matchMode = document.getElementById('curator-match-mode').value;
        const statusId = 'curator-search-status';

        if (filters.length === 0) {
            window.curatorSetStatus(statusId, 'Please add at least one filter rule.', 'error');
            return;
        }

        const runBtn = document.getElementById('curator-search-run');
        const sendBtn = document.getElementById('curator-search-send');
        if (runBtn) runBtn.disabled = true;
        if (sendBtn) sendBtn.disabled = true;
        window.curatorSetStatus(statusId, 'Searching…', 'loading');

        const payload = {
            filters: filters,
            match_mode: matchMode,
            max_songs: 1000,
            similarity_threshold: 1.0,
            included_ids: [],
            excluded_ids: [],
            search_only: true,
        };

        try {
            const res = await fetch('/api/curator/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Search failed');
            lastResults = Array.isArray(data.results) ? data.results : [];
            skippedIds = new Set();
            renderResults();
            window.curatorSetStatus(statusId, '', '');
        } catch (e) {
            window.curatorSetStatus(statusId, e.message || 'Search failed', 'error');
        } finally {
            if (runBtn) runBtn.disabled = false;
            if (sendBtn) sendBtn.disabled = false;
        }
    }

    // ---------- Render results ----------
    function visibleResults() {
        return lastResults.filter(t => !skippedIds.has(t.item_id));
    }

    function renderResults() {
        const section = document.getElementById('curator-results-section');
        if (!section) return;
        section.classList.remove('hidden');

        const headCount = document.getElementById('curator-results-count');
        const headSkipped = document.getElementById('curator-results-skipped');
        const visible = visibleResults();
        if (headCount) headCount.textContent = visible.length;
        if (headSkipped) {
            headSkipped.textContent = skippedIds.size > 0 ? ` · ${skippedIds.size} skipped` : '';
        }

        const sendBtn = document.getElementById('curator-search-send');
        if (sendBtn) {
            sendBtn.classList.toggle('hidden', visible.length === 0);
            const lbl = `Send ${visible.length} to Extender`;
            sendBtn.innerHTML = `${escHtml(lbl)} <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg>`;
        }

        const wrap = document.getElementById('curator-results-table-wrap');
        const cards = document.getElementById('curator-results-cards');
        if (visible.length === 0) {
            const empty = `<div class="curator-empty-state">No tracks match these rules. Try loosening one.</div>`;
            if (wrap) wrap.innerHTML = empty;
            if (cards) cards.innerHTML = empty;
            return;
        }

        if (wrap) wrap.innerHTML = renderTable(visible);
        if (cards) cards.innerHTML = renderCards(visible);
    }

    function renderTable(rows) {
        const trs = rows.map(t => rowHtml(t)).join('');
        return `<div class="curator-card flush">
            <table class="curator-table">
                <thead><tr>
                    <th class="col-play"></th>
                    <th>Title / Artist</th>
                    <th>Album</th>
                    <th class="col-year">Year</th>
                    <th class="col-bpm">BPM</th>
                    <th class="col-actions">Action</th>
                </tr></thead>
                <tbody>${trs}</tbody>
            </table>
        </div>`;
    }

    function rowHtml(t) {
        const inWb = window.workbenchHas(t.item_id);
        const id = escHtml(t.item_id);
        const artist = escHtml(t.song_artist || t.author || 'Unknown');
        const title = escHtml(t.title || 'Unknown');
        const album = escHtml(t.album || '-');
        const year = t.year ? escHtml(t.year) : '';
        const bpm = (t.bpm != null ? Math.round(t.bpm) : (t.tempo != null ? Math.round(t.tempo) : ''));
        const stream = '/api/curator/stream/' + encodeURIComponent(t.item_id);

        const actionHtml = inWb ? `
            <div class="curator-row-actions">
                <span class="curator-pill" data-tone="success">${ICONS.check} In Workbench</span>
                <button type="button" class="curator-remove-x" data-wb-remove="${id}" title="Remove from Workbench">${ICONS.x}</button>
            </div>` : `
            <div class="curator-row-actions">
                <button type="button" class="curator-btn" data-kind="success" data-size="sm" data-search-add="${id}">${ICONS.plus} Add</button>
                <button type="button" class="curator-btn" data-kind="secondary" data-size="sm" data-search-skip="${id}">Skip</button>
            </div>`;

        return `<tr class="${inWb ? 'in-wb' : ''}" data-row-id="${id}">
            <td class="col-play">
                <button type="button" class="curator-icon-btn" data-stream="${escHtml(stream)}" data-item-id="${id}" data-title="${title}" data-artist="${artist}">${ICONS.play}</button>
            </td>
            <td>
                <div class="curator-track-cell-title">${title}</div>
                <div class="curator-track-cell-sub">${artist}</div>
            </td>
            <td class="col-album">${album}</td>
            <td class="col-year">${year}</td>
            <td class="col-bpm">${bpm}</td>
            <td class="col-actions">${actionHtml}</td>
        </tr>`;
    }

    function renderCards(rows) {
        return rows.map(t => {
            const inWb = window.workbenchHas(t.item_id);
            const id = escHtml(t.item_id);
            const artist = escHtml(t.song_artist || t.author || 'Unknown');
            const title = escHtml(t.title || 'Unknown');
            const yearText = t.year ? escHtml(t.year) : '';
            const stream = '/api/curator/stream/' + encodeURIComponent(t.item_id);

            const actions = inWb ? `
                <button type="button" class="curator-added-text" data-wb-remove="${id}">${ICONS.check} Added</button>` : `
                <div class="curator-track-card-actions">
                    <button type="button" class="curator-btn" data-kind="success" data-size="sm" data-search-add="${id}">+ Add</button>
                    <button type="button" class="curator-btn" data-kind="secondary" data-size="sm" data-search-skip="${id}">Skip</button>
                </div>`;
            return `<div class="curator-track-card ${inWb ? 'in-wb' : ''}" data-row-id="${id}">
                <button type="button" class="curator-icon-btn" data-stream="${escHtml(stream)}" data-item-id="${id}" data-title="${title}" data-artist="${artist}">${ICONS.play}</button>
                <div class="curator-track-card-meta">
                    <div class="curator-track-card-title">${title}</div>
                    <div class="curator-track-card-sub">${artist}${yearText ? ' · ' + yearText : ''}</div>
                </div>
                ${actions}
            </div>`;
        }).join('');
    }

    // ---------- Send to Extender ----------
    function sendAllToExtender() {
        const visible = visibleResults();
        if (visible.length === 0) return;
        const added = window.workbenchAddBulk(visible, 'search');
        window.curatorToast(`Sent ${added > 0 ? added : visible.length} track${(added || visible.length) === 1 ? '' : 's'} to Extender.`, 'success');
        window.location.href = '/playlist_curator/extender';
    }

    // ---------- Loading filter options ----------
    async function loadFilterOptions() {
        try {
            const res = await fetch('/api/curator/filter_options');
            const data = await res.json();
            if (data) filterOptions = Object.assign(filterOptions, data);
        } catch (e) {
            console.warn('Failed to load filter options:', e);
        }
    }

    // ---------- Init ----------
    async function init() {
        await loadFilterOptions();
        addFilterRow();

        // Add rule button
        const addBtn = document.getElementById('curator-add-rule');
        if (addBtn) addBtn.addEventListener('click', () => addFilterRow());

        // Run search
        const runBtn = document.getElementById('curator-search-run');
        if (runBtn) runBtn.addEventListener('click', runSearch);

        // Clear all
        const clearBtn = document.getElementById('curator-search-clear');
        if (clearBtn) clearBtn.addEventListener('click', () => {
            const rows = document.getElementById('curator-filter-rows');
            if (rows) rows.innerHTML = '';
            addFilterRow();
            lastResults = [];
            skippedIds = new Set();
            const section = document.getElementById('curator-results-section');
            if (section) section.classList.add('hidden');
            window.curatorSetStatus('curator-search-status', '', '');
        });

        // Send to Extender
        const sendBtn = document.getElementById('curator-search-send');
        if (sendBtn) sendBtn.addEventListener('click', sendAllToExtender);

        // Add / Skip delegation
        document.addEventListener('click', (e) => {
            const addBtn = e.target.closest('[data-search-add]');
            if (addBtn) {
                e.preventDefault();
                const id = addBtn.dataset.searchAdd;
                const track = lastResults.find(r => r.item_id === id);
                if (track) window.workbenchAdd(track, 'search');
                return;
            }
            const skipBtn = e.target.closest('[data-search-skip]');
            if (skipBtn) {
                e.preventDefault();
                skippedIds.add(skipBtn.dataset.searchSkip);
                renderResults();
                return;
            }
        });

        // Re-paint results when workbench changes (other tab, sheet ×, etc.)
        document.addEventListener('curator:workbench:changed', () => {
            if (lastResults.length > 0) renderResults();
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
