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
    let renderToken = 0;

    // Pagination state — Smart Search returns pages of `per_page`. lastResults
    // is the union of all pages currently loaded (single page after Run search;
    // grows when "Load all" walks remaining pages).
    let lastPayload = null;
    let lastTotal = 0;
    let lastPage = 1;
    let lastPerPage = 500;
    let loadAllAbort = null; // { aborted: bool }

    // Render `rows` into `container` in chunks across animation frames so a
    // 5 000-row paint doesn't block the main thread. The first chunk is large
    // (so above-the-fold rows appear instantly); later chunks are smaller.
    // Subsequent calls cancel any in-flight chunked render via renderToken.
    function renderInChunks(container, rows, rowFn, opts) {
        const opt = opts || {};
        const firstChunk = opt.firstChunk || 500;
        const chunkSize = opt.chunkSize || 200;
        const myToken = ++renderToken;
        let i = 0;
        const renderSlice = (n) => {
            if (myToken !== renderToken || !container.isConnected) return;
            const slice = rows.slice(i, Math.min(i + n, rows.length));
            if (slice.length === 0) return;
            container.insertAdjacentHTML('beforeend', slice.map(rowFn).join(''));
            i += slice.length;
            if (i < rows.length) requestAnimationFrame(() => renderSlice(chunkSize));
        };
        renderSlice(firstChunk);
    }

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
            page: 1,
            per_page: 500,
        };

        // Cancel any in-flight Load-all from a previous search
        if (loadAllAbort) loadAllAbort.aborted = true;
        loadAllAbort = null;

        try {
            const res = await fetch('/api/curator/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Search failed');
            lastResults = Array.isArray(data.results) ? data.results : [];
            lastPayload = payload;
            lastTotal = data.total != null ? data.total : lastResults.length;
            lastPage = data.page || 1;
            lastPerPage = data.per_page || 500;
            skippedIds = new Set();
            renderResults();
            renderPagination();
            window.curatorSetStatus(statusId, '', '');
        } catch (e) {
            window.curatorSetStatus(statusId, e.message || 'Search failed', 'error');
        } finally {
            if (runBtn) runBtn.disabled = false;
            if (sendBtn) sendBtn.disabled = false;
        }
    }

    // ---------- Pagination ----------
    async function fetchPage(page) {
        if (!lastPayload) return;
        const statusId = 'curator-search-status';
        const payload = Object.assign({}, lastPayload, { page });
        window.curatorSetStatus(statusId, `Loading page ${page}…`, 'loading');
        try {
            const res = await fetch('/api/curator/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Page load failed');
            lastResults = Array.isArray(data.results) ? data.results : [];
            lastTotal = data.total != null ? data.total : lastResults.length;
            lastPage = data.page || page;
            lastPerPage = data.per_page || lastPerPage;
            skippedIds = new Set();
            renderResults();
            renderPagination();
            window.curatorSetStatus(statusId, '', '');
        } catch (e) {
            window.curatorSetStatus(statusId, e.message || 'Page load failed', 'error');
        }
    }

    async function loadAllPages() {
        if (!lastPayload || lastTotal <= lastResults.length) return;
        const statusId = 'curator-loadall-status';
        const totalPages = Math.ceil(lastTotal / lastPerPage);
        // Discover starting point: lastPage already loaded, fetch remaining
        const abort = { aborted: false };
        loadAllAbort = abort;
        const cancelBtn = document.getElementById('curator-page-loadall-cancel');
        const loadBtn = document.getElementById('curator-page-loadall');
        const prevBtn = document.getElementById('curator-page-prev');
        const nextBtn = document.getElementById('curator-page-next');
        if (cancelBtn) cancelBtn.classList.remove('hidden');
        if (loadBtn) loadBtn.disabled = true;
        if (prevBtn) prevBtn.disabled = true;
        if (nextBtn) nextBtn.disabled = true;

        try {
            for (let p = lastPage + 1; p <= totalPages; p++) {
                if (abort.aborted) break;
                const statusEl = document.getElementById(statusId);
                if (statusEl) statusEl.textContent = `Loading page ${p} of ${totalPages}…`;
                const payload = Object.assign({}, lastPayload, { page: p });
                const res = await fetch('/api/curator/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                if (abort.aborted) break;
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || 'Load all failed');
                if (Array.isArray(data.results) && data.results.length > 0) {
                    lastResults = lastResults.concat(data.results);
                    lastPage = data.page || p;
                }
            }
            if (!abort.aborted) {
                renderResults();
                renderPagination();
                const statusEl = document.getElementById(statusId);
                if (statusEl) statusEl.textContent = `Loaded ${lastResults.length} of ${lastTotal}`;
            }
        } catch (e) {
            const statusEl = document.getElementById(statusId);
            if (statusEl) statusEl.textContent = e.message || 'Load all failed';
        } finally {
            if (cancelBtn) cancelBtn.classList.add('hidden');
            if (loadBtn) loadBtn.disabled = false;
            renderPagination();
            if (loadAllAbort === abort) loadAllAbort = null;
        }
    }

    function renderPagination() {
        const wrap = document.getElementById('curator-pagination');
        const indicator = document.getElementById('curator-page-indicator');
        const prev = document.getElementById('curator-page-prev');
        const next = document.getElementById('curator-page-next');
        const loadAll = document.getElementById('curator-page-loadall');
        const totalEl = document.getElementById('curator-results-total');

        const totalPages = lastPerPage > 0 ? Math.ceil(lastTotal / lastPerPage) : 1;
        const showPager = lastTotal > lastPerPage;

        if (wrap) wrap.classList.toggle('hidden', !showPager);
        if (indicator) indicator.textContent = `Page ${lastPage} / ${totalPages || 1}`;
        if (prev) prev.disabled = lastPage <= 1;
        if (next) next.disabled = lastPage >= totalPages;
        if (loadAll) {
            const remaining = lastTotal - lastResults.length;
            loadAll.disabled = remaining <= 0;
            loadAll.textContent = remaining > 0 ? `Load all (${lastTotal})` : 'All loaded';
        }
        if (totalEl) {
            totalEl.textContent = lastTotal > lastResults.length
                ? ` of ${lastTotal} on page ${lastPage}/${totalPages}`
                : (lastTotal > 0 ? ` of ${lastTotal}` : '');
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
            // Cancel any in-flight chunked render before swapping in the empty state
            renderToken++;
            const empty = `<div class="curator-empty-state">No tracks match these rules. Try loosening one.</div>`;
            if (wrap) wrap.innerHTML = empty;
            if (cards) cards.innerHTML = empty;
            return;
        }

        if (wrap) {
            wrap.innerHTML = renderTableShell();
            const tbody = wrap.querySelector('tbody');
            if (tbody) renderInChunks(tbody, visible, rowHtml);
        }
        if (cards) {
            cards.innerHTML = '';
            renderInChunks(cards, visible, cardHtml);
        }
    }

    function renderTableShell() {
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
                <tbody></tbody>
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
                <span class="curator-pill" data-tone="success" title="In Workbench">${ICONS.check} Added</span>
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

    function cardHtml(t) {
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
    }

    // Surgical per-row swap. Returns true when at least one row was updated;
    // caller falls back to renderResults() when false (e.g. id not on this page).
    function updateRowsForChanges(changedIds) {
        if (!Array.isArray(changedIds) || changedIds.length === 0) return false;
        if (lastResults.length === 0) return false;
        const idSet = new Set(changedIds.map(String));
        const tracks = lastResults.filter(r => idSet.has(String(r.item_id)));
        if (tracks.length === 0) return false;
        let updated = false;
        tracks.forEach(t => {
            const sel = `[data-row-id="${CSS.escape(String(t.item_id))}"]`;
            const tr = document.querySelector('tr' + sel);
            if (tr) {
                const tmp = document.createElement('tbody');
                tmp.innerHTML = rowHtml(t);
                const next = tmp.firstElementChild;
                if (next) { tr.replaceWith(next); updated = true; }
            }
            const card = document.querySelector('.curator-track-card' + sel);
            if (card) {
                const tmp = document.createElement('div');
                tmp.innerHTML = cardHtml(t);
                const next = tmp.firstElementChild;
                if (next) { card.replaceWith(next); updated = true; }
            }
        });
        return updated;
    }

    // ---------- Send to Extender ----------
    function sendAllToExtender() {
        const visible = visibleResults();
        if (visible.length === 0) return;
        const added = window.workbenchAddBulk(visible, 'search');
        // The save normally runs on a 250ms debounce. Force a synchronous flush
        // so the localStorage write completes before navigation aborts the page.
        if (typeof window.workbenchFlushSync === 'function') window.workbenchFlushSync();
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
            lastPayload = null;
            lastTotal = 0;
            lastPage = 1;
            if (loadAllAbort) loadAllAbort.aborted = true;
            const section = document.getElementById('curator-results-section');
            if (section) section.classList.add('hidden');
            window.curatorSetStatus('curator-search-status', '', '');
        });

        // Send to Extender
        const sendBtn = document.getElementById('curator-search-send');
        if (sendBtn) sendBtn.addEventListener('click', sendAllToExtender);

        // Pagination
        const prevBtn = document.getElementById('curator-page-prev');
        if (prevBtn) prevBtn.addEventListener('click', () => {
            if (lastPage > 1) fetchPage(lastPage - 1);
        });
        const nextBtn = document.getElementById('curator-page-next');
        if (nextBtn) nextBtn.addEventListener('click', () => {
            const totalPages = lastPerPage > 0 ? Math.ceil(lastTotal / lastPerPage) : 1;
            if (lastPage < totalPages) fetchPage(lastPage + 1);
        });
        const loadAllBtn = document.getElementById('curator-page-loadall');
        if (loadAllBtn) loadAllBtn.addEventListener('click', loadAllPages);
        const cancelBtn = document.getElementById('curator-page-loadall-cancel');
        if (cancelBtn) cancelBtn.addEventListener('click', () => {
            if (loadAllAbort) loadAllAbort.aborted = true;
        });

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
        document.addEventListener('curator:workbench:changed', (e) => {
            if (lastResults.length === 0) return;
            const changedIds = e && e.detail ? e.detail.changedIds : null;
            if (changedIds && updateRowsForChanges(changedIds)) {
                // Surgical swap succeeded; "Send N to Extender" button label may need a refresh.
                const sendBtn = document.getElementById('curator-search-send');
                if (sendBtn && typeof visibleResults === 'function') {
                    const visible = visibleResults();
                    sendBtn.classList.toggle('hidden', visible.length === 0);
                    if (visible.length > 0) {
                        sendBtn.innerHTML = `Send ${visible.length} to Extender <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg>`;
                    }
                }
                return;
            }
            renderResults();
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
