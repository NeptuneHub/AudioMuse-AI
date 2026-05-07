/* ============================================================================
   AudioMuse-AI — Setup Wizard
   Companion script that runs ALONGSIDE the existing setup.js.

   setup.js renders all advanced field-rows into a single #advanced-fields
   container. This script:
     1. Watches that container with a MutationObserver.
     2. After the first batch of rows lands, walks them and moves each row
        into the right group container based on the input's `name` attribute.
     3. Wires conditional visibility (AI provider, clustering algorithm,
        lyrics master toggle, lyrics-API toggle).
     4. Adds a search filter that hides non-matching rows across all groups.
     5. Adds rail scroll-spy + smooth-scroll on rail-item click.

   Hard rule: every input keeps its original name and id. Visibility uses
   .style.display so values still submit when fields are hidden.
   ============================================================================ */

(function () {
    'use strict';

    // ----------------------- group classification ------------------------

    // Which group each known field name belongs to. First match wins.
    // Anything not matched ends up in "everything-else".
    var GROUPS = [
        {
            id: 'group-server-extras',
            names: ['PROBE_TOP_PLAYED_LIMIT', 'MIGRATION_UNMATCHED_ALBUMS_PAYLOAD_LIMIT']
        },
        {
            id: 'group-auth-extras',
            names: ['ENABLE_PROXY_FIX']
        },
        {
            id: 'group-ai-fields',
            names: [
                'AI_MODEL_PROVIDER', 'AI_REQUEST_TIMEOUT_SECONDS', 'MAX_SONGS_IN_AI_PROMPT'
            ],
            prefixes: ['OLLAMA_', 'OPENAI_', 'GEMINI_', 'MISTRAL_']
        },
        {
            id: 'group-lyrics-fields',
            names: [
                'LYRICS_ENABLED', 'LYRICS_LLM_ENABLED', 'LYRICS_API_ENABLE',
                'MUSICSERVER_LYRICS_TIMEOUT'
            ]
        },
        {
            id: 'group-clustering-fields',
            names: [
                'CLUSTER_ALGORITHM', 'ENABLE_CLUSTERING_EMBEDDINGS', 'USE_GPU_CLUSTERING',
                'TOP_K_MOODS_FOR_PURITY_CALCULATION',
                'OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY',
                'TOP_N_MOODS', 'TOP_N_OTHER_FEATURES',
                'EMBEDDING_DIMENSION', 'ENERGY_MIN', 'ENERGY_MAX',
                'MIN_SONGS_PER_GENRE_FOR_STRATIFICATION',
                'STRATIFIED_SAMPLING_TARGET_PERCENTILE',
                'SAMPLING_PERCENTAGE_CHANGE_PER_RUN',
                'AUDIO_LOAD_TIMEOUT', 'REBUILD_INDEX_BATCH_SIZE',
                'MAX_QUEUED_ANALYSIS_JOBS',
                'ITERATIONS_PER_BATCH_JOB', 'MAX_CONCURRENT_BATCH_JOBS',
                'DB_FETCH_CHUNK_SIZE', 'CLEANING_SAFETY_LIMIT',
                'TOP_N_ELITES'
            ],
            prefixes: [
                'CLUSTERING_', 'DBSCAN_', 'NUM_CLUSTERS_', 'GMM_', 'SPECTRAL_',
                'PCA_COMPONENTS_', 'EXPLOITATION_', 'MUTATION_', 'SCORE_WEIGHT_'
            ]
        },
        {
            id: 'group-playlists-fields',
            names: [
                'TOP_N_PLAYLISTS', 'MIN_PLAYLIST_SIZE_FOR_TOP_N',
                'MAX_DISTANCE',
                'MAX_SONGS_PER_CLUSTER', 'MAX_SONGS_PER_ARTIST',
                'MAX_SONGS_PER_ARTIST_PLAYLIST',
                'PLAYLIST_ENERGY_ARC',
                'NUM_RECENT_ALBUMS',
                'ARTIST_INDEX_MAX_PART_SIZE_MB',
                'PER_SONG_MODEL_RELOAD'
            ],
            prefixes: [
                'SIMILARITY_', 'VOYAGER_', 'PATH_', 'ALCHEMY_',
                'SONIC_FINGERPRINT_', 'CLAP_', 'SEM_GROVE_',
                'DUPLICATE_DISTANCE_', 'MOOD_SIMILARITY_'
            ]
        }
    ];

    // The rail anchor each card maps to (used for scroll-spy and search).
    var CARD_TO_RAIL = {
        'group-server':           'group-server',
        'group-auth':             'group-auth',
        'group-ai':               'group-ai',
        'group-lyrics':           'group-lyrics',
        'group-clustering':       'group-clustering',
        'group-playlists':        'group-playlists',
        'group-everything-else':  'group-everything-else'
    };

    // Map child container id → owning card id.
    var CONTAINER_TO_CARD = {
        'group-server-extras':         'group-server',
        'group-auth-extras':           'group-auth',
        'group-ai-fields':             'group-ai',
        'group-lyrics-fields':         'group-lyrics',
        'group-clustering-fields':     'group-clustering',
        'group-playlists-fields':      'group-playlists',
        'group-everything-else-fields':'group-everything-else'
    };

    function classifyField(name) {
        if (!name) return 'group-everything-else-fields';
        for (var i = 0; i < GROUPS.length; i++) {
            var g = GROUPS[i];
            if (g.names && g.names.indexOf(name) !== -1) return g.id;
            if (g.prefixes) {
                for (var j = 0; j < g.prefixes.length; j++) {
                    if (name.indexOf(g.prefixes[j]) === 0) return g.id;
                }
            }
        }
        return 'group-everything-else-fields';
    }

    // -------------------- redistribution: move rows into groups ------------------

    function findInputName(row) {
        // Each rendered .field-row contains exactly one input/select/textarea
        // whose name attribute matches the underlying config key.
        var el = row.querySelector('input[name], select[name], textarea[name]');
        return el ? el.getAttribute('name') : null;
    }

    function relabelRowFromName(row, name) {
        // setup.js renders advanced fields with the raw config key as the
        // visible label (e.g. "OLLAMA_SERVER_URL"). That's noisy; humanize
        // the visible text while keeping the input untouched.
        var label = row.querySelector('label');
        if (!label) return;
        var labelTextNode = null;
        for (var i = 0; i < label.childNodes.length; i++) {
            if (label.childNodes[i].nodeType === Node.TEXT_NODE) {
                labelTextNode = label.childNodes[i];
                break;
            }
        }
        if (!labelTextNode) return;
        // Only humanize when the existing text is the bare config key.
        var current = String(labelTextNode.nodeValue || '').trim();
        if (current && current.toUpperCase() === name) {
            labelTextNode.nodeValue = humanizeName(name) + ' ';
        }
    }

    function humanizeName(name) {
        // Lowercase + spaces, but keep common acronyms uppercase.
        var ACRONYMS = {
            ai: 'AI', api: 'API', url: 'URL', id: 'ID',
            jwt: 'JWT', json: 'JSON', cpu: 'CPU', gpu: 'GPU',
            db: 'DB', llm: 'LLM', clap: 'CLAP', mulan: 'MuLan',
            pca: 'PCA', kmeans: 'K-Means', dbscan: 'DBSCAN',
            gmm: 'GMM', mb: 'MB', bpm: 'BPM',
            mpd: 'MPD', mcp: 'MCP'
        };
        var parts = name.toLowerCase().split('_').filter(Boolean);
        return parts.map(function (p, idx) {
            if (ACRONYMS[p]) return ACRONYMS[p];
            if (idx === 0) return p.charAt(0).toUpperCase() + p.slice(1);
            return p;
        }).join(' ');
    }

    function tagRowForSearch(row, name) {
        var input = row.querySelector('input[name], select[name], textarea[name]');
        var labelText = '';
        var labelEl = row.querySelector('label');
        if (labelEl) {
            labelText = (labelEl.textContent || '').trim();
        }
        var hint = '';
        var hintEl = row.querySelector('small');
        if (hintEl) hint = (hintEl.textContent || '').trim();
        var haystack = [name, labelText, hint].join(' ').toLowerCase();
        row.dataset.searchKey = haystack;
        row.dataset.fieldName = name;
        if (input) {
            row.dataset.fieldId = input.id || '';
        }
    }

    // Convert the rendered text input for fields that should really be
    // dropdowns. Preserves name + value. setup.js's collectConfigFromForm
    // walks FormData by name, so the submitted payload is unchanged.
    var SELECT_OVERRIDES = {
        AI_MODEL_PROVIDER: ['NONE', 'OLLAMA', 'OPENAI', 'GEMINI', 'MISTRAL'],
        CLUSTER_ALGORITHM: ['kmeans', 'dbscan', 'gmm', 'spectral'],
        GMM_COVARIANCE_TYPE: ['full', 'tied', 'diag', 'spherical'],
        VOYAGER_METRIC: ['angular', 'euclidean', 'dot'],
        PATH_DISTANCE_METRIC: ['angular', 'euclidean']
    };

    function maybeUpgradeToSelect(row, name) {
        var options = SELECT_OVERRIDES[name];
        if (!options) return null;
        var input = row.querySelector('input[name="' + name + '"]');
        if (!input) return null;
        var current = String(input.value || '').trim();
        var sel = document.createElement('select');
        sel.id = input.id;
        sel.name = input.name;
        if (input.required) sel.required = true;
        // setup.js stamps dataset.originalValue on every input so
        // collectConfigFromForm() can detect "unchanged" and skip the field.
        // Mirror it onto the select so that contract still holds.
        sel.dataset.originalValue = input.dataset.originalValue !== undefined
            ? input.dataset.originalValue
            : current;
        var foundCurrent = false;
        for (var i = 0; i < options.length; i++) {
            var opt = document.createElement('option');
            opt.value = options[i];
            opt.textContent = options[i];
            if (current && options[i].toUpperCase() === current.toUpperCase()) {
                opt.selected = true;
                foundCurrent = true;
            }
            sel.appendChild(opt);
        }
        if (!foundCurrent && current) {
            // Preserve any unexpected legacy value as an extra option so
            // the form still round-trips the original payload.
            var legacy = document.createElement('option');
            legacy.value = current;
            legacy.textContent = current + ' (current)';
            legacy.selected = true;
            sel.insertBefore(legacy, sel.firstChild);
        }
        input.parentNode.replaceChild(sel, input);
        return sel;
    }

    function placeRow(row) {
        var name = findInputName(row);
        if (!name) return;
        relabelRowFromName(row, name);
        maybeUpgradeToSelect(row, name);
        tagRowForSearch(row, name);
        var targetId = classifyField(name);
        var target = document.getElementById(targetId);
        if (!target) {
            // Fallback to "everything else" if the target container is missing.
            target = document.getElementById('group-everything-else-fields');
        }
        if (target) {
            target.appendChild(row);
        }
    }

    function redistribute() {
        var staging = document.getElementById('advanced-fields');
        if (!staging) return false;
        if (!staging.firstChild) return false;
        // Snapshot children first — appending while iterating mutates the live list.
        var rows = Array.prototype.slice.call(staging.children);
        rows.forEach(function (row) {
            // Only handle .field-row entries; skip anything else.
            if (row.classList && row.classList.contains('field-row')) {
                placeRow(row);
            } else {
                // Be permissive: setup.js renders each field as a div without
                // checking for the .field-row class on every render. Treat
                // any direct child as a row candidate.
                placeRow(row);
            }
        });
        // Hide the empty-state hint if "everything else" got something.
        var emptyHint = document.getElementById('group-everything-else-empty');
        var ee = document.getElementById('group-everything-else-fields');
        if (emptyHint && ee) {
            emptyHint.hidden = ee.childElementCount > 0;
        }
        return true;
    }

    // --------------------- conditional visibility -----------------------

    function setRowVisibility(row, visible) {
        if (!row) return;
        if (visible) {
            row.style.removeProperty('display');
            row.removeAttribute('data-conditionally-hidden');
        } else {
            row.style.display = 'none';
            row.setAttribute('data-conditionally-hidden', '1');
        }
    }

    function findRowByName(name) {
        // After redistribution, .field-row entries live inside group containers.
        // Find them by the input/select name.
        var input = document.querySelector(
            '#setup-form input[name="' + name + '"], ' +
            '#setup-form select[name="' + name + '"], ' +
            '#setup-form textarea[name="' + name + '"]'
        );
        if (!input) return null;
        return input.closest('.field-row') || input.parentElement;
    }

    function getStringValue(name) {
        var el = document.querySelector(
            '#setup-form [name="' + name + '"]'
        );
        if (!el) return '';
        return String(el.value || '').trim();
    }

    function isTruthyText(value) {
        var v = String(value || '').trim().toLowerCase();
        return v === '1' || v === 'true' || v === 'yes' || v === 'on';
    }

    function applyAiProviderVisibility() {
        var providerRaw = getStringValue('AI_MODEL_PROVIDER').toUpperCase();
        var BY_PROVIDER = {
            OLLAMA: ['OLLAMA_'],
            OPENAI: ['OPENAI_'],
            GEMINI: ['GEMINI_'],
            MISTRAL: ['MISTRAL_']
        };
        var allPrefixes = ['OLLAMA_', 'OPENAI_', 'GEMINI_', 'MISTRAL_'];
        // Collect every provider-specific row inside the AI group.
        // tagRowForSearch() stamps each relocated row with data-field-name,
        // which is what we match on here.
        var aiContainer = document.getElementById('group-ai-fields');
        if (!aiContainer) return;
        var rows = aiContainer.querySelectorAll('[data-field-name]');
        for (var i = 0; i < rows.length; i++) {
            var row = rows[i];
            var name = row.dataset.fieldName || '';
            var matchedPrefix = null;
            for (var j = 0; j < allPrefixes.length; j++) {
                if (name.indexOf(allPrefixes[j]) === 0) {
                    matchedPrefix = allPrefixes[j];
                    break;
                }
            }
            if (!matchedPrefix) continue; // shared field (timeout, max_songs)
            var allowed = BY_PROVIDER[providerRaw] || [];
            setRowVisibility(row, allowed.indexOf(matchedPrefix) !== -1);
        }
    }

    function applyClusteringAlgorithmVisibility() {
        var algo = getStringValue('CLUSTER_ALGORITHM').toLowerCase();
        var BY_ALGO = {
            kmeans:   ['NUM_CLUSTERS_'],
            dbscan:   ['DBSCAN_'],
            gmm:      ['GMM_'],
            spectral: ['SPECTRAL_']
        };
        var allPrefixes = ['NUM_CLUSTERS_', 'DBSCAN_', 'GMM_', 'SPECTRAL_'];
        var container = document.getElementById('group-clustering-fields');
        if (!container) return;
        var rows = container.querySelectorAll('[data-field-name]');
        for (var i = 0; i < rows.length; i++) {
            var row = rows[i];
            var name = row.dataset.fieldName || '';
            var matchedPrefix = null;
            for (var j = 0; j < allPrefixes.length; j++) {
                if (name.indexOf(allPrefixes[j]) === 0) {
                    matchedPrefix = allPrefixes[j];
                    break;
                }
            }
            if (!matchedPrefix) continue;
            var allowed = BY_ALGO[algo] || [];
            setRowVisibility(row, allowed.indexOf(matchedPrefix) !== -1);
        }
    }

    function applyLyricsVisibility() {
        var lyricsOn = isTruthyText(getStringValue('LYRICS_ENABLED'));
        var apiOn    = isTruthyText(getStringValue('LYRICS_API_ENABLE'));

        var dependentNames = ['LYRICS_LLM_ENABLED', 'LYRICS_API_ENABLE', 'MUSICSERVER_LYRICS_TIMEOUT'];
        for (var i = 0; i < dependentNames.length; i++) {
            var row = findRowByName(dependentNames[i]);
            if (row) setRowVisibility(row, lyricsOn);
        }

        var apiBlock = document.getElementById('lyrics-api-config');
        if (apiBlock) {
            apiBlock.style.display = (lyricsOn && apiOn) ? '' : 'none';
        }
    }

    function wireConditionalListeners() {
        // Provider / algorithm — listen on the (possibly upgraded-to-select) field.
        ['AI_MODEL_PROVIDER', 'CLUSTER_ALGORITHM'].forEach(function (name) {
            var el = document.querySelector('#setup-form [name="' + name + '"]');
            if (!el) return;
            var handler = name === 'AI_MODEL_PROVIDER'
                ? applyAiProviderVisibility
                : applyClusteringAlgorithmVisibility;
            el.addEventListener('change', handler);
            el.addEventListener('input', handler);
        });
        // Lyrics: text inputs (true/false), so listen for input.
        ['LYRICS_ENABLED', 'LYRICS_API_ENABLE'].forEach(function (name) {
            var el = document.querySelector('#setup-form [name="' + name + '"]');
            if (!el) return;
            el.addEventListener('input', applyLyricsVisibility);
            el.addEventListener('change', applyLyricsVisibility);
        });
    }

    function applyAllConditionalVisibility() {
        applyAiProviderVisibility();
        applyClusteringAlgorithmVisibility();
        applyLyricsVisibility();
    }

    // ------------------------- search -------------------------

    function applySearch(query) {
        var q = String(query || '').trim().toLowerCase();
        var clearBtn = document.getElementById('wizard-search-clear');
        if (clearBtn) clearBtn.hidden = q.length === 0;

        var rows = document.querySelectorAll('#setup-form .grouped-fields > [data-search-key], #setup-form .grouped-fields > [data-field-name]');
        var hitCounts = {}; // cardId -> count

        // First pass: tag matching rows.
        for (var i = 0; i < rows.length; i++) {
            var row = rows[i];
            var key = row.dataset.searchKey || '';
            var matches = !q || key.indexOf(q) !== -1;
            if (matches) {
                row.classList.remove('is-search-hidden');
                // But if it's hidden by conditional visibility, leave that alone.
                // (we only toggle the search-hidden class; conditional
                // visibility uses inline display:none.)
            } else {
                row.classList.add('is-search-hidden');
            }
        }

        // Cards: hide a card entirely if every row inside is hidden by search
        // AND the card has no template-static fields. Cards with template
        // markup (server, auth, lyrics-api) always stay visible because they
        // host fields that are not data-attribute tagged.
        var TEMPLATE_CARDS = ['group-server', 'group-auth', 'group-lyrics'];
        var allCards = document.querySelectorAll('#setup-form .wizard-card');
        for (var k = 0; k < allCards.length; k++) {
            var card = allCards[k];
            var cardId = card.id;
            if (!q) {
                card.classList.remove('is-search-hidden');
                continue;
            }
            if (TEMPLATE_CARDS.indexOf(cardId) !== -1) {
                // Match against the card's own header text + every visible
                // field-row inside it.
                var headerText = (card.querySelector('.wizard-card-head').textContent || '').toLowerCase();
                var hasMatch = headerText.indexOf(q) !== -1
                    || card.querySelector('.field-row:not(.is-search-hidden) [name]')
                    || card.querySelector('[data-search-key]:not(.is-search-hidden)');
                card.classList.toggle('is-search-hidden', !hasMatch);
            } else {
                var visibleRows = card.querySelectorAll('[data-search-key]:not(.is-search-hidden), [data-field-name]:not(.is-search-hidden)');
                card.classList.toggle('is-search-hidden', visibleRows.length === 0);
            }
        }

        // Mirror visibility onto the rail items so the rail collapses too.
        var railItems = document.querySelectorAll('.wizard-rail-item');
        for (var m = 0; m < railItems.length; m++) {
            var ri = railItems[m];
            var targetId = (ri.getAttribute('href') || '').replace(/^#/, '');
            var card = document.getElementById(targetId);
            if (!card) continue;
            ri.classList.toggle('is-search-hidden', card.classList.contains('is-search-hidden'));
        }
    }

    function wireSearch() {
        var input = document.getElementById('wizard-search-input');
        var clearBtn = document.getElementById('wizard-search-clear');
        if (!input) return;
        input.addEventListener('input', function () {
            applySearch(input.value);
        });
        if (clearBtn) {
            clearBtn.addEventListener('click', function () {
                input.value = '';
                applySearch('');
                input.focus();
            });
        }
    }

    // ------------------------- rail scroll-spy -------------------------

    function wireRail() {
        var items = document.querySelectorAll('.wizard-rail-item');
        if (!items.length) return;

        items.forEach(function (item) {
            item.addEventListener('click', function (e) {
                var href = item.getAttribute('href') || '';
                if (!href.startsWith('#')) return;
                var target = document.getElementById(href.slice(1));
                if (!target) return;
                e.preventDefault();
                // Set active state immediately for snappy feedback.
                items.forEach(function (it) { it.classList.remove('is-active'); });
                item.classList.add('is-active');
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                // Move focus to the card heading for screen readers.
                var heading = target.querySelector('h2');
                if (heading) {
                    heading.setAttribute('tabindex', '-1');
                    heading.focus({ preventScroll: true });
                }
            });
        });

        // Scroll-spy: activate the rail item whose card is currently in view.
        if (!('IntersectionObserver' in window)) return;
        var observer = new IntersectionObserver(function (entries) {
            // Pick the entry closest to the top of the viewport.
            var visible = entries
                .filter(function (e) { return e.isIntersecting; })
                .sort(function (a, b) { return a.boundingClientRect.top - b.boundingClientRect.top; });
            if (visible.length === 0) return;
            var topEntry = visible[0];
            var id = topEntry.target.id;
            items.forEach(function (it) {
                var href = (it.getAttribute('href') || '').slice(1);
                it.classList.toggle('is-active', href === id);
            });
        }, {
            rootMargin: '-20% 0px -55% 0px',
            threshold: 0
        });
        document.querySelectorAll('#setup-form .wizard-card').forEach(function (card) {
            observer.observe(card);
        });
    }

    // ------------------------- bootstrap -------------------------

    function init() {
        wireRail();
        wireSearch();

        var staging = document.getElementById('advanced-fields');
        if (!staging) return;

        // setup.js fetches /api/setup async, so #advanced-fields starts empty.
        // Wait for the first batch of children, redistribute, then run again
        // a tick later to catch any late additions before disconnecting.
        var done = false;
        function runOnce() {
            if (done) return;
            if (!staging.firstChild) return;
            // Drain everything currently in the staging container.
            redistribute();
            done = true;
            observer.disconnect();
            // Tag any rows that were already template-static (in server/auth/lyrics
            // cards) so search can find them too.
            tagStaticRowsForSearch();
            wireConditionalListeners();
            applyAllConditionalVisibility();
        }

        var observer = new MutationObserver(function () {
            if (staging.firstChild) {
                // Microtask delay — let setup.js finish its append loop first.
                Promise.resolve().then(runOnce);
            }
        });
        observer.observe(staging, { childList: true });
        // Also try immediately in case the data was already there (rare).
        if (staging.firstChild) {
            runOnce();
        }
    }

    function tagStaticRowsForSearch() {
        // Apply data-search-key to template-rendered rows so the search
        // input can match against them too.
        var rows = document.querySelectorAll('#setup-form .field-row');
        for (var i = 0; i < rows.length; i++) {
            var row = rows[i];
            if (row.dataset.searchKey) continue; // already tagged
            var input = row.querySelector('input[name], select[name], textarea[name]');
            var name = input ? input.getAttribute('name') : '';
            var labelEl = row.querySelector('label');
            var labelText = labelEl ? (labelEl.textContent || '').trim() : '';
            row.dataset.searchKey = (name + ' ' + labelText).toLowerCase();
            if (name) row.dataset.fieldName = name;
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
