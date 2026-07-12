// AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
// Copyright (C) 2025 NeptuneHub
// SPDX-License-Identifier: AGPL-3.0-only
//
// Setup-page admin for the media-server registry: one list of all configured
// servers with Add / Edit / Delete / Set-default / Test / Sweep, backed by the
// /api/servers endpoints. Secondary servers are edited inline here; the default
// server is edited through the existing default-server editor (setup.js), which
// writes the global config. Only rendered on the setup page.

(function () {
    var CRED_MASK = '__unchanged__';
    var CRED_FIELDS = {
        jellyfin: [
            { key: 'url', label: 'Server URL', placeholder: 'http://jellyfin:8096' },
            { key: 'user_id', label: 'User ID' },
            { key: 'token', label: 'API Token', secret: true }
        ],
        emby: [
            { key: 'url', label: 'Server URL', placeholder: 'http://emby:8096' },
            { key: 'user_id', label: 'User ID' },
            { key: 'token', label: 'API Token', secret: true }
        ],
        navidrome: [
            { key: 'url', label: 'Server URL', placeholder: 'http://navidrome:4533' },
            { key: 'user', label: 'Username' },
            { key: 'password', label: 'Password', secret: true }
        ],
        lyrion: [
            { key: 'url', label: 'Server URL', placeholder: 'http://lyrion:9000' }
        ],
        plex: [
            { key: 'url', label: 'Server URL', placeholder: 'http://plex:32400' },
            { key: 'token', label: 'Plex Token', secret: true }
        ]
    };

    var panel = document.getElementById('music-servers-section');
    if (!panel) {
        return;
    }

    function el(id) { return document.getElementById(id); }

    function feedback(node, message, ok) {
        if (!node) { return; }
        node.textContent = message;
        node.style.display = message ? 'block' : 'none';
        node.style.color = ok ? '' : '#c0392b';
    }

    function showRegistryForm() {
        var editor = el('default-server-editor');
        if (editor) { editor.style.display = 'none'; }
        el('music-server-form').style.display = 'block';
        el('music-server-form').scrollIntoView({ behavior: 'smooth' });
    }

    function hideRegistryForm() {
        if (window.PlexLink) { window.PlexLink.stop(); }
        el('music-server-form').style.display = 'none';
    }

    function showDefaultEditor() {
        hideRegistryForm();
        var editor = el('default-server-editor');
        if (editor) {
            editor.style.display = 'block';
            editor.scrollIntoView({ behavior: 'smooth' });
        }
    }

    function currentType() { return el('ms-type').value; }

    function renderCredFields(values, editing) {
        if (window.PlexLink) { window.PlexLink.stop(); }
        var fields = CRED_FIELDS[currentType()] || [];
        var mount = el('ms-cred-fields');
        mount.innerHTML = '';
        fields.forEach(function (f) {
            var wrap = document.createElement('div');
            wrap.style.display = 'flex';
            wrap.style.flexDirection = 'column';
            wrap.style.gap = '0.25rem';
            var label = document.createElement('label');
            label.textContent = f.label;
            var input = document.createElement('input');
            input.id = 'ms-cred-' + f.key;
            input.setAttribute('data-cred', f.key);
            input.type = f.secret ? 'password' : 'text';
            input.style.width = '100%';
            if (f.placeholder) { input.placeholder = f.placeholder; }
            var v = values ? values[f.key] : '';
            if (f.secret) {
                input.value = '';
                input.placeholder = editing ? 'Leave blank to keep current' : (f.placeholder || '');
            } else {
                input.value = (v == null) ? '' : v;
            }
            wrap.appendChild(label);
            wrap.appendChild(input);
            mount.appendChild(wrap);
        });
        if (currentType() === 'plex' && window.PlexLink) {
            var plexRow = document.createElement('div');
            plexRow.id = 'ms-plex-link';
            plexRow.style.marginTop = '0.25rem';
            mount.appendChild(plexRow);
            window.PlexLink.attach(plexRow, {
                getTokenInput: function () { return el('ms-cred-token'); }
            });
        }
    }

    function clearLibraryBoxes() {
        var boxes = el('ms-libraries-boxes');
        boxes.innerHTML = '';
        boxes.style.display = 'none';
    }

    function syncLibraryBoxesToInput() {
        var boxes = el('ms-libraries-boxes');
        var all = boxes.querySelector('input[data-lib-all]');
        var picks = [];
        boxes.querySelectorAll('input[data-lib-name]').forEach(function (cb) {
            cb.disabled = !!(all && all.checked);
            if (cb.checked) { picks.push(cb.getAttribute('data-lib-name')); }
        });
        el('ms-libraries').value = (all && all.checked) ? '' : picks.join(',');
    }

    function renderLibraryBoxes(libraries) {
        var boxes = el('ms-libraries-boxes');
        boxes.innerHTML = '';
        if (!Array.isArray(libraries) || !libraries.length) {
            clearLibraryBoxes();
            return;
        }
        var title = document.createElement('label');
        title.textContent = 'Music libraries to use';
        boxes.appendChild(title);
        var selected = el('ms-libraries').value.split(',')
            .map(function (s) { return s.trim().toLowerCase(); })
            .filter(function (s) { return s; });
        function row(labelText, attrs, checked) {
            var label = document.createElement('label');
            label.style.cssText = 'display:flex; align-items:center; gap:0.5rem;';
            var cb = document.createElement('input');
            cb.type = 'checkbox';
            Object.keys(attrs).forEach(function (k) { cb.setAttribute(k, attrs[k]); });
            cb.checked = checked;
            cb.addEventListener('change', syncLibraryBoxesToInput);
            label.appendChild(cb);
            label.appendChild(document.createTextNode(labelText));
            return label;
        }
        boxes.appendChild(row('No restriction (use all libraries)', { 'data-lib-all': '1' }, selected.length === 0));
        libraries.forEach(function (lib) {
            var name = lib.name || lib;
            boxes.appendChild(row(name, { 'data-lib-name': name }, selected.indexOf(String(name).toLowerCase()) !== -1));
        });
        boxes.style.display = 'flex';
        syncLibraryBoxesToInput();
    }

    function loadLibrariesIntoForm() {
        var editing = !!el('ms-edit-id').value;
        var payload = { server_type: currentType(), creds: collectCreds(editing) };
        if (editing) { payload.server_id = el('ms-edit-id').value; }
        jsonPost('/api/servers/libraries', payload)
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (d) {
                if (d && Array.isArray(d.libraries) && d.libraries.length) {
                    renderLibraryBoxes(d.libraries);
                } else {
                    clearLibraryBoxes();
                }
            })
            .catch(function () { clearLibraryBoxes(); });
    }

    function collectCreds(editing) {
        var creds = {};
        var fields = CRED_FIELDS[currentType()] || [];
        fields.forEach(function (f) {
            var input = el('ms-cred-' + f.key);
            var value = input ? input.value.trim() : '';
            if (f.secret && !value) {
                if (editing) { creds[f.key] = CRED_MASK; }
            } else {
                creds[f.key] = value;
            }
        });
        return creds;
    }

    function resetForm() {
        el('ms-edit-id').value = '';
        el('ms-name').value = '';
        el('ms-type').value = 'jellyfin';
        el('ms-libraries').value = '';
        el('ms-make-default').checked = false;
        el('ms-make-default').parentElement.style.display = '';
        el('music-server-form-title').textContent = 'Add a server';
        feedback(el('ms-feedback'), '', true);
        clearLibraryBoxes();
        renderCredFields(null, false);
    }

    function startAdd() {
        resetForm();
        showRegistryForm();
    }

    function startEditSecondary(server) {
        el('ms-edit-id').value = server.server_id;
        el('ms-name').value = server.name;
        el('ms-type').value = server.server_type;
        el('ms-libraries').value = server.music_libraries || '';
        el('ms-make-default').checked = false;
        el('ms-make-default').parentElement.style.display = 'none';
        el('music-server-form-title').textContent = 'Edit ' + server.name;
        feedback(el('ms-feedback'), '', true);
        clearLibraryBoxes();
        renderCredFields(server.creds, true);
        showRegistryForm();
        loadLibrariesIntoForm();
    }

    function actionButton(text, handler) {
        var b = document.createElement('button');
        b.type = 'button';
        b.textContent = text;
        b.style.marginRight = '0.35rem';
        b.addEventListener('click', handler);
        return b;
    }

    function renderTable(data) {
        var tbody = el('music-servers-tbody');
        tbody.innerHTML = '';
        var servers = data.servers || [];
        el('music-servers-empty').style.display = servers.length ? 'none' : 'block';
        servers.forEach(function (s) {
            var tr = document.createElement('tr');
            tr.style.borderTop = '1px solid rgba(128,128,128,0.3)';
            function cell(content, center) {
                var td = document.createElement('td');
                td.style.padding = '0.35rem';
                if (center) { td.style.textAlign = 'center'; }
                if (typeof content === 'string') { td.textContent = content; }
                else { td.appendChild(content); }
                return td;
            }
            tr.appendChild(cell(s.name));
            tr.appendChild(cell(s.server_type));
            tr.appendChild(cell(s.is_default ? 'yes' : '', true));

            var actions = document.createElement('div');
            actions.style.whiteSpace = 'nowrap';
            if (s.is_default) {
                actions.appendChild(actionButton('Edit', showDefaultEditor));
            } else {
                actions.appendChild(actionButton('Edit', function () { startEditSecondary(s); }));
                actions.appendChild(actionButton('Set default', function () { setDefault(s.server_id); }));
                actions.appendChild(actionButton('Delete', function () { removeServer(s); }));
            }
            tr.appendChild(cell(actions));
            tbody.appendChild(tr);
        });
    }

    var sweepTimer = null;
    var currentSweepTaskId = null;
    var ACTIVE_STATES = ['PENDING', 'STARTED', 'PROGRESS', 'queued', 'started', 'deferred', 'scheduled'];

    function renderSweepProgress(pct, message, active, failed) {
        var box = el('sweep-progress');
        if (!box) { return; }
        box.style.display = 'block';
        el('sweep-progress-bar').style.width = Math.max(0, Math.min(100, pct)) + '%';
        el('sweep-progress-bar').style.background = failed ? '#c0392b' : '#4a90d9';
        el('sweep-progress-pct').textContent = Math.round(pct) + '%';
        el('sweep-progress-text').textContent = message || (active ? 'Working...' : '');
        el('sweep-cancel-btn').style.display = active ? '' : 'none';
    }

    function stopSweepPolling() {
        if (sweepTimer) {
            clearInterval(sweepTimer);
            sweepTimer = null;
        }
        currentSweepTaskId = null;
        var btn = el('sweep-cancel-btn');
        if (btn) { btn.style.display = 'none'; }
    }

    function pollSweep(taskId) {
        if (!taskId) { return; }
        stopSweepPolling();
        currentSweepTaskId = taskId;
        function tick() {
            fetch('/api/status/' + encodeURIComponent(taskId), { headers: { 'Accept': 'application/json' } })
                .then(function (r) { return r.ok ? r.json() : Promise.reject(r); })
                .then(function (d) {
                    var msg = (d.details && (d.details.status_message || d.details.message)) || d.status_message || d.state || '';
                    var state = d.state || '';
                    var terminal = ['SUCCESS', 'FAILURE', 'REVOKED', 'finished', 'failed', 'canceled'].indexOf(state) !== -1;
                    var failed = state === 'FAILURE' || state === 'failed';
                    renderSweepProgress(terminal ? 100 : (d.progress || 0), msg, !terminal, failed);
                    if (terminal) {
                        stopSweepPolling();
                        loadServers();
                    }
                })
                .catch(function () { stopSweepPolling(); });
        }
        tick();
        sweepTimer = setInterval(tick, 3000);
    }

    function maybeResumeSweep(data) {
        var t = data && data.sweep_task;
        if (!t || !t.task_id) { return; }
        if (ACTIVE_STATES.indexOf(t.status) !== -1) {
            renderSweepProgress(t.progress || 0, t.message || 'Sweep in progress...', true, false);
            pollSweep(t.task_id);
        }
    }

    function loadServers() {
        fetch('/api/servers', { headers: { 'Accept': 'application/json' } })
            .then(function (r) { return r.ok ? r.json() : Promise.reject(r); })
            .then(function (data) {
                renderTable(data);
                maybeResumeSweep(data);
            })
            .catch(function () {
                feedback(el('music-servers-error'), 'Could not load servers.', false);
            });
    }

    function jsonPost(url, body, method) {
        return fetch(url, {
            method: method || 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: body ? JSON.stringify(body) : null
        });
    }

    function save() {
        var editing = !!el('ms-edit-id').value;
        var payload = {
            name: el('ms-name').value.trim(),
            server_type: currentType(),
            creds: collectCreds(editing),
            music_libraries: el('ms-libraries').value.trim(),
            make_default: el('ms-make-default').checked
        };
        if (!payload.name) {
            feedback(el('ms-feedback'), 'A display name is required.', false);
            return;
        }
        var url = editing ? '/api/servers/' + encodeURIComponent(el('ms-edit-id').value) : '/api/servers';
        jsonPost(url, payload, editing ? 'PUT' : 'POST')
            .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); })
            .then(function (res) {
                if (!res.ok) {
                    feedback(el('ms-feedback'), (res.d && res.d.error) || 'Save failed.', false);
                    return;
                }
                hideRegistryForm();
                resetForm();
                loadServers();
                if (res.d && res.d.sweep_task_id) {
                    renderSweepProgress(0, 'Matching sweep queued...', true, false);
                    pollSweep(res.d.sweep_task_id);
                }
            })
            .catch(function () { feedback(el('ms-feedback'), 'Save failed.', false); });
    }

    function test() {
        var editing = !!el('ms-edit-id').value;
        var payload = { server_type: currentType(), creds: collectCreds(editing) };
        if (editing) { payload.server_id = el('ms-edit-id').value; }
        feedback(el('ms-feedback'), 'Testing...', true);
        jsonPost('/api/servers/test', payload)
            .then(function (r) { return r.json(); })
            .then(function (d) {
                if (d.ok) {
                    feedback(el('ms-feedback'), 'Connection OK (' + (d.sample_count || 0) + ' sample tracks).', true);
                    loadLibrariesIntoForm();
                } else {
                    feedback(el('ms-feedback'), 'Failed: ' + (d.error || 'unknown error'), false);
                }
            })
            .catch(function () { feedback(el('ms-feedback'), 'Test failed.', false); });
    }

    function setDefault(serverId) {
        jsonPost('/api/servers/' + encodeURIComponent(serverId) + '/default')
            .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); })
            .then(function (res) {
                if (!res.ok) {
                    feedback(el('music-servers-error'), (res.d && res.d.error) || 'Could not set the default server.', false);
                    return;
                }
                loadServers();
            })
            .catch(function (err) {
                console.error('Set default failed:', err);
                feedback(el('music-servers-error'), 'Could not set the default server.', false);
            });
    }

    function alignServers() {
        jsonPost('/api/servers/align')
            .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); })
            .then(function (res) {
                if (!res.ok) {
                    feedback(el('music-servers-error'), (res.d && res.d.error) || 'Could not start the alignment.', false);
                    return;
                }
                if (res.d && res.d.task_id) {
                    renderSweepProgress(0, 'Music server alignment queued...', true, false);
                    pollSweep(res.d.task_id);
                }
            })
            .catch(function (err) {
                console.error('Alignment request failed:', err);
                feedback(el('music-servers-error'), 'Could not start the alignment.', false);
            });
    }

    function removeServer(server) {
        if (!window.confirm('Delete server "' + server.name + '"? Its cross-server track mappings are removed too.')) {
            return;
        }
        fetch('/api/servers/' + encodeURIComponent(server.server_id), { method: 'DELETE' })
            .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); })
            .then(function (res) {
                if (!res.ok) {
                    feedback(el('music-servers-error'), (res.d && res.d.error) || 'Delete failed.', false);
                    return;
                }
                loadServers();
            })
            .catch(function (err) {
                console.error('Delete failed:', err);
                feedback(el('music-servers-error'), 'Delete failed.', false);
            });
    }

    el('ms-type').addEventListener('change', function () {
        clearLibraryBoxes();
        renderCredFields(null, !!el('ms-edit-id').value);
    });
    el('sweep-cancel-btn').addEventListener('click', function () {
        if (!currentSweepTaskId) { return; }
        if (!window.confirm('Cancel the running background task? Only one batch task runs at a time, so this stops whatever is currently running.')) {
            return;
        }
        jsonPost('/api/cancel/' + encodeURIComponent(currentSweepTaskId))
            .then(function () {
                el('sweep-progress-text').textContent = 'Cancelling...';
            });
    });
    el('ms-add-btn').addEventListener('click', startAdd);
    el('ms-align-btn').addEventListener('click', alignServers);
    el('ms-save-btn').addEventListener('click', save);
    el('ms-test-btn').addEventListener('click', test);
    el('ms-cancel-btn').addEventListener('click', hideRegistryForm);

    resetForm();
    hideRegistryForm();
    loadServers();
})();
