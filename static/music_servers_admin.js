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
        renderCredFields(server.creds, true);
        showRegistryForm();
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
            tr.appendChild(cell(s.enabled ? 'yes' : 'no', true));

            var actions = document.createElement('div');
            if (s.is_default) {
                actions.appendChild(actionButton('Edit', showDefaultEditor));
            } else {
                actions.appendChild(actionButton('Edit', function () { startEditSecondary(s); }));
                actions.appendChild(actionButton('Set default', function () { setDefault(s.server_id); }));
                actions.appendChild(actionButton('Sweep', function () { sweep(s.server_id); }));
                actions.appendChild(actionButton('Delete', function () { removeServer(s); }));
            }
            tr.appendChild(cell(actions));
            tbody.appendChild(tr);
        });
    }

    function loadServers() {
        fetch('/api/servers', { headers: { 'Accept': 'application/json' } })
            .then(function (r) { return r.ok ? r.json() : Promise.reject(r); })
            .then(renderTable)
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
                } else {
                    feedback(el('ms-feedback'), 'Failed: ' + (d.error || 'unknown error'), false);
                }
            })
            .catch(function () { feedback(el('ms-feedback'), 'Test failed.', false); });
    }

    function setDefault(serverId) {
        jsonPost('/api/servers/' + encodeURIComponent(serverId) + '/default')
            .then(function () { loadServers(); });
    }

    function sweep(serverId) {
        jsonPost('/api/servers/' + encodeURIComponent(serverId) + '/sweep')
            .then(function (r) { return r.json(); })
            .then(function () {
                feedback(el('music-servers-error'), 'Matching sweep started for this server.', true);
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
            });
    }

    el('ms-type').addEventListener('change', function () {
        renderCredFields(null, !!el('ms-edit-id').value);
    });
    el('ms-add-btn').addEventListener('click', startAdd);
    el('ms-save-btn').addEventListener('click', save);
    el('ms-test-btn').addEventListener('click', test);
    el('ms-cancel-btn').addEventListener('click', hideRegistryForm);

    resetForm();
    hideRegistryForm();
    loadServers();
})();
