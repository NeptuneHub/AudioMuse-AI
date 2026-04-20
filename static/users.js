// Users management for the Users page. Supports normal users and admins.
// Server enforces:
// - only admins can create/delete accounts or change another user's password
// - non-admins only see themselves in /api/users and can only change their
//   own password
// Mirrored in the UI so admins see everyone plus create/delete controls,
// and non-admins only see their own row with a "Change password" button.
(function () {
    const nameInput = document.getElementById('additional-user-name');
    const passInput = document.getElementById('additional-user-password');
    const passConfirmInput = document.getElementById('additional-user-password-confirm');
    const roleInput = document.getElementById('additional-user-role');
    const addBtn = document.getElementById('additional-user-add');
    const addToggleBtn = document.getElementById('add-user-toggle');
    const addCancelBtn = document.getElementById('add-user-cancel');
    const addPanel = document.getElementById('add-user-panel');
    const feedback = document.getElementById('additional-user-feedback');
    const tbody = document.getElementById('additional-users-tbody');
    const table = document.getElementById('additional-users-table');
    if (!tbody || !table) return;

    const currentUser = table.getAttribute('data-current-user') || '';
    const isAdmin = (table.getAttribute('data-is-admin') || 'false') === 'true';

    const pwPanel = document.getElementById('change-password-panel');
    const pwTarget = document.getElementById('change-password-target');
    const pwNew = document.getElementById('change-password-new');
    const pwConfirm = document.getElementById('change-password-confirm');
    const pwSave = document.getElementById('change-password-save');
    const pwCancel = document.getElementById('change-password-cancel');
    const pwFeedback = document.getElementById('change-password-feedback');
    let pwTargetId = null;
    let pwTargetName = '';

    function showFeedback(msg, kind) {
        if (!feedback) return;
        feedback.textContent = msg || '';
        feedback.className = 'inline-feedback';
        if (kind) feedback.classList.add('status-' + kind);
        feedback.style.display = msg ? '' : 'none';
    }

    function showPwFeedback(msg, kind) {
        if (!pwFeedback) return;
        pwFeedback.textContent = msg || '';
        pwFeedback.className = 'inline-feedback';
        if (kind) pwFeedback.classList.add('status-' + kind);
        pwFeedback.style.display = msg ? '' : 'none';
    }

    function formatDate(iso) {
        if (!iso) return '';
        try {
            const d = new Date(iso);
            if (isNaN(d.getTime())) return iso;
            return d.toLocaleString();
        } catch (e) {
            return iso;
        }
    }

    function roleLabel(role) {
        return role === 'admin' ? 'admin' : 'user';
    }

    function openPasswordPanel(id, username) {
        closeAddPanel();
        pwTargetId = id;
        pwTargetName = username;
        if (pwTarget) pwTarget.textContent = 'Target user: ' + username;
        if (pwNew) pwNew.value = '';
        if (pwConfirm) pwConfirm.value = '';
        showPwFeedback('', null);
        if (pwPanel) {
            pwPanel.style.display = '';
            pwPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
        if (pwNew) pwNew.focus();
    }

    function closePasswordPanel() {
        pwTargetId = null;
        pwTargetName = '';
        if (pwPanel) pwPanel.style.display = 'none';
        if (pwNew) pwNew.value = '';
        if (pwConfirm) pwConfirm.value = '';
        showPwFeedback('', null);
    }

    function openAddPanel() {
        closePasswordPanel();
        if (nameInput) nameInput.value = '';
        if (passInput) passInput.value = '';
        if (passConfirmInput) passConfirmInput.value = '';
        if (roleInput) roleInput.value = 'user';
        showFeedback('', null);
        if (addPanel) {
            addPanel.style.display = '';
            addPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
        if (nameInput) nameInput.focus();
    }

    function closeAddPanel() {
        if (addPanel) addPanel.style.display = 'none';
        if (nameInput) nameInput.value = '';
        if (passInput) passInput.value = '';
        if (passConfirmInput) passConfirmInput.value = '';
        if (roleInput) roleInput.value = 'user';
        showFeedback('', null);
    }

    function renderUsers(users) {
        tbody.innerHTML = '';
        if (!users || users.length === 0) {
            const tr = document.createElement('tr');
            const td = document.createElement('td');
            td.colSpan = 4;
            td.style.padding = '0.75rem';
            td.style.opacity = '0.7';
            td.textContent = 'No users to display.';
            tr.appendChild(td);
            tbody.appendChild(tr);
            return;
        }
        users.forEach(function (u) {
            const tr = document.createElement('tr');
            const tdName = document.createElement('td');
            tdName.style.padding = '0.5rem';
            tdName.textContent = u.username;
            const tdRole = document.createElement('td');
            tdRole.style.padding = '0.5rem';
            tdRole.textContent = roleLabel(u.role);
            const tdCreated = document.createElement('td');
            tdCreated.style.padding = '0.5rem';
            tdCreated.textContent = formatDate(u.created_at);
            const tdAct = document.createElement('td');
            tdAct.style.padding = '0.5rem';
            tdAct.style.textAlign = 'right';

            const isSelf = currentUser && u.username === currentUser;

            // Admin can change any password; non-admin can change their own.
            if (isAdmin || isSelf) {
                const pw = document.createElement('button');
                pw.type = 'button';
                pw.className = 'btn btn-primary';
                pw.textContent = 'Change password';
                pw.style.marginRight = '0.5rem';
                pw.addEventListener('click', function () { openPasswordPanel(u.id, u.username); });
                tdAct.appendChild(pw);
            }

            if (isAdmin) {
                if (isSelf) {
                    const span = document.createElement('span');
                    span.style.opacity = '0.7';
                    span.textContent = '(current user)';
                    tdAct.appendChild(span);
                } else {
                    const del = document.createElement('button');
                    del.type = 'button';
                    del.className = 'btn btn-danger';
                    del.textContent = 'Delete';
                    del.addEventListener('click', function () { deleteUser(u.id, u.username); });
                    tdAct.appendChild(del);
                }
            } else if (isSelf) {
                const span = document.createElement('span');
                span.style.opacity = '0.7';
                span.textContent = '(current user)';
                tdAct.appendChild(span);
            }

            tr.appendChild(tdName);
            tr.appendChild(tdRole);
            tr.appendChild(tdCreated);
            tr.appendChild(tdAct);
            tbody.appendChild(tr);
        });
    }

    function loadUsers() {
        fetch('/api/users', { credentials: 'same-origin' })
            .then(function (r) {
                if (!r.ok) throw new Error('Failed to load users (' + r.status + ').');
                return r.json();
            })
            .then(function (data) { renderUsers((data && data.users) || []); })
            .catch(function (err) { showFeedback(err.message || 'Failed to load users.', 'error'); });
    }

    function addUser() {
        const username = (nameInput.value || '').trim();
        const password = passInput.value || '';
        const passwordConfirm = (passConfirmInput && passConfirmInput.value) || '';
        const role = (roleInput && roleInput.value) || 'user';
        if (!username || !password) {
            showFeedback('Username and password are required.', 'error');
            return;
        }
        if (password !== passwordConfirm) {
            showFeedback('Passwords do not match.', 'error');
            return;
        }
        addBtn.disabled = true;
        showFeedback('Creating user...', 'info');
        fetch('/api/users', {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: username, password: password, role: role })
        })
            .then(function (r) {
                return r.json().then(function (data) { return { ok: r.ok, status: r.status, data: data }; });
            })
            .then(function (res) {
                if (!res.ok) throw new Error((res.data && res.data.error) || ('Failed to create user (' + res.status + ').'));
                showFeedback('User "' + username + '" created.', 'success');
                loadUsers();
                closeAddPanel();
            })
            .catch(function (err) { showFeedback(err.message || 'Failed to create user.', 'error'); })
            .finally(function () { addBtn.disabled = false; });
    }

    function deleteUser(id, username) {
        if (!window.confirm('Delete user "' + username + '"? This cannot be undone.')) return;
        fetch('/api/users/' + encodeURIComponent(id), {
            method: 'DELETE',
            credentials: 'same-origin'
        })
            .then(function (r) {
                return r.json().then(function (data) { return { ok: r.ok, status: r.status, data: data }; });
            })
            .then(function (res) {
                if (!res.ok) throw new Error((res.data && res.data.error) || ('Failed to delete user (' + res.status + ').'));
                showFeedback('User "' + username + '" deleted.', 'success');
                loadUsers();
            })
            .catch(function (err) { showFeedback(err.message || 'Failed to delete user.', 'error'); });
    }

    function savePassword() {
        if (pwTargetId === null) return;
        const newPw = (pwNew && pwNew.value) || '';
        const confirmPw = (pwConfirm && pwConfirm.value) || '';
        if (!newPw) {
            showPwFeedback('New password cannot be empty.', 'error');
            return;
        }
        if (newPw !== confirmPw) {
            showPwFeedback('Passwords do not match.', 'error');
            return;
        }
        pwSave.disabled = true;
        showPwFeedback('Updating password...', 'info');
        const targetName = pwTargetName;
        fetch('/api/users/' + encodeURIComponent(pwTargetId) + '/password', {
            method: 'PUT',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ password: newPw })
        })
            .then(function (r) {
                return r.json().then(function (data) { return { ok: r.ok, status: r.status, data: data }; });
            })
            .then(function (res) {
                if (!res.ok) throw new Error((res.data && res.data.error) || ('Failed to update password (' + res.status + ').'));
                showPwFeedback('Password for "' + targetName + '" updated.', 'success');
                if (pwNew) pwNew.value = '';
                if (pwConfirm) pwConfirm.value = '';
                closePasswordPanel();
            })
            .catch(function (err) { showPwFeedback(err.message || 'Failed to update password.', 'error'); })
            .finally(function () { pwSave.disabled = false; });
    }

    if (addBtn) addBtn.addEventListener('click', addUser);
    if (addToggleBtn) addToggleBtn.addEventListener('click', openAddPanel);
    if (addCancelBtn) addCancelBtn.addEventListener('click', closeAddPanel);
    if (pwSave) pwSave.addEventListener('click', savePassword);
    if (pwCancel) pwCancel.addEventListener('click', closePasswordPanel);
    loadUsers();
})();
