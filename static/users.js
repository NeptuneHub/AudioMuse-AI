(function () {
    const nameInput = document.getElementById('additional-user-name');
    const passInput = document.getElementById('additional-user-password');
    const passConfirmInput = document.getElementById('additional-user-password-confirm');
    const currentInput = document.getElementById('additional-user-current-password');
    const roleInput = document.getElementById('additional-user-role');
    const addBtn = document.getElementById('additional-user-add');
    const addToggleBtn = document.getElementById('add-user-toggle');
    const addCancelBtn = document.getElementById('add-user-cancel');
    const addPanel = document.getElementById('add-user-panel');
    const feedback = document.getElementById('additional-user-feedback');
    const pageFeedback = document.getElementById('users-page-feedback');
    const tbody = document.getElementById('additional-users-tbody');
    const table = document.getElementById('additional-users-table');
    if (!tbody || !table) return;

    const currentUser = table.getAttribute('data-current-user') || '';
    const isAdmin = (table.getAttribute('data-is-admin') || 'false') === 'true';

    const pwPanel = document.getElementById('change-password-panel');
    const pwTarget = document.getElementById('change-password-target');
    const pwNew = document.getElementById('change-password-new');
    const pwConfirm = document.getElementById('change-password-confirm');
    const pwCurrent = document.getElementById('change-password-current');
    const pwCurrentLabel = document.getElementById('change-password-current-label');
    const pwConfirmHint = document.getElementById('change-password-confirm-hint');
    const pwSave = document.getElementById('change-password-save');
    const pwCancel = document.getElementById('change-password-cancel');
    const pwFeedback = document.getElementById('change-password-feedback');
    let pwTargetId = null;
    let pwTargetName = '';

    const delPanel = document.getElementById('delete-user-panel');
    const delTarget = document.getElementById('delete-user-target');
    const delCurrent = document.getElementById('delete-user-current');
    const delConfirmBtn = document.getElementById('delete-user-confirm');
    const delCancelBtn = document.getElementById('delete-user-cancel');
    const delFeedback = document.getElementById('delete-user-feedback');
    let delTargetId = null;
    let delTargetName = '';

    function feedbackRenderer(el) {
        return function (msg, kind) {
            if (!el) return;
            el.textContent = msg || '';
            el.className = 'inline-feedback';
            if (kind) el.classList.add('status-' + kind);
            el.style.display = msg ? '' : 'none';
        };
    }

    const showFeedback = feedbackRenderer(feedback);
    const showPageFeedback = feedbackRenderer(pageFeedback);
    const showPwFeedback = feedbackRenderer(pwFeedback);
    const showDelFeedback = feedbackRenderer(delFeedback);

    function formatDate(iso) {
        return iso ? iso.replace('T', ' ').substring(0, 19) : '';
    }

    function roleLabel(role) {
        return role === 'admin' ? 'admin' : 'user';
    }

    function openPasswordPanel(id, username) {
        closeAddPanel();
        closeDeletePanel();
        pwTargetId = id;
        pwTargetName = username;
        if (pwTarget) pwTarget.textContent = 'Target user: ' + username;
        const isSelf = currentUser && username === currentUser;
        if (isSelf) {
            if (pwConfirmHint) pwConfirmHint.textContent = 'Confirm this change with your current password.';
            if (pwCurrentLabel) pwCurrentLabel.textContent = 'Current password';
            if (pwCurrent) pwCurrent.placeholder = 'your current password';
        } else {
            if (pwConfirmHint) pwConfirmHint.textContent = 'You are changing the password of "' + username + '". Enter your admin password to confirm this operation.';
            if (pwCurrentLabel) pwCurrentLabel.textContent = 'Your admin password';
            if (pwCurrent) pwCurrent.placeholder = 'your admin password';
        }
        if (pwNew) pwNew.value = '';
        if (pwConfirm) pwConfirm.value = '';
        if (pwCurrent) pwCurrent.value = '';
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
        if (pwCurrent) pwCurrent.value = '';
        showPwFeedback('', null);
    }

    function openDeletePanel(id, username) {
        closeAddPanel();
        closePasswordPanel();
        delTargetId = id;
        delTargetName = username;
        if (delTarget) delTarget.textContent = 'Target user: ' + username;
        if (delCurrent) delCurrent.value = '';
        showDelFeedback('', null);
        if (delPanel) {
            delPanel.style.display = '';
            delPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
        if (delCurrent) delCurrent.focus();
    }

    function closeDeletePanel() {
        delTargetId = null;
        delTargetName = '';
        if (delPanel) delPanel.style.display = 'none';
        if (delCurrent) delCurrent.value = '';
        showDelFeedback('', null);
    }

    function openAddPanel() {
        closePasswordPanel();
        closeDeletePanel();
        if (nameInput) nameInput.value = '';
        if (passInput) passInput.value = '';
        if (passConfirmInput) passConfirmInput.value = '';
        if (currentInput) currentInput.value = '';
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
        if (currentInput) currentInput.value = '';
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
                    del.addEventListener('click', function () { openDeletePanel(u.id, u.username); });
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
            .catch(function (err) { showPageFeedback(err.message || 'Failed to load users.', 'error'); });
    }

    function addUser() {
        const username = (nameInput.value || '').trim();
        const password = passInput.value || '';
        const passwordConfirm = (passConfirmInput && passConfirmInput.value) || '';
        const currentPw = (currentInput && currentInput.value) || '';
        const role = (roleInput && roleInput.value) || 'user';
        if (!username || !password) {
            showFeedback('Username and password are required.', 'error');
            return;
        }
        if (password !== passwordConfirm) {
            showFeedback('Passwords do not match.', 'error');
            return;
        }
        if (!currentPw) {
            showFeedback('Your admin password is required to confirm.', 'error');
            return;
        }
        addBtn.disabled = true;
        showFeedback('Creating user...', 'info');
        fetch('/api/users', {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                username: username,
                password: password,
                current_password: currentPw,
                role: role
            })
        })
            .then(function (r) {
                return r.json().then(function (data) { return { ok: r.ok, status: r.status, data: data }; });
            })
            .then(function (res) {
                if (!res.ok) throw new Error((res.data && res.data.error) || ('Failed to create user (' + res.status + ').'));
                closeAddPanel();
                showPageFeedback('User "' + username + '" created.', 'success');
                loadUsers();
            })
            .catch(function (err) {
                if (currentInput) currentInput.value = '';
                showFeedback(err.message || 'Failed to create user.', 'error');
            })
            .finally(function () { addBtn.disabled = false; });
    }

    function deleteUser() {
        if (delTargetId === null) return;
        const currentPw = (delCurrent && delCurrent.value) || '';
        if (!currentPw) {
            showDelFeedback('Your admin password is required to confirm.', 'error');
            return;
        }
        delConfirmBtn.disabled = true;
        showDelFeedback('Deleting user...', 'info');
        const targetName = delTargetName;
        fetch('/api/users/' + encodeURIComponent(delTargetId), {
            method: 'DELETE',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ current_password: currentPw })
        })
            .then(function (r) {
                return r.json().then(function (data) { return { ok: r.ok, status: r.status, data: data }; });
            })
            .then(function (res) {
                if (!res.ok) throw new Error((res.data && res.data.error) || ('Failed to delete user (' + res.status + ').'));
                closeDeletePanel();
                showPageFeedback('User "' + targetName + '" deleted.', 'success');
                loadUsers();
            })
            .catch(function (err) {
                if (delCurrent) delCurrent.value = '';
                showDelFeedback(err.message || 'Failed to delete user.', 'error');
            })
            .finally(function () { delConfirmBtn.disabled = false; });
    }

    function savePassword() {
        if (pwTargetId === null) return;
        const newPw = (pwNew && pwNew.value) || '';
        const confirmPw = (pwConfirm && pwConfirm.value) || '';
        const currentPw = (pwCurrent && pwCurrent.value) || '';
        if (!newPw) {
            showPwFeedback('New password cannot be empty.', 'error');
            return;
        }
        if (newPw !== confirmPw) {
            showPwFeedback('Passwords do not match.', 'error');
            return;
        }
        if (!currentPw) {
            showPwFeedback('Your password is required to confirm this change.', 'error');
            return;
        }
        pwSave.disabled = true;
        showPwFeedback('Updating password...', 'info');
        const targetName = pwTargetName;
        fetch('/api/users/' + encodeURIComponent(pwTargetId) + '/password', {
            method: 'PUT',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ password: newPw, current_password: currentPw })
        })
            .then(function (r) {
                return r.json().then(function (data) { return { ok: r.ok, status: r.status, data: data }; });
            })
            .then(function (res) {
                if (!res.ok) throw new Error((res.data && res.data.error) || ('Failed to update password (' + res.status + ').'));
                showPwFeedback('Password for "' + targetName + '" updated.', 'success');
                if (pwNew) pwNew.value = '';
                if (pwConfirm) pwConfirm.value = '';
                if (pwCurrent) pwCurrent.value = '';
                closePasswordPanel();
            })
            .catch(function (err) {
                if (pwCurrent) pwCurrent.value = '';
                showPwFeedback(err.message || 'Failed to update password.', 'error');
            })
            .finally(function () { pwSave.disabled = false; });
    }

    function submitOnEnter(el, action) {
        if (!el) return;
        el.addEventListener('keydown', function (e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                action();
            }
        });
    }

    if (addBtn) addBtn.addEventListener('click', addUser);
    if (addToggleBtn) addToggleBtn.addEventListener('click', openAddPanel);
    if (addCancelBtn) addCancelBtn.addEventListener('click', closeAddPanel);
    if (pwSave) pwSave.addEventListener('click', savePassword);
    if (pwCancel) pwCancel.addEventListener('click', closePasswordPanel);
    if (delConfirmBtn) delConfirmBtn.addEventListener('click', deleteUser);
    if (delCancelBtn) delCancelBtn.addEventListener('click', closeDeletePanel);
    [nameInput, passInput, passConfirmInput, currentInput].forEach(function (el) {
        submitOnEnter(el, addUser);
    });
    [pwNew, pwConfirm, pwCurrent].forEach(function (el) {
        submitOnEnter(el, savePassword);
    });
    submitOnEnter(delCurrent, deleteUser);
    loadUsers();
})();
