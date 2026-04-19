// Additional (non-admin) users management for the Users admin page.
(function () {
    const nameInput = document.getElementById('additional-user-name');
    const passInput = document.getElementById('additional-user-password');
    const addBtn = document.getElementById('additional-user-add');
    const feedback = document.getElementById('additional-user-feedback');
    const tbody = document.getElementById('additional-users-tbody');
    if (!addBtn || !tbody) return;

    function showFeedback(msg, kind) {
        if (!feedback) return;
        feedback.textContent = msg || '';
        feedback.className = 'inline-feedback';
        if (kind) feedback.classList.add('status-' + kind);
        feedback.style.display = msg ? '' : 'none';
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

    function renderUsers(users) {
        tbody.innerHTML = '';
        if (!users || users.length === 0) {
            const tr = document.createElement('tr');
            const td = document.createElement('td');
            td.colSpan = 3;
            td.style.padding = '0.75rem';
            td.style.opacity = '0.7';
            td.textContent = 'No additional users configured.';
            tr.appendChild(td);
            tbody.appendChild(tr);
            return;
        }
        users.forEach(function (u) {
            const tr = document.createElement('tr');
            const tdName = document.createElement('td');
            tdName.style.padding = '0.5rem';
            tdName.textContent = u.username;
            const tdCreated = document.createElement('td');
            tdCreated.style.padding = '0.5rem';
            tdCreated.textContent = formatDate(u.created_at);
            const tdAct = document.createElement('td');
            tdAct.style.padding = '0.5rem';
            tdAct.style.textAlign = 'right';
            const del = document.createElement('button');
            del.type = 'button';
            del.className = 'btn btn-danger';
            del.textContent = 'Delete';
            del.addEventListener('click', function () { deleteUser(u.id, u.username); });
            tdAct.appendChild(del);
            tr.appendChild(tdName);
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
        if (!username || !password) {
            showFeedback('Username and password are required.', 'error');
            return;
        }
        addBtn.disabled = true;
        showFeedback('Creating user...', 'info');
        fetch('/api/users', {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: username, password: password })
        })
            .then(function (r) {
                return r.json().then(function (data) { return { ok: r.ok, status: r.status, data: data }; });
            })
            .then(function (res) {
                if (!res.ok) throw new Error((res.data && res.data.error) || ('Failed to create user (' + res.status + ').'));
                showFeedback('User "' + username + '" created.', 'success');
                nameInput.value = '';
                passInput.value = '';
                loadUsers();
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

    addBtn.addEventListener('click', addUser);
    loadUsers();
})();
