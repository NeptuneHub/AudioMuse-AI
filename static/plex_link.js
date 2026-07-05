window.PlexLink = (function () {
    var pollTimer = null;

    function stop() {
        if (pollTimer) {
            clearInterval(pollTimer);
            pollTimer = null;
        }
    }

    function clientId() {
        var key = 'audiomuse_plex_client_id';
        var id = null;
        try { id = window.localStorage.getItem(key); } catch (e) { id = null; }
        if (!id) {
            if (window.crypto && typeof window.crypto.randomUUID === 'function') {
                id = 'audiomuse-' + window.crypto.randomUUID();
            } else {
                id = 'audiomuse-' + Date.now().toString(16) + '-' + Math.random().toString(16).slice(2);
            }
            try { window.localStorage.setItem(key, id); } catch (e) { id = id; }
        }
        return id;
    }

    function setStatus(el, type, text) {
        el.style.display = 'block';
        el.textContent = text;
        if (type === 'err') {
            el.style.color = '#c0392b';
        } else if (type === 'ok') {
            el.style.color = '#1e7e34';
        } else {
            el.style.color = '';
        }
    }

    function fillToken(opts, token, statusEl) {
        var input = opts.getTokenInput ? opts.getTokenInput() : null;
        if (input) {
            input.value = token;
            input.dataset.originalValue = '';
            input.dispatchEvent(new Event('input', { bubbles: true }));
        }
        setStatus(statusEl, 'ok', 'Linked with Plex. Token filled in.');
        if (typeof opts.onFilled === 'function') {
            opts.onFilled(token);
        }
    }

    function poll(pinId, cid, opts, statusEl) {
        stop();
        var deadline = Date.now() + 180000;
        pollTimer = setInterval(function () {
            if (Date.now() > deadline) {
                stop();
                setStatus(statusEl, 'err', 'Timed out waiting for Plex. Start again to retry.');
                return;
            }
            var url = '/api/setup/plex/pin/' + encodeURIComponent(pinId)
                + '?client_id=' + encodeURIComponent(cid) + '&_=' + Date.now();
            fetch(url, { cache: 'no-store' })
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    if (data && data.token) {
                        stop();
                        fillToken(opts, data.token, statusEl);
                    }
                })
                .catch(function () { return null; });
        }, 1500);
    }

    function renderCode(codeEl, code, opts) {
        codeEl.innerHTML = '';
        codeEl.style.display = 'block';

        var step1 = document.createElement('div');
        step1.appendChild(document.createTextNode('1. Copy this code: '));

        var codeInput = document.createElement('input');
        codeInput.type = 'text';
        codeInput.readOnly = true;
        codeInput.value = code;
        codeInput.style.width = '6rem';
        codeInput.style.fontWeight = '700';
        codeInput.style.letterSpacing = '0.15em';
        codeInput.style.textAlign = 'center';
        codeInput.addEventListener('focus', function () { codeInput.select(); });
        step1.appendChild(codeInput);

        var copyBtn = document.createElement('button');
        copyBtn.type = 'button';
        copyBtn.textContent = 'Copy';
        if (opts && opts.buttonClass) {
            copyBtn.className = opts.buttonClass;
        }
        copyBtn.style.marginLeft = '0.5rem';
        copyBtn.style.width = 'auto';
        copyBtn.addEventListener('click', function () {
            codeInput.focus();
            codeInput.select();
            var copied = false;
            if (window.navigator && navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(code).then(function () {
                    copyBtn.textContent = 'Copied';
                }).catch(function () { return null; });
                copied = true;
            }
            if (!copied) {
                try {
                    document.execCommand('copy');
                    copyBtn.textContent = 'Copied';
                } catch (e) { return null; }
            }
        });
        step1.appendChild(copyBtn);

        var step2 = document.createElement('div');
        step2.style.marginTop = '0.4rem';
        step2.appendChild(document.createTextNode('2. '));
        var openLink = document.createElement('a');
        openLink.href = 'https://plex.tv/link';
        openLink.target = '_blank';
        openLink.rel = 'noopener';
        openLink.textContent = 'Open plex.tv/link';
        step2.appendChild(openLink);
        step2.appendChild(document.createTextNode(', paste the code, and accept.'));

        codeEl.appendChild(step1);
        codeEl.appendChild(step2);
    }

    function begin(opts, statusEl, codeEl) {
        stop();
        var cid = clientId();
        codeEl.style.display = 'none';
        codeEl.innerHTML = '';
        setStatus(statusEl, 'pending', 'Contacting Plex...');
        fetch('/api/setup/plex/pin', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            cache: 'no-store',
            body: JSON.stringify({ client_id: cid })
        }).then(function (r) {
            return r.json().then(function (data) {
                if (!r.ok) { throw new Error(data.error || 'Unable to start Plex linking.'); }
                return data;
            });
        }).then(function (data) {
            renderCode(codeEl, String(data.code), opts);
            setStatus(statusEl, 'pending', 'Waiting for you to enter the code in Plex and accept...');
            poll(String(data.id), cid, opts, statusEl);
        }).catch(function (err) {
            setStatus(statusEl, 'err', err.message || 'Unable to start Plex linking.');
        });
    }

    function attach(mount, opts) {
        stop();
        opts = opts || {};
        mount.innerHTML = '';

        var link = document.createElement('a');
        link.href = '#';
        link.textContent = opts.linkText || 'Or sign in with Plex to fill the token automatically';

        var codeEl = document.createElement('div');
        codeEl.style.display = 'none';
        codeEl.style.marginTop = '0.5rem';

        var statusEl = document.createElement('div');
        statusEl.style.display = 'none';
        statusEl.style.marginTop = '0.4rem';

        link.addEventListener('click', function (ev) {
            ev.preventDefault();
            begin(opts, statusEl, codeEl);
        });

        mount.appendChild(link);
        mount.appendChild(codeEl);
        mount.appendChild(statusEl);
    }

    return { attach: attach, stop: stop };
})();
