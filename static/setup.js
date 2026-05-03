var serverFields = {
    jellyfin: [
        {name: 'JELLYFIN_URL', label: 'Jellyfin URL', placeholder: 'http://your-jellyfin-server:8096', tooltip: 'Base URL of your Jellyfin server, including http:// or https:// and the port. Must be reachable from the AudioMuse-AI container.'},
        {name: 'JELLYFIN_USER_ID', label: 'Jellyfin user ID', placeholder: 'your-user-id', tooltip: "The Jellyfin user whose library AudioMuse-AI will read. Find the ID in Jellyfin under Dashboard \u2192 Users \u2192 (your user) \u2192 the URL contains userId=..."},
        {name: 'JELLYFIN_TOKEN', label: 'Jellyfin API token', placeholder: 'your-api-token', tooltip: 'API key for that Jellyfin user. Create one in Jellyfin under Dashboard \u2192 API Keys.'}
    ],
    navidrome: [
        {name: 'NAVIDROME_URL', label: 'Navidrome URL', placeholder: 'http://your-navidrome-server:4533', tooltip: 'Base URL of your Navidrome server, including http:// or https:// and the port.'},
        {name: 'NAVIDROME_USER', label: 'Navidrome username', placeholder: 'your-username', tooltip: 'Username of a Navidrome account that can read the music library.'},
        {name: 'NAVIDROME_PASSWORD', label: 'Navidrome password', placeholder: 'your-password', tooltip: 'Password for the Navidrome user above.'}
    ],
    lyrion: [
        {name: 'LYRION_URL', label: 'Lyrion URL', placeholder: 'http://your-lyrion-server:9000', tooltip: 'Base URL of your Lyrion (Logitech Media Server) instance, including http:// and the port.'}
    ],
    emby: [
        {name: 'EMBY_URL', label: 'Emby URL', placeholder: 'http://your-emby-server:8096', tooltip: 'Base URL of your Emby server, including http:// or https:// and the port.'},
        {name: 'EMBY_USER_ID', label: 'Emby user ID', placeholder: 'your-user-id', tooltip: 'The Emby user whose library AudioMuse-AI will read. Find the ID in Emby under Dashboard \u2192 Users \u2192 (your user).'},
        {name: 'EMBY_TOKEN', label: 'Emby API token', placeholder: 'your-api-token', tooltip: 'API key for that Emby user. Create one in Emby under Dashboard \u2192 API Keys.'}
    ]
};

var testFeedback = document.getElementById('test-feedback');
var saveFeedback = document.getElementById('save-feedback');
var saveButton = document.getElementById('save-button');
var serverConfigFields = document.getElementById('server-config-fields');
var advancedFields = document.getElementById('advanced-fields');
var authCredentials = document.getElementById('auth-credentials');
var authAdminExists = document.getElementById('auth-admin-exists');
var apiTokenRow = document.getElementById('api-token-row');
var authCredentialInputs = [
    document.getElementById('AUDIOMUSE_USER'),
    document.getElementById('AUDIOMUSE_PASSWORD'),
    document.getElementById('AUDIOMUSE_PASSWORD_CONFIRM'),
    document.getElementById('JWT_SECRET')
];
var setupForm = document.getElementById('setup-form');
var musicLibrariesSection = document.getElementById('music-libraries-section');
var musicLibrariesList = document.getElementById('music-libraries-list');
var musicLibrariesHint = document.getElementById('music-libraries-hint');
var serverValues = {};
var serverSecretHasValue = {};
var originalValues = {};
var currentSelectedLibraries = [];  // comma-split MUSIC_LIBRARIES from /api/setup
var currentLibraryCheckboxes = [];  // array of HTMLInputElement (checkbox) rendered in the section
var currentNoRestrictionCheckbox = null;  // pseudo-entry: when checked, MUSIC_LIBRARIES = '' (scan all, auto-grow)
// Set from GET /api/setup: when true, an admin already exists in
// audiomuse_users and the setup wizard must not allow editing admin
// credentials here. User management happens in /users instead.
var hasAdminUser = false;

function updateAuthVisibility() {
    var authEnabled = document.getElementById('AUTH_ENABLED').value === 'true';
    var showAdminCreds = authEnabled && !hasAdminUser;
    authCredentials.style.display = authEnabled ? 'grid' : 'none';
    if (authAdminExists) {
        authAdminExists.style.display = (authEnabled && hasAdminUser) ? 'block' : 'none';
    }
    // Hide the three admin-credential wrappers when an admin already exists.
    var adminWrappers = document.querySelectorAll('.auth-admin-credential');
    for (var i = 0; i < adminWrappers.length; i++) {
        adminWrappers[i].style.display = showAdminCreds ? '' : 'none';
    }
    apiTokenRow.style.display = authEnabled ? 'block' : 'none';
    authCredentialInputs.forEach(function(input) {
        if (!input) {
            return;
        }
        // JWT_SECRET stays editable whenever auth is enabled, regardless of
        // whether an admin already exists.
        var isAdminCred = input.id !== 'JWT_SECRET';
        var enabledForInput = isAdminCred ? showAdminCreds : authEnabled;
        input.disabled = !enabledForInput;
        var label = document.querySelector('label[for="' + input.id + '"]');
        if (isAdminCred) {
            input.required = enabledForInput;
            if (label) {
                if (enabledForInput) {
                    label.classList.add('required-label');
                } else {
                    label.classList.remove('required-label');
                }
            }
        }
    });
    var apiTokenInput = document.getElementById('API_TOKEN');
    if (apiTokenInput) {
        apiTokenInput.disabled = !authEnabled;
        apiTokenInput.required = false;
        var label = document.querySelector('label[for="API_TOKEN"]');
        if (label) {
            // Update only the leading text node so the info-tooltip span is preserved.
            var newText = authEnabled ? 'API token (optional) ' : 'API token ';
            if (label.firstChild && label.firstChild.nodeType === Node.TEXT_NODE) {
                label.firstChild.nodeValue = newText;
            } else {
                label.insertBefore(document.createTextNode(newText), label.firstChild);
            }
        }
    }
}

function createInputField(field, value) {
    var row = document.createElement('div');
    row.className = 'field-row';
    var label = document.createElement('label');
    label.setAttribute('for', field.name);
    if (field.tooltip) {
        label.classList.add('label-with-tooltip');
        label.appendChild(document.createTextNode(field.label));
        var tt = document.createElement('span');
        tt.className = 'info-tooltip';
        tt.setAttribute('tabindex', '0');
        var icon = document.createElement('span');
        icon.className = 'info-icon';
        var text = document.createElement('span');
        text.className = 'tooltip-text';
        text.textContent = field.tooltip;
        tt.appendChild(icon);
        tt.appendChild(text);
        label.appendChild(document.createTextNode(' '));
        label.appendChild(tt);
    } else {
        label.textContent = field.label;
    }
    var input;
    if (field.type === 'textarea') {
        input = document.createElement('textarea');
    } else {
        input = document.createElement('input');
    }
    input.id = field.name;
    input.name = field.name;
    if (field.inputType) {
        input.type = field.inputType;
    } else {
        input.type = 'text';
    }
    if (field.placeholder) {
        input.placeholder = field.placeholder;
    }
    if (field.required) {
        label.classList.add('required-label');
        input.required = true;
    }
    var hasSecretValue = false;
    if (field.secret) {
        if (field.has_value) {
            hasSecretValue = true;
        }
    }
    if (field.secret) {
        if (field.name === 'AUDIOMUSE_PASSWORD') {
            input.value = '';
            input.dataset.originalValue = field.originalValue !== undefined ? field.originalValue : '';
        } else if (hasSecretValue) {
            input.value = '********';
            input.dataset.originalValue = field.originalValue !== undefined ? field.originalValue : '********';
        } else {
            if (value) {
                input.value = value;
            } else {
                input.value = '';
            }
            input.dataset.originalValue = field.originalValue !== undefined ? field.originalValue : input.value;
        }
    } else {
        if (value) {
            input.value = value;
        } else {
            input.value = '';
        }
        input.dataset.originalValue = field.originalValue !== undefined ? field.originalValue : input.value;
    }
    if (field.type === 'boolean') {
        input.type = 'text';
        input.placeholder = 'true or false';
    }
    if (field.secret) {
        input.type = 'password';
    }
    row.appendChild(label);
    row.appendChild(input);
    if (field.description) {
        var hint = document.createElement('small');
        hint.textContent = field.description;
        row.appendChild(hint);
    }
    return row;
}

function renderServerFields(serverType, values, hasValueMap) {
    hasValueMap = hasValueMap || {};
    serverConfigFields.innerHTML = '';
    if (!serverFields[serverType]) {
        updateTestButtonState();
        return;
    }
    var fields = serverFields[serverType];
    fields.forEach(function(field) {
        var value = '';
        if (values[field.name]) {
            value = values[field.name];
        }
        var secret = false;
        var secretKeys = ['NAVIDROME_PASSWORD', 'AUDIOMUSE_PASSWORD', 'API_TOKEN', 'JELLYFIN_TOKEN', 'EMBY_TOKEN'];
        for (var i = 0; i < secretKeys.length; i++) {
            if (secretKeys[i] === field.name) {
                secret = true;
                break;
            }
        }
        if (field.name.indexOf('_API_KEY') !== -1) {
            secret = true;
        }
        var hasValue = false;
        if (hasValueMap) {
            if (hasValueMap[field.name]) {
                hasValue = true;
            }
        }
        var fieldCopy = {
            name: field.name,
            label: field.label,
            placeholder: field.placeholder,
            required: true,
            secret: secret,
            has_value: hasValue,
            tooltip: field.tooltip,
            originalValue: originalValues[field.name] !== undefined ? originalValues[field.name] : value
        };
        serverConfigFields.appendChild(createInputField(fieldCopy, value));
    });
    updateTestButtonState();
}

function renderAdvancedFields(fields) {
    advancedFields.innerHTML = '';
    if (!fields) {
        return;
    }
    fields.forEach(function(field) {
        var secret = false;
        if (field.secret) {
            secret = true;
        }
        if (field.name.indexOf('_API_KEY') !== -1) {
            secret = true;
        }
        var fieldConfig = {
            name: field.name,
            label: field.name,
            placeholder: field.default ? field.default : '',
            type: field.type === 'bool' ? 'boolean' : field.type,
            inputType: field.type === 'boolean' ? 'text' : 'text',
            secret: secret,
            has_value: field.has_value,
            originalValue: originalValues[field.name] !== undefined ? originalValues[field.name] : (field.value || '')
        };
        advancedFields.appendChild(createInputField(fieldConfig, field.value));
    });
}

function loadSetupData() {
    fetch('/api/setup').then(function(response) {
        if (!response.ok) {
            throw new Error('Failed to load setup data');
        }
        return response.json();
    }).then(function(data) {
        hasAdminUser = !!data.has_admin_user;
        var basicData = {};
        var secretHasValue = {};
        data.basic_fields.forEach(function(item) {
            basicData[item.name] = item.value;
            if (item.secret) {
                secretHasValue[item.name] = item.has_value;
            }
        });
        serverSecretHasValue = secretHasValue;
        var advancedData = data.advanced_fields;
        var mediaServerSelect = document.getElementById('MEDIASERVER_TYPE');
        if (basicData.MEDIASERVER_TYPE) {
            mediaServerSelect.value = basicData.MEDIASERVER_TYPE;
        } else {
            mediaServerSelect.value = 'jellyfin';
        }
        var authEnabledSelect = document.getElementById('AUTH_ENABLED');
        if (basicData.AUTH_ENABLED) {
            authEnabledSelect.value = String(basicData.AUTH_ENABLED).toLowerCase();
        } else {
            authEnabledSelect.value = 'true';
        }
        var usernameInput = document.getElementById('AUDIOMUSE_USER');
        if (basicData.AUDIOMUSE_USER) {
            usernameInput.value = basicData.AUDIOMUSE_USER;
        } else {
            usernameInput.value = '';
        }
        var passwordInput = document.getElementById('AUDIOMUSE_PASSWORD');
        var confirmInput = document.getElementById('AUDIOMUSE_PASSWORD_CONFIRM');
        var tokenInput = document.getElementById('API_TOKEN');
        if (passwordInput && secretHasValue.AUDIOMUSE_PASSWORD) {
            passwordInput.value = '********';
            passwordInput.dataset.originalValue = '********';
            passwordInput.placeholder = '********';
        } else if (passwordInput) {
            passwordInput.value = '';
            passwordInput.dataset.originalValue = '';
        }
        if (confirmInput && secretHasValue.AUDIOMUSE_PASSWORD) {
            confirmInput.value = '********';
            confirmInput.dataset.originalValue = '********';
            confirmInput.placeholder = '********';
        } else if (confirmInput) {
            confirmInput.value = '';
            confirmInput.dataset.originalValue = '';
        }
        if (tokenInput) {
            if (secretHasValue.API_TOKEN) {
                tokenInput.value = '********';
                tokenInput.dataset.originalValue = '********';
            } else {
                tokenInput.value = basicData.API_TOKEN || '';
                tokenInput.dataset.originalValue = tokenInput.value;
            }
        }
        var jwtInput = document.getElementById('JWT_SECRET');
        if (jwtInput) {
            if (secretHasValue.JWT_SECRET) {
                jwtInput.value = '********';
                jwtInput.dataset.originalValue = '********';
            } else {
                if (basicData.JWT_SECRET) {
                    jwtInput.value = basicData.JWT_SECRET;
                } else {
                    jwtInput.value = '';
                }
                jwtInput.dataset.originalValue = jwtInput.value;
            }
        }
        var visibleAdvancedData = Array.isArray(advancedData)
            ? advancedData.filter(function(f) { return f && f.name !== 'MUSIC_LIBRARIES'; })
            : advancedData;
        currentSelectedLibraries = splitLibraryList(data.music_libraries);
        originalValues = {};
        data.basic_fields.forEach(function(item) {
            originalValues[item.name] = item.value || '';
            if (item.secret && item.has_value && !item.value) {
                originalValues[item.name] = '********';
            }
        });
        data.advanced_fields.forEach(function(item) {
            originalValues[item.name] = item.value || '';
            if (item.secret && item.has_value && !item.value) {
                originalValues[item.name] = '********';
            }
        });
        serverValues = basicData; // keep the full current server-related values
        renderServerFields(mediaServerSelect.value, basicData, secretHasValue);
        renderAdvancedFields(visibleAdvancedData);
        populateLyricsApiFields(data.lyrics_api_fields);
        updateAuthVisibility();
        // If the provider is already configured (server returned `has_value`
        // for the credential fields), auto-fetch the library list so the
        // checkbox state matches the saved MUSIC_LIBRARIES value.
        if (providerCredsHaveSavedValues(mediaServerSelect.value, secretHasValue, basicData)) {
            fetchProviderLibraries(mediaServerSelect.value);
        }
    }).catch(function(err) {
        saveFeedback.className = 'status-failure inline-feedback';
        saveFeedback.style.display = 'block';
        saveFeedback.textContent = 'Unable to load setup data. Refresh the page or check the server logs.';
    });
}

function saveCurrentServerValues() {
    var currentServerType = document.getElementById('MEDIASERVER_TYPE').value;
    var keys = ['JELLYFIN_URL', 'JELLYFIN_USER_ID', 'JELLYFIN_TOKEN', 'NAVIDROME_URL', 'NAVIDROME_USER', 'NAVIDROME_PASSWORD', 'LYRION_URL', 'EMBY_URL', 'EMBY_USER_ID', 'EMBY_TOKEN'];
    keys.forEach(function(key) {
        var input = document.getElementById(key);
        if (input) {
            serverValues[key] = input.value;
        }
    });
}

function testConfigFieldsFilled() {
    var requiredFields = serverConfigFields.querySelectorAll('input[required], textarea[required], select[required]');
    if (!requiredFields.length) {
        return false;
    }
    return Array.prototype.every.call(requiredFields, function(input) {
        if (input.disabled) {
            return true;
        }
        return input.value.trim() !== '';
    });
}

function updateTestButtonState() {
    var testButton = document.getElementById('test-button');
    testButton.disabled = !testConfigFieldsFilled();
}

function updateServerFields() {
    saveCurrentServerValues();
    var serverType = document.getElementById('MEDIASERVER_TYPE').value;
    renderServerFields(serverType, serverValues, serverSecretHasValue);
    // Hide the checkbox list (it only matches the prior provider's library
    // names) but keep ``currentSelectedLibraries`` intact: it reflects the
    // *saved* MUSIC_LIBRARIES value, which is provider-agnostic in storage.
    // If the user flips back to the original provider, the next render will
    // re-check the matching names. The renderer's case-insensitive name
    // match means stale names against a new provider's libraries simply
    // miss and leave their boxes unchecked — no leakage into the save.
    hideMusicLibrariesSection();
}

function splitLibraryList(value) {
    if (!value) {
        return [];
    }
    return String(value).split(',').map(function(s) { return s.trim(); }).filter(Boolean);
}

function providerCredsHaveSavedValues(serverType, secretHasValue, basicData) {
    var fields = serverFields[serverType];
    if (!fields) return false;
    for (var i = 0; i < fields.length; i++) {
        var name = fields[i].name;
        // For secret fields the server returns has_value=true when a value is
        // stored; for non-secret it just returns the actual string.
        if (secretHasValue && secretHasValue[name]) continue;
        if (basicData && basicData[name]) continue;
        return false;
    }
    return true;
}

function hideMusicLibrariesSection() {
    if (!musicLibrariesSection) return;
    musicLibrariesSection.style.display = 'none';
    musicLibrariesList.innerHTML = '';
    currentLibraryCheckboxes = [];
    currentNoRestrictionCheckbox = null;
    if (musicLibrariesHint) musicLibrariesHint.style.display = 'none';
}

function fetchProviderLibraries(serverType, configOverride) {
    if (!musicLibrariesSection) return;
    if (!serverFields[serverType]) {
        hideMusicLibrariesSection();
        return;
    }
    var configPayload = configOverride || collectConfigFromForm(true);
    // MEDIASERVER_TYPE may be dropped by collectConfigFromForm if unchanged.
    configPayload.MEDIASERVER_TYPE = serverType;
    fetch('/api/setup/providers/libraries', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: configPayload })
    }).then(function(resp) {
        return resp.json().then(function(data) {
            if (!resp.ok) {
                throw new Error(data.error || 'Unable to list libraries.');
            }
            return data;
        });
    }).then(function(data) {
        if (data.unsupported || !Array.isArray(data.libraries) || data.libraries.length === 0) {
            hideMusicLibrariesSection();
            return;
        }
        // Preserve the user's in-flight checkbox toggles across re-renders
        // (e.g. after clicking Test Connection again). Otherwise we would
        // reset to currentSelectedLibraries, which only reflects the value
        // last loaded from /api/setup.
        var selectedForRender = currentSelectedLibraries;
        var forceNoRestriction = false;
        // Track whether this is a re-render (state from a previous render
        // exists) vs the first render after page load. The "honor live UI
        // state" fallback below must only run on re-renders, otherwise an
        // empty DB selection would be treated as "user just unchecked
        // everything" and we'd uncheck the default No-restriction box.
        var isRerender = (currentLibraryCheckboxes.length > 0 || !!currentNoRestrictionCheckbox);
        if (isRerender) {
            // "No restriction" wins: render as the unrestricted state (empty list).
            if (currentNoRestrictionCheckbox && currentNoRestrictionCheckbox.checked) {
                selectedForRender = [];
                forceNoRestriction = true;
            } else {
                var checkedNames = [];
                for (var k = 0; k < currentLibraryCheckboxes.length; k++) {
                    if (currentLibraryCheckboxes[k].checked) {
                        checkedNames.push(currentLibraryCheckboxes[k].dataset.libraryName);
                    }
                }
                selectedForRender = checkedNames;
            }
        }
        renderLibraryCheckboxes(data.libraries, selectedForRender);
        // Re-render only: if the user explicitly turned No-restriction off
        // and unchecked all rows, renderLibraryCheckboxes would have
        // defaulted back to "no restriction" (stale-selection fallback).
        // Honor the live UI state instead. On the first render we want the
        // default behavior (empty saved selection → No-restriction checked).
        if (isRerender && currentNoRestrictionCheckbox && !forceNoRestriction
                && Array.isArray(selectedForRender) && selectedForRender.length === 0) {
            currentNoRestrictionCheckbox.checked = false;
            applyNoRestrictionState();
            updateMusicLibrariesHint();
        }
    }).catch(function() {
        // Don't block the user on list failures — the free-text value still
        // works on save (empty string = scan everything).
        hideMusicLibrariesSection();
    });
}

function renderLibraryCheckboxes(libraries, selectedNames) {
    if (!musicLibrariesList) return;
    musicLibrariesList.innerHTML = '';
    currentLibraryCheckboxes = [];
    currentNoRestrictionCheckbox = null;

    // Map saved names to lowercase for case-insensitive lookup.
    var selectedLower = {};
    var rawHasSelection = Array.isArray(selectedNames) && selectedNames.length > 0;
    if (rawHasSelection) {
        for (var i = 0; i < selectedNames.length; i++) {
            selectedLower[String(selectedNames[i]).toLowerCase()] = true;
        }
    }
    // "No restriction" = empty saved selection. Backend reads MUSIC_LIBRARIES=''
    // as "scan everything" across every media-server adapter, and new libraries
    // added later are picked up automatically.
    var noRestriction = !rawHasSelection;
    // If a saved selection exists but has no overlap with this provider's
    // libraries (e.g. names were saved for a different provider), treat it as
    // stale and fall back to "no restriction" rather than rendering all
    // unchecked which would look broken.
    if (rawHasSelection) {
        var anyMatch = false;
        for (var j = 0; j < libraries.length; j++) {
            var libName = libraries[j] && libraries[j].name ? String(libraries[j].name).toLowerCase() : '';
            if (libName && selectedLower[libName]) { anyMatch = true; break; }
        }
        if (!anyMatch) noRestriction = true;
    }

    // --- "No restriction" pseudo-row at the top ---
    var noRow = document.createElement('label');
    noRow.style.display = 'flex';
    noRow.style.alignItems = 'center';
    noRow.style.gap = '0.5rem';
    noRow.style.fontWeight = '500';
    var noCb = document.createElement('input');
    noCb.type = 'checkbox';
    noCb.checked = noRestriction;
    noCb.style.width = 'auto';
    noCb.style.flex = '0 0 auto';
    noCb.style.margin = '0';
    noCb.addEventListener('change', function() {
        applyNoRestrictionState();
        updateMusicLibrariesHint();
    });
    noRow.appendChild(noCb);
    noRow.appendChild(document.createTextNode('No restriction (scan all libraries, including ones added later)'));
    musicLibrariesList.appendChild(noRow);
    currentNoRestrictionCheckbox = noCb;

    libraries.forEach(function(lib) {
        var name = lib && lib.name ? String(lib.name) : '';
        if (!name) return;
        var row = document.createElement('label');
        row.style.display = 'flex';
        row.style.alignItems = 'center';
        row.style.gap = '0.5rem';
        row.style.fontWeight = '400';
        var cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.dataset.libraryName = name;
        // When "No restriction" is on, individual rows are visually checked
        // but disabled (the user shouldn't be picking from a list that is
        // already overridden). When off, honor the saved selection.
        cb.checked = noRestriction ? false : !!selectedLower[name.toLowerCase()];
        // Override `.field-row input { width: 100% }` from setup.html — that
        // global rule would stretch each checkbox across the row and push
        // the label text to the far right.
        cb.style.width = 'auto';
        cb.style.flex = '0 0 auto';
        cb.style.margin = '0';
        cb.addEventListener('change', updateMusicLibrariesHint);
        row.appendChild(cb);
        row.appendChild(document.createTextNode(name));
        row.dataset.libraryRow = '1';
        musicLibrariesList.appendChild(row);
        currentLibraryCheckboxes.push(cb);
    });
    applyNoRestrictionState();
    musicLibrariesSection.style.display = 'flex';
    updateMusicLibrariesHint();
}

function applyNoRestrictionState() {
    // Disable per-library checkboxes (and dim their rows) whenever the
    // "No restriction" pseudo-entry is checked, so the UI matches the
    // semantics: empty MUSIC_LIBRARIES means "scan everything".
    if (!currentNoRestrictionCheckbox) return;
    var disabled = !!currentNoRestrictionCheckbox.checked;
    for (var i = 0; i < currentLibraryCheckboxes.length; i++) {
        var cb = currentLibraryCheckboxes[i];
        cb.disabled = disabled;
        if (disabled) cb.checked = false;
        if (cb.parentElement) cb.parentElement.style.opacity = disabled ? '0.5' : '1';
    }
}

function updateMusicLibrariesHint() {
    if (!musicLibrariesHint) return;
    // Hint = "you'll scan nothing": only relevant when No-restriction is OFF
    // and the user has unchecked every library row.
    var noRestriction = currentNoRestrictionCheckbox && currentNoRestrictionCheckbox.checked;
    var anyChecked = currentLibraryCheckboxes.some(function(cb) { return cb.checked; });
    musicLibrariesHint.style.display = (!noRestriction && currentLibraryCheckboxes.length > 0 && !anyChecked)
        ? 'block' : 'none';
}

function collectMusicLibrariesValue() {
    // Returns the MUSIC_LIBRARIES value to store, or null to skip writing.
    if (!currentLibraryCheckboxes.length && !currentNoRestrictionCheckbox) {
        // Section isn't rendered (MPD or provider doesn't support it, or the
        // fetch failed). Don't touch MUSIC_LIBRARIES.
        return null;
    }
    // "No restriction" → empty (= scan everything, every adapter treats this
    // as "no filter" and will include libraries added in the future).
    if (currentNoRestrictionCheckbox && currentNoRestrictionCheckbox.checked) {
        return '';
    }
    var checked = currentLibraryCheckboxes.filter(function(cb) { return cb.checked; });
    // None checked while No-restriction is OFF → still empty (scan all).
    // "Scan nothing" is a footgun and the hint already warns the user; we
    // refuse to persist that state.
    if (checked.length === 0) {
        return '';
    }
    // MUSIC_LIBRARIES is stored as a comma-separated string, so a comma in a
    // library name would corrupt the round-trip. Skip writing rather than
    // sending a poisoned value; the hint makes the case visible to the user.
    var names = checked.map(function(cb) { return cb.dataset.libraryName; });
    if (names.some(function(n) { return n.indexOf(',') !== -1; })) {
        return null;
    }
    return names.join(',');
}

function collectConfigFromForm(testMode) {
    var formData = new FormData(setupForm);
    var config = {};
    formData.forEach(function(value, key) {
        var input = document.getElementById(key);
        if (!input) {
            return;
        }
        var original = input.dataset.originalValue;
        if (!testMode) {
            if (original !== undefined && value === original) {
                return;
            }
            if (value === '' && original === undefined) {
                return;
            }
        } else {
            if (input.type === 'password' && original === '********' && value === '********') {
                return;
            }
        }
        config[key] = value;
    });
    return config;
}

function testConnection() {
    var testButton = document.getElementById('test-button');
    var passwordInput = document.getElementById('AUDIOMUSE_PASSWORD');
    var confirmInput = document.getElementById('AUDIOMUSE_PASSWORD_CONFIRM');
    var passwordValue = '';
    if (passwordInput) {
        passwordValue = passwordInput.value;
    }
    var confirmValue = '';
    if (confirmInput) {
        confirmValue = confirmInput.value;
    }
    var passwordUnchanged = (passwordValue === '********');
    if (passwordUnchanged && !confirmValue) {
        passwordUnchanged = true;
    } else {
        passwordUnchanged = false;
    }
    if (!passwordUnchanged && (passwordValue || confirmValue)) {
        if (passwordValue !== confirmValue) {
            testFeedback.className = 'status-failure inline-feedback';
            testFeedback.style.display = 'block';
            testFeedback.textContent = 'Password and confirmation do not match.';
            return;
        }
    }
    testButton.disabled = true;
    saveButton.disabled = true;
    testFeedback.className = 'status-pending inline-feedback';
    testFeedback.style.display = 'block';
    testFeedback.textContent = 'Testing connection...';
    var config = collectConfigFromForm(true);
    fetch('/api/setup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: config, test_connection: true })
    }).then(function(resp) {
        return resp.json().then(function(data) {
            if (!resp.ok) {
                throw new Error(data.error || 'Unable to test connection.');
            }
            return data;
        });
    }).then(function(data) {
        testFeedback.className = 'status-success inline-feedback';
        testFeedback.style.display = 'block';
        var serverName = data.media_server ? data.media_server.charAt(0).toUpperCase() + data.media_server.slice(1) : 'media server';
        var count = (typeof data.probe_count === 'number') ? data.probe_count : 0;
        if (data.probe_limit_hit) {
            testFeedback.textContent = '✓ Connected to ' + serverName + '. At least ' + count + ' recent top-played items were returned.';
        } else if (count === 1) {
            testFeedback.textContent = '✓ Connected to ' + serverName + '. 1 top-played item was returned.';
        } else {
            testFeedback.textContent = '✓ Connected to ' + serverName + '. ' + count + ' top-played items were returned.';
        }
        // Populate the library checkbox list using the same config payload
        // (so secret placeholders fall back to saved values server-side).
        var serverType = document.getElementById('MEDIASERVER_TYPE').value;
        fetchProviderLibraries(serverType, config);
    }).catch(function(err) {
        testFeedback.className = 'status-failure inline-feedback';
        testFeedback.style.display = 'block';
        testFeedback.textContent = '✕ Connection test failed: ' + err.message;
    }).finally(function() {
        testButton.disabled = false;
        saveButton.disabled = false;
    });
}

setupForm.addEventListener('submit', function(event) {
    event.preventDefault();
    saveButton.disabled = true;
    saveFeedback.style.display = 'none';
    var passwordInput = document.getElementById('AUDIOMUSE_PASSWORD');
    var confirmInput = document.getElementById('AUDIOMUSE_PASSWORD_CONFIRM');
    var passwordValue = '';
    if (passwordInput) {
        passwordValue = passwordInput.value;
    }
    var confirmValue = '';
    if (confirmInput) {
        confirmValue = confirmInput.value;
    }
    var passwordUnchanged = (passwordValue === '********');
    if (passwordUnchanged && !confirmValue) {
        passwordUnchanged = true;
    } else {
        passwordUnchanged = false;
    }
    if (!passwordUnchanged && (passwordValue || confirmValue)) {
        if (passwordValue !== confirmValue) {
            saveFeedback.className = 'status-failure inline-feedback';
            saveFeedback.style.display = 'block';
            saveFeedback.textContent = 'Password and confirmation do not match.';
            saveButton.disabled = false;
            return;
        }
    }
    var config = collectConfigFromForm();
    var mlValue = collectMusicLibrariesValue();
    if (mlValue !== null) {
        config.MUSIC_LIBRARIES = mlValue;
    }
    fetch('/api/setup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: config })
    }).then(function(resp) {
        return resp.json().then(function(data) {
            if (!resp.ok) {
                throw new Error(data.error || 'Unable to save configuration.');
            }
            return data;
        });
    }).then(function(data) {
        saveFeedback.className = 'status-success inline-feedback';
        saveFeedback.style.display = 'block';
        var countdown = 20;
        saveFeedback.textContent = 'Configuration saved. Redirecting in ' + countdown + ' seconds...';
        var countdownInterval = setInterval(function() {
            countdown -= 1;
            if (countdown > 0) {
                saveFeedback.textContent = 'Configuration saved. Redirecting in ' + countdown + ' seconds...';
            } else {
                clearInterval(countdownInterval);
                window.location.href = '/';
            }
        }, 1000);
    }).catch(function(err) {
        saveFeedback.className = 'status-failure inline-feedback';
        saveFeedback.style.display = 'block';
        var message = err.message || 'Unable to save configuration.';
        if (message === 'Forbidden' || message === 'Setup required' || message === 'Auth not configured') {
            message = 'Error saving configuration. Please refresh the page and try again.';
        } else if (!message.toLowerCase().includes('refresh')) {
            message = message + ' Please refresh the page or check the server logs.';
        }
        saveFeedback.textContent = '✕ ' + message;
    }).finally(function() {
        saveButton.disabled = false;
    });
});

document.getElementById('test-button').addEventListener('click', testConnection);
serverConfigFields.addEventListener('input', updateTestButtonState);
document.getElementById('MEDIASERVER_TYPE').addEventListener('change', updateServerFields);
document.getElementById('AUTH_ENABLED').addEventListener('change', updateAuthVisibility);

// ---------------------------------------------------------------------------
// Lyrics API section — interactive analyze & configure
// ---------------------------------------------------------------------------
var lyricsApiState = {};
[1, 2].forEach(function(s) {
    lyricsApiState[s] = {exampleUrl: '', params: {}, paramRoles: {}, pathSegments: [], pathRoles: {}, jsonObj: null, selectedField: null};
});

function populateLyricsApiFields(lyricsApiData) {
    if (!lyricsApiData) return;
    [1, 2].forEach(function(slot) {
        var pre = 'LYRICS_API_' + slot + '_';
        var get = function(k) { return lyricsApiData[pre + k] || {}; };
        var urlTemplate = get('URL_TEMPLATE').value || '';
        var artistParam = get('ARTIST_PARAM').value || '';
        var titleParam  = get('TITLE_PARAM').value  || '';
        var lyricsField = get('LYRICS_FIELD').value || '';
        var apikeyParam = get('APIKEY_PARAM').value || '';
        var apikeyHasVal = get('APIKEY_VALUE').has_value || false;
        var timeout     = get('TIMEOUT').value || '5';
        function setHidden(name, val) {
            var el = document.getElementById(name);
            if (el) { el.value = val; el.dataset.originalValue = val; }
        }
        setHidden(pre + 'URL_TEMPLATE', urlTemplate);
        setHidden(pre + 'ARTIST_PARAM', artistParam);
        setHidden(pre + 'TITLE_PARAM',  titleParam);
        setHidden(pre + 'LYRICS_FIELD', lyricsField);
        setHidden(pre + 'APIKEY_PARAM', apikeyParam);
        var akEl = document.getElementById(pre + 'APIKEY_VALUE');
        if (akEl) { akEl.value = apikeyHasVal ? '********' : ''; akEl.dataset.originalValue = akEl.value; }
        var toEl = document.getElementById(pre + 'TIMEOUT');
        if (toEl) { toEl.value = timeout; toEl.dataset.originalValue = timeout; }
        var isPathBased = urlTemplate.indexOf('{artist}') !== -1 && urlTemplate.indexOf('{title}') !== -1;
        var configComplete = urlTemplate && lyricsField && (isPathBased || (artistParam && titleParam));
        if (configComplete) {
            showLyricsApiSlotSummary(slot, urlTemplate, artistParam, titleParam, lyricsField, apikeyParam, apikeyHasVal);
            var inputRow = document.getElementById('lyrics-api-' + slot + '-input-row');
            if (inputRow) inputRow.style.display = 'none';
            var toRow = document.getElementById('lyrics-api-' + slot + '-timeout-row');
            if (toRow) toRow.style.display = 'flex';
        }
    });
}

function showLyricsApiStatus(slot, type, msg) {
    var row = document.getElementById('lyrics-api-' + slot + '-analyze-status');
    var el  = document.getElementById('lyrics-api-' + slot + '-status-msg');
    if (!row || !el) return;
    row.style.display = 'block';
    el.className = 'inline-feedback status-' + type;
    el.textContent = msg;
}

function analyzeLyricsApiSlot(slot) {
    var urlEl = document.getElementById('lyrics-api-' + slot + '-example-url');
    var url = urlEl ? urlEl.value.trim() : '';
    if (!url) { showLyricsApiStatus(slot, 'failure', 'Please enter an example URL.'); return; }
    var btn = document.getElementById('lyrics-api-' + slot + '-analyze-btn');
    if (btn) btn.disabled = true;
    showLyricsApiStatus(slot, 'pending', 'Calling the API\u2026');
    fetch('/api/setup/lyrics-api/analyze', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({example_url: url})
    }).then(function(r) { return r.json(); }).then(function(data) {
        if (btn) btn.disabled = false;
        if (data.error && !data.json_obj && !data.params) {
            showLyricsApiStatus(slot, 'failure', '\u2715 ' + data.error);
            return;
        }
        showLyricsApiStatus(slot, data.error ? 'pending' : 'success',
            data.error ? ('\u26a0 HTTP error: ' + data.error) : '\u2713 API responded \u2014 select the lyrics field below.');
        var state = lyricsApiState[slot];
        state.exampleUrl    = url;
        state.params        = data.params        || {};
        state.pathSegments  = data.path_segments || [];
        state.pathRoles     = {};
        state.jsonObj       = (data.json_obj !== undefined) ? data.json_obj : null;
        state.selectedField = (data.guesses && data.guesses.lyrics_field) || null;
        state.paramRoles    = {};
        var g = data.guesses || {};
        Object.keys(state.params).forEach(function(pname) {
            if      (pname === g.artist_param) state.paramRoles[pname] = 'artist';
            else if (pname === g.title_param)  state.paramRoles[pname] = 'title';
            else if (pname === g.apikey_param) state.paramRoles[pname] = 'apikey';
            else                               state.paramRoles[pname] = 'none';
        });
        // Auto-suggest timeout: actual response time + 20%, minimum 2 extra seconds
        if (data.elapsed_ms != null) {
            var elapsed = data.elapsed_ms / 1000;
            var suggested = Math.max(elapsed * 1.2, elapsed + 2);
            suggested = Math.round(suggested * 2) / 2; // round to nearest 0.5s
            state.suggestedTimeout = Math.max(suggested, 2);
        } else {
            state.suggestedTimeout = 5;
        }
        renderLyricsApiAnalysis(slot);
    }).catch(function(err) {
        if (btn) btn.disabled = false;
        showLyricsApiStatus(slot, 'failure', '\u2715 ' + (err.message || err));
    });
}

function renderLyricsApiAnalysis(slot) {
    var area = document.getElementById('lyrics-api-' + slot + '-analyze-area');
    if (!area) return;
    area.innerHTML = '';
    area.style.display = 'block';
    var state = lyricsApiState[slot];
    var paramNames = Object.keys(state.params);
    if (paramNames.length > 0) {
        var sec = document.createElement('div');
        sec.style.marginBottom = '1.25rem';
        var hdr = document.createElement('p');
        hdr.style.cssText = 'font-weight:600; margin-bottom:0.5rem;';
        hdr.textContent = 'URL parameters \u2014 assign the role of each:';
        sec.appendChild(hdr);
        var grid = document.createElement('div');
        grid.style.cssText = 'display:grid; grid-template-columns:auto 1fr auto; gap:0.4rem 0.9rem; align-items:center;';
        paramNames.forEach(function(pname) {
            var nameEl = document.createElement('code');
            nameEl.textContent = pname;
            nameEl.style.cssText = 'background:var(--bg-card,#f0f0f0); padding:0.15rem 0.45rem; border-radius:4px; white-space:nowrap;';
            var val = String(state.params[pname] || '');
            var valEl = document.createElement('span');
            valEl.textContent = val.length > 45 ? val.substring(0, 45) + '\u2026' : val;
            valEl.style.cssText = 'color:var(--text-muted,#666); font-size:0.9rem; min-width:0; overflow:hidden;';
            var sel = document.createElement('select');
            sel.style.cssText = 'width:auto; margin-bottom:0; padding:0.35rem 0.55rem; font-size:0.88rem;';
            [['none','\u2014 ignore \u2014'],['artist','Artist'],['title','Title'],['apikey','API key']].forEach(function(r) {
                var opt = document.createElement('option');
                opt.value = r[0]; opt.textContent = r[1]; sel.appendChild(opt);
            });
            sel.value = state.paramRoles[pname] || 'none';
            (function(pn) {
                sel.addEventListener('change', function() {
                    state.paramRoles[pn] = sel.value;
                    updateLyricsApiHiddenInputs(slot);
                });
            })(pname);
            grid.appendChild(nameEl);
            grid.appendChild(valEl);
            grid.appendChild(sel);
        });
        sec.appendChild(grid);
        area.appendChild(sec);
    }
    var pathSegs = state.pathSegments || [];
    if (pathSegs.length > 0) {
        var pathSec = document.createElement('div');
        pathSec.style.marginBottom = '1.25rem';
        var pathHdr = document.createElement('p');
        pathHdr.style.cssText = 'font-weight:600; margin-bottom:0.5rem;';
        pathHdr.textContent = 'URL path values \u2014 assign the role of each:';
        pathSec.appendChild(pathHdr);
        var pathGrid = document.createElement('div');
        pathGrid.style.cssText = 'display:grid; grid-template-columns:auto 1fr auto; gap:0.4rem 0.9rem; align-items:center;';
        pathSegs.forEach(function(seg) {
            var nameEl = document.createElement('code');
            nameEl.textContent = '/' + seg.value;
            nameEl.style.cssText = 'background:var(--bg-card,#f0f0f0); padding:0.15rem 0.45rem; border-radius:4px; white-space:nowrap;';
            var emptyEl = document.createElement('span');
            var sel = document.createElement('select');
            sel.style.cssText = 'width:auto; margin-bottom:0; padding:0.35rem 0.55rem; font-size:0.88rem;';
            [['none','\u2014 ignore \u2014'],['artist','Artist'],['title','Title']].forEach(function(r) {
                var opt = document.createElement('option');
                opt.value = r[0]; opt.textContent = r[1]; sel.appendChild(opt);
            });
            sel.value = (state.pathRoles || {})[seg.index] || 'none';
            (function(idx) {
                sel.addEventListener('change', function() {
                    state.pathRoles[idx] = sel.value;
                    updateLyricsApiHiddenInputs(slot);
                });
            })(seg.index);
            pathGrid.appendChild(nameEl);
            pathGrid.appendChild(emptyEl);
            pathGrid.appendChild(sel);
        });
        pathSec.appendChild(pathGrid);
        area.appendChild(pathSec);
    }
    if (state.jsonObj !== null && typeof state.jsonObj === 'object') {
        var treeHdr = document.createElement('p');
        treeHdr.style.cssText = 'font-weight:600; margin-bottom:0.5rem;';
        treeHdr.textContent = 'API response \u2014 click Select next to the field that contains the lyrics:';
        area.appendChild(treeHdr);
        var treeWrap = document.createElement('div');
        treeWrap.id = 'lyrics-api-' + slot + '-tree';
        treeWrap.style.cssText = 'font-family:monospace; font-size:0.87rem; background:var(--bg-card,#f8f9fb); border:1px solid var(--border-color,#ccc); border-radius:8px; padding:1rem; max-height:360px; overflow-y:auto;';
        treeWrap.appendChild(buildJsonTreeEl(state.jsonObj, '', slot, 0));
        area.appendChild(treeWrap);
    } else if (state.jsonObj === null) {
        var noJson = document.createElement('p');
        noJson.style.cssText = 'color:var(--text-muted,#666); font-size:0.9rem; margin-top:0.5rem;';
        noJson.textContent = 'The API did not return valid JSON. Check the URL and try again.';
        area.appendChild(noJson);
    }
    var selRow = document.createElement('div');
    selRow.id = 'lyrics-api-' + slot + '-selected-display';
    selRow.style.cssText = 'margin-top:0.75rem; font-weight:500; font-size:0.95rem;';
    if (state.selectedField) {
        selRow.textContent = 'Lyrics field: ' + state.selectedField;
        selRow.style.color = 'var(--status-success-text,#1f4f1f)';
    } else {
        selRow.textContent = 'No lyrics field selected yet \u2014 click Select on a field above.';
        selRow.style.color = 'var(--text-muted,#666)';
    }
    area.appendChild(selRow);
    updateLyricsApiHiddenInputs(slot);
    highlightSelectedField(slot);
    // Show timeout field and apply suggested value
    var toRow = document.getElementById('lyrics-api-' + slot + '-timeout-row');
    if (toRow) toRow.style.display = 'flex';
    var toInput = document.getElementById('LYRICS_API_' + slot + '_TIMEOUT');
    if (toInput && state.suggestedTimeout) {
        toInput.value = state.suggestedTimeout;
    }
}

function buildJsonTreeEl(obj, path, slot, depth) {
    var el = document.createElement('div');
    el.style.marginLeft = depth > 0 ? '1.2rem' : '0';
    if (typeof obj !== 'object' || obj === null) { el.textContent = String(obj); return el; }
    if (Array.isArray(obj)) {
        var note = document.createElement('span');
        note.style.color = '#888';
        note.textContent = '[' + obj.length + ' items]';
        el.appendChild(note);
        obj.slice(0, 3).forEach(function(item, i) {
            el.appendChild(buildJsonTreeEl(item, path + '[' + i + ']', slot, depth + 1));
        });
        return el;
    }
    Object.keys(obj).forEach(function(k) {
        var childPath = path ? path + '.' + k : k;
        var v = obj[k];
        var row = document.createElement('div');
        row.style.cssText = 'display:flex; align-items:flex-start; gap:0.4rem; margin-bottom:0.2rem; flex-wrap:wrap;';
        var keyEl = document.createElement('span');
        keyEl.style.cssText = 'color:var(--accent-color,#3675f1); font-weight:600; white-space:nowrap;';
        keyEl.textContent = k + ': ';
        row.appendChild(keyEl);
        if (typeof v === 'object' && v !== null) {
            var nested = document.createElement('div');
            nested.style.flex = '1 1 100%';
            nested.appendChild(buildJsonTreeEl(v, childPath, slot, depth + 1));
            row.appendChild(nested);
        } else {
            var strVal = v === null ? 'null' : String(v);
            var isText = typeof v === 'string' && v.length > 0;
            var isLong = isText && (v.length > 30 || v.indexOf('\n') !== -1);
            var valEl = document.createElement('span');
            valEl.textContent = strVal.length > 80 ? strVal.substring(0, 80) + '\u2026' : strVal;
            valEl.style.cssText = 'flex:1; min-width:0; word-break:break-word; color:' + (typeof v === 'string' ? 'var(--text-main,#111)' : '#888') + ';';
            row.appendChild(valEl);
            if (isText) {
                var btn = document.createElement('button');
                btn.type = 'button';
                btn.textContent = 'Select';
                btn.dataset.fieldPath = childPath;
                btn.style.cssText = 'font-size:0.78rem; padding:0.15rem 0.55rem; white-space:nowrap; border-radius:5px; border:none; cursor:pointer; flex-shrink:0;' +
                    (isLong ? 'background:var(--accent-color,#3675f1); color:#fff;' : 'background:var(--bg-card,#e0e0e0); color:var(--text-main,#333);');
                (function(cp, s) {
                    btn.addEventListener('click', function() { selectLyricsField(s, cp); });
                })(childPath, slot);
                row.appendChild(btn);
            }
        }
        el.appendChild(row);
    });
    return el;
}

function selectLyricsField(slot, path) {
    lyricsApiState[slot].selectedField = path;
    var disp = document.getElementById('lyrics-api-' + slot + '-selected-display');
    if (disp) { disp.textContent = 'Lyrics field: ' + path; disp.style.color = 'var(--status-success-text,#1f4f1f)'; }
    highlightSelectedField(slot);
    updateLyricsApiHiddenInputs(slot);
}

function highlightSelectedField(slot) {
    var tree = document.getElementById('lyrics-api-' + slot + '-tree');
    if (!tree) return;
    var selectedPath = lyricsApiState[slot].selectedField;
    tree.querySelectorAll('button[data-field-path]').forEach(function(btn) {
        if (btn.dataset.fieldPath === selectedPath) {
            btn.textContent = '\u2713 Selected';
            btn.style.background = 'var(--status-success-bg,#e6f4ea)';
            btn.style.color = 'var(--status-success-text,#1f4f1f)';
        } else if (btn.textContent === '\u2713 Selected') {
            btn.textContent = 'Select';
            btn.style.background = 'var(--bg-card,#e0e0e0)';
            btn.style.color = 'var(--text-main,#333)';
        }
    });
}

function buildUrlTemplate(exampleUrl, paramRoles, params, pathSegments, pathRoles) {
    try {
        var parsed = new URL(exampleUrl);
        var artistParam = null, titleParam = null, apikeyParam = null, apikeyValue = null;
        Object.keys(paramRoles).forEach(function(pname) {
            var role = paramRoles[pname];
            if      (role === 'artist') artistParam = pname;
            else if (role === 'title')  titleParam  = pname;
            else if (role === 'apikey') { apikeyParam = pname; apikeyValue = String(params[pname] || ''); }
        });
        var artistInPath = false, titleInPath = false;
        (pathSegments || []).forEach(function(seg) {
            var role = (pathRoles || {})[seg.index] || 'none';
            if (role === 'artist') artistInPath = true;
            if (role === 'title')  titleInPath  = true;
        });
        if ((!artistParam && !artistInPath) || (!titleParam && !titleInPath)) return null;
        // Substitute dynamic path segments
        var pathParts = parsed.pathname.split('/');
        var segIdx = 0;
        var allSegs = pathSegments || [];
        for (var i = 0; i < pathParts.length; i++) {
            if (!pathParts[i]) { continue; }
            var thisSeg = null;
            for (var j = 0; j < allSegs.length; j++) {
                if (allSegs[j].index === segIdx) { thisSeg = allSegs[j]; break; }
            }
            if (thisSeg) {
                var segRole = (pathRoles || {})[thisSeg.index] || 'none';
                if      (segRole === 'artist') pathParts[i] = '{artist}';
                else if (segRole === 'title')  pathParts[i] = '{title}';
            }
            segIdx++;
        }
        var newPath = pathParts.join('/');
        var newParams = new URLSearchParams();
        (new URLSearchParams(parsed.search)).forEach(function(val, key) {
            var role = paramRoles[key] || 'none';
            if      (role === 'artist') newParams.set(key, '{artist}');
            else if (role === 'title')  newParams.set(key, '{title}');
            else if (role === 'apikey') { /* omit from template */ }
            else                        newParams.set(key, val);
        });
        var qs = newParams.toString()
            .replace(/%7Bartist%7D/gi, '{artist}')
            .replace(/%7Btitle%7D/gi,  '{title}');
        var template = parsed.origin + newPath + (qs ? '?' + qs : '');
        return {template: template, artistParam: artistParam, titleParam: titleParam, apikeyParam: apikeyParam, apikeyValue: apikeyValue};
    } catch(e) { return null; }
}

function updateLyricsApiHiddenInputs(slot) {
    var state = lyricsApiState[slot];
    var result = buildUrlTemplate(state.exampleUrl, state.paramRoles, state.params, state.pathSegments, state.pathRoles);
    var pre = 'LYRICS_API_' + slot + '_';
    function setVal(id, val) { var el = document.getElementById(id); if (el) el.value = val; }
    if (result) {
        setVal(pre + 'URL_TEMPLATE', result.template);
        setVal(pre + 'ARTIST_PARAM', result.artistParam || '');
        setVal(pre + 'TITLE_PARAM',  result.titleParam  || '');
        setVal(pre + 'APIKEY_PARAM', result.apikeyParam || '');
        if (result.apikeyValue !== null) setVal(pre + 'APIKEY_VALUE', result.apikeyValue);
    }
    if (state.selectedField !== null) setVal(pre + 'LYRICS_FIELD', state.selectedField);
    if (result && state.selectedField) {
        showLyricsApiSlotSummary(slot, result.template, result.artistParam, result.titleParam,
            state.selectedField, result.apikeyParam, !!(result.apikeyValue));
    }
}

function showLyricsApiSlotSummary(slot, template, artistParam, titleParam, lyricsField, apikeyParam, apikeyHasVal) {
    var summary = document.getElementById('lyrics-api-' + slot + '-config-summary');
    if (!summary) return;
    summary.style.display = 'block';
    summary.innerHTML = '';
    // Header row: title + delete button
    var headerRow = document.createElement('div');
    headerRow.style.cssText = 'display:flex; align-items:center; justify-content:space-between; margin-bottom:0.6rem;';
    var title = document.createElement('p');
    title.style.cssText = 'font-weight:700; margin:0; color:var(--status-success-text,#1f4f1f);';
    title.textContent = '\u2713 Slot ' + slot + ' configured \u2014 will be saved with the form';
    var deleteBtn = document.createElement('button');
    deleteBtn.type = 'button';
    deleteBtn.textContent = '\u2715 Delete';
    deleteBtn.style.cssText = 'background:var(--status-failure-bg,#fdecea); color:var(--status-failure-text,#7a1f1f); border:1px solid var(--status-failure-border,#f5c2c0); font-weight:600; font-size:0.85rem; padding:0.3rem 0.7rem; border-radius:6px; cursor:pointer; white-space:nowrap;';
    (function(s, tpl, ap, tp, lf, akp, akv) {
        deleteBtn.addEventListener('click', function() { pendingDeleteLyricsApiSlot(s, tpl, ap, tp, lf, akp, akv); });
    })(slot, template, artistParam, titleParam, lyricsField, apikeyParam, apikeyHasVal);
    headerRow.appendChild(title);
    headerRow.appendChild(deleteBtn);
    summary.appendChild(headerRow);
    var info = [['URL template', template]];
    if (artistParam) info.push(['Artist param', artistParam]);
    if (titleParam)  info.push(['Title param',  titleParam]);
    info.push(['Lyrics field', lyricsField]);
    if (apikeyParam) info.push(['API key param', apikeyParam]);
    if (apikeyHasVal) info.push(['API key', '(set)']);
    var dl = document.createElement('dl');
    dl.style.cssText = 'margin:0; display:grid; grid-template-columns:auto 1fr; gap:0.2rem 1rem; font-size:0.92rem;';
    info.forEach(function(pair) {
        var dt = document.createElement('dt'); dt.style.fontWeight = '600'; dt.textContent = pair[0];
        var dd = document.createElement('dd'); dd.style.cssText = 'margin:0; word-break:break-all;'; dd.textContent = pair[1];
        dl.appendChild(dt); dl.appendChild(dd);
    });
    summary.appendChild(dl);
    var editBtn = document.createElement('button');
    editBtn.type = 'button';
    editBtn.textContent = 'Re-analyze';
    editBtn.style.cssText = 'margin-top:0.75rem; background:var(--bg-input,#fff); color:var(--text-main,#333); border:1px solid var(--border-color,#ccc); font-weight:400;';
    (function(s) {
        editBtn.addEventListener('click', function() {
            var inputRow = document.getElementById('lyrics-api-' + s + '-input-row');
            if (inputRow) inputRow.style.display = 'flex';
            var analyzeArea = document.getElementById('lyrics-api-' + s + '-analyze-area');
            if (analyzeArea && lyricsApiState[s].jsonObj) analyzeArea.style.display = 'block';
        });
    })(slot);
    summary.appendChild(editBtn);
}

// Show a "pending deletion" warning — hidden inputs are cleared so the next Save
// will commit the deletion, but the user can Undo before that.
function pendingDeleteLyricsApiSlot(slot, template, artistParam, titleParam, lyricsField, apikeyParam, apikeyHasVal) {
    var pre = 'LYRICS_API_' + slot + '_';
    // Snapshot current values for Undo
    var snapshot = {};
    ['URL_TEMPLATE','ARTIST_PARAM','TITLE_PARAM','LYRICS_FIELD','APIKEY_PARAM','APIKEY_VALUE','TIMEOUT'].forEach(function(k) {
        var el = document.getElementById(pre + k);
        snapshot[k] = el ? el.value : '';
    });
    // Clear hidden inputs so Save will delete this slot
    ['URL_TEMPLATE','ARTIST_PARAM','TITLE_PARAM','LYRICS_FIELD','APIKEY_PARAM','APIKEY_VALUE'].forEach(function(k) {
        var el = document.getElementById(pre + k);
        if (el) el.value = '';
    });
    var toInput = document.getElementById(pre + 'TIMEOUT');
    if (toInput) toInput.value = '5';
    // Replace summary with a warning banner
    var summary = document.getElementById('lyrics-api-' + slot + '-config-summary');
    if (!summary) return;
    summary.style.display = 'block';
    summary.style.background = 'var(--status-warning-bg,#fff8e1)';
    summary.style.borderColor = 'var(--status-warning-border,#ffe082)';
    summary.innerHTML = '';
    var row = document.createElement('div');
    row.style.cssText = 'display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.5rem;';
    var msg = document.createElement('span');
    msg.style.cssText = 'font-weight:600; color:var(--status-warning-text,#7a5c00);';
    msg.textContent = '\u26a0\ufe0f Slot ' + slot + ' will be deleted when you save the form.';
    var undoBtn = document.createElement('button');
    undoBtn.type = 'button';
    undoBtn.textContent = 'Undo';
    undoBtn.style.cssText = 'font-weight:600; font-size:0.85rem; padding:0.3rem 0.8rem; border-radius:6px; cursor:pointer;';
    (function(s, snap, tpl, ap, tp, lf, akp, akv) {
        undoBtn.addEventListener('click', function() {
            // Restore hidden inputs
            ['URL_TEMPLATE','ARTIST_PARAM','TITLE_PARAM','LYRICS_FIELD','APIKEY_PARAM','APIKEY_VALUE','TIMEOUT'].forEach(function(k) {
                var el = document.getElementById('LYRICS_API_' + s + '_' + k);
                if (el) el.value = snap[k] || '';
            });
            // Restore summary card
            summary.style.background = '';
            summary.style.borderColor = '';
            showLyricsApiSlotSummary(s, tpl, ap, tp, lf, akp, akv);
        });
    })(slot, snapshot, template, artistParam, titleParam, lyricsField, apikeyParam, apikeyHasVal);
    row.appendChild(msg);
    row.appendChild(undoBtn);
    summary.appendChild(row);
}

function clearLyricsApiSlot(slot) {
    var state = lyricsApiState[slot];
    state.exampleUrl = ''; state.params = {}; state.paramRoles = {};
    state.pathSegments = []; state.pathRoles = {};
    state.jsonObj = null; state.selectedField = null; state.suggestedTimeout = null;
    var pre = 'LYRICS_API_' + slot + '_';
    ['URL_TEMPLATE','ARTIST_PARAM','TITLE_PARAM','LYRICS_FIELD','APIKEY_PARAM','APIKEY_VALUE'].forEach(function(k) {
        var el = document.getElementById(pre + k);
        if (el) { el.value = ''; el.dataset.originalValue = ''; }
    });
    var toInput = document.getElementById(pre + 'TIMEOUT');
    if (toInput) { toInput.value = '5'; toInput.dataset.originalValue = '5'; }
    var summary = document.getElementById('lyrics-api-' + slot + '-config-summary');
    if (summary) { summary.style.display = 'none'; summary.innerHTML = ''; summary.style.background = ''; summary.style.borderColor = ''; }
    var analyzeArea = document.getElementById('lyrics-api-' + slot + '-analyze-area');
    if (analyzeArea) { analyzeArea.style.display = 'none'; analyzeArea.innerHTML = ''; }
    var statusRow = document.getElementById('lyrics-api-' + slot + '-analyze-status');
    if (statusRow) statusRow.style.display = 'none';
    var toRow = document.getElementById('lyrics-api-' + slot + '-timeout-row');
    if (toRow) toRow.style.display = 'none';
    var inputRow = document.getElementById('lyrics-api-' + slot + '-input-row');
    if (inputRow) { inputRow.style.display = 'flex'; }
    var urlInput = document.getElementById('lyrics-api-' + slot + '-example-url');
    if (urlInput) urlInput.value = '';
}

document.getElementById('lyrics-api-1-analyze-btn').addEventListener('click', function() { analyzeLyricsApiSlot(1); });
document.getElementById('lyrics-api-2-analyze-btn').addEventListener('click', function() { analyzeLyricsApiSlot(2); });

loadSetupData();
