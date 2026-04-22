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
var serverValues = {};
var serverSecretHasValue = {};
var originalValues = {};
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
        var visibleAdvancedData = advancedData;
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
        updateAuthVisibility();
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
loadSetupData();
