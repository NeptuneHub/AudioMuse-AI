var serverFields = {
    jellyfin: [
        {name: 'JELLYFIN_URL', label: 'Jellyfin URL', placeholder: 'http://your-jellyfin-server:8096'},
        {name: 'JELLYFIN_USER_ID', label: 'Jellyfin user ID', placeholder: 'your-user-id'},
        {name: 'JELLYFIN_TOKEN', label: 'Jellyfin API token', placeholder: 'your-api-token'}
    ],
    navidrome: [
        {name: 'NAVIDROME_URL', label: 'Navidrome URL', placeholder: 'http://your-navidrome-server:4533'},
        {name: 'NAVIDROME_USER', label: 'Navidrome username', placeholder: 'your-username'},
        {name: 'NAVIDROME_PASSWORD', label: 'Navidrome password', placeholder: 'your-password'}
    ],
    lyrion: [
        {name: 'LYRION_URL', label: 'Lyrion URL', placeholder: 'http://your-lyrion-server:9000'}
    ],
    emby: [
        {name: 'EMBY_URL', label: 'Emby URL', placeholder: 'http://your-emby-server:8096'},
        {name: 'EMBY_USER_ID', label: 'Emby user ID', placeholder: 'your-user-id'},
        {name: 'EMBY_TOKEN', label: 'Emby API token', placeholder: 'your-api-token'}
    ]
};

var testFeedback = document.getElementById('test-feedback');
var saveFeedback = document.getElementById('save-feedback');
var saveButton = document.getElementById('save-button');
var serverConfigFields = document.getElementById('server-config-fields');
var advancedFields = document.getElementById('advanced-fields');
var authCredentials = document.getElementById('auth-credentials');
var apiTokenRow = document.getElementById('api-token-row');
var authCredentialInputs = [
    document.getElementById('AUDIOMUSE_USER'),
    document.getElementById('AUDIOMUSE_PASSWORD'),
    document.getElementById('AUDIOMUSE_PASSWORD_CONFIRM'),
    document.getElementById('JWT_SECRET')
];
var setupForm = document.getElementById('setup-form');
var serverValues = {};
var originalValues = {};

function updateAuthVisibility() {
    var authEnabled = document.getElementById('AUTH_ENABLED').value === 'true';
    authCredentials.style.display = authEnabled ? 'grid' : 'none';
    apiTokenRow.style.display = authEnabled ? 'block' : 'none';
    authCredentialInputs.forEach(function(input) {
        if (!input) {
            return;
        }
        input.disabled = !authEnabled;
        var label = document.querySelector('label[for="' + input.id + '"]');
        if (input.id !== 'JWT_SECRET') {
            input.required = authEnabled;
            if (label) {
                if (authEnabled) {
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
            if (authEnabled) {
                label.textContent = 'API token (optional)';
            } else {
                label.textContent = 'API token';
            }
        }
    }
}

function createInputField(field, value) {
    var row = document.createElement('div');
    row.className = 'field-row';
    var label = document.createElement('label');
    label.setAttribute('for', field.name);
    label.textContent = field.label;
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
    serverConfigFields.innerHTML = '';
    if (!serverFields[serverType]) {
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
            originalValue: originalValues[field.name] !== undefined ? originalValues[field.name] : value
        };
        serverConfigFields.appendChild(createInputField(fieldCopy, value));
    });
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
        var basicData = {};
        var secretHasValue = {};
        data.basic_fields.forEach(function(item) {
            basicData[item.name] = item.value;
            if (item.secret) {
                secretHasValue[item.name] = item.has_value;
            }
        });
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

function updateServerFields() {
    saveCurrentServerValues();
    var serverType = document.getElementById('MEDIASERVER_TYPE').value;
    renderServerFields(serverType, serverValues);
}

function waitForHealthAndRedirect(redirectUrl) {
    var attempts = 0;
    var maxAttempts = 80;
    var consecutiveOk = 0;
    saveFeedback.className = 'status-pending inline-feedback';
    saveFeedback.style.display = 'block';
    saveFeedback.textContent = 'Configuration saved. Restarting services — please wait...';
    function checkHealth() {
        attempts += 1;
        fetch('/api/health', { cache: 'no-store' })
            .then(function(resp) {
                if (!resp.ok) {
                    throw new Error('Service not ready');
                }
                return resp.json();
            })
            .then(function(data) {
                if (data && data.status === 'ok') {
                    consecutiveOk += 1;
                    if (consecutiveOk >= 2) {
                        window.location.href = redirectUrl;
                        return;
                    }
                    setTimeout(checkHealth, 1500);
                } else {
                    consecutiveOk = 0;
                    throw new Error('Service not ready');
                }
            })
            .catch(function() {
                consecutiveOk = 0;
                if (attempts < maxAttempts) {
                    setTimeout(checkHealth, 1500);
                } else {
                    saveFeedback.className = 'status-failure inline-feedback';
                    saveFeedback.style.display = 'block';
                    saveFeedback.textContent = 'Restart timeout. Please refresh the page in a moment.';
                    saveButton.disabled = false;
                }
            });
    }

    setTimeout(checkHealth, 3000);
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
        waitForHealthAndRedirect('/');
    }).catch(function(err) {
        saveFeedback.className = 'status-failure inline-feedback';
        saveFeedback.style.display = 'block';
        saveFeedback.textContent = err.message;
    }).finally(function() {
        saveButton.disabled = false;
    });
});

document.getElementById('test-button').addEventListener('click', testConnection);
document.getElementById('MEDIASERVER_TYPE').addEventListener('change', updateServerFields);
document.getElementById('AUTH_ENABLED').addEventListener('change', updateAuthVisibility);
loadSetupData();