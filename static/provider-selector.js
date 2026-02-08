/**
 * Provider Selector Component for Multi-Provider Playlist Support
 *
 * Usage:
 * 1. Include this script in your template
 * 2. Add a container div: <div id="provider-selector-container"></div>
 * 3. Call initProviderSelector() after DOM is loaded
 * 4. Get selected value with getSelectedProviders() when creating playlist
 */

function _escapeHtml(str) {
    if (str == null) return '';
    return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;')
        .replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#039;');
}

let _providers = [];
let _selectedProviderValue = null; // null = primary/default, 'all' = all providers, number = specific provider

/**
 * Initialize the provider selector component.
 * Fetches enabled providers and renders the dropdown.
 *
 * @param {string} containerId - ID of the container element
 * @param {object} options - Configuration options
 * @param {boolean} options.showAllOption - Whether to show "All Providers" option (default: true)
 * @param {boolean} options.showLabel - Whether to show label (default: true)
 * @param {string} options.labelText - Label text (default: "Save to:")
 */
async function initProviderSelector(containerId = 'provider-selector-container', options = {}) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.warn(`Provider selector container '${containerId}' not found`);
        return;
    }

    const showAllOption = options.showAllOption !== false;
    const showLabel = options.showLabel !== false;
    const labelText = options.labelText || 'Save to:';

    try {
        const response = await fetch('/api/providers/enabled');
        _providers = await response.json();
    } catch (err) {
        console.error('Failed to load providers:', err);
        _providers = [];
    }

    // Only show selector if there are multiple providers or showAllOption is true
    if (_providers.length <= 1 && !showAllOption) {
        container.style.display = 'none';
        return;
    }

    // Build the selector HTML
    let html = '<div class="provider-selector">';

    if (showLabel) {
        html += `<label for="provider-select">${labelText}</label>`;
    }

    html += '<select id="provider-select" class="provider-select">';
    html += '<option value="">Primary Provider</option>';

    if (showAllOption && _providers.length > 1) {
        html += '<option value="all">All Providers</option>';
    }

    _providers.forEach(p => {
        html += `<option value="${_escapeHtml(p.id)}">${_escapeHtml(p.name)}</option>`;
    });

    html += '</select></div>';

    container.innerHTML = html;

    // Add event listener
    const select = document.getElementById('provider-select');
    if (select) {
        select.addEventListener('change', function() {
            const value = this.value;
            if (value === '') {
                _selectedProviderValue = null;
            } else if (value === 'all') {
                _selectedProviderValue = 'all';
            } else {
                _selectedProviderValue = parseInt(value, 10);
            }
        });
    }
}

/**
 * Get the currently selected provider value.
 *
 * @returns {null|string|number} null for primary, 'all' for all, or provider ID
 */
function getSelectedProviders() {
    return _selectedProviderValue;
}

/**
 * Get the list of loaded providers.
 *
 * @returns {Array} List of provider objects
 */
function getProviderList() {
    return _providers;
}

/**
 * Check if multiple providers are available.
 *
 * @returns {boolean}
 */
function hasMultipleProviders() {
    return _providers.length > 1;
}

/**
 * Add provider_ids to a playlist creation payload.
 *
 * @param {object} payload - The existing payload object
 * @returns {object} Payload with provider_ids added if applicable
 */
function addProviderToPayload(payload) {
    const selected = getSelectedProviders();
    if (selected !== null) {
        payload.provider_ids = selected;
    }
    return payload;
}

// CSS styles for the provider selector
const providerSelectorStyles = `
.provider-selector {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
}

.provider-selector label {
    font-weight: 500;
    font-size: 0.9rem;
    white-space: nowrap;
}

.provider-select {
    padding: 0.4rem 0.75rem;
    border: 1px solid var(--border-color, #ccc);
    border-radius: 4px;
    background: var(--bg-input, #fff);
    color: var(--text-main, #333);
    font-size: 0.9rem;
    min-width: 150px;
}

.provider-select:focus {
    outline: none;
    border-color: var(--color-primary, #007bff);
}

/* Compact variant for inline use */
.provider-selector.compact {
    margin-bottom: 0;
}

.provider-selector.compact label {
    font-size: 0.85rem;
}

.provider-selector.compact .provider-select {
    padding: 0.3rem 0.5rem;
    font-size: 0.85rem;
    min-width: 120px;
}
`;

// Inject styles when script loads
(function() {
    const styleEl = document.createElement('style');
    styleEl.textContent = providerSelectorStyles;
    document.head.appendChild(styleEl);
})();
