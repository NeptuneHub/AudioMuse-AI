/**
 * Shared utility functions for AudioMuse-AI frontend.
 */

/**
 * Escape HTML special characters to prevent XSS.
 * @param {*} str - Value to escape
 * @returns {string} Escaped string safe for innerHTML
 */
function escapeHtml(str) {
    if (str == null) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
