function renderTaskError(task) {
    var block = document.getElementById('status-error-block');
    if (!block) {
        return;
    }
    var err = task && task.details && task.details.error;
    if (err && typeof err === 'object' && err.error_code) {
        var codeEl = document.getElementById('status-error-code');
        var classEl = document.getElementById('status-error-class');
        var messageEl = document.getElementById('status-error-message');
        if (codeEl) { codeEl.textContent = err.error_code; }
        if (classEl) { classEl.textContent = err.error_class || 'N/A'; }
        if (messageEl) { messageEl.textContent = err.error_message || 'N/A'; }
        block.style.display = '';
    } else {
        block.style.display = 'none';
    }
}

function formatErrorText(errObj) {
    if (errObj && typeof errObj === 'object' && errObj.error_code) {
        var message = (typeof errObj.error === 'string' && errObj.error) ? errObj.error : (errObj.error_message || '');
        return '[' + errObj.error_code + '] ' + (errObj.error_class || 'Error') + ': ' + message;
    }
    if (typeof errObj === 'string') {
        return errObj;
    }
    return '';
}

function apiErrorText(body, fallback) {
    if (body && typeof body === 'object') {
        if (body.error_code) {
            return formatErrorText(body);
        }
        if (typeof body.error === 'string' && body.error) {
            return body.error;
        }
    }
    return fallback;
}
