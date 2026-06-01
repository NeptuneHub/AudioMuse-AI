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
        return '[' + errObj.error_code + '] ' + (errObj.error_class || 'Error') + ': ' + (errObj.error_message || '');
    }
    if (typeof errObj === 'string') {
        return errObj;
    }
    return '';
}
