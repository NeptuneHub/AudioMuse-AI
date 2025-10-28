// /static/collection_script.js

document.addEventListener('DOMContentLoaded', () => {
    // --- State Variables ---
    let pb = null; // PocketBase instance
    let currentTaskId = null;

    // --- DOM Element References ---
    const loginSection = document.getElementById('login-section');
    const githubLoginBtn = document.getElementById('github-login-btn');
    const pocketbaseUrlInput = document.getElementById('pocketbase-url');
    const privacyCheckbox = document.getElementById('privacy-ack');
    const privacyLink = document.getElementById('privacy-link');
    const appContent = document.getElementById('app-content');
    const userInfoDisplay = document.getElementById('user-info');
    const logoutBtn = document.getElementById('logout-btn');
    const startSyncBtn = document.getElementById('start-sync-btn');
    const cancelSyncBtn = document.getElementById('cancel-sync-btn');
    const statusTaskId = document.getElementById('status-task-id');
    const statusRunningTime = document.getElementById('status-running-time');
    const statusTaskType = document.getElementById('status-task-type');
    const statusStatus = document.getElementById('status-status');
    const statusProgress = document.getElementById('status-progress');
    const progressBar = document.getElementById('progress-bar');
    const statusLog = document.getElementById('status-log');
    const statusDetails = document.getElementById('status-details');

    // --- FIXED: References for moving the privacy policy element ---
    const privacyGroup = document.querySelector('.privacy-ack-group');
    const loginButtonContainer = githubLoginBtn.parentElement;
    const logoutButtonContainer = logoutBtn.parentElement;

    // --- Core & Auth Functions ---
    function initialize() {
        try {
            const url = pocketbaseUrlInput.value || localStorage.getItem('pocketbaseUrl');
            if (url) {
                pocketbaseUrlInput.value = url;
                pb = new PocketBase(url);
            }
        } catch (error) {
            console.error("Failed to initialize PocketBase:", error);
            showMessageBox("Error", "Invalid PocketBase URL.");
            pb = null;
        }
        updateUI();
    }

    function updateUI() {
        if (pb && pb.authStore.isValid) {
            // --- User is LOGGED IN ---
            loginSection.style.display = 'none';
            appContent.style.display = 'block';
            const user = pb.authStore.model;
            userInfoDisplay.textContent = user.email || user.username || user.id;

            // --- FIXED: Move privacy policy group to the logged-in section ---
            // It's placed just before the logout button's container.
            logoutButtonContainer.before(privacyGroup);
            privacyCheckbox.checked = true;
            privacyCheckbox.disabled = true; // Make it uncheckable

        } else {
            // --- User is LOGGED OUT ---
            loginSection.style.display = 'block';
            appContent.style.display = 'none';

            // --- FIXED: Move privacy policy group back to the login section ---
            // It's placed just before the login button's container.
            loginButtonContainer.before(privacyGroup);
            privacyCheckbox.disabled = false; // Make it checkable again
            githubLoginBtn.disabled = !privacyCheckbox.checked; // Re-evaluate button state
        }
    }

    async function loginWithGithub() {
        const url = pocketbaseUrlInput.value;
        if (!url) {
            showMessageBox('Error', 'Please enter the PocketBase Server URL.');
            return;
        }
        try {
             pb = new PocketBase(url);
        } catch(e) {
             showMessageBox('Error', 'Invalid PocketBase URL provided.');
             return;
        }
        try {
            await pb.collection('users').authWithOAuth2({ provider: 'github' });
            localStorage.setItem('pocketbaseUrl', url);
            showMessageBox('Success', 'Successfully logged in with GitHub.');
            updateUI();
        } catch (error) {
            console.error('GitHub login failed:', error);
            showMessageBox('Login Failed', 'Could not authenticate with GitHub.');
            pb.authStore.clear();
            updateUI();
        }
    }

    function logout() {
        if (pb) pb.authStore.clear();
        // --- FIXED: Also uncheck the privacy box on logout ---
        privacyCheckbox.checked = false;
        showMessageBox('Logged Out', 'You have been successfully logged out.');
        updateUI(); // This call will move the element and update all UI states correctly.
    }

    // --- Task Management Functions ---
    function formatRunningTime(totalSeconds) {
        if (totalSeconds === null || isNaN(totalSeconds) || totalSeconds < 0) return '-- : -- : --';
        const hours = Math.floor(totalSeconds / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = Math.floor(totalSeconds % 60);
        const pad = (num) => String(num).padStart(2, '0');
        return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
    }

    function displayTaskStatus(task) {
        statusTaskId.textContent = task.task_id || 'N/A';
        statusRunningTime.textContent = formatRunningTime(task.running_time_seconds);
        statusTaskType.textContent = task.task_type || 'N/A';
        const stateUpper = (task.state || task.status || 'IDLE').toUpperCase();
        statusStatus.textContent = stateUpper;
        statusProgress.textContent = task.progress || 0;
        progressBar.style.width = `${task.progress || 0}%`;

        let statusClass = 'status-idle';
        if (['SUCCESS', 'FINISHED'].includes(stateUpper)) statusClass = 'status-success';
        else if (['FAILURE', 'FAILED', 'REVOKED', 'CANCELED'].includes(stateUpper)) statusClass = 'status-failure';
        else if (['PENDING', 'STARTED', 'PROGRESS'].includes(stateUpper)) statusClass = 'status-pending';
        statusStatus.className = `status-text ${statusClass}`;
        
        let statusMessage = 'N/A';
        if (task.details && typeof task.details === 'object') {
             statusMessage = (Array.isArray(task.details.log) && task.details.log.length > 0) 
                 ? task.details.log[task.details.log.length - 1] 
                 : (task.details.message || 'No message.');
        } else if (task.details) {
             statusMessage = task.details.toString();
        }
        statusLog.textContent = statusMessage;
        statusDetails.textContent = typeof task.details === 'object' ? JSON.stringify(task.details, null, 2) : task.details;
    }
    
    function disableAllTaskButtons(isDisabled) {
        startSyncBtn.disabled = isDisabled;
        if(isDisabled) {
            startSyncBtn.style.cursor = 'not-allowed';
            startSyncBtn.style.backgroundColor = '#9ca3af';
        } else {
            startSyncBtn.style.cursor = '';
            startSyncBtn.style.backgroundColor = '';
        }
    }

    function updateCancelButtonState(isDisabled) {
        cancelSyncBtn.disabled = isDisabled;
    }

    async function checkActiveTasks() {
        try {
            const response = await fetch(getActiveTasksEndpointUrl);
            const activeTask = await response.json();

            if (activeTask && activeTask.task_id) {
                currentTaskId = activeTask.task_id;
                displayTaskStatus(activeTask);
                disableAllTaskButtons(true);
                updateCancelButtonState(false);
                return true;
            } else if (currentTaskId) {
                const finishedTaskId = currentTaskId;
                currentTaskId = null;
                try {
                    const finalStatusResponse = await fetch(getTaskStatusEndpointUrl.replace(':taskId:', encodeURIComponent(finishedTaskId)));
                    if (finalStatusResponse.ok) {
                        const finalStatusData = await finalStatusResponse.json();
                        displayTaskStatus(finalStatusData);
                    }
                } finally {
                    disableAllTaskButtons(false);
                    updateCancelButtonState(true);
                }
                return true;
            } else {
                disableAllTaskButtons(false);
                updateCancelButtonState(true);
            }
        } catch (error) {
            console.error('Error checking active tasks:', error);
            currentTaskId = null;
            disableAllTaskButtons(false);
            updateCancelButtonState(true);
        }
        return false;
    }

    async function fetchAndDisplayLastCollectionTask() {
        try {
            const response = await fetch(getLastCollectionTaskUrl);
            if (response.ok) {
                const lastTask = await response.json();
                if (lastTask && lastTask.task_id && lastTask.status !== 'NO_PREVIOUS_TASK') {
                     displayTaskStatus(lastTask);
                }
            }
        } catch (error) {
            console.error('Error fetching last collection task:', error);
        }
    }

    async function startSync() {
        if (!pb || !pb.authStore.isValid) {
            showMessageBox('Error', 'You are not logged in. Please log in first.');
            updateUI();
            return;
        }
        const payload = {
            url: pocketbaseUrlInput.value,
            token: pb.authStore.token,
            num_albums: parseInt(document.getElementById('num-albums').value, 10)
        };
        disableAllTaskButtons(true);
        try {
            const response = await fetch(startCollectionSyncUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const result = await response.json();
            if (!response.ok || !result.task_id) throw new Error(result.message || 'Failed to start sync.');
            
            currentTaskId = result.task_id;
            displayTaskStatus({ task_id: result.task_id, task_type: "main_collection_sync", status: 'PENDING', progress: 0, details: 'Task enqueued.' });
            updateCancelButtonState(false);
            showMessageBox('Task Started', `Sync task enqueued with ID: ${result.task_id}`);
        } catch (error) {
            console.error('Error starting sync task:', error);
            showMessageBox('Error', `Could not start sync task: ${error.message}`);
            disableAllTaskButtons(false);
        }
    }

    async function cancelSync() {
        if (!currentTaskId) return;
        updateCancelButtonState(true);
        try {
            const response = await fetch(cancelTaskEndpointUrl.replace(':taskId:', encodeURIComponent(currentTaskId)), { method: 'POST' });
            const result = await response.json();
            if (!response.ok) throw new Error(result.message || 'Failed to send cancellation request.');
            showMessageBox('Cancellation Sent', result.message);
        } catch (error) {
            console.error('Error cancelling task:', error);
            showMessageBox('Error', `Could not cancel task: ${error.message}`);
            updateCancelButtonState(false);
        }
    }

    function showMessageBox(title, message) {
        const boxId = 'custom-message-box';
        document.getElementById(boxId)?.remove();
        const messageBox = document.createElement('div');
        messageBox.id = boxId;
        messageBox.style.cssText = 'position: fixed; top: 20px; right: 20px; background-color: #fff; color: #1F2937; padding: 20px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); z-index: 1000; border: 1px solid #E5E7EB; max-width: 400px; text-align: left;';
        messageBox.innerHTML = `<h3 style="font-weight: 600; margin-top:0; margin-bottom: 10px; color: #111827;">${title}</h3><p style="margin:0;">${message}</p><button style="position: absolute; top: 10px; right: 10px; background: none; border: none; font-size: 1.5rem; color: #9CA3AF; cursor: pointer;" onclick="this.parentNode.remove()">&times;</button>`;
        document.body.appendChild(messageBox);
        setTimeout(() => messageBox.remove(), 5000);
    }

    // --- Event Listeners & Initialization ---
    githubLoginBtn.addEventListener('click', loginWithGithub);
    logoutBtn.addEventListener('click', logout);
    startSyncBtn.addEventListener('click', startSync);
    cancelSyncBtn.addEventListener('click', cancelSync);
    privacyLink.addEventListener('click', () => setTimeout(() => {
        if (!privacyCheckbox.checked) {
            privacyCheckbox.checked = true;
            privacyCheckbox.dispatchEvent(new Event('change'));
        }
    }, 100));
    privacyCheckbox.addEventListener('change', () => githubLoginBtn.disabled = !privacyCheckbox.checked);
    
    async function mainInit() {
        console.assert(getActiveTasksEndpointUrl, "Missing getActiveTasksEndpointUrl url variable");
        console.assert(getTaskStatusEndpointUrl, "Missing getTaskStatusEndpointUrl url variable");
        console.assert(cancelTaskEndpointUrl, "Missing cancelTaskEndpointUrl url variable");
        console.assert(startCollectionSyncUrl, "Missing startCollectionSyncUrl url variable");
        console.assert(getLastCollectionTaskUrl, "Missing getLastCollectionTaskUrl url variable");

        initialize();
        if (!await checkActiveTasks()) {
            await fetchAndDisplayLastCollectionTask();
            updateCancelButtonState(true);
        }
        setInterval(checkActiveTasks, 3000);
    }
    mainInit();
});
