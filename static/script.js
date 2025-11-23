// --- DOM Element References ---
const mainContent = document.getElementById('main-content');

// View switcher
const basicViewBtn = document.getElementById('basic-view-btn');
const advancedViewBtn = document.getElementById('advanced-view-btn');
const advancedParams = document.querySelectorAll('.advanced-param');
const kmeansParamsBasic = document.getElementById('kmeans-params-basic');
const basicAlgorithmDisplay = document.getElementById('basic-algorithm-display');


// Config Form
const clusterAlgorithmSelect = document.getElementById('config-cluster_algorithm');
const dbscanParamsDiv = document.getElementById('dbscan-params');
const gmmParamsDiv = document.getElementById('gmm-params');
const spectralParamsDiv = document.getElementById('spectral-params'); // New reference for Spectral
const aiModelProviderSelect = document.getElementById('config-ai_model_provider');
const ollamaConfigGroup = document.getElementById('ollama-config-group');
const openaiConfigGroup = document.getElementById('openai-config-group');
const geminiConfigGroup = document.getElementById('gemini-config-group');
const mistralConfigGroup = document.getElementById('mistral-config-group');

// Task Buttons
const startAnalysisBtn = document.getElementById('start-analysis-btn');
const startClusteringBtn = document.getElementById('start-clustering-btn');
const fetchPlaylistsBtn = document.getElementById('fetch-playlists-btn');
const cancelTaskBtn = document.getElementById('cancel-task-btn');

// Task Status Display
const statusTaskId = document.getElementById('status-task-id');
const statusRunningTime = document.getElementById('status-running-time');
const statusTaskType = document.getElementById('status-task-type');
const statusStatus = document.getElementById('status-status');
const statusProgress = document.getElementById('status-progress');
const progressBar = document.getElementById('progress-bar');
const statusLog = document.getElementById('status-log');
const statusDetails = document.getElementById('status-details');

// Playlists
const playlistsSection = document.getElementById('playlists-section');
const playlistsContainer = document.getElementById('playlists-container');

// --- State Variables ---
let currentTaskId = null;
let lastPolledTaskDetails = {};

// --- Functions ---

/**
 * Formats a duration in seconds into a HH : MM : SS string.
 * @param {number} totalSeconds The total seconds to format.
 * @returns {string} The formatted time string.
 */
function formatRunningTime(totalSeconds) {
    if (totalSeconds === null || totalSeconds === undefined || isNaN(totalSeconds) || totalSeconds < 0) {
        return '-- : -- : --';
    }
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = Math.floor(totalSeconds % 60);

    const pad = (num) => String(num).padStart(2, '0');

    return `${pad(hours)} : ${pad(minutes)} : ${pad(seconds)}`;
}

/**
 * Switches between basic and advanced configuration views.
 * @param {('basic'|'advanced')} viewToShow The view to display.
 */
function switchView(viewToShow) {
    if (viewToShow === 'basic') {
        basicViewBtn.classList.add('active');
        advancedViewBtn.classList.remove('active');
        advancedParams.forEach(el => el.classList.add('hidden'));
        if (kmeansParamsBasic) kmeansParamsBasic.classList.remove('hidden'); // Show K-Means params in basic
        if (basicAlgorithmDisplay) {
            basicAlgorithmDisplay.classList.remove('hidden');
            const pElement = basicAlgorithmDisplay.querySelector('p');
            if (pElement) {
                pElement.style.color = '';
            }
        }
        if (clusterAlgorithmSelect) clusterAlgorithmSelect.classList.add('hidden'); // Hide algorithm dropdown in basic

        // In basic view, we only support K-Means
        if (clusterAlgorithmSelect) {
            clusterAlgorithmSelect.value = 'kmeans';
        }

    } else { // advanced view
        basicViewBtn.classList.remove('active');
        advancedViewBtn.classList.add('active');
        advancedParams.forEach(el => el.classList.remove('hidden'));
        kmeansParamsBasic.classList.add('hidden'); // Hide the basic K-Means inputs
        if (basicAlgorithmDisplay) basicAlgorithmDisplay.classList.add('hidden');
        if (clusterAlgorithmSelect) clusterAlgorithmSelect.classList.remove('hidden'); // Show algorithm dropdown in advanced
    }
    // This function handles showing/hiding algorithm-specific params
    toggleClusteringParams();
}

async function fetchConfig() {
    try {
        const response = await fetch(getConfigEndpointUrl);
        const config = await response.json();
        renderConfig(config);
        // Call switchView here to ensure the view is set correctly *before* showing the content
        switchView('basic');
        toggleAiConfig();
    } catch (error) {
        console.error('Error fetching config:', error);
        showMessageBox('Error', 'Failed to load configuration. Please check the backend server.');
    }
}

function renderConfig(config) {
    // MODIFIED: Removed rendering for Jellyfin config as the inputs are gone.

    // Analysis
    document.getElementById('config-num_recent_albums').value = config.num_recent_albums || 0;
    document.getElementById('config-top_n_moods').value = config.top_n_moods || 0;

    // Clustering
    document.getElementById('config-top_n_playlists').value = config.top_n_playlists || 0;
    clusterAlgorithmSelect.value = (config.cluster_algorithm === 'dbscan' || config.cluster_algorithm === 'gmm' || config.cluster_algorithm === 'spectral') ? config.cluster_algorithm : 'kmeans';
    document.getElementById('config-max_distance').value = config.max_distance || 0;
    document.getElementById('config-max_songs_per_cluster').value = config.max_songs_per_cluster || 0;
    document.getElementById('config-pca_components_min').value = config.pca_components_min || 0;
    document.getElementById('config-pca_components_max').value = config.pca_components_max || 0;
    document.getElementById('config-clustering_runs').value = config.clustering_runs || 0;
    document.getElementById('config-min_songs_per_genre_for_stratification').value = config.min_songs_per_genre_for_stratification || 0;
    document.getElementById('config-stratified_sampling_target_percentile').value = config.stratified_sampling_target_percentile || 0;
    document.getElementById('config-score_weight_diversity').value = config.score_weight_diversity || 0;
    document.getElementById('config-score_weight_purity').value = config.score_weight_purity || 0;
    document.getElementById('config-score_weight_silhouette').value = config.score_weight_silhouette || 0;
    document.getElementById('config-score_weight_davies_bouldin').value = config.score_weight_davies_bouldin || 0;
    document.getElementById('config-score_weight_calinski_harabasz').value = config.score_weight_calinski_harabasz || 0;
    // CORRECTED: Fixed variable names to match the Python config.
    document.getElementById('config-score_weight_other_feature_diversity').value = config.score_weight_other_feature_diversity || 0;
    document.getElementById('config-score_weight_other_feature_purity').value = config.score_weight_other_feature_purity || 0;
    document.getElementById('config-enable_clustering_embeddings').checked = config.enable_clustering_embeddings;

    // Algorithm Specific
    document.getElementById('config-dbscan_eps_min').value = config.dbscan_eps_min || 0;
    document.getElementById('config-dbscan_eps_max').value = config.dbscan_eps_max || 0;
    document.getElementById('config-dbscan_min_samples_min').value = config.dbscan_min_samples_min || 0;
    document.getElementById('config-dbscan_min_samples_max').value = config.dbscan_min_samples_max || 0;
    document.getElementById('config-num_clusters_min').value = config.num_clusters_min || 0;
    document.getElementById('config-num_clusters_max').value = config.num_clusters_max || 0;
    document.getElementById('config-gmm_n_components_min').value = config.gmm_n_components_min || 0;
    document.getElementById('config-gmm_n_components_max').value = config.gmm_n_components_max || 0;
    document.getElementById('config-spectral_n_clusters_min').value = config.spectral_n_clusters_min || 0;
    document.getElementById('config-spectral_n_clusters_max').value = config.spectral_n_clusters_max || 0;


    // AI Naming
    aiModelProviderSelect.value = config.ai_model_provider || 'NONE';
    document.getElementById('config-ollama_server_url').value = config.ollama_server_url || 'http://127.0.0.1:11434/api/generate';
    document.getElementById('config-ollama_model_name').value = config.ollama_model_name || 'mistral:7b';
    document.getElementById('config-openai_server_url').value = config.openai_server_url || 'https://openrouter.ai/api/v1/chat/completions';
    document.getElementById('config-openai_model_name').value = config.openai_model_name || '';
    document.getElementById('config-gemini_model_name').value = config.gemini_model_name || 'gemini-2.5-pro';
    document.getElementById('config-mistral_model_name').value = config.mistral_model_name || 'ministral-3b-latest';
}

function toggleClusteringParams() {
    const selectedAlgorithm = clusterAlgorithmSelect.value;
    dbscanParamsDiv.classList.add('hidden');
    gmmParamsDiv.classList.add('hidden');
    spectralParamsDiv.classList.add('hidden'); // Hide spectral params by default
    // K-Means params (kmeansParamsBasic) are handled based on view below

    // Only show algorithm-specific params in advanced view
    if (advancedViewBtn.classList.contains('active')) {
        // First, ensure K-Means params are hidden if K-Means is NOT selected
        if (selectedAlgorithm !== 'kmeans' && kmeansParamsBasic) {
            kmeansParamsBasic.classList.add('hidden');
        }

        if (selectedAlgorithm === 'dbscan') {
            dbscanParamsDiv.classList.remove('hidden');
        } else if (selectedAlgorithm === 'gmm') {
            gmmParamsDiv.classList.remove('hidden');
        } else if (selectedAlgorithm === 'spectral') {
            spectralParamsDiv.classList.remove('hidden');
        } else if (selectedAlgorithm === 'kmeans' && kmeansParamsBasic) {
            kmeansParamsBasic.classList.remove('hidden'); // Show K-Means params if K-Means is selected
        }
    }
}

function toggleAiConfig() {
    const provider = aiModelProviderSelect.value;
    ollamaConfigGroup.classList.add('hidden');
    openaiConfigGroup.classList.add('hidden');
    geminiConfigGroup.classList.add('hidden');
    mistralConfigGroup.classList.add('hidden');

    if (provider === 'OLLAMA') {
        ollamaConfigGroup.classList.remove('hidden');
    } else if (provider === 'OPENAI') {
        openaiConfigGroup.classList.remove('hidden');
    } else if (provider === 'GEMINI') {
        geminiConfigGroup.classList.remove('hidden');
    } else if (provider === 'MISTRAL') {
        mistralConfigGroup.classList.remove('hidden');
    }
}

function updateCancelButtonState(isDisabled) {
    cancelTaskBtn.disabled = isDisabled;
}

async function checkActiveTasks() {
    try {
        const response = await fetch(getActiveTasksEndpointUrl);
        const mainActiveTask = await response.json();

        if (mainActiveTask && mainActiveTask.task_id) {
            currentTaskId = mainActiveTask.task_id;
            const currentStatusUpper = (mainActiveTask.status || mainActiveTask.state || 'UNKNOWN').toUpperCase();
            const terminalStates = ['SUCCESS', 'FINISHED', 'FAILURE', 'FAILED', 'REVOKED', 'CANCELED'];

            displayTaskStatus(mainActiveTask);

            const previousStateWasTerminal = lastPolledTaskDetails[currentTaskId]?.state && terminalStates.includes(lastPolledTaskDetails[currentTaskId].state.toUpperCase());

            if (terminalStates.includes(currentStatusUpper) && !previousStateWasTerminal) {
                let alertTitle = 'Task Update';
                let alertMessage = `Task ${mainActiveTask.task_id} (${mainActiveTask.task_type_from_db || 'Unknown Type'}) has ${currentStatusUpper.toLowerCase()}.`;
                if (['SUCCESS', 'FINISHED'].includes(currentStatusUpper)) alertTitle = 'Task Completed';
                else if (['FAILURE', 'FAILED'].includes(currentStatusUpper)) alertTitle = 'Task Failed';
                else if (['REVOKED', 'CANCELED'].includes(currentStatusUpper)) alertTitle = 'Task Canceled';

                showMessageBox(alertTitle, alertMessage);
            }
            lastPolledTaskDetails[currentTaskId] = { state: currentStatusUpper, ...mainActiveTask };
            disableTaskButtons(true);
            updateCancelButtonState(false);
            return true;
        } else if (currentTaskId) {
            const finishedTaskId = currentTaskId;
            const previousDetails = lastPolledTaskDetails[finishedTaskId];
            currentTaskId = null;

            try {
                const finalStatusResponse = await fetch(getTaskStatusEndpointUrl.replace(':taskId:', encodeURIComponent(finishedTaskId)));
                if (finalStatusResponse.ok) {
                    const finalStatusData = await finalStatusResponse.json();
                    const upperFinalStatus = (finalStatusData.state || 'UNKNOWN').toUpperCase();
                    const terminalStates = ['SUCCESS', 'FINISHED', 'FAILURE', 'FAILED', 'REVOKED', 'CANCELED'];

                    const finalStatusIsTerminal = terminalStates.includes(upperFinalStatus);
                    const previousStateWasTerminal = previousDetails && previousDetails.state && terminalStates.includes(previousDetails.state.toUpperCase());

                    if (finalStatusIsTerminal && !previousStateWasTerminal) {
                        let alertTitle = 'Task Finished';
                        let alertMessage = `Task ${finalStatusData.task_id} (${finalStatusData.task_type_from_db || 'Unknown Type'}) has ${upperFinalStatus.toLowerCase()}.`;
                        if (['SUCCESS', 'FINISHED'].includes(upperFinalStatus)) alertTitle = 'Task Completed';
                        else if (['FAILURE', 'FAILED'].includes(upperFinalStatus)) alertTitle = 'Task Failed';
                        else if (['REVOKED', 'CANCELED'].includes(upperFinalStatus)) alertTitle = 'Task Canceled';

                        showMessageBox(alertTitle, alertMessage);
                    }
                    displayTaskStatus(finalStatusData);
                } else {
                    showMessageBox('Task Finished', `Task ${finishedTaskId} is no longer active. Final status could not be retrieved.`);
                    displayTaskStatus({ task_id: finishedTaskId, state: 'FINISHED', progress: 100 });
                }
            } catch (e) {
                console.error(`Error fetching final status for ${finishedTaskId}:`, e);
            } finally {
                delete lastPolledTaskDetails[finishedTaskId];
                disableTaskButtons(false);
                updateCancelButtonState(true);
            }
            return true;
        } else {
            disableTaskButtons(false);
            updateCancelButtonState(true);
        }
    } catch (error) {
        console.error('Error checking active tasks:', error);
        displayTaskStatus({ task_id: 'Error', task_type: 'Error', state: 'ERROR', progress: 0, details: `Polling error: ${error.message}` });
        currentTaskId = null;
        disableTaskButtons(false);
        updateCancelButtonState(true);
    }
    return false;
}

function disableTaskButtons(isDisabled) {
    const buttons = [startAnalysisBtn, startClusteringBtn, fetchPlaylistsBtn];
    buttons.forEach(button => {
        button.disabled = isDisabled;
        if (isDisabled) {
            button.style.backgroundColor = '#93C5FD'; // Disabled color from CSS
            button.style.cursor = 'not-allowed';
        } else {
            button.style.backgroundColor = '';
            button.style.cursor = '';
        }
    });
}

function displayTaskStatus(task) {
    statusTaskId.textContent = task.task_id || 'N/A';
    statusRunningTime.textContent = formatRunningTime(task.running_time_seconds);
    statusTaskType.textContent = task.task_type_from_db || task.task_type || 'N/A';
    const stateUpper = (task.state || task.status || 'IDLE').toUpperCase();
    statusStatus.textContent = stateUpper;
    statusProgress.textContent = task.progress || 0;
    progressBar.style.width = `${task.progress || 0}%`;

    statusStatus.className = 'status-text'; // Reset classes
    let statusClass = 'status-pending';
    if (['SUCCESS', 'FINISHED'].includes(stateUpper)) {
        statusClass = 'status-success';
    } else if (['FAILURE', 'FAILED', 'REVOKED', 'CANCELED'].includes(stateUpper)) {
        statusClass = 'status-failure';
    } else if (stateUpper === 'IDLE') {
        statusClass = 'status-idle';
    }
    statusStatus.classList.add('status-status', statusClass);


    if (['SUCCESS', 'FINISHED'].includes(stateUpper) && (task.task_type_from_db || task.task_type || '').toLowerCase().includes('clustering')) {
        fetchPlaylists();
    }
    if (['SUCCESS', 'FINISHED'].includes(stateUpper) && (task.task_type_from_db || task.task_type || '').toLowerCase().includes('fetch_playlists')) {
        fetchPlaylists();
    }

    let statusMessage = 'N/A';
    if (task.details) {
        if (task.details.status_message) {
            statusMessage = task.details.status_message;
        } else if (Array.isArray(task.details.log) && task.details.log.length > 0) {
            statusMessage = task.details.log[task.details.log.length - 1];
        } else if (task.details.message) {
            statusMessage = task.details.message;
        }
    }
    statusLog.textContent = statusMessage;


    statusDetails.textContent = typeof task.details === 'object' ? JSON.stringify(task.details, null, 2) : task.details;
    statusDetails.scrollTop = statusDetails.scrollHeight;
}

async function startTask(taskType) {
    disableTaskButtons(true);
    updateCancelButtonState(true);

    // MODIFIED: Removed Jellyfin parameters from the payload.
    // The backend now gets these from its own configuration.
    const payload = {};

    if (taskType === 'analysis') {
        payload.num_recent_albums = parseInt(document.getElementById('config-num_recent_albums').value);
        payload.top_n_moods = parseInt(document.getElementById('config-top_n_moods').value);
    } else if (taskType === 'clustering') {
        playlistsSection.style.display = 'none';
        playlistsContainer.innerHTML = '';

        Object.assign(payload, {
            clustering_method: clusterAlgorithmSelect.value,
            top_n_playlists: parseInt(document.getElementById('config-top_n_playlists').value),
            max_distance: parseFloat(document.getElementById('config-max_distance').value),
            max_songs_per_cluster: parseInt(document.getElementById('config-max_songs_per_cluster').value),
            pca_components_min: parseInt(document.getElementById('config-pca_components_min').value),
            pca_components_max: parseInt(document.getElementById('config-pca_components_max').value),
            clustering_runs: parseInt(document.getElementById('config-clustering_runs').value),
            min_songs_per_genre_for_stratification: parseInt(document.getElementById('config-min_songs_per_genre_for_stratification').value),
            stratified_sampling_target_percentile: parseInt(document.getElementById('config-stratified_sampling_target_percentile').value),
            score_weight_diversity: parseFloat(document.getElementById('config-score_weight_diversity').value),
            score_weight_purity: parseFloat(document.getElementById('config-score_weight_purity').value),
            score_weight_silhouette: parseFloat(document.getElementById('config-score_weight_silhouette').value),
            score_weight_davies_bouldin: parseFloat(document.getElementById('config-score_weight_davies_bouldin').value),
            score_weight_calinski_harabasz: parseFloat(document.getElementById('config-score_weight_calinski_harabasz').value),
            // CORRECTED: Fixed variable names to match the Python config.
            score_weight_other_feature_diversity: parseFloat(document.getElementById('config-score_weight_other_feature_diversity').value),
            score_weight_other_feature_purity: parseFloat(document.getElementById('config-score_weight_other_feature_purity').value),
            dbscan_eps_min: parseFloat(document.getElementById('config-dbscan_eps_min').value),
            dbscan_eps_max: parseFloat(document.getElementById('config-dbscan_eps_max').value),
            dbscan_min_samples_min: parseInt(document.getElementById('config-dbscan_min_samples_min').value),
            dbscan_min_samples_max: parseInt(document.getElementById('config-dbscan_min_samples_max').value),
            num_clusters_min: parseInt(document.getElementById('config-num_clusters_min').value),
            num_clusters_max: parseInt(document.getElementById('config-num_clusters_max').value),
            gmm_n_components_min: parseInt(document.getElementById('config-gmm_n_components_min').value),
            gmm_n_components_max: parseInt(document.getElementById('config-gmm_n_components_max').value),
            spectral_n_clusters_min: parseInt(document.getElementById('config-spectral_n_clusters_min').value),
            spectral_n_clusters_max: parseInt(document.getElementById('config-spectral_n_clusters_max').value),
            ai_model_provider: aiModelProviderSelect.value,
            ollama_server_url: document.getElementById('config-ollama_server_url').value,
            ollama_model_name: document.getElementById('config-ollama_model_name').value,
            openai_server_url: document.getElementById('config-openai_server_url').value,
            openai_model_name: document.getElementById('config-openai_model_name').value,
            gemini_model_name: document.getElementById('config-gemini_model_name').value,
            mistral_model_name: document.getElementById('config-mistral_model_name').value,
            enable_clustering_embeddings: document.getElementById('config-enable_clustering_embeddings').checked
        });
    }

    try {
        let url = startAnalysisEndpointUrl.replace('/analysis/', `/${encodeURIComponent(taskType)}/`);
        if (taskType === 'fetch_playlists') {
            url = startFetchPlaylistsEndpointUrl;
        }

        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const result = await response.json();
        if (response.ok && result.task_id) {
            currentTaskId = result.task_id;
            displayTaskStatus({ task_id: result.task_id, task_type: result.task_type, state: 'PENDING', progress: 0, details: 'Task enqueued.' });
            lastPolledTaskDetails[result.task_id] = { state: 'PENDING', task_type: result.task_type, task_id: result.task_id };
            updateCancelButtonState(false);
        } else {
            throw new Error(result.message || 'Failed to start task.');
        }
    } catch (error) {
        console.error(`Error starting ${taskType} task:`, error);
        showMessageBox('Error', `Failed to start ${taskType} task: ${error.message}`);
        disableTaskButtons(false);
        updateCancelButtonState(true);
    }
}

async function cancelTask() {
    if (!currentTaskId) return;
    updateCancelButtonState(true);
    try {
        const response = await fetch(cancelTaskEndpointUrl.replace(':taskId:', encodeURIComponent(currentTaskId)), { method: 'POST' });
        const result = await response.json();
        if (response.ok) {
            showMessageBox('Success', result.message);
            checkActiveTasks();
        } else {
            throw new Error(result.message || 'Failed to cancel task.');
        }
    } catch (error) {
        console.error('Error cancelling task:', error);
        showMessageBox('Error', `Failed to cancel task: ${error.message}`);
        updateCancelButtonState(false);
    }
}

async function fetchPlaylists() {
    playlistsContainer.innerHTML = '<p>Fetching playlists...</p>';
    playlistsSection.style.display = 'block';
    try {
        const response = await fetch(getPlaylistsEndpointUrl);
        if (!response.ok) throw new Error(`Server responded with ${response.status}`);
        const playlistsData = await response.json();
        renderPlaylists(playlistsData);
    } catch (error) {
        console.error('Error fetching playlists:', error);
        playlistsContainer.innerHTML = `<p class="status-failure">Error fetching playlists: ${error.message}</p>`;
    }
}

function renderPlaylists(playlistsData) {
    playlistsContainer.innerHTML = '';
    if (!playlistsData || Object.keys(playlistsData).length === 0) {
        playlistsContainer.innerHTML = '<p>No playlists found.</p>';
        return;
    }
    for (const [playlistName, songs] of Object.entries(playlistsData)) {
        const playlistDiv = document.createElement('div');
        playlistDiv.className = 'playlist-item';
        playlistDiv.innerHTML = `
            <div class="playlist-header">
                <strong class="playlist-name">${playlistName}</strong>
                <span class="playlist-song-count">(${songs.length} songs)</span>
                <button class="show-songs-btn">SHOW</button>
            </div>
            <ul class="song-list" style="display: none;">
                ${songs.map(song => `<li>${song.title} by ${song.author}</li>`).join('')}
            </ul>`;
        playlistDiv.querySelector('.show-songs-btn').addEventListener('click', e => {
            const btn = e.target;
            const list = btn.closest('.playlist-item').querySelector('.song-list');
            list.style.display = list.style.display === 'none' ? 'block' : 'none';
            btn.textContent = list.style.display === 'none' ? 'SHOW' : 'HIDE';
        });
        playlistsContainer.appendChild(playlistDiv);
    }
}

function showMessageBox(title, message) {
    const boxId = 'custom-message-box';
    document.getElementById(boxId)?.remove();
    const messageBox = document.createElement('div');
    messageBox.id = boxId;
    messageBox.style.cssText = 'position: fixed; top: 20px; right: 20px; background-color: #fff; color: #1F2937; padding: 20px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); z-index: 1000; border: 1px solid #E5E7EB; max-width: 400px; text-align: left;';
    messageBox.innerHTML = `<h3 style="font-weight: 600; margin-top:0; margin-bottom: 10px; color: #111827;">${title}</h3><p style="margin:0;">${message}</p><button style="position: absolute; top: 10px; right: 10px; background: none; border: none; font-size: 1.5rem; color: #9CA3AF; cursor: pointer;" onclick="this.parentNode.remove()">&times;</button>`;

    setTimeout(() => {
        messageBox.remove();
    }, 5000);

    document.body.appendChild(messageBox);
}

async function fetchAndDisplayOverallLastTask() {
    try {
        const response = await fetch(getLastOverallTaskStatusEndpointUrl);
        if (response.ok) {
            const lastTask = await response.json();
            if (lastTask && lastTask.task_id) displayTaskStatus(lastTask);
            else displayTaskStatus({ state: 'IDLE', details: 'No previous task found.' });
        } else {
            displayTaskStatus({ state: 'IDLE', details: 'Could not fetch last task status.' });
        }
    } catch (error) {
        console.error('Error fetching last task status:', error);
        displayTaskStatus({ state: 'IDLE', details: 'Error fetching last task status.' });
    }
}

// --- Event Listeners & Initialization ---
document.addEventListener('DOMContentLoaded', async () => {
    // do work behind a reverse proxy we need to get the URLs resolved set and expect them in the following variables
    console.assert(getActiveTasksEndpointUrl, "Missing getActiveTasksEndpointUrl url variable");
    console.assert(getConfigEndpointUrl, "Missing getConfigEndpointUrl url variable");
    console.assert(getTaskStatusEndpointUrl, "Missing getTaskStatusEndpointUrl url variable");
    console.assert(startAnalysisEndpointUrl, "Missing startAnalysisEndpointUrl url variable");
    console.assert(getLastOverallTaskStatusEndpointUrl, "Missing getLastOverallTaskStatusEndpointUrl url variable");
    console.assert(getPlaylistsEndpointUrl, "Missing getPlaylistsEndpointUrl url variable");
    console.assert(cancelTaskEndpointUrl, "Missing cancelTaskEndpointUrl url variable");
    console.assert(startFetchPlaylistsEndpointUrl, "Missing startFetchPlaylistsEndpointUrl url variable");

    // Attach event listeners to buttons
    startAnalysisBtn.addEventListener('click', () => startTask('analysis'));
    startClusteringBtn.addEventListener('click', () => startTask('clustering'));
    fetchPlaylistsBtn.addEventListener('click', () => startTask('fetch_playlists'));
    cancelTaskBtn.addEventListener('click', cancelTask);

    // View switcher buttons
    basicViewBtn.addEventListener('click', () => switchView('basic'));
    advancedViewBtn.addEventListener('click', () => switchView('advanced'));

    // Algorithm and AI provider change listeners
    clusterAlgorithmSelect.addEventListener('change', toggleClusteringParams);
    aiModelProviderSelect.addEventListener('change', toggleAiConfig);

    await fetchConfig();
    if (!await checkActiveTasks()) {
        await fetchAndDisplayOverallLastTask();
        updateCancelButtonState(true);
    }
    setInterval(checkActiveTasks, 3000);
});


