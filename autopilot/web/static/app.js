/* Autopilot Video Console — client-side JavaScript */

/**
 * Display a toast notification.
 * @param {string} message - The message to display.
 * @param {'success'|'error'|'info'} type - The toast type.
 * @param {number} duration - Duration in milliseconds before auto-dismiss.
 */
function showToast(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s ease-out';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

/**
 * SSE connection handler stub.
 * Connects to the server-sent events endpoint for real-time updates.
 * @param {string} url - The SSE endpoint URL.
 */
function connectSSE(url) {
    if (!url) url = '/api/events';

    const source = new EventSource(url);

    source.addEventListener('notification', function(event) {
        try {
            const data = JSON.parse(event.data);
            showToast(data.message, data.type || 'info');
        } catch (e) {
            console.error('SSE notification parse error:', e, event.data);
        }
    });

    source.onerror = function() {
        console.warn('SSE connection lost, will retry...');
    };

    return source;
}

/**
 * Refresh a single stage card via HTMX ajax.
 * @param {string} stage - The pipeline stage name.
 */
function refreshStageCard(stage) {
    if (typeof htmx !== 'undefined') {
        htmx.ajax('GET', '/dashboard/stage/' + stage, {
            target: '#stage-' + stage,
            swap: 'outerHTML'
        });
    }
}

/**
 * Create an SSE event handler that parses JSON, refreshes the stage card,
 * and optionally shows a toast notification.
 * @param {string} eventType - The SSE event type name (for error logging).
 * @param {string} [toastMsg] - Optional toast message. Use '{stage}' as placeholder.
 * @param {'success'|'error'|'info'} [toastType] - Toast type when toastMsg is provided.
 * @returns {function} Event handler function.
 */
function makeStageHandler(eventType, toastMsg, toastType) {
    return function(event) {
        try {
            var data = JSON.parse(event.data);
            var stage = data.stage;
            if (stage) {
                refreshStageCard(stage);
                if (toastMsg) {
                    showToast(toastMsg.replace('{stage}', stage), toastType || 'info');
                }
            } else {
                console.warn('SSE ' + eventType + ': missing stage field', data);
                if (toastMsg) {
                    showToast(toastMsg.replace('{stage}', 'unknown'), toastType || 'info');
                }
            }
        } catch (e) {
            console.error('SSE ' + eventType + ' parse error:', e);
        }
    };
}

/**
 * Set up SSE event listeners for dashboard stage card updates.
 * @param {EventSource} source - The SSE EventSource connection.
 */
function setupDashboardSSE(source) {
    source.addEventListener('stage_started', makeStageHandler('stage_started'));
    source.addEventListener('stage_completed', makeStageHandler('stage_completed'));
    source.addEventListener('job_progress', makeStageHandler('job_progress'));
}

/**
 * Update the notification bell badge visibility and count.
 * @param {number} count - Number of pending notifications. 0 hides the badge.
 */
function updateNotificationBadge(count) {
    var badge = document.getElementById('notification-badge');
    if (!badge) return;
    if (count > 0) {
        badge.textContent = count > 9 ? '9+' : String(count);
        badge.classList.remove('hidden');
    } else {
        badge.classList.add('hidden');
    }
}

/**
 * Set up SSE event listeners for notification bell updates.
 * Shows the badge when gate_waiting events arrive and hides it
 * when gates are approved or skipped.
 * @param {EventSource} source - The SSE EventSource connection.
 */
function setupNotificationSSE(source) {
    var pendingCount = 0;

    source.addEventListener('gate_waiting', function(event) {
        pendingCount++;
        updateNotificationBadge(pendingCount);
        try {
            var data = JSON.parse(event.data);
            showToast('Gate waiting: ' + (data.stage || 'unknown'), 'info');
        } catch (e) {
            console.error('SSE gate_waiting parse error:', e);
        }
    });

    source.addEventListener('gate_approved', function() {
        pendingCount = Math.max(0, pendingCount - 1);
        updateNotificationBadge(pendingCount);
    });

    source.addEventListener('gate_skipped', function() {
        pendingCount = Math.max(0, pendingCount - 1);
        updateNotificationBadge(pendingCount);
    });

    source.addEventListener('stage_error', makeStageHandler('stage_error', 'Error in {stage} stage', 'error'));

    source.addEventListener('run_completed', function(event) {
        try {
            showToast('Pipeline run completed!', 'success', 6000);
        } catch (e) {
            console.error('SSE run_completed error:', e);
        }
    });

    source.addEventListener('run_failed', function(event) {
        try {
            showToast('Pipeline run failed', 'error', 8000);
        } catch (e) {
            console.error('SSE run_failed error:', e);
        }
    });
}

/* Initialize on DOM ready */
document.addEventListener('DOMContentLoaded', function() {
    console.log('Autopilot Video Console loaded');

    /* Connect SSE on all pages for notifications */
    var source = connectSSE('/api/events');
    setupNotificationSSE(source);

    /* If the dashboard grid is present, also wire up stage card updates */
    if (document.getElementById('dashboard-grid')) {
        setupDashboardSSE(source);
    }
});
