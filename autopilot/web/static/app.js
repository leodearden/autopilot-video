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

    source.onerror = function() {
        /*
         * NOTE: Events emitted by the server during reconnection are lost.
         * EventSource will auto-reconnect, but there is no replay mechanism.
         * A robust fix would be to track Last-Event-ID and have the server
         * replay missed events on reconnect.
         */
        console.warn('SSE connection lost, will retry...');
    };

    return source;
}

/* Per-stage debounce timers — coalesces burst SSE updates for the same stage */
var _refreshTimers = {};
/* Debounce delay in ms — minimum interval between DOM refreshes for a given stage */
var DEBOUNCE_MS = 150;

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
 * Create an SSE event handler that parses JSON, optionally refreshes the stage card,
 * and optionally shows a toast notification.
 * @param {string} eventType - The SSE event type name (for error logging).
 * @param {string} [toastMsg] - Optional toast message. Use '{stage}' as placeholder.
 * @param {'success'|'error'|'info'} [toastType] - Toast type when toastMsg is provided.
 * @param {number} [toastDuration] - Optional toast duration in ms (defaults to showToast's default).
 * @param {boolean} [refreshCard=true] - Whether to call refreshStageCard. Set false for error-only handlers.
 * @returns {function} Event handler function.
 */
function makeStageHandler(eventType, toastMsg, toastType, toastDuration, refreshCard) {
    if (refreshCard === undefined) refreshCard = true;
    return function(event) {
        try {
            var data = JSON.parse(event.data);
            var stage = data.stage;
            if (stage) {
                if (refreshCard) {
                    refreshStageCard(stage);
                }
                if (toastMsg) {
                    showToast(toastMsg.replace('{stage}', stage), toastType || 'info', toastDuration);
                }
            } else {
                console.warn('SSE ' + eventType + ': missing stage field', data);
                if (toastMsg) {
                    showToast(toastMsg.replace('{stage}', 'unknown'), toastType || 'info', toastDuration);
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

var _permissionRequested = false;

/**
 * Set up SSE event listeners for notification bell updates.
 * Shows the badge when gate_waiting events arrive and hides it
 * when gates are approved or skipped.
 * @param {EventSource} source - The SSE EventSource connection.
 */
function setupNotificationSSE(source) {
    var unreadCount = 0;

    source.addEventListener('notification', function(event) {
        try {
            var data = JSON.parse(event.data);
            if (!data || typeof data !== 'object') {
                console.warn('SSE notification: unexpected payload', event.data);
                return;
            }
            var msg = (data && typeof data.message === 'string') ? data.message : '(no message)';
            unreadCount++;
            updateNotificationBadge(unreadCount);
            showToast(msg, data.type || 'info');

            /* Browser Notification API (best-effort) */
            if (typeof Notification !== 'undefined') {
                if (Notification.permission === 'granted') {
                    new Notification('Autopilot Video', { body: msg });
                } else if (Notification.permission !== 'denied') {
                    if (!_permissionRequested) {
                        _permissionRequested = true;
                        Notification.requestPermission().then(function(perm) {
                            if (perm === 'granted') {
                                new Notification('Autopilot Video', { body: msg });
                            }
                        });
                    }
                }
            }
        } catch (e) {
            console.error('SSE notification parse error:', e, event.data);
        }
    });

    /* Bell click clears unread count */
    var bell = document.getElementById('notification-bell');
    if (bell) {
        bell.addEventListener('click', function() {
            unreadCount = 0;
            updateNotificationBadge(0);
        });
    }
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
