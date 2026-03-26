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

/* Per-stage debounce timer map and delay for coalescing burst SSE updates */
var _refreshTimers = {};
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
 * Debounced wrapper around refreshStageCard.
 * Coalesces burst SSE events for the same stage so only the trailing
 * event triggers the actual HTMX refresh.
 * @param {string} stage - The pipeline stage name.
 */
function debouncedRefreshStageCard(stage) {
    if (_refreshTimers[stage]) {
        clearTimeout(_refreshTimers[stage]);
    }
    _refreshTimers[stage] = setTimeout(function() {
        delete _refreshTimers[stage];
        refreshStageCard(stage);
    }, DEBOUNCE_MS);
}

/**
 * Set up SSE event listeners for dashboard stage card updates.
 * @param {EventSource} source - The SSE EventSource connection.
 */
function setupDashboardSSE(source) {
    source.addEventListener('stage_started', function(event) {
        try {
            var data = JSON.parse(event.data);
            var stage = data.stage;
            if (stage) {
                refreshStageCard(stage);
            }
        } catch (e) {
            console.error('SSE stage_started parse error:', e);
        }
    });

    source.addEventListener('stage_completed', function(event) {
        try {
            var data = JSON.parse(event.data);
            var stage = data.stage;
            if (stage) {
                refreshStageCard(stage);
            }
        } catch (e) {
            console.error('SSE stage_completed parse error:', e);
        }
    });

    source.addEventListener('job_progress', function(event) {
        try {
            var data = JSON.parse(event.data);
            var stage = data.stage;
            if (stage) {
                refreshStageCard(stage);
            }
        } catch (e) {
            console.error('SSE job_progress parse error:', e);
        }
    });

    source.addEventListener('gate_waiting', function(event) {
        try {
            var data = JSON.parse(event.data);
            var stage = data.stage;
            if (stage) {
                refreshStageCard(stage);
                showToast('Pipeline paused at ' + stage + '. Review required.', 'info');
            }
        } catch (e) {
            console.error('SSE gate_waiting parse error:', e);
        }
    });

    source.addEventListener('gate_approved', function(event) {
        try {
            var data = JSON.parse(event.data);
            var stage = data.stage;
            if (stage) {
                refreshStageCard(stage);
                showToast('Gate approved: ' + stage, 'success');
            }
        } catch (e) {
            console.error('SSE gate_approved parse error:', e);
        }
    });

    source.addEventListener('gate_skipped', function(event) {
        try {
            var data = JSON.parse(event.data);
            var stage = data.stage;
            if (stage) {
                refreshStageCard(stage);
                showToast('Gate skipped: ' + stage, 'info');
            }
        } catch (e) {
            console.error('SSE gate_skipped parse error:', e);
        }
    });

    source.addEventListener('stage_error', function(event) {
        try {
            var data = JSON.parse(event.data);
            var stage = data.stage;
            if (stage) {
                refreshStageCard(stage);
                showToast('Stage error: ' + stage, 'error');
            }
        } catch (e) {
            console.error('SSE stage_error parse error:', e);
        }
    });

    source.addEventListener('run_completed', function(event) {
        showToast('Pipeline run completed successfully.', 'success');
    });

    source.addEventListener('run_failed', function(event) {
        showToast('Pipeline run failed.', 'error');
    });
}

/* Initialize on DOM ready */
document.addEventListener('DOMContentLoaded', function() {
    console.log('Autopilot Video Console loaded');

    /* If the dashboard grid is present, wire up SSE for live updates */
    if (document.getElementById('dashboard-grid')) {
        var source = connectSSE('/api/events');
        setupDashboardSSE(source);
    }
});
