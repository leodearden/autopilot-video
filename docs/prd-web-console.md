# PRD: Autopilot Video — Web Management Console

## Product Overview

A self-hosted web console that provides real-time visibility and control over the Autopilot Video pipeline. The console replaces the current CLI-only workflow with a browser-based interface where the creator can monitor pipeline progress, browse cataloged media and analysis results, review and approve LLM-generated artifacts (narratives, scripts, edit plans), and configure per-stage gate controls that determine whether the pipeline runs autonomously or pauses for human intervention.

### Why

The current pipeline operates as a single `autopilot run` command with a CLI-based human review checkpoint at the NARRATE stage. This has three problems:

1. **No visibility during long runs.** The 16–20 hour analysis phase produces no output the creator can inspect until it completes. If something goes wrong at hour 12, the creator doesn't know until hour 16.

2. **Coarse-grained intervention.** The only intervention point is narrative approval. But the creator may want to review activity classifications before they become narratives, or review scripts before EDL generation, or inspect edit plans before committing to a multi-hour render. Today this requires stopping the pipeline, manually querying SQLite, and restarting.

3. **No asset or job overview.** There's no way to see which media files have been analyzed, what the analysis found, which narratives are in progress, or what the render queue looks like — without writing SQL queries.

### Hardware & Deployment Target

The console runs on the same machine as the pipeline (RTX 3090 workstation). It is a single-user application — no multi-tenancy, no authentication (bound to localhost by default). The console must not compete with the pipeline for GPU resources; it is CPU/IO only.

### Key Constraints

- **Zero added cost.** No SaaS dependencies, no cloud hosting, no paid UI frameworks.
- **Minimal footprint.** The console should not add significant complexity to the project. It is a tool for the pipeline, not a product in its own right.
- **Non-blocking.** The console must never block pipeline execution. Reads are against the SQLite WAL snapshot; writes (approvals, gate changes) go through a coordination layer that the orchestrator polls.
- **Works unattended.** If the creator never opens the console, the pipeline should still run to completion (respecting the configured gate defaults). The console enhances the workflow; it doesn't gate it by default.

---

## Design Principles

1. **Read from the catalog, write to the gate.** The console reads all display data directly from the SQLite catalog (WAL mode supports concurrent readers). The only writes are gate decisions (approve/reject/skip) and gate configuration changes, which go through a lightweight coordination mechanism (a `pipeline_control` table or a small JSON file the orchestrator polls).

2. **Progressive disclosure.** The top-level view is a pipeline status board showing stage progress. Drilling down reveals per-stage details: media files in INGEST, analysis results in ANALYZE, clusters in CLASSIFY, etc. The creator should never need to write SQL.

3. **Gate-first interaction model.** Every stage transition has a configurable gate: `auto` (proceed without waiting), `pause` (wait for explicit approval), or `notify` (proceed but send a notification). Gates are the primary control surface — they turn the pipeline from fully-automated to human-in-the-loop at any granularity.

4. **Offline-first rendering.** The console renders all views server-side or with minimal client JS. No heavy frontend framework. The pipeline workstation may not have internet access during a run, and the creator shouldn't need to install Node.js or run a build step.

---

## Architecture

### Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| HTTP server | **FastAPI** | Already a Python project; async support for SSE; no new language runtime |
| Templates | **Jinja2** | Server-rendered HTML; ships with FastAPI/Starlette |
| Interactivity | **HTMX** + minimal vanilla JS | Partial page updates without a JS framework; < 14 KB |
| Styling | **Tailwind CSS** (CDN or pre-built) | Utility-first; no build step if using CDN play |
| Real-time updates | **Server-Sent Events (SSE)** | Unidirectional push from server; simpler than WebSocket for status updates |
| Data | **SQLite** (existing catalog) | Read directly from existing tables; no data duplication |
| Coordination | **`pipeline_gates` table** in SQLite | Orchestrator polls for gate decisions; console writes approvals |

### New Components

```
autopilot/
├── web/
│   ├── __init__.py
│   ├── app.py              # FastAPI application factory
│   ├── routes/
│   │   ├── dashboard.py    # Pipeline overview, stage status
│   │   ├── media.py        # Media file browser, analysis results
│   │   ├── pipeline.py     # Stage detail views, job tracking
│   │   ├── review.py       # Gate review UI (approve/reject artifacts)
│   │   ├── gates.py        # Gate configuration API
│   │   ├── narratives.py   # Narrative/script/EDL detail views
│   │   └── sse.py          # Server-Sent Events endpoint
│   ├── templates/
│   │   ├── base.html       # Layout, nav, HTMX/Tailwind includes
│   │   ├── dashboard.html  # Pipeline board
│   │   ├── media/
│   │   │   ├── list.html   # Media file table with filters
│   │   │   └── detail.html # Single file: metadata + analysis
│   │   ├── pipeline/
│   │   │   ├── stage.html  # Per-stage detail (jobs, progress)
│   │   │   └── jobs.html   # Job list with status
│   │   ├── review/
│   │   │   ├── clusters.html    # Activity cluster review
│   │   │   ├── narratives.html  # Narrative proposal review
│   │   │   ├── scripts.html     # Script review + edit
│   │   │   ├── edl.html         # EDL review + timeline viz
│   │   │   └── renders.html     # Render preview + approval
│   │   ├── gates/
│   │   │   └── config.html # Gate toggle panel
│   │   └── partials/       # HTMX fragment templates
│   │       ├── stage_card.html
│   │       ├── job_row.html
│   │       ├── media_row.html
│   │       └── gate_toggle.html
│   └── static/
│       ├── app.css         # Minimal custom styles
│       └── app.js          # Minimal JS (SSE handler, timeline viz)
```

### Database Additions

Two new tables in the existing catalog database:

```sql
-- Gate configuration and state
CREATE TABLE pipeline_gates (
    stage TEXT PRIMARY KEY,           -- INGEST, ANALYZE, CLASSIFY, etc.
    mode TEXT DEFAULT 'auto',         -- auto | pause | notify
    status TEXT DEFAULT 'idle',       -- idle | waiting | approved | rejected | skipped
    decided_at TEXT,                  -- ISO 8601 timestamp of last decision
    decided_by TEXT DEFAULT 'system', -- system | console | cli
    notes TEXT                        -- Optional reviewer notes
);

-- Job-level tracking (sub-stage granularity)
CREATE TABLE pipeline_jobs (
    job_id TEXT PRIMARY KEY,          -- UUID
    stage TEXT NOT NULL,              -- Parent pipeline stage
    job_type TEXT NOT NULL,           -- asr | yolo | face | embed | cluster | narrate | script | edl | render_clip | upload
    target_id TEXT,                   -- media_id, narrative_id, or cluster_id being processed
    target_label TEXT,                -- Human-readable label (filename, narrative title)
    status TEXT DEFAULT 'pending',    -- pending | running | done | error | skipped
    started_at TEXT,
    finished_at TEXT,
    duration_seconds REAL,
    progress_pct REAL,               -- 0.0–100.0 for long-running jobs
    error_message TEXT,
    worker TEXT                       -- gpu | cpu-0 | cpu-1 | ... for resource tracking
);

-- Pipeline run metadata
CREATE TABLE pipeline_runs (
    run_id TEXT PRIMARY KEY,          -- UUID
    started_at TEXT NOT NULL,
    finished_at TEXT,
    config_snapshot TEXT,             -- YAML config at run start
    current_stage TEXT,               -- Currently executing stage
    status TEXT DEFAULT 'running',    -- running | paused | completed | failed | cancelled
    wall_clock_seconds REAL,
    budget_remaining_seconds REAL
);
```

### Orchestrator Integration

The existing `PipelineOrchestrator` needs modifications to:

1. **Write job records.** Before and after each sub-task (per-file analysis, per-narrative scripting, per-clip rendering), insert/update rows in `pipeline_jobs`.

2. **Check gates.** Before transitioning between stages, query `pipeline_gates` for the next stage. If `mode = 'pause'`, set `status = 'waiting'` and poll until it changes to `approved` or `skipped`. If `mode = 'auto'`, proceed immediately. If `mode = 'notify'`, proceed but emit an SSE event.

3. **Emit events.** Write stage/job transitions to a lightweight event stream (append-only table or in-memory queue) that the SSE endpoint reads.

```python
# New method on PipelineOrchestrator
async def _check_gate(self, stage: str) -> str:
    """Check if a gate allows proceeding. Returns 'approved' or 'skipped'."""
    gate = self.db.get_gate(stage)
    if gate.mode == 'auto':
        self.db.update_gate(stage, status='approved', decided_by='system')
        return 'approved'
    if gate.mode == 'notify':
        self._emit_event('gate_passed', stage=stage)
        self.db.update_gate(stage, status='approved', decided_by='system')
        return 'approved'
    # mode == 'pause'
    self.db.update_gate(stage, status='waiting')
    self._emit_event('gate_waiting', stage=stage)
    while True:
        gate = self.db.get_gate(stage)
        if gate.status in ('approved', 'skipped'):
            return gate.status
        await asyncio.sleep(2)  # Poll interval
```

---

## Features

### 1. Pipeline Dashboard

The primary view. Shows the full pipeline as a horizontal stage board with real-time status.

#### Stage Cards

Each of the 9 pipeline stages is represented as a card showing:

| Element | Content |
|---------|---------|
| **Stage name** | INGEST, ANALYZE, CLASSIFY, NARRATE, SCRIPT, EDL, SOURCE_ASSETS, RENDER, UPLOAD |
| **Status indicator** | Color-coded: grey (idle), blue (running), amber (waiting at gate), green (complete), red (error) |
| **Progress** | For multi-job stages: "47/128 files" or "3/5 narratives" with progress bar |
| **Elapsed / ETA** | Wall-clock time spent in this stage; estimated time remaining based on throughput |
| **Gate badge** | Shows current gate mode (auto/pause/notify) with quick-toggle |
| **Error count** | Red badge if any jobs in this stage errored |

#### Pipeline Timeline

Below the stage cards, a horizontal timeline bar shows:

- Overall pipeline progress as percentage of estimated total time
- Budget consumed vs. remaining (out of 36-hour budget)
- Current run duration
- Time-of-day estimate for completion

#### Live Updates

The dashboard subscribes to an SSE endpoint. Events trigger HTMX partial swaps to update individual stage cards without full page reload. Event types:

- `stage_started` / `stage_completed` / `stage_error`
- `job_started` / `job_completed` / `job_error` / `job_progress`
- `gate_waiting` / `gate_approved` / `gate_skipped`
- `run_completed` / `run_failed`

### 2. Media Browser

Browse all ingested media files with rich filtering and analysis drill-down.

#### List View

A sortable, filterable table of all media files:

| Column | Content | Filter/Sort |
|--------|---------|-------------|
| Thumbnail | First frame (extracted during ingest) | — |
| Filename | Original filename | Text search |
| Duration | `HH:MM:SS` | Sort, range filter |
| Resolution | `4096×4096` | Filter by resolution class |
| Created | Date/time from EXIF | Date range picker, sort |
| Location | Reverse-geocoded label or GPS coords | Text search |
| Status | `ingested` / `analyzing` / `analyzed` / `error` | Filter |
| Analysis | Icon badges for completed analyses (ASR, YOLO, faces, embeddings, audio) | Filter by "has transcript", "has detections", etc. |

#### Detail View

Clicking a media file opens a detail page with tabs:

**Metadata tab:**
- Full technical metadata (codec, FPS, audio channels, GPS, etc.)
- File path, SHA-256 prefix, dedup status

**Transcript tab:**
- Scrollable transcript with timestamps and speaker labels
- Click a timestamp to seek the video preview (if preview generation is enabled)

**Detections tab:**
- Summary: top object classes detected, total tracks, frame coverage
- Expandable list of tracked objects (track_id, class, first/last frame, bbox count)
- Optionally: a few sample frames with detection bboxes overlaid (pre-rendered thumbnails)

**Faces tab:**
- Grid of detected face crops grouped by cluster
- Cluster label (if assigned) with link to face cluster management

**Audio Events tab:**
- Timeline visualization of audio event classifications
- Dominant event per time window, filterable by event class

**Embeddings tab:**
- Embedding coverage (frames sampled out of total)
- Nearest-neighbor query: "find similar frames across all media" (runs a FAISS search on demand)

### 3. Pipeline Stage Detail Views

Each stage has a dedicated detail page accessible from the dashboard.

#### INGEST Detail
- File discovery progress (files found / scanned / inserted)
- Deduplication results (duplicates found, bytes saved)
- Audio normalization status per file
- Error log for files that failed to ingest

#### ANALYZE Detail
- Sub-stage breakdown: ASR, scenes, objects, faces, embeddings, audio events, captions
- Per-sub-stage: job list showing each media file's analysis status
- GPU utilization indicator (which model is loaded, VRAM usage)
- Throughput metrics (FPS processed, real-time multiple for ASR)

#### CLASSIFY Detail
- Activity cluster list with:
  - Label, description, time range, location
  - Number of clips in cluster
  - Expandable clip list with thumbnails
- Cluster merge/split controls (if gate is set to `pause`)

#### NARRATE Detail
- Master storyboard preview (the structured text sent to the LLM)
- Proposed narratives with approval controls (see Review section)
- LLM prompt/response log (collapsible, for debugging prompt quality)

#### SCRIPT Detail
- Per-narrative script viewer
- Scene list with shot type, voiceover text, music mood, visual direction
- Script edit capability (if gate is set to `pause`)

#### EDL Detail
- Per-narrative EDL viewer
- Visual timeline representation (see Timeline Visualization)
- Validation results (pass/fail per check, with details on failures)
- OTIO export download link

#### SOURCE_ASSETS Detail
- Asset resolution status: resolved (with file path), pending, failed
- Music tracks: generated (MusicGen) vs. sourced vs. fetch-list
- Voiceover clips: generated audio with playback
- B-roll: sourced clips with preview
- Fetch list: unresolved assets requiring manual action

#### RENDER Detail
- Render queue: clips sorted by routing (fast-path FFmpeg vs. slow-path MoviePy)
- Per-clip progress: pending / encoding / done / error
- Rendered output preview (video player for completed renders)
- Quality validation results per narrative

#### UPLOAD Detail
- Upload queue with status per narrative
- YouTube video IDs and URLs for completed uploads
- Privacy status and metadata sent

### 4. Review & Approval Interface

The core human-in-the-loop feature. When a gate is set to `pause`, the pipeline halts at that stage transition and the review interface presents the stage's output for approval.

#### Review: Activity Clusters (CLASSIFY → NARRATE gate)

**What the creator sees:**
- All activity clusters in a card grid
- Each card shows: label, description, time range, location, clip count, representative thumbnail
- Expandable: full clip list with thumbnails and durations

**Actions:**
- **Approve all** — proceed to narrative generation with these clusters
- **Merge clusters** — select 2+ clusters to combine (drag-and-drop or checkbox)
- **Split cluster** — select a cluster and specify a split point (temporal or manual clip assignment)
- **Relabel** — edit a cluster's label or description
- **Exclude cluster** — remove a cluster from narrative consideration (clips remain in catalog)
- **Approve with changes** — apply modifications and proceed

#### Review: Narratives (NARRATE → SCRIPT gate)

**What the creator sees:**
- Proposed narratives in a list, each showing:
  - Title, description, proposed duration
  - Emotional arc notes
  - Activity clusters included
  - Status: proposed / approved / rejected

**Actions:**
- **Approve** / **Reject** per narrative (existing functionality, now in a UI instead of CLI)
- **Edit** — modify title, description, duration target, or included clusters
- **Reorder** — drag to set render priority
- **Request regeneration** — send back to LLM with notes ("combine the hiking narratives", "shorter duration target")
- **Approve selected** — approve checked narratives and proceed

#### Review: Scripts (SCRIPT → EDL gate)

**What the creator sees:**
- Per-narrative script displayed as an ordered scene list
- Each scene shows: scene number, shot type (interview/broll/music), estimated duration, voiceover text, visual direction, music mood

**Actions:**
- **Approve** — proceed to EDL generation with this script
- **Edit inline** — modify voiceover text, scene order (drag), shot type, music mood
- **Delete scene** — remove a scene from the script
- **Add scene** — insert a new scene (manual entry or LLM-assisted)
- **Regenerate** — send back to LLM with notes
- **Preview storyboard** — see the source material (clips, transcripts) that this script draws from

#### Review: Edit Plans (EDL → SOURCE_ASSETS gate)

**What the creator sees:**
- Timeline visualization of the EDL (see Timeline Visualization below)
- Validation report summary (pass/fail)
- Per-clip details: source file, in/out timecodes, crop mode, transitions
- Audio track layout: music, voiceover, source audio with levels

**Actions:**
- **Approve** — proceed to asset sourcing and rendering
- **Adjust clip** — change in/out points, crop mode, transition type
- **Remove clip** — delete a clip from the timeline
- **Swap clip** — replace a clip with an alternative from the same activity cluster
- **Re-validate** — run validation checks after manual edits
- **Download OTIO** — export for review in DaVinci Resolve before approving
- **Regenerate** — send back to LLM with notes

#### Review: Rendered Output (RENDER → UPLOAD gate)

**What the creator sees:**
- Video player with the rendered output
- Quality validation results
- Side-by-side: script scene list next to timeline position indicator

**Actions:**
- **Approve for upload** — proceed to YouTube upload
- **Reject** — flag for re-render (with notes on what to fix)
- **Download** — get the rendered file without uploading
- **Skip upload** — mark as complete but don't upload

### 5. Gate Configuration Panel

A dedicated settings page for configuring stage gates.

#### Gate Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| **Auto** | Pipeline proceeds through this stage transition without pausing | Stages the creator trusts (e.g., INGEST → ANALYZE) |
| **Pause** | Pipeline halts and waits for explicit approval via the console | Stages where the creator wants review (e.g., NARRATE → SCRIPT) |
| **Notify** | Pipeline proceeds but sends a browser notification and logs the transition | Stages the creator wants awareness of but doesn't want to block on |

#### Gate Configuration UI

A vertical list of all stage transitions with toggle switches:

```
INGEST    ──[auto ▼]──▶  ANALYZE
ANALYZE   ──[auto ▼]──▶  CLASSIFY
CLASSIFY  ──[pause ▼]──▶  NARRATE      ← "Review activity clusters before narrative generation"
NARRATE   ──[pause ▼]──▶  SCRIPT       ← "Review narrative proposals before scripting"
SCRIPT    ──[pause ▼]──▶  EDL          ← "Review scripts before EDL generation"
EDL       ──[auto ▼]──▶  SOURCE_ASSETS
SOURCE    ──[pause ▼]──▶  RENDER       ← "Review edit plans + resolved assets before render"
RENDER    ──[pause ▼]──▶  UPLOAD       ← "Review rendered output before publishing"
```

#### Presets

Quick-apply gate presets:

| Preset | Description | Gates set to `pause` |
|--------|-------------|---------------------|
| **Full Auto** | Run everything without stopping | None |
| **Review Creative** | Pause at creative decision points | CLASSIFY→NARRATE, NARRATE→SCRIPT, RENDER→UPLOAD |
| **Review Everything** | Pause at every transition | All |
| **Review Before Render** | Let planning run, pause before committing GPU time | EDL→SOURCE, RENDER→UPLOAD |

#### Gate Timeout

Optional per-gate timeout: if no decision is made within N hours, auto-approve and proceed. This prevents the pipeline from stalling indefinitely if the creator forgets to check the console. Default: no timeout (wait forever). Recommended: 4 hours for non-critical gates.

### 6. Job Tracking

Fine-grained tracking of individual work items within each stage.

#### Job List View

A filterable table of all jobs across the current pipeline run:

| Column | Content |
|--------|---------|
| Job ID | Short UUID |
| Stage | Parent pipeline stage |
| Type | `asr`, `yolo`, `face`, `embed`, `cluster`, `narrate`, `render_clip`, etc. |
| Target | Media filename or narrative title |
| Status | `pending` / `running` / `done` / `error` / `skipped` |
| Duration | Elapsed time (running) or total time (done) |
| Progress | Percentage bar for long-running jobs |
| Worker | `gpu`, `cpu-0`, `cpu-1`, etc. |
| Error | Error message (if failed), expandable |

#### Job Filters

- By stage (dropdown)
- By status (multi-select: show only errors, show only running)
- By type (multi-select)
- Text search on target label

#### Error Drilldown

Clicking an errored job shows:
- Full error traceback
- Job parameters (which file, what model config)
- Retry button (re-queues the job)
- Skip button (marks as skipped, pipeline continues)

### 7. Timeline Visualization

A lightweight visual timeline for EDL review. Not a full NLE — just enough to understand the edit structure.

#### Layout

- Horizontal timeline with time ruler (minutes:seconds)
- Multiple tracks stacked vertically:
  - **V1**: Primary video clips (colored blocks showing source file, with in/out timecodes)
  - **V2**: B-roll overlay clips (if any)
  - **A1**: Source audio (waveform outline)
  - **A2**: Music track (colored block with label)
  - **A3**: Voiceover (colored block with text preview)
  - **Subtitles**: Caption text markers

#### Clip Blocks

Each clip block shows:
- Source filename (truncated)
- Duration
- Crop mode indicator (icon: center/auto/stabilize)
- Transition type at edges (cut/crossfade)
- Color coding by source file for visual grouping

#### Interactions

- Hover a clip block to see full metadata (source path, timecodes, crop settings)
- Click to open source media detail page
- Zoom in/out on timeline (scroll wheel or buttons)
- No drag-and-drop editing — the timeline is read-only for visualization; edits go through the review action buttons

#### Implementation

Render the timeline as SVG or HTML5 Canvas. Given the "no heavy framework" constraint, a server-rendered SVG with HTMX-triggered tooltip overlays is the simplest approach. For a richer experience, a ~200-line vanilla JS Canvas renderer handles zoom/pan.

### 8. Notifications

Browser notifications and in-console alerts for key events.

| Event | Notification |
|-------|-------------|
| Gate waiting | "Pipeline paused at NARRATE → SCRIPT. Review required." (links to review page) |
| Stage error | "ANALYZE failed: YOLO out of memory on file X." (links to error detail) |
| Pipeline complete | "Pipeline finished. 4 narratives rendered. 1 pending upload." |
| Gate timeout approaching | "CLASSIFY gate will auto-approve in 30 minutes." |
| Long-running job stalled | "ASR job for file X has been running for 2 hours (expected: 20 min)." |

Notifications use the browser Notification API (requires one-time permission grant) and are also shown as toast messages in the console UI. A notification bell icon in the nav bar shows unread count.

---

## Pages & Navigation

### Navigation Structure

```
┌─ Dashboard          (pipeline overview, stage board)
├─ Media              (media file browser)
│   └─ Media Detail   (per-file analysis)
├─ Pipeline
│   ├─ Stage Detail   (per-stage jobs and progress)
│   └─ Jobs           (cross-stage job list)
├─ Review             (pending gate reviews — only shown when gates are waiting)
│   ├─ Clusters
│   ├─ Narratives
│   ├─ Scripts
│   ├─ Edit Plans
│   └─ Renders
├─ Gates              (gate configuration panel)
└─ Settings           (console preferences, notification config)
```

### URL Scheme

```
/                           → Dashboard
/media                      → Media list
/media/{media_id}           → Media detail
/pipeline/{stage}           → Stage detail
/pipeline/jobs              → All jobs
/pipeline/jobs/{job_id}     → Job detail
/review                     → Review hub (shows what's waiting)
/review/clusters            → Cluster review
/review/narratives          → Narrative review
/review/scripts/{id}        → Script review for narrative
/review/edl/{id}            → EDL review for narrative
/review/render/{id}         → Render review for narrative
/gates                      → Gate configuration
/api/gates/{stage}          → Gate API (GET/PUT)
/api/events                 → SSE event stream
/api/jobs/{job_id}/retry    → Retry a failed job
/api/jobs/{job_id}/skip     → Skip a failed job
```

---

## API Endpoints

### REST API

All API endpoints return JSON. The HTML pages are server-rendered and use HTMX to call these endpoints for dynamic updates.

```
GET  /api/run                        → Current pipeline run status
GET  /api/run/history                → Past pipeline runs

GET  /api/stages                     → All stages with status summary
GET  /api/stages/{stage}             → Stage detail with job counts

GET  /api/jobs                       → Job list (filterable: ?stage=&status=&type=)
GET  /api/jobs/{job_id}              → Job detail
POST /api/jobs/{job_id}/retry        → Re-queue a failed job
POST /api/jobs/{job_id}/skip         → Skip a failed job

GET  /api/media                      → Media list (filterable, paginated: ?status=&q=&page=)
GET  /api/media/{media_id}           → Media detail with all analysis
GET  /api/media/{media_id}/transcript → Transcript segments
GET  /api/media/{media_id}/detections → Detection summary

GET  /api/clusters                   → Activity cluster list
GET  /api/clusters/{cluster_id}      → Cluster detail with clips
POST /api/clusters/merge             → Merge clusters (body: {cluster_ids: [...]})
POST /api/clusters/{id}/relabel      → Update cluster label/description

GET  /api/narratives                 → Narrative list
GET  /api/narratives/{id}            → Narrative detail
POST /api/narratives/{id}/approve    → Approve narrative
POST /api/narratives/{id}/reject     → Reject narrative
PUT  /api/narratives/{id}            → Edit narrative

GET  /api/scripts/{narrative_id}     → Script for narrative
PUT  /api/scripts/{narrative_id}     → Update script
POST /api/scripts/{narrative_id}/regenerate → Re-run LLM script generation

GET  /api/edl/{narrative_id}         → EDL for narrative
PUT  /api/edl/{narrative_id}         → Update EDL
POST /api/edl/{narrative_id}/validate → Run validation
GET  /api/edl/{narrative_id}/otio    → Download OTIO file

GET  /api/renders/{narrative_id}     → Render status and output path
GET  /api/renders/{narrative_id}/video → Stream rendered video file

GET  /api/uploads                    → Upload status list

GET  /api/gates                      → All gate configurations
GET  /api/gates/{stage}              → Gate for specific stage
PUT  /api/gates/{stage}              → Update gate mode/status
POST /api/gates/{stage}/approve      → Approve a waiting gate
POST /api/gates/{stage}/skip         → Skip a waiting gate
PUT  /api/gates/preset/{preset}      → Apply a gate preset

GET  /api/events                     → SSE event stream
```

---

## Data Flow & Coordination

### Read Path (Console → SQLite)

```
Browser ──HTTP──▶ FastAPI ──read──▶ SQLite (WAL snapshot)
                     │
                     └── SSE push ◀── Event queue ◀── Orchestrator writes
```

All data displayed in the console is read from existing catalog tables. No data duplication. The console opens a read-only SQLite connection (or a connection that only writes to `pipeline_gates`).

### Write Path (Console → Orchestrator)

```
Browser ──HTTP POST──▶ FastAPI ──write──▶ pipeline_gates table
                                              │
                                              ▼
                                     Orchestrator polls
                                     (2-second interval)
```

The only writes from the console are:
1. Gate decisions (approve/skip/reject) → `pipeline_gates.status`
2. Gate configuration changes (auto/pause/notify) → `pipeline_gates.mode`
3. Artifact edits (narrative title, script text, cluster labels) → existing catalog tables
4. Job retry/skip → `pipeline_jobs.status`

The orchestrator polls `pipeline_gates` when it reaches a gate. It polls `pipeline_jobs` for retry/skip signals during error handling. This polling approach avoids the need for IPC, message queues, or shared-memory coordination between the console server and the orchestrator process.

### Event Flow (Orchestrator → Console)

```
Orchestrator ──insert──▶ pipeline_events table
                              │
               FastAPI SSE ◀──┘ (polls or uses SQLite update hook)
                   │
                   ▼
              Browser (EventSource)
                   │
                   ▼
              HTMX swap (partial page update)
```

New table for the event stream:

```sql
CREATE TABLE pipeline_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,        -- stage_started, job_completed, gate_waiting, etc.
    stage TEXT,
    job_id TEXT,
    payload_json TEXT,               -- Event-specific data
    created_at TEXT DEFAULT (datetime('now'))
);
```

The SSE endpoint tracks its last-seen `event_id` and polls for new events every 1 second. Events older than 24 hours are automatically pruned.

---

## Non-Functional Requirements

### Performance

| Metric | Target |
|--------|--------|
| Dashboard page load | < 500ms |
| SSE event latency (orchestrator write → browser update) | < 3 seconds |
| Media list with 1000 files | < 1 second (paginated, 50 per page) |
| Timeline visualization for 15-minute EDL | < 200ms render |
| SQLite read contention with running pipeline | Zero — WAL mode guarantees non-blocking reads |

### Resource Usage

| Resource | Budget |
|----------|--------|
| CPU | < 5% sustained (event polling, template rendering) |
| RAM | < 200 MB (FastAPI + in-memory caches) |
| GPU | Zero — the console never touches the GPU |
| Disk | < 50 MB (templates, static assets, event log) |
| Network | Localhost only by default; < 1 Mbps for SSE + page loads |

### Reliability

- The console process is fully independent of the orchestrator process. Either can restart without affecting the other.
- If the console crashes, the pipeline continues running (gates default to the last-configured mode; `auto` gates proceed, `pause` gates hold).
- If the orchestrator crashes, the console shows the last known state and displays a "pipeline not running" banner.
- SQLite WAL mode ensures the console never blocks the orchestrator and vice versa.

---

## Implementation Phases

### Phase 1: Dashboard + Status Tracking

Build the foundation: FastAPI app, pipeline dashboard, job tracking, SSE event stream.

**Scope:**
- FastAPI application with Jinja2 templates
- Dashboard page with stage cards (reads existing DB state)
- `pipeline_jobs` and `pipeline_events` tables
- Orchestrator instrumented to write job/event records
- SSE endpoint with live stage/job updates
- Basic nav structure

**Deliverable:** A working dashboard that shows real-time pipeline status. No intervention capability yet — read-only.

### Phase 2: Media Browser

**Scope:**
- Media list page with sorting, filtering, pagination
- Media detail page with analysis tabs (transcript, detections, faces, audio events)
- Thumbnail extraction during ingest (if not already present)

**Deliverable:** Full media browsing experience. Creator can inspect what the pipeline found without writing SQL.

### Phase 3: Gates + Narrative Review

The first intervention capability. This replaces the existing CLI-based narrative approval.

**Scope:**
- `pipeline_gates` table and orchestrator gate-checking logic
- Gate configuration page with toggle UI
- Narrative review page (approve/reject/edit)
- Gate presets

**Deliverable:** Creator can configure gates and approve/reject narratives through the web console instead of the CLI.

### Phase 4: Full Review Suite

Extend review to all stage transitions.

**Scope:**
- Cluster review page (merge, split, relabel, exclude)
- Script review page (inline editing, regeneration)
- EDL review page (clip detail, validation display, OTIO download)
- Timeline visualization (SVG or Canvas)
- Render preview page (video player, quality validation)
- Upload approval

**Deliverable:** Full human-in-the-loop capability at every stage transition.

### Phase 5: Notifications + Polish

**Scope:**
- Browser notification support
- Toast messages in the console
- Gate timeout configuration
- Job retry/skip from the UI
- Error drilldown views
- Pipeline run history
- Mobile-responsive layout (for checking status on phone)

**Deliverable:** Production-quality console with all planned features.

---

## Open Questions

1. **Video preview in browser.** Serving raw 4K source files for in-browser preview is impractical. Should we generate low-res proxies during ingest (adds ~30 min to ingest), or rely on thumbnail grids with the option to open files in a local player (VLC/mpv via `xdg-open`)?

2. **Script editing depth.** How rich should the script editor be? Options range from plain textarea (simple, no dependencies) to a structured scene editor (more useful but more code). The structured editor is better but could be deferred to Phase 5.

3. **Multi-run support.** Should the console support viewing past pipeline runs, or only the current/most-recent run? Past runs are useful for comparison but add complexity to every query (must scope by `run_id`).

4. **Remote access.** The console binds to localhost by default. Should there be a flag to bind to `0.0.0.0` for access from other devices on the LAN (e.g., checking pipeline status from a phone)? If so, should we add basic auth (even a static token)?

5. **Cluster visualization.** For the CLASSIFY review, should we include a map view (clusters plotted on a map by GPS) and/or a timeline view (clusters on a temporal axis)? Both are valuable but add frontend complexity. A map could use Leaflet.js (~40 KB) with OpenStreetMap tiles.
