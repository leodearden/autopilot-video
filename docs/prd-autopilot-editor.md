# PRD: Autopilot Video Editor

## Product Overview

**Autopilot** is a self-hosted, LLM-orchestrated video editing pipeline that ingests raw footage and audio from multi-day activities, catalogs the media, organizes it into coherent narratives, scripts and plans edits, sources supplementary assets, renders finished videos, and uploads them to YouTube as unlisted drafts.

The system is designed for a single creator who shoots on a mix of cameras (primarily DJI Osmo Action 6 in 4K×4K square mode on a head-strap mount) and records separate audio tracks. The creator dumps all media from a 3–20 day period into a directory and expects the system to produce review-ready videos within 36 hours of wall-clock time.

### Hardware Target

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 3090 (24 GB VRAM, Ampere, 10,496 CUDA cores) |
| CPU | AMD 16-core (assumed Ryzen 9 / Threadripper class) |
| RAM | 128 GB DDR4/DDR5 |
| Storage | NVMe SSD (sufficient for multi-TB working sets) |
| LLM Access | Claude Max subscription (Opus/Sonnet via claude.ai or API) |

### Key Constraints

- **Budget**: £0–15/month ongoing (excluding hardware amortization and Claude Max subscription)
- **Speed/quality tradeoff**: Biased strongly toward quality; 36-hour wall-clock budget is generous
- **Open source preferred**: Commercial tools only where open source has clear quality gaps
- **Human-in-the-loop**: Final review before publish; the system should not auto-publish

---

## Design Principles

1. **L-Storyboard as universal interchange.** Every piece of media analysis (transcript, detected objects, scene boundaries, face IDs, CLIP embeddings, audio events) is materialized as structured text in a per-clip "storyboard row." All LLM reasoning operates on text — the LLM never needs to see video frames directly. This decouples expensive visual analysis (run once) from creative decisions (iterable via re-prompting).

2. **Catalog-first architecture.** The media catalog is the persistent, queryable ground truth. Every pipeline stage reads from and writes to the catalog. This enables incremental ingestion (add footage mid-trip), re-editing without re-analysis, and debugging any stage independently.

3. **Quality over throughput.** With a 36-hour budget and typically 20–80 hours of raw footage per trip, we can afford to run the largest feasible models at every stage. We never downsample or skip analysis to save time unless the budget is demonstrably exceeded.

4. **Deterministic rendering.** Edit plans are serialized as JSON EDLs (Edit Decision Lists) and optionally exported as OpenTimelineIO files. A given EDL + source media always produces the same output. Human review can happen at the EDL level (fast, text-based) or the rendered output level (slow, visual).

5. **Graceful degradation with human fetch lists.** When the system can't automatically source an asset (specific B-roll, licensed music), it produces a prioritized fetch list for the human rather than silently degrading quality.

---

## Pipeline Stages

### Stage 0: Ingest and Normalize

**Goal:** Discover all media files in the input directory tree, extract technical metadata, normalize audio, and prepare files for analysis.

**Inputs:** A directory path containing a mix of `.mp4`, `.mov`, `.wav`, `.mp3`, `.aac`, and image files from cameras, phones, and audio recorders.

**Outputs:** A populated `media_catalog` database table with one row per source file, containing file path, SHA-256 hash (for deduplication), codec info, resolution, frame rate, duration, creation timestamp (from EXIF/metadata or filesystem), GPS coordinates (if available), audio channel layout, and a status field (`ingested`, `analyzing`, `analyzed`, `error`).

**Technical approach:**

- Use `ffprobe` (via `python-ffmpeg` or direct subprocess) for codec/resolution/duration extraction.
- Use `exiftool` (Perl, pre-installed on most Linux) for EXIF, GPS, and creation timestamp. The DJI Osmo Action 6 embeds GPS and timestamp in MP4 metadata; extract these as primary signals for temporal/spatial clustering in Stage 2.
- Audio normalization: Run `ffmpeg -af loudnorm=I=-16:TP=-1.5:LRA=11` (EBU R128) on all audio to a consistent level before ASR. Store normalized audio as separate `.wav` files to avoid re-encoding source video.
- Deduplication: SHA-256 on the first 64 MB of each file to catch exact duplicates (e.g., files copied to multiple cards).
- Database: **SQLite** via Python `sqlite3`. Single-file database, no server, trivially backed up. Schema supports JSON columns for variable-structure metadata (detected objects, embeddings, etc.).

**Estimated time for 50 hours of footage:** ~5 minutes (I/O-bound, parallelizable across CPU cores).

---

### Stage 1: Catalog — Deep Media Analysis

**Goal:** For every media file, produce a rich, queryable annotation: transcript with word-level timestamps and speaker IDs, scene/shot boundaries, per-frame object detections with tracking IDs, face detections and recognition clusters, audio event classifications, and dense CLIP embeddings for semantic search.

This is the most compute-intensive stage and the foundation for everything downstream. Quality here directly determines the quality of all subsequent LLM reasoning.

#### 1a. Speech Transcription and Diarization

**Model choice: WhisperX with faster-whisper large-v3**

Rationale: We favor `large-v3` (1.55B params, ~3 GB VRAM in FP16 via CTranslate2) over `large-v3-turbo` (809M params) because we're quality-biased and the RTX 3090 has ample VRAM. The quality difference is small but measurable on noisy outdoor audio — exactly our use case. WhisperX adds:
- **pyannote VAD** for robust voice activity detection (critical for action footage with long non-speech segments)
- **wav2vec2 forced alignment** for accurate word-level timestamps (essential for tight edit cuts)
- **pyannote speaker diarization** for identifying who's speaking (useful for narrative organization)

**Performance estimate on RTX 3090:** faster-whisper large-v3 achieves roughly 25–35× real-time on Ampere GPUs for long-form audio. Conservatively using 25×, transcribing 50 hours of audio takes **~2 hours**. Since ASR runs on extracted audio (not video), it can run in parallel with video analysis stages on the same GPU by scheduling sequentially or using CUDA MPS.

**Output schema per file:**
```json
{
  "segments": [
    {
      "start": 12.34,
      "end": 15.67,
      "text": "Look at that view over there",
      "speaker": "SPEAKER_01",
      "words": [
        {"word": "Look", "start": 12.34, "end": 12.55, "score": 0.98},
        {"word": "at", "start": 12.56, "end": 12.65, "score": 0.97}
      ]
    }
  ]
}
```

**Dependencies:**
- `whisperx` (BSD license)
- `faster-whisper` (MIT license)
- `pyannote.audio` (MIT license, requires HuggingFace token for model download)
- A HuggingFace account and accepted user agreements for pyannote models

#### 1b. Shot Boundary Detection

**Model choice: TransNetV2**

Rationale: TransNetV2 is a 3D CNN purpose-built for shot boundary detection, achieving state-of-the-art F1 on ClipShots and BBC Planet Earth benchmarks. It handles gradual transitions (dissolves, fades) that trip up PySceneDetect's threshold-based approach. Since we're quality-biased, use TransNetV2 as primary and PySceneDetect `detect-adaptive` as fallback/validation.

**Performance:** TransNetV2 processes video at roughly 200–400 FPS on an RTX 3090 (it operates on downsampled 48×27 frames). 50 hours of footage at 30fps = ~5.4M frames → ~4–7 hours if run on raw frames. However, TransNetV2 batches efficiently and can be fed pre-extracted thumbnails, reducing this substantially.

**Fallback:** PySceneDetect (BSD license) with `detect-adaptive` mode, threshold 27.0, for simple content-change detection when TransNetV2 fails or for validation.

**Output:** List of `(start_frame, end_frame, transition_type)` tuples per file, stored in the catalog.

#### 1c. Object Detection and Tracking

**Model choice: YOLO11x with ByteTrack**

Rationale: YOLO11x (the "extra-large" variant) maximizes detection accuracy (mAP 54.7 on COCO) at the cost of speed. On an RTX 3090 at 640×640 input resolution, YOLO11x runs at approximately 40–60 FPS. We choose quality over the faster YOLO11n/s/m variants. ByteTrack (built into Ultralytics `model.track()`) provides robust multi-object tracking with identity persistence through occlusions.

**Critical for auto-crop:** The per-frame bounding boxes from this stage are the primary input to the auto-crop viewport computation in Stage 7. Every frame needs detection data, which means we must run YOLO on every frame (not subsampled) for clips that will be auto-cropped. For cataloging-only clips, 1-per-second sampling is sufficient.

**Processing strategy:**
- **Full-frame detection (for auto-crop candidates):** Run YOLO11x on every frame. At 40 FPS on RTX 3090, 50 hours × 30fps = 5.4M frames → ~37 hours. This exceeds our 36-hour budget if run serially. **Mitigation:** Run on every 3rd frame (10 FPS equivalent) and interpolate bounding boxes for intermediate frames using ByteTrack's Kalman filter predictions. This reduces to ~12.5 hours and produces smooth-enough tracking for crop path computation.
- **Sparse detection (for cataloging):** 1 frame per second, yielding ~180K frames → ~75 minutes.

**Resolution strategy:** The source is 4K×4K. YOLO operates on resized input (640×640 is standard). Downsampling from 4096→640 loses small objects, but for our use case (people, vehicles, animals, equipment in action footage), subjects are typically large in frame. If small-object sensitivity becomes an issue, switch to tiled inference (crop 4K into overlapping 640×640 tiles), but this 4× the compute cost and should only be used selectively.

**Output:** Per-frame list of `(track_id, class, bbox_xywh, confidence)` stored in catalog. Track IDs are consistent within a clip (ByteTrack handles ID assignment).

**Dependencies:**
- `ultralytics` (AGPL-3.0 for the package; models are also AGPL-3.0 — acceptable for personal non-distributed use)

#### 1d. Face Detection and Clustering

**Model choice: InsightFace (SCRFD + ArcFace)**

Rationale: InsightFace's `buffalo_l` model pack provides SCRFD for detection (fast, handles varied poses) and ArcFace for 512-dimensional face embeddings (99.4% LFW accuracy). We run face detection on the same frame samples as YOLO (leveraging the GPU while it's already loaded), extract face crops, compute embeddings, and cluster across all footage using DBSCAN or agglomerative clustering to identify unique individuals.

This enables the LLM to reference people by name (after the human labels the clusters once): "Person A is talking to Person B at the market."

**Performance:** SCRFD runs at ~100 FPS on RTX 3090 at 640×640. ArcFace embedding extraction is ~0.5ms per face crop. With sparse sampling (1 FPS), face processing for 50 hours is under 30 minutes.

**Output:** Per-frame list of `(face_id, bbox, embedding_vector)` and a global `face_clusters` table mapping cluster IDs to human-assigned labels.

**Dependencies:**
- `insightface` (MIT license)
- `scikit-learn` for clustering

#### 1e. Semantic Frame Embeddings (CLIP/SigLIP)

**Model choice: SigLIP 2 (ViT-SO400M/14)**

Rationale: SigLIP 2 (Google, Feb 2025) provides superior zero-shot classification and retrieval compared to OpenAI CLIP, with native dynamic resolution support. The ViT-SO400M variant fits comfortably in 24 GB VRAM alongside other models. These embeddings enable natural-language search over footage ("person cooking outdoors", "group photo at sunset") and zero-shot activity classification without custom training.

**Sampling:** 1 frame per 2 seconds. For 50 hours of footage, this is ~90K frames. At ~200 embeddings/second on RTX 3090, this takes ~7.5 minutes.

**Output:** 768-dimensional float32 embedding vector per sampled frame, stored in the catalog. Build a FAISS IVF index for fast approximate nearest-neighbor search.

**Dependencies:**
- `transformers` (Apache 2.0)
- `faiss-cpu` or `faiss-gpu` (MIT license)

#### 1f. Audio Event Classification

**Model choice: PANNs (CNN14)**

Rationale: PANNs (Pretrained Audio Neural Networks) achieve 0.431 mAP on AudioSet's 527 event classes. This detects non-speech audio events — applause, music, vehicle engines, wind, water, animal sounds — that are useful for activity classification and narrative organization. CNN14 is the best accuracy/speed tradeoff at 80M params.

**Performance:** CNN14 processes audio at >100× real-time on GPU. 50 hours → ~30 minutes.

**Output:** Per-second list of `(event_class, probability)` for top-5 audio events, stored in catalog.

**Dependencies:**
- `panns_inference` (MIT license)

#### 1g. Video Captioning (Selective)

**Model choice: Qwen2.5-VL-7B-Instruct (local via vLLM or transformers)**

Rationale: For clips where YOLO detections and CLIP embeddings are insufficient to describe the scene (scenic shots, complex activities, unusual subjects), run a video language model to generate natural-language descriptions. The 7B variant fits in RTX 3090 VRAM (FP16 ~14 GB) and provides strong scene understanding with temporal grounding.

**Strategy:** Run selectively — only on clips where the LLM planner (Stage 3) requests additional context, or on a random 10% sample during initial cataloging to validate that YOLO + CLIP + ASR provide sufficient coverage. This is an on-demand capability, not a batch-everything step.

**Performance:** ~2–5 seconds per 30-second clip at 7B FP16 on RTX 3090. Budget for ~500 clip descriptions per editing run.

**Dependencies:**
- `transformers` (Apache 2.0)
- `vllm` (Apache 2.0) for batched inference if running many descriptions

#### Stage 1 — Total Time Estimate

| Sub-stage | Items | Rate on RTX 3090 | Estimated Time |
|-----------|-------|-------------------|----------------|
| 1a. ASR (WhisperX large-v3) | 50h audio | 25× real-time | ~2.0 h |
| 1b. Shot boundaries (TransNetV2) | 5.4M frames | ~300 FPS | ~5.0 h |
| 1c. Object detection (YOLO11x, every 3rd frame) | 1.8M frames | ~50 FPS | ~10.0 h |
| 1d. Face detection (SCRFD, 1 FPS) | 180K frames | ~100 FPS | ~0.5 h |
| 1e. CLIP embeddings (SigLIP 2, 0.5 FPS) | 90K frames | ~200/s | ~0.15 h |
| 1f. Audio events (PANNs) | 50h audio | 100× RT | ~0.5 h |
| **Subtotal (serial)** | | | **~18.2 h** |

**Parallelization strategy:** Sub-stages 1a and 1f operate on audio only (no GPU contention with video stages). Run 1a and 1f concurrently with 1b/1c/1d/1e on the GPU. The GPU-heavy stages (1b, 1c, 1d, 1e) should run sequentially on the GPU to avoid VRAM contention (YOLO11x alone uses ~8 GB). With audio stages overlapped, effective wall-clock for Stage 1: **~16 hours**.

This leaves ~20 hours for Stages 2–8, which is ample — the remaining stages are far less compute-intensive.

---

### Stage 2: Activity Classification

**Goal:** Group the cataloged clips into activity clusters. An "activity" is a coherent event or experience (e.g., "morning hike to waterfall", "cooking dinner at campsite", "market visit in Chiang Mai").

**Approach:**

1. **Temporal-spatial clustering:** Group clips by (creation_timestamp, GPS_coordinates) proximity. Clips shot within 30 minutes and 500 meters of each other are candidates for the same activity. Use DBSCAN with a custom distance metric combining temporal and spatial dimensions.

2. **Semantic refinement:** Within each temporal-spatial cluster, use SigLIP embeddings and transcript text to sub-cluster if the cluster contains clearly distinct activities (e.g., "hiking" and "swimming" at the same location). Compute mean CLIP embedding per temporal window (5-minute sliding), detect embedding-space discontinuities (cosine distance > threshold) as activity boundaries.

3. **LLM labeling:** Feed each cluster's summary (transcripts excerpts, top YOLO classes, top audio events, GPS reverse-geocode, time range) to Claude Sonnet via the API. Prompt: "Given these clip summaries, provide a short descriptive label for this activity and a 1-paragraph description. Also flag if this cluster appears to contain multiple distinct activities that should be split."

**Output:** `activity_clusters` table in the catalog, each with: cluster_id, label, description, time_range, location, list of clip_ids.

**Estimated time:** <30 minutes (mostly LLM API calls; clustering computation is trivial).

---

### Stage 3: Narrative Organization

**Goal:** From the set of activity clusters, propose a set of video narratives. A narrative is a story that a viewer would watch — it might span one activity ("The waterfall hike") or weave together several related activities ("Three days in northern Thailand").

**Approach:**

This is a fundamentally creative task best handled by a large LLM with the full context.

1. **Construct the master storyboard:** For each activity cluster, produce a structured summary: label, duration of usable footage, key moments (from transcript highlights and YOLO event density), people present (from face clusters), emotional tone (inferred from transcript sentiment and audio events), visual quality notes (from YOLO detection confidence distributions — low confidence correlates with motion blur or poor framing).

2. **Prompt Claude Opus** (via Claude Max subscription) with the complete master storyboard and a system prompt defining the creator's style preferences, target video durations, and audience. The prompt should request:
   - A proposed set of narratives with titles
   - For each narrative: which activities it includes, proposed duration, narrative arc (beginning/middle/end), emotional journey, target audience
   - Reasoning for why these narratives were chosen and what was excluded

3. **Human review checkpoint:** Present the narrative proposals to the creator for approval, modification, or rejection. This is a natural pause point — the creator may merge narratives, split them, or add/remove activities.

**Output:** `narratives` table with: narrative_id, title, description, proposed_duration, list of activity_cluster_ids, arc_notes.

**Estimated time:** ~15 minutes (one large Claude Opus call + human review).

---

### Stage 4: Script Each Narrative

**Goal:** For each approved narrative, produce a detailed script: narration text (if voiceover is used), on-screen text/titles, and a beat-by-beat description of what the viewer sees and hears at each moment.

**Approach:**

1. **Construct the narrative storyboard:** For the activities in this narrative, assemble the full L-Storyboard: every shot with its transcript, visual description, detected objects, people, audio events, duration. Include timestamps relative to both the source clip and the narrative timeline.

2. **Prompt Claude Opus** with the narrative storyboard, the approved narrative description from Stage 3, and a script-writing system prompt. Request:
   - Scene-by-scene script with estimated timing
   - Voiceover narration text (if applicable)
   - On-screen text/titles at key moments
   - Music mood suggestions per scene (e.g., "upbeat acoustic", "ambient tension")
   - B-roll needs: specific shots that would enhance the narrative but aren't in the source footage
   - Quality flags: scenes where source footage may be insufficient (too dark, too shaky, key moment not captured)

**Output:** `narrative_scripts` table with the full script JSON per narrative.

**Estimated time:** ~10 minutes per narrative (Claude Opus call). For 5 narratives: ~50 minutes.

---

### Stage 5: Edit Plan (EDL Generation)

**Goal:** For each scripted narrative, produce a precise Edit Decision List: which source clips to use, exact in/out points (frame-accurate), transitions, audio mixing instructions, title card specifications, and auto-crop directives.

**Approach:**

1. **Prompt Claude Opus** with the script (from Stage 4) and the detailed storyboard data for all candidate clips. The prompt includes function-calling tools that the LLM uses to construct the edit:

   - `select_clip(clip_id, in_timecode, out_timecode, track)` — place a clip on the timeline
   - `add_transition(type, duration, position)` — crossfade, cut, etc.
   - `set_crop_mode(clip_id, mode, subject_track_id)` — "auto_subject" (track a specific person/object), "center", "manual_offset"
   - `add_title(text, style, position, duration)` — title card or lower third
   - `set_audio(clip_id, level_db, fade_in, fade_out)` — per-clip audio levels
   - `add_music(mood, duration, start_time)` — request music track (resolved in Stage 6)
   - `add_voiceover(text, start_time, duration)` — request TTS voiceover (resolved in Stage 6)
   - `request_broll(description, duration, start_time)` — request stock B-roll (resolved in Stage 6)

2. **Validate the EDL:** Automated checks for:
   - No overlapping clips on the same track
   - Total duration within ±10% of the scripted target
   - All referenced clip_ids exist in the catalog
   - All in/out points within clip duration bounds
   - Audio levels within broadcast-safe ranges

3. **Export as OpenTimelineIO** for optional human review in DaVinci Resolve or other NLE before rendering.

**Output:** `edit_plans` table with JSON EDL per narrative, plus `.otio` files in the output directory.

**Estimated time:** ~15 minutes per narrative. For 5 narratives: ~75 minutes.

---

### Stage 6: Asset Sourcing

**Goal:** Resolve all external asset requests from the edit plans — find or generate music tracks, source B-roll footage, and generate voiceover audio.

#### 6a. Music

**Approach (ordered by preference):**

1. **MusicGen (Meta AudioCraft)** — open source, runs locally on RTX 3090. Generate 30-second instrumental tracks from text prompts (e.g., "upbeat acoustic guitar, travel vlog, 120 BPM"). Model: `facebook/musicgen-large` (3.3B params, ~7 GB VRAM FP16). Generates ~30 seconds in ~15 seconds on RTX 3090. **Limitation:** CC-BY-NC model weights — acceptable for personal YouTube but not for commercial licensing. Quality is good for background music but not production-grade.

2. **Freesound.org API** — CC-licensed sound effects and ambient audio. Free, REST API, 410K+ sounds. Use for ambient pads, transitions, and sound effects.

3. **Fetch list fallback** — For narratives where the creator wants specific licensed music, output a fetch list: "Track needed: upbeat acoustic, 2:30 duration, for 'Morning Hike' narrative. Suggested search: Epidemic Sound / Artlist / YouTube Audio Library."

#### 6b. B-Roll

**Approach:**

1. **Pexels API** — Free, excellent REST API, HD/4K stock video. Rate limit: 200 requests/hour (sufficient). Search by keyword from the edit plan's `request_broll(description)` calls. Download top-3 results per request for LLM to select from.

2. **Pixabay API** — Secondary free source. 100 requests/minute.

3. **Fetch list fallback** — "B-roll needed: aerial shot of Thai rice paddies, 5 seconds. None found in Pexels/Pixabay. Suggest: creator's drone footage or Storyblocks."

#### 6c. Voiceover

**Model choice: Kokoro (Apache 2.0, 82M params)**

Rationale: Kokoro delivers near-commercial TTS quality at 82M params with sub-0.3s latency. It's fully open source (Apache 2.0), runs on CPU easily given its small size, and supports multiple English voices. For a personal YouTube channel where the creator isn't narrating on-camera, Kokoro provides natural-sounding voiceover without any API cost.

**Alternative (higher quality, small cost): ElevenLabs Starter ($5/month, ~30 minutes of audio).** If the creator wants a specific cloned voice or premium quality, ElevenLabs is the cost-effective commercial option within budget.

**Performance:** Kokoro generates ~10× real-time on CPU alone. A typical 10-minute video with 3 minutes of voiceover takes ~18 seconds to generate.

**Output:** Downloaded/generated asset files in the project's `assets/` directory. Updated EDL with resolved file paths replacing placeholder requests. Any unresolved requests added to `fetch_list.md` for human action.

**Estimated time:** ~30 minutes (dominated by API calls and MusicGen generation).

---

### Stage 7: Render

**Goal:** Execute the finalized edit plans to produce rendered video files.

#### 7a. Auto-Crop Viewport Computation

**Context:** Source footage is 4K×4K (4096×4096 or 3840×3840) from the DJI Osmo Action 6 head-strap mount. Target output is 1920×1080 (16:9) and/or 1080×1920 (9:16). The crop window is 47–50% of the source width and 26–28% of the source height (for 16:9), giving substantial room for intelligent reframing.

**Algorithm:**

1. **Load tracking data** from Stage 1c for the target clip. Each frame has bounding boxes for tracked objects with IDs.

2. **Select subject(s):** The EDL specifies a `subject_track_id` (or "auto" to select the most prominent person). For "auto", select the track ID with the highest cumulative bbox area × frame count.

3. **Compute raw crop center:** For each frame, place the crop center such that the subject's bounding box is positioned according to the rule of thirds (horizontally: subject center at 1/3 or 2/3 of frame width; vertically: subject eyes at 1/3 from top). If multiple subjects, compute the bounding box that contains all of them and center on that.

4. **Smooth the crop path:** Apply a Kalman filter (or exponential moving average with τ = 0.5 seconds) to the crop center coordinates to eliminate jitter from frame-to-frame detection noise. The smoothing time constant should be tunable per-activity type:
   - **Slow activities** (walking, talking, cooking): τ = 1.0s — very smooth, cinematic panning
   - **Medium activities** (hiking, sightseeing): τ = 0.5s — responsive but stable
   - **Fast activities** (biking, sports, action): τ = 0.2s — tight tracking

5. **Clamp to frame bounds:** Ensure the crop window never extends beyond the source frame. If the subject is near an edge, bias the crop toward the subject (allow some edge approach) rather than losing tracking.

6. **Handle detection gaps:** When the subject is not detected (occluded, out of frame), hold the last known crop position for up to 2 seconds, then smoothly drift toward frame center over the next 1 second. If the subject reappears, smoothly transition back.

7. **Output:** A per-frame `(crop_x, crop_y)` array for the top-left corner of the crop window. Store alongside the EDL.

**Fallback modes:**
- `center`: Fixed center crop (for scenic shots with no clear subject)
- `manual_offset`: Creator specifies a static offset (e.g., "crop 200px left of center")
- `stabilize_only`: Use the gyro data from the DJI Osmo Action 6 (embedded in metadata) for electronic stabilization within the crop window, no subject tracking

#### 7b. FFmpeg Rendering Pipeline

**Engine: FFmpeg with NVENC (GPU-accelerated encoding)**

**Rendering steps per narrative:**

1. **Decode source clips:** FFmpeg `cuvid` hardware decoder for H.264/H.265 → GPU memory. This avoids CPU-GPU transfers for decode.

2. **Apply crop:** Use FFmpeg's `crop` filter with per-frame coordinates from the crop path. For dynamic crops, generate a `sendcmd` filter script or use MoviePy's frame-by-frame processing with PyAV for low-level control.

3. **Scale to output resolution:** `scale_cuda` filter: crop window → 1920×1080 or 1080×1920.

4. **Apply transitions:** Crossfades via `xfade` filter between clips. Cut transitions are just concatenation.

5. **Mix audio:** Overlay music at specified levels, mix voiceover, normalize final audio to -16 LUFS (EBU R128) via `loudnorm` filter.

6. **Burn subtitles (optional):** From ASR output, render captions using `subtitles` filter or `drawtext`.

7. **Encode:** NVENC H.264 High Profile, CRF-equivalent quality targeting ~15 Mbps for 1080p (YouTube recommended). Audio: AAC 256 kbps stereo.

**Performance:** NVENC on RTX 3090 encodes 1080p H.264 at 300–500 FPS. A 10-minute video renders in ~30–60 seconds for the encode step. The bottleneck is the crop-path computation and filter graph setup, not encoding.

**Alternative approach for complex edits: MoviePy + PyAV**

For edits requiring frame-level manipulation (dynamic crops, picture-in-picture, custom overlays), use MoviePy 2.x for composition and PyAV for I/O. MoviePy processes frames as numpy arrays, enabling arbitrary Python transformations per frame. This is slower than pure FFmpeg filter graphs (~10–30 FPS for complex compositions) but far more flexible.

**Recommended hybrid:** Use FFmpeg filter graphs for simple cuts/concatenations/audio mixing (fast path), and MoviePy for clips requiring dynamic crop or complex overlays (slow path). The EDL should tag each clip with its rendering complexity to route accordingly.

**Total render time estimate:** For 5 narratives averaging 10 minutes each:
- Simple cuts (70% of clips): FFmpeg filter graph, ~2 minutes total
- Dynamic crop clips (30%): MoviePy at ~20 FPS, ~15 minutes total
- Audio mixing and normalization: ~5 minutes total
- **Total: ~25 minutes**

#### 7c. Quality Validation

After rendering, run automated checks:
- Duration matches EDL target (±1 second)
- Audio loudness is within -16 ±1 LUFS
- No black frames (frame analysis for mean pixel value < 5)
- No silent audio gaps > 2 seconds (unless intentional per EDL)
- Resolution and codec match target specs
- File size is reasonable (8–15 MB per minute for 1080p H.264)

**Output:** Rendered `.mp4` files in `output/` directory. `validation_report.json` per narrative.

---

### Stage 8: Upload to YouTube

**Goal:** Upload rendered videos to YouTube as unlisted drafts with metadata (title, description, tags, thumbnail).

**Approach:**

- **YouTube Data API v3** via `google-api-python-client`.
- Authentication: OAuth 2.0 with offline refresh token (one-time browser auth, then unattended).
- Upload with `privacyStatus: "unlisted"` (visible only via direct link until creator publishes).
- Set title, description (from narrative script), tags (from activity labels + detected objects), and category.
- Upload a thumbnail: Extract the "best frame" from the rendered video (highest YOLO detection confidence + rule-of-thirds composition score) and upload via `thumbnails.set`.

**Quota:** Default 10,000 units/day. Each upload costs 1,600 units → 6 uploads/day. For 5 narratives, this is well within limits.

**Output:** `upload_results` table with YouTube video IDs and URLs. Summary message to creator with links.

**Estimated time:** ~5 minutes per upload (limited by upload bandwidth, not API).

---

## Time Budget Summary

| Stage | Description | Estimated Wall-Clock Time |
|-------|-------------|--------------------------|
| 0 | Ingest and normalize | 0.1 h |
| 1 | Deep media analysis | 16.0 h |
| 2 | Activity classification | 0.5 h |
| 3 | Narrative organization | 0.25 h |
| 4 | Script each narrative | 0.8 h |
| 5 | Edit plan / EDL generation | 1.25 h |
| 6 | Asset sourcing | 0.5 h |
| 7 | Render | 0.5 h |
| 8 | Upload | 0.5 h |
| | **Total** | **~20.4 h** |

**Margin:** ~15.6 hours of slack against the 36-hour budget. This margin absorbs:
- Larger-than-expected footage volumes (up to ~100 hours)
- Re-runs of LLM stages after human feedback
- Network delays in API calls and uploads
- Unexpected model loading/initialization overhead

---

## Technology Stack

### Core Languages
- **Python 3.11+** — primary language for all pipeline code
- **Bash** — for FFmpeg command composition and system orchestration

### Database
- **SQLite 3** — single-file embedded database. JSON1 extension for structured metadata columns. WAL mode for concurrent reads during pipeline execution.

### ML/AI Models (all running locally on RTX 3090)

| Model | Task | Size (VRAM) | License |
|-------|------|-------------|---------|
| faster-whisper large-v3 | ASR | ~3 GB FP16 | MIT |
| pyannote/speaker-diarization-3.1 | Diarization | ~1 GB | MIT |
| TransNetV2 | Shot detection | <1 GB | MIT |
| YOLO11x | Object detection | ~8 GB FP16 | AGPL-3.0 |
| InsightFace buffalo_l | Face det+rec | ~2 GB | MIT |
| SigLIP 2 ViT-SO400M/14 | Embeddings | ~3 GB FP16 | Apache 2.0 |
| PANNs CNN14 | Audio events | <1 GB | MIT |
| Qwen2.5-VL-7B-Instruct | Video captioning | ~14 GB FP16 | Apache 2.0 |
| MusicGen-large | Music generation | ~7 GB FP16 | MIT (code), CC-BY-NC (weights) |
| Kokoro | TTS | <1 GB | Apache 2.0 |

**VRAM management:** These models must NOT all be loaded simultaneously. The pipeline orchestrator loads/unloads models per stage. Peak VRAM usage is ~14 GB (Qwen2.5-VL-7B) or ~8 GB (YOLO11x). The RTX 3090's 24 GB provides comfortable headroom.

### LLM (Cloud API)

| Model | Task | Access |
|-------|------|--------|
| Claude Opus (latest) | Narrative planning, scripting, edit planning | Claude Max subscription |
| Claude Sonnet (latest) | Activity labeling, metadata enrichment | Claude Max subscription |

### Key Python Libraries

| Library | Purpose | License |
|---------|---------|---------|
| `whisperx` | ASR + alignment + diarization | BSD |
| `ultralytics` | YOLO detection + tracking | AGPL-3.0 |
| `insightface` | Face detection + recognition | MIT |
| `transformers` | SigLIP, Qwen2.5-VL, model loading | Apache 2.0 |
| `faiss-gpu` | Vector similarity search | MIT |
| `moviepy` (v2.x) | Programmatic video editing | MIT |
| `pyav` | Low-level FFmpeg bindings | BSD |
| `opentimelineio` | Timeline interchange format | Apache 2.0 |
| `anthropic` | Claude API client | MIT |
| `google-api-python-client` | YouTube Data API | Apache 2.0 |
| `panns_inference` | Audio event classification | MIT |
| `audiocraft` | MusicGen music generation | MIT (code) |
| `kokoro` | Text-to-speech | Apache 2.0 |
| `scenedetect` | Fallback scene detection | BSD |
| `scikit-learn` | Clustering (DBSCAN, agglomerative) | BSD |

### External APIs

| API | Purpose | Cost |
|-----|---------|------|
| Pexels Video API | B-roll sourcing | Free (200 req/hr) |
| Pixabay API | B-roll + music sourcing | Free (100 req/min) |
| Freesound.org API | Sound effects | Free (CC-licensed content) |
| YouTube Data API v3 | Video upload | Free (10K units/day) |

---

## Data Model

### SQLite Schema (Key Tables)

```sql
CREATE TABLE media_files (
    id TEXT PRIMARY KEY,           -- UUID
    file_path TEXT NOT NULL,
    sha256_prefix TEXT,            -- First 64MB hash for dedup
    codec TEXT,
    resolution_w INTEGER,
    resolution_h INTEGER,
    fps REAL,
    duration_seconds REAL,
    created_at TEXT,               -- ISO 8601 from EXIF
    gps_lat REAL,
    gps_lon REAL,
    audio_channels INTEGER,
    status TEXT DEFAULT 'ingested', -- ingested|analyzing|analyzed|error
    metadata_json TEXT             -- Extensible JSON blob
);

CREATE TABLE transcripts (
    media_id TEXT REFERENCES media_files(id),
    segments_json TEXT,            -- WhisperX output
    language TEXT,
    PRIMARY KEY (media_id)
);

CREATE TABLE shot_boundaries (
    media_id TEXT REFERENCES media_files(id),
    boundaries_json TEXT,          -- [(start_frame, end_frame, type), ...]
    method TEXT,                   -- transnetv2|pyscenedetect
    PRIMARY KEY (media_id, method)
);

CREATE TABLE detections (
    media_id TEXT REFERENCES media_files(id),
    frame_number INTEGER,
    detections_json TEXT,          -- [{track_id, class, bbox, conf}, ...]
    PRIMARY KEY (media_id, frame_number)
);

CREATE TABLE face_clusters (
    cluster_id INTEGER PRIMARY KEY,
    label TEXT,                    -- Human-assigned name
    representative_embedding BLOB, -- 512-dim float32
    sample_image_paths TEXT        -- JSON array of face crop paths
);

CREATE TABLE clip_embeddings (
    media_id TEXT REFERENCES media_files(id),
    frame_number INTEGER,
    embedding BLOB,               -- 768-dim float32
    PRIMARY KEY (media_id, frame_number)
);

CREATE TABLE audio_events (
    media_id TEXT REFERENCES media_files(id),
    timestamp_seconds REAL,
    events_json TEXT,              -- [{class, probability}, ...]
    PRIMARY KEY (media_id, timestamp_seconds)
);

CREATE TABLE activity_clusters (
    cluster_id TEXT PRIMARY KEY,
    label TEXT,
    description TEXT,
    time_start TEXT,
    time_end TEXT,
    location_label TEXT,
    gps_center_lat REAL,
    gps_center_lon REAL,
    clip_ids_json TEXT             -- JSON array of media_file IDs
);

CREATE TABLE narratives (
    narrative_id TEXT PRIMARY KEY,
    title TEXT,
    description TEXT,
    proposed_duration_seconds REAL,
    activity_cluster_ids_json TEXT,
    arc_notes TEXT,
    status TEXT DEFAULT 'proposed' -- proposed|approved|scripted|planned|rendered|uploaded
);

CREATE TABLE edit_plans (
    narrative_id TEXT REFERENCES narratives(narrative_id),
    edl_json TEXT,                 -- Full EDL with clip refs, timings, effects
    otio_path TEXT,                -- Path to .otio file
    validation_json TEXT,
    PRIMARY KEY (narrative_id)
);

CREATE TABLE crop_paths (
    media_id TEXT REFERENCES media_files(id),
    target_aspect TEXT,            -- "16:9" or "9:16"
    subject_track_id INTEGER,
    smoothing_tau REAL,
    path_data BLOB,               -- Packed array of (frame, x, y) tuples
    PRIMARY KEY (media_id, target_aspect, subject_track_id)
);

CREATE TABLE uploads (
    narrative_id TEXT REFERENCES narratives(narrative_id),
    youtube_video_id TEXT,
    youtube_url TEXT,
    uploaded_at TEXT,
    privacy_status TEXT DEFAULT 'unlisted',
    PRIMARY KEY (narrative_id)
);
```

---

## Project Structure

```
autopilot/
├── README.md
├── pyproject.toml
├── config.yaml                    # User-editable configuration
├── autopilot/
│   ├── __init__.py
│   ├── cli.py                     # Main entry point
│   ├── orchestrator.py            # Pipeline DAG runner
│   ├── config.py                  # Configuration loading
│   ├── db.py                      # SQLite catalog interface
│   │
│   ├── ingest/
│   │   ├── scanner.py             # Directory walker + metadata extraction
│   │   ├── normalizer.py          # Audio normalization
│   │   └── dedup.py               # SHA-256 deduplication
│   │
│   ├── analyze/
│   │   ├── asr.py                 # WhisperX transcription
│   │   ├── scenes.py              # TransNetV2 + PySceneDetect
│   │   ├── objects.py             # YOLO11x + ByteTrack
│   │   ├── faces.py               # InsightFace SCRFD + ArcFace
│   │   ├── embeddings.py          # SigLIP 2 frame embeddings
│   │   ├── audio_events.py        # PANNs CNN14
│   │   ├── captions.py            # Qwen2.5-VL selective captioning
│   │   └── gpu_scheduler.py       # VRAM-aware model load/unload
│   │
│   ├── organize/
│   │   ├── cluster.py             # Temporal-spatial-semantic clustering
│   │   ├── classify.py            # LLM-based activity labeling
│   │   └── narratives.py          # LLM-based narrative proposals
│   │
│   ├── plan/
│   │   ├── script.py              # LLM script generation
│   │   ├── edl.py                 # LLM EDL generation with tool use
│   │   ├── validator.py           # EDL validation rules
│   │   └── otio_export.py         # OpenTimelineIO export
│   │
│   ├── source/
│   │   ├── music.py               # MusicGen + Freesound
│   │   ├── broll.py               # Pexels + Pixabay API
│   │   ├── voiceover.py           # Kokoro / ElevenLabs TTS
│   │   └── fetch_list.py          # Unresolved asset reporting
│   │
│   ├── render/
│   │   ├── crop.py                # Auto-crop viewport computation
│   │   ├── ffmpeg_render.py       # FFmpeg filter graph rendering
│   │   ├── moviepy_render.py      # MoviePy complex composition
│   │   ├── router.py              # Route clips to fast/slow render path
│   │   └── validate.py            # Post-render quality checks
│   │
│   ├── upload/
│   │   ├── youtube.py             # YouTube Data API v3 upload
│   │   └── thumbnail.py           # Best-frame thumbnail extraction
│   │
│   └── prompts/
│       ├── activity_label.md      # System prompt for activity labeling
│       ├── narrative_planner.md   # System prompt for narrative proposals
│       ├── script_writer.md       # System prompt for script generation
│       └── edit_planner.md        # System prompt for EDL generation (with tool schemas)
│
├── tests/
│   ├── test_ingest.py
│   ├── test_analyze.py
│   ├── test_crop.py
│   ├── test_render.py
│   └── fixtures/                  # Small test media files
│
├── scripts/
│   ├── setup_models.sh            # Download all models
│   ├── setup_youtube_oauth.py     # One-time YouTube auth
│   └── label_faces.py             # Interactive face cluster labeling
│
└── output/                        # Rendered videos + reports
    ├── {narrative_title}/
    │   ├── final.mp4
    │   ├── edl.json
    │   ├── edl.otio
    │   ├── validation_report.json
    │   └── assets/
    └── fetch_list.md              # Unresolved asset requests
```

---

## Configuration

```yaml
# config.yaml

input_dir: ~/footage/trip-thailand-2026/
output_dir: ~/output/trip-thailand-2026/

# Creator profile (used in LLM prompts)
creator:
  name: "Your Name"
  channel_style: "travel vlog, relaxed, observational"
  target_audience: "travel enthusiasts, 25-40"
  default_video_duration_minutes: 8-15
  narration_style: "minimal voiceover, let footage speak"
  music_preference: "acoustic, ambient, no vocals"

# Camera profiles
cameras:
  dji_osmo_action_6:
    source_resolution: [4096, 4096]  # Or [3840, 3840]
    aspect_mode: "square"
    has_gyro_data: true
    default_crop_target: "16:9"
    crop_smoothing_tau: 0.5

# Output targets
output:
  primary_aspect: "16:9"
  resolution: [1920, 1080]
  codec: "h264"
  quality_crf: 18                   # Lower = higher quality
  audio_bitrate: "256k"
  target_loudness_lufs: -16

# Model preferences
models:
  whisper_size: "large-v3"          # large-v3 | large-v3-turbo
  yolo_variant: "yolo11x"          # yolo11x | yolo11l | yolo11m
  yolo_sample_every_n_frames: 3    # 1 = every frame, 3 = every 3rd
  clip_model: "google/siglip2-so400m-patch14"
  face_model: "buffalo_l"
  tts_engine: "kokoro"             # kokoro | elevenlabs
  music_engine: "musicgen"         # musicgen | fetch_list_only

# LLM
llm:
  provider: "anthropic"
  planning_model: "claude-opus-4-20250514"
  utility_model: "claude-sonnet-4-20250514"

# YouTube
youtube:
  privacy_status: "unlisted"       # unlisted | private
  default_category: "19"           # Travel & Events
  credentials_path: ~/.config/autopilot/youtube_oauth.json

# Processing
processing:
  max_wall_clock_hours: 36
  gpu_device: 0
  num_cpu_workers: 12              # Leave 4 cores for system
  batch_size_yolo: 16
  batch_size_whisper: 24
```

---

## Success Criteria

### Quantitative

| Metric | Target | Measurement |
|--------|--------|-------------|
| Wall-clock time for 50h of footage | < 36 hours | Timer from `autopilot run` to final upload |
| Transcript word error rate | < 8% on clear speech | Manual spot-check of 10 random 1-min segments |
| Auto-crop subject retention | Subject in frame > 90% of tracked clip duration | Automated: check if subject bbox center is within crop window |
| Auto-crop smoothness | No visible jitter in rendered output | Manual review: no frame-to-frame crop jumps > 2% of frame width |
| Render quality | No encoding artifacts visible at 1080p | Manual review + automated VMAF > 90 vs source crop |
| Upload success rate | 100% for well-formed renders | YouTube API response code check |
| EDL validity rate | > 95% of LLM-generated EDLs pass validation without correction | Automated validation pass/fail rate |

### Qualitative

| Criterion | Target |
|-----------|--------|
| Activity classification accuracy | Creator agrees with >80% of auto-assigned labels |
| Narrative quality | Creator approves ≥3 of 5 proposed narratives without major restructuring |
| Script quality | Creator needs <30 min of manual script editing per narrative |
| Edit plan quality | Rendered output requires <20% of clips to be manually re-cut |
| Overall time savings | Creator spends <4 hours on review/corrections for a full trip's worth of videos (vs. estimated 20-40 hours of fully manual editing) |

---

## Implementation Phases

### Phase 1: Foundation (Stages 0, 1a, 1b, 1c)
Build the ingest pipeline, SQLite catalog, ASR, shot detection, and object detection. This is the minimum viable analysis backbone. Test with 1 hour of real footage.

**Deliverable:** A catalog database with transcripts, shot boundaries, and per-frame detections for all ingested media. CLI command: `autopilot ingest /path/to/footage`.

### Phase 2: Intelligence (Stages 1d, 1e, 1f, 2, 3)
Add face clustering, CLIP embeddings, audio events, activity classification, and narrative organization. This is where the LLM integration begins.

**Deliverable:** Activity clusters with labels and narrative proposals. CLI command: `autopilot analyze` and `autopilot plan --narratives`.

### Phase 3: Editing (Stages 4, 5, 7a, 7b)
Build the script writer, EDL generator, auto-crop computation, and rendering pipeline. This is the core creative automation.

**Deliverable:** Rendered videos from LLM-generated edit plans. CLI command: `autopilot edit` and `autopilot render`.

### Phase 4: Polish (Stages 6, 7c, 8)
Add asset sourcing (music, B-roll, TTS), quality validation, and YouTube upload. This completes the end-to-end pipeline.

**Deliverable:** Full pipeline from `autopilot run` (ingest → upload). Fetch list for unresolved assets.

### Phase 5: Iteration
Based on real-world usage, tune:
- LLM prompts for narrative quality and edit planning
- Auto-crop smoothing parameters per activity type
- YOLO sampling rate vs. crop quality tradeoffs
- Face clustering thresholds
- Music generation prompts

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| YOLO11x every-3rd-frame too slow for 100h+ trips | Medium | Exceeds 36h budget | Fall back to YOLO11l (2× faster) or increase sampling interval to every 5th frame |
| LLM edit plans produce invalid EDLs | High initially | Wasted render time | Strict JSON schema validation + retry loop (up to 3 attempts with error feedback) |
| Auto-crop loses subject during fast action | Medium | Poor framing in rendered output | Activity-aware smoothing τ; human review checkpoint before render |
| MusicGen quality insufficient for final output | Medium | Creator rejects background music | Default to fetch-list-only mode; creator provides music manually |
| Claude API rate limits during batch planning | Low | Pipeline stalls at planning stages | Implement exponential backoff; batch prompts efficiently; use Sonnet for utility tasks |
| AGPL-3.0 YOLO license concerns | Low | Legal ambiguity for personal use | AGPL only triggers on distribution; personal use on own hardware is unambiguous |
| Qwen2.5-VL-7B + YOLO11x VRAM co-residency | None | N/A | Pipeline orchestrator ensures only one large model loaded at a time |
| YouTube API quota exhaustion | Very Low | Can't upload >6 videos/day | Request quota increase (free) if needed; batch uploads across days |
