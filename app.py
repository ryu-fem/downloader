import json
import mimetypes
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests
import yt_dlp
from flask import Flask, request, jsonify, render_template_string, send_file, abort

app = Flask(__name__)

OUTPUT_DIR = str(Path(__file__).resolve().parent / "downloads")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_TTL_SECONDS = 60 * 30  # finished/errored jobs are forgotten after this

# Bounded pool instead of one raw thread per request — keeps memory/CPU use
# predictable under concurrent downloads instead of growing unbounded.
CPU_COUNT = os.cpu_count() or 2
EXECUTOR = ThreadPoolExecutor(max_workers=max(4, CPU_COUNT * 2))

# One pooled HTTP session reused across requests instead of opening a fresh
# TCP/TLS connection per probe/download.
HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "Mozilla/5.0 (compatible; Puller/1.0)"})

RESERVED_NAMES = set()
FILENAME_LOCK = threading.Lock()

# Audio is always transcoded down to this constant mp3 bitrate on download,
# so this is the number to use for size estimates, not the source stream's
# own bitrate.
TARGET_AUDIO_KBPS = 192

# YouTube increasingly challenges requests from datacenter IPs (the kind
# any cloud host uses) with a "Sign in to confirm you're not a bot" wall.
# Which player client avoids that depends on whether we're authenticated:
#   - No cookies: identify as tv/web_safari, which bypasses the anonymous
#     bot wall in most cases.
#   - With cookies: yt-dlp defaults to a client called "tv_downgraded" for
#     logged-in sessions, which currently has an active, unresolved
#     upstream bug throwing "The page needs to be reloaded." for many
#     videos (see yt-dlp issues #17389 / #17405, both still open as of
#     Aug 2026). Explicitly excluding it and falling back to default +
#     web_embedded is the best known workaround right now, but this is a
#     moving target — YouTube changes this often enough that no client
#     list is guaranteed to keep working.
YOUTUBE_PLAYER_CLIENTS_ANON = ["tv", "web_safari"]
YOUTUBE_PLAYER_CLIENTS_AUTH = ["default", "-tv_downgraded", "web_embedded"]

# Optional: path to a cookies.txt file (Netscape format) exported from a
# real, signed-in YouTube session. If present, it's used as a fallback for
# videos that still get blocked after the player-client change above. See
# the deployment notes for how to generate and mount this file.
COOKIES_FILE = os.environ.get("YTDLP_COOKIES_FILE", "/app/cookies.txt")

# Railway (and most host-based platforms) have no easy way to upload a raw
# file, but they do let you set an environment variable. If YTDLP_COOKIES
# holds the *contents* of a cookies.txt file, write it out once at startup
# so COOKIES_FILE above picks it up normally.
_cookies_env = os.environ.get("YTDLP_COOKIES")
if _cookies_env and not Path(COOKIES_FILE).exists():
    try:
        Path(COOKIES_FILE).write_text(_cookies_env)
    except OSError:
        pass


def client_opts(mode):
    """yt-dlp options for a specific client mode. 'auth' uses cookies plus
    the client list that works with a logged-in session; 'anon' uses no
    cookies plus the client list that avoids the anonymous bot wall.
    Format IDs are only valid within the client mode that produced them,
    so whichever mode lists the formats must be the same mode that later
    downloads them."""
    if mode == "auth" and Path(COOKIES_FILE).exists():
        return {
            "extractor_args": {"youtube": {"player_client": YOUTUBE_PLAYER_CLIENTS_AUTH}},
            "cookiefile": COOKIES_FILE,
        }
    return {"extractor_args": {"youtube": {"player_client": YOUTUBE_PLAYER_CLIENTS_ANON}}}


def default_client_mode():
    return "auth" if Path(COOKIES_FILE).exists() else "anon"


def base_ydl_opts(mode=None):
    """Shared yt-dlp options that help avoid YouTube's bot-detection wall.
    Applied to both the format-listing call and the actual download."""
    return client_opts(mode or default_client_mode())


def extract_with_fallback(ydl_opts, url, download, mode=None):
    """Run yt-dlp's extract_info, and if it fails on the known
    "page needs to be reloaded" bug (tv_downgraded client, only hit when
    cookies are in play), retry once without cookies/auth so a public
    video still gets through even while that upstream bug is unresolved.
    Returns (info, mode_used) so callers can keep later requests (like the
    actual download) pinned to whichever client mode actually worked."""
    mode = mode or default_client_mode()
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=download), mode
    except Exception as e:
        if mode == "auth" and "reload" in str(e).lower():
            fallback_opts = {**ydl_opts, **client_opts("anon")}
            fallback_opts.pop("cookiefile", None)
            with yt_dlp.YoutubeDL(fallback_opts) as ydl:
                return ydl.extract_info(url, download=download), "anon"
        raise

# Lets yt-dlp pull fragmented (DASH/HLS) streams over several connections at
# once instead of one fragment at a time — the single biggest download-speed
# win available here.
CONCURRENT_FRAGMENTS = 5

# Optional post-download re-encode for video. "original" skips compression
# entirely (fast remux only). The other two run the finished mp4 back
# through ffmpeg with a higher CRF (more compression) and, for "smaller",
# also cap the resolution. Audio is copied instead of re-encoded whenever
# the source track is already AAC, since re-encoding AAC->AAC only costs
# time and quality for no size benefit.
COMPRESS_PRESETS = {
    "balanced": {"crf": 25, "preset": "veryfast", "audio_bitrate": "128k", "max_height": None},
    "smaller": {"crf": 29, "preset": "veryfast", "audio_bitrate": "96k", "max_height": 720},
}


def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()


def unique_output_path(title, ext):
    """Pick a filename that doesn't collide with an existing file or an
    in-flight download, appending ' (1)', ' (2)', etc. as needed."""
    base = sanitize_filename(title) or "file"
    ext = (ext or "bin").lstrip(".").strip() or "bin"
    with FILENAME_LOCK:
        candidate_name = f"{base}.{ext}"
        counter = 1
        while (Path(OUTPUT_DIR) / candidate_name).exists() or candidate_name in RESERVED_NAMES:
            candidate_name = f"{base} ({counter}).{ext}"
            counter += 1
        RESERVED_NAMES.add(candidate_name)
    return Path(OUTPUT_DIR) / candidate_name


def set_job(job_id, payload):
    payload = dict(payload)
    payload["_ts"] = time.time()
    with JOBS_LOCK:
        JOBS[job_id] = payload
        # Opportunistic cleanup so JOBS doesn't grow forever on a long-lived
        # process — cheap since it only scans on writes.
        if len(JOBS) > 200:
            cutoff = time.time() - JOB_TTL_SECONDS
            for jid in [j for j, v in JOBS.items() if v.get("_ts", 0) < cutoff]:
                stale = JOBS.pop(jid, None)
                stale_path = (stale or {}).get("path")
                if stale_path:
                    Path(stale_path).unlink(missing_ok=True)


def get_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id, {"status": "unknown"})
        return {k: v for k, v in job.items() if k != "_ts"}


PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stub</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #14171f;
    --ticket: #1d212b;
    --ticket-edge: #333a4a;
    --gold: #e8b64b;
    --gold-dim: #a9853c;
    --red: #c1443c;
    --ivory: #f2ead9;
    --muted: #9b917f;
    --muted-dim: #66604f;
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    background: var(--bg);
    color: var(--ivory);
    font-family: 'IBM Plex Sans', sans-serif;
    -webkit-font-smoothing: antialiased;
    min-height: 100vh;
    padding: 44px 16px 60px;
  }

  ::selection { background: rgba(232,182,75,0.3); color: var(--ivory); }

  .wrap { max-width: 560px; margin: 0 auto; }

  .bulbs {
    height: 12px;
    background-image: radial-gradient(circle, var(--gold) 2.4px, transparent 2.8px);
    background-size: 22px 12px;
    background-repeat: repeat-x;
    background-position: center;
    margin-bottom: -1px;
  }

  .marquee {
    background: var(--gold);
    color: #241a06;
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    padding: 12px 22px 10px;
    border-radius: 10px 10px 0 0;
  }
  .marquee .brand {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 28px;
    letter-spacing: 3px;
  }
  .marquee .admit {
    font-size: 11px;
    letter-spacing: 1.5px;
    color: #5c4315;
  }

  .perf {
    height: 17px;
    background-image: radial-gradient(circle, var(--bg) 7px, transparent 7.3px);
    background-size: 26px 17px;
    background-repeat: repeat-x;
    background-position: center top;
    background-color: var(--gold);
  }

  .ticket {
    background: var(--ticket);
    border: 1px solid var(--ticket-edge);
    border-top: none;
    border-radius: 0 0 10px 10px;
    padding: 26px 26px 22px;
  }

  .tagline {
    color: var(--muted);
    font-size: 14.5px;
    margin: 0 0 22px;
    line-height: 1.5;
  }

  .field-row {
    display: flex;
    gap: 10px;
  }

  .field-row input {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--ticket-edge);
    border-radius: 7px;
    color: var(--ivory);
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14.5px;
    padding: 12px 14px;
    outline: none;
    transition: border-color 0.15s ease;
  }
  .field-row input::placeholder { color: var(--muted-dim); }
  .field-row input:focus { border-color: var(--gold); }

  .btn {
    border: none;
    border-radius: 7px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    font-size: 14px;
    cursor: pointer;
    transition: filter 0.15s ease, background 0.15s ease, color 0.15s ease;
  }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn:focus-visible { outline: 2px solid var(--gold); outline-offset: 2px; }

  .btn-fetch {
    background: var(--gold);
    color: #241a06;
    padding: 12px 20px;
    white-space: nowrap;
  }
  .btn-fetch:hover:not(:disabled) { filter: brightness(1.08); }

  .btn-get {
    background: transparent;
    border: 1px solid var(--ticket-edge);
    color: var(--ivory);
    padding: 6px 14px;
    font-size: 13px;
  }
  .btn-get:hover:not(:disabled) { border-color: var(--gold); color: var(--gold); }

  .meta {
    display: none;
    margin-top: 14px;
    font-size: 13.5px;
    color: var(--muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .switches {
    display: none;
    gap: 18px;
    margin-top: 22px;
    padding-top: 18px;
    border-top: 1px dashed var(--ticket-edge);
  }
  .switch-group { display: flex; gap: 6px; }
  .switch-btn {
    background: transparent;
    border: 1px solid var(--ticket-edge);
    color: var(--muted);
    border-radius: 999px;
    padding: 6px 14px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 12.5px;
    font-weight: 500;
    cursor: pointer;
    transition: border-color 0.15s ease, color 0.15s ease, background 0.15s ease;
  }
  .switch-btn.active { background: rgba(232,182,75,0.12); border-color: var(--gold); color: var(--gold); }
  .switch-btn:hover:not(.active) { color: var(--ivory); }
  .switch-btn:focus-visible { outline: 2px solid var(--gold); outline-offset: 2px; }

  .results {
    margin-top: 18px;
    display: none;
    flex-direction: column;
    gap: 2px;
  }

  .result-row {
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 10px 0;
  }
  .result-row .label { font-size: 14.5px; color: var(--ivory); flex-shrink: 0; }
  .result-row .note { font-size: 11.5px; color: var(--muted-dim); flex-shrink: 0; }
  .result-row .leader {
    flex: 1;
    overflow: hidden;
    white-space: nowrap;
    color: var(--muted-dim);
    font-size: 13px;
  }
  .result-row .leader::after {
    content: " . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ";
  }
  .result-row .size { font-size: 13px; color: var(--muted); flex-shrink: 0; }

  .progress-box {
    display: none;
    margin-top: 18px;
    padding-top: 18px;
    border-top: 1px dashed var(--ticket-edge);
  }
  .progress-line { font-size: 13.5px; color: var(--muted); margin-bottom: 10px; }
  .progress-track {
    height: 8px;
    background: var(--bg);
    border-radius: 999px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%; width: 0%;
    background: var(--gold);
    transition: width 0.4s ease;
    border-radius: 999px;
  }
  .progress-pct { font-size: 12px; color: var(--muted); margin-top: 6px; text-align: right; }

  .result-note {
    margin-top: 18px;
    padding-top: 18px;
    border-top: 1px dashed var(--ticket-edge);
    display: none;
    align-items: center;
    gap: 12px;
    font-size: 14px;
  }
  .result-note.ok { color: var(--gold); }
  .result-note.err { color: var(--red); }
  .result-note .get-link {
    margin-left: auto;
    color: #241a06;
    background: var(--gold);
    text-decoration: none;
    border-radius: 7px;
    padding: 7px 15px;
    font-size: 13px;
    font-weight: 600;
    flex-shrink: 0;
  }
  .result-note .get-link:hover { filter: brightness(1.08); }

  .credit {
    text-align: center;
    margin: 18px auto 0;
    color: var(--muted-dim);
    font-size: 12px;
  }
  .credit span { color: var(--muted); }

  @media (max-width: 560px) {
    .field-row { flex-direction: column; }
    .btn-fetch { width: 100%; }
    .marquee .brand { font-size: 24px; }
    .result-note { flex-wrap: wrap; }
    .result-note .get-link { margin-left: 0; width: 100%; text-align: center; }
  }
</style>
</head>
<body>

<div class="wrap">
  <div class="bulbs"></div>
  <div class="marquee">
    <span class="brand">STUB</span>
    <span class="admit">ADMIT ONE</span>
  </div>
  <div class="perf"></div>

  <div class="ticket">
    <p class="tagline">Paste a link from any video site, or a direct file link. Pick a format and it's yours.</p>

    <div class="field-row">
      <input type="text" id="url" placeholder="Paste your link" autocomplete="off">
      <button class="btn btn-fetch" id="loadBtn" onclick="fetchFormats()">Fetch</button>
    </div>

    <div class="meta" id="meta"></div>

    <div class="switches" id="kindToggle">
      <div class="switch-group">
        <button class="switch-btn active" id="videoTab" onclick="switchKind('video')">Video</button>
        <button class="switch-btn" id="audioTab" onclick="switchKind('audio')">Audio only</button>
      </div>
    </div>

    <div class="switches" id="compressToggle">
      <div class="switch-group">
        <button class="switch-btn active" id="compressOriginal" onclick="switchCompress('original')">Original</button>
        <button class="switch-btn" id="compressBalanced" onclick="switchCompress('balanced')">Balanced</button>
        <button class="switch-btn" id="compressSmaller" onclick="switchCompress('smaller')">Smaller</button>
      </div>
    </div>

    <div class="results" id="results"></div>

    <div class="progress-box" id="progressBox">
      <div class="progress-line" id="progressLine">Starting…</div>
      <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
      <div class="progress-pct" id="progressPct">0%</div>
    </div>

    <div class="result-note" id="resultNote"></div>
  </div>
</div>

<p class="credit">Made by <span>الشيخ قطة</span></p>

<script>
const el = id => document.getElementById(id);
let currentUrl = '';
let currentTitle = '';
let currentClientMode = null;
let videoFormats = [];
let audioFormats = [];
let activeKind = 'video';
let isDirect = false;
let directFormat = null;
let compressLevel = 'original';

async function fetchFormats() {
  const url = el('url').value.trim();
  const loadBtn = el('loadBtn');
  el('resultNote').style.display = 'none';
  el('results').style.display = 'none';
  el('results').innerHTML = '';
  el('meta').style.display = 'none';
  el('kindToggle').style.display = 'none';
  el('compressToggle').style.display = 'none';

  if (!url) { showResult(false, 'Paste a link first.'); return; }

  loadBtn.disabled = true;
  loadBtn.textContent = 'Reading…';
  try {
    const res = await fetch('/formats', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({url})
    });
    const data = await res.json();
    if (data.error) { showResult(false, data.error); return; }

    if (data.title) {
      el('meta').textContent = data.title;
      el('meta').style.display = 'block';
    }

    currentUrl = url;
    currentTitle = data.title || '';
    currentClientMode = data.client_mode || null;
    isDirect = !!data.is_direct;

    if (isDirect) {
      directFormat = data.direct_format;
      renderDirectResult();
    } else {
      videoFormats = data.video_formats || [];
      audioFormats = data.audio_formats || [];
      activeKind = 'video';
      compressLevel = 'original';
      setCompressButtons();
      el('videoTab').classList.add('active');
      el('audioTab').classList.remove('active');
      el('kindToggle').style.display = 'flex';
      renderResults();
    }
  } catch (e) {
    showResult(false, 'Could not reach that source — check the link and try again.');
  } finally {
    loadBtn.disabled = false;
    loadBtn.textContent = 'Fetch';
  }
}

function switchKind(kind) {
  activeKind = kind;
  el('videoTab').classList.toggle('active', kind === 'video');
  el('audioTab').classList.toggle('active', kind === 'audio');
  renderResults();
}

function switchCompress(level) {
  compressLevel = level;
  setCompressButtons();
}

function setCompressButtons() {
  el('compressOriginal').classList.toggle('active', compressLevel === 'original');
  el('compressBalanced').classList.toggle('active', compressLevel === 'balanced');
  el('compressSmaller').classList.toggle('active', compressLevel === 'smaller');
}

function renderResults() {
  el('compressToggle').style.display = activeKind === 'video' ? 'flex' : 'none';

  const list = el('results');
  list.innerHTML = '';
  const items = activeKind === 'video' ? videoFormats : audioFormats;

  if (!items.length) {
    const row = document.createElement('div');
    row.className = 'result-row';
    row.innerHTML = `<span class="label">No ${activeKind} formats available for this source.</span>`;
    list.appendChild(row);
    list.style.display = 'flex';
    return;
  }

  items.forEach(f => {
    const row = document.createElement('div');
    row.className = 'result-row';
    const resText = activeKind === 'video' ? f.resolution : (f.abr ? f.abr + ' kbps' : 'Audio');
    const badge = activeKind === 'video' ? 'mp4' : f.ext;
    const noteText = (activeKind === 'video' && f.codec_note) ? f.codec_note.replace(' · ', '') : '';
    const sizeText = f.size ? `${f.size_approx ? '~' : ''}${f.size}` : 'size unknown';
    row.innerHTML = `
      <span class="label">${resText}</span>
      ${noteText ? `<span class="note">${noteText}</span>` : ''}
      <span class="leader"></span>
      <span class="size">${sizeText} · ${badge}</span>
      <button class="btn btn-get">Get</button>
    `;
    row.querySelector('button').addEventListener('click', () =>
      startDownload(currentUrl, f.format_id, f.has_audio, activeKind, currentTitle, null,
        activeKind === 'video' ? compressLevel : 'original', currentClientMode));
    list.appendChild(row);
  });
  list.style.display = 'flex';
}

function renderDirectResult() {
  const list = el('results');
  list.innerHTML = '';
  const row = document.createElement('div');
  row.className = 'result-row';
  const badge = (directFormat.ext || 'file').toLowerCase();
  const resText = directFormat.label || 'Direct file';
  const sizeText = directFormat.size ? `${directFormat.size_approx ? '~' : ''}${directFormat.size}` : 'size unknown';
  row.innerHTML = `
    <span class="label">${resText}</span>
    <span class="leader"></span>
    <span class="size">${sizeText} · ${badge}</span>
    <button class="btn btn-get">Get</button>
  `;
  row.querySelector('button').addEventListener('click', () =>
    startDownload(currentUrl, '__direct__', false, 'direct', currentTitle, directFormat.ext, 'original', null));
  list.appendChild(row);
  list.style.display = 'flex';
}

async function startDownload(url, formatId, hasAudio, kind, title, ext, compress, clientMode) {
  el('resultNote').style.display = 'none';
  el('progressBox').style.display = 'block';
  el('progressLine').textContent = 'Starting…';
  el('progressFill').style.width = '0%';
  el('progressPct').textContent = '0%';

  try {
    const res = await fetch('/download', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({url, format_id: formatId, has_audio: hasAudio, kind: kind, title: title, ext: ext, compress: compress || 'original', client_mode: clientMode})
    });
    const data = await res.json();
    if (data.error) { finishProgress(false, data.error); return; }
    pollProgress(data.job_id);
  } catch (e) {
    finishProgress(false, 'Could not start the download.');
  }
}

function pollProgress(jobId) {
  const interval = setInterval(async () => {
    const res = await fetch('/progress/' + jobId);
    const data = await res.json();
    if (data.status === 'downloading') {
      const pctText = (data.percent || '0%').trim();
      el('progressLine').textContent = 'Downloading…';
      el('progressPct').textContent = pctText;
      const num = parseFloat(pctText);
      if (!isNaN(num)) el('progressFill').style.width = num + '%';
    } else if (data.status === 'processing') {
      el('progressLine').textContent = 'Finishing up…';
    } else if (data.status === 'compressing') {
      const pctText = (data.percent || '').trim();
      el('progressLine').textContent = 'Compressing…';
      el('progressPct').textContent = pctText || '…';
      const num = parseFloat(pctText);
      if (!isNaN(num)) el('progressFill').style.width = num + '%';
    } else if (data.status === 'finished') {
      el('progressFill').style.width = '100%';
      el('progressPct').textContent = '100%';
      clearInterval(interval);
      finishProgress(true, data.filename, data.download_url);
    } else if (data.status === 'error') {
      clearInterval(interval);
      finishProgress(false, data.error);
    }
  }, 1000);
}

function finishProgress(ok, message, downloadUrl) {
  el('progressBox').style.display = 'none';
  showResult(ok, message, downloadUrl);
}

function showResult(ok, message, downloadUrl) {
  const r = el('resultNote');
  r.className = 'result-note ' + (ok ? 'ok' : 'err');
  r.innerHTML = '';

  const text = document.createElement('span');
  text.textContent = (ok ? 'Saved as ' : 'Error: ') + message;
  r.appendChild(text);

  if (ok && downloadUrl) {
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.textContent = 'get file';
    link.className = 'get-link';
    link.setAttribute('download', '');
    r.appendChild(link);
    // Kick off the browser download automatically too, so the person
    // doesn't have to click if they don't need to.
    const auto = document.createElement('iframe');
    auto.style.display = 'none';
    auto.src = downloadUrl;
    document.body.appendChild(auto);
  }

  r.style.display = 'flex';
}

el('url').addEventListener('keydown', e => {
  if (e.key === 'Enter') fetchFormats();
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


def format_size(num_bytes):
    if not num_bytes:
        return None
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def estimate_video_size_bytes(f, duration):
    """Return (bytes, is_approx) for a single video format, falling back to
    a bitrate * duration estimate when yt-dlp doesn't report a real size.
    Video is only remuxed (not re-encoded) to mp4 on download, so the
    source stream's own size/bitrate is still a valid estimate."""
    exact = f.get("filesize") or f.get("filesize_approx")
    if exact:
        return exact, False
    bitrate = f.get("tbr") or ((f.get("vbr") or 0) + (f.get("abr") or 0)) or None
    dur = duration or f.get("duration")
    if bitrate and dur:
        return bitrate * 1000 / 8 * dur, True
    return None, False


def estimate_audio_output_bytes(duration):
    """Audio downloads are always transcoded to a constant TARGET_AUDIO_KBPS
    mp3, regardless of the source stream's own bitrate, so the estimate has
    to be computed from that target, not from the source format's size."""
    if not duration:
        return None, False
    return TARGET_AUDIO_KBPS * 1000 / 8 * duration, True


def probe_direct_url(url):
    """Try to treat `url` as a plain downloadable file (pdf, zip, image,
    mp3, etc.) rather than a video-site link. Returns a dict describing the
    file, or None if it doesn't look like a downloadable file at all."""
    content_type = None
    content_length = None
    content_disposition = ""

    try:
        r = HTTP.head(url, allow_redirects=True, timeout=10)
        if r.status_code < 400:
            content_type = r.headers.get("Content-Type", "").split(";")[0].strip()
            cl = r.headers.get("Content-Length")
            content_length = int(cl) if cl and cl.isdigit() else None
            content_disposition = r.headers.get("Content-Disposition", "")
    except requests.RequestException:
        pass

    # Some servers don't implement HEAD properly — confirm with a streamed
    # GET, closed the instant we've read the headers, so we don't pull the
    # whole body.
    if content_type is None or content_length is None:
        try:
            with HTTP.get(url, stream=True, timeout=10) as r2:
                if r2.status_code >= 400:
                    return None
                if content_type is None:
                    content_type = r2.headers.get("Content-Type", "").split(";")[0].strip()
                if content_length is None:
                    cl = r2.headers.get("Content-Length")
                    content_length = int(cl) if cl and cl.isdigit() else None
                if not content_disposition:
                    content_disposition = r2.headers.get("Content-Disposition", "")
        except requests.RequestException:
            return None

    if content_type and "text/html" in content_type:
        return None

    filename = None
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', content_disposition)
    if m:
        filename = unquote(m.group(1).strip())
    if not filename:
        path_name = Path(urlparse(url).path).name
        if path_name:
            filename = unquote(path_name)

    ext = None
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
    if not ext and content_type:
        guessed = mimetypes.guess_extension(content_type)
        if guessed:
            ext = guessed.lstrip(".")
    ext = ext or "bin"

    title = filename.rsplit(".", 1)[0] if filename and "." in filename else (filename or "file")

    return {
        "title": title,
        "ext": ext,
        "content_type": content_type or "file",
        "size": format_size(content_length),
        "size_bytes": content_length,
    }


@app.route("/formats", methods=["POST"])
def formats():
    url = request.json.get("url", "").strip()
    if not url:
        return jsonify({"error": "The link is empty."}), 400

    try:
        ydl_opts = {
            **base_ydl_opts(),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "socket_timeout": 15,
        }
        info, client_mode = extract_with_fallback(ydl_opts, url, download=False)
        all_formats = info.get("formats") or [info]
        duration = info.get("duration")

        video_candidates = [f for f in all_formats if f.get("vcodec", "none") != "none"]
        video_merged = [f for f in video_candidates if f.get("acodec", "none") != "none"]
        if not video_merged:
            video_merged = video_candidates

        audio_only = [f for f in all_formats if f.get("vcodec", "none") == "none" and f.get("acodec", "none") != "none"]
        audio_only.sort(key=lambda f: f.get("abr") or 0, reverse=True)
        best_audio_bytes, best_audio_approx = (None, False)
        if audio_only:
            best_audio_bytes, best_audio_approx = estimate_video_size_bytes(audio_only[0], duration)

        # Sort so that, at each resolution, an H.264 (avc1) stream is
        # picked over VP9/AV1 when both exist. H.264 plays reliably in
        # mobile editing apps (CapCut, InShot, VN, etc.); VP9/AV1 — which
        # is what YouTube serves by default at 1080p+ — often doesn't
        # open cleanly in them, which is the "1080p downloads don't work
        # on mobile" symptom.
        def is_h264(f):
            return (f.get("vcodec") or "").startswith(("avc1", "h264"))

        video_merged.sort(key=lambda f: (is_h264(f), f.get("tbr") or 0), reverse=True)
        seen_heights = set()
        video_result = []
        for f in video_merged:
            height = f.get("height")
            if height in seen_heights:
                continue
            seen_heights.add(height)
            label = f"{height}p" if height else (f.get("resolution") or "N/A")

            size_bytes, is_approx = estimate_video_size_bytes(f, duration)
            has_audio = f.get("acodec", "none") != "none"
            if not has_audio:
                if size_bytes is not None and best_audio_bytes is not None:
                    size_bytes += best_audio_bytes
                    is_approx = is_approx or best_audio_approx
                else:
                    size_bytes = None

            video_result.append({
                "format_id": f.get("format_id", "N/A"),
                "ext": "mp4",
                "resolution": label,
                "has_audio": has_audio,
                "codec_note": "" if is_h264(f) else " · may not open in some mobile editors",
                "size": format_size(size_bytes),
                "size_approx": is_approx,
            })
        video_result.sort(
            key=lambda r: int(r["resolution"][:-1]) if r["resolution"].endswith("p") and r["resolution"][:-1].isdigit() else 0,
            reverse=True,
        )

        seen = set()
        audio_result = []
        audio_duration = duration or (audio_only[0].get("duration") if audio_only else None)
        audio_size_bytes, audio_size_approx = estimate_audio_output_bytes(audio_duration)
        for f in audio_only:
            key = (f.get("ext"), f.get("abr"))
            if key in seen:
                continue
            seen.add(key)
            audio_result.append({
                "format_id": f.get("format_id", "N/A"),
                "ext": "mp3",
                "abr": round(f.get("abr")) if f.get("abr") else None,
                "has_audio": True,
                "size": format_size(audio_size_bytes),
                "size_approx": audio_size_approx,
            })

        return jsonify({
            "title": info.get("title", ""),
            "is_direct": False,
            "video_formats": video_result,
            "audio_formats": audio_result,
            "client_mode": client_mode,
        })
    except Exception as e:
        direct = probe_direct_url(url)
        if direct:
            return jsonify({
                "title": direct["title"],
                "is_direct": True,
                "direct_format": {
                    "format_id": "__direct__",
                    "ext": direct["ext"],
                    "label": direct["content_type"],
                    "size": direct["size"],
                    "size_approx": False,
                },
            })

        msg = str(e)
        if "Network is unreachable" in msg or "Failed to resolve" in msg or "Connection refused" in msg:
            msg = ("Could not reach that source from this server. If you're on "
                   "PythonAnywhere's free tier, this is their outbound network "
                   "restriction blocking the request, not a bug in the app.")
        return jsonify({"error": msg}), 500


def probe_duration_seconds(path):
    """Seconds of media in `path`, via ffprobe. Returns None if unavailable —
    compression still runs, just without a percent."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        return float(out.stdout.strip())
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        return None


def probe_audio_codec(path):
    """Codec name of the first audio stream, via ffprobe. Returns None if
    unavailable/no audio track."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name", "-of",
             "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        codec = out.stdout.strip()
        return codec or None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def compress_video(input_path, output_path, preset_name, job_id, duration=None):
    """Re-encode input_path into output_path at a smaller size using the
    given COMPRESS_PRESETS entry, updating the job's progress as it goes.
    Raises RuntimeError on failure."""
    preset = COMPRESS_PRESETS[preset_name]

    # Skip re-encoding the audio track entirely when it's already AAC and
    # we're not changing the frame size — copying is both faster and
    # avoids a needless quality/size hit from a second lossy pass.
    audio_codec = probe_audio_codec(input_path)
    can_copy_audio = audio_codec == "aac"

    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-c:v", "libx264", "-crf", str(preset["crf"]), "-preset", preset["preset"],
        "-threads", "0",
    ]
    if can_copy_audio:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "aac", "-b:a", preset["audio_bitrate"]]
    cmd += ["-movflags", "+faststart"]
    if preset["max_height"]:
        cmd += ["-vf", f"scale=-2:'min({preset['max_height']},ih)'"]
    cmd += ["-progress", "pipe:1", "-nostats", str(output_path)]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_ms="):
            try:
                done_seconds = int(line.split("=", 1)[1]) / 1_000_000
                if duration:
                    pct = min(99.9, done_seconds / duration * 100)
                    set_job(job_id, {"status": "compressing", "percent": f"{pct:.1f}%"})
                else:
                    set_job(job_id, {"status": "compressing", "percent": ""})
            except ValueError:
                pass
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("Compression failed (ffmpeg error).")


def run_download(job_id, url, format_id, has_audio, kind, output_path, compress="original", client_mode=None):
    def progress_hook(d):
        if d["status"] == "downloading":
            percent = d.get("_percent_str", "").strip()
            set_job(job_id, {"status": "downloading", "percent": percent})
        elif d["status"] == "finished":
            set_job(job_id, {"status": "processing"})

    stem = str(output_path.with_suffix(""))
    outtmpl = f"{stem}.%(ext)s"

    # Pinned to the exact client mode that produced this format_id (see
    # /formats), since a format ID from one YouTube client isn't
    # necessarily valid to request from another.
    common_opts = {
        **base_ydl_opts(client_mode),
        "outtmpl": outtmpl,
        "noplaylist": True,
        "ignoreerrors": True,
        "no_warnings": True,
        "windowsfilenames": True,
        "progress_hooks": [progress_hook],
        "concurrent_fragment_downloads": CONCURRENT_FRAGMENTS,
        "socket_timeout": 20,
        "retries": 3,
    }

    try:
        if kind == "audio":
            ydl_opts = {
                **common_opts,
                "format": format_id,
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": str(TARGET_AUDIO_KBPS),
                }],
            }
        else:
            format_spec = format_id if has_audio else f"{format_id}+bestaudio/best"
            ydl_opts = {
                **common_opts,
                "format": format_spec,
                "merge_output_format": "mp4",
                "postprocessors": [{"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}],
            }

        extract_with_fallback(ydl_opts, url, download=True, mode=client_mode)

        if kind == "video" and compress != "original" and output_path.exists():
            set_job(job_id, {"status": "compressing", "percent": ""})
            tmp_path = output_path.with_name(output_path.stem + ".compressing.mp4")
            duration = probe_duration_seconds(output_path)
            try:
                compress_video(output_path, tmp_path, compress, job_id, duration)
                tmp_path.replace(output_path)
            finally:
                tmp_path.unlink(missing_ok=True)

        set_job(job_id, {
            "status": "finished",
            "filename": output_path.name,
            "path": str(output_path),
            "download_url": f"/file/{job_id}",
        })
    except Exception as e:
        set_job(job_id, {"status": "error", "error": str(e)})
    finally:
        RESERVED_NAMES.discard(output_path.name)


def run_direct_download(job_id, url, output_path):
    """Stream a plain file straight to disk (no yt-dlp involved) for links
    that aren't a supported video/audio site."""
    try:
        with HTTP.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = r.headers.get("Content-Length")
            total = int(total) if total and total.isdigit() else None
            downloaded = 0
            set_job(job_id, {"status": "downloading", "percent": "0%"})
            with open(output_path, "wb") as out:
                for chunk in r.iter_content(chunk_size=512 * 1024):
                    if not chunk:
                        continue
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        set_job(job_id, {"status": "downloading", "percent": f"{pct:.1f}%"})
                    else:
                        set_job(job_id, {"status": "downloading", "percent": format_size(downloaded) or "…"})
        set_job(job_id, {
            "status": "finished",
            "filename": output_path.name,
            "path": str(output_path),
            "download_url": f"/file/{job_id}",
        })
    except Exception as e:
        set_job(job_id, {"status": "error", "error": str(e)})
    finally:
        RESERVED_NAMES.discard(output_path.name)


@app.route("/download", methods=["POST"])
def download():
    data = request.json
    url = data.get("url", "").strip()
    format_id = data.get("format_id", "").strip()
    has_audio = bool(data.get("has_audio", False))
    kind = data.get("kind", "video")
    title = data.get("title", "").strip()
    compress = data.get("compress", "original")
    if compress not in COMPRESS_PRESETS:
        compress = "original"
    client_mode = data.get("client_mode")
    if client_mode not in ("auth", "anon"):
        client_mode = None
    if not url or not format_id:
        return jsonify({"error": "Missing data."}), 400

    job_id = f"{int(time.time() * 1000)}-{threading.get_ident()}"
    set_job(job_id, {"status": "starting"})

    if kind == "direct":
        ext = (data.get("ext") or "bin").strip()
        output_path = unique_output_path(title or "file", ext)
        EXECUTOR.submit(run_direct_download, job_id, url, output_path)
    else:
        ext = "mp3" if kind == "audio" else "mp4"
        output_path = unique_output_path(title or "video", ext)
        EXECUTOR.submit(run_download, job_id, url, format_id, has_audio, kind, output_path, compress, client_mode)

    return jsonify({"job_id": job_id})


@app.route("/progress/<job_id>")
def progress(job_id):
    return jsonify(get_job(job_id))


@app.route("/file/<job_id>")
def get_file(job_id):
    job = get_job(job_id)
    if job.get("status") != "finished" or not job.get("path"):
        abort(404)
    path = Path(job["path"])
    if not path.exists():
        abort(404)
    return send_file(path, as_attachment=True, download_name=job.get("filename") or path.name)


if __name__ == "__main__":
    # Local development only — in production this file is served by
    # gunicorn (see Dockerfile), which never runs this block.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
