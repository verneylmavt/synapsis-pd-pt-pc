(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const ACTIVE = new Set(['starting', 'running', 'reconnecting', 'stopping']);
  const STORAGE_KEY = 'synapsis.selection';
  const state = {
    sources: [], source: null, areas: [], runs: [], run: null, sourceActive: null,
    areaId: null, points: [], selectedPoint: -1, undo: [], preview: null, previewUrl: null,
    sourceEpoch: 0, selectionEpoch: 0, sourceController: null, pollController: null,
    timer: null, failures: 0, busy: false, lastStats: null, upload: null,
    listEpoch: 0, sourceListController: null, lastHistory: null, lastForecast: null
  };
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); } catch { /* Storage is optional. */ }
  const copyPoints = points => points.map(point => ({ x: point.x, y: point.y }));
  const active = run => Boolean(run && ACTIVE.has(run.status));
  const locked = () => active(state.sourceActive) || active(state.run);
  const selectedArea = () => state.areas.find(area => String(area.id) === String(state.areaId));
  const statusLabel = status => String(status || 'No run').replace(/^./, char => char.toUpperCase());
  const valueOrDash = value => value == null ? '—' : String(value);

  function element(tag, text, attributes = {}) {
    const node = document.createElement(tag);
    if (text != null) node.textContent = text;
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    return node;
  }
  function message(id, text, kind = '') {
    $(id).textContent = text;
    $(id).className = `form-message ${kind}`;
  }
  function notice(text, kind = '') {
    $('notice').textContent = text;
    $('notice').className = `notice ${kind}`;
    $('notice').hidden = !text;
  }
  function safeError(error) {
    return String(error.message || 'Please try again.').replace(/(?:rtsp|https?):\/\/\S+/gi, '[stream URL]');
  }
  async function request(url, { signal, method = 'GET', body } = {}) {
    const response = await fetch(url, {
      signal, method, headers: body ? { 'Content-Type': 'application/json' } : {},
      ...(body ? { body: JSON.stringify(body) } : {})
    });
    if (!response.ok) {
      let detail;
      try { detail = (await response.json()).detail; } catch { /* Non-JSON failures. */ }
      if (Array.isArray(detail)) detail = detail.map(item => item.msg).join('; ');
      const error = new Error(typeof detail === 'string' ? detail : `Request failed (${response.status}).`);
      error.status = response.status;
      throw error;
    }
    return response.json();
  }
  function persist() {
    saved = { sourceId: state.source?.id, areaId: state.areaId, runId: state.run?.id };
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(saved)); } catch { /* Private browsing may block storage. */ }
  }
  function stopPolling() {
    clearTimeout(state.timer);
    state.timer = null;
    state.pollController?.abort();
    state.pollController = null;
    state.selectionEpoch++;
  }
  function drawSourceList() {
    $('sourceCount').textContent = state.sources.length;
    $('sourceList').replaceChildren();
    if (!state.sources.length) $('sourceList').append(element('p', 'No sources yet. Add your first video or camera.', { class: 'hint' }));
    for (const source of state.sources) {
      const button = element('button', null, { type: 'button', class: 'source-item', 'aria-current': String(source.id === state.source?.id) });
      button.append(element('span', source.kind === 'live' ? '◉' : '▷', { class: 'source-icon', 'aria-hidden': 'true' }));
      const text = element('span');
      text.append(element('span', source.name, { class: 'source-name' }), element('span', source.kind === 'live' ? 'Live camera' : 'Video recording', { class: 'source-kind' }));
      button.append(text);
      button.disabled = !source.enabled;
      button.addEventListener('click', () => selectSource(source.id));
      $('sourceList').append(button);
    }
  }
  async function loadSources(preferredId) {
    const epoch = state.sourceEpoch, listEpoch = ++state.listEpoch;
    state.sourceListController?.abort();
    const controller = new AbortController(); state.sourceListController = controller;
    try {
      const sources = await request('/api/video-sources', {signal: controller.signal});
      if (epoch !== state.sourceEpoch || listEpoch !== state.listEpoch) return;
      state.sources = sources;
      drawSourceList();
      const id = preferredId ?? state.source?.id ?? saved.sourceId;
      const source = state.sources.find(item => String(item.id) === String(id) && item.enabled) || state.sources.find(item => item.enabled);
      if (source) await selectSource(source.id);
      else notice('Add a video or live camera to begin.');
    } catch (error) { if (error.name !== 'AbortError' && epoch === state.sourceEpoch && listEpoch === state.listEpoch) notice(`Sources could not be loaded. ${safeError(error)}`, 'error'); }
  }
  function clearPreview() {
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = null;
    state.preview = null;
    $('sourcePicture').hidden = true;
    $('sourcePicture').removeAttribute('src');
    $('polygonCanvas').hidden = true;
    $('streamImage').hidden = true;
    $('streamImage').removeAttribute('src');
    $('mediaEmpty').hidden = false;
    $('mediaCaption').hidden = true;
  }
  function resetAnalytics() {
    state.lastStats = null;
    state.lastHistory = null; state.lastForecast = null;
    for (const id of ['occupancyValue', 'entriesValue', 'exitsValue', 'netValue']) $(id).textContent = '—';
    $('occupancyNote').textContent = 'No current observation';
    $('historyMessage').textContent = 'Start a run to see its observed history.';
    $('forecastMessage').textContent = 'A forecast needs enough complete history from one run.';
    $('forecastMethod').textContent = '';
    emptyTable('historyBody', 4, 'No history yet.');
    emptyTable('forecastBody', 3, 'Not enough history yet.');
    emptyTable('eventsBody', 3, 'No confirmed crossings yet.');
    $('eventsCount').textContent = '0 events';
    $('historyChart').replaceChildren(); $('forecastChart').replaceChildren();
    $('updatedAt').textContent = 'Waiting for data';
  }
  async function selectSource(id) {
    stopPolling();
    state.sourceListController?.abort();
    state.sourceController?.abort();
    state.sourceController = new AbortController();
    const signal = state.sourceController.signal;
    const epoch = ++state.sourceEpoch;
    state.source = state.sources.find(source => source.id === id);
    state.run = null; state.sourceActive = null; state.runs = []; state.areas = []; state.areaId = null;
    state.points = []; state.undo = []; state.selectedPoint = -1; state.busy = false;
    clearPreview(); resetAnalytics(); drawSourceList(); renderAreaOptions(); renderRunOptions(); renderEditor(); renderRun();
    $('sourceHeading').textContent = state.source.name;
    const metadata = [state.source.kind === 'live' ? 'Live camera' : 'Video recording'];
    if (state.source.width && state.source.height) metadata.push(`${state.source.width} × ${state.source.height}`);
    if (state.source.fps) metadata.push(`${Number(state.source.fps).toFixed(1)} FPS`);
    $('sourceMeta').textContent = metadata.join(' · ');
    $('retryPreview').disabled = false;
    notice('Loading areas and processing runs…');
    loadPreview(epoch, signal);
    try {
      const [areas, runs] = await Promise.all([
        request(`/api/areas?video_source_id=${id}`, { signal }), request(`/api/video-sources/${id}/runs`, { signal })
      ]);
      if (epoch !== state.sourceEpoch) return;
      state.areas = areas; state.runs = runs;
      state.sourceActive = runs.find(active) || null;
      state.run = state.sourceActive || runs.find(run => run.id === saved.runId) || runs[0] || null;
      const preferred = areas.find(area => String(area.id) === String(saved.areaId)) || areas.find(area => area.active) || areas[0];
      state.areaId = preferred?.id ?? null;
      renderAreaOptions(); renderRunOptions(); loadAreaEditor(); applySettings(state.run); renderRun(); persist();
      notice(areas.length ? '' : 'Define an active area before starting processing.');
      if (state.run) refreshAnalytics();
    } catch (error) {
      if (error.name !== 'AbortError' && epoch === state.sourceEpoch) notice(`Source details could not be loaded. ${safeError(error)}`, 'error');
    }
  }
  async function loadPreview(epoch = state.sourceEpoch, signal = state.sourceController?.signal) {
    const sourceId = state.source?.id;
    if (!sourceId) return;
    try {
      const response = await fetch(`/api/preview/${sourceId}`, { signal, cache: 'no-store' });
      if (!response.ok) throw new Error('The source preview is unavailable.');
      const blob = await response.blob();
      if (epoch !== state.sourceEpoch) return;
      const url = URL.createObjectURL(blob);
      const image = new Image();
      await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = () => reject(new Error('The preview image could not be read.')); image.src = url; });
      if (epoch !== state.sourceEpoch || signal?.aborted) { URL.revokeObjectURL(url); return; }
      if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
      state.previewUrl = url; state.preview = image;
      $('sourcePicture').src = url; $('sourcePicture').hidden = false;
      $('polygonCanvas').width = image.naturalWidth; $('polygonCanvas').height = image.naturalHeight;
      $('mediaStage').style.aspectRatio = `${image.naturalWidth}/${image.naturalHeight}`;
      $('mediaEmpty').hidden = true;
      drawPolygon(); renderRun();
    } catch (error) {
      if (error.name !== 'AbortError' && epoch === state.sourceEpoch) notice(`Preview unavailable. Check the source connection and try Refresh preview. ${safeError(error)}`, 'error');
    }
  }
  function renderAreaOptions() {
    $('areaSelect').replaceChildren();
    if (!state.areas.length) $('areaSelect').append(element('option', 'Create your first area', { value: '' }));
    for (const area of state.areas) $('areaSelect').append(element('option', `${area.name}${area.active ? '' : ' (inactive)'}`, { value: area.id }));
    $('areaSelect').disabled = !state.areas.length;
    $('areaSelect').value = state.areaId ?? '';
  }
  function renderRunOptions() {
    $('runSelect').replaceChildren();
    if (!state.runs.length) $('runSelect').append(element('option', 'No runs yet', { value: '' }));
    for (const run of state.runs) {
      const when = run.started_at ? new Date(run.started_at).toLocaleString() : String(run.id).slice(0, 8);
      $('runSelect').append(element('option', `${statusLabel(run.status)} · ${when}`, { value: run.id }));
    }
    $('runSelect').disabled = !state.runs.length;
    $('runSelect').value = state.run?.id ?? '';
  }
  function renderRun() {
    const run = state.run;
    $('runStatus').textContent = run ? statusLabel(run.status) : 'No run';
    $('runStatus').className = `run-status ${run?.status || ''}`;
    $('startRun').disabled = !state.source || !state.areas.some(area => area.active) || locked() || state.busy;
    $('stopRun').disabled = !active(run) || run?.status === 'stopping' || state.busy;
    $('processingSettings').disabled = locked() || state.busy;
    $('areaSelect').disabled = !state.areas.length || state.busy;
    $('runSelect').disabled = !state.runs.length || state.busy;
    $('areaEditor').disabled = !state.source || locked() || state.busy;
    $('saveArea').disabled = !state.source || locked() || state.busy;
    $('polygonCanvas').classList.toggle('locked', locked());
    $('editorHint').textContent = locked()
      ? 'Area changes are locked while a run is active. Stop processing before editing.'
      : 'Changes apply to the next run. Click the preview to add points, drag to adjust, or enter coordinates below. Coordinates are normalized from 0 to 1.';
    renderAreaSnapshotNote();
    if (run) {
      const fps = run.fps == null ? '—' : Number(run.fps).toFixed(1);
      const frame = run.last_frame_index ?? '—';
      const dropped = run.dropped_frames ?? 0;
      const heartbeat = run.heartbeat_at ? new Date(run.heartbeat_at).toLocaleTimeString('en-GB', {timeZone:'UTC'}) : null;
      $('runDetail').textContent = `Frame ${frame} · ${fps} processed FPS · ${dropped} dropped${heartbeat ? ` · Heartbeat ${heartbeat} UTC` : ''}${run.error_code ? ` · ${String(run.error_code).replace(/_/g, ' ')}` : ''}`;
      if (active(run)) {
        const stream = `/api/runs/${encodeURIComponent(run.id)}/stream`;
        if ($('streamImage').getAttribute('src') !== stream) $('streamImage').src = stream;
        $('streamImage').hidden = false; $('sourcePicture').hidden = true; $('polygonCanvas').hidden = true;
        $('mediaEmpty').hidden = true;
        $('mediaCaption').textContent = `${statusLabel(run.status)} · ${state.source?.kind === 'live' ? 'Live stream' : 'Video playback'}`;
        $('mediaCaption').hidden = false;
      } else {
        $('streamImage').hidden = true; $('streamImage').removeAttribute('src');
        $('sourcePicture').hidden = !state.preview;
        $('mediaCaption').textContent = `${statusLabel(run.status)} · Preview`;
        $('mediaCaption').hidden = !state.preview;
        $('mediaEmpty').hidden = Boolean(state.preview);
        drawPolygon();
        if (state.lastStats) renderStats(state.lastStats);
      }
    } else {
      $('runDetail').textContent = 'Processing starts only when you press Start.';
      drawPolygon();
    }
  }
  function renderAreaSnapshotNote() {
    const current = selectedArea(), snapshots = state.run?.areas;
    const note = $('areaSnapshotNote');
    note.hidden = true; note.textContent = '';
    if (!state.run || state.areaId == null || !Array.isArray(snapshots)) return;
    const snapshot = snapshots.find(area => String(area.id) === String(state.areaId));
    if (!snapshot) {
      note.textContent = 'This area was not included in the selected run. Choose another area or start a new run to see its counts.';
      note.hidden = false; return;
    }
    const different = current && (current.name !== snapshot.name || current.polygon.length !== snapshot.polygon.length || current.polygon.some((point,i) => Math.abs(point.x-snapshot.polygon[i].x)>1e-9 || Math.abs(point.y-snapshot.polygon[i].y)>1e-9));
    if (different) {
      note.textContent = `The current area differs from the saved run area. Counts use “${snapshot.name}” as saved when this run began. Editor changes apply to the next run.`;
      note.hidden = false;
    }
  }
  function loadAreaEditor() {
    const area = selectedArea();
    state.points = copyPoints(area?.polygon || []); state.undo = []; state.selectedPoint = -1;
    $('areaName').value = area?.name || ''; $('areaActive').checked = area?.active ?? true;
    message('polygonMessage', ''); renderEditor();
  }
  function rememberPoints() {
    state.undo.push(copyPoints(state.points));
    if (state.undo.length > 50) state.undo.shift();
  }
  function renderEditor() {
    $('pointCount').textContent = `${state.points.length} points`;
    $('vertexRows').replaceChildren();
    state.points.forEach((point, index) => {
      const row = element('tr', null, { class: index === state.selectedPoint ? 'selected' : '' });
      row.append(element('td', index + 1));
      for (const axis of ['x', 'y']) {
        const cell = element('td');
        const input = element('input', null, { type: 'number', min: 0, max: 1, step: '0.001', 'aria-label': `Point ${index + 1} ${axis.toUpperCase()} coordinate` });
        input.value = Number.isFinite(point[axis]) ? Number(point[axis].toFixed(4)) : '';
        input.addEventListener('focus', () => { state.selectedPoint = index; drawPolygon(); });
        input.addEventListener('change', () => {
          rememberPoints(); state.points[index][axis] = input.value === '' ? NaN : Number(input.value);
          drawPolygon(); message('polygonMessage', 'Unsaved area changes.');
        });
        cell.append(input); row.append(cell);
      }
      const cell = element('td');
      const remove = element('button', 'Remove', { type: 'button', 'aria-label': `Remove point ${index + 1}` });
      remove.addEventListener('click', () => { rememberPoints(); state.points.splice(index, 1); state.selectedPoint = -1; renderEditor(); });
      cell.append(remove); row.append(cell); $('vertexRows').append(row);
    });
    $('undoPoints').disabled = !state.undo.length;
    drawPolygon(); renderRunControlsOnly();
  }
  function renderRunControlsOnly() {
    $('areaEditor').disabled = !state.source || locked() || state.busy;
    $('saveArea').disabled = !state.source || locked() || state.busy;
  }
  function drawPolygon() {
    const canvas = $('polygonCanvas');
    canvas.hidden = !state.preview || active(state.run);
    if (!state.preview) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const points = state.points.filter(point => Number.isFinite(point.x) && Number.isFinite(point.y));
    if (!points.length) return;
    const scale = canvas.width / Math.max(canvas.clientWidth, 1);
    ctx.beginPath();
    points.forEach((point, index) => index ? ctx.lineTo(point.x * canvas.width, point.y * canvas.height) : ctx.moveTo(point.x * canvas.width, point.y * canvas.height));
    if (points.length >= 3) ctx.closePath();
    ctx.fillStyle = 'rgba(8,135,121,.12)'; if (points.length >= 3) ctx.fill();
    ctx.strokeStyle = '#088779'; ctx.lineWidth = 2 * scale; ctx.stroke();
    points.forEach((point, index) => {
      const x = point.x * canvas.width, y = point.y * canvas.height;
      ctx.beginPath(); ctx.arc(x, y, (index === state.selectedPoint ? 6 : 4) * scale, 0, Math.PI * 2);
      ctx.fillStyle = index === state.selectedPoint ? '#2559ad' : '#fff'; ctx.fill();
      ctx.strokeStyle = '#088779'; ctx.lineWidth = 2 * scale; ctx.stroke();
      ctx.font = `600 ${11 * scale}px sans-serif`; ctx.fillStyle = '#173f42';
      ctx.fillText(index + 1, x + 8 * scale, y - 8 * scale);
    });
  }
  function polygonError(points) {
    if (points.length < 3) return 'Add at least three distinct points.';
    if (points.some(point => !Number.isFinite(point.x) || !Number.isFinite(point.y) || point.x < 0 || point.x > 1 || point.y < 0 || point.y > 1)) return 'Every coordinate must be a number between 0 and 1.';
    if (points.some((point, i) => points.some((other, j) => j > i && Math.hypot(point.x - other.x, point.y - other.y) < 1e-9))) return 'Each polygon point must be distinct.';
    const cross = (a, b, c) => (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    const on = (a, b, p) => Math.abs(cross(a,b,p)) < 1e-10 && p.x >= Math.min(a.x,b.x)-1e-10 && p.x <= Math.max(a.x,b.x)+1e-10 && p.y >= Math.min(a.y,b.y)-1e-10 && p.y <= Math.max(a.y,b.y)+1e-10;
    const intersects = (a,b,c,d) => (cross(a,b,c)*cross(a,b,d) < 0 && cross(c,d,a)*cross(c,d,b) < 0) || on(a,b,c) || on(a,b,d) || on(c,d,a) || on(c,d,b);
    for (let i = 0; i < points.length; i++) for (let j = i + 1; j < points.length; j++) {
      if (j === i+1 || (i === 0 && j === points.length-1)) continue;
      if (intersects(points[i],points[(i+1)%points.length],points[j],points[(j+1)%points.length])) return 'Polygon edges must not intersect. Move or remove a crossing point.';
    }
    const twiceArea = points.reduce((sum, point, i) => { const next = points[(i+1)%points.length]; return sum + point.x*next.y - next.x*point.y; }, 0);
    if (Math.abs(twiceArea) < 1e-10) return 'The polygon must have a nonzero area. Move points off the same line.';
    return null;
  }
  async function saveArea() {
    if (!state.source || locked() || state.busy) return;
    const error = polygonError(state.points);
    if (error) { message('polygonMessage', error, 'error'); return; }
    const name = $('areaName').value.trim();
    if (!name) { message('polygonMessage', 'Give this area a name.', 'error'); $('areaName').focus(); return; }
    const epoch = state.sourceEpoch;
    const area = selectedArea();
    const payload = { name, polygon: copyPoints(state.points), active: $('areaActive').checked };
    state.busy = true; renderRun(); message('polygonMessage', 'Saving area…');
    try {
      const result = await request(area ? `/api/areas/${area.id}` : '/api/areas', { method: area ? 'PUT' : 'POST', body: area ? payload : { ...payload, video_source_id: state.source.id } });
      if (epoch !== state.sourceEpoch) return;
      const index = state.areas.findIndex(item => item.id === result.id);
      if (index < 0) state.areas.push(result); else state.areas[index] = result;
      state.areaId = result.id; state.undo = []; renderAreaOptions(); renderEditor(); persist();
      message('polygonMessage', 'Area saved.', 'success'); notice('');
      if (state.run) refreshAnalytics();
    } catch (error) {
      if (epoch === state.sourceEpoch) message('polygonMessage', `Area could not be saved. ${safeError(error)}`, 'error');
      if (error.status === 409 && epoch === state.sourceEpoch) loadSources(state.source.id);
    } finally { if (epoch === state.sourceEpoch) { state.busy = false; renderRun(); } }
  }
  function applySettings(run) {
    const settings = run?.settings;
    if (!settings) return;
    for (const [id, key] of [['modelSelect','model'],['deviceSelect','device'],['confidenceInput','conf'],['iouInput','iou'],['imageSizeInput','imgsz']]) {
      if (settings[key] == null) continue;
      const control = $(id);
      const value = key === 'device' ? String(settings[key]).replace(/^cuda:/, '') : String(settings[key]);
      if (key === 'device' && /^\d+$/.test(value) && ![...control.options].some(option => option.value === value)) control.append(element('option', `GPU ${value}`, {value}));
      if (key === 'imgsz' && Number(value) >= 320 && Number(value) <= 1280 && Number(value)%32 === 0 && ![...control.options].some(option => option.value === value)) control.append(element('option', `${value} pixels`, {value}));
      control.value = value;
    }
  }
  async function runAction(action) {
    if (state.busy || !state.source) return;
    const epoch = state.sourceEpoch;
    const selection = state.selectionEpoch;
    const sourceId = state.source.id;
    const settings = { model: $('modelSelect').value, device: $('deviceSelect').value, conf: Number($('confidenceInput').value), iou: Number($('iouInput').value), imgsz: Number($('imageSizeInput').value) };
    if (action === 'start' && (!Number.isFinite(settings.conf) || settings.conf < .01 || settings.conf > .95 || !Number.isFinite(settings.iou) || settings.iou < .05 || settings.iou > .95)) { notice('Confidence must be between 0.01 and 0.95; overlap threshold between 0.05 and 0.95.', 'error'); return; }
    state.busy = true; renderRun(); notice(action === 'start' ? 'Starting processing…' : 'Stopping processing…');
    try {
      const run = await request(action === 'start' ? `/api/video-sources/${sourceId}/runs` : `/api/runs/${encodeURIComponent(state.run.id)}/stop`, { method: 'POST', ...(action === 'start' ? { body: settings } : {}) });
      if (epoch !== state.sourceEpoch || selection !== state.selectionEpoch) return;
      state.run = run; state.sourceActive = active(run) ? run : null;
      const index = state.runs.findIndex(item => item.id === run.id);
      if (index >= 0) state.runs[index] = run; else state.runs.unshift(run);
      resetAnalytics(); renderRunOptions(); persist(); notice(''); refreshAnalytics();
    } catch (error) {
      if (epoch === state.sourceEpoch) notice(`Processing could not ${action}. ${safeError(error)}`, 'error');
      if (error.status === 409 && epoch === state.sourceEpoch) {
        await loadSources(sourceId);
        if (state.source?.id === sourceId) notice(`Processing could not ${action}. ${safeError(error)}`, 'error');
      }
    } finally { if (epoch === state.sourceEpoch) { state.busy = false; renderRun(); } }
  }
  function markStale() {
    $('occupancyValue').textContent = '—';
    const last = state.lastStats?.last_observed_inside ?? state.lastStats?.currently_inside;
    $('occupancyNote').textContent = `Stale${last == null ? ' · no current observation' : ` · Last observed: ${last}`}`;
  }
  async function refreshAnalytics() {
    stopPolling();
    if (!state.run || state.areaId == null) return;
    const sourceEpoch = state.sourceEpoch, selectionEpoch = state.selectionEpoch;
    const controller = new AbortController(); state.pollController = controller;
    const current = () => sourceEpoch === state.sourceEpoch && selectionEpoch === state.selectionEpoch && !controller.signal.aborted;
    async function poll() {
      if (!current()) return;
      const runId = state.run.id;
      const params = new URLSearchParams({ video_source_id: state.source.id, area_id: state.areaId, run_id: runId });
      const scoped = new URLSearchParams({ run_id: runId, area_id: state.areaId, granularity: 'minute' });
      const areaInRun = !Array.isArray(state.run.areas) || state.run.areas.some(area => String(area.id) === String(state.areaId));
      const round = new AbortController();
      const cancelRound = () => round.abort();
      controller.signal.addEventListener('abort', cancelRound, {once: true});
      try {
        const [run, stats, history, forecast, events] = await Promise.all([
          request(`/api/runs/${encodeURIComponent(runId)}`, { signal: round.signal }),
          areaInRun ? request(`/api/stats/live?${params}`, { signal: round.signal }) : Promise.resolve(null),
          areaInRun ? request(`/api/stats?${scoped}`, { signal: round.signal }) : Promise.resolve(null),
          areaInRun ? request(`/api/forecast?${scoped}`, { signal: round.signal }) : Promise.resolve(null),
          areaInRun ? request(`/api/runs/${encodeURIComponent(runId)}/events?area_id=${state.areaId}`, { signal: round.signal }) : Promise.resolve(null)
        ]);
        if (!current()) return;
        state.run = run;
        if (state.sourceActive?.id === run.id) state.sourceActive = active(run) ? run : null;
        const index = state.runs.findIndex(item => item.id === run.id); if (index >= 0) state.runs[index] = run;
        state.failures = 0; state.lastStats = stats;
        renderRunOptions(); renderRun();
        if (stats) { renderStats(stats); renderHistory(history); renderForecast(forecast); renderEvents(events.items || [], history.timeline); }
        if (run.status === 'failed' || run.status === 'interrupted') notice(`This run ${run.status}. ${run.error_code ? String(run.error_code).replace(/_/g,' ') + '. ' : ''}Start a new run when the source is ready.`, 'error');
        else if (stats?.freshness === 'stale') notice('The latest observation is stale. Current occupancy is unknown.', 'stale');
        else notice('');
        $('updatedAt').textContent = `Updated ${new Date().toLocaleTimeString()}`;
      } catch (error) {
        if (!current() || error.name === 'AbortError') return;
        state.failures++; markStale();
        notice(`Updates are temporarily unavailable; retrying. ${safeError(error)}`, 'stale');
      } finally {
        round.abort();
        controller.signal.removeEventListener('abort', cancelRound);
        if (current() && (active(state.run) || state.failures)) state.timer = setTimeout(poll, Math.min(10000, 2000 * Math.max(1, state.failures)));
      }
    }
    poll();
  }
  function renderStats(stats) {
    const current = stats.freshness === 'current' && active(state.run) && stats.currently_inside != null;
    $('occupancyValue').textContent = current ? stats.currently_inside : '—';
    const last = stats.last_observed_inside ?? stats.currently_inside;
    $('occupancyNote').textContent = current ? 'Current observation' : `${stats.freshness === 'stale' ? 'Stale' : 'Last observation'}${last == null ? ' · unknown' : ` · Last observed: ${last}`}`;
    $('entriesValue').textContent = valueOrDash(stats.total_in);
    $('exitsValue').textContent = valueOrDash(stats.total_out);
    $('netValue').textContent = valueOrDash(stats.net_entries);
  }
  function emptyTable(id, columns, text) {
    const row = element('tr'); row.append(element('td', text, { colspan: columns })); $(id).replaceChildren(row);
  }
  function elapsed(seconds) {
    const total = Math.max(0, Math.floor(Number(seconds) || 0));
    return `${String(Math.floor(total/60)).padStart(2,'0')}:${String(total%60).padStart(2,'0')}`;
  }
  function timeLabel(value, timeline) {
    if (timeline === 'file') return elapsed(value);
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? 'Unknown' : date.toLocaleTimeString('en-GB', {hour:'2-digit',minute:'2-digit',timeZone:'UTC'});
  }
  function svgElement(tag, attributes = {}, text) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text != null) node.textContent = text;
    return node;
  }
  function chart(id, rows, labels, forecast = false) {
    const svg = $(id), width = Math.max(240, Math.round(svg.clientWidth)), height = 240;
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.replaceChildren(svgElement('title', {}, forecast ? 'Estimated entries and exits, next five minutes' : 'Observed entries and exits by minute'));
    if (!rows.length) { svg.append(svgElement('text', {x:width/2,y:120,'text-anchor':'middle'}, 'No complete observations yet')); return; }
    const left=35, right=width-15, top=18, bottom=205;
    const max = Math.max(1, ...rows.flatMap(row => [row.in_count, row.out_count]).filter(Number.isFinite));
    const ceiling = Math.max(1, Math.ceil(max));
    for (let i=0;i<=4;i++) {
      const y=bottom-(bottom-top)*i/4;
      svg.append(svgElement('line',{x1:left,x2:right,y1:y,y2:y,class:'gridline'}), svgElement('text',{x:left-7,y:y+3,'text-anchor':'end'},Number((ceiling*i/4).toFixed(1))));
    }
    const x = i => rows.length === 1 ? (left+right)/2 : left+(right-left)*i/(rows.length-1);
    for (const [field, suffix] of [['in_count','in'],['out_count','out']]) {
      let path='', continuing=false;
      rows.forEach((row,i) => {
        const value=row[field];
        if (value == null || !Number.isFinite(value)) { continuing=false; return; }
        const y=bottom-(bottom-top)*value/ceiling;
        path += `${continuing?'L':'M'}${x(i)},${y} `; continuing=true;
        svg.append(svgElement('circle',{cx:x(i),cy:y,r:3,class:`dot-${suffix}`}));
      });
      svg.append(svgElement('path',{d:path,class:`series-${suffix}${forecast?' forecast-line':''}`}));
    }
    labels.forEach((label,i) => { if (i===0 || i===labels.length-1 || i%Math.max(1,Math.ceil(labels.length/5))===0) svg.append(svgElement('text',{x:x(i),y:227,'text-anchor':'middle'},label)); });
  }
  function renderHistory(history) {
    state.lastHistory = history;
    const buckets=(history.buckets || []).slice(-60);
    $('historyMessage').textContent = `${history.timeline === 'file' ? 'Video time' : 'Camera UTC time'} · One-minute windows${(history.buckets?.length || 0)>60?' · Most recent 60 windows':''}. Gaps are unobserved, not zero.`;
    chart('historyChart',buckets,buckets.map(bucket=>timeLabel(bucket.window_start,history.timeline)));
    $('historyBody').replaceChildren();
    for (const bucket of buckets) {
      const row=element('tr');
      [timeLabel(bucket.window_start,history.timeline),valueOrDash(bucket.in_count),valueOrDash(bucket.out_count),!bucket.observed?'Unobserved':bucket.complete?'Complete':'Partial'].forEach(value=>row.append(element('td',value)));
      $('historyBody').append(row);
    }
    if (!buckets.length) emptyTable('historyBody',4,'No observed history yet.');
  }
  function renderForecast(forecast) {
    state.lastForecast = forecast;
    $('forecastBody').replaceChildren(); $('forecastMethod').textContent='';
    if (forecast.status !== 'ready') {
      const reasons = {
        requires_20_observed_contiguous_minutes: `Twenty complete, contiguous minute buckets are required${forecast.history_minutes == null ? '.' : `; ${forecast.history_minutes} available.`}`,
        multiple_runs: 'Choose one processing run to view its forecast.',
        no_run: 'Start processing to build the history needed for a forecast.',
        no_observations: 'No complete observations are available for this area yet.',
        live_run_not_current: 'Live observation is unavailable. Resume a healthy camera run to forecast.'
      };
      $('forecastMessage').textContent=reasons[forecast.reason] || (forecast.reason ? String(forecast.reason).replace(/_/g,' ') : 'Twenty complete, contiguous minute buckets are required.');
      emptyTable('forecastBody',3,'Not enough history yet.'); chart('forecastChart',[],[]); return;
    }
    const predictions=forecast.predictions || [];
    $('forecastMessage').textContent=forecast.label || `Next five minutes of ${forecast.timeline === 'file' ? 'video time' : 'camera time'}. Estimates may differ from observed movement.`;
    chart('forecastChart',predictions,predictions.map(item=>`+${item.step}m`),true);
    for (const item of predictions) {
      const row=element('tr'); [`+${item.step} min`,Number(item.in_count).toFixed(1),Number(item.out_count).toFixed(1)].forEach(value=>row.append(element('td',value))); $('forecastBody').append(row);
    }
    const methods=forecast.methods || {}, mae=forecast.backtest_mae;
    const score = direction => {
      const result = mae?.[direction];
      const value = typeof result === 'object' && result != null ? result[methods[direction] === 'ewma' ? 'ewma' : 'naive'] : result;
      return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(2) : '—';
    };
    const method = direction => methods[direction] === 'last_value' ? 'Last observed rate' : methods[direction] === 'ewma' ? 'Smoothed recent rate' : 'Baseline';
    $('forecastMethod').textContent = `Entries: ${method('in')} · Exits: ${method('out')}${mae && Object.keys(mae).length ? ` · Backtest MAE ${typeof mae === 'number' ? mae.toFixed(2) : `${score('in')} / ${score('out')}`}` : ''}`;
  }
  function renderEvents(items, timeline) {
    const recent=items.slice(0,50); $('eventsCount').textContent=`${recent.length} recent events`;
    $('eventsBody').replaceChildren();
    for (const event of recent) {
      const row=element('tr');
      row.append(element('td',timeline === 'file' ? elapsed(event.media_time_ms/1000) : `${new Date(event.timestamp).toLocaleString('en-GB',{timeZone:'UTC',hour12:false})} UTC`));
      const movement=element('td'); movement.append(element('span',event.type === 'enter'?'Entry':'Exit',{class:`event-pill ${event.type==='exit'?'exit':''}`}));
      row.append(movement,element('td',event.tracker_id ?? '—')); $('eventsBody').append(row);
    }
    if (!recent.length) emptyTable('eventsBody',3,'No confirmed crossings yet.');
  }

  $('areaSelect').addEventListener('change', () => {
    state.areaId = $('areaSelect').value ? Number($('areaSelect').value) : null;
    stopPolling(); resetAnalytics(); loadAreaEditor(); persist(); refreshAnalytics();
  });
  $('runSelect').addEventListener('change', () => {
    stopPolling(); state.run=state.runs.find(run=>run.id===$('runSelect').value)||null;
    resetAnalytics(); applySettings(state.run); renderRun(); persist(); refreshAnalytics();
  });
  $('startRun').addEventListener('click',()=>runAction('start'));
  $('stopRun').addEventListener('click',()=>runAction('stop'));
  $('refreshSources').addEventListener('click',()=>loadSources(state.source?.id));
  $('retryPreview').addEventListener('click',()=>loadPreview());
  $('newArea').addEventListener('click',()=>{stopPolling();resetAnalytics();state.areaId=null;state.points=[];state.undo=[];state.selectedPoint=-1;$('areaName').value='';$('areaActive').checked=true;renderAreaOptions();renderEditor();renderRun();persist();message('polygonMessage','Draw a new polygon and save it. Changes apply to the next run.');});
  $('addVertex').addEventListener('click',()=>{rememberPoints();state.points.push({x:.5,y:.5});state.selectedPoint=state.points.length-1;renderEditor();$('vertexRows').lastElementChild?.querySelector('input')?.focus();});
  $('clearPoints').addEventListener('click',()=>{rememberPoints();state.points=[];state.selectedPoint=-1;renderEditor();});
  $('undoPoints').addEventListener('click',()=>{if(state.undo.length){state.points=state.undo.pop();state.selectedPoint=-1;renderEditor();}});
  $('saveArea').addEventListener('click',saveArea);
  const canvas=$('polygonCanvas'); let drag=null;
  const canvasPoint=event=>{const rect=canvas.getBoundingClientRect();return{x:Math.min(1,Math.max(0,(event.clientX-rect.left)/rect.width)),y:Math.min(1,Math.max(0,(event.clientY-rect.top)/rect.height))};};
  canvas.addEventListener('pointerdown',event=>{
    if(locked()||state.busy||!state.preview)return;
    event.preventDefault();canvas.focus();const point=canvasPoint(event),rect=canvas.getBoundingClientRect();
    const index=state.points.findIndex(item=>Math.hypot((item.x-point.x)*rect.width,(item.y-point.y)*rect.height)<13);
    rememberPoints();
    if(index>=0){state.selectedPoint=index;drag=index;canvas.setPointerCapture(event.pointerId);}else{state.points.push(point);state.selectedPoint=state.points.length-1;}
    renderEditor();message('polygonMessage','Unsaved area changes.');
  });
  canvas.addEventListener('pointermove',event=>{if(drag!=null&&!locked()){state.points[drag]=canvasPoint(event);drawPolygon();}});
  const finishDrag=()=>{if(drag!=null){drag=null;renderEditor();}};
  canvas.addEventListener('pointerup',finishDrag);canvas.addEventListener('pointercancel',finishDrag);
  canvas.addEventListener('keydown',event=>{
    if(locked()||state.selectedPoint<0)return;
    const point=state.points[state.selectedPoint],step=event.shiftKey?.02:.005;
    if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Delete','Backspace'].includes(event.key))return;
    event.preventDefault();rememberPoints();
    if(['Delete','Backspace'].includes(event.key)){state.points.splice(state.selectedPoint,1);state.selectedPoint=-1;}
    else{if(event.key==='ArrowLeft')point.x-=step;if(event.key==='ArrowRight')point.x+=step;if(event.key==='ArrowUp')point.y-=step;if(event.key==='ArrowDown')point.y+=step;point.x=Math.min(1,Math.max(0,point.x));point.y=Math.min(1,Math.max(0,point.y));}
    renderEditor();
  });
  new ResizeObserver(drawPolygon).observe($('mediaStage'));
  new ResizeObserver(() => {
    if (state.lastHistory) {
      const history = state.lastHistory, buckets = (history.buckets || []).slice(-60);
      chart('historyChart', buckets, buckets.map(bucket => timeLabel(bucket.window_start, history.timeline)));
    }
    if (state.lastForecast) {
      const forecast = state.lastForecast, predictions = forecast.status === 'ready' ? forecast.predictions || [] : [];
      chart('forecastChart', predictions, predictions.map(item => `+${item.step}m`), true);
    }
  }).observe(document.querySelector('.analytics-grid'));
  $('streamImage').addEventListener('load',()=>{const image=$('streamImage');if(image.naturalWidth&&image.naturalHeight)$('mediaStage').style.aspectRatio=`${image.naturalWidth}/${image.naturalHeight}`;});
  $('streamImage').addEventListener('error',()=>{if(active(state.run))notice('The video view disconnected. Processing status and counts continue to update. Refresh this page to reconnect the view.','stale');});
  $('cameraForm').addEventListener('submit',async event=>{
    event.preventDefault();const name=$('cameraName').value.trim(),uri=$('cameraUri').value.trim();
    if(!/^(rtsp|https?):\/\//i.test(uri)){message('cameraMessage','Use an RTSP, HTTP, or HTTPS stream URL.','error');return;}
    const button=event.target.querySelector('button');button.disabled=true;message('cameraMessage','Connecting camera…');
    try{const source=await request('/api/video-sources',{method:'POST',body:{name,uri}});$('cameraUri').value='';$('cameraName').value='';message('cameraMessage','Camera added.','success');await loadSources(source.id);}
    catch(error){message('cameraMessage',`Camera could not be added. ${safeError(error)}`,'error');}
    finally{button.disabled=false;}
  });
  $('uploadForm').addEventListener('submit',event=>{
    event.preventDefault();const file=$('videoFile').files[0];if(!file||state.upload)return;
    const data=new FormData();data.append('file',file);if($('videoName').value.trim())data.append('name',$('videoName').value.trim());
    const xhr=new XMLHttpRequest();state.upload=xhr;$('uploadButton').disabled=true;$('uploadProgressWrap').hidden=false;$('uploadProgress').value=0;
    message('uploadMessage','Uploading video…');
    const finish=()=>{if(state.upload===xhr)state.upload=null;$('uploadButton').disabled=false;$('uploadProgressWrap').hidden=true;};
    xhr.open('POST','/api/upload-video');xhr.responseType='json';xhr.timeout=300000;
    xhr.upload.addEventListener('progress',progress=>{if(progress.lengthComputable){const percent=Math.round(progress.loaded/progress.total*100);$('uploadProgress').value=percent;message('uploadMessage',percent===100?'Upload complete. Checking video…':`Uploading… ${percent}%`);}else $('uploadProgress').removeAttribute('value');});
    xhr.addEventListener('load',async()=>{finish();if(xhr.status>=200&&xhr.status<300&&xhr.response?.id){message('uploadMessage','Video uploaded.','success');$('videoFile').value='';$('videoName').value='';await loadSources(xhr.response.id);}else message('uploadMessage',`Upload failed. ${safeError(new Error(typeof xhr.response?.detail==='string'?xhr.response.detail:'Check the file and try again.'))}`,'error');});
    xhr.addEventListener('error',()=>{finish();message('uploadMessage','The upload connection failed. Try again.','error');});
    xhr.addEventListener('timeout',()=>{finish();message('uploadMessage','The upload timed out. Try a smaller file or check the connection.','error');});
    xhr.addEventListener('abort',()=>{finish();message('uploadMessage','Upload cancelled.');});
    xhr.send(data);
  });
  $('cancelUpload').addEventListener('click',()=>state.upload?.abort());
  window.addEventListener('pagehide',()=>{stopPolling();state.sourceController?.abort();state.sourceListController?.abort();state.upload?.abort();if(state.previewUrl)URL.revokeObjectURL(state.previewUrl);});
  loadSources();
})();
