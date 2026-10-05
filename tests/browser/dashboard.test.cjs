/* Run with: node --test tests/browser/dashboard.test.cjs (requires Playwright). */
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs/promises');
const path = require('node:path');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '../..');
let server, browser, base;
const preview = '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540"><rect width="960" height="540" fill="#e2e8f0"/></svg>';
const area = { id: 3, video_source_id: 1, name: 'Main entrance', active: true, polygon: [{x: .15,y: .15},{x: .85,y: .15},{x: .85,y: .85},{x: .15,y: .85}] };
const running = { id: 'run-1', video_source_id: 1, status: 'running', settings: {}, areas: [area], started_at: '2026-10-05T00:00:00Z', heartbeat_at: new Date().toISOString(), last_frame_index: 30, media_time_ms: 1000, fps: 15, dropped_frames: 0 };
before(async () => {
  server = http.createServer(async (req, res) => {
    const relative = req.url.split('?')[0] === '/dashboard' ? 'app/templates/dashboard.html' : 'app' + req.url.split('?')[0];
    try {
      const data = await fs.readFile(path.join(root, relative));
      res.setHeader('Content-Type', relative.endsWith('.css') ? 'text/css' : relative.endsWith('.js') ? 'text/javascript' : 'text/html');
      res.end(data);
    } catch { res.writeHead(404); res.end('Not found'); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${server.address().port}`;
  browser = await chromium.launch({ headless: true });
});
after(async () => { await browser?.close(); await new Promise(resolve => server?.close(resolve)); });

async function setup(options = {}) {
  const context = await browser.newContext({ viewport: options.viewport || { width: 1440, height: 1000 }, timezoneId: 'Asia/Bangkok' });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const requests = [];
  let sourceReads = 0;
  let run = Object.hasOwn(options, 'run') ? options.run : running;
  const sources = [{id: 1, name: 'Entrance clip', kind: 'file', enabled: true, width: 960, height: 540, fps: 30}, {id: 2, name: 'Side camera', kind: 'live', enabled: true}];
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url());
    const p = url.pathname;
    const method = route.request().method();
    requests.push({ path: p, query: url.search, method, body: route.request().postData() });
    const json = data => route.fulfill({ json: data });
    if (p.startsWith('/api/preview/')) return route.fulfill({ contentType: 'image/svg+xml', body: options.preview || preview });
    if (p.endsWith('/stream')) return route.fulfill({ contentType: 'image/svg+xml', body: preview });
    if (p === '/api/video-sources' && method === 'POST') {
      const source = {id: 4, name: JSON.parse(route.request().postData()).name, kind: 'live', enabled: true};
      sources.push(source); return json(source);
    }
    if (p === '/api/video-sources') { if (options.delayRefresh && sourceReads++ > 0) await new Promise(resolve => setTimeout(resolve, 600)); return json(sources); }
    if (p === '/api/areas' && method === 'GET') {
      if (options.delayAreas && url.searchParams.get('video_source_id') === '1') await new Promise(resolve => setTimeout(resolve, 600));
      return json(url.searchParams.get('video_source_id') === '1' ? options.areas || [area] : []);
    }
    if (p === '/api/areas' && method === 'POST') return json({ ...JSON.parse(route.request().postData()), id: 5, active: true });
    if (p.startsWith('/api/areas/') && method === 'PUT') return json({ ...area, ...JSON.parse(route.request().postData()) });
    if (p.match(/\/video-sources\/\d+\/runs$/) && method === 'POST') { if (options.delayRunStart) await new Promise(resolve => setTimeout(resolve, 600)); run = { ...running, status: 'starting' }; return json(run); }
    if (p.match(/\/video-sources\/\d+\/runs$/)) return json(p.includes('/1/') && run ? [run] : []);
    if (p.endsWith('/stop')) { run = { ...run, status: 'stopped' }; return json(run); }
    if (p.endsWith('/events')) return json({ items: [{ id: 1, timestamp: '2026-10-05T00:00:01Z', media_time_ms: 1000, tracker_id: '0:7', type: 'enter', area_id: 3 }] });
    if (p.startsWith('/api/runs/')) return json(run);
    if (p === '/api/stats/live') return json(options.stats || { currently_inside: 2, last_observed_inside: 2, total_in: 7, total_out: 5, net_entries: 2, freshness: 'current', run_id: 'run-1', status: run?.status, ts: new Date().toISOString() });
    if (p === '/api/stats') return json(options.history || { timeline: 'file', buckets: [{window_start: 0, window_end: 60, in_count: 7, out_count: 5, observed: true, complete: true, currently_inside: 2}, {window_start: 60, window_end: 120, in_count: null, out_count: null, observed: false, complete: false, currently_inside: null}], summary: {} });
    if (p === '/api/forecast') return json(options.forecast || { status: 'insufficient_data', reason: 'Twenty complete minute buckets are required.', predictions: [], horizon_minutes: 5, timeline: 'file' });
    if (p === '/api/upload-video') { await new Promise(resolve => setTimeout(resolve, 2000)); return json({ id: 6, name: 'Uploaded', kind: 'file', enabled: true }); }
    return route.fulfill({ status: 404, json: { detail: 'Unknown test API route' } });
  });
  if (options.saved) await context.addInitScript(value => localStorage.setItem('synapsis.selection', JSON.stringify(value)), options.saved);
  await page.goto(`${base}/dashboard`);
  return { page, context, requests, errors };
}

test('restores a selected source and run without starting another producer, and locks area editing', async () => {
  const { page, context, requests, errors } = await setup({ saved: { sourceId: 1, areaId: 3 } });
  try {
    await page.getByRole('heading', { name: 'People analytics' }).waitFor();
    await page.locator('#runStatus').getByText('Running', { exact: true }).waitFor();
    assert.equal(await page.locator('#occupancyValue').textContent(), '2');
    assert.equal(await page.locator('#saveArea').isDisabled(), true);
    assert.equal(await page.locator('#stopRun').isDisabled(), false);
    assert.equal(requests.filter(r => r.method === 'POST' && r.path.endsWith('/runs')).length, 0);
    assert.deepEqual(errors, [], 'No unhandled browser errors');
    await page.reload();
    await page.locator('#runStatus').getByText('Running', { exact: true }).waitFor();
    assert.equal(requests.filter(r => r.method === 'POST' && r.path.endsWith('/runs')).length, 0);
  } finally { await context.close(); }
});

for (const width of [390, 768, 1440]) test(`fits a ${width}px viewport without horizontal overflow`, async () => {
  const {page, context} = await setup({ viewport: { width, height: 900 } });
  try {
    await page.getByRole('heading', { name: 'People analytics' }).waitFor();
    await page.locator('#runStatus').getByText('Running', { exact: true }).waitFor();
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    const panel = await page.locator('#mediaStage').boundingBox();
    assert.ok(panel.width > 100 && panel.x >= 0 && panel.x + panel.width <= width + 1);
    const image = await page.locator('#streamImage').boundingBox();
    assert.ok(Math.abs(image.width / image.height - 960 / 540) < 0.02);
    if (process.env.DASHBOARD_SCREENSHOTS === '1') {
      await fs.mkdir(path.join(root, 'output/playwright'), {recursive:true});
      await page.screenshot({path:path.join(root, `output/playwright/dashboard-${width}.png`),fullPage:true});
    }
  } finally { await context.close(); }
});

test('shows stale occupancy as unknown while preserving the last observation and signed net count', async () => {
  const {page, context} = await setup({ stats: { currently_inside: null, last_observed_inside: 4, total_in: 2, total_out: 5, net_entries: -3, freshness: 'stale', status: 'running' } });
  try {
    await page.locator('#occupancyNote').getByText(/Last observed: 4/).waitFor();
    assert.equal(await page.locator('#occupancyValue').textContent(), '—');
    assert.equal(await page.locator('#netValue').textContent(), '-3');
    assert.match(await page.locator('#forecastMessage').textContent(), /Twenty complete/);
    assert.match(await page.locator('#historyBody').textContent(), /Unobserved/);
  } finally { await context.close(); }
});

test('explicit stop changes run state; a new run only starts after pressing Start', async () => {
  const {page, context, requests} = await setup();
  try {
    await page.locator('#runStatus').getByText('Running', { exact: true }).waitFor();
    await page.locator('#stopRun').click();
    await page.locator('#runStatus').getByText('Stopped', { exact: true }).waitFor();
    assert.equal(requests.filter(r => r.path.endsWith('/stop')).length, 1);
    await page.locator('#startRun').click();
    await page.locator('#runStatus').getByText('Starting', { exact: true }).waitFor();
    assert.equal(requests.filter(r => r.method === 'POST' && r.path.endsWith('/runs')).length, 1);
  } finally { await context.close(); }
});

test('rejects crossing polygon edges and saves a valid normalized polygon from keyboard inputs', async () => {
  const {page, context, requests} = await setup({ run: null });
  try {
    await page.locator('#areaSelect').selectOption('3');
    await page.locator('#vertexRows input').first().waitFor();
    await page.locator('#clearPoints').click();
    const points = [[.1,.1],[.8,.8],[.1,.8],[.8,.1]];
    for (const [x,y] of points) {
      await page.locator('#addVertex').click();
      const row = page.locator('#vertexRows tr').last();
      await row.locator('input').nth(0).fill(String(x)); await row.locator('input').nth(0).dispatchEvent('change');
      await row.locator('input').nth(1).fill(String(y)); await row.locator('input').nth(1).dispatchEvent('change');
    }
    await page.locator('#saveArea').click();
    assert.match(await page.locator('#polygonMessage').textContent(), /intersect|cross/i);
    assert.equal(requests.filter(r => r.method === 'PUT').length, 0);
    await page.locator('#vertexRows tr').last().getByRole('button', {name: /Remove/}).click();
    await page.locator('#saveArea').click();
    await page.locator('#polygonMessage').getByText(/saved/i).waitFor();
    const saved = JSON.parse(requests.find(r => r.method === 'PUT').body);
    assert.deepEqual(saved.polygon, [{x:.1,y:.1},{x:.8,y:.8},{x:.1,y:.8}]);
  } finally { await context.close(); }
});

test('ignores a delayed response from a previously selected source', async () => {
  const {page, context} = await setup({ delayAreas: true, run: null });
  try {
    await page.getByRole('button', { name: /Side camera/ }).click();
    await page.waitForTimeout(750);
    assert.match(await page.locator('#sourceHeading').textContent(), /Side camera/);
    assert.equal(await page.locator('#areaSelect option[value="3"]').count(), 0);
    assert.equal(await page.locator('#occupancyValue').textContent(), '—');
  } finally { await context.close(); }
});

test('registers a camera and clears the credential-bearing input after success', async () => {
  const {page, context} = await setup({ run: null });
  try {
    await page.locator('#cameraDetails').evaluate(el => el.open = true);
    await page.locator('#cameraName').fill('Warehouse');
    await page.locator('#cameraUri').fill('rtsp://user:secret@camera.local/stream');
    await page.locator('#cameraForm').getByRole('button', { name: 'Add camera' }).click();
    await page.locator('#sourceHeading').getByText('Warehouse', { exact: true }).waitFor();
    assert.equal(await page.locator('#cameraUri').inputValue(), '');
    assert.equal(await page.getByText('rtsp://user:secret@camera.local/stream', {exact:true}).count(), 0);
  } finally { await context.close(); }
});

test('renders a ready forecast as accessible SVG plus a five-step table', async () => {
  const {page, context} = await setup({forecast: {status:'ready', horizon_minutes:5, predictions:[1,2,3,4,5].map(step=>({step,in_count:3,out_count:2})),methods:{in:'EWMA',out:'last_value'},backtest_mae:{in:0.8,out:0.5},timeline:'file',label:'Next five minutes of video time'}});
  try {
    await page.locator('#forecastBody tr').nth(4).waitFor();
    assert.equal(await page.locator('#forecastChart').getAttribute('role'), 'img');
    assert.match(await page.locator('#forecastMessage').textContent(), /video time/i);
    assert.equal(await page.locator('#forecastBody tr').count(), 5);
  } finally { await context.close(); }
});

test('keeps a portrait preview and its polygon canvas aligned without letterboxing', async () => {
  const {page, context} = await setup({ run: null, preview: '<svg xmlns="http://www.w3.org/2000/svg" width="540" height="960"><rect width="540" height="960" fill="#e2e8f0"/></svg>' });
  try {
    await page.locator('#polygonCanvas').waitFor({state:'visible'});
    const box = await page.locator('#mediaStage').boundingBox();
    assert.ok(Math.abs(box.width/box.height - 540/960) < 0.01, 'Preview stage must retain the actual source aspect ratio');
  } finally { await context.close(); }
});

test('a late source refresh cannot undo a more recent source selection', async () => {
  const {page, context} = await setup({run:null,delayRefresh:true});
  try {
    await page.locator('#areaSelect option[value="3"]').waitFor({state:'attached'});
    await page.locator('#refreshSources').click();
    await page.getByRole('button', {name:/Side camera/}).click();
    await page.waitForTimeout(750);
    assert.equal(await page.locator('#sourceHeading').textContent(), 'Side camera');
  } finally { await context.close(); }
});

test('keeps a quiet running source active through more than four event-free polls', async () => {
  const {page, context, requests} = await setup({ stats: {currently_inside:0,last_observed_inside:0,total_in:0,total_out:0,net_entries:0,freshness:'current',status:'running'} });
  try {
    await page.waitForFunction(() => document.querySelector('#runStatus').textContent === 'Running');
    while (requests.filter(r => r.path === '/api/stats/live').length < 5) await page.waitForTimeout(250);
    assert.equal(await page.locator('#stopRun').isDisabled(), false);
    assert.equal(await page.locator('#streamImage').isVisible(), true);
    assert.equal(requests.filter(r => r.path.endsWith('/stop')).length, 0);
  } finally { await context.close(); }
});

test('cancels an in-flight upload without switching to a newly created source', async () => {
  const {page, context} = await setup({run:null});
  try {
    await page.locator('#cameraDetails').waitFor();
    await page.locator('#uploadDetails').evaluate(el => el.open = true);
    await page.locator('#videoFile').setInputFiles({name:'sample.mp4',mimeType:'video/mp4',buffer:Buffer.alloc(2048)});
    await page.locator('#uploadButton').click();
    await page.locator('#cancelUpload').click();
    await page.locator('#uploadMessage').getByText('Upload cancelled.', {exact:true}).waitFor();
    assert.equal(await page.locator('#uploadButton').isDisabled(), false);
    assert.equal(await page.locator('#sourceHeading').textContent(), 'Entrance clip');
  } finally { await context.close(); }
});

test('moves a polygon vertex by pointer and keyboard and can undo the keyboard change', async () => {
  const {page, context, errors} = await setup({run:null});
  try {
    await page.locator('#polygonCanvas').waitFor({state:'visible'});
    await page.locator('#vertexRows tr').nth(3).waitFor();
    const box = await page.locator('#polygonCanvas').boundingBox();
    await page.mouse.move(box.x + box.width*.15, box.y + box.height*.15);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width*.25, box.y + box.height*.25);
    await page.mouse.up();
    assert.ok(Math.abs(Number(await page.locator('#vertexRows input').first().inputValue())-.25)<.002);
    await page.locator('#polygonCanvas').focus();
    await page.keyboard.press('ArrowLeft');
    assert.ok(Math.abs(Number(await page.locator('#vertexRows input').first().inputValue())-.245)<.002);
    await page.locator('#undoPoints').click();
    assert.ok(Math.abs(Number(await page.locator('#vertexRows input').first().inputValue())-.25)<.002);
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('keeps chart labels readable at mobile width instead of shrinking a desktop SVG', async () => {
  const {page, context} = await setup({viewport:{width:390,height:900}});
  try {
    await page.locator('#historyChart text').nth(2).waitFor();
    const fontPixels = await page.locator('#historyChart text').nth(2).evaluate(el => parseFloat(getComputedStyle(el).fontSize)*el.getScreenCTM().a);
    assert.ok(fontPixels >= 9, `Axis text renders at ${fontPixels}px`);
  } finally { await context.close(); }
});

test('warns when current polygon edits differ from the historical run area', async () => {
  const snapshot = {...area,polygon:[{x:.2,y:.2},{x:.8,y:.2},{x:.8,y:.8},{x:.2,y:.8}]};
  const {page, context} = await setup({run:{...running,status:'completed',areas:[snapshot]}});
  try {
    await page.locator('#runStatus').getByText('Completed',{exact:true}).waitFor();
    assert.match(await page.locator('#areaSnapshotNote').textContent(), /saved.*area|area.*start/i);
    assert.equal(await page.locator('#areaSnapshotNote').isVisible(), true);
    assert.match(await page.locator('#editorHint').textContent(), /next run/i);
  } finally { await context.close(); }
});

test('uses UTC for live history even when the viewer has another timezone', async () => {
  const {page,context} = await setup({history:{timeline:'live',buckets:[{window_start:'2026-10-05T00:00:00Z',window_end:'2026-10-05T00:01:00Z',in_count:1,out_count:0,observed:true,complete:true}],summary:{}}});
  try {
    await page.locator('#historyMessage').getByText(/Camera UTC time/).waitFor();
    assert.equal(await page.locator('#historyBody tr td').first().textContent(), '00:00');
  } finally {await context.close();}
});

test('an area absent from a run snapshot has unknown counts but still follows run lifecycle', async () => {
  const added = {...area,id:9,name:'New area',active:false};
  const {page,context,requests} = await setup({areas:[area,added],saved:{sourceId:1,areaId:9}});
  try {
    await page.locator('#runStatus').getByText('Running',{exact:true}).waitFor();
    await page.waitForTimeout(150);
    assert.equal(await page.locator('#occupancyValue').textContent(), '—');
    assert.equal(requests.filter(r => r.path === '/api/stats/live').length, 0);
    assert.match(await page.locator('#areaSnapshotNote').textContent(), /not included/i);
    assert.ok(requests.some(r => r.path === '/api/runs/run-1'));
  } finally {await context.close();}
});

test('locks the selected area while starting a run so a pending action cannot lose its result', async () => {
  const {page,context} = await setup({run:null,delayRunStart:true});
  try {
    await page.locator('#areaSelect option[value="3"]').waitFor({state:'attached'});
    await page.locator('#startRun').click();
    assert.equal(await page.locator('#areaSelect').isDisabled(), true);
    await page.locator('#runStatus').getByText('Starting',{exact:true}).waitFor();
    assert.equal(await page.locator('#areaSelect').isDisabled(), false);
  } finally {await context.close();}
});

test('restores a CUDA device setting without losing the valid device selection', async () => {
  const {page,context,requests} = await setup({run:{...running,status:'completed',settings:{model:'yolov8n.pt',device:'cuda:0',conf:.25,iou:.45,imgsz:480}}});
  try {
    await page.locator('#runStatus').getByText('Completed',{exact:true}).waitFor();
    assert.equal(await page.locator('#deviceSelect').inputValue(), '0');
    assert.equal(await page.locator('#imageSizeInput').inputValue(), '480');
    await page.locator('#startRun').click();
    await page.locator('#runStatus').getByText('Starting',{exact:true}).waitFor();
    const body=JSON.parse(requests.find(r=>r.method==='POST'&&r.path.endsWith('/runs')).body);
    assert.equal(body.device,'0'); assert.equal(body.imgsz,480);
  } finally {await context.close();}
});

test('respects server freshness within its configured heartbeat grace period', async () => {
  const {page,context} = await setup({run:{...running,heartbeat_at:new Date(Date.now()-11000).toISOString()}});
  try {
    await page.locator('#updatedAt').getByText(/Updated/).waitFor();
    assert.equal(await page.locator('#occupancyValue').textContent(),'2');
  } finally {await context.close();}
});
