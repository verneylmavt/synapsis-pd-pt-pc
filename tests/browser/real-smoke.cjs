/* Opt-in real API smoke. Creates a source/area/run, preserves all existing data.
 * Set SMOKE_URL and SMOKE_DEVICE=cpu for Docker; otherwise uses host GPU 0.
 */
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const base = process.env.SMOKE_URL || 'http://127.0.0.1:8010';
const video = process.env.SMOKE_VIDEO || path.join(root, 'data/benchmarks/mot15/PETS09-S2L1/video.mp4');
const output = path.join(root, 'output/playwright');
const active = new Set(['starting','running','stopping','reconnecting']);
const pause = milliseconds => new Promise(resolve=>setTimeout(resolve,milliseconds));

(async()=>{
  const browser = await chromium.launch({headless:true});
  const context = await browser.newContext({viewport:{width:1440,height:1000}});
  const errors=[], postRuns=[], consoles=[], failedResponses=[];
  const instrument=page=>{
    page.on('pageerror',error=>errors.push(error.message));
    page.on('console',entry=>{if(entry.type()==='error')consoles.push(entry.text());});
    page.on('response',response=>{if(response.status()>=400)failedResponses.push({status:response.status(),url:response.url()});});
    page.on('request',request=>{if(request.method()==='POST'&&/\/video-sources\/\d+\/runs$/.test(new URL(request.url()).pathname))postRuns.push(request.url());});
  };
  let source, area, run;
  const report={base,started_at:new Date().toISOString(),checks:[]};
  const check=text=>{report.checks.push(text);process.stdout.write(text+'\n');};
  const api=async url=>{const response=await context.request.get(base+url);assert.ok(response.ok(),`GET ${url}: ${response.status()}`);return response.json();};
  try{
    await fs.mkdir(output,{recursive:true});
    const readyDeadline=Date.now()+60000;
    let ready=false;
    while(Date.now()<readyDeadline){
      try{ready=(await context.request.get(base+'/readyz',{timeout:3000})).ok();}catch{/* Server startup may reset a connection. */}
      if(ready)break;await pause(500);
    }
    assert.ok(ready,'Application did not become ready within 60 seconds');
    // Seed a decoded upload in this server's filesystem. Old host/container
    // sources may have metadata but a path unavailable in the current runtime.
    const initialResponse=await context.request.post(base+'/api/upload',{multipart:{
      name:'Smoke preview seed',file:{name:path.basename(video),mimeType:'video/mp4',buffer:await fs.readFile(video)}
    }});
    assert.ok(initialResponse.ok());const initial=await initialResponse.json();
    await context.addInitScript(id=>{if(!localStorage.getItem('synapsis.selection'))localStorage.setItem('synapsis.selection',JSON.stringify({sourceId:id}));},initial.id);
    const page=await context.newPage();instrument(page);
    await page.goto(base+'/dashboard',{waitUntil:'domcontentloaded'});
    await page.getByRole('heading',{name:'People analytics'}).waitFor();
    await page.locator('#uploadDetails').evaluate(el=>el.open=true);
    const name='Browser smoke PETS '+new Date().toISOString();
    await page.locator('#videoName').fill(name);await page.locator('#videoFile').setInputFiles(video);
    const uploaded=page.waitForResponse(response=>new URL(response.url()).pathname==='/api/upload-video'&&response.request().method()==='POST',{timeout:60000});
    await page.locator('#uploadButton').click();
    const uploadResponse=await uploaded;assert.ok(uploadResponse.ok(),`Upload status ${uploadResponse.status()}`);source=await uploadResponse.json();
    report.source_id=source.id;
    report.source_name=source.name;
    await page.locator('#sourceHeading').getByText(source.name,{exact:true}).waitFor();
    await page.locator('#polygonCanvas').waitFor({state:'visible'});
    const stage=await page.locator('#mediaStage').boundingBox();assert.ok(Math.abs(stage.width/stage.height-768/576)<.01);
    check('Real upload selected; 768×576 preview and canvas retain 4:3 aspect ratio');
    await page.locator('#areaName').fill('Smoke central area');
    for(const [x,y] of [[.15,.15],[.85,.15],[.85,.85],[.15,.85]]){
      await page.locator('#addVertex').click();const row=page.locator('#vertexRows tr').last();
      await row.locator('input').nth(0).fill(String(x));await row.locator('input').nth(0).dispatchEvent('change');
      await row.locator('input').nth(1).fill(String(y));await row.locator('input').nth(1).dispatchEvent('change');
    }
    const areaSaved=page.waitForResponse(response=>new URL(response.url()).pathname==='/api/areas'&&response.request().method()==='POST');
    await page.locator('#saveArea').click();const savedResponse=await areaSaved;assert.ok(savedResponse.ok());area=await savedResponse.json();report.area_id=area.id;
    await page.locator('#polygonMessage').getByText('Area saved.',{exact:true}).waitFor();
    check('Normalized polygon saved through the real API');
    await page.locator('#settingsDetails').evaluate(el=>el.open=true);await page.locator('#deviceSelect').selectOption(process.env.SMOKE_DEVICE || '0');
    const started=page.waitForResponse(response=>new URL(response.url()).pathname===`/api/video-sources/${source.id}/runs`&&response.request().method()==='POST',{timeout:15000});
    await page.locator('#startRun').click();const startedResponse=await started;assert.ok(startedResponse.ok(),`Start status ${startedResponse.status()}`);run=await startedResponse.json();report.run_id=run.id;
    const deadline=Date.now()+60000;
    while(Date.now()<deadline){run=await api(`/api/runs/${run.id}`);if(run.status==='running'&&run.last_frame_index>=3)break;if(!active.has(run.status))throw new Error(`Run ended during startup: ${run.status} / ${run.error_code}`);await pause(500);}
    assert.equal(run.status,'running');assert.ok(run.last_frame_index>=3);
    await page.locator('#runStatus').getByText('Running',{exact:true}).waitFor({timeout:15000});
    await page.waitForFunction(()=>document.querySelector('#streamImage').naturalWidth>0);
    assert.equal(await page.locator('#saveArea').isDisabled(),true);
    check('Explicit Start runs pretrained inference; committed frames appear; area edits locked');
    await page.reload({waitUntil:'domcontentloaded'});await page.locator('#runStatus').getByText('Running',{exact:true}).waitFor({timeout:15000});
    assert.equal(await page.locator('#runSelect').inputValue(),run.id);assert.equal(postRuns.length,1);
    const viewer=await context.newPage();instrument(viewer);await viewer.goto(base+'/dashboard',{waitUntil:'domcontentloaded'});await viewer.locator('#runStatus').getByText('Running',{exact:true}).waitFor({timeout:15000});
    await viewer.waitForFunction(()=>document.querySelector('#streamImage').naturalWidth>0);
    assert.equal(await viewer.locator('#runSelect').inputValue(),run.id);assert.equal(postRuns.length,1);
    check('Refresh plus a second viewer recover the same run without another producer POST');
    const stats=await api(`/api/stats/live?video_source_id=${source.id}&area_id=${area.id}&run_id=${run.id}`);
    assert.equal(stats.run_id,run.id);assert.equal(stats.freshness,'current');assert.equal(typeof stats.currently_inside,'number');
    const history=await api(`/api/stats?run_id=${run.id}&area_id=${area.id}&granularity=minute`);assert.equal(history.timeline,'file');assert.ok(history.buckets.length>0);
    const forecast=await api(`/api/forecast?run_id=${run.id}&area_id=${area.id}`);assert.equal(forecast.status,'insufficient_data');
    report.observation={stats,history_buckets:history.buckets.length,forecast};check('Live occupancy, run history, and insufficient forecast use the real database');
    await page.screenshot({path:path.join(output,'real-dashboard-1440.png'),fullPage:true});
    for(const width of [768,390]){await viewer.setViewportSize({width,height:900});assert.equal(await viewer.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);await viewer.screenshot({path:path.join(output,`real-dashboard-${width}.png`),fullPage:true});}
    check('Real dashboard screenshots captured at 1440/768/390; no horizontal overflow');
    await page.locator('#stopRun').click();
    const stopDeadline=Date.now()+15000;
    while(Date.now()<stopDeadline){run=await api(`/api/runs/${run.id}`);if(!active.has(run.status))break;await pause(250);}
    assert.equal(run.status,'stopped');await page.locator('#runStatus').getByText('Stopped',{exact:true}).waitFor({timeout:15000});
    await page.waitForFunction(()=>document.querySelector('#occupancyValue').textContent==='—');
    check('Explicit Stop reaches stopped; current occupancy becomes unknown while history remains');
    report.final_run=run;
    assert.deepEqual(errors,[]);assert.deepEqual(consoles,[]);check('No unhandled JavaScript or browser console errors');
    report.result='passed';
  }catch(error){report.result='failed';report.error=error.message;throw error;}
  finally{
    if(run&&active.has(run.status)){try{await context.request.post(`${base}/api/runs/${run.id}/stop`);}catch{/* Preserve original error. */}}
    report.browser_errors=errors;report.console_errors=consoles;report.failed_responses=failedResponses;report.finished_at=new Date().toISOString();
    await fs.writeFile(path.join(output,'real-smoke-report.json'),JSON.stringify(report,null,2));await context.close();await browser.close();
  }
})().catch(error=>{process.stderr.write(error.stack+'\n');process.exitCode=1;});
