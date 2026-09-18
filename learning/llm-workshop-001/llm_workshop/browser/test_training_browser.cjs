/* Real-browser smoke tests using Chrome's local debugging pipe, no dependencies. */
const {spawn} = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const assert = require('node:assert/strict');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(),'llm-browser-test-'));
const chrome = spawn(process.env.CHROME_BIN || 'google-chrome', [
  '--headless','--no-sandbox','--disable-dev-shm-usage','--disable-gpu',
  '--no-first-run','--no-default-browser-check','--disable-background-networking',
  '--remote-debugging-pipe',`--user-data-dir=${path.join(temporary,'profile')}`
],{stdio:['ignore','ignore','pipe','pipe','pipe']});
let count=0,buffer='',sessionId;
const pending=new Map(),exceptions=[],requests=[];
chrome.on('error',error=>{for(const item of pending.values())item.reject(error);});
chrome.stderr.on('data',chunk=>{if(process.env.DEBUG_BROWSER)process.stderr.write(chunk);});
for(const pipe of [chrome.stdio[3],chrome.stdio[4]])pipe.on('error',error=>{
  for(const item of pending.values()){clearTimeout(item.timer);item.reject(error);}
  pending.clear();
});
chrome.stdio[4].on('data',chunk=>{
  buffer+=chunk.toString();
  let index;
  while((index=buffer.indexOf('\0'))>=0){
    const message=JSON.parse(buffer.slice(0,index));buffer=buffer.slice(index+1);
    if(message.id && pending.has(message.id)) {
      const {resolve,reject,timer}=pending.get(message.id);clearTimeout(timer);pending.delete(message.id);
      message.error?reject(new Error(JSON.stringify(message.error))):resolve(message.result);
    } else if(message.method==='Runtime.exceptionThrown') exceptions.push(message.params.exceptionDetails);
    else if(message.method==='Network.requestWillBeSent') requests.push(message.params.request.url);
  }
});
function send(method,params={},session=sessionId){
  const id=++count;
  return new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>{pending.delete(id);reject(new Error(`Timeout: ${method}`));},20000);
    pending.set(id,{resolve,reject,timer});
    chrome.stdio[3].write(JSON.stringify({id,method,params,...session?{sessionId:session}:{}})+'\0');
  });
}
async function evaluate(expression){
  const result=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
  if(result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
  return result.result.value;
}
async function waitFor(expression){
  const end=Date.now()+10000;
  while(Date.now()<end){if(await evaluate(expression))return;await new Promise(resolve=>setTimeout(resolve,50));}
  throw new Error(`Page condition timed out: ${expression}`);
}
async function click(selector){
  await evaluate(`(()=>{const el=document.querySelector(${JSON.stringify(selector)});if(!el)throw Error('Missing '+${JSON.stringify(selector)});el.click();})()`);
}
async function change(id,value){
  await evaluate(`(()=>{const el=document.getElementById(${JSON.stringify(id)});el.value=${JSON.stringify(value)};el.dispatchEvent(new Event('change',{bubbles:true}));})()`);
}
async function prompt(text){
  await evaluate(`(()=>{document.getElementById('prompt').value=${JSON.stringify(text)};document.getElementById('prompt-form').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}));})()`);
}
async function screenshot(name,width=1440,height=1050){
  await send('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
  await evaluate('window.scrollTo(0,0)');
  const result=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  fs.writeFileSync(path.join(temporary,name+'.png'),Buffer.from(result.data,'base64'));
}

(async()=>{try{
const target=await send('Target.createTarget',{url:'about:blank'},null);sessionId=(await send('Target.attachToTarget',{targetId:target.targetId,flatten:true},null)).sessionId;await send('Page.enable');await send('Runtime.enable');await send('Page.navigate',{url:pathToFileURL(path.join(__dirname,'index.html')).href});await waitFor('document.querySelector("#scene .scene-head") !== null');
assert.equal(await evaluate('document.querySelectorAll("#chapters small").length'),0);
await click('#chapters [data-chapter="5"]');
await click('[data-model="untrained"]');const before=await evaluate('document.querySelector(".checkpoint-preview pre").textContent');await click('[data-model="trained"]');assert.notEqual(await evaluate('document.querySelector(".checkpoint-preview pre").textContent'),before);assert.equal(await evaluate('document.querySelector("[data-model=trained]").getAttribute("aria-pressed")'),'true');
for(const run of await evaluate('WORKSHOP_EXPERIMENTS.runs.map(r=>({id:r.id,device:r.device,after:r.after,before:r.before,steps:r.steps}))')){
await change('training-run',run.id);assert.equal(await evaluate('document.getElementById("model").value'),run.after);assert.match(await evaluate('document.querySelector(".scene-head .badge").textContent'),new RegExp(run.device==='cuda'?'GPU':'CPU'));
await click('[data-model="'+run.before+'"]');assert.equal(await evaluate('document.getElementById("training-run").value'),run.id);assert.equal(await evaluate('document.getElementById("model").value'),run.before);
await click('[data-model="'+run.after+'"]');assert.match(await evaluate('document.querySelector(".checkpoint-preview h3").textContent'),new RegExp(run.steps+' updates'));
}
for(const width of [1440,1024,390]){await send('Emulation.setDeviceMetricsOverride',{width,height:1100,deviceScaleFactor:1,mobile:false});for(const run of ['legacy','controlled_cuda_1000','adapter_biology','adapter_finance']){await change('training-run',run);assert(await evaluate('document.documentElement.scrollWidth<=innerWidth'),'Overflow '+width+' '+run);}await evaluate('document.querySelector("#scene").scrollIntoView()');const r=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(temporary,'training-'+width+'.png'),Buffer.from(r.data,'base64'));}
await change('training-run','controlled_cuda_1000');await click('#chapters [data-chapter="4"]');await click('#generate');await change('model','adapter_biology');assert.equal(await evaluate('document.querySelector(".generation-text mark").textContent'),'');assert.match(await evaluate('document.getElementById("scene").textContent'),/adapter updates/);assert(await evaluate('document.documentElement.scrollWidth<=innerWidth'));
assert.equal(exceptions.length,0);console.log('PASS: all recorded CPU/GPU and LoRA runs selectable; visibly different before/after; keep-run behavior, generation reset, no menu times and responsive layouts: '+temporary);
}finally{await send('Browser.close',{},null).catch(()=>{});chrome.kill();for(const p of pending.values())clearTimeout(p.timer);}})().catch(e=>{console.error(e);process.exitCode=1;});