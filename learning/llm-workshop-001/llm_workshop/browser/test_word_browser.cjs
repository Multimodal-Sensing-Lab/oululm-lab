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
async function screenshot(name,width=1440,height=1050){
  await send('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
  await evaluate('window.scrollTo(0,0)');
  const result=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  fs.writeFileSync(path.join(temporary,name+'.png'),Buffer.from(result.data,'base64'));
}
async function wordPrompt(text) {
 await evaluate(`(()=>{document.getElementById('word-prompt').value=${JSON.stringify(text)};document.getElementById('word-form').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}));})()`);
}
(async()=>{
 try {
  const target=await send('Target.createTarget',{url:'about:blank'},null);
  sessionId=(await send('Target.attachToTarget',{targetId:target.targetId,flatten:true},null)).sessionId;
  await send('Page.enable');await send('Runtime.enable');await send('Network.enable');
  await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1050,deviceScaleFactor:1,mobile:false});
  await send('Page.navigate',{url:pathToFileURL(path.join(__dirname,'words.html')).href});
  await waitFor('document.querySelectorAll(".word-token").length === 4');
  assert.match(await evaluate('document.getElementById("word-status").textContent'),/TinyGPT words.*random initialization.*4 \/ 16/);
  assert.match(await evaluate('document.getElementById("word-model-info").textContent'),/432.*600/);
  await screenshot('word-tokens');
  await wordPrompt('the cat cat');
  const labels=await evaluate('[...document.querySelectorAll(".word-token small")].map(x=>x.textContent.split("ID ")[1])');
  assert.equal(labels[2],labels[3]);
  await click('[data-stage="1"]');
  const rows=await evaluate('[...document.querySelectorAll(".word-table tbody tr")].map(r=>[...r.querySelectorAll("button")].map(b=>b.textContent))');
  assert.deepEqual(rows[2],rows[3]);
  await click('[data-cell="3,0"]');
  assert.match(await evaluate('document.querySelector(".word-selection").textContent'),/position 3/);
  await change('word-matrix','positions');
  const positions=await evaluate('[...document.querySelectorAll(".word-table tbody tr")].map(r=>[...r.querySelectorAll("button")].map(b=>b.textContent))');
  assert.notDeepEqual(positions[2],positions[3]);
  await evaluate('document.getElementById("word-feature").focus()');
  await change('word-feature','63');
  assert.equal(await evaluate('document.activeElement.id'),'word-feature');
  assert.match(await evaluate('document.querySelector(".word-selection").textContent'),/feature 63/);
  await click('[data-view="attention"]');
  await click('[data-pair="0,3"]');
  assert.match(await evaluate('document.querySelector(".word-selection").textContent'),/After mask: −∞.*0.000000/);
  await click('[data-pair="0,0"]');
  assert.match(await evaluate('document.querySelector(".word-selection").textContent'),/1.000000/);
  await change('word-block','1');await change('word-head','1');
  await screenshot('word-attention');
  await click('[data-view="flow"]');
  assert.match(await evaluate('document.getElementById("word-scene").textContent'),/52 logits/);
  await wordPrompt('the small cat');
  await change('word-model','trained');
  await click('[data-stage="2"]');
  const original=await evaluate('[...document.querySelectorAll(".word-prob")].map(x=>x.textContent)');
  await change('word-temperature','2');
  assert.notDeepEqual(await evaluate('[...document.querySelectorAll(".word-prob")].map(x=>x.textContent)'),original);
  for(let i=0;i<12;i++){
   if(await evaluate('document.getElementById("word-generate").disabled'))break;
   await click('#word-generate');
  }
  assert.match(await evaluate('document.querySelector(".word-selection").textContent'),/END was generated/);
  assert.equal(await evaluate('document.querySelector(".word-output").textContent'),await evaluate('WORKSHOP_WORD_MODEL.training.samples.find(s=>s.prompt==="the small cat").after.text'));
  await screenshot('word-generation');
  await click('#word-reset');
  assert.equal(await evaluate('document.querySelector(".word-output").textContent'),'the small cat');
  await wordPrompt('the dragon');
  assert.equal(await evaluate('document.getElementById("word-error").hidden'),false);
  await wordPrompt('cat '.repeat(16));
  assert.equal(await evaluate('document.getElementById("word-error").hidden'),false);
  await wordPrompt('cat '.repeat(15));
  assert.equal(await evaluate('document.getElementById("word-generate").disabled'),true);
  await wordPrompt('the small cat');
  await click('[data-stage="3"]');
  assert.equal(await evaluate('document.querySelectorAll(".word-samples .mini-card").length'),4);
  assert((await evaluate('document.querySelector(".word-table").textContent')).includes(await evaluate('WORKSHOP_WORD_MODEL.training.test.after.toFixed(4)')));
  await screenshot('word-training');
  await click('[data-model="untrained"]');
  assert.equal(await evaluate('document.getElementById("word-model").value'),'untrained');
  for(const width of [1440,1024,390]){
   await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
   for(let stage=0;stage<4;stage++){
    await click('[data-stage="'+stage+'"]');
    const info=await evaluate('(()=>{const ids=[...document.querySelectorAll("[id]")].map(x=>x.id);return {overflow:document.documentElement.scrollWidth-innerWidth,duplicates:ids.length-new Set(ids).size};})()');
    assert(info.overflow<=1,'Overflow '+width+' stage '+stage+': '+info.overflow);
    assert.equal(info.duplicates,0);
    await click('#word-explanation summary');
    assert.match(await evaluate('document.querySelector("#word-explanation pre").textContent'),/ids|optimizer/);
   }
  }
  await click('[data-stage="0"]');await screenshot('word-mobile',390,1000);
  assert.equal(exceptions.length,0,JSON.stringify(exceptions));
  assert(requests.every(url=>url.startsWith('file:')||url.startsWith('data:')),'Unexpected external page request');
  console.log('PASS: four word stages, repeated IDs/embeddings, positions, masking, block/head controls, greedy END, temperature, reset, unknown words, token limits, checkpoint selection, Python panels and responsive layouts. Screenshots: '+temporary);
 } finally {await send('Browser.close',{},null).catch(()=>{});chrome.kill();for(const item of pending.values())clearTimeout(item.timer);}
})().catch(error=>{console.error(error);process.exitCode=1;});
