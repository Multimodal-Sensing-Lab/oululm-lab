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
const target=await send('Target.createTarget',{url:'about:blank'},null);sessionId=(await send('Target.attachToTarget',{targetId:target.targetId,flatten:true},null)).sessionId;await send('Page.enable');await send('Runtime.enable');await send('Page.navigate',{url:pathToFileURL(path.join(__dirname,'data-lab.html')).href});await waitFor('document.querySelector("#data-rows tr") !== null');
assert.equal(await evaluate('document.querySelectorAll("#data-set option").length'),8);
assert.match(await evaluate('document.querySelector("#run-info").textContent'),/435,335/);
for(const model of ['0','1']){await change('lab-model',model);for(const stage of ['biology_base','finance_only','finance_replay']){await change('lab-stage',stage);assert.equal(await evaluate('document.querySelectorAll("#samples .mini-card").length'),4);assert.equal(await evaluate('document.querySelectorAll("#curve polyline").length'),3);}}
await change('data-set','finance');await change('data-split','validation');assert.match(await evaluate('document.getElementById("data-count").textContent'),/200 matching/);await change('data-category','calculations');assert.match(await evaluate('document.getElementById("data-count").textContent'),/125 matching/);await click('#data-next');assert.match(await evaluate('document.getElementById("data-count").textContent'),/21–40/);
await evaluate('document.getElementById("data-search").value="zzzzmissing";document.getElementById("data-search").dispatchEvent(new Event("input"))');assert.equal(await evaluate('document.querySelectorAll("#data-rows tr").length'),0);assert(await evaluate('document.getElementById("data-next").disabled'));
await evaluate('document.getElementById("data-search").value=""');await change('data-split','all');await change('data-set','stories');assert.match(await evaluate('document.getElementById("data-info").textContent'),/517/);await change('data-split','train');assert.equal(await evaluate('document.querySelectorAll("#data-rows tr").length'),3);
for(const width of [1440,1024,390]){await send('Emulation.setDeviceMetricsOverride',{width,height:1100,deviceScaleFactor:1,mobile:false});await change('lab-model','0');await change('lab-stage','finance_replay');await change('data-set','finance');assert(await evaluate('document.documentElement.scrollWidth<=innerWidth'),'Overflow at '+width);await screenshot('data-lab-'+width,width,1100);await evaluate('document.getElementById("data").scrollIntoView()');const shot=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(temporary,'data-records-'+width+'.png'),Buffer.from(shot.data,'base64'));}
assert.equal(exceptions.length,0);console.log('PASS: data lab curves, checkpoint specs, exact split/category filters, pagination, empty search and mobile: '+temporary);
}finally{await send('Browser.close',{},null).catch(()=>{});chrome.kill();for(const p of pending.values())clearTimeout(p.timer);}})().catch(e=>{console.error(e);process.exitCode=1;});
