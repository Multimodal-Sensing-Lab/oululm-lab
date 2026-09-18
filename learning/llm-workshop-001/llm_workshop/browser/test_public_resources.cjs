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
async function input(id,value){
  await evaluate(`(()=>{const el=document.getElementById(${JSON.stringify(id)});el.value=${JSON.stringify(value)};el.dispatchEvent(new Event('input',{bubbles:true}));})()`);
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
const target=await send('Target.createTarget',{url:'about:blank'},null);sessionId=(await send('Target.attachToTarget',{targetId:target.targetId,flatten:true},null)).sessionId;await send('Page.enable');await send('Runtime.enable');await send('Network.enable');
for(const file of ['../glossary.html','../001_text_to_predictions/walkthrough.html','../python-lab.html','../references.html','../student-guide.html','../002_embeddings_and_positions/index.html']){
 await send('Page.navigate',{url:pathToFileURL(path.join(__dirname,file)).href});await waitFor('document.querySelector(".resource-brand img")?.complete');
 assert(await evaluate('document.querySelector(".resource-brand img").naturalWidth>0'),'Missing logo '+file);
 assert.equal(await evaluate('document.querySelectorAll("a[href*=INSTRUCTOR],a[href*=instructor],a[href*=presentations]").length'),0);
 for(const width of [1440,390]){await send('Emulation.setDeviceMetricsOverride',{width,height:1050,deviceScaleFactor:1,mobile:false});assert(await evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Overflow '+file+' '+width);await screenshot(path.basename(file,'.html')+'-'+width,width,1050);}
}
assert.equal(exceptions.length,0,JSON.stringify(exceptions));assert(requests.every(u=>u.startsWith('file:')||u.startsWith('data:')));console.log('PASS: participant resources, logos, navigation and mobile layouts. Screenshots: '+temporary);
}finally{await send('Browser.close',{},null).catch(()=>{});chrome.kill();for(const p of pending.values())clearTimeout(p.timer);}})().catch(e=>{console.error(e);process.exitCode=1;});
