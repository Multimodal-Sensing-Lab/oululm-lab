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
await send('Page.navigate',{url:pathToFileURL(path.join(__dirname,'domain-lab.html')).href});await waitFor('document.querySelector("#domain-specs p")!==null');
assert.equal(await evaluate('document.getElementById("domain-generate").disabled'),true);
assert.match(await evaluate('document.getElementById("domain-specs").textContent'),/same base.pt.*full-weight/);
for(const kind of ['word','character']){await change('domain-kind',kind);for(const stage of ['base','biology','email','finance','qa']){await change('domain-stage',stage);assert.match(await evaluate('document.getElementById("domain-specs").textContent'),new RegExp(stage==='base'?'1,000 updates':'500 updates'));await change('prepared-split','validation');assert.match(await evaluate('document.getElementById("prepared-view").textContent'),/validation/);}}
await change('domain-stage','email');assert.match(await evaluate('document.getElementById("prepared-view").textContent'),/Prompt.*loss masked[\s\S]*Response/);
await evaluate('document.getElementById("prepared-search").value="impossible_search_12345";document.getElementById("prepared-search").dispatchEvent(new Event("input"))');assert.match(await evaluate('document.getElementById("prepared-view").textContent'),/No matching records/);
for(const width of [1440,390]){await send('Emulation.setDeviceMetricsOverride',{width,height:1100,deviceScaleFactor:1,mobile:false});for(const s of ['biology','email','qa']){await change('domain-stage',s);assert(await evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Overflow '+width+' '+s);}await screenshot('domain-'+width,width,1100);}
assert(requests.every(u=>u.startsWith('file:')||u.startsWith('data:')),'Offline page made a network request');
await send('Page.navigate',{url:(process.env.WORKSHOP_URL||'http://127.0.0.1:8765')+'/llm_workshop/browser/domain-lab.html?domain=biology'});await waitFor('document.getElementById("domain-generate")&&!document.getElementById("domain-generate").disabled');
// Reproduce the reported failures and check that remedies preserve the user's text.
await change('domain-task','email');
await input('domain-input','Write an answer');
await input('domain-incoming','Hello,\n\nI am arriving late');
assert.equal(await evaluate('document.getElementById("task-warning").hidden'),false);
assert.match(await evaluate('document.getElementById("input-model").textContent'),/Words · Biology.*Trained task: prose/);
assert.match(await evaluate('document.getElementById("vocabulary-message").textContent'),/am, arriving, late/);
await click('#use-task-branch');
assert.equal(await evaluate('document.getElementById("domain-stage").value'),'email');
assert.equal(await evaluate('document.getElementById("task-warning").hidden'),true);
assert.equal(await evaluate('document.getElementById("vocabulary-check").hidden'),false);
await click('#use-characters');
assert.equal(await evaluate('document.getElementById("domain-input").value'),'Write an answer');
assert.equal(await evaluate('document.getElementById("domain-incoming").value'),'Hello,\n\nI am arriving late');
assert.equal(await evaluate('document.getElementById("vocabulary-check").hidden'),true);
assert.match(await evaluate('document.getElementById("input-status").textContent'),/94 input tokens/);
await click('#domain-generate');await waitFor('document.getElementById("output-label").textContent.startsWith("LIVE")');
assert.match(await evaluate('document.getElementById("output-label").textContent'),/Characters · Email/);
assert.match(await evaluate('document.getElementById("serialized-input").textContent'),/I am arriving late/);
await change('domain-kind','word');await change('domain-stage','biology');await change('domain-task','qa');await input('domain-input','where is the cat?');
assert.equal(await evaluate('document.getElementById("incoming-wrap").hidden'),true);
assert.equal(await evaluate('document.getElementById("domain-stage").value'),'biology');
assert.match(await evaluate('document.getElementById("input-status").textContent'),/10 input tokens/);
await click('#domain-generate');await waitFor('document.getElementById("output-label").textContent.startsWith("LIVE")');
assert.equal(await evaluate('document.getElementById("domain-output").textContent'),'.');
assert.match(await evaluate('document.getElementById("output-explanation").textContent'),/no words or numbers.*EOS/);
await click('#use-task-branch');
assert.equal(await evaluate('document.getElementById("domain-input").value'),'where is the cat?');
assert.equal(await evaluate('document.getElementById("domain-stage").value'),'qa');
await click('#domain-generate');await waitFor('document.getElementById("output-label").textContent.startsWith("LIVE")');
assert.equal(await evaluate('document.getElementById("domain-output").textContent'),'an a cat is a small mammal.');
assert.match(await evaluate('document.getElementById("task-description").textContent'),/location/);
// Long mismatch and vocabulary notices must fit both laptop and phone layouts.
await change('domain-stage','biology');await change('domain-task','email');await input('domain-input','Write an answer');await input('domain-incoming','Hello,\n\nI am arriving late');
for(const width of [1440,390]){await send('Emulation.setDeviceMetricsOverride',{width,height:1100,deviceScaleFactor:1,mobile:false});assert(await evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Input warnings overflow '+width);await evaluate('document.getElementById("task-description").scrollIntoView()');const shot=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(temporary,'input-warning-'+width+'.png'),Buffer.from(shot.data,'base64'));}
await change('domain-stage','qa');
for(const kind of ['word','character']){await change('domain-kind',kind);await click('#domain-generate');await waitFor('document.getElementById("output-label").textContent.startsWith("LIVE")');assert.equal(await evaluate('document.getElementById("domain-output").textContent'),'a cat is a small mammal. it has fur and is often kept as a pet.');}
await change('domain-kind','word');await change('domain-inspect','base');assert.equal(await evaluate('document.getElementById("domain-input").value'),'what is a cat?');await click('#domain-generate');await waitFor('document.getElementById("output-label").textContent.startsWith("LIVE")');assert.notEqual(await evaluate('document.getElementById("domain-output").textContent'),'a cat is a small mammal. it has fur and is often kept as a pet.');
await evaluate('document.getElementById("domain-input").value="zzzzunknownword";document.getElementById("domain-input").dispatchEvent(new Event("input"))');await click('#domain-generate');await waitFor('!document.getElementById("domain-error").hidden');assert.match(await evaluate('document.getElementById("domain-error").textContent'),/Outside this checkpoint vocabulary/);
assert.equal(exceptions.length,0,JSON.stringify(exceptions));console.log('PASS: all 10 checkpoint branches, source/target records, offline mode, reported email/vocabulary and biology/Q&A failures, model/task guidance, input-preserving remedies, actual local generation, base/after comparisons, desktop/mobile layouts. Screenshots: '+temporary);
}finally{await send('Browser.close',{},null).catch(()=>{});chrome.kill();for(const p of pending.values())clearTimeout(p.timer);}})().catch(e=>{console.error(e);process.exitCode=1;});
