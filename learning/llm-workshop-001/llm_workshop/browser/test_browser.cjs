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
(async()=>{
  try{
    const target=await send('Target.createTarget',{url:'about:blank'},null);
    sessionId=(await send('Target.attachToTarget',{targetId:target.targetId,flatten:true},null)).sessionId;
    await send('Page.enable');await send('Runtime.enable');await send('Network.enable');
    await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1050,deviceScaleFactor:1,mobile:false});
    await send('Page.navigate',{url:pathToFileURL(path.join(__dirname,'index.html')).href});
    await waitFor('document.querySelector("#scene .scene-head") !== null');
    assert.match(await evaluate('document.getElementById("model-meta").textContent'),/12 \/ 64/);
    await screenshot('overview');
    for(let chapter=0;chapter<8;chapter++){
      await click(`#chapters [data-chapter="${chapter}"]`);
      assert(await evaluate('document.querySelector("#scene .scene-head h2").textContent.length > 0'));
      for(const tab of ['explain','numbers','python']){
        await click(`[data-tab="${tab}"]`);
        assert(await evaluate('document.getElementById("detail").textContent.length > 150'));
      }
    }
    await click('[data-tab="explain"]');
    await click('#chapters [data-chapter="2"]');
    await prompt('aaa');
    await click('[data-embed="0"]');
    assert.equal(await evaluate('document.querySelectorAll(".heatmap").length'),1);
    await click('[data-embed="2"]');
    assert.equal(await evaluate('document.querySelectorAll(".heatmap").length'),3);
    await click('[data-cell="1,2"]');
    assert.match(await evaluate('document.querySelector(".cell-readout").textContent'),/POSITION 1.*FEATURE 2/s);
    await change('features','56');
    assert.match(await evaluate('document.querySelector(".feature-slider").textContent'),/56–63/);
    await change('features','0');
    await prompt('the cat sat.');
    await screenshot('embeddings');
    await click('#chapters [data-chapter="3"]');
    for(let stage=0;stage<7;stage++) await click(`[data-micro="${stage}"]`);
    await click('[data-micro="2"]');
    await click('.arithmetic summary');
    assert.equal(await evaluate('document.querySelectorAll(".arithmetic tbody tr").length'),32);
    await click('[data-micro="3"]');
    assert.equal(await evaluate('document.querySelectorAll(".heat-cell.masked").length'),66);
    await click('[data-pair="4,10"]');
    assert.match(await evaluate('document.querySelector(".cell-readout").textContent'),/−∞/);
    await click('[data-micro="4"]');
    assert.equal(await evaluate('document.querySelectorAll(".arithmetic tbody tr").length'),12);
    await change('reader','0');
    assert.match(await evaluate('document.querySelector(".cell-readout").textContent'),/1\.000000/);
    await change('reader','4');
    await screenshot('attention');
    await change('head','1');await change('block','1');
    await click('[data-micro="6"]');
    const before=await evaluate('document.querySelector(".cell-readout").textContent');
    await change('head','0');
    assert.equal(await evaluate('document.querySelector(".cell-readout").textContent'),before);
    await click('#chapters [data-chapter="4"]');
    await change('model','trained');
    const probs=await evaluate('document.querySelector(".prob-chart").textContent');
    await change('temperature','2');
    assert.notEqual(await evaluate('document.querySelector(".prob-chart").textContent'),probs);
    await click('#generate');
    assert.match(await evaluate('document.getElementById("model-meta").textContent'),/13 \/ 64/);
    await click('#reset-generation');
    assert.match(await evaluate('document.getElementById("model-meta").textContent'),/12 \/ 64/);
    await screenshot('prediction');
    await prompt('🦉');
    assert.equal(await evaluate('document.getElementById("error").hidden'),false);
    await prompt('a'.repeat(64));
    assert.equal(await evaluate('document.getElementById("generate").disabled'),true);
    await prompt('the cat sat.');
    await click('#chapters [data-chapter="5"]');await screenshot('training');
    await click('#chapters [data-chapter="6"]');
    await evaluate(`(()=>{const el=document.getElementById('passages');el.value=el.value.replace('K142','M219');el.dispatchEvent(new Event('input',{bubbles:true}));})()`);
    assert.match(await evaluate('document.querySelector("#evidence-results pre").textContent'),/M219/);
    await evaluate(`(()=>{const el=document.getElementById('question');el.value='xyzzy';el.dispatchEvent(new Event('input',{bubbles:true}));})()`);
    assert.match(await evaluate('document.querySelector("#evidence-results pre").textContent'),/No matching passage/);
    // Text is escaped rather than interpreted as HTML.
    await evaluate(`(()=>{const el=document.getElementById('passages');el.value='<img src=x onerror="throw Error(123)">';el.dispatchEvent(new Event('input',{bubbles:true}));})()`);
    assert.equal(await evaluate('document.querySelectorAll("#evidence-results img").length'),0);
    for(const scenario of ['feedback','methods','evidence']) await click(`[data-scenario="${scenario}"]`);
    await click('#chapters [data-chapter="7"]');
    await click('[data-answer="0,1"]');
    assert.equal(await evaluate('document.querySelectorAll(".feedback.wrong").length'),1);
    await click('[data-answer="0,0"]');
    assert.equal(await evaluate('document.querySelectorAll(".feedback.wrong").length'),0);
    for(const [question,correct] of [[0,0],[1,1],[2,2],[3,0]]){
      for(let choice=0;choice<3;choice++){
        await click(`[data-answer="${question},${choice}"]`);
        const feedback=await evaluate(`(()=>{const card=document.querySelectorAll('.quiz-card')[${question}],p=card.querySelector('.feedback');return {wrong:p.classList.contains('wrong'),text:p.textContent,selected:card.querySelector('[aria-pressed="true"]').dataset.answer};})()`);
        assert.equal(feedback.wrong,choice!==correct);
        assert.equal(feedback.selected,`${question},${choice}`);
        assert.match(feedback.text,/You chose:.*Correct answer:/);
        if(question===1)assert.match(feedback.text,/mixing weights total 100%.*not raw QK scores or vocabulary probabilities/);
      }
    }
    await click('[data-tab="explain"]');
    await evaluate(`document.getElementById('tab-explain').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}))`);
    assert.equal(await evaluate('document.activeElement.id'),'tab-numbers');
    // Ensure each rendered chapter has no duplicate IDs or document-level overflow.
    for(const width of [1440,1024,390]){
      await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
      for(let chapter=0;chapter<8;chapter++){
        await click(`#chapters [data-chapter="${chapter}"]`);
        const info=await evaluate(`(()=>{const ids=[...document.querySelectorAll('[id]')].map(e=>e.id);return {overflow:document.documentElement.scrollWidth-innerWidth,duplicate:ids.length-new Set(ids).size};})()`);
        assert.equal(info.duplicate,0,`Duplicate IDs, chapter ${chapter}`);
        assert(info.overflow<=1,`Document overflow ${info.overflow}px at ${width}px, chapter ${chapter}`);
      }
    }
    await click('#chapters [data-chapter="2"]');await screenshot('mobile',390,1000);
    await send('Page.navigate',{url:pathToFileURL(path.join(__dirname,'../student-guide.html')).href});
    await waitFor('document.querySelector("#route") !== null');
    await screenshot('participant-guide',1280,1000);
    assert.equal(exceptions.length,0,JSON.stringify(exceptions));
    assert(requests.every(url=>url.startsWith('file:')||url.startsWith('data:')),`Unexpected external requests: ${requests}`);
    console.log(`PASS: eight chapters, every attention operation, tabs and keyboard, cell selection, masking, model switching, generation, invalid input, 64-token limit, evidence edits, quiz feedback, mobile layouts. No JS exceptions or external page requests. Screenshots: ${temporary}`);
  } finally {
    await send('Browser.close',{},null).catch(()=>{});
    chrome.kill();
    for(const item of pending.values())clearTimeout(item.timer);
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
