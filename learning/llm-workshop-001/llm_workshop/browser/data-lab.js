/* Recorded metrics and exact data only. No optimizer or hidden model call here. */
(() => {
'use strict';
const D=globalThis.WORKSHOP_DATA_LAB,$=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=n=>Number(n).toLocaleString(),f=n=>Number(n).toFixed(3);
let page=0;
$('lab-model').innerHTML=D.runs.map((r,i)=>`<option value="${i}">${r.manifest.tokenizer.kind==='word'?'Words + digits + punctuation':'Characters'}</option>`).join('');
$('data-set').innerHTML=D.datasets.map(d=>`<option value="${d.id}">${esc(d.label)}</option>`).join('');
function chart(stage){
 const h=stage.history,max=Math.ceil(Math.max(...h.flatMap(p=>[p.biology_validation,p.finance_validation,p.train_batch||0]))),width=780;
 const keys=[['biology_validation','#087e70','Biology validation'],['finance_validation','#a25225','Finance validation'],['train_batch','#7165a8','Last training batch']];
 const y=v=>225-v/max*165,x=v=>55+v/stage.steps*680;
 return `<svg viewBox="0 0 ${width} 275" role="img" aria-label="Measured loss curves for ${esc(stage.name)}. Lower is better. Step-by-step values in the linked report.">${keys.map(([k,c,l],i)=>`<text x="${55+i*240}" y="20" font-size="13" fill="${c}">${l}</text>`).join('')}${[0,.25,.5,.75,1].map(t=>`<line x1="55" x2="735" y1="${y(t*max)}" y2="${y(t*max)}" stroke="#dde3dd"/><text x="12" y="${y(t*max)+4}" font-size="12">${(max*t).toFixed(1)}</text>`).join('')}${keys.map(([k,c])=>`<polyline points="${h.filter(p=>p[k]!==null).map(p=>`${x(p.step)},${y(p[k])}`).join(' ')}" fill="none" stroke="${c}" stroke-width="2.5"/>`).join('')}<text x="55" y="248" font-size="12">0 updates</text><text x="635" y="248" font-size="12">${stage.steps} updates</text><text x="280" y="270" font-size="12">Cross-entropy · nats per target token</text></svg>`;
}
function renderRun(){
 const r=D.runs[Number($('lab-model').value)],s=r.stages.find(s=>s.name===$('lab-stage').value),m=r.manifest,c=m.config;
 $('run-info').innerHTML=`<h2>${s.name==='biology_base'?'Biology pretraining':s.name==='finance_only'?'Finance fine-tuning':'Finance fine-tuning with biology replay'}</h2><p class="method-note">Input: <b>${num(m.splits[s.domain].train.records)} ${s.domain} training records</b>${s.name==='finance_replay'?' + biology training records sampled for replay':''}. Holdout: <b>${num(m.splits.biology.validation.records)} biology + ${num(m.splits.finance.validation.records)} finance</b> records.</p><div class="specs">${[[num(s.steps),'optimizer updates'],[s.epochs,'epochs over '+s.domain],[s.learning_rate,'learning rate'],[s.batch_size,'maximum batch records'],[num(c.vocab_size),'fixed vocabulary entries'],[c.block_size,'context positions'],[num(r.parameter_count),'trainable parameters'],[s.training_seconds.toFixed(1)+' s','measured loop + evaluation + plots']].map(([v,k])=>`<div><strong>${v}</strong><span>${k}</span></div>`).join('')}</div><p class="subtle">${esc(r.device_name)} · ${c.n_layer} blocks · ${c.n_head} heads · ${c.n_embd} features · AdamW · dropout ${c.dropout} · seed ${r.seed}. Full-weight training. ${num(s.target_exposures)} target predictions presented, including repeated passes.</p><p class="subtle">${esc(r.timing)} This is one small-model recording, not a runtime promise for larger models. <a href="../../${r.path}/report.json">Full settings, metrics and samples</a> · <a href="../../${r.path}/data_manifest.json">Input hashes and split counts</a> · <a href="../../${r.path}/${s.name}_progress.png">Saved Python plot</a></p>`;
 $('curve').innerHTML=chart(s);
 $('comparison').innerHTML=`<h3>Compare final checkpoints on the same held-out data</h3><table class="data-table results"><thead><tr><th>Checkpoint</th><th>Biology loss ↓</th><th>Finance loss ↓</th></tr></thead><tbody>${r.stages.map(stage=>`<tr><th>${{biology_base:'Biology base',finance_only:'Finance only',finance_replay:'Finance + replay'}[stage.name]}</th><td>${f(stage.history.at(-1).biology_validation)}</td><td>${f(stage.history.at(-1).finance_validation)}</td></tr>`).join('')}</tbody></table>`;
 $('samples').innerHTML=`<h3>Actual greedy continuations · same four prompts</h3><p class="subtle">These are saved outputs from the selected checkpoint, including its errors. Fluent-looking biology and formatted numbers can still be wrong. This model is not a calculator or an instruction assistant.</p><div class="word-samples">${s.samples.map(p=>`<div class="mini-card"><strong>Input: ${esc(p.prompt)}</strong><p>${esc(p.text)}</p></div>`).join('')}</div><details><summary>Run this experiment in Python</summary><pre class="code">python -m llm_workshop.domains.train --tokenization ${m.tokenizer.kind} --device cuda \
  --output-dir runs/my_${m.tokenizer.kind}_experiment --live-plot

python -m llm_workshop.domains.demo \
  --checkpoint ${r.path}/${s.checkpoint} --text 'the blue mussel'</pre><p>Use a new output directory each time. Omit --live-plot for headless runs: PNG plots still update on disk. This page loads bundled recordings; use the documented export command to show a new run here.</p></details>`;
}
function dataset(){return D.datasets.find(d=>d.id===$('data-set').value);}
function setCategories(){const d=dataset();$('data-category').innerHTML='<option value="all">All categories</option>'+Object.keys(d.stats.categories).sort().map(c=>`<option>${esc(c)}</option>`).join('');page=0;renderData();}
function renderData(){
 const d=dataset(),split=$('data-split').value,cat=$('data-category').value,q=$('data-search').value.toLowerCase();
 const rows=d.rows.filter(r=>(split==='all'||r.split===split)&&(cat==='all'||r.category===cat)&&r.sentence.toLowerCase().includes(q));
 page=Math.min(page,Math.max(0,Math.ceil(rows.length/20)-1));
 const scope=d.rows.filter(r=>split==='all'||r.split===split),chars=scope.reduce((n,r)=>n+r.sentence.length,0),words=scope.reduce((n,r)=>n+r.sentence.trim().split(/\s+/).length,0);
 $('data-info').innerHTML=`<p><b>${num(scope.length)} records · ${num(words)} whitespace words · ${num(chars)} text characters</b> in ${split==='all'?'all splits':esc(split)} (before category/search filters). ${d.stats.bytes?num(d.stats.bytes)+' bytes in the complete source CSV.':''}</p><p class="subtle">${esc(d.counting)} ${esc(d.note)}</p>${d.stats.path?`<p><a href="../../${esc(d.stats.path.replace(/^.*?OuluLLM Lab\//,''))}">Open source CSV</a> · SHA-256: <code class="hash">${esc(d.stats.sha256)}</code></p>`:''}<details><summary>Category counts · complete dataset</summary><p>${Object.entries(d.stats.categories).map(([c,n])=>`${esc(c)}: ${num(n)} (${(n/d.stats.records*100).toFixed(1)}%)`).join(' · ')}</p></details>`;
 $('data-count').textContent=`${num(rows.length)} matching records · ${rows.length?page*20+1:0}–${Math.min(rows.length,(page+1)*20)} shown`;
 $('data-rows').innerHTML=rows.slice(page*20,(page+1)*20).map(r=>`<tr><td>${esc(r.id)}</td><td>${esc(r.split)}<br>${esc(r.category)}</td><td>${esc(r.sentence)}</td></tr>`).join('');
 $('data-prev').disabled=page===0;$('data-next').disabled=(page+1)*20>=rows.length;
}
for(const id of ['lab-model','lab-stage'])$(id).addEventListener('change',renderRun);
$('data-set').addEventListener('change',setCategories);
for(const id of ['data-split','data-category','data-search'])$(id).addEventListener(id==='data-search'?'input':'change',()=>{page=0;renderData();});
$('data-prev').onclick=()=>{page--;renderData();};$('data-next').onclick=()=>{page++;renderData();};
const query=new URLSearchParams(location.search);if(D.datasets.some(d=>d.id===query.get('dataset')))$('data-set').value=query.get('dataset');if(query.get('tokenization')==='character')$('lab-model').value=String(D.runs.findIndex(r=>r.manifest.tokenizer.kind==='character'));
renderRun();setCategories();
})();
