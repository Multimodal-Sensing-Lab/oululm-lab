/* Validate recorded checkpoints against independent PyTorch exports and LoRA identity. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
require('./model-data.js');require('./experiment-data.js');
const E=require('./engine.js'),D=globalThis.WORKSHOP_MODEL,X=globalThis.WORKSHOP_EXPERIMENTS;
assert.deepEqual(X.config,D.config);assert.deepEqual(X.vocabulary,D.vocabulary);
const models={...D.models,...X.models};
let comparisons=0,maxError=0;
for(const fixture of X.reference_logits){
  const actual=E.forward(E.encode(fixture.text,D.vocabulary,64),models[fixture.model],D.config).logits;
  assert.equal(actual.length,fixture.logits.length);
  actual.forEach((row,i)=>row.forEach((value,j)=>{
    const expected=fixture.logits[i][j],error=Math.abs(value-expected);
    maxError=Math.max(maxError,error);comparisons++;
    assert(error<=3e-5+Math.abs(expected)*1e-5,fixture.model+' logits mismatch');
  }));
}
for(const run of X.runs){
  assert(models[run.before]&&models[run.after]);
  assert.equal(run.history[0].step,0);assert.equal(run.history.at(-1).step,run.steps);
  for(const point of run.history)assert(Number.isFinite(point.train)&&Number.isFinite(point.validation));
  if(run.device==='cuda')assert.match(run.device_name,/NVIDIA/);
  for(const data of Object.values(run.corpus)){
    const file=path.resolve(__dirname,'../..',data.path);
    const rows=fs.readFileSync(file,'utf8').trim().split('\n').map(JSON.parse);
    const text=rows.map(r=>r.text).join('\n\n');
    assert.equal(crypto.createHash('sha256').update(text).digest('hex'),data.sha256);
    assert.equal(text.length,data.characters);assert.equal(rows.length,data.documents);
  }
  if(run.kind==='adapter'){
    const base=models[run.before].weights,adapted=models[run.after].weights;
    assert.equal(run.trainable_parameters,run.rank*(D.config.n_embd+D.config.vocab_size));
    for(const key of Object.keys(base)){
      if(key!=='lm_head.weight')assert.deepEqual(adapted[key],base[key],'Frozen parameter changed: '+key);
    }
    for(let row=0;row<D.config.vocab_size;row++)for(let f=0;f<D.config.n_embd;f++){
      const delta=run.b[row].reduce((sum,b,k)=>sum+b*run.a[k][f],0)*run.scale;
      assert(Math.abs(base['lm_head.weight'][row][f]+delta-adapted['lm_head.weight'][row][f])<2e-6,'LoRA merge mismatch');
    }
  }
}
assert.deepEqual(models.stories_cpu_200.weights,models.trained.weights,'Original CPU training should reproduce');
console.log(`PASS: ${comparisons} exported PyTorch logits (max error ${maxError.toExponential(2)}), corpus hashes/counts, CPU reproduction, and frozen-base/LoRA merge for all domains.`);
