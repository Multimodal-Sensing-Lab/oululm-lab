/* Compare the bundled word browser's calculations with measured Python tensors. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
require('./word-model-data.js');
const E=require('./engine.js'),D=globalThis.WORKSHOP_WORD_MODEL;
const fixtures=JSON.parse(fs.readFileSync(path.join(__dirname,'word-reference.json'),'utf8'));
let comparisons=0,maxError=0;
function close(actual,expected,label){
 if(Array.isArray(expected)){assert.equal(actual.length,expected.length,label);expected.forEach((v,i)=>close(actual[i],v,label+'['+i+']'));}
 else if(expected===null)assert.equal(actual,-Infinity,label);
 else {const error=Math.abs(Number(actual)-Number(expected));assert(error<=3e-5+Math.abs(Number(expected))*1e-5,`${label}: ${actual} != ${expected}`);maxError=Math.max(error,maxError);comparisons++;}
}
for(const fixture of fixtures){
 const ids=E.encodeWords(fixture.text,D.vocabulary,D.config.block_size),trace=E.forward(ids,D.models[fixture.model],D.config);
 close(trace.logits,fixture.logits,'logits');
 for(const key of ['tokens','positions','combined'])close(trace[key],fixture.embeddings[key],key);
 fixture.blocks.forEach((heads,b)=>heads.forEach((expected,h)=>{
  for(const key of Object.keys(expected))close(trace.blocks[b].heads[h][key]||trace.blocks[b][key],expected[key],`${b}/${h}/${key}`);
 }));
}
assert.deepEqual(E.encodeWords('  THE\tSMALL cat ',D.vocabulary,16),E.encodeWords('the small cat',D.vocabulary,16));
assert.throws(()=>E.encodeWords('the dragon',D.vocabulary,16));
assert.throws(()=>E.encodeWords('the € cat',D.vocabulary,16));
assert.throws(()=>E.encodeWords('',D.vocabulary,16));
assert.throws(()=>E.encodeWords('cat '.repeat(16),D.vocabulary,16));
const repeated=E.encodeWords('the cat cat',D.vocabulary,16),trace=E.forward(repeated,D.models.trained,D.config);
close(trace.tokens[2],trace.tokens[3],'Repeated word lookup');
assert.notDeepEqual(trace.positions[2],trace.positions[3]);
const changed=E.forward(E.encodeWords('the cat dog',D.vocabulary,16),D.models.trained,D.config);
close(trace.logits.slice(0,3),changed.logits.slice(0,3),'Causal independence');
for(const block of trace.blocks)for(const head of block.heads)head.attention.forEach((row,i)=>{
 assert(Math.abs(row.reduce((a,b)=>a+b,0)-1)<1e-12);assert(row.slice(i+1).every(v=>v===0));
});
for(const sample of D.training.samples)for(const name of ['untrained','trained']){
 const ids=E.encodeWords(sample.prompt,D.vocabulary,16);
 for(let n=0;n<12&&ids.length<16;n++){
  const logits=E.forward(ids,D.models[name],D.config).logits.at(-1),chosen=logits.indexOf(Math.max(...logits));
  ids.push(chosen);if(chosen===1)break;
 }
 const expected=sample[name==='untrained'?'before':'after'];
 assert.deepEqual(ids,expected.ids,'Fixed greedy generation including EOS/context stop');
 assert.equal(E.decodeWords(ids,D.vocabulary),expected.text);
}
assert.equal(E.decodeWords([0,2,1],D.vocabulary),'<UNK>');
const full=E.forward([0,...Array(15).fill(D.vocabulary.indexOf('cat'))],D.models.trained,D.config);
assert(full.logits.flat().every(Number.isFinite));
console.log(`PASS: word tokenizer, ${comparisons} Python/JS scalar comparisons (max error ${maxError}), repeated words, causality, masks, full context and before/after greedy continuations.`);
