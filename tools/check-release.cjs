/* Run from any directory: node tools/check-release.cjs */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),course=path.join(root,'learning/llm-workshop-001');
function walk(dir){return fs.readdirSync(dir,{withFileTypes:true}).filter(e=>!['.git','__pycache__','.venv','.pytest_cache','node_modules'].includes(e.name)).flatMap(e=>e.isDirectory()?walk(path.join(dir,e.name)):[path.join(dir,e.name)]);}
const files=walk(root),errors=[];let links=0,checkpoints=0;
for(const file of files){
 const rel=path.relative(root,file);
 if(fs.statSync(file).size>=100*1024*1024)errors.push('Oversized file: '+rel);
 if(/(^|\/)(INSTRUCTOR[^/]*|instructor\.html|guide\.html|\.env|\.idea|\.agent_harnesses.json)(\/|$)/i.test(rel)||/\.(pptx|mp4|mov)$/.test(rel))errors.push('Excluded material: '+rel);
 if(!/\.(html|md)$/.test(file))continue;
 const text=fs.readFileSync(file,'utf8');
 for(const m of [...text.matchAll(/(?:href|src)=["']([^"']+)["']/g),...text.matchAll(/\]\(([^)]+)\)/g)]){
  let url=m[1];if(/^(https?:|mailto:|data:|#)/.test(url)||/[{}<>]/.test(url))continue;
  url=url.split('#')[0].split('?')[0];if(!url)continue;
  const base=text.match(/<base\s+href=["']([^"']+)["']/i);
  const directory=base&&url!==base[1]?path.resolve(path.dirname(file),base[1]):path.dirname(file);
  const target=path.resolve(directory,decodeURI(url));links++;
  if(!target.startsWith(root+path.sep)||!fs.existsSync(target))errors.push('Broken/local-outside link: '+rel+' -> '+url);
 }
}
const catalog=JSON.parse(fs.readFileSync(path.join(course,'llm_workshop/browser/domain-catalog.json')));
for(const item of catalog.runs){
 const folder=path.join(course,item.path),report=JSON.parse(fs.readFileSync(path.join(folder,'report.json')));
 for(const stage of report.stages){const digest=crypto.createHash('sha256').update(fs.readFileSync(path.join(folder,stage.checkpoint))).digest('hex');assert.equal(digest,stage.sha256,item.id+' '+stage.name);checkpoints++;}
}
assert.equal(fs.existsSync(path.join(course,'llm_workshop/index.html')),false,'Private workshop hub must not be included');
assert.equal(errors.length,0,errors.join('\n'));
console.log('PASS: '+files.length+' release files, '+links+' local links, '+checkpoints+' live checkpoint hashes; no excluded files or files over 100 MiB.');
