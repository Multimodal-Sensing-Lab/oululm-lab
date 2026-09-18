"""Serve the workshop locally and generate with the actual saved domain checkpoints.

python -m llm_workshop.domains.serve
Open http://127.0.0.1:8765/llm_workshop/browser/domain-lab.html
No external APIs, training, email sending, or checkpoint writes.
"""
import argparse,json,threading
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit
import torch
from llm_workshop.io import sha256
from .prepare import ROOT
from .train import load
from .finetune import prompt_for,predict

LOCK=threading.Lock()


@lru_cache(maxsize=4)
def checkpoint(kind,stage):
    catalog=json.loads((ROOT/'llm_workshop/browser/domain-catalog.json').read_text())
    item=next((r for r in catalog['runs'] if r['id']==kind),None)
    if item is None:raise ValueError('Choose a listed tokenization.')
    folder=ROOT/item['path'];report=json.loads((folder/'report.json').read_text())
    record=next((s for s in report['stages'] if s['name']==stage),None)
    if record is None:raise ValueError('Choose a listed checkpoint.')
    path=folder/record['checkpoint']
    if sha256(path)!=record['sha256']:raise ValueError('Checkpoint hash differs from its report; re-export a verified recording.')
    model,tok,state=load(path)
    return model,tok,record


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT),**kw)
    def reply(self,status,payload):
        data=json.dumps(payload,ensure_ascii=False).encode();self.send_response(status)
        self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_GET(self):
        if urlsplit(self.path).path=='/api/workshop/health':return self.reply(200,{'service':'llm-workshop-domain-inference','device':'CPU','external_api':False})
        if self.path=='/':
            self.send_response(302);self.send_header('Location','/llm_workshop/browser/domain-lab.html');self.end_headers();return
        super().do_GET()
    def do_POST(self):
        if self.path!='/api/workshop/generate':return self.reply(404,{'error':'Unknown endpoint.'})
        if self.headers.get_content_type()!='application/json':return self.reply(415,{'error':'Use application/json.'})
        origin=self.headers.get('Origin')
        if origin and urlsplit(origin).netloc!=self.headers.get('Host'):return self.reply(403,{'error':'Open the page from this local server.'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=16000:raise ValueError('Request is too large or empty.')
            body=json.loads(self.rfile.read(length));text=body.get('text','');incoming=body.get('incoming','')
            if not isinstance(text,str) or not 1<=len(text)<=3000 or not isinstance(incoming,str) or len(incoming)>3000:raise ValueError('Use a nonempty input of at most 3,000 characters per field.')
            budget=body.get('tokens',96)
            if type(budget)!=int or not 1<=budget<=200:raise ValueError('Choose 1–200 new tokens.')
            task=body.get('task','prose')
            if task not in ['prose','email','qa']:raise ValueError('Choose prose, email, or qa format.')
            with LOCK:
                model,tok,record=checkpoint(body.get('kind'),body.get('stage'))
                prompt=prompt_for(task,text,incoming)
                result=predict(model,tok,prompt,budget)
            self.reply(200,{**result,'serialized_input':prompt,'checkpoint':record['name'],'checkpoint_sha256':record['sha256'],'task':task})
        except (ValueError,KeyError,TypeError,json.JSONDecodeError) as e:self.reply(400,{'error':str(e)})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8765);a=p.parse_args()
    torch.set_num_threads(2)
    with ThreadingHTTPServer(('127.0.0.1',a.port),Handler) as server:
        print(f'Workshop: http://127.0.0.1:{a.port}/llm_workshop/browser/domain-lab.html',flush=True);server.serve_forever()


if __name__=='__main__':main()
