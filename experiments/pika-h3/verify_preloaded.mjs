import {spawn} from 'node:child_process';
import {resolve} from 'node:path';
import {writeFile} from 'node:fs/promises';
const root=resolve(process.argv[2]||'agent/pika-fp8/pika-preloaded-five-video-comparison-20261102');
const server=spawn('/usr/bin/python3',['-m','http.server','18979','--bind','127.0.0.1','--directory',root],{stdio:'ignore'});
await new Promise(r=>setTimeout(r,500));
const browser=spawn(process.env.CHROME_BINARY||resolve(root,'../browser-runtime/chrome-linux64/chrome'),[
 '--headless','--disable-gpu','--no-sandbox','--mute-audio','--autoplay-policy=no-user-gesture-required','--remote-debugging-port=0',
 `--user-data-dir=${resolve(root,'../browser-check')}`,'http://127.0.0.1:18979/comparison.html'],{stdio:['ignore','ignore','pipe']});
let socket;
try{
 const port=await new Promise((resolvePort,reject)=>{const timer=setTimeout(()=>reject(new Error('Browser startup timeout')),15000);browser.once('error',reject);browser.stderr.on('data',buffer=>{const match=buffer.toString().match(/DevTools listening on ws:\/\/127\.0\.0\.1:(\d+)/);if(match){clearTimeout(timer);resolvePort(match[1])}})});
 const page=(await(await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find(p=>p.type==='page');
 socket=new WebSocket(page.webSocketDebuggerUrl);
 await new Promise((r,j)=>{socket.addEventListener('open',r,{once:true});socket.addEventListener('error',j,{once:true})});
 let sequence=0;const pending=new Map();const errors=[];
 socket.addEventListener('message',event=>{const response=JSON.parse(event.data),entry=pending.get(response.id);if(entry){clearTimeout(entry.timer);pending.delete(response.id);response.error?entry.reject(new Error(JSON.stringify(response.error))):entry.resolve(response.result)}else if(response.method==='Runtime.exceptionThrown')errors.push(response.params)});
 const rpc=(method,params={})=>new Promise((resolveRpc,reject)=>{const id=++sequence,timer=setTimeout(()=>reject(new Error(method+' timeout')),25000);pending.set(id,{resolve:resolveRpc,reject,timer});socket.send(JSON.stringify({id,method,params}))});
 await rpc('Runtime.enable');
 const evaluate=async expression=>{const reply=await rpc('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(reply.exceptionDetails)throw new Error(reply.exceptionDetails.text);return reply.result.value};
 let count=0;for(let i=0;i<40&&count!==5;i++){count=await evaluate('document.querySelectorAll(".case").length');if(count!==5)await new Promise(r=>setTimeout(r,250))}
 if(count!==5)throw new Error('Expected five cards; got '+count);
 const rows=await evaluate('document.querySelectorAll("#rows > tr").length');if(rows!==5)throw new Error('Expected five metric rows');
 const assets=await evaluate('[...document.querySelectorAll("source[src],a[href],img[src]")].map(e=>e.getAttribute("src")||e.getAttribute("href")).filter(p=>p&&!p.startsWith("http")&&!p.startsWith("#"))');
 const broken=[];for(const asset of [...new Set(assets)]){const response=await fetch(`http://127.0.0.1:18979/${asset}`,{method:'HEAD'});if(!response.ok)broken.push({asset,status:response.status})}if(broken.length)throw new Error('Broken assets '+JSON.stringify(broken));
 const filtered=await evaluate(`(()=>{const s=document.querySelector('#filter');s.value='owl';s.dispatchEvent(new Event('input'));return document.querySelectorAll('.case').length})()`);if(filtered!==1)throw new Error('Filter failed');
 await evaluate(`(()=>{const s=document.querySelector('#filter');s.value='';s.dispatchEvent(new Event('input'));document.querySelector('.play').click()})()`);
 await new Promise(r=>setTimeout(r,10000));
 const videos=await evaluate('[...document.querySelector(".case").querySelectorAll("video")].map(v=>({time:v.currentTime,paused:v.paused,ready:v.readyState,error:v.error?.message}))');
 if(videos.length!==3||videos.some(v=>v.paused||v.time<.25||v.error))throw new Error('Playback failed '+JSON.stringify(videos)+' '+JSON.stringify(await evaluate('[...document.querySelector(".case").querySelectorAll("video")].map(v=>({src:v.currentSrc,network:v.networkState,canplay:v.canPlayType(\'video/mp4; codecs="avc1.64001f"\')}))')));
 const drift=Math.max(...videos.map(v=>v.time))-Math.min(...videos.map(v=>v.time));if(drift>.2)throw new Error('Playback drift '+drift);
 await evaluate('document.querySelectorAll("video").forEach(v=>v.pause())');
 await rpc('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
 let shot=await rpc('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await writeFile(resolve(root,'../desktop.png'),Buffer.from(shot.data,'base64'));
 await rpc('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 const mobile=await evaluate('({viewport:innerWidth,document_width:document.documentElement.scrollWidth})');if(mobile.document_width>391||mobile.viewport>391)throw new Error('Mobile overflow '+JSON.stringify(mobile));
 shot=await rpc('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await writeFile(resolve(root,'../mobile.png'),Buffer.from(shot.data,'base64'));
 if(errors.length)throw new Error('JS errors '+JSON.stringify(errors));
 const result={cases:count,metric_rows:rows,relative_assets:[...new Set(assets)].length,missing_assets:broken,filter:'PASS',paired_playback:'PASS',drift_seconds:drift,mobile,js_errors:errors};
 await writeFile(resolve(root,'browser-validation.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}finally{if(socket)socket.close();browser.kill('SIGTERM');server.kill('SIGTERM')}
