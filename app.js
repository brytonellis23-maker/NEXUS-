const $=id=>document.getElementById(id);
let model=null,aiRunning=false,cameraStream=null,motors=false,nanoBase="",mode="idle",lastSeen=false,lastSpoken="",lastSpeech=0,lastAdjust=0;
let vlReady=false;

/* A browser-safe vision-language ranking layer. It ranks detector crops against
   the user's natural-language description using CLIP in Transformers.js.
   If CLIP cannot load, the app falls back to the detector/category match. */
let clipPipeline=null;
async function loadVisionLanguage(){
  if(clipPipeline)return true;
  $("aiStatus").textContent="AI: LOADING VISION + LANGUAGE";
  try{
    const mod=await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm");
    clipPipeline=await mod.pipeline("zero-shot-image-classification","Xenova/clip-vit-base-patch32");
    vlReady=true; return true;
  }catch(e){
    console.warn("Vision-language model unavailable; using detector fallback.",e);
    vlReady=false; return false;
  }
}
function speak(msg,force=false){
 if(!$("voiceToggle").checked||!("speechSynthesis"in window))return;
 const now=Date.now();if(!force&&msg===lastSpoken&&now-lastSpeech<3500)return;
 lastSpoken=msg;lastSpeech=now;speechSynthesis.cancel();
 const u=new SpeechSynthesisUtterance(msg);u.volume=+$("voiceVolume").value;u.rate=+$("voiceRate").value;speechSynthesis.speak(u);
}
function state(s){
 const labels={idle:"● IDLE",scanning:"● SCANNING",found:"● TARGET FOUND",locked:"● TARGET LOCKED",tracking:"● TRACKING",lost:"● TARGET LOST"};
 $("trackState").textContent=labels[s]||s;if(s===mode)return;mode=s;
 const t=$("targetText").value||"target";
 const w={scanning:`Scanning for ${t}.`,found:"Target found.",locked:"Target locked and centered.",tracking:"Tracking target.",lost:"Target lost. Searching.",idle:"Manual control."};
 if(w[s])speak(w[s]);
}
function target(){return $("targetText").value.trim()||"target"}
async function startCamera(){try{cameraStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:"environment"},audio:false});video.srcObject=cameraStream;$("status").textContent="● WEBCAM READY";speak("NEXUS CAM online. Webcam ready.",true)}catch(e){alert("Webcam permission failed: "+e.message)}}
async function startAI(){
 if(!video.srcObject)await startCamera();
 $("aiStatus").textContent="AI: LOADING";
 try{
  model=await cocoSsd.load();
  await loadVisionLanguage();
  aiRunning=true;$("aiStatus").textContent=vlReady?"AI: VISION + LANGUAGE":"AI: OBJECT DETECTION";
  speak(vlReady?"Vision and language AI ready.":"Object detection AI ready.",true);detectLoop();
 }catch(e){$("aiStatus").textContent="AI: ERROR";console.error(e)}
}
function stopAI(){aiRunning=false;ctx.clearRect(0,0,canvas.width,canvas.height);$("aiStatus").textContent="AI: OFF";state("idle")}
async function send(cmd){if(cmd==="estop")motors=false;if(!nanoBase)return;try{await fetch(nanoBase+"/api/command",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({command:cmd,panSpeed:+$("panSpeed").value,tiltSpeed:+$("tiltSpeed").value,motors})})}catch(e){$("connText").textContent="Command failed"}}

function detectorCandidates(preds){
 // Detector supplies spatial proposals. We keep several likely regions so the
 // language model can choose among them instead of blindly choosing the first.
 return preds.filter(p=>p.score>=+$("confidence").value).slice(0,8);
}
async function rankCandidates(cands){
 if(!vlReady||!cands.length)return cands.map((p,i)=>({p,score:p.score}));
 const wanted=target();
 // Crop each proposal from the current frame. Use canvas as an image source.
 const tmp=document.createElement("canvas"),tctx=tmp.getContext("2d");
 const labels=[
  wanted,
  `a photo of ${wanted}`,
  `an object that is ${wanted}`,
  "an unrelated object"
 ];
 const out=[];
 for(const p of cands){
   const [x,y,w,h]=p.bbox; tmp.width=Math.max(32,Math.round(w));tmp.height=Math.max(32,Math.round(h));
   tctx.drawImage(video,x,y,w,h,0,0,tmp.width,tmp.height);
   try{
     const r=await clipPipeline(tmp,labels);
     const best=r.find(v=>v.label!==labels[3]);
     const unrelated=r.find(v=>v.label===labels[3])?.score||0;
     out.push({p,score:(best?.score||0)-unrelated});
   }catch(e){out.push({p,score:p.score})}
 }
 return out.sort((a,b)=>b.score-a.score);
}
async function detectLoop(){
 if(!aiRunning||!model||video.readyState<2){if(aiRunning)requestAnimationFrame(detectLoop);return}
 canvas.width=video.videoWidth||640;canvas.height=video.videoHeight||480;
 const preds=await model.detect(video,20,+$("confidence").value);
 const candidates=detectorCandidates(preds);
 const ranked=(mode!=="idle"&&candidates.length)?await rankCandidates(candidates):candidates.map(p=>({p,score:p.score}));
 ctx.clearRect(0,0,canvas.width,canvas.height);ctx.lineWidth=3;ctx.font="16px Arial";
 let best=null;
 if(mode!=="idle"&&ranked.length)best=ranked[0];
 for(const p of preds){
   const [x,y,w,h]=p.bbox, hit=best&&p===best.p;
   ctx.strokeStyle=hit?"#19e879":"#1594ff";ctx.strokeRect(x,y,w,h);
   const lab=`${p.class} ${(p.score*100).toFixed(0)}%`;
   ctx.fillStyle="rgba(0,0,0,.75)";ctx.fillRect(x,y-24,ctx.measureText(lab).width+12,24);
   ctx.fillStyle="#fff";ctx.fillText(lab,x+6,y-7);
 }
 $("detections").textContent=`${preds.length} objects`;
 if(best){
   const p=best.p,[x,y,w,h]=p.bbox,cx=x+w/2,cy=y+h/2,dx=cx-canvas.width/2,dy=cy-canvas.height/2;
   $("matchScore").textContent=(Math.max(0,best.score)*100).toFixed(0)+"%";
   $("targetReadout").textContent=`Target: ${target()} • ${p.class} • X ${Math.round(dx)} • Y ${Math.round(dy)}`;
   if(!lastSeen){lastSeen=true;state("found")} if(mode==="scanning"||mode==="found")state("locked");
   if(mode==="tracking"){
     const now=Date.now();
     if(now-lastAdjust>220){
       if(Math.abs(dx)>canvas.width*.08)send(dx>0?"pan_right":"pan_left");
       else if(Math.abs(dy)>canvas.height*.08)send(dy>0?"tilt_down":"tilt_up");
       else{send("stop");state("locked")}
       lastAdjust=now;
     }
   }
 }else if(mode!=="idle"){
   $("matchScore").textContent="—";
   if(lastSeen){lastSeen=false;state("lost")}else state("scanning");
   send("stop");
 }
 if(aiRunning)requestAnimationFrame(detectLoop)
}

$("camera").onclick=startCamera;$("startAI").onclick=startAI;$("stopAI").onclick=stopAI;
$("scanTarget").onclick=async()=>{if(!aiRunning)await startAI();lastSeen=false;state("scanning")};
$("trackTarget").onclick=async()=>{if(!aiRunning)await startAI();state("tracking")};
$("unlockTarget").onclick=()=>{state("idle");lastSeen=false;send("stop")};
$("targetText").oninput=()=>{$("targetReadout").textContent="Target: "+target()};
$("testVoice").onclick=()=>speak("NEXUS CAM voice system is working.",true);
$("voiceCommand").onclick=()=>{
 const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
 if(!SR){speak("Voice commands are not supported in this browser.",true);return}
 speak("Listening.",true);const r=new SR();r.lang="en-US";r.interimResults=false;
 r.onresult=e=>{const t=e.results[0][0].transcript.toLowerCase();const m=t.match(/(?:lock onto|track|find|scan for)\s+(.+)/);
 if(m){$("targetText").value=m[1].trim();$("targetReadout").textContent="Target: "+m[1].trim();speak("Understood. Starting scan for "+m[1].trim()+".",true);$("scanTarget").click()}
 else speak("I did not understand the target.",true)};r.start();
};
$("connect").onclick=()=>{let v=$("nanoIP").value.trim();if(v&&!/^https?:\/\//.test(v))v="http://"+v;nanoBase=v.replace(/\/$/,"");$("connText").textContent=nanoBase?"Configured: "+nanoBase:"Not connected";$("status").textContent="● NANO READY";speak("Nano connection configured.",true)};
$("enable").onclick=()=>{motors=true;send("enable");speak("Motors enabled.",true)};
$("estop").onclick=()=>{motors=false;send("estop");speak("Emergency stop activated.",true);state("idle")};
$("fullscreen").onclick=()=>document.documentElement.requestFullscreen?.();
document.querySelectorAll("[data-cmd]").forEach(b=>{b.onpointerdown=()=>send(b.dataset.cmd);b.onpointerup=()=>send("stop");b.onpointerleave=()=>send("stop")});
if("serviceWorker"in navigator)navigator.serviceWorker.register("./sw.js").catch(()=>{});
