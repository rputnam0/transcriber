'use strict';
const $ = id => document.getElementById(id);
const audio = $('audio');
let data, review, selected = 0, saveQueue = Promise.resolve(), revision = 0, lastActive = '';
const palette = ['#678877','#ba8a65','#859dc2','#b095b5','#b7a350','#769b9b'];
const time = s => `${Math.floor(s/60).toString().padStart(2,'0')}:${(s%60).toFixed(1).padStart(4,'0')}`;
const el = (tag, text, className) => {const node=document.createElement(tag); if(text!==undefined)node.textContent=text; if(className)node.className=className; return node;};
const item = i => review.reviews[i] || {verdict:'ungraded',speaker_slot:null,text_error:false,note:''};
const voice = i => item(i).speaker_slot == null ? data.segments[i].speaker : review.roster[item(i).speaker_slot];
function save() {
  review.updated_at = new Date().toISOString();
  const snapshot=JSON.stringify(review), current=++revision;
  try {localStorage.setItem(`speaker-review:${review.transcript_sha256}`,snapshot);} catch (_) {}
  $('saveStatus').textContent='Saving…';
  saveQueue=saveQueue.catch(()=>{}).then(async()=>{
    const response=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':data.token},body:snapshot});
    if(!response.ok) throw new Error((await response.json()).error || 'Save failed');
    if(current===revision)$('saveStatus').textContent='Saved on this Mac';
  }).catch(error=>{ $('saveStatus').textContent=`Not saved to disk — ${error.message}. Export a backup.`; });
}
function change(i, patch, rerender=true) {
  review.reviews[i]={...item(i),...patch,updated_at:new Date().toISOString()};
  save(); if(rerender)render();
}
function grade(i, verdict) {
  const patch={verdict};
  change(i,patch);
}
function assign(i, slot) {
  const verdict=slot==null?'ungraded':review.roster[slot]===data.segments[i].speaker?'correct':'wrong';
  change(i,{speaker_slot:slot,verdict});
}
function visible(i) {
  const s=data.segments[i], r=item(i), filter=$('filter').value, query=$('search').value.toLowerCase();
  return (!query || `${s.text} ${s.speaker} ${voice(i)}`.toLowerCase().includes(query)) &&
    (filter==='all' || filter==='flagged' && s.review_required || r.verdict===filter);
}
function select(i, scroll=false) {
  selected=i;
  document.querySelectorAll('.turn.selected').forEach(n=>n.classList.remove('selected'));
  const row=$(`turn-${i}`); if(row){row.classList.add('selected');if(scroll)row.scrollIntoView({block:'center',behavior:'smooth'});}
}
function seek(i) {
  select(i);audio.currentTime=Math.max(0,data.segments[i].start-1);
  audio.play().catch(()=>{$('saveStatus').textContent='Press play to start audio';});
}
function next() {
  const candidates=data.segments.map((_,i)=>i).filter(i=>item(i).verdict==='ungraded' && visible(i));
  const target=candidates.find(i=>i>selected) ?? candidates[0];
  if(target!==undefined){select(target,true);seek(target);} else $('saveStatus').textContent='No ungraded turns in this view';
}
function render() {
  const fragment=document.createDocumentFragment();
  const speakers=[...new Set(data.segments.map(s=>s.speaker))].sort();
  let count=0;
  data.segments.forEach((s,i)=>{
    if(!visible(i))return;count++;
    const r=item(i), row=el('article',undefined,`turn${i===selected?' selected':''}`);row.id=`turn-${i}`;
    row.style.setProperty('--voice',palette[speakers.indexOf(s.speaker)%palette.length]);
    const stamp=el('button',time(s.start),'time');stamp.title=`Play turn ${i+1} with context`;stamp.append(el('small',`to ${time(s.end)}`));stamp.onclick=()=>seek(i);row.append(stamp);
    const body=el('div'), line=el('div',undefined,'speaker-line');line.append(el('span',voice(i),'speaker'));
    if(s.review_required){const badge=el('span','Review suggested','badge');badge.title=s.review_reasons.join(' · ');line.append(badge);}
    if(r.verdict!=='ungraded')line.append(el('span',r.verdict==='correct'?'✓ Correct':r.verdict==='wrong'?'↳ Speaker wrong':'? Unsure','verdict'));
    body.append(line,el('p',s.text,'words'));
    const controls=el('div',undefined,'controls');
    for(const [v,label] of [['correct','✓ Correct'],['wrong','✕ Wrong'],['unsure','? Unsure']]){
      const b=el('button',label,r.verdict===v?'chosen':'');b.setAttribute('aria-pressed',String(r.verdict===v));b.onclick=()=>{select(i);grade(i,v);};controls.append(b);
    }
    const choose=el('select');choose.setAttribute('aria-label',`Correct speaker for turn ${i+1}`);
    const placeholder=el('option','Assign one of 4 voices…');placeholder.value='';choose.append(placeholder);
    review.roster.forEach((name,slot)=>{const option=el('option',`${slot+1} · ${name}`);option.value=slot;choose.append(option);});
    choose.value=r.speaker_slot??'';choose.onchange=()=>{select(i);assign(i,choose.value===''?null:Number(choose.value));};controls.append(choose);
    const reset=el('button','Undo grade');reset.onclick=()=>change(i,{verdict:'ungraded',speaker_slot:null});controls.append(reset);
    const textLabel=el('label',undefined,'text-error'), checkbox=el('input');checkbox.type='checkbox';checkbox.checked=!!r.text_error;checkbox.onchange=()=>change(i,{text_error:checkbox.checked},false);textLabel.append(checkbox,document.createTextNode(' Words / timing wrong'));controls.append(textLabel);body.append(controls);
    if(r.speaker_slot!=null)body.append(el('div',`Original prediction: ${s.speaker}`,'original'));
    const note=el('input');note.className='notes';note.placeholder='Optional note: missing speech, overlap, or timing…';note.value=r.note||'';note.setAttribute('aria-label',`Note for turn ${i+1}`);note.onchange=()=>change(i,{note:note.value},false);body.append(note);
    row.onclick=()=>select(i);row.append(body);fragment.append(row);
  });
  $('turns').replaceChildren(fragment);$('empty').hidden=!!count;
  const values=Object.values(review.reviews), correct=values.filter(r=>r.verdict==='correct').length, wrong=values.filter(r=>r.verdict==='wrong').length, unsure=values.filter(r=>r.verdict==='unsure').length;
  const accuracy=correct+wrong?` · ${Math.round(100*correct/(correct+wrong))}% correct among ${correct+wrong} judged turns`:'';
  $('progress').textContent=`${correct+wrong+unsure} / ${data.segments.length} graded · ${unsure} unsure${accuracy}`;
  lastActive='';updateActive();
}
function updateActive() {
  if(!data)return;
  const active=data.segments.map((s,i)=>s.start<=audio.currentTime && audio.currentTime<s.end?i:-1).filter(i=>i>=0);
  const key=active.join(',');if(key===lastActive)return;lastActive=key;
  document.querySelectorAll('.turn.active').forEach(n=>n.classList.remove('active'));
  active.forEach(i=>$(`turn-${i}`)?.classList.add('active'));
  $('now').textContent=active.length?[...new Set(active.map(voice))].join(' + '):'Between turns';
  if(active.length && !audio.paused && $('follow').checked){
    const nextSelected=active.includes(selected)?selected:active[0];
    const changed=nextSelected!==selected;
    select(nextSelected,changed);
  }
}
$('audio').addEventListener('timeupdate',updateActive);
$('audio').addEventListener('error',()=>{$('saveStatus').textContent='Audio could not load. Check the local server.';});
$('playPause').onclick=()=>audio.paused?audio.play().catch(()=>{}):audio.pause();
audio.addEventListener('play',()=>{$('playPause').textContent='Pause';});
audio.addEventListener('pause',()=>{$('playPause').textContent='Play';});
$('speed').onchange=()=>audio.playbackRate=Number($('speed').value);
$('back').onclick=()=>audio.currentTime=Math.max(0,audio.currentTime-5);
$('replay').onclick=()=>seek(selected);
$('next').onclick=next;
$('search').oninput=render;$('filter').onchange=render;
$('export').onclick=()=>{
  const output={...review,title:data.title,original_predictions_preserved:true,segments:data.segments.map((s,i)=>({...s,turn_id:i,review:item(i),corrected_speaker:item(i).speaker_slot==null?null:voice(i)}))};
  const url=URL.createObjectURL(new Blob([JSON.stringify(output,null,2)],{type:'application/json'}));
  const a=el('a');a.href=url;a.download='session1-speaker-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
document.addEventListener('keydown',event=>{
  if(!data || event.ctrlKey || event.metaKey || event.altKey || /INPUT|SELECT|TEXTAREA|BUTTON/.test(event.target.tagName))return;
  const key=event.key.toLowerCase();
  if(key===' '){event.preventDefault();audio.paused?audio.play().catch(()=>{}):audio.pause();}
  else if(key==='c')grade(selected,'correct');else if(key==='w')grade(selected,'wrong');else if(key==='u')grade(selected,'unsure');else if(key==='n')next();else if(/^[1-4]$/.test(key))assign(selected,Number(key)-1);
});
(async()=>{
  try{
    const response=await fetch('/api/data');if(!response.ok)throw new Error('Could not load transcript');
    data=await response.json();review=data.review;
    let recovered=false;
    try{const cached=JSON.parse(localStorage.getItem(`speaker-review:${review.transcript_sha256}`));if(cached && cached.transcript_sha256===review.transcript_sha256 && (cached.updated_at||'')>(review.updated_at||'')){review=cached;recovered=true;}}catch(_){}
    review.roster.forEach((name,slot)=>{
      const label=el('label',slot===0?'1 · Dungeon master':`${slot+1} · Player ${slot}`), input=el('input');input.value=name;
      input.onchange=()=>{const value=input.value.trim();if(!value || review.roster.some((s,i)=>i!==slot && s===value)){input.value=review.roster[slot];$('saveStatus').textContent='Each voice needs a distinct name';return;}review.roster[slot]=value;save();render();};label.append(input);$('roster').append(label);
    });
    render();$('saveStatus').textContent='Ready · grades save automatically on this Mac';if(recovered)save();
  }catch(error){$('progress').textContent=error.message;$('saveStatus').textContent='Could not open review';}
})();
