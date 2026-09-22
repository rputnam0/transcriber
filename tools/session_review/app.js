'use strict';
const $ = id => document.getElementById(id);
const audio = $('audio');
let data, review, selected = 0, saveQueue = Promise.resolve(), revision = 0, lastActive = '';
let page=0, visibleTurns=[], pending=0, saveFailed=false, serverRevision=0;
let clip=null;
const phraseMode=()=>$('needsReview').checked;
const pageSize=80;
let sessionId=new URLSearchParams(location.search).get('session');
const endpoint=path=>`${path}?session=${encodeURIComponent(sessionId)}`;
const cacheKey=()=>`session-review:${sessionId}:${review.transcript_sha256}`;
const viewKey=()=>`session-review-view:${sessionId}`;
function saveView() {
  try {localStorage.setItem(viewKey(),JSON.stringify({needsReview:$('needsReview').checked,filter:$('filter').value,speaker:$('speakerFilter').value}));} catch (_) {}
}
function restoreView() {
  try {
    const view=JSON.parse(localStorage.getItem(viewKey()));
    if(!view)return;
    $('needsReview').checked=view.needsReview===true;
    if(['all','ungraded','wrong','unsure'].includes(view.filter))$('filter').value=view.filter;
    if([...$('speakerFilter').options].some(option=>option.value===view.speaker))$('speakerFilter').value=view.speaker;
  } catch (_) {}
}
const palette = ['#678877','#ba8a65','#859dc2','#b095b5','#b7a350','#769b9b'];
const time = s => `${Math.floor(s/3600).toString().padStart(2,'0')}:${Math.floor(s%3600/60).toString().padStart(2,'0')}:${(s%60).toFixed(1).padStart(4,'0')}`;
const el = (tag, text, className) => {const node=document.createElement(tag); if(text!==undefined)node.textContent=text; if(className)node.className=className; return node;};
const item = i => review.reviews[i] || {verdict:'ungraded',speaker_slot:null,text_error:false,note:''};
const voice = i => {
  const slot=item(i).speaker_slot ?? data.roster.indexOf(data.segments[i].speaker);
  return slot<0 ? data.segments[i].speaker : review.roster[slot];
};
const speakerKey = i => {
  const slot=item(i).speaker_slot ?? data.roster.indexOf(data.segments[i].speaker);
  return slot<0 ? `name:${data.segments[i].speaker}` : `slot:${slot}`;
};
function renderSpeakerFilter() {
  const select=$('speakerFilter'), previous=select.value;
  const choices=[['','All speakers'],...review.roster.map((name,slot)=>[`slot:${slot}`,name])];
  [...new Set(data.segments.map(s=>s.speaker))].filter(name=>!data.roster.includes(name)).sort()
    .forEach(name=>choices.push([`name:${name}`,name]));
  select.replaceChildren(...choices.map(([value,label])=>{const option=el('option',label);option.value=value;return option;}));
  if(choices.some(([value])=>value===previous))select.value=previous;
}
function save() {
  review.updated_at = new Date().toISOString();
  const snapshot=JSON.stringify(review), current=++revision;
  try {localStorage.setItem(cacheKey(),snapshot);} catch (_) {}
  pending++; $('saveStatus').textContent='Saving…';
  saveQueue=saveQueue.catch(()=>{}).then(async()=>{
    const body=JSON.parse(snapshot);body.revision=serverRevision;
    const response=await fetch(endpoint('/api/review'),{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':data.token},body:JSON.stringify(body)});
    if(!response.ok) throw new Error((await response.json()).error || 'Save failed');
    serverRevision=(await response.json()).revision;review.revision=serverRevision;saveFailed=false;
    if(current===revision){$('saveStatus').textContent='Saved on this Mac';try{localStorage.setItem(cacheKey(),JSON.stringify(review));}catch(_){}}
  }).catch(error=>{saveFailed=true;$('saveStatus').textContent=`Not saved — ${error.message} Export a backup.`;}).finally(()=>pending--);
}
function change(i, patch, rerender=true) {
  review.reviews[i]={...item(i),...patch,updated_at:new Date().toISOString()};
  save(); if(rerender)render();
}
function grade(i, verdict) {
  const patch=verdict==='correct'?{verdict,speaker_slot:null}:{verdict};
  change(i,patch);
  if(phraseMode())next(i);
}
function assign(i, slot) {
  const verdict=slot==null?'ungraded':data.roster[slot]===data.segments[i].speaker?'correct':'wrong';
  change(i,{speaker_slot:slot,verdict});
  if(phraseMode() && slot!==null)next(i);
}
function visible(i) {
  const s=data.segments[i], r=item(i), filter=$('filter').value, query=$('search').value.toLowerCase();
  return (!query || `${s.text} ${s.speaker} ${voice(i)}`.toLowerCase().includes(query)) &&
    (!$('speakerFilter').value || speakerKey(i)===$('speakerFilter').value) &&
    (!$('needsReview').checked || s.review_required || s.roster_review_required || s.decode_review_required) &&
    (filter==='all' || r.verdict===filter);
}
function select(i, scroll=false) {
  selected=i;
  const position=visibleTurns.indexOf(i);
  if(position>=0 && Math.floor(position/pageSize)!==page){page=Math.floor(position/pageSize);render();}
  document.querySelectorAll('.turn.selected').forEach(n=>n.classList.remove('selected'));
  const row=$(`turn-${i}`); if(row){row.classList.add('selected');if(scroll)row.scrollIntoView({block:'center',behavior:'smooth'});}
}
function seek(i) {
  if(!data || i==null)return;
  clip=phraseMode()?{index:i,start:Math.max(0,data.segments[i].start-1),end:Math.min(data.duration,data.segments[i].end+0.25),finished:false}:null;
  select(i,true);audio.currentTime=Math.max(0,data.segments[i].start-1);lastActive='';updateActive();
  $('clipStatus').textContent=clip?`Reviewing ${voice(i)} at ${time(data.segments[i].start)} · grade or choose a speaker to jump ahead.`:'';
  audio.play().catch(()=>{$('saveStatus').textContent='Press play to start audio';});
}
function next(after=selected) {
  const candidates=data.segments.map((_,i)=>i).filter(i=>item(i).verdict==='ungraded' && visible(i));
  const target=candidates.find(i=>i>after) ?? candidates[0];
  if(target!==undefined){seek(target);} else {audio.pause();clip=null;$('clipStatus').textContent='No ungraded turns in this view.';}
}
function playPause() {
  if(!data)return;
  if(!audio.paused){audio.pause();return;}
  if(phraseMode() && (!clip || clip.finished || !visible(clip.index))){
    const target=visible(selected)?selected:visibleTurns.find(i=>i>selected)??visibleTurns[0];
    if(target!==undefined)seek(target);
    return;
  }
  audio.play().catch(()=>{});
}
function render() {
  renderSpeakerFilter();
  const fragment=document.createDocumentFragment();
  const speakers=[...new Set(data.segments.map(s=>s.speaker))].sort();
  let count=0;
  visibleTurns=data.segments.map((_,i)=>i).filter(visible);
  page=Math.max(0,Math.min(page,Math.ceil(visibleTurns.length/pageSize)-1));
  visibleTurns.slice(page*pageSize,(page+1)*pageSize).forEach(i=>{
    const s=data.segments[i];count++;
    const r=item(i), row=el('article',undefined,`turn${i===selected?' selected':''}`);row.id=`turn-${i}`;
    row.style.setProperty('--voice',palette[speakers.indexOf(s.speaker)%palette.length]);
    const stamp=el('button',time(s.start),'time');stamp.title=`Play turn ${i+1} with context`;stamp.append(el('small',`to ${time(s.end)}`));stamp.onclick=()=>seek(i);row.append(stamp);
    const body=el('div'), line=el('div',undefined,'speaker-line');line.append(el('span',voice(i),'speaker'));
    if(s.review_required || s.roster_review_required || s.decode_review_required){const badge=el('span','Review suggested','badge');badge.title=[...(s.review_reasons||[]),s.roster_review_required?s.roster_reason:'',s.decode_review_required?'Check decoded passage':''].filter(Boolean).join(' · ');line.append(badge);}
    if(r.verdict!=='ungraded')line.append(el('span',r.verdict==='correct'?'✓ Correct':r.verdict==='wrong'?'↳ Speaker wrong':'? Unsure','verdict'));
    body.append(line,el('p',s.text,'words'));
    const controls=el('div',undefined,'controls');
    for(const [v,label] of [['correct','✓ Correct'],['wrong','✕ Wrong'],['unsure','? Unsure']]){
      const b=el('button',label,r.verdict===v?'chosen':'');b.setAttribute('aria-pressed',String(r.verdict===v));b.onclick=()=>{select(i);grade(i,v);};controls.append(b);
    }
    const choose=el('select');choose.setAttribute('aria-label',`Correct speaker for turn ${i+1}`);
    const placeholder=el('option',`Assign one of ${review.roster.length} voices…`);placeholder.value='';choose.append(placeholder);
    review.roster.forEach((name,slot)=>{const option=el('option',`${slot+1} · ${name}`);option.value=slot;choose.append(option);});
    choose.value=r.speaker_slot??'';choose.onchange=()=>{select(i);assign(i,choose.value===''?null:Number(choose.value));};controls.append(choose);
    const reset=el('button','Undo grade');reset.onclick=()=>change(i,{verdict:'ungraded',speaker_slot:null});controls.append(reset);
    const textLabel=el('label',undefined,'text-error'), checkbox=el('input');checkbox.type='checkbox';checkbox.checked=!!r.text_error;checkbox.onchange=()=>change(i,{text_error:checkbox.checked},false);textLabel.append(checkbox,document.createTextNode(' Words / timing wrong'));controls.append(textLabel);body.append(controls);
    if(r.speaker_slot!=null)body.append(el('div',`Original prediction: ${s.speaker}`,'original'));
    const note=el('input');note.className='notes';note.placeholder='Optional note: missing speech, overlap, or timing…';note.value=r.note||'';note.maxLength=5000;note.setAttribute('aria-label',`Note for turn ${i+1}`);note.oninput=()=>change(i,{note:note.value},false);body.append(note);
    row.onclick=event=>{if(!event.target.closest('button,select,input,label'))select(i);};row.append(body);fragment.append(row);
  });
  $('turns').replaceChildren(fragment);$('empty').hidden=!!count;
  const values=Object.values(review.reviews), correct=values.filter(r=>r.verdict==='correct').length, wrong=values.filter(r=>r.verdict==='wrong').length, unsure=values.filter(r=>r.verdict==='unsure').length;
  const accuracy=correct+wrong?` · ${Math.round(100*correct/(correct+wrong))}% correct among ${correct+wrong} judged turns`:'';
  $('progress').textContent=`${correct+wrong+unsure} / ${data.segments.length} graded · ${unsure} unsure${accuracy}`;
  $('pageInfo').textContent=visibleTurns.length?`${page*pageSize+1}–${Math.min((page+1)*pageSize,visibleTurns.length)} of ${visibleTurns.length} matching turns`: 'No matching turns';
  $('pagePrev').disabled=page===0;$('pageNext').disabled=(page+1)*pageSize>=visibleTurns.length;
  lastActive='';
}
function updateActive() {
  if(!data)return;
  if(clip && !clip.finished && audio.currentTime>=clip.end){
    clip.finished=true;audio.pause();
    $('clipStatus').textContent=`Phrase finished: ${voice(clip.index)} · grade it, choose a speaker, or press Next ungraded.`;
  }
  const active=data.segments.map((s,i)=>s.start<=audio.currentTime && audio.currentTime<s.end?i:-1).filter(i=>i>=0);
  const key=active.join(',');if(key===lastActive && $('now').textContent!=='Ready to listen')return;lastActive=key;
  document.querySelectorAll('.turn.active').forEach(n=>n.classList.remove('active'));
  active.forEach(i=>$(`turn-${i}`)?.classList.add('active'));
  $('now').textContent=active.length?[...new Set(active.map(voice))].join(' + '):'Between turns';
  if(active.length && !clip && !audio.paused && $('follow').checked){
    const nextSelected=active.includes(selected)?selected:active[0];
    const changed=nextSelected!==selected;
    select(nextSelected,changed);
  }
}
$('audio').addEventListener('timeupdate',updateActive);
$('audio').addEventListener('error',()=>{$('saveStatus').textContent='Audio could not load. Check the local server.';});
$('playPause').onclick=playPause;
audio.addEventListener('play',()=>{
  $('playPause').textContent='Pause';
  // Native audio controls must use the same bounded phrase playback.
  if(phraseMode() && data && (!clip || clip.finished)){
    const target=visible(selected)?selected:visibleTurns.find(i=>i>selected)??visibleTurns[0];
    if(target!==undefined)seek(target);else audio.pause();
  }
});
audio.addEventListener('pause',()=>{$('playPause').textContent='Play';});
$('speed').onchange=()=>audio.playbackRate=Number($('speed').value);
$('back').onclick=()=>{audio.currentTime=Math.max(clip?.start??0,audio.currentTime-5);if(clip)clip.finished=false;};
$('replay').onclick=()=>seek(selected);
$('next').onclick=()=>next();
$('previous').onclick=()=>{const target=visibleTurns.filter(i=>i<selected).at(-1);if(target!==undefined)seek(target);};
$('search').oninput=()=>{page=0;render();};
$('filter').onchange=$('needsReview').onchange=$('speakerFilter').onchange=event=>{
  if(event.target.id==='speakerFilter' && $('speakerFilter').value)$('needsReview').checked=true;
  audio.pause();clip=null;saveView();page=0;if(data)render();
  $('clipStatus').textContent=phraseMode()?'Phrase review: play → grade → next. Unflagged stretches are skipped.':'';
};
$('pagePrev').onclick=()=>{page--;render();};$('pageNext').onclick=()=>{page++;render();};
$('retrySave').onclick=()=>data&&save();
$('export').onclick=()=>{
  const output={...review,title:data.title,original_predictions_preserved:true,segments:data.segments.map((s,i)=>({...s,turn_id:i,review:item(i),corrected_speaker:item(i).speaker_slot==null?null:voice(i)}))};
  const url=URL.createObjectURL(new Blob([JSON.stringify(output,null,2)],{type:'application/json'}));
  const a=el('a');a.href=url;a.download=`session${sessionId}-speaker-review.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
document.addEventListener('keydown',event=>{
  if(!data || event.ctrlKey || event.metaKey || event.altKey || /INPUT|SELECT|TEXTAREA|BUTTON/.test(event.target.tagName))return;
  const key=event.key.toLowerCase();
  if(key===' '){event.preventDefault();playPause();}
  else if(key==='c')grade(selected,'correct');else if(key==='w')grade(selected,'wrong');else if(key==='u')grade(selected,'unsure');else if(key==='n')next();else if(/^[1-6]$/.test(key) && Number(key)<=review.roster.length)assign(selected,Number(key)-1);
});
$('jumpForm').onsubmit=event=>{
  event.preventDefault();const value=$('jumpTime').value.trim();
  if(!/^\d+(?::[0-5]?\d){0,2}(?:\.\d+)?$/.test(value)){$('jumpStatus').textContent='Use seconds, MM:SS, or HH:MM:SS';return;}
  const seconds=value.split(':').reduce((a,b)=>a*60+Number(b),0);
  if(seconds<0 || seconds>=data.duration){$('jumpStatus').textContent='That time is outside this recording';return;}
  audio.pause();clip=null;audio.currentTime=seconds;lastActive='';updateActive();
  const i=data.segments.findIndex(s=>s.end>seconds);if(i>=0)select(i,true);
  $('jumpStatus').textContent=`At ${time(seconds)} · press Play to listen`;
};
window.addEventListener('beforeunload',e=>{if(pending||saveFailed){e.preventDefault();e.returnValue='';}});
$('sessionPicker').onchange=async()=>{
  audio.pause();await saveQueue;
  if(saveFailed){$('sessionPicker').value=sessionId;return;}
  location.href=`/?session=${encodeURIComponent($('sessionPicker').value)}`;
};
(async()=>{
  try{
    const sessions=await (await fetch('/api/sessions')).json();
    sessions.forEach(s=>{const option=el('option',`${s.title} · ${time(s.duration).split('.')[0]}`);option.value=s.id;$('sessionPicker').append(option);});
    if(!sessions.some(s=>s.id===sessionId))sessionId=sessions[0]?.id;
    if(!sessionId)throw new Error('No completed sessions are available yet');
    $('sessionPicker').value=sessionId;
    history.replaceState(null,'',`/?session=${encodeURIComponent(sessionId)}`);
    const response=await fetch(endpoint('/api/data'));if(!response.ok)throw new Error('Could not load transcript');
    data=await response.json();review=data.review;serverRevision=review.revision;
    document.title=`${data.title} · Speaker review`;$('sessionTitle').textContent=data.title;
    $('subtitle').textContent=`${data.title} / ${time(data.duration).split('.')[0]} / ${review.roster.length} voices`;
    $('sourceNote').textContent=data.note;
    audio.src=endpoint('/audio');
    let recovered=false;
    try{const cached=JSON.parse(localStorage.getItem(cacheKey()));if(cached && cached.session===sessionId && cached.transcript_sha256===review.transcript_sha256 && cached.revision>=review.revision && (cached.updated_at||'')>(review.updated_at||'')){review=cached;recovered=true;}}catch(_){}
    review.roster.forEach((name,slot)=>{
      const label=el('label',`${slot+1} · ${data.roster[slot]}`), input=el('input');input.value=name;input.maxLength=100;
      input.onchange=()=>{const value=input.value.trim();if(!value || review.roster.some((s,i)=>i!==slot && s===value)){input.value=review.roster[slot];$('saveStatus').textContent='Each voice needs a distinct name';return;}review.roster[slot]=value;save();render();};label.append(input);$('roster').append(label);
    });
    renderSpeakerFilter();restoreView();render();$('clipStatus').textContent=phraseMode()?'Phrase review: play → grade → next. Unflagged stretches are skipped.':'';
    $('saveStatus').textContent='Ready · grades save automatically on this Mac';if(recovered)save();
  }catch(error){$('progress').textContent=error.message;$('saveStatus').textContent='Could not open review';}
})();
