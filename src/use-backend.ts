import { useCallback, useEffect, useState } from 'react';
import { getJson, parseCatalogue, parseEnvironment, type BackendEnvironment, type EnvironmentSummary } from './backend';

export function useBackend(enabled:boolean){
 const [catalogue,setCatalogue]=useState<EnvironmentSummary[]>([]),[id,setId]=useState(''),[environment,setEnvironment]=useState<BackendEnvironment|null>(null),[error,setError]=useState(''),[pending,setPending]=useState(false),[refreshKey,setRefreshKey]=useState(0);
 const refresh=useCallback(()=>setRefreshKey(v=>v+1),[]);
 useEffect(()=>{
   if(!enabled)return;
   const controller=new AbortController();let timer=0;setPending(true);setError('');
   async function poll(){
     try{
       const list=parseCatalogue(await getJson('/api/environments',controller.signal));if(controller.signal.aborted)return;setCatalogue(list);
       const entry=list.find(e=>e.id===id)||(!id?list[0]:undefined);
       if(!entry)throw new Error(id?'The selected environment is no longer available.':'No saved environments are available yet.');
       const next=parseEnvironment(await getJson(entry.url,controller.signal));if(controller.signal.aborted)return;
       if(next.id!==entry.id)throw new Error('Backend returned a different environment.');
       setEnvironment(previous=>JSON.stringify(previous)===JSON.stringify(next)?previous:next);setError('');
     }catch(e){if(!controller.signal.aborted)setError(e instanceof Error?e.message:'Cannot reach the environment backend.');}
     finally{if(!controller.signal.aborted){setPending(false);timer=window.setTimeout(poll,15000);}}
   }
   void poll();return()=>{controller.abort();clearTimeout(timer)};
 },[enabled,id,refreshKey]);
 const select=useCallback((next:string)=>{setEnvironment(null);setId(next)},[]);
 return {catalogue,environment,error,pending,refresh,select};
}
