import {pathToFileURL} from 'node:url';
const {chromium} = await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE).href);
const browser = await chromium.launch({executablePath:process.env.CHROMIUM_PATH,args:['--no-sandbox']});
try {
  const page = await browser.newPage();
  await page.goto(process.env.CANVAS_URL || 'http://127.0.0.1:18188');
  await page.waitForFunction(()=>window.LiteGraph?.registered_node_types?.TuringUtilsVideoMaskGuidedCrop);
  await page.waitForTimeout(1500);
  console.log(await page.evaluate(async()=>{
    if (LiteGraph.vueNodesMode) throw Error('Classic canvas required');
    app.graph.clear();
    const add=type=>{const n=LiteGraph.createNode(type);app.graph.add(n);return n;};
    let crop=add('TuringUtilsVideoMaskGuidedCrop');
    const source=add('PrimitiveInt');
    const check=(ok,msg)=>{if(!ok)throw Error(msg);};
    const wait=()=>new Promise(r=>setTimeout(r,150));
    const names=['width','height','context_scale','missing_mode','smooth_window','mask_threshold'];
    const rows=()=>names.map(name=>{
      const w=crop.widgets.find(w=>w.name===name);
      check(crop.getLayoutWidgets().includes(w)&&!w.hidden&&!w.advanced,'Missing ordinary control '+name);
      return w.y;
    });
    await wait();const before=rows();
    for(const name of ['width','height']) {
      const w=crop.widgets.find(w=>w.name===name);
      let i=crop.inputs.findIndex(s=>s.name===name);
      if(i<0){crop.addInput(name,'INT',{widget:{name}});i=crop.inputs.length-1;}
      check(source.connect(0,crop,i),'Could not connect '+name);
      // Force the legacy hide path even on a frontend that no longer hides it.
      w.type='converted-widget';w.hidden=true;w.computeSize=()=>[0,-4];
      crop.getLayoutWidgets();await wait();
      check(w.type==='number'&&w.disabled,'Connected numeric editor not retained');
      check(JSON.stringify(rows())===JSON.stringify(before),'Rows moved');
    }
    let prompt=await app.graphToPrompt();
    for(const name of ['width','height'])check(Array.isArray(prompt.output[crop.id].inputs[name]),'Link lost in prompt');
    const id=crop.id;
    await app.loadGraphData(app.graph.serialize());crop=app.graph.getNodeById(id);await wait();rows();
    for(const name of ['width','height']){
      const i=crop.inputs.findIndex(s=>s.name===name);
      check(crop.inputs[i].link!=null,'Link lost on reload');crop.disconnectInput(i);await wait();rows();
      check(!crop.widgets.find(w=>w.name===name).disabled,'Editor did not unlock');
    }
    return 'Crop classic canvas: visible controls, width/height links, legacy conversion, API links and reload/disconnect passed';
  }));
} finally {await browser.close();}
