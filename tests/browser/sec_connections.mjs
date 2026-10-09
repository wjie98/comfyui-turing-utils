import {pathToFileURL} from 'node:url';
const {chromium} = await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE).href);
const browser = await chromium.launch({executablePath:process.env.CHROMIUM_PATH,args:['--no-sandbox']});
try {
  const page = await browser.newPage();
  await page.goto('http://127.0.0.1:18188');
  await page.waitForFunction(()=>window.LiteGraph?.registered_node_types?.TuringUtilsSeCTrackVisualConcept);
  await page.waitForTimeout(1500);
  console.log(await page.evaluate(async()=>{
    app.graph.clear();
    let target=LiteGraph.createNode('TuringUtilsSeCTrackVisualConcept');app.graph.add(target);
    let source=LiteGraph.createNode('TuringUtilsMaskToVisualPrompts');app.graph.add(source);
    const check=(ok,message)=>{if(!ok)throw Error(message);};
    const wait=()=>new Promise(r=>setTimeout(r,200));
    const positions=()=>['positive_coords','negative_coords'].map(name=>{
      const widget=target.widgets.find(w=>w.name===name);
      check(target.getLayoutWidgets().includes(widget), 'Missing coordinate row: '+name);
      check(!widget.hidden && widget.type==='text', 'Hidden/converted coordinate: '+name);
      const i=target.inputs.findIndex(s=>s.name===name);
      check(target.inputs[i]?.widget?.name===name,'Detached coordinate socket '+name);
      check(Math.abs(target.getConnectionPos(true,i)[1]-target.pos[1]-widget.y-LiteGraph.NODE_SLOT_HEIGHT/2)<0.1,'Socket not on coordinate row '+name);
      return widget.y;
    });
    await wait(); const before=positions();
    for (const name of ['positive_coords','negative_coords']) {
      check(target.inputs.find(s=>s.name===name).widget?.name===name, 'Coordinate socket lost native binding');
    }
    for(const name of ['positive_coords','negative_coords']) {
      const output=source.outputs.findIndex(s=>s.name===name),input=target.inputs.findIndex(s=>s.name===name);
      check(source.connect(output,target,input), 'Connection failed: '+name);
      await wait();
      check(JSON.stringify(positions())===JSON.stringify(before), 'Coordinate rows moved on connection');
      check(target.widgets.find(w=>w.name===name).disabled, 'Connected field remains editable');
    }
    check(before[1]>before[0], 'Overlapping coordinates');
    const prompt=await app.graphToPrompt();
    for(const name of ['positive_coords','negative_coords']) {
      const link=prompt.output[String(target.id)].inputs[name];
      check(Array.isArray(link) && String(link[0])===String(source.id) && link[1]===source.outputs.findIndex(s=>s.name===name), 'Local text replaced upstream connection: '+name);
    }
    // Exercise legacy/extension conversion even on frontends which keep widgets.
    const positive=target.widgets.find(w=>w.name==='positive_coords');
    positive.type='converted-widget';positive.hidden=true;positive.computeSize=()=>[0,-4];
    target.getLayoutWidgets();await wait();
    check(JSON.stringify(positions())===JSON.stringify(before), 'Converted row did not recover');
    const saved=app.graph.serialize(),targetId=target.id,sourceId=source.id;
    await app.loadGraphData(saved);target=app.graph.getNodeById(targetId);source=app.graph.getNodeById(sourceId);
    await wait();positions();
    for(const name of ['positive_coords','negative_coords']) {
      const input=target.inputs.findIndex(s=>s.name===name);
      check(target.inputs[input].link!=null,'Link lost during reload');
      target.disconnectInput(input);await wait();positions();
      check(!target.widgets.find(w=>w.name===name).disabled,'Field did not unlock');
      check(source.connect(source.outputs.findIndex(s=>s.name===name),target,input),'Reconnect failed');
    }
    return 'SeC: stable rows, independent connections, legacy conversion, reload and reconnect OK';
  }));
} finally {await browser.close();}
