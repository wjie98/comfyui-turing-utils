import {pathToFileURL} from 'node:url';
const {chromium}=await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE).href);
const browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH,args:['--no-sandbox']});
try {
  const page=await browser.newPage();
  await page.goto(process.env.CANVAS_URL || 'http://127.0.0.1:18188');
  await page.waitForFunction(()=>window.LiteGraph?.registered_node_types?.TuringUtilsVideoMaskGuidedCrop);
  await page.waitForTimeout(1500);
  console.log(await page.evaluate(async()=>{
    if(LiteGraph.vueNodesMode)throw Error('Classic canvas required');
    app.graph.clear();
    const check=(ok,msg)=>{if(!ok)throw Error(msg);};
    const types=Object.keys(LiteGraph.registered_node_types).filter(n=>n.startsWith('TuringUtils')&&n!=='TuringUtilsStagePath');
    const nodes=types.map(type=>{const n=LiteGraph.createNode(type);app.graph.add(n);return n;});
    const saved=new Map(nodes.map(n=>[n.type,JSON.stringify(n.serialize().widgets_values)]));
    const {subgraph,node:container}=app.graph.convertToSubgraph(new Set(nodes));
    const audit=graph=>{
      for(const n of graph.nodes){
        if(!saved.has(n.type))continue;
        check(JSON.stringify(n.serialize().widgets_values)===saved.get(n.type),'Subgraph changed values '+n.type);
        if(n.type!=='TuringUtilsMultimodalPromptChat'){
          check((n.widgets??[]).every(w=>!w.advanced),'Subgraph added advanced '+n.type);
          check(JSON.stringify(n.getLayoutWidgets().map(w=>w.name))===JSON.stringify((n.widgets??[]).filter(w=>n.isWidgetVisible(w)).map(w=>w.name)),'Subgraph reordered '+n.type);
        }
      }
    };
    app.canvas.setGraph(subgraph);audit(subgraph);
    app.canvas.setGraph(app.graph);
    const id=subgraph.id;
    await app.loadGraphData(app.graph.serialize());
    const restored=app.graph.subgraphs.get(id);check(restored,'Subgraph missing on reload');
    app.canvas.setGraph(restored);audit(restored);
    app.canvas.setGraph(app.graph);
    return `Classic subgraph enter/reload: ${saved.size} ordinary nodes retain values and parameter order`;
  }));
} finally {await browser.close();}
