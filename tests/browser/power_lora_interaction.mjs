import {pathToFileURL} from 'node:url';
const {chromium}=await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE).href);
const browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH,args:['--no-sandbox']});
try {
  const page=await browser.newPage({viewport:{width:1600,height:1200}});
  await page.route('**/turing/canvas/loras',r=>r.fulfill({json:{names:['test.safetensors','second.safetensors']}}));
  await page.goto('http://127.0.0.1:18188');
  await page.waitForFunction(()=>window.LiteGraph?.registered_node_types?.TuringCanvasH3Settings);
  await page.waitForTimeout(1800);
  console.log(await page.evaluate(()=>{
    app.graph.clear();
    const root=LiteGraph.createNode('TuringCanvasSettings'); app.graph.add(root);root.pos=[2000,0];
    const node=LiteGraph.createNode('TuringCanvasH3Settings'); app.graph.add(node);node.pos=[150,100];
    const storage=node.widgets.find(w=>w.name==='loras');storage.value=JSON.stringify([{name:'test.safetensors',on:true,strength:1}]);storage.callback();
    node.setSize([600,node.computeSize()[1]]);app.canvas.ds.scale=1;app.canvas.ds.offset=[0,0];app.graph.setDirtyCanvas(true,true);
    window.testLora=node;
    if (LiteGraph.vueNodesMode) throw Error('Must test the classic canvas');
    return {bound:app.canvas._events_binded,interactive:app.canvas.allow_interaction,vue:LiteGraph.vueNodesMode};
  }));
  await page.waitForTimeout(600);
  // Dispatch DOM pointer events through LiteGraph's canvas listener. This is
  // intentionally not a direct row.mouse call; CI has no physical pointer.
  for (const [index, width] of [600, 420, 800, 390, 600].entries()) {
  await page.evaluate(width=>{
    const node=window.testLora;
    node.setSize([width,node.size[1]]);
    // Reproduce an obsolete widget-width cache after parameter/layout changes.
    node.widgets.find(w=>w.canvasLoraEntry).width=250;
    app.graph.setDirtyCanvas(true,true);
  },width);
  await page.waitForTimeout(150);
  const point=await page.evaluate(()=>{
    const node=window.testLora,row=node.widgets.find(w=>w.canvasLoraEntry), b=row.hitAreas.strengthInc;
    if (Math.abs(b[0]+b[2]-(node.size[0]-10-20/3))>.01) throw Error('Stale width changed painted hit area');
    const rect=app.canvas.canvas.getBoundingClientRect();
    return {x:rect.x+node.pos[0]+b[0]+b[2]/2,y:rect.y+node.pos[1]+b[1]+b[3]/2};
  });
  await page.evaluate(({x,y})=>{
    const canvas=app.canvas.canvas;
    for(const type of ['pointermove','pointerdown','pointerup'])canvas.dispatchEvent(new PointerEvent(type,{bubbles:true,clientX:x,clientY:y,button:0,buttons:type==='pointerdown'?1:0,pointerId:1,isPrimary:true,pointerType:'mouse'}));
  },point);
  await page.waitForTimeout(150);
  await page.evaluate(expected=>{
    const node=window.testLora,storage=node.widgets.find(w=>w.name==='loras');
    if (Math.abs(JSON.parse(storage.value)[0].strength-expected)>1e-8) throw Error('Canvas pointer missed strength control');
    if (node.getLayoutWidgets().includes(storage)) throw Error('Storage overlaps LoRA row');
  },1+(index+1)*.05);
  }
  console.log('Classic canvas: repeated redraw/resize with stale widget width and DOM pointer clicks passed');
  if(process.env.CANVAS_SCREENSHOT) await page.screenshot({path:process.env.CANVAS_SCREENSHOT});
} finally {await browser.close();}
