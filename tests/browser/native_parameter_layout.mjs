import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const {chromium} = await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE).href);
const browser = await chromium.launch({executablePath:process.env.CHROMIUM_PATH,args:['--no-sandbox']});
try {
  const page = await browser.newPage();
  await page.goto(process.env.CANVAS_URL || 'http://127.0.0.1:18188');
  await page.waitForFunction(()=>window.LiteGraph?.registered_node_types?.TuringUtilsVideoMaskGuidedCrop);
  const report = await page.evaluate(async()=>{
    const {app} = await import('/scripts/app.js');
    if (LiteGraph.vueNodesMode) throw Error('Classic canvas required');
    const defs = await (await fetch('/object_info')).json();
    const check=(ok,msg)=>{if(!ok)throw Error(msg);};
    const tick=()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
    let nodes=0, fields=0, branches=0;
    const scalar = new Set(['number','text','string','combo','toggle']);
    const audit = (node, native, phase) => {
      const size=[Math.max(node.size[0],native.size[0]),Math.max(node.size[1],native.size[1])];
      node.setSize(size);native.setSize(size);
      node.arrange();native.arrange();
      node.arrange();native.arrange();
      check(JSON.stringify((node.widgets??[]).map(w=>w.name))===JSON.stringify((native.widgets??[]).map(w=>w.name)),`${node.type} ${phase}: widget order differs from native`);
      for(const ref of native.widgets??[]) {
        if(!scalar.has(ref.type) || ref.hidden) continue;
        const w=node.widgets.find(w=>w.name===ref.name);
        check(w.type===ref.type && !w.hidden && !w.advanced,`${node.type}.${w.name} ${phase}: type/visibility differs`);
        const slot=node.inputs.findIndex(s=>s.name===w.name);
        const nativeSlot=native.inputs.findIndex(s=>s.name===w.name);
        if(nativeSlot<0 || !native.inputs[nativeSlot].widget)continue;
        check(node.inputs[slot]?.widget?.name===w.name,`${node.type}.${w.name} ${phase}: detached binding`);
        check(Math.abs(w.y-ref.y)<0.1,`${node.type}.${w.name} ${phase}: row differs from native (${w.y}/${ref.y})`);
        const offset=node.getConnectionPos(true,slot)[1]-node.pos[1]-w.y;
        const nativeOffset=native.getConnectionPos(true,nativeSlot)[1]-native.pos[1]-ref.y;
        check(Math.abs(offset-nativeOffset)<0.1,`${node.type}.${w.name} ${phase}: socket offset differs from native (${offset}/${nativeOffset})`);
        check(Math.abs(offset-LiteGraph.NODE_SLOT_HEIGHT/2)<0.1,`${node.type}.${w.name} ${phase}: socket outside widget row (${offset}, native ${nativeOffset}, expected ${LiteGraph.NODE_SLOT_HEIGHT/2})`);
        fields++;
      }
    };
    for(const [name, def] of Object.entries(defs)) {
      if(!name.startsWith('TuringUtils') || def.is_dev_only || name==='TuringUtilsMultimodalPromptChat' || name==='TuringUtilsStagePath')continue;
      app.graph.clear();
      const nativeName='NativeLayoutAudit_'+name;
      await app.registerNodeDef(nativeName,{...def,name:nativeName,display_name:def.display_name});
      let node=LiteGraph.createNode(name), native=LiteGraph.createNode(nativeName);
      app.graph.add(node);app.graph.add(native);await tick();
      audit(node,native,'created');
      node.setSize([node.size[0]+200,node.size[1]+100]);
      audit(node,native,'resized');
      // Exercise every mode and nested DynamicCombo branch using the backend schema.
      async function modes(inputs, prefix='') {
        for(const [key,spec] of Object.entries({...inputs.required,...inputs.optional})) {
          if(spec[0]!=='COMFY_DYNAMICCOMBO_V3')continue;
          const field=prefix+key;
          const original=node.widgets.find(w=>w.name===field)?.value;
          for(const option of spec[1].options) {
            for(const n of [node,native]) {
              const w=n.widgets.find(w=>w.name===field);
              check(w,`${name}: missing dynamic selector ${field}`);
              w.value=option.key;w.callback?.(w.value);
            }
            await tick();audit(node,native,field+'='+option.key);branches++;
            await modes(option.inputs,field+'.');
          }
          for(const n of [node,native]) {const w=n.widgets.find(w=>w.name===field);w.value=original;w.callback?.(original);}
          await tick();
        }
      }
      await modes(def.input);
      const names=(native.widgets??[]).filter(w=>scalar.has(w.type)&&!w.hidden&&native.inputs.some(s=>s.widget?.name===w.name)).map(w=>w.name);
      // Native PrimitiveNode derives the correct output type and widget config.
      for(const field of names) {
        for(const n of [node,native]) {
          const i=n.inputs.findIndex(s=>s.name===field);
          n.onInputDblClick(i);
          check(n.inputs[i].link!=null,`${name}.${field}: primitive connection failed`);
        }
        await tick();audit(node,native,'connected '+field);
      }
      const id=node.id, nativeId=native.id;
      const saved=app.graph.serialize();
      await app.loadGraphData(saved);node=app.graph.getNodeById(id);native=app.graph.getNodeById(nativeId);
      await tick();audit(node,native,'reloaded');
      const prompt=await app.graphToPrompt();
      for(const field of names) {
        check(node.inputs.find(s=>s.name===field).link!=null,`${name}.${field}: reload lost link`);
        // Frontend PrimitiveNode is intentionally inlined by native graphToPrompt.
        // Stage Barrier is deliberately removed by its execution compiler.
        if(name!=='TuringUtilsStageBarrier') {
          check(prompt.output[node.id]&&prompt.output[native.id],`${name}: missing API node`);
          check(JSON.stringify(prompt.output[node.id].inputs[field])===JSON.stringify(prompt.output[native.id].inputs[field]),`${name}.${field}: exported value differs from native`);
        }
        for(const n of [node,native])n.disconnectInput(n.inputs.findIndex(s=>s.name===field));
      }
      await tick();audit(node,native,'disconnected');
      for(const field of names)check(!node.widgets.find(w=>w.name===field).disabled,`${name}.${field}: still disabled`);
      nodes++;
    }
    app.graph.clear();
    const chat=LiteGraph.createNode('TuringUtilsMultimodalPromptChat');app.graph.add(chat);await tick();
    const values=JSON.stringify(chat.serialize().widgets_values);
    for(const shown of [false,true,false]) {
      chat.showAdvanced=shown;chat.arrange();
      const visible=chat.getLayoutWidgets();
      for(const w of chat.widgets) {
        if(w.advanced)check(visible.includes(w)===shown,`Chat advanced visibility: ${w.name}`);
      }
      for(const name of ['prompt','system_prompt'])check(visible.some(w=>w.name===name),`Chat ordinary field hidden: ${name}`);
      check(JSON.stringify(chat.serialize().widgets_values)===values,'Chat toggle changed persisted values/order');
    }
    return {nodes,fields,branches,chatToggle:true};
  });
  assert.ok(report.nodes>30);
  console.log('Native-reference parameter audit:',report);
} finally {await browser.close();}
