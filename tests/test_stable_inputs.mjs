import assert from 'node:assert/strict';
import {test} from 'node:test';
import {stableInputRows, stableRowNames} from '../web/lib/stable_inputs.js';

test('connected coordinate controls retain their rows, values and sockets', () => {
  const computeSize = () => [300,20];
  const widgets = ['positive_coords','negative_coords'].map(name=>({name,type:'text',computeSize,value:'[]',options:{}}));
  const inputs = widgets.map(w=>({name:w.name,widget:{name:w.name},link:null}));
  const node = {widgets, inputs}, restore = stableInputRows(node, widgets.map(w=>w.name));
  restore();
  assert.ok(inputs.every(input => input.widget?.name === input.name));
  inputs[0].link = 42;
  widgets[0].hidden = true; widgets[0].type = 'converted-widget'; widgets[0].computeSize = () => [0,-4];
  restore();
  assert.equal(widgets[0].type,'text'); assert.equal(widgets[0].computeSize,computeSize);
  assert.equal(widgets[0].hidden,false); assert.equal(widgets[0].disabled,true);
  assert.equal(widgets[1].disabled,false); assert.equal(inputs[0].link,42);
  inputs[1].link = 43; restore(); assert.equal(widgets[1].disabled,true);
  inputs[0].link=null; restore(); assert.equal(widgets[0].disabled,false);
  assert.equal(widgets[0].value,'[]'); assert.equal(node.inputs,inputs);
});

test('crop numeric and combo controls survive late legacy input conversion', () => {
  const names = stableRowNames.TuringUtilsVideoMaskGuidedCrop;
  assert.ok(names.includes('width') && names.includes('height'));
  const widgets = names.map(name => ({name, type: name === 'missing_mode' ? 'combo' : 'number',
    value: name === 'missing_mode' ? 'interpolate' : 768, options: {}, computeSize: () => [300,20]}));
  const node = {widgets, inputs: [{name:'images',type:'IMAGE',link:1},{name:'masks',type:'MASK',link:2}]};
  const restore = stableInputRows(node, names);
  restore();
  for (const [index, widget] of widgets.entries()) {
    const type = widget.type, value = widget.value;
    // Older frontends create this input when converting a widget for wiring.
    const slot = {name:widget.name,type:type === 'combo' ? ['interpolate','hold'] : 'INT',
      widget:{name:widget.name},link:100+index,pos:[0,0]};
    node.inputs.push(slot);
    widget.type='converted-widget';widget.hidden=true;widget.computeSize=()=>[0,-4];
    restore();
    assert.equal(widget.type,type);assert.equal(widget.hidden,false);assert.equal(widget.disabled,true);
    assert.equal(widget.value,value);assert.equal(slot.link,100+index);assert.equal(slot.widget.name,widget.name);
    assert.deepEqual(slot.pos,[0,0]);
    assert.equal(widget.computeSize()[1],20);
    slot.link=null;restore();assert.equal(widget.disabled,false);
  }
  assert.equal(node.inputs[0].link,1);assert.equal(node.inputs[1].link,2);
});

test('saved detached sockets recover native metadata without replacing links or slots', () => {
  const config = Symbol('native widget config');
  const binding = {name:'width', [config]: () => ['INT', {min:1}]};
  const widget = {name:'width', type:'number', value:768, options:{}};
  const node = {widgets:[widget],inputs:[{name:'width',widget:binding,link:null}]};
  const restore = stableInputRows(node,['width']);
  const savedSlot = {name:'width',link:42};
  node.inputs = [savedSlot];
  restore();
  assert.equal(savedSlot.widget,binding);
  assert.equal(savedSlot.widget[config]()[0],'INT');
  assert.equal(node.inputs[0],savedSlot);
  assert.equal(savedSlot.link,42);
});
