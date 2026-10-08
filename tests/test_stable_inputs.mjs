import assert from 'node:assert/strict';
import {test} from 'node:test';
import {stableInputRows} from '../web/lib/stable_inputs.js';

test('connected coordinate controls retain their rows, values and sockets', () => {
  const computeSize = () => [300,20];
  const widgets = ['positive_coords','negative_coords'].map(name=>({name,type:'text',computeSize,value:'[]',options:{}}));
  const inputs = widgets.map(w=>({name:w.name,widget:{name:w.name},link:null}));
  const node = {widgets, inputs}, restore = stableInputRows(node, widgets.map(w=>w.name));
  restore();
  assert.ok(inputs.every(input => !input.widget));
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
