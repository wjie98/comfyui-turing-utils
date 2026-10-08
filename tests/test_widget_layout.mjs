import assert from 'node:assert/strict';
import {test} from 'node:test';
import {advancedLast, isInternalNode, hideInternalNode} from '../web/lib/widget_layout.js';

test('advanced controls move visually without changing saved parameter order', () => {
  const widgets = [{name:'model'}, {name:'attention',advanced:true}, {name:'points'}, {name:'memory',advanced:true}];
  assert.deepEqual(advancedLast(widgets).map(w=>w.name), ['model','points','attention','memory']);
  assert.deepEqual(widgets.map(w=>w.name), ['model','attention','points','memory']);
  assert.equal(advancedLast(widgets)[2], widgets[1]);
  assert.deepEqual(advancedLast(widgets.filter(w=>!w.advanced)).map(w=>w.name), ['model','points']);
});

test('internal nodes remain hidden when developer mode resets palette flags', () => {
  for (const name of ['_TuringUtilsSeCApply', '_TuringCanvasRead', 'TuringUtilsStagePath']) assert.ok(isInternalNode(name));
  for (const name of ['TuringUtilsMultimodalPromptChat', 'TuringUtilsMiniMaxH3LatentUpscale', 'OtherPluginInternal']) assert.ok(!isInternalNode(name));
  const type = {}; hideInternalNode(type);
  type.skip_list = false;
  assert.equal(type.skip_list, true);
});
