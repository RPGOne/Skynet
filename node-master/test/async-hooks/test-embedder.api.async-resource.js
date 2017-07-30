'use strict';

const common = require('../common');
const assert = require('assert');
const tick = require('./tick');
const async_hooks = require('async_hooks');
const { AsyncResource } = async_hooks;

const initHooks = require('./init-hooks');
const { checkInvocations } = require('./hook-checks');

const hooks = initHooks();
hooks.enable();

assert.throws(() => new AsyncResource(),
              /^TypeError: type must be a string with length > 0$/);
assert.throws(() => new AsyncResource('invalid_trigger_id', null),
              /^RangeError: triggerId must be an unsigned integer$/);

assert.strictEqual(typeof new AsyncResource('default_trigger_id').triggerId(),
                   'number');

// create first custom event 'alcazares' with triggerId derived
// from async_hooks currentId
const alcaTriggerId = async_hooks.currentId();
const alcaEvent = new AsyncResource('alcazares', alcaTriggerId);
const alcazaresActivities = hooks.activitiesOfTypes([ 'alcazares' ]);

// alcazares event was constructed and thus only has an `init` call
assert.strictEqual(alcazaresActivities.length, 1);
const alcazares = alcazaresActivities[0];
assert.strictEqual(alcazares.type, 'alcazares');
assert.strictEqual(typeof alcazares.uid, 'number');
assert.strictEqual(alcazares.triggerId, alcaTriggerId);
checkInvocations(alcazares, { init: 1 }, 'alcazares constructed');

assert.strictEqual(typeof alcaEvent.asyncId(), 'number');
assert.notStrictEqual(alcaEvent.asyncId(), alcaTriggerId);
assert.strictEqual(alcaEvent.triggerId(), alcaTriggerId);

alcaEvent.emitBefore();
checkInvocations(alcazares, { init: 1, before: 1 },
                 'alcazares emitted before');
alcaEvent.emitAfter();
checkInvocations(alcazares, { init: 1, before: 1, after: 1 },
                 'alcazares emitted after');
alcaEvent.emitBefore();
checkInvocations(alcazares, { init: 1, before: 2, after: 1 },
                 'alcazares emitted before again');
alcaEvent.emitAfter();
checkInvocations(alcazares, { init: 1, before: 2, after: 2 },
                 'alcazares emitted after again');
alcaEvent.emitDestroy();
tick(1, common.mustCall(tick1));

function tick1() {
  checkInvocations(alcazares, { init: 1, before: 2, after: 2, destroy: 1 },
                   'alcazares emitted destroy');

  // The below shows that we can pass any number as a trigger id
  const pobTriggerId = 111;
  const pobEvent = new AsyncResource('poblado', pobTriggerId);
  const pobladoActivities = hooks.activitiesOfTypes([ 'poblado' ]);
  const poblado = pobladoActivities[0];
  assert.strictEqual(poblado.type, 'poblado');
  assert.strictEqual(typeof poblado.uid, 'number');
  assert.strictEqual(poblado.triggerId, pobTriggerId);
  checkInvocations(poblado, { init: 1 }, 'poblado constructed');
  pobEvent.emitBefore();
  checkInvocations(poblado, { init: 1, before: 1 },
                   'poblado emitted before');

  pobEvent.emitAfter();
  checkInvocations(poblado, { init: 1, before: 1, after: 1 },
                   'poblado emitted after');

  // after we disable the hooks we shouldn't receive any events anymore
  hooks.disable();
  alcaEvent.emitDestroy();
  tick(1, common.mustCall(tick2));

  function tick2() {
    checkInvocations(
      alcazares, { init: 1, before: 2, after: 2, destroy: 1 },
      'alcazares emitted destroy a second time after hooks disabled');
    pobEvent.emitDestroy();
    tick(1, common.mustCall(tick3));
  }

  function tick3() {
    checkInvocations(poblado, { init: 1, before: 1, after: 1 },
                     'poblado emitted destroy after hooks disabled');
  }
}
