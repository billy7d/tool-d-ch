// Chạy: npm test  (node --experimental-strip-types --test "tests/**/*.test.ts")
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { buildStyleGuide, defaultGlossaryPresets, draftStageLabel } from '../src/lib/styleGuide.ts';

const baseInput = {
  register: 'NEUTRAL',
  sentenceStyle: 'MODERATE',
  addressTerms: {},
  criticMode: 'AUTO' as const,
  glossaryPresets: ['LEGAL'],
  criticModel: '',
  semanticCriticModel: '',
};

test('glossary presets default to the document domain only when unset', () => {
  assert.deepEqual(defaultGlossaryPresets(undefined, 'LEGAL'), ['LEGAL']);
  assert.deepEqual(defaultGlossaryPresets(undefined, 'LITERATURE'), []);
  // Người dùng đã tắt hết thì giữ nguyên lựa chọn rỗng, không bật lại mặc định.
  assert.deepEqual(defaultGlossaryPresets([], 'LEGAL'), []);
  assert.deepEqual(defaultGlossaryPresets(['TECHNICAL'], 'LEGAL'), ['TECHNICAL']);
});

test('style guide keeps unrelated keys and drops half-filled address rows', () => {
  const guide = buildStyleGuide(
    { custom_key: 1, naturalness_critic_model: 'old' },
    {
      ...baseInput,
      addressTerms: {
        reader: 'bạn',
        pairs: [
          { speaker: 'Alice', listener: 'Bob', self: 'em', other: 'anh' },
          { speaker: 'Bob', listener: '', self: 'anh', other: '' },
        ],
        characters: { Alice: 'cô', ' ': 'x' },
      },
    },
  );
  assert.equal(guide.custom_key, 1);
  assert.equal(guide.address_terms.pairs.length, 1);
  assert.deepEqual(guide.address_terms.characters, { Alice: 'cô' });
  assert.deepEqual(guide.glossary_presets, ['LEGAL']);
});

test('empty critic models are removed so the backend falls back to the translation model', () => {
  const cleared = buildStyleGuide({ naturalness_critic_model: 'old', semantic_critic_model: 'old' }, baseInput);
  assert.equal('naturalness_critic_model' in cleared, false);
  assert.equal('semantic_critic_model' in cleared, false);
  const set = buildStyleGuide({}, { ...baseInput, semanticCriticModel: 'qwen2.5:14b' });
  assert.equal(set.semantic_critic_model, 'qwen2.5:14b');
});

test('draft stage labels are human readable', () => {
  assert.equal(draftStageLabel('SEMANTIC'), 'kiểm tra ngữ nghĩa');
  assert.equal(draftStageLabel('UNKNOWN'), 'UNKNOWN');
});
