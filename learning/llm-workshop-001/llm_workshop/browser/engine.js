/* TinyGPT forward pass, independent of the interface. Float64 arithmetic in JS.
   Weights and reference outputs come from the repository's actual PyTorch model. */
(function (root) {
  'use strict';
  const add = (a, b) => a.map((r, i) => r.map((x, j) => x + b[i][j]));
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  function linear(x, weight, bias) {
    return x.map(row => weight.map((w, j) => dot(row, w) + bias[j]));
  }
  function norm(x, weight, bias) {
    return x.map(row => {
      const mean = row.reduce((a, b) => a + b, 0) / row.length;
      const variance = row.reduce((s, v) => s + (v - mean) ** 2, 0) / row.length;
      return row.map((v, j) => (v - mean) / Math.sqrt(variance + 1e-5) * weight[j] + bias[j]);
    });
  }
  // Abramowitz–Stegun erf approximation; maximum absolute error about 1.5e-7.
  function erf(x) {
    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x);
    const t = 1 / (1 + .3275911 * x);
    return sign * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - .284496736) * t + .254829592) * t * Math.exp(-x * x));
  }
  const gelu = x => .5 * x * (1 + erf(x / Math.SQRT2));
  function softmax(values, temperature = 1) {
    if (!Number.isFinite(temperature) || temperature <= 0) throw new Error('Temperature must be positive.');
    const max = Math.max(...values);
    if (!Number.isFinite(max)) throw new Error('Softmax needs a finite allowed score.');
    const exp = values.map(v => Math.exp((v - max) / temperature));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(v => v / sum);
  }
  function encode(text, vocabulary, blockSize) {
    const chars = Array.from(text);
    if (!chars.length || chars.length > blockSize) throw new Error(`Use 1–${blockSize} characters.`);
    const ids = chars.map(c => vocabulary.indexOf(c));
    if (ids.includes(-1)) throw new Error(`Outside this teaching vocabulary: ${[...new Set(chars.filter((_, i) => ids[i] < 0))].map(c => JSON.stringify(c)).join(', ')}. Try lowercase English text.`);
    return ids;
  }
  // Mirror words/tokenizer.py. Keep word IDs throughout generation; display
  // spellings of special tokens must never be fed back into tokenization.
  function wordPieces(text) {
    if (typeof text !== 'string' || !text.trim()) throw new Error('Enter a few words, for example: the small cat');
    if (/[^a-zA-Z\s.!?]/.test(text)) throw new Error('Use English letters, spaces and . ! ? in this teaching tokenizer.');
    return text.toLowerCase().match(/[a-z]+|[.!?]/g) || [];
  }
  function encodeWords(text, vocabulary, blockSize) {
    const tokens = wordPieces(text);
    const missing = [...new Set(tokens.filter(token => !vocabulary.includes(token)))];
    if (missing.length) throw new Error(`Outside the fixed word vocabulary: ${missing.join(', ')}. Try a corpus example.`);
    const ids = [0, ...tokens.map(token => vocabulary.indexOf(token))];
    if (ids.length > blockSize) throw new Error(`Use at most ${blockSize - 1} word/punctuation tokens; BOS takes one position.`);
    return ids;
  }
  function decodeWords(ids, vocabulary) {
    const tokens = ids.map(i => vocabulary[i]);
    if (tokens[0] === '<BOS>') tokens.shift();
    if (tokens.at(-1) === '<EOS>') tokens.pop();
    return tokens.join(' ').replace(/\s+([.!?])/g, '$1');
  }
  function forward(ids, model, cfg) {
    if (!ids.length || ids.length > cfg.block_size || ids.some(i => !Number.isInteger(i) || i < 0 || i >= cfg.vocab_size)) throw new Error('Invalid token IDs or context length.');
    const w = model.weights, d = cfg.n_embd, dh = d / cfg.n_head;
    const tokens = ids.map(id => w['token_embedding.weight'][id].slice());
    const positions = ids.map((_, i) => w['position_embedding.weight'][i].slice());
    const combined = add(tokens, positions);
    let hidden = combined;
    const blocks = [];
    for (let b = 0; b < cfg.n_layer; b++) {
      const p = `blocks.${b}.`;
      const project = (x, name) => linear(x, w[p + name + '.weight'], w[p + name + '.bias']);
      const input = hidden;
      const normalized = norm(input, w[p + 'ln1.weight'], w[p + 'ln1.bias']);
      const qkv = project(normalized, 'attn.qkv');
      const heads = Array.from({length: cfg.n_head}, (_, h) => {
        const [Q, K, V] = [0, 1, 2].map(part => qkv.map(row => row.slice(part * d + h * dh, part * d + (h + 1) * dh)));
        const scores = Q.map(q => K.map(k => dot(q, k) / Math.sqrt(dh)));
        const mask = scores.map((row, i) => row.map((_, j) => Number(j <= i)));
        const masked_scores = scores.map((row, i) => row.map((v, j) => j <= i ? v : -Infinity));
        const attention = masked_scores.map(row => softmax(row));
        const weighted_values = attention.map(row => Array.from({length: dh}, (_, f) => row.reduce((s, a, j) => s + a * V[j][f], 0)));
        return {Q, K, V, scores, mask, masked_scores, attention, weighted_values};
      });
      const concatenated = ids.map((_, i) => heads.flatMap(head => head.weighted_values[i]));
      const attention_output = project(concatenated, 'attn.proj');
      const after_attention_residual = add(input, attention_output);
      const normalized_ffn = norm(after_attention_residual, w[p + 'ln2.weight'], w[p + 'ln2.bias']);
      const expanded = project(normalized_ffn, 'mlp.0');
      const activated = expanded.map(row => row.map(gelu));
      const feed_forward = project(activated, 'mlp.2');
      const output = add(after_attention_residual, feed_forward);
      blocks.push({input, normalized, heads, concatenated, attention_output, after_attention_residual, normalized_ffn, expanded, activated, feed_forward, output});
      hidden = output;
    }
    const final = norm(hidden, w['ln_f.weight'], w['ln_f.bias']);
    const logits = linear(final, w['lm_head.weight'], w['lm_head.bias']);
    return {tokens, positions, combined, blocks, final, logits};
  }
  function choose(probabilities, greedy = false, random = Math.random) {
    if (greedy) return probabilities.indexOf(Math.max(...probabilities));
    const value = random();
    let sum = 0;
    for (let i = 0; i < probabilities.length; i++) {
      sum += probabilities[i];
      if (value < sum) return i;
    }
    return probabilities.length - 1;
  }
  // Transparent lexical retrieval for the pedagogical evidence activity.
  // This is intentionally separate from token embeddings and neural generation.
  function retrieve(question, passages) {
    const stop = new Set(['the', 'a', 'an', 'is', 'in', 'of', 'to', 'and', 'where', 'what', 'when']);
    const terms = s => [...new Set((s.toLowerCase().match(/[a-z0-9]+/g) || []).filter(t => !stop.has(t)))];
    const query = terms(question);
    return passages.map((text, index) => {
      const words = new Set(terms(text));
      const matched = query.filter(t => words.has(t));
      return {text, index, matched, score: query.length ? matched.length / query.length : 0};
    }).sort((a, b) => b.score - a.score || a.index - b.index);
  }
  const api = {add, dot, linear, norm, gelu, softmax, encode, wordPieces, encodeWords, decodeWords, forward, choose, retrieve};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.TinyEngine = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
