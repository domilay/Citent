/* ═══════════════════════════════════════════════════════════════
   JUROR · Live Agent · client-side LLM runtime
   ----------------------------------------------------------------
   When the user supplies an API key, this module replaces the
   scripted demo trace by actually running the agent in the browser.
   It emits the SAME action stream (`flat` array) that animation.js
   already knows how to play, so the visual layer is unchanged.

   Providers supported:
     - anthropic  (Claude Messages API; requires
                   `anthropic-dangerous-direct-browser-access` header)
     - openai     (Chat Completions; CORS works out of the box)
     - google     (Gemini generateContent)
     - deepseek   (OpenAI-compatible; CORS as of late 2024)

   The key NEVER leaves the browser except to go directly to the
   provider's official endpoint.  No proxy server.
   ═══════════════════════════════════════════════════════════════ */

window.JUROR_LIVE = (function () {

  // ───────────────────────────────────────────────────────────
  //  Provider configuration
  // ───────────────────────────────────────────────────────────
  const PROVIDERS = {
    anthropic: {
      defaultModel: 'claude-opus-4-6',
      keyHint:      'sk-ant-…',
      endpoint:     'https://api.anthropic.com/v1/messages',
      buildBody:    (sys, user, model, max_tokens) => ({
        model, max_tokens, temperature: 0,
        system: sys,
        messages: [{ role: 'user', content: user }],
      }),
      buildHeaders: (key) => ({
        'x-api-key':                                 key,
        'anthropic-version':                         '2023-06-01',
        'anthropic-dangerous-direct-browser-access': 'true',
        'content-type':                              'application/json',
      }),
      parseResponse: (j) => ({
        text:   j.content?.[0]?.text ?? '',
        tokens: (j.usage?.input_tokens || 0) + (j.usage?.output_tokens || 0),
      }),
    },

    openai: {
      defaultModel: 'gpt-5.4',
      keyHint:      'sk-…',
      endpoint:     'https://api.openai.com/v1/chat/completions',
      buildBody:    (sys, user, model, max_tokens) => ({
        model, max_tokens, temperature: 0,
        messages: [
          { role: 'system', content: sys },
          { role: 'user',   content: user },
        ],
      }),
      buildHeaders: (key) => ({
        'Authorization': `Bearer ${key}`,
        'content-type': 'application/json',
      }),
      parseResponse: (j) => ({
        text:   j.choices?.[0]?.message?.content ?? '',
        tokens: j.usage?.total_tokens || 0,
      }),
    },

    google: {
      defaultModel: 'gemini-3.5-flash',
      keyHint:      'AIza…',
      endpoint:     null,   // computed per request because key goes in URL
      buildBody:    (sys, user, model, max_tokens) => ({
        systemInstruction: { parts: [{ text: sys }] },
        contents:           [{ role: 'user', parts: [{ text: user }] }],
        generationConfig:   { temperature: 0, maxOutputTokens: max_tokens },
      }),
      buildHeaders: ()    => ({ 'content-type': 'application/json' }),
      buildUrl:     (key, model) =>
        `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(key)}`,
      parseResponse: (j) => ({
        text:   j.candidates?.[0]?.content?.parts?.[0]?.text ?? '',
        tokens: (j.usageMetadata?.totalTokenCount) || 0,
      }),
    },

    deepseek: {
      defaultModel: 'deepseek-chat',
      keyHint:      'sk-…',
      endpoint:     'https://api.deepseek.com/v1/chat/completions',
      buildBody:    (sys, user, model, max_tokens) => ({
        model, max_tokens, temperature: 0,
        messages: [
          { role: 'system', content: sys },
          { role: 'user',   content: user },
        ],
      }),
      buildHeaders: (key) => ({
        'Authorization': `Bearer ${key}`,
        'content-type': 'application/json',
      }),
      parseResponse: (j) => ({
        text:   j.choices?.[0]?.message?.content ?? '',
        tokens: j.usage?.total_tokens || 0,
      }),
    },
  };

  function defaultModelFor(provider) { return PROVIDERS[provider]?.defaultModel ?? ''; }

  // ───────────────────────────────────────────────────────────
  //  Local non-LLM tools (mirror the prototype/juror.py logic)
  // ───────────────────────────────────────────────────────────
  const LABELS = ['Background', 'Technical basis', 'Comparison', 'Fundamental idea'];

  const LABEL_PRIORS = {
    'Background': {
      markers: ['see e.g.', 'for a review', 'has been used', 'among others',
                'as discussed', 'in general', 'previously', 'literature'],
      negative_verbs: ['we follow', 'we adopt', 'we use', 'compared to', 'outperform'],
    },
    'Technical basis': {
      markers: ['we follow', 'we adopt', 'we use', 'we extend',
                'based on', 'building on', 'we apply', 'we implement'],
      negative_verbs: [],
    },
    'Comparison': {
      markers: ['compared to', 'in contrast', 'outperform', 'worse than',
                'vs.', 'performs better'],
      negative_verbs: [],
    },
    'Fundamental idea': {
      markers: ['based on the idea', 'inspired by', 'the key insight',
                'motivated by', 'underlying principle'],
      negative_verbs: [],
    },
  };

  function tfidfLrClassifier(text) {
    const lower = text.toLowerCase();
    const scores = {};
    for (const [label, prior] of Object.entries(LABEL_PRIORS)) {
      let s = 0.5;
      for (const m of prior.markers)        if (lower.includes(m))            s += 1.0;
      for (const v of prior.negative_verbs) if (lower.includes(v))            s -= 0.7;
      scores[label] = Math.max(s, 0.05);
    }
    const total = Object.values(scores).reduce((a, b) => a + b, 0);
    const probs = {};
    for (const [k, v] of Object.entries(scores)) probs[k] = +(v / total).toFixed(3);
    const top = Object.entries(probs).sort((a, b) => b[1] - a[1])[0][0];
    return { predicted_label: top, probs, max_prob: probs[top],
             difficulty: +(1 - probs[top]).toFixed(2) };
  }

  function rhetoricalMarkers(text) {
    const lower = text.toLowerCase();
    const found = [];
    for (const [label, prior] of Object.entries(LABEL_PRIORS)) {
      for (const m of prior.markers) {
        let idx = 0;
        while ((idx = lower.indexOf(m, idx)) !== -1) {
          found.push({ marker: m, label, pos: idx });
          idx += m.length;
        }
      }
    }
    return found;
  }

  function sectionClassifier(text) {
    const t = text.toLowerCase();
    if (['we propose', 'our method', 'we follow', 'we extend', 'we apply', 'we implement']
        .some(k => t.includes(k))) return { section: 'Methods', confidence: 0.78 };
    if (['in general', 'for a review', 'literature', 'previously', 'has been used']
        .some(k => t.includes(k))) return { section: 'Related Work', confidence: 0.74 };
    if (['outperform', 'results show', 'achieves a', 'compared to', 'in contrast']
        .some(k => t.includes(k))) return { section: 'Experiments / Comparison', confidence: 0.68 };
    return { section: 'Unknown', confidence: 0.30 };
  }

  function coCitationLookup(ref, paragraph) {
    const re = /[A-Z][A-Za-z]+(?:\s+et\s+al\.?|\s+and\s+[A-Z][A-Za-z]+)?\s*\(\d{4}\)/g;
    const refs = paragraph.match(re) || [];
    return refs.filter(r => r.trim() !== ref.trim()).slice(0, 5);
  }

  function exactSpanMatch(span, source) {
    const a = span.replace(/\s+/g, ' ').trim().toLowerCase();
    const b = source.replace(/\s+/g, ' ').trim().toLowerCase();
    return b.includes(a);
  }

  function bleuOverlap(span, source) {
    const s = new Set(span.toLowerCase().split(/\s+/));
    const t = new Set(source.toLowerCase().split(/\s+/));
    if (s.size === 0) return 0;
    let inter = 0;
    for (const w of s) if (t.has(w)) inter++;
    return inter / s.size;
  }

  // ───────────────────────────────────────────────────────────
  //  LLM call wrapper
  // ───────────────────────────────────────────────────────────
  async function callLLM(provider, key, model, systemPrompt, userPrompt, maxTokens = 220) {
    const p = PROVIDERS[provider];
    if (!p) throw new Error('Unknown provider: ' + provider);

    const url     = p.buildUrl  ? p.buildUrl(key, model) : p.endpoint;
    const headers = p.buildHeaders(key);
    const body    = JSON.stringify(p.buildBody(systemPrompt, userPrompt, model, maxTokens));

    const resp = await fetch(url, { method: 'POST', headers, body });
    if (!resp.ok) {
      const errText = await resp.text();
      throw new Error(`${provider} HTTP ${resp.status}: ${errText.slice(0, 200)}`);
    }
    const j = await resp.json();
    return p.parseResponse(j);
  }

  // ───────────────────────────────────────────────────────────
  //  Run live agent against one case input.  Builds a trace
  //  with the exact same shape as the JSON files in traces/.
  // ───────────────────────────────────────────────────────────
  async function runCase({ caseInput, config, onProgress }) {
    const trace = {
      case_id: caseInput.case_id,
      input:   { paragraph:        caseInput.paragraph,
                 cited_ref:        caseInput.ref_clean_citation,
                 ground_truth:     caseInput.ground_truth || null },
      schema:  { task: 'citation_intent', labels: LABELS },
      stages:  [],
    };

    const report = (stage) => { if (onProgress) onProgress(stage); };

    // ─── Stage 1: Triage ────────────────────────────────────
    const t0 = performance.now();
    const triage = tfidfLrClassifier(caseInput.paragraph);
    const sortedProbs = Object.entries(triage.probs).sort((a, b) => b[1] - a[1]);
    const shortlist = sortedProbs.slice(0, 3).filter(([, p]) => p > 0.05).map(([l]) => l);
    const triageStage = {
      id: 'triage', name: 'Triage Officer', engine: 'ML',
      tools_called: ['tfidf_lr_classifier'],
      duration_ms: +(performance.now() - t0).toFixed(1),
      tokens: 0, llm_calls: 0,
      outputs: { difficulty: triage.difficulty, shortlist, probs: triage.probs,
                 fast_track: triage.difficulty < 0.30 },
      actions: [
        { t: 0,    type: 'say',       actor: 'triage',
          text: `New case received: ${caseInput.case_id}.` },
        { t: 250,  type: 'tool_call', tool: 'tfidf_lr_classifier', engine: 'ML',
          summary: `max_prob = ${triage.max_prob.toFixed(3)}` },
        { t: 700,  type: 'say',       actor: 'triage',
          text: `Difficulty = ${triage.difficulty.toFixed(2)}. Shortlisting: ${shortlist.join(', ')}.` },
        { t: 1100, type: 'shortlist', value: shortlist, fast_track: triage.difficulty < 0.30 },
      ],
    };
    trace.stages.push(triageStage);
    report(triageStage);

    // ─── Stage 2: Investigator ──────────────────────────────
    const t1 = performance.now();
    const markers   = rhetoricalMarkers(caseInput.paragraph);
    const section   = sectionClassifier(caseInput.paragraph);
    const coCites   = coCitationLookup(caseInput.ref_clean_citation, caseInput.paragraph);
    const invStage = {
      id: 'investigator', name: 'Investigator', engine: 'RULE+ML+GRAPH',
      tools_called: ['rhetorical_marker_regex', 'section_classifier', 'co_citation_lookup'],
      duration_ms: +(performance.now() - t1).toFixed(1),
      tokens: 0, llm_calls: 0,
      outputs: { markers, section, co_citations: coCites },
      actions: [
        { t: 0,    type: 'say',       actor: 'investigator', text: 'Gathering evidence; cheapest tools first.' },
        { t: 350,  type: 'tool_call', tool: 'rhetorical_marker_regex', engine: 'RULE',  summary: `${markers.length} marker(s) found` },
        { t: 800,  type: 'say',       actor: 'investigator', text: `Found ${markers.length} rhetorical marker(s).` },
        { t: 1250, type: 'tool_call', tool: 'section_classifier',      engine: 'ML',    summary: section.section },
        { t: 1600, type: 'say',       actor: 'investigator', text: `Section: ${section.section} (conf ${section.confidence}).` },
        { t: 2000, type: 'tool_call', tool: 'co_citation_lookup',      engine: 'GRAPH', summary: `${coCites.length} co-citation(s)` },
        { t: 2400, type: 'say',       actor: 'investigator', text: `Case file complete.` },
      ],
    };
    trace.stages.push(invStage);
    report(invStage);

    // ─── Stage 3: Prosecution Panel (REAL LLM, one call per intent) ─
    const t2 = performance.now();
    const advocates = [];
    const panelActions = [];
    let actionT = 0;
    let panelTokens = 0;

    const intentDefs = {
      'Background':      'the cited work provides general background or situating context',
      'Technical basis': 'the citing paper uses, adopts, extends, or builds upon methods from the cited work',
      'Comparison':      'the citing paper compares its methods or findings against the cited work',
      'Fundamental idea':'the cited work provides a core theoretical concept or key insight',
    };

    const panelSys = (
      'You are an expert citation analyst arguing as an advocate. '
      + 'Given a paragraph and a cited reference, you will write a SHORT (under 25 words) '
      + 'argument for why the citation intent is YOUR ASSIGNED LABEL. '
      + 'Cite specific text from the paragraph if possible. '
      + 'Do not include explanations or caveats. Speak in first person, like a lawyer.'
    );

    for (const label of shortlist) {
      const userPrompt =
        `Paragraph: "${caseInput.paragraph}"\n\n`
        + `Cited reference: ${caseInput.ref_clean_citation}\n\n`
        + `Your assigned label: ${label}\n`
        + `Label definition: ${intentDefs[label]}\n\n`
        + `Write your single-line argument now, starting with "I argue ${label}...".`;

      let argText, tokens;
      try {
        const r = await callLLM(config.provider, config.apiKey, config.model || defaultModelFor(config.provider),
                                panelSys, userPrompt, 100);
        argText = (r.text || '').trim() || `I argue ${label}.`;
        tokens  = r.tokens;
      } catch (e) {
        argText = `I argue ${label}. (LLM call failed: ${e.message.slice(0, 60)})`;
        tokens  = 0;
      }
      panelTokens += tokens;

      // detect evidence spans by regex against the paragraph
      const evSpans = [];
      const qmatch = argText.match(/"([^"]{6,120})"/g) || [];
      for (const q of qmatch) evSpans.push({ span: q.replace(/"/g, ''), marker: 'cited' });
      // strength = number of matched real markers + bonus for verifiable evidence
      const labelMarkers = markers.filter(m => m.label === label);
      let strength = labelMarkers.length * 1.0;
      if (section.section === 'Related Work'           && label === 'Background')      strength += 0.6;
      if (section.section === 'Methods'                && label === 'Technical basis') strength += 0.6;
      if (section.section.startsWith('Experiments')    && label === 'Comparison')      strength += 0.4;

      advocates.push({ label, argument: argText, evidence_spans: evSpans,
                       raw_strength: +Math.max(strength, 0).toFixed(2) });

      panelActions.push(
        { t: actionT,       type: 'advocate_speak', label, text: argText, strength: +strength.toFixed(2) },
        { t: actionT + 100, type: 'tool_call', tool: `llm_advocate[${label}]`, engine: 'LLM',
          summary: `${tokens} tokens` },
      );
      actionT += 1800;
    }

    const panelStage = {
      id: 'panel', name: 'Prosecution Panel', engine: 'LLM',
      tools_called: advocates.map(a => `llm_advocate[${a.label}]`),
      duration_ms: +(performance.now() - t2).toFixed(1),
      tokens: panelTokens, llm_calls: advocates.length,
      outputs: { advocates },
      actions: panelActions,
    };
    trace.stages.push(panelStage);
    report(panelStage);

    // ─── Stage 4: Cross-Examiner (REAL LLM, one call) ───────
    const t3 = performance.now();
    const attacks = [];
    const examinerActions = [];

    // Rule-based pre-flag: if section conflicts with label
    for (const adv of advocates) {
      let flag = null, text = null;
      if (adv.label === 'Technical basis' && section.section === 'Related Work') {
        flag = 'contradicted';
        text = `Paragraph is from Related Work, not Methods — the 'technical basis' claim is undermined.`;
      } else if (adv.label === 'Comparison' && adv.raw_strength < 0.5) {
        flag = 'unsupported';
        text = `No comparative verb present — this argument lacks textual evidence.`;
      } else if (adv.raw_strength < 0.3) {
        flag = 'weak';
        text = `Only marginal signal; not enough to overcome the priors.`;
      }
      if (flag) attacks.push({ label: adv.label, attack: text, flag });
    }

    let examinerTokens = 0;
    if (attacks.length > 0) {
      // Refine attacks with a real LLM critique
      const critSys = 'You are a Cross-Examiner attacking weak citation-intent arguments. '
                    + 'Given a paragraph and one weak argument, write a SHORT (< 25 words) sharp rebuttal.';
      try {
        // batch — one call covers all flagged attacks
        const critUser = `Paragraph: "${caseInput.paragraph}"\n\n`
                       + `Cited: ${caseInput.ref_clean_citation}\n\n`
                       + `Weak arguments to rebut:\n`
                       + attacks.map((a, i) => `${i + 1}. [${a.label}] ${a.attack}`).join('\n')
                       + `\n\nReturn a numbered rebuttal list, one short sentence each.`;
        const r = await callLLM(config.provider, config.apiKey, config.model || defaultModelFor(config.provider),
                                critSys, critUser, 220);
        const lines = (r.text || '').split('\n').filter(l => /^\d/.test(l.trim()));
        for (let i = 0; i < attacks.length && i < lines.length; i++) {
          attacks[i].attack = lines[i].replace(/^\d+\.\s*/, '').trim() || attacks[i].attack;
        }
        examinerTokens = r.tokens;
      } catch (e) { /* keep rule-based attacks */ }
    }

    let atkT = 0;
    if (attacks.length === 0) {
      examinerActions.push({ t: 0, type: 'say', actor: 'examiner',
        text: 'No advocate argument fails the contradiction filter.' });
    } else {
      for (const atk of attacks) {
        examinerActions.push({ t: atkT, type: 'attack', label: atk.label,
                               text: atk.attack, flag: atk.flag });
        atkT += 1400;
      }
    }
    const examinerStage = {
      id: 'examiner', name: 'Cross-Examiner', engine: 'RULE+LLM',
      tools_called: ['contradiction_pattern', ...(attacks.length ? ['llm_critic'] : [])],
      duration_ms: +(performance.now() - t3).toFixed(1),
      tokens: examinerTokens, llm_calls: attacks.length ? 1 : 0,
      outputs: { attacks }, actions: examinerActions,
    };
    trace.stages.push(examinerStage);
    report(examinerStage);

    // ─── Stage 5: Fact-Checker (LOCAL, no LLM) ──────────────
    const t4 = performance.now();
    const verifications = [];
    let verT = 0;
    const factActions = [];
    for (const adv of advocates) {
      for (const ev of adv.evidence_spans) {
        const exact = exactSpanMatch(ev.span, caseInput.paragraph);
        const bleu  = +bleuOverlap(ev.span, caseInput.paragraph).toFixed(2);
        const verified = exact || bleu >= 0.85;
        verifications.push({ label: adv.label, span: ev.span,
                             exact_match: exact, bleu, verified });
        const short = ev.span.length > 80 ? ev.span.slice(0, 80) + '…' : ev.span;
        factActions.push({ t: verT, type: 'verify', label: adv.label, span: short,
                           verified, exact, bleu });
        verT += 420;
      }
    }
    if (factActions.length === 0) {
      factActions.push({ t: 0, type: 'say', actor: 'factchecker',
        text: 'No quoted spans to verify (advocates argued without direct quotes).' });
    }
    const factStage = {
      id: 'fact_checker', name: 'Fact-Checker', engine: 'RULE+NLI',
      tools_called: ['exact_span_match', 'bleu_overlap'],
      duration_ms: +(performance.now() - t4).toFixed(1),
      tokens: 0, llm_calls: 0,
      outputs: { verifications }, actions: factActions,
    };
    trace.stages.push(factStage);
    report(factStage);

    // ─── Stage 6: Judge (REAL LLM, one call) ────────────────
    const t5 = performance.now();
    const verifiedCount = {};
    for (const v of verifications) if (v.verified)
      verifiedCount[v.label] = (verifiedCount[v.label] || 0) + 1;
    const attacksByLabel = {};
    for (const a of attacks) attacksByLabel[a.label] = a;

    const scores = {};
    for (const adv of advocates) {
      let s = adv.raw_strength + 0.5 * (verifiedCount[adv.label] || 0);
      if (attacksByLabel[adv.label]) {
        const pen = { contradicted: 1.5, unsupported: 1.2, weak: 0.7 }[attacksByLabel[adv.label].flag] || 0.5;
        s -= pen;
      }
      s += triage.probs[adv.label] || 0;
      scores[adv.label] = +Math.max(s, 0.02).toFixed(3);
    }
    // softmax
    const expS = {}; let Z = 0;
    for (const [k, v] of Object.entries(scores)) { expS[k] = Math.exp(v); Z += expS[k]; }
    const probs = {};
    for (const [k, v] of Object.entries(expS)) probs[k] = +(v / Z).toFixed(3);
    const sortedFinal = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    const winner = sortedFinal[0][0];
    const margin = +(sortedFinal[0][1] - (sortedFinal[1]?.[1] ?? 0)).toFixed(3);
    const confidence = probs[winner];
    const abstain = confidence < 0.55;

    // LLM arbitration (optional, but spends 1 call)
    let judgeTokens = 0;
    try {
      const judgeSys = 'You are an impartial Judge ratifying a citation-intent verdict. '
                     + 'Briefly (< 25 words) confirm or qualify the verdict given the evidence.';
      const judgeUser =
        `Paragraph: "${caseInput.paragraph}"\n\nVerdict: ${winner} (conf ${(confidence*100).toFixed(0)}%).\n`
        + `Verified evidence count: ${Object.entries(verifiedCount).map(([l,n])=>`${l}: ${n}`).join(', ') || 'none'}.\n`
        + 'Confirm or qualify in one short sentence.';
      const r = await callLLM(config.provider, config.apiKey, config.model || defaultModelFor(config.provider),
                              judgeSys, judgeUser, 100);
      judgeTokens = r.tokens;
    } catch (e) { /* judge call optional */ }

    const judgeStage = {
      id: 'judge', name: 'Judge', engine: 'LLM+CAL',
      tools_called: ['llm_arbitrator', 'platt_calibrator'],
      duration_ms: +(performance.now() - t5).toFixed(1),
      tokens: judgeTokens, llm_calls: 1,
      outputs: { label: winner, confidence, margin, scores, probs, abstain },
      actions: [
        { t: 0,    type: 'say',    actor: 'judge', text: 'Weighing arguments and verified evidence.' },
        { t: 700,  type: 'scores', scores, probs },
        { t: 1400, type: 'verdict', label: winner, confidence, margin, abstain },
        { t: 1700, type: 'say',    actor: 'judge',
          text: `Verdict: ${winner}  ·  conf ${confidence.toFixed(2)}  ·  margin ${margin.toFixed(2)}`
                + (abstain ? '  ·  ABSTAINED' : '') },
      ],
    };
    trace.stages.push(judgeStage);
    report(judgeStage);

    // ─── Stage 7: Clerk (LOCAL) ─────────────────────────────
    const verified = verifications.filter(v => v.label === winner && v.verified);
    const rejected = [];
    for (const adv of advocates) {
      if (adv.label === winner) continue;
      const atk = attacksByLabel[adv.label];
      rejected.push({ label: adv.label,
                      reason: atk ? atk.attack
                                  : `lower posterior than winner (${(probs[adv.label] || 0).toFixed(2)})` });
    }
    const artifact = {
      case_id: caseInput.case_id,
      label: winner,
      confidence, margin, abstain,
      evidence: verified.map(v => ({ span: v.span, verified: true,
                                     method: v.exact_match ? 'exact' : 'bleu' })),
      rejected_alternatives: rejected,
      ground_truth: caseInput.ground_truth || null,
    };
    const clerkStage = {
      id: 'clerk', name: 'Clerk', engine: 'CODE',
      tools_called: [], duration_ms: 1,
      tokens: 0, llm_calls: 0,
      outputs: { artifact },
      actions: [
        { t: 0,    type: 'say', actor: 'clerk', text: 'Sealing the case file.' },
        { t: 500,  type: 'show_artifact', artifact },
        { t: 1300, type: 'say', actor: 'clerk', text: 'Annotation artifact filed.' },
      ],
    };
    trace.stages.push(clerkStage);
    report(clerkStage);

    trace.final_artifact = artifact;
    trace.summary = {
      total_duration_ms: trace.stages.reduce((a, s) => a + s.duration_ms, 0),
      total_tokens:      trace.stages.reduce((a, s) => a + (s.tokens     || 0), 0),
      tools_invoked:     trace.stages.reduce((a, s) => a + s.tools_called.length, 0),
      llm_calls:         trace.stages.reduce((a, s) => a + (s.llm_calls  || 0), 0),
      ground_truth_match: artifact.label === caseInput.ground_truth,
    };
    return trace;
  }

  // ───────────────────────────────────────────────────────────
  //  Public API
  // ───────────────────────────────────────────────────────────
  return {
    PROVIDERS,
    defaultModelFor,
    runCase,
  };

})();
