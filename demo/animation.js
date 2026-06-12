/* ═══════════════════════════════════════════════════════════════
   JUROR · Workflow Demo · animation engine
   Plays back a JSON trace produced by ../prototype/juror.py as a
   left → right pipeline.  Four fixed intent lanes, one per intent.
   ═══════════════════════════════════════════════════════════════ */

const STAGE_BUFFER_MS = 700;       // pause between stages

// fixed mapping: each intent owns one lane (one sub-agent per intent)
const INTENT_SLOT = {
  'Background':      'BG',
  'Technical basis': 'TB',
  'Comparison':      'CP',
  'Fundamental idea':'FI',
};
const ALL_SLOTS = ['BG', 'TB', 'CP', 'FI'];
const SAY_STATIONS = ['triage', 'investigator', 'examiner', 'factchecker', 'judge', 'clerk'];
function slotOf(label) { return INTENT_SLOT[label] || null; }

// ---------------------------------------------------------------
//  Engine type helper
// ---------------------------------------------------------------
function engineChip(engine) {
  if (!engine) return '';
  const key = engine.toLowerCase();
  const classes = [];
  if (key.includes('ml'))    classes.push('ml');
  if (key.includes('rule'))  classes.push('rule');
  if (key.includes('graph')) classes.push('graph');
  if (key.includes('nli'))   classes.push('nli');
  if (key.includes('llm'))   classes.push('llm');
  if (key.includes('code') || key.includes('cal')) classes.push('code');
  const cls = classes.length ? classes[0] : 'code';
  return `<span class="engine-chip ${cls}">${engine}</span>`;
}

// ---------------------------------------------------------------
//  Main player
// ---------------------------------------------------------------
class JurorPlayer {
  constructor() {
    this.trace = null;
    this.flat = [];           // flattened actions with globalT
    this.idx  = 0;
    this.elapsedMs = 0;
    this.playing = false;
    this.lastFrame = null;
    this.speed = 1;
    this.cumLLMCalls = 0;
    this.cumTokens   = 0;
    this.cumTools    = 0;

    this.actors = ['triage', 'investigator',
                   'adv-BG', 'adv-TB', 'adv-CP', 'adv-FI',
                   'examiner', 'factchecker', 'judge', 'clerk'];
  }

  async loadCase(n) {
    this.stop();
    this.reset();
    try {
      const resp = await fetch(`traces/case_${String(n).padStart(3, '0')}.json`);
      if (!resp.ok) throw new Error(`status ${resp.status}`);
      this.trace = await resp.json();
      this.bindCaseInfo();
      this.flatten();
      this.updateClockTotal();
    } catch (err) {
      this.showError(`Could not load trace file <code>traces/case_${String(n).padStart(3, '0')}.json</code>.
                      Run <code>python -m http.server 8000</code> from the demo folder
                      and open http://localhost:8000  (file:// fetch is blocked).
                      <br><br>Error: ${err.message}`);
    }
  }

  reset() {
    this.idx = 0;
    this.elapsedMs = 0;
    this.lastFrame = null;
    this.cumLLMCalls = 0;
    this.cumTokens = 0;
    this.cumTools = 0;

    // clear actors (stations + intent lanes)
    for (const a of this.actors) {
      const el = document.getElementById('actor-' + a);
      if (!el) continue;
      el.classList.remove('active', 'dim', 'winner');
      el.style.opacity = '';
      const strike = el.querySelector('.strike-overlay');
      if (strike) strike.classList.remove('show');
      el.querySelectorAll('.stamp').forEach(s => s.remove());
    }
    // clear intent-lane arguments + strength chips
    ALL_SLOTS.forEach(s => {
      const arg = document.getElementById('arg-' + s);
      if (arg) { arg.textContent = ''; arg.classList.remove('show'); }
      const str = document.getElementById('str-' + s);
      if (str) { str.textContent = ''; str.classList.remove('show'); }
    });
    // clear station status lines
    SAY_STATIONS.forEach(k => {
      const s = document.getElementById('say-' + k);
      if (s) { s.textContent = ''; s.classList.remove('show'); }
    });
    const panelFrame = document.getElementById('station-panel');
    if (panelFrame) panelFrame.classList.remove('active', 'dim');
    document.querySelectorAll('.flow-arrow').forEach(ar => ar.classList.remove('lit', 'next'));

    // clear transcript
    document.getElementById('transcript-log').innerHTML = '';

    // meters
    this.setMeter('engine', '—');
    this.setMeter('tools', '0');
    this.setMeter('llm', '0');
    this.setMeter('tokens', '0');
    this.setMeter('stage', '—');
    const verdict = document.getElementById('m-verdict');
    verdict.textContent = 'pending…';
    verdict.classList.remove('final', 'abstain');

    document.getElementById('artifact-drawer').classList.add('hidden');
    document.getElementById('clock').textContent = '00:00.0';

    this.unsetButtons();
  }

  bindCaseInfo() {
    document.getElementById('case-id').textContent        = this.trace.case_id;
    document.getElementById('case-ref-input').value       = this.trace.input.cited_ref;
    document.getElementById('case-gt').textContent        = this.trace.input.ground_truth || '—';
    document.getElementById('case-paragraph-input').value = this.trace.input.paragraph;
    const err = document.getElementById('case-error');
    if (err) err.innerHTML = '';
  }

  flatten() {
    // give each action a global timeline timestamp
    this.flat = [];
    let stageStart = 0;
    for (const stage of this.trace.stages) {
      // stage entry pseudo-action (lights up the actor + updates meters)
      this.flat.push({
        globalT: stageStart,
        type: '_enter_stage',
        stageId: stage.id,
        stageName: stage.name,
        engine: stage.engine,
      });
      if (stage.id === 'panel') {
        this.flat.push({
          globalT: stageStart + 10,
          type: '_setup_advocates',
          advocates: stage.outputs.advocates,
        });
      }

      for (const action of stage.actions) {
        this.flat.push({
          globalT: stageStart + action.t,
          stageId: stage.id,
          stageEngine: stage.engine,
          ...action,
        });
      }

      // stage exit
      const maxT = stage.actions.length
        ? Math.max(...stage.actions.map(a => a.t))
        : 0;
      this.flat.push({
        globalT: stageStart + maxT + 200,
        type: '_exit_stage',
        stageId: stage.id,
        tools: stage.tools_called.length,
        tokens: stage.tokens,
        llmCalls: stage.llm_calls,
      });
      stageStart += maxT + STAGE_BUFFER_MS;
    }
    this.totalMs = stageStart;
  }

  // ---------------------------------------------------------------
  //  Playback loop
  // ---------------------------------------------------------------
  play() {
    if (!this.trace) return;
    if (this.idx >= this.flat.length) this.reset();
    this.playing = true;
    this.setButtons();
    this.lastFrame = null;
    requestAnimationFrame(this.loop.bind(this));
  }
  pause() {
    this.playing = false;
    this.lastFrame = null;
    this.unsetButtons();
  }
  stop() { this.pause(); }

  loop(ts) {
    if (!this.playing) return;
    if (this.lastFrame === null) this.lastFrame = ts;
    this.elapsedMs += (ts - this.lastFrame) * this.speed;
    this.lastFrame = ts;

    // fire all due actions
    while (this.idx < this.flat.length && this.flat[this.idx].globalT <= this.elapsedMs) {
      this.fireAction(this.flat[this.idx]);
      this.idx++;
    }

    document.getElementById('clock').textContent = this.formatTime(this.elapsedMs);

    if (this.idx >= this.flat.length) {
      this.playing = false;
      this.unsetButtons();
      return;
    }
    requestAnimationFrame(this.loop.bind(this));
  }

  // ---------------------------------------------------------------
  //  Action dispatcher
  // ---------------------------------------------------------------
  fireAction(a) {
    switch (a.type) {
      case '_enter_stage':     return this.enterStage(a);
      case '_exit_stage':      return this.exitStage(a);
      case '_setup_advocates': return this.setupAdvocates(a);
      case 'say':              return this.act_say(a);
      case 'tool_call':        return this.act_tool(a);
      case 'shortlist':        return this.act_shortlist(a);
      case 'advocate_speak':   return this.act_advocateSpeak(a);
      case 'attack':           return this.act_attack(a);
      case 'verify':           return this.act_verify(a);
      case 'scores':           return this.act_scores(a);
      case 'verdict':          return this.act_verdict(a);
      case 'show_artifact':    return this.act_showArtifact(a);
      default:
        console.warn('unknown action type', a.type, a);
    }
  }

  // -------- stage lights --------
  enterStage(a) {
    this._stageLLMSeen = 0;
    // dim everyone, hide every station status line
    for (const id of this.actors) {
      const el = document.getElementById('actor-' + id);
      if (el) { el.classList.remove('active'); el.classList.add('dim'); }
    }
    SAY_STATIONS.forEach(k => {
      const s = document.getElementById('say-' + k);
      if (s) s.classList.remove('show');
    });
    const panelFrame = document.getElementById('station-panel');
    if (panelFrame) panelFrame.classList.remove('active');

    let owner;
    switch (a.stageId) {
      case 'triage':       owner = 'triage'; break;
      case 'investigator': owner = 'investigator'; break;
      case 'panel':        owner = null; break;       // handled per-advocate
      case 'examiner':     owner = 'examiner'; break;
      case 'fact_checker': owner = 'factchecker'; break;
      case 'judge':        owner = 'judge'; break;
      case 'clerk':        owner = 'clerk'; break;
    }
    if (owner) {
      const el = document.getElementById('actor-' + owner);
      if (el) { el.classList.add('active'); el.classList.remove('dim'); }
    } else if (a.stageId === 'panel') {
      if (panelFrame) panelFrame.classList.add('active');
      // bring all four intent lanes up to base visibility
      ALL_SLOTS.forEach(s => {
        const el = document.getElementById('actor-adv-' + s);
        if (el) el.classList.remove('dim');
      });
    }
    // light the pipeline up to (and pulsing into) the current stage
    const order = { triage: 0, investigator: 1, panel: 2, examiner: 3,
                    fact_checker: 4, judge: 5, clerk: 6 };
    const here = order[a.stageId] ?? 0;
    document.querySelectorAll('.flow-arrow').forEach((ar, i) => {
      ar.classList.toggle('lit', i < here);
      ar.classList.toggle('next', i === here - 1);
    });

    this.setMeter('engine', a.engine);
    this.setMeter('stage', a.stageName);
    this.appendLog('system', '──── ' + a.stageName + ' ' + engineChip(a.engine) + ' ────', {asHtml: true});
  }

  exitStage(a) {
    // Reconcile per-stage LLM and token counts with the trace metadata.
    const expectedLLM = a.llmCalls || 0;
    const seenLLM = this._stageLLMSeen || 0;
    if (expectedLLM > seenLLM) {
      const diff = expectedLLM - seenLLM;
      this.cumLLMCalls += diff;
      this.setMeter('llm', this.cumLLMCalls);
      this.flashMeter('m-llm');
    }
    if (a.tokens > 0) {
      this.cumTokens += a.tokens;
      this.setMeter('tokens', this.cumTokens);
      this.flashMeter('m-tokens');
    }
    this._stageLLMSeen = 0;
  }

  setupAdvocates(a) {
    // intent lanes are static; just make sure their argument lines start empty
    ALL_SLOTS.forEach(s => {
      const arg = document.getElementById('arg-' + s);
      if (arg) { arg.textContent = ''; arg.classList.remove('show'); }
      const str = document.getElementById('str-' + s);
      if (str) { str.textContent = ''; str.classList.remove('show'); }
    });
  }

  // -------- speech (inline station status line) --------
  act_say(a) {
    if (SAY_STATIONS.includes(a.actor)) {
      this.showSay(a.actor, a.text);
    }
    this.appendLog(a.actor, a.text);
  }

  // -------- tool call --------
  act_tool(a) {
    this.cumTools++;
    this.setMeter('tools', this.cumTools);
    const isLLM = (a.engine || '').toUpperCase() === 'LLM';
    if (isLLM) {
      this._stageLLMSeen = (this._stageLLMSeen || 0) + 1;
      this.cumLLMCalls++;
      this.setMeter('llm', this.cumLLMCalls);
      this.flashMeter('m-llm');
    }
    const msg = `<code>${a.tool}</code> → ${a.summary || 'ok'}`;
    this.appendLog('tool', msg + ' ' + engineChip(a.engine), {asHtml: true});
  }

  // -------- panel convened --------
  act_shortlist(a) {
    this.appendLog('system',
      'Panel convened · one advocate per intent: <b>' + a.value.join(', ') + '</b>'
      + (a.fast_track ? ' <span style="color:#4ADE80">[fast-track candidate]</span>' : ''),
      {asHtml: true});
  }

  // -------- advocate speaks (fills its own intent lane) --------
  act_advocateSpeak(a) {
    const slot = slotOf(a.label);
    if (!slot) return;
    // highlight the speaking lane; others fall back to base visibility
    ALL_SLOTS.forEach(s => {
      const el = document.getElementById('actor-adv-' + s);
      if (!el) return;
      if (s === slot) { el.classList.add('active'); el.classList.remove('dim'); }
      else            { el.classList.remove('active'); el.classList.remove('dim'); }
    });
    const arg = document.getElementById('arg-' + slot);
    if (arg) { arg.textContent = a.text; arg.classList.add('show'); }
    const str = document.getElementById('str-' + slot);
    if (str && a.strength !== undefined) {
      str.textContent = 'raw strength ' + a.strength;
      str.classList.add('show');
    }
    this.appendLog('advocate', `<b>${a.label}</b>: ${a.text}`, {asHtml: true});
  }

  // -------- attack from cross-examiner --------
  act_attack(a) {
    const slot = slotOf(a.label);
    if (slot) {
      const card = document.getElementById('actor-adv-' + slot);
      if (card) {
        const strike = card.querySelector('.strike-overlay');
        if (strike) strike.classList.add('show');
        card.classList.add('dim');
      }
    }
    this.showSay('examiner', a.text);
    const el = document.getElementById('actor-examiner');
    if (el) { el.classList.add('active'); el.classList.remove('dim'); }
    this.appendLog('examiner',
      `attack <b>${a.label}</b>: ${a.text} <span class="engine-chip rule">${a.flag || 'flagged'}</span>`,
      {asHtml: true});
  }

  // -------- fact-checker verify (stamp on the lane) --------
  act_verify(a) {
    const slot = slotOf(a.label);
    if (slot) {
      const card = document.getElementById('actor-adv-' + slot);
      if (card) {
        // bring the lane under inspection forward so its stamp reads clearly
        card.classList.remove('dim');
        this.stampOn(card, a.verified);
      }
    }
    const fc = document.getElementById('actor-factchecker');
    if (fc) { fc.classList.add('active'); fc.classList.remove('dim'); }
    const passFail = a.verified
      ? '<span class="verify-pass">✓ verified</span>'
      : '<span class="verify-fail">✗ unverified</span>';
    const meta = a.exact ? 'exact match' : `bleu ${a.bleu}`;
    this.appendLog('factchecker',
      `${passFail} <b>[${a.label}]</b> &nbsp; <i>"${a.span}"</i> &nbsp;&nbsp; <span style="color:#9CA3AF">(${meta})</span>`,
      {asHtml: true});
  }

  // -------- judge scores breakdown --------
  act_scores(a) {
    const rows = Object.entries(a.probs)
      .sort((x, y) => y[1] - x[1])
      .map(([lbl, p]) => `<span style="display:inline-block;min-width:140px">${lbl}</span> ${'█'.repeat(Math.round(p * 20))} ${(p*100).toFixed(1)}%`);
    this.appendLog('judge',
      'Calibrated posterior:<br>' + rows.map(r => '&nbsp;&nbsp;' + r).join('<br>'),
      {asHtml: true});
  }

  // -------- judge verdict --------
  act_verdict(a) {
    const slot = slotOf(a.label);
    if (slot) {
      const card = document.getElementById('actor-adv-' + slot);
      if (card) { card.classList.add('winner'); card.classList.remove('dim'); }
    }
    // strike the losers
    ALL_SLOTS.forEach(s => {
      const card = document.getElementById('actor-adv-' + s);
      if (card && !card.classList.contains('winner')) {
        const strike = card.querySelector('.strike-overlay');
        if (strike) strike.classList.add('show');
        card.classList.add('dim');
      }
    });

    const verdictEl = document.getElementById('m-verdict');
    verdictEl.textContent = `${a.label}  ·  conf ${(a.confidence*100).toFixed(1)}%`;
    verdictEl.classList.add('final');
    if (a.abstain) verdictEl.classList.add('abstain');
  }

  // -------- clerk shows artifact --------
  act_showArtifact(a) {
    const drawer = document.getElementById('artifact-drawer');
    const pre    = document.getElementById('artifact-json');
    pre.textContent = JSON.stringify(a.artifact, null, 2);
    drawer.classList.remove('hidden');
    pre.scrollTop = 0;
  }

  // ---------------------------------------------------------------
  //  Visual primitives
  // ---------------------------------------------------------------
  showSay(stationKey, text) {
    const el = document.getElementById('say-' + stationKey);
    if (!el) return;
    el.textContent = text;
    void el.offsetWidth;        // reflow so the transition runs
    el.classList.add('show');
  }

  stampOn(card, verified) {
    const stamp = document.createElement('div');
    stamp.className = 'stamp ' + (verified ? 'pass' : 'fail');
    stamp.textContent = verified ? '✓ VERIFIED' : '✗ UNVERIFIED';
    stamp.style.top  = '6px';
    stamp.style.right = '6px';
    card.appendChild(stamp);
    setTimeout(() => stamp.remove(), 1700 / this.speed);
  }

  appendLog(actorClass, message, opts = {}) {
    const log = document.getElementById('transcript-log');
    const li = document.createElement('li');
    const ts = document.createElement('span');
    ts.className = 'ts';
    ts.textContent = this.formatTime(this.elapsedMs);
    const tag = document.createElement('span');
    tag.className = 'actor-tag ' + actorClass;
    tag.textContent = actorClass.toUpperCase();
    const msg = document.createElement('span');
    msg.className = 'msg' + (actorClass === 'tool' ? ' tool' : '');
    if (opts.asHtml) msg.innerHTML = message; else msg.textContent = message;

    li.appendChild(ts);
    li.appendChild(tag);
    li.appendChild(msg);
    log.appendChild(li);
    log.scrollTop = log.scrollHeight;
  }

  flashMeter(elementId) {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.classList.add('active');
    clearTimeout(el._flashTimer);
    el._flashTimer = setTimeout(() => el.classList.remove('active'), 500);
  }

  setMeter(name, value) {
    const map = { engine: 'm-engine', tools: 'm-tools', llm: 'm-llm',
                  tokens: 'm-tokens', stage: 'm-stage' };
    const el = document.getElementById(map[name]);
    if (el) el.textContent = value;
  }

  formatTime(ms) {
    const s = ms / 1000;
    const mm = Math.floor(s / 60);
    const ss = (s - mm * 60).toFixed(1);
    return `${String(mm).padStart(2,'0')}:${String(ss).padStart(4,'0')}`;
  }

  updateClockTotal() { /* hook for showing total length if desired */ }

  setButtons()   {
    document.getElementById('btn-play').disabled = true;
    document.getElementById('btn-pause').disabled = false;
  }
  unsetButtons() {
    document.getElementById('btn-play').disabled = false;
    document.getElementById('btn-pause').disabled = true;
  }

  showError(html) {
    const err = document.getElementById('case-error');
    if (err) err.innerHTML = `<div class="error-banner">${html}</div>`;
  }
}

// ═══════════════════════════════════════════════════════════════
//  Live-mode glue
// ═══════════════════════════════════════════════════════════════
const liveState = {
  mode:           'demo',          // 'demo' | 'live'
  provider:       'anthropic',
  apiKey:         '',
  model:          '',
  urlPreset:      null,            // e.g. 'google' from ?preset=google
  // Server-real mode wires the UI to a FastAPI backend that runs the agent
  // with real trained models (LR triage, NLI fact-checker).  Detected at
  // boot by GET-ing /api/status on the same origin.
  serverReady:    false,
  serverStatus:   null,             // last /api/status payload
  preferServer:   true,             // user can disable to force in-browser mode
};

// URL presets: ?preset=google → server-side Google + gemini-3.5-flash
const URL_PRESETS = {
  google: { provider: 'google', model: 'gemini-3.5-flash' },
};

// Built-in sample inputs (in case the user wants Live without an existing trace)
const LIVE_CASES = {
  1: {
    case_id: 'ID_183878',
    paragraph:
      'In general, simultaneous confidence bands for a function f are constructed ' +
      'by studying the asymptotic distribution of the sup. The approach of ' +
      'Bickel and Rosenblatt (1973) relates this to a study of the distribution ' +
      'of a Gaussian process. This approach to constructing confidence bands has ' +
      'been used in the context of nonparametric estimation by, among others, ' +
      'Hardle (1989) for M-estimators and Claeskens and Van Keilegom (2003) for ' +
      'local polynomial likelihood estimators.',
    ref_clean_citation: 'Hardle (1989)',
    ground_truth: 'Background',
  },
  2: {
    case_id: 'ID_99812',
    paragraph:
      'We follow the boosting framework of Buehlmann and Hothorn (2007), in which ' +
      'regularization is achieved indirectly via the application of penalized ' +
      'base learners. We extend their componentwise gradient boosting to handle ' +
      'conditional transformation models, but the iterative scheme and the ' +
      'stopping criterion remain as in the original work.',
    ref_clean_citation: 'Buehlmann and Hothorn (2007)',
    ground_truth: 'Technical basis',
  },
  3: {
    case_id: 'ID_77541',
    paragraph:
      'Our proposed kernel estimator achieves a parametric rate of convergence ' +
      'on subsets, which is faster than the non-parametric rate reported by ' +
      'Yao et al. (2005) under the sparsely observed regime. In contrast to ' +
      'their approach, our method does not require functional principal ' +
      'component analysis as a preprocessing step.',
    ref_clean_citation: 'Yao et al. (2005)',
    ground_truth: 'Comparison',
  },
};

function setApiStatus(level, text) {
  const dot = document.querySelector('#api-status .status-dot');
  const txt = document.getElementById('api-status-text');
  if (dot) { dot.className = 'status-dot ' + level; }
  if (txt) { txt.textContent = text; }
}

function applyMode(mode) {
  liveState.mode = mode;
  document.querySelectorAll('.mode-opt').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === mode);
  });
  if (mode === 'live' && !liveState.apiKey && !liveState.urlPreset) {
    document.getElementById('api-drawer').classList.add('open');
  }
}

async function probeServer() {
  // Try GET /api/status on the same origin; flip serverReady accordingly.
  try {
    const resp = await fetch('/api/status', { method: 'GET' });
    if (!resp.ok) throw new Error('status ' + resp.status);
    const j = await resp.json();
    liveState.serverReady  = true;
    liveState.serverStatus = j;
    return j;
  } catch (e) {
    liveState.serverReady  = false;
    liveState.serverStatus = null;
    return null;
  }
}

function applyUrlPreset(presetName) {
  const preset = URL_PRESETS[presetName];
  if (!preset) return false;

  liveState.urlPreset  = presetName;
  liveState.provider   = preset.provider;
  liveState.model      = preset.model;
  liveState.preferServer = true;

  document.querySelectorAll('.prov-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.provider === preset.provider);
  });
  const modelInput = document.getElementById('api-model-input');
  if (modelInput) {
    modelInput.placeholder = preset.model;
    modelInput.value       = preset.model;
  }

  applyMode('live');
  return true;
}

function describeServerStatus(j) {
  if (!j) return 'Server not detected · running in-browser mode.';
  const trAcc   = j.triage?.metrics?.accuracy;
  const trMsg   = j.triage?.ready
    ? `Triage = real TF-IDF + LR (held-out acc ${trAcc ?? '?'})`
    : 'Triage NOT loaded';
  const nliMsg  = j.fact_checker?.nli_available
    ? `NLI = ${j.fact_checker.model}`
    : 'NLI = BLEU fallback';
  const provOk  = Object.entries(j.llm_providers || {}).filter(([, v]) => v).map(([k]) => k);
  const provMsg = provOk.length
    ? `LLM keys ready: ${provOk.join(', ')}`
    : 'No LLM keys configured on server — set provider env vars.';
  return `${trMsg} · ${nliMsg} · ${provMsg}`;
}

async function runViaServer(caseInput, providerName, modelName) {
  const payload = { case: caseInput, provider: providerName };
  if (modelName) payload.model = modelName;
  const resp = await fetch('/api/run-case', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`server /api/run-case → ${resp.status}: ${body.slice(0, 200)}`);
  }
  return await resp.json();
}

function getCaseInputFromUI() {
  const paragraph = document.getElementById('case-paragraph-input').value.trim();
  const ref = document.getElementById('case-ref-input').value.trim();
  const caseId = document.getElementById('case-id').textContent;
  const gt = document.getElementById('case-gt').textContent;
  return {
    case_id: caseId === '—' ? 'CUSTOM' : caseId,
    paragraph,
    ref_clean_citation: ref,
    ground_truth: gt === '—' ? null : gt,
  };
}

function fillCaseFields(c) {
  document.getElementById('case-id').textContent = c.case_id;
  document.getElementById('case-ref-input').value = c.ref_clean_citation;
  document.getElementById('case-gt').textContent = c.ground_truth || '—';
  document.getElementById('case-paragraph-input').value = c.paragraph;
  const err = document.getElementById('case-error');
  if (err) err.innerHTML = '';
}

function selectCustomCase() {
  player.pause();
  player.reset();
  player.trace = null;
  document.getElementById('case-id').textContent = 'CUSTOM';
  document.getElementById('case-ref-input').value = '';
  document.getElementById('case-gt').textContent = '—';
  document.getElementById('case-paragraph-input').value = '';
  const err = document.getElementById('case-error');
  if (err) err.innerHTML = '';
  document.getElementById('case-paragraph-input').focus();
}

async function runLiveCase(caseInput) {
  if (!caseInput.paragraph || !caseInput.ref_clean_citation) {
    setApiStatus('error', 'Enter a paragraph and cited reference before running.');
    return;
  }
  player.pause(); player.reset();
  document.getElementById('btn-play').disabled = true;

  // 1) Server-real path (preferred) ────────────────────────────────
  if (liveState.preferServer && liveState.serverReady) {
    const provider = liveState.provider;
    const provOk   = liveState.serverStatus?.llm_providers?.[provider];
    if (!provOk) {
      setApiStatus('error',
        `Server has no key configured for '${provider}'. Ask the engineer to set the env var, or switch provider.`);
      document.getElementById('btn-play').disabled = false;
      return;
    }
    const model = liveState.model || null;
    const modelHint = model ? ` / ${model}` : '';
    setApiStatus('ok',
      `Server mode · calling ${provider}${modelHint} on the backend (real TF-IDF + NLI + LLM)…`);
    try {
      const trace = await runViaServer(caseInput, provider, model);
      setApiStatus('ok',
        `Server run complete · ${trace.summary.llm_calls} LLM calls · ${trace.summary.total_tokens} tokens.`);
      player.trace = trace;
      player.bindCaseInfo();
      player.flatten();
      player.idx = 0; player.elapsedMs = 0;
      player.play();
    } catch (e) {
      setApiStatus('error', `Server run failed: ${e.message.slice(0, 160)}`);
    } finally {
      document.getElementById('btn-play').disabled = false;
    }
    return;
  }

  // 2) In-browser fallback path ────────────────────────────────────
  if (!window.JUROR_LIVE) {
    setApiStatus('error', 'Live agent module failed to load.');
    document.getElementById('btn-play').disabled = false;
    return;
  }
  if (!liveState.apiKey) {
    setApiStatus('error',
      'No backend detected and no browser API key set. Either deploy the server or paste a key above.');
    document.getElementById('api-drawer').classList.add('open');
    document.getElementById('btn-play').disabled = false;
    return;
  }
  setApiStatus('ok',
    `Browser mode · calling ${liveState.provider} directly from your browser…`);
  try {
    const trace = await window.JUROR_LIVE.runCase({
      caseInput,
      config: { provider: liveState.provider, apiKey: liveState.apiKey, model: liveState.model },
    });
    setApiStatus('ok',
      `Browser run complete · ${trace.summary.llm_calls} LLM calls · ${trace.summary.total_tokens} tokens.`);
    player.trace = trace;
    player.bindCaseInfo();
    player.flatten();
    player.idx = 0; player.elapsedMs = 0;
    player.play();
  } catch (e) {
    setApiStatus('error', `Browser run failed: ${e.message.slice(0, 160)}`);
  } finally {
    document.getElementById('btn-play').disabled = false;
  }
}

// ═══════════════════════════════════════════════════════════════
//  Wire up UI
// ═══════════════════════════════════════════════════════════════
const player = new JurorPlayer();

document.addEventListener('DOMContentLoaded', async () => {
  player.loadCase(1);

  // Detect whether a real server is reachable on the same origin.
  // If yes — Live mode prefers the server (real TF-IDF + NLI + LLM).
  // If no  — Live mode falls back to in-browser direct calls.
  const status = await probeServer();

  const urlPreset = new URLSearchParams(window.location.search).get('preset');
  const presetApplied = urlPreset && applyUrlPreset(urlPreset);

  if (status && status.triage?.ready) {
    const liveBtn = document.querySelector('.mode-opt[data-mode="live"]');
    if (liveBtn) liveBtn.innerHTML =
      '<span class="mode-dot live-dot"></span> Live <span class="badge-no-llm" style="margin-left:6px">server</span>';

    if (presetApplied) {
      const provOk = status.llm_providers?.[liveState.provider];
      if (provOk) {
        setApiStatus('ok',
          `Preset "${urlPreset}" · server ${liveState.provider} / ${liveState.model} ready. Click Run.`);
      } else {
        setApiStatus('error',
          `Preset "${urlPreset}" needs ${liveState.provider.toUpperCase()}_API_KEY on the server.`);
      }
    } else {
      setApiStatus('ok', describeServerStatus(status));
    }
  } else {
    if (presetApplied) {
      setApiStatus('error',
        `Preset "${urlPreset}" requires the backend server — start uvicorn in server/.`);
    } else {
      setApiStatus('pending',
        'No backend detected on this origin. Live mode will run in-browser using the key you paste below.');
    }
  }

  // ─── Play / Pause / Restart ───
  document.getElementById('btn-play').addEventListener('click', () => {
    if (liveState.mode === 'live') {
      runLiveCase(getCaseInputFromUI());
    } else {
      player.play();
    }
  });
  document.getElementById('btn-pause').addEventListener('click', () => player.pause());
  document.getElementById('btn-restart').addEventListener('click', () => {
    player.pause(); player.reset();
  });

  // ─── Speed ───
  document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      player.speed = parseFloat(btn.dataset.speed);
    });
  });

  // ─── Case selector ───
  document.querySelectorAll('.case-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      document.querySelectorAll('.case-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (btn.dataset.case === 'custom') {
        selectCustomCase();
        return;
      }
      if (liveState.mode === 'demo') {
        await player.loadCase(parseInt(btn.dataset.case));
      } else {
        const c = LIVE_CASES[parseInt(btn.dataset.case)];
        player.reset();
        player.trace = { case_id: c.case_id,
                         input: { paragraph: c.paragraph, cited_ref: c.ref_clean_citation,
                                  ground_truth: c.ground_truth } };
        fillCaseFields(c);
      }
    });
  });

  // ─── Mode switch ───
  document.querySelectorAll('.mode-opt').forEach(btn => {
    btn.addEventListener('click', () => applyMode(btn.dataset.mode));
  });

  // ─── Settings drawer toggle ───
  const drawer = document.getElementById('api-drawer');
  document.getElementById('btn-settings').addEventListener('click', () => {
    drawer.classList.toggle('open');
  });

  // ─── Provider selector ───
  const modelInput = document.getElementById('api-model-input');
  document.querySelectorAll('.prov-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.prov-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      liveState.provider = btn.dataset.provider;
      if (window.JUROR_LIVE) {
        const model = window.JUROR_LIVE.defaultModelFor(liveState.provider);
        modelInput.placeholder = model;
        if (!modelInput.value) modelInput.value = model;
      }
    });
  });
  // initialize default model
  if (window.JUROR_LIVE) {
    modelInput.placeholder = window.JUROR_LIVE.defaultModelFor(liveState.provider);
    modelInput.value       = window.JUROR_LIVE.defaultModelFor(liveState.provider);
  }

  // ─── Activate key ───
  document.getElementById('btn-save-key').addEventListener('click', () => {
    const key   = document.getElementById('api-key-input').value.trim();
    const model = modelInput.value.trim();
    if (!key) {
      setApiStatus('error', 'Paste an API key above first.');
      return;
    }
    liveState.apiKey = key;
    liveState.model  = model || window.JUROR_LIVE.defaultModelFor(liveState.provider);
    // wipe the password field so it isn't sitting in the DOM after activation
    document.getElementById('api-key-input').value = '';
    setApiStatus('ok',
      `Key active for ${liveState.provider} / ${liveState.model}. Click Run to make real LLM calls.`);
    applyMode('live');
  });

  // ─── Keyboard shortcuts ───
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === ' ' || e.key === 'k') {
      e.preventDefault();
      if (liveState.mode === 'live') {
        document.getElementById('btn-play').click();
      } else {
        player.playing ? player.pause() : player.play();
      }
    } else if (e.key === 'r') {
      player.pause(); player.reset();
    } else if (e.key === '1' || e.key === '2' || e.key === '3') {
      document.querySelector(`.case-btn[data-case="${e.key}"]`).click();
    }
  });
});
