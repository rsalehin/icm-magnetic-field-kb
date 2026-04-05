// ── State ───────────────────────────────────────────────────────────────────
// Evidence registry — keyed by message UID to avoid cross-message collisions
const evidenceRegistry = {};
let currentConvId    = null;
let conversations    = [];
let evidencePapers   = new Set();
let evidenceDetails  = [];
let graphData        = null;
let simulation       = null;
let svg              = null;
let zoomBehavior     = null;
let gContainer       = null;
let showEdgeTypes    = { cites: true, co_cited: true };
let graphInited      = false;
let queryGraphMode   = false;

// ── Tab switching ────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach((b, i) => {
    b.classList.toggle('active', ['chat','graph','stats'][i] === name);
  });
  document.querySelectorAll('.tab-content').forEach(c => {
    c.classList.toggle('active', c.id === name + '-tab');
  });
  if (name === 'graph' && !graphInited) initGraph();
  if (name === 'stats') loadStats();
}

// ── Conversation management ──────────────────────────────────────────────────
async function loadConversations() {
  const res  = await fetch('/api/conversations');
  conversations = await res.json();
  renderConvList();
}

function renderConvList() {
  const el = document.getElementById('conv-list');
  el.innerHTML = conversations.map(c => `
    <div class="conv-item ${c.id === currentConvId ? 'active' : ''}"
         onclick="loadConversation('${c.id}')">
      <button class="conv-delete" onclick="deleteConv(event,'${c.id}')">×</button>
      <div class="conv-title">${escHtml(c.title)}</div>
      <div class="conv-meta">${c.message_count} msgs · ${formatDate(c.updated_at)}</div>
    </div>
  `).join('');
}

async function loadConversation(convId) {
  currentConvId = convId;
  renderConvList();
  const msgs = await fetch(`/api/conversation/${convId}`).then(r => r.json());
  const container = document.getElementById('messages');
  container.innerHTML = '';
  msgs.forEach(m => {
    const ev = m.pipeline_trace?.evidence || [];
    appendMessage(m.role, m.content, m.intent, m.abstained, m.pipeline_trace, ev);
  });
  container.scrollTop = container.scrollHeight;
}

function newConversation() {
  currentConvId = null;
  const container = document.getElementById('messages');
  container.innerHTML = '';
  const welcome = document.createElement('div');
  welcome.id = 'welcome';
  welcome.innerHTML = `
    <h2>Intracluster Magnetic Fields</h2>
    <p>Query 251 astrophysics papers with grounded evidence and citations.</p>
    <div class="welcome-suggestions">
      <button class="suggestion-btn" onclick="askSuggestion(this)">What spectral index did Murgia 2004 find for the magnetic field power spectrum in Abell 119?</button>
      <button class="suggestion-btn" onclick="askSuggestion(this)">How does the GRF magnetic field model differ from MHD cosmological simulations?</button>
      <button class="suggestion-btn" onclick="askSuggestion(this)">What is the typical central magnetic field strength B₀ observed in galaxy clusters?</button>
      <button class="suggestion-btn" onclick="askSuggestion(this)">What papers studied Faraday rotation in the Coma cluster?</button>
    </div>
  `;
  container.appendChild(welcome);
  renderConvList();
}

async function deleteConv(e, convId) {
  e.stopPropagation();
  await fetch(`/api/conversation/${convId}`, { method: 'DELETE' });
  if (currentConvId === convId) newConversation();
  await loadConversations();
}

// ── Math + text renderer ─────────────────────────────────────────────────────
function renderMathAndText(text) {
  /**
   * Pipeline:
   * 1. Extract math blocks to protect them from markdown parser
   * 2. Run markdown renderer
   * 3. Restore math blocks and render with KaTeX
   */

  // Step 1 — extract math into placeholders
  const mathBlocks = [];
  let protected_text = text;

  // Display math $$...$$ first
  protected_text = protected_text.replace(/\$\$([\s\S]+?)\$\$/g, (match, math) => {
    mathBlocks.push({ type: 'display', math: math.trim() });
    return `%%MATH_DISPLAY_${mathBlocks.length - 1}%%`;
  });

  // Inline math $...$
  protected_text = protected_text.replace(/\$([^$\n]+?)\$/g, (match, math) => {
    mathBlocks.push({ type: 'inline', math: math.trim() });
    return `%%MATH_INLINE_${mathBlocks.length - 1}%%`;
  });

  // Step 2 — render markdown
  marked.setOptions({
    breaks:   true,   // \n → <br>
    gfm:      true,   // GitHub flavored markdown
  });

  let html = marked.parse(protected_text);

  // Step 3 — restore math blocks and render with KaTeX
  html = html.replace(/%%MATH_DISPLAY_(\d+)%%/g, (match, idx) => {
    const block = mathBlocks[parseInt(idx)];
    try {
      return katex.renderToString(block.math, {
        displayMode:  true,
        throwOnError: false,
      });
    } catch(e) {
      return `<code>${escHtml(block.math)}</code>`;
    }
  });

  html = html.replace(/%%MATH_INLINE_(\d+)%%/g, (match, idx) => {
    const block = mathBlocks[parseInt(idx)];
    try {
      return katex.renderToString(block.math, {
        displayMode:  false,
        throwOnError: false,
      });
    } catch(e) {
      return `<code>${escHtml(block.math)}</code>`;
    }
  });

  return html;
}

// ── Messaging ────────────────────────────────────────────────────────────────
async function sendQuestion() {
  const input = document.getElementById('question-input');
  const q     = input.value.trim();
  if (!q) return;

  input.value = '';
  input.style.height = '';

  if (genMode === 'export') {
    await sendExportMode(q);
  } else {
    await sendBuiltinMode(q);
  }
}

async function sendBuiltinMode(q) {
  const container = document.getElementById('messages');
  const welcome   = document.getElementById('welcome');
  if (welcome && welcome.parentNode === container) container.removeChild(welcome);

  appendMessage('user', q);

  const thinkEl = document.createElement('div');
  thinkEl.className = 'message assistant';

  let elapsed = 0;
  const timer = setInterval(() => {
    elapsed++;
    const timeEl = thinkEl.querySelector('#think-elapsed');
    if (timeEl) timeEl.textContent = `${elapsed}s`;
  }, 1000);

  thinkEl.innerHTML = `
    <div class="message-role">SYSTEM</div>
    <div class="thinking">
      <div class="thinking-dots"><span></span><span></span><span></span></div>
      retrieving evidence &amp; generating... <span id="think-elapsed" style="color:var(--amber);margin-left:6px">0s</span>
    </div>
  `;
  container.appendChild(thinkEl);
  container.scrollTop = container.scrollHeight;

  setStatus('querying...');
  document.getElementById('send-btn').disabled = true;

  try {
    const res  = await fetch('/api/query', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ question: q, conversation_id: currentConvId }),
    });
    const data = await res.json();
    clearInterval(timer);
    container.removeChild(thinkEl);
    currentConvId = data.conversation_id;

    evidenceDetails = data.pipeline_trace?.evidence || [];
    evidencePapers  = new Set(evidenceDetails.map(e => e.paper_id));

    appendMessage(
      'assistant', data.answer, data.intent,
      data.abstained, data.pipeline_trace, evidenceDetails
    );

    updateGraphEvidence();
    updateQueryStats(evidenceDetails);
    await loadConversations();
    setStatus('ready');

  } catch (err) {
    clearInterval(timer);
    container.removeChild(thinkEl);
    appendMessage('assistant', '⚠ Error: ' + err.message);
    setStatus('error');
  }

  document.getElementById('send-btn').disabled = false;
  container.scrollTop = container.scrollHeight;
}

function appendMessage(role, content, intent, abstained, trace, evidence) {
  const container = document.getElementById('messages');
  const el        = document.createElement('div');
  el.className    = `message ${role}`;

  const roleLabel    = role === 'user' ? 'YOU' : 'ASSISTANT';
  const intentBadge  = intent
    ? `<span class="intent-badge intent-${intent}">${intent}</span>` : '';
  const abstainBadge = abstained
    ? `<span class="abstain-badge">ABSTAINED</span> ` : '';

  const ev = evidence || [];

  // Register evidence for this message under a unique key
  const msgKey = 'msg_' + Math.random().toString(36).slice(2);
  evidenceRegistry[msgKey] = ev;

  let rendered;

  if (role === 'user') {
    // User messages — plain escaped text only, no markdown
    rendered = escHtml(content);
  } else {
    // Assistant messages — full markdown + math + citation links
    rendered = renderMathAndText(content).replace(
      /\[E(\d+)\]/g,
      (match, num) => {
        const idx  = parseInt(num) - 1;
        const item = ev[idx];
        if (item) {
          const page = item.page_num || 1;
          const searchTerm = (item.text_snippet || '')
            .replace(/[^\w\s]/g, ' ')
            .split(/\s+/)
            .filter(w => w.length > 3)
            .slice(0, 6)
            .join(' ');
          const url = '/paper-viewer'
          + '?chunk_id=' + encodeURIComponent(item.chunk_id)
          + '&page='     + page
          + '&snippet='  + encodeURIComponent((item.text_snippet || '').slice(0, 150))
          + '&year='     + (item.year || '')
          + '&title='    + encodeURIComponent(item.title || '')
          + '&paper_id=' + encodeURIComponent(item.paper_id);

          return `<a class="cite-ref"
            href="${url}"
            target="_blank"
            rel="noopener"
            data-msgkey="${msgKey}"
            data-idx="${idx}"
            onmouseenter="showCiteTooltipByKey(this,event)"
            onmouseleave="hideCiteTooltip()">[E${num}]</a>`;
        }
        return `<span style="color:var(--amber);font-weight:500">[E${num}]</span>`;
      }
    );
  }

  const traceHtml = (trace && role === 'assistant') ? buildTraceHtml(trace, ev) : '';

  el.innerHTML = `
    <div class="message-role">${roleLabel}${intentBadge}</div>
    <div class="message-content">${abstainBadge}${rendered}</div>
    ${traceHtml}
  `;
  container.appendChild(el);
}

function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const toggle  = document.getElementById('sidebar-toggle');
  const collapsed = sidebar.classList.toggle('collapsed');
  toggle.textContent = collapsed ? '›' : '‹';
  toggle.style.left  = collapsed ? '0px' : '240px';
}

function buildTraceHtml(trace, evidence) {
  const p  = trace.planner   || {};
  const r  = trace.retriever || {};
  const rk = trace.reranker  || {};
  const a  = trace.assembler || {};
  const ev = evidence        || [];

  const uid = 'tr_' + Math.random().toString(36).slice(2);

  return `
    <div class="pipeline-toggle">
      <button class="pipeline-toggle-btn" onclick="togglePipeline('${uid}')">
        ▸ pipeline trace
      </button>
    </div>
    <div class="pipeline-panel" id="${uid}">
      <div class="pipeline-section">
        <div class="pipeline-section-title">Planner</div>
        <div class="pipeline-kv">
          <span class="pipeline-k">intent</span>
          <span class="pipeline-v">${p.intent||'—'}</span>
          <span class="pipeline-k">mode</span>
          <span class="pipeline-v">${p.synthesis_mode||'—'}</span>
        </div>
      </div>
      <div class="pipeline-section">
        <div class="pipeline-section-title">Retriever</div>
        <div class="pipeline-kv">
          <span class="pipeline-k">paper_k</span>
          <span class="pipeline-v">${p.budgets?.paper_k||'—'}</span>
          <span class="pipeline-k">chunks fused</span>
          <span class="pipeline-v">${r.chunks_fused||'—'}</span>
        </div>
      </div>
      <div class="pipeline-section">
        <div class="pipeline-section-title">Reranker</div>
        <div class="pipeline-kv">
          <span class="pipeline-k">input → output</span>
          <span class="pipeline-v">${rk.input_chunks||'—'} → ${rk.output_chunks||'—'}</span>
          <span class="pipeline-k">distinct papers</span>
          <span class="pipeline-v">${rk.distinct_papers||'—'}</span>
          <span class="pipeline-k">max score</span>
          <span class="pipeline-v" style="color:var(--green)">${(rk.max_rerank_score||0).toFixed(4)}</span>
        </div>
      </div>
      <div class="pipeline-section">
        <div class="pipeline-section-title">Assembler</div>
        <div class="pipeline-kv">
          <span class="pipeline-k">deduplicated</span>
          <span class="pipeline-v">${a.deduplicated||0}</span>
          <span class="pipeline-k">disagreement</span>
          <span class="pipeline-v" style="color:${a.has_disagreement?'var(--amber)':'var(--text-dim)'}">${a.has_disagreement?'YES':'no'}</span>
          <span class="pipeline-k">abstain</span>
          <span class="pipeline-v" style="color:${a.abstain?'var(--red)':'var(--text-dim)'}">${a.abstain?'YES':'no'}</span>
        </div>
      </div>
      ${ev.length > 0 ? `
      <div class="pipeline-section">
        <div class="pipeline-section-title">Evidence</div>
        ${ev.slice(0,8).map(e => `
          <div class="evidence-item">
            <span class="evidence-rank">[E${e.rank}]</span>
            <span class="evidence-score">${(e.rerank_score||0).toFixed(3)}</span>
            <div style="font-family:var(--mono);font-size:10px;color:var(--cyan);margin:3px 0">
              ${escHtml(e.title || e.paper_id || '—')}
            </div>
            <div style="font-family:var(--mono);font-size:9px;color:var(--text-dim);margin-bottom:2px">
              ${escHtml((e.paper_id||'').replace('arxiv:',''))} · ${e.year||'?'} · §${escHtml(e.section||'')}
            </div>
            <div class="evidence-snippet">${escHtml(e.text_snippet||'')}</div>
          </div>
        `).join('')}
      </div>` : ''}
    </div>
  `;
}

function togglePipeline(uid) {
  const el  = document.getElementById(uid);
  el.classList.toggle('open');
  const btn = el.previousElementSibling.querySelector('button');
  btn.textContent = el.classList.contains('open')
    ? '▾ pipeline trace' : '▸ pipeline trace';
}

function askSuggestion(btn) {
  document.getElementById('question-input').value = btn.textContent.trim();
  sendQuestion();
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuestion(); }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

// ── Citation tooltip ─────────────────────────────────────────────────────────
function showCiteTooltipByKey(el, event) {
  const key  = el.dataset.msgkey;
  const idx  = parseInt(el.dataset.idx);
  const item = (evidenceRegistry[key] || [])[idx];
  if (!item) return;

  document.getElementById('ct-rank').textContent    = `[E${idx + 1}]`;
  document.getElementById('ct-title').textContent   = item.title   || '—';
  document.getElementById('ct-year').textContent    = item.year    || '—';
  document.getElementById('ct-journal').textContent = item.journal || '—';
  document.getElementById('ct-section').textContent = item.section || '—';
  document.getElementById('ct-id').textContent      = item.paper_id || '—';

  const x = Math.min(event.clientX + 12, window.innerWidth  - 360);
  const y = Math.min(event.clientY + 12, window.innerHeight - 200);
  const tip = document.getElementById('cite-tooltip');
  tip.style.left = x + 'px';
  tip.style.top  = y + 'px';
  tip.classList.add('visible');
}

function hideCiteTooltip() {
  document.getElementById('cite-tooltip').classList.remove('visible');
}

// ── Tooltip hover persistence ─────────────────────────────────────────────────
let _tooltipHovered = false;

document.addEventListener('DOMContentLoaded', () => {
  const tipEl = document.getElementById('cite-tooltip');

  tipEl.addEventListener('mouseenter', () => {
    _tooltipHovered = true;
  });
  tipEl.addEventListener('mouseleave', () => {
    _tooltipHovered = false;
    hideCiteTooltip();
  });
});

// ── Delegated citation events ─────────────────────────────────────────────────
document.addEventListener('mouseover', e => {
  const ref = e.target.closest('.cite-ref');
  if (ref) showCiteTooltipByKey(ref, e);
});

document.addEventListener('mouseout', e => {
  const ref = e.target.closest('.cite-ref');
  if (ref) {
    setTimeout(() => {
      if (!_tooltipHovered) hideCiteTooltip();
    }, 200);
  }
});

function hideCiteTooltip() {
  document.getElementById('cite-tooltip').classList.remove('visible');
}

// ── Graph ────────────────────────────────────────────────────────────────────
async function initGraph() {
  graphInited = true;
  const res  = await fetch('/api/graph');
  graphData  = await res.json();
  document.getElementById('graph-loading').style.display = 'none';
  renderGraph();
}

function renderGraph() {
  if (!graphData) return;
  const container = document.getElementById('graph-container');
  const W = container.clientWidth;
  const H = container.clientHeight;

  svg = d3.select('#graph-svg');
  svg.selectAll('*').remove();

  zoomBehavior = d3.zoom()
    .scaleExtent([0.1, 8])
    .on('zoom', e => gContainer.attr('transform', e.transform));
  svg.call(zoomBehavior);
  gContainer = svg.append('g');

  if (queryGraphMode && evidencePapers.size > 0) {
    renderQueryGraph(W, H);
  } else {
    renderCorpusLanes(W, H);
  }
}

// ── Query mode — force graph of evidence + neighbours ────────────────────────
function renderQueryGraph(W, H) {
  // Build node set: evidence + 1-hop neighbours
  const visibleIds = new Set(evidencePapers);
  graphData.edges.forEach(e => {
    const src = e.source?.id || e.source;
    const tgt = e.target?.id || e.target;
    if (evidencePapers.has(src)) visibleIds.add(tgt);
    if (evidencePapers.has(tgt)) visibleIds.add(src);
  });

  const nodes = graphData.nodes.filter(n => visibleIds.has(n.id));
  const nodeSet = new Set(nodes.map(n => n.id));
  // In query mode only show CITES edges — co-citation causes instability
  const edges = graphData.edges.filter(e => {
    const src = e.source?.id || e.source;
    const tgt = e.target?.id || e.target;
    return nodeSet.has(src) && nodeSet.has(tgt) && e.type !== 'co_cited';
  });

  const simNodes = nodes.map(n => ({ ...n }));
  const nodeById = {};
  simNodes.forEach(n => nodeById[n.id] = n);

  const simEdges = edges.map(e => ({
    ...e,
    source: nodeById[e.source?.id || e.source],
    target: nodeById[e.target?.id || e.target],
  })).filter(e => e.source && e.target);

  // Arrows
  svg.append('defs').append('marker')
    .attr('id', 'arrow')
    .attr('viewBox', '0 -4 8 8')
    .attr('refX', 18).attr('refY', 0)
    .attr('markerWidth', 5).attr('markerHeight', 5)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-4L8,0L0,4')
    .attr('fill', '#1a3050');

  // Links
  const link = gContainer.append('g')
    .selectAll('line').data(simEdges).join('line')
    .attr('stroke',         d => d.type === 'co_cited' ? '#1a3045' : '#1e3555')
    .attr('stroke-width',   d => d.type === 'co_cited' ? 1 : 1.5)
    .attr('stroke-dasharray', d => d.type === 'co_cited' ? '3,3' : null)
    .attr('marker-end',     d => d.type !== 'co_cited' ? 'url(#arrow)' : null);

  // Nodes
  const isEvidence = d => evidencePapers.has(d.id);

  const node = gContainer.append('g')
    .selectAll('circle').data(simNodes).join('circle')
    .attr('r',            d => isEvidence(d) ? 14 : 8)
    .attr('fill',         d => isEvidence(d) ? 'var(--amber)' : '#1a3050')
    .attr('stroke',       d => isEvidence(d) ? '#ffcc60' : '#2a4570')
    .attr('stroke-width', d => isEvidence(d) ? 2 : 1)
    .style('cursor', 'pointer')
    .on('click', (e, d) => showNodeInfo(d))
    .on('mouseover', function(e, d) {
      d3.select(this).attr('stroke', 'var(--cyan)').attr('stroke-width', 2);
    })
    .on('mouseout', function(e, d) {
      d3.select(this)
        .attr('stroke',       isEvidence(d) ? '#ffcc60' : '#2a4570')
        .attr('stroke-width', isEvidence(d) ? 2 : 1);
    })
    .call(d3.drag()
      .on('start', (e,d) => { if(!e.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
      .on('drag',  (e,d) => { d.fx=e.x; d.fy=e.y; })
      .on('end',   (e,d) => { if(!e.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; })
    );

  // Labels — always visible in query mode
  const label = gContainer.append('g')
    .selectAll('text').data(simNodes).join('text')
    .text(d => {
      const id = d.id.replace('arxiv:','').replace('astro-ph/','');
      return isEvidence(d)
        ? (d.title || id).slice(0, 22)
        : id.slice(0, 14);
    })
    .attr('font-family', "'DM Mono', monospace")
    .attr('font-size',   d => isEvidence(d) ? '9px' : '7px')
    .attr('fill',        d => isEvidence(d) ? 'var(--amber)' : 'var(--text-dim)')
    .attr('dy',          d => -(isEvidence(d) ? 18 : 12))
    .attr('text-anchor', 'middle')
    .style('pointer-events', 'none');

  // Rank badges on evidence nodes
  const rank = gContainer.append('g')
    .selectAll('text')
    .data(simNodes.filter(n => evidencePapers.has(n.id)))
    .join('text')
    .text(d => {
      const ev = [...evidencePapers].indexOf(d.id) + 1;
      return `E${ev}`;
    })
    .attr('font-family', "'DM Mono', monospace")
    .attr('font-size',   '8px')
    .attr('font-weight', '600')
    .attr('fill',        '#000')
    .attr('text-anchor', 'middle')
    .attr('dy',          '4px')
    .style('pointer-events', 'none');

  simulation = d3.forceSimulation(simNodes)
    .force('link',      d3.forceLink(simEdges).id(d => d.id).distance(100).strength(0.3))
    .force('charge',    d3.forceManyBody().strength(-300))
    .force('center',    d3.forceCenter(W/2, H/2))
    .force('collision', d3.forceCollide(d => isEvidence(d) ? 30 : 20))
    .alphaDecay(0.05)      // settle faster (default 0.0228)
    .alphaMin(0.001)       // stop sooner
    .on('tick', () => {
      link
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      node.attr('cx', d => d.x).attr('cy', d => d.y);
      label.attr('x', d => d.x).attr('y', d => d.y);
      rank.attr('x', d => d.x).attr('y', d => d.y);
    })
    .on('end', () => {
      // Pin all nodes once settled — no more movement
      simNodes.forEach(d => { d.fx = d.x; d.fy = d.y; });
    });
}

// ── Full corpus mode — static year-lane layout ───────────────────────────────
function renderCorpusLanes(W, H) {
  if (simulation) { simulation.stop(); simulation = null; }

  const eras = [
    { label: '1995–1999', min: 1995, max: 1999, color: '#2d5a3d' },
    { label: '2000–2004', min: 2000, max: 2004, color: '#3d6b2d' },
    { label: '2005–2009', min: 2005, max: 2009, color: '#6b7a2d' },
    { label: '2010–2014', min: 2010, max: 2014, color: '#7a5c2d' },
    { label: '2015–2019', min: 2015, max: 2019, color: '#7a3d2d' },
    { label: '2020–2026', min: 2020, max: 2026, color: '#5a2d7a' },
  ];

  const laneH    = H / eras.length;
  const paddingX = 60;
  const paddingY = 14;

  // Draw lane backgrounds
  eras.forEach((era, i) => {
    gContainer.append('rect')
      .attr('x', 0).attr('y', i * laneH)
      .attr('width', W).attr('height', laneH)
      .attr('fill', i % 2 === 0 ? '#090c14' : '#07090f')
      .attr('opacity', 0.8);

    gContainer.append('text')
      .attr('x', 8).attr('y', i * laneH + 16)
      .text(era.label)
      .attr('font-family', "'Barlow Condensed', sans-serif")
      .attr('font-size', '11px')
      .attr('font-weight', '600')
      .attr('fill', era.color)
      .attr('opacity', 0.8);

    // Lane separator
    gContainer.append('line')
      .attr('x1', 0).attr('y1', i * laneH)
      .attr('x2', W).attr('y2', i * laneH)
      .attr('stroke', '#1a2035').attr('stroke-width', 1);
  });

  // Assign nodes to lanes and position
  const nodePositions = {};
  eras.forEach((era, laneIdx) => {
    const laneNodes = graphData.nodes
      .filter(n => n.year >= era.min && n.year <= era.max)
      .sort((a, b) => (b.citation_count || 0) - (a.citation_count || 0));

    const usableW  = W - paddingX * 2;
    const cols     = Math.ceil(Math.sqrt(laneNodes.length * (usableW / laneH)));
    const cellW    = usableW / Math.max(cols, 1);
    const rowH     = (laneH - paddingY * 2) / Math.max(Math.ceil(laneNodes.length / cols), 1);

    laneNodes.forEach((n, i) => {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const x   = paddingX + col * cellW + cellW / 2;
      const y   = laneIdx * laneH + paddingY + row * rowH + rowH / 2;
      nodePositions[n.id] = { x, y };
    });
  });

  // Draw nodes
  const nodeData = graphData.nodes.filter(n => nodePositions[n.id]);

  const nodeG = gContainer.append('g')
    .selectAll('g').data(nodeData).join('g')
    .attr('transform', d => {
      const p = nodePositions[d.id];
      return `translate(${p.x},${p.y})`;
    })
    .style('cursor', 'pointer')
    .on('click', (e, d) => showNodeInfo(d));

  nodeG.append('circle')
    .attr('r', d => {
      const isEv = evidencePapers.has(d.id);
      const size = Math.max(3, Math.min(10, (d.citation_count || 0) / 50));
      return isEv ? size + 4 : size;
    })
    .attr('fill',         d => evidencePapers.has(d.id) ? 'var(--amber)' : d.color)
    .attr('stroke',       d => evidencePapers.has(d.id) ? '#ffcc60' : 'none')
    .attr('stroke-width', 2)
    .attr('opacity',      d => evidencePapers.has(d.id) ? 1 : 0.7)
    .on('mouseover', function(e, d) {
      d3.select(this).attr('opacity', 1).attr('stroke', 'var(--cyan)').attr('stroke-width', 1.5);
      // Show tooltip with title
      const tip = gContainer.append('g').attr('class', 'node-tip')
        .attr('transform', `translate(${nodePositions[d.id].x + 10},${nodePositions[d.id].y - 10})`);
      tip.append('rect')
        .attr('x', 0).attr('y', -12)
        .attr('width', Math.min(d.title.length * 5.5, 200)).attr('height', 16)
        .attr('fill', '#0c0f18').attr('rx', 2);
      tip.append('text')
        .text((d.title || d.id).slice(0, 35))
        .attr('font-family', "'DM Mono',monospace")
        .attr('font-size', '8px')
        .attr('fill', 'var(--text-bright)')
        .attr('x', 4).attr('y', 0);
    })
    .on('mouseout', function(e, d) {
      const isEv = evidencePapers.has(d.id);
      d3.select(this)
        .attr('opacity',      isEv ? 1 : 0.7)
        .attr('stroke',       isEv ? '#ffcc60' : 'none')
        .attr('stroke-width', isEv ? 2 : 0);
      gContainer.selectAll('.node-tip').remove();
    });

  // Evidence node labels
  nodeG.filter(d => evidencePapers.has(d.id))
    .append('text')
    .text(d => (d.id.replace('arxiv:','')).slice(0, 12))
    .attr('font-family', "'DM Mono',monospace")
    .attr('font-size',   '7px')
    .attr('fill',        'var(--amber)')
    .attr('text-anchor', 'middle')
    .attr('dy', d => {
      const size = Math.max(3, Math.min(10, (d.citation_count || 0) / 50)) + 4;
      return -(size + 3);
    })
    .style('pointer-events', 'none');

  // Search filter
  currentFilterFn = (search, yearMin, yearMax) => {
    nodeG.attr('opacity', d => {
      const matches = (
        (!search || (d.title||'').toLowerCase().includes(search)) &&
        (!d.year  || (d.year >= yearMin && d.year <= yearMax))
      );
      return matches ? 1 : 0.06;
    });
  };
}

let currentFilterFn = null;

function toggleQueryGraph(btn) {
  if (evidencePapers.size === 0) {
    document.getElementById('evidence-count-label').textContent = 'run a query first';
    return;
  }
  queryGraphMode = !queryGraphMode;
  btn.classList.toggle('active', queryGraphMode);
  if (simulation) simulation.stop();
  renderGraph();
}

function showFullGraph() {
  queryGraphMode = false;
  const btn = document.getElementById('query-graph-btn');
  if (btn) btn.classList.remove('active');
  if (simulation) simulation.stop();
  renderGraph();
}

function updateGraphEvidence() {
  const label = document.getElementById('evidence-count-label');
  label.textContent = evidencePapers.size > 0
    ? `${evidencePapers.size} evidence papers`
    : 'no query yet';

  // Auto-enable query focus mode when evidence is available
  if (evidencePapers.size > 0) {
    queryGraphMode = true;
    const btn = document.getElementById('query-graph-btn');
    if (btn) btn.classList.add('active');
  }

  if (graphInited && simulation) renderGraph();
}

let genMode = 'builtin';

function setGenMode(mode) {
  genMode = mode;
  const exportSelect = document.getElementById('export-target-select');
  const sendBtn      = document.getElementById('send-btn');
  if (mode === 'export') {
    exportSelect.style.display = 'inline';
    sendBtn.textContent = 'Retrieve →';
  } else {
    exportSelect.style.display = 'none';
    sendBtn.textContent = 'Send';
  }
}

async function sendQuestion() {
  const input = document.getElementById('question-input');
  const q     = input.value.trim();
  if (!q) return;

  if (genMode === 'export') {
    await sendExportMode(q);
  } else {
    await sendBuiltinMode(q);
  }
}

async function sendExportMode(q) {
  const input = document.getElementById('question-input');
  input.value = '';
  input.style.height = '';

  const container = document.getElementById('messages');
  const welcome   = document.getElementById('welcome');
  if (welcome && welcome.parentNode === container) container.removeChild(welcome);

  appendMessage('user', q);

  const thinkEl = document.createElement('div');
  thinkEl.className = 'message assistant';
  thinkEl.innerHTML = `
    <div class="message-role">SYSTEM</div>
    <div class="thinking">
      <div class="thinking-dots"><span></span><span></span><span></span></div>
      retrieving evidence...
    </div>
  `;
  container.appendChild(thinkEl);
  container.scrollTop = container.scrollHeight;
  setStatus('retrieving...');
  document.getElementById('send-btn').disabled = true;

  try {
    // Step 1 — retrieve evidence
    const res  = await fetch('/api/retrieve-only', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ question: q }),
    });
    const data = await res.json();

    // Step 2 — store prompt on server for extension to fetch
    const storeRes = await fetch('/api/store-prompt', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ prompt: data.prompt }),
    });
    const { prompt_id } = await storeRes.json();

    container.removeChild(thinkEl);

    const target    = document.getElementById('export-target').value;
    const targetMap = {
  claude:  { 
    name: 'Claude',  
    url: 'https://claude.ai/new',  // /new forces browser, skips app
    color: '#c9a227' 
  },
  chatgpt: { name: 'ChatGPT', url: 'https://chatgpt.com',           color: '#19c37d' },
  gemini:  { name: 'Gemini',  url: 'https://gemini.google.com/app', color: '#4285f4' },
};
    const t = targetMap[target];

    // Step 3 — open LLM with prompt_id in URL
    const targetUrl = `${t.url}?kb_prompt_id=${prompt_id}`;

    const exportEl = document.createElement('div');
    exportEl.className = 'message assistant';
    exportEl.innerHTML = `
      <div class="message-role">RETRIEVAL COMPLETE
        <span class="intent-badge intent-${data.intent}">${data.intent}</span>
      </div>
      <div style="
        background:var(--bg-panel); border:1px solid var(--border);
        border-radius:3px; padding:14px; margin-top:8px;
      ">
        <div style="font-family:var(--mono);font-size:11px;color:var(--text-dim);margin-bottom:10px">
          Retrieved <span style="color:var(--cyan)">${data.distinct_chunks} chunks</span>
          from <span style="color:var(--cyan)">${data.distinct_papers} papers</span>
          ${data.abstain ? `<br><span style="color:var(--red)">⚠ ${data.abstain_reason}</span>` : ''}
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
          <a href="${targetUrl}" target="_blank" rel="noopener" style="
            padding:8px 20px; background:${t.color}22;
            border:1px solid ${t.color}; border-radius:2px;
            color:${t.color}; font-family:var(--cond); font-size:13px;
            font-weight:700; letter-spacing:0.08em; text-transform:uppercase;
            text-decoration:none; display:inline-block;
            transition: background 0.15s;
          ">→ Send to ${t.name}</a>
          <button onclick="copyPrompt(this)" data-prompt="${escAttr(data.prompt)}"
            style="
              padding:8px 14px; background:none;
              border:1px solid var(--border); border-radius:2px;
              color:var(--text-dim); font-family:var(--cond); font-size:12px;
              font-weight:600; letter-spacing:0.06em; text-transform:uppercase;
              cursor:pointer;
            "
          >📋 Copy prompt</button>
          <span style="font-family:var(--mono);font-size:10px;color:var(--text-dim)">
            (requires ICM-KB extension)
          </span>
        </div>
      </div>
    `;
    container.appendChild(exportEl);
    setStatus('ready');

  } catch(err) {
    container.removeChild(thinkEl);
    appendMessage('assistant', '⚠ Error: ' + err.message);
    setStatus('error');
  }

  document.getElementById('send-btn').disabled = false;
  container.scrollTop = container.scrollHeight;
}

async function copyPrompt(btn) {
  const prompt = btn.dataset.prompt;
  await navigator.clipboard.writeText(prompt);

  // Show confirm message
  const confirm = btn.parentElement.nextElementSibling;
  if (confirm) {
    confirm.style.display = 'block';
    setTimeout(() => { confirm.style.display = 'none'; }, 3000);
  }

  btn.textContent = '✓ Copied';
  setTimeout(() => { btn.textContent = '📋 Copy prompt'; }, 2000);
}

function showNodeInfo(d) {
  const el = document.getElementById('node-info');
  document.getElementById('node-info-title').textContent = d.title || d.id;
  document.getElementById('ni-year').textContent    = d.year    || '—';
  document.getElementById('ni-journal').textContent = (d.journal||'—').slice(0,30);
  document.getElementById('ni-cites').textContent   = d.citation_count || 0;
  document.getElementById('ni-id').textContent      = d.id;
  el.classList.add('visible');
}

function filterGraph() {
  if (!graphData) return;
  const search  = document.getElementById('graph-search').value.toLowerCase();
  const yearMin = parseInt(document.getElementById('year-min').value) || 0;
  const yearMax = parseInt(document.getElementById('year-max').value) || 9999;

  if (currentFilterFn) {
    currentFilterFn(search, yearMin, yearMax);
  } else {
    d3.selectAll('#graph-svg circle')
      .attr('opacity', d => {
        const matches = (
          (!search || (d.title||'').toLowerCase().includes(search)) &&
          (!d.year  || (d.year >= yearMin && d.year <= yearMax))
        );
        return matches ? 1 : 0.06;
      });
  }
}

function toggleEdge(type, btn) {
  showEdgeTypes[type] = !showEdgeTypes[type];
  btn.classList.toggle('active', showEdgeTypes[type]);
  if (graphData) renderGraph();
}

function graphZoom(factor) {
  if (!svg || !zoomBehavior) return;
  svg.transition().duration(300).call(zoomBehavior.scaleBy, factor);
}

function graphReset() {
  if (!svg || !zoomBehavior) return;
  svg.transition().duration(400).call(zoomBehavior.transform, d3.zoomIdentity);
}

// ── Stats ────────────────────────────────────────────────────────────────────
async function loadStats() {
  const data = await fetch('/api/stats').then(r => r.json());

  document.getElementById('stats-cards').innerHTML = `
    <div class="stat-card"><div class="stat-value">${data.total_papers}</div><div class="stat-label">Papers</div></div>
    <div class="stat-card"><div class="stat-value">${(data.total_chunks/1000).toFixed(1)}k</div><div class="stat-label">Chunks indexed</div></div>
    <div class="stat-card"><div class="stat-value">${data.graph_nodes.toLocaleString()}</div><div class="stat-label">Graph nodes</div></div>
    <div class="stat-card"><div class="stat-value">${(data.graph_edges/1000).toFixed(1)}k</div><div class="stat-label">Citation edges</div></div>
  `;

  const maxEra = Math.max(...data.by_era.map(e => e.count));
  document.getElementById('era-chart').innerHTML = data.by_era.map(e => `
    <div class="bar-row">
      <span class="bar-label">${e.era}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${e.count/maxEra*100}%"></div></div>
      <span class="bar-val">${e.count}</span>
    </div>
  `).join('');

  const maxJ = Math.max(...data.by_journal.map(j => j.count));
  document.getElementById('journal-chart').innerHTML = data.by_journal.map(j => `
    <div class="bar-row">
      <span class="bar-label" title="${j.journal}">${j.journal
        .replace('Monthly Notices of the Royal Astronomical Society','MNRAS')
        .replace('Astronomy and Astrophysics','A&A')
        .replace('The Astrophysical Journal','ApJ')}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${j.count/maxJ*100}%"></div></div>
      <span class="bar-val">${j.count}</span>
    </div>
  `).join('');

  const maxA = Math.max(...data.top_authors.map(a => a.count));
  document.getElementById('author-chart').innerHTML = data.top_authors.map(a => `
    <div class="bar-row">
      <span class="bar-label">${a.name}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${a.count/maxA*100}%"></div></div>
      <span class="bar-val">${a.count}</span>
    </div>
  `).join('');

  if (evidenceDetails.length > 0) updateQueryStats(evidenceDetails);
}

function updateQueryStats(evidence) {
  const section = document.getElementById('query-stats-section');
  const content = document.getElementById('query-stats-content');
  if (!evidence || evidence.length === 0) return;

  const eraCounts     = {};
  const journalCounts = {};
  evidence.forEach(e => {
    const yr  = e.year || 0;
    const era = yr < 2000 ? '1995–1999'
      : yr < 2005 ? '2000–2004'
      : yr < 2010 ? '2005–2009'
      : yr < 2015 ? '2010–2014'
      : yr < 2020 ? '2015–2019' : '2020–2026';
    eraCounts[era] = (eraCounts[era] || 0) + 1;
    const j = (e.journal || 'Unknown')
      .replace('Monthly Notices of the Royal Astronomical Society','MNRAS')
      .replace('Astronomy and Astrophysics','A&A')
      .replace('The Astrophysical Journal','ApJ');
    journalCounts[j] = (journalCounts[j] || 0) + 1;
  });

  const eraRows = Object.entries(eraCounts)
    .sort((a,b) => a[0].localeCompare(b[0]))
    .map(([k,v]) => `<div class="query-paper-item">${k} <span>${v}</span></div>`)
    .join('');

  const journalRows = Object.entries(journalCounts)
    .sort((a,b) => b[1]-a[1])
    .map(([k,v]) => `<div class="query-paper-item">${k} <span>${v}</span></div>`)
    .join('');

  const paperRows = evidence.slice(0,6).map(e => `
    <div class="query-paper-item" style="flex-direction:column;align-items:flex-start;gap:2px">
      <span style="color:var(--cyan);font-size:9px">${(e.paper_id||'').replace('arxiv:','')}</span>
      <span style="color:var(--text-dim)">${(e.title||'').slice(0,45)} (${e.year||'?'})</span>
    </div>
  `).join('');

  content.innerHTML = `
    <div class="query-stats-grid">
      <div class="query-stat-block">
        <div class="query-stat-label">By era</div>
        <div class="query-paper-list">${eraRows}</div>
      </div>
      <div class="query-stat-block">
        <div class="query-stat-label">By journal</div>
        <div class="query-paper-list">${journalRows}</div>
      </div>
    </div>
    <div class="query-stat-block" style="margin-top:12px">
      <div class="query-stat-label">Evidence papers (${evidence.length})</div>
      <div class="query-paper-list">${paperRows}</div>
    </div>
  `;

  section.classList.add('visible');
}

// ── Utilities ────────────────────────────────────────────────────────────────
function escHtml(s) {
  if (!s) return '';
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function escAttr(s) {
  if (!s) return '';
  return String(s).replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function formatDate(ts) {
  if (!ts) return '';
  return new Date(ts).toLocaleDateString('en-GB', { day:'numeric', month:'short' });
}

function setStatus(txt) {
  document.getElementById('status-text').textContent = txt;
}
document.getElementById('sidebar-toggle').style.left = '240px';

// ── Init ─────────────────────────────────────────────────────────────────────
loadConversations();