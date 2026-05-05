let currentSessionId = null;   // active session UUID
let history = [];              // local copy for rendering new batches only
let isLoading = false;

const API_URL = () => document.getElementById('api-url').value.replace(/\/$/, '');

// health check
async function checkHealth() {
  const dot = document.getElementById('status-dot');
  const lbl = document.getElementById('status-label');
  try {
    const r = await fetch(API_URL() + '/health', { signal: AbortSignal.timeout(3000) });
    if (r.ok) {
      dot.className = 'status-dot ok';
      lbl.textContent = 'connected';
      lbl.style.color = '#1D9E75';
    } else {
      throw new Error('not ok');
    }
  } catch {
    dot.className = 'status-dot err';
    lbl.textContent = 'not reachable';
    lbl.style.color = '#E24B4A';
  }
}

document.getElementById('api-url').addEventListener('change', checkHealth);
checkHealth();
setInterval(checkHealth, 15000);

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function sendPreset(text) {
  document.getElementById('user-input').value = text;
  sendMessage();
}

function addMessage(role, content, isTable) {
  const msgs = document.getElementById('messages');
  const wrap = document.createElement('div');
  wrap.className = 'msg' + (role === 'user' ? ' user' : '');

  const avatar = document.createElement('div');
  avatar.className = 'avatar ' + (role === 'user' ? 'user' : 'ai');
  avatar.textContent = role === 'user' ? 'You' : 'AI';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  if (isTable && Array.isArray(isTable)) {
    bubble.appendChild(renderTable(isTable));
  } else {
    bubble.innerHTML = formatText(content);
  }

  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  msgs.appendChild(wrap);
  msgs.scrollTop = msgs.scrollHeight;
  return bubble;
}

function formatText(text) {
  // convert markdown-like formatting
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code style="background:#f0ede8;padding:1px 4px;border-radius:4px;font-size:12px">$1</code>')
    .replace(/\n/g, '<br>');
}

function renderTable(patients) {
  if (!patients.length) return document.createTextNode('No data.');
  const labs = Object.keys(patients[0]);

  const wrap = document.createElement('div');
  wrap.className = 'data-table-wrap';

  const tbl = document.createElement('table');
  tbl.className = 'lab-table';

  const thead = document.createElement('thead');
  const hr = document.createElement('tr');
  ['Patient', ...labs].forEach(h => {
    const th = document.createElement('th');
    th.textContent = h;
    hr.appendChild(th);
  });
  thead.appendChild(hr);
  tbl.appendChild(thead);

  const tbody = document.createElement('tbody');
  patients.forEach((p, i) => {
    const tr = document.createElement('tr');
    const idx = document.createElement('td');
    idx.textContent = `#${i + 1}`;
    idx.style.fontWeight = '500';
    idx.style.color = '#0f6e56';
    tr.appendChild(idx);
    labs.forEach(lab => {
      const td = document.createElement('td');
      td.textContent = typeof p[lab] === 'number' ? p[lab].toFixed(3) : p[lab];
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  tbl.appendChild(tbody);
  wrap.appendChild(tbl);
  return wrap;
}

function addTyping() {
  const msgs = document.getElementById('messages');
  const wrap = document.createElement('div');
  wrap.className = 'msg'; wrap.id = 'typing-indicator';
  const avatar = document.createElement('div');
  avatar.className = 'avatar ai'; avatar.textContent = 'AI';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
  wrap.appendChild(avatar); wrap.appendChild(bubble);
  msgs.appendChild(wrap);
  msgs.scrollTop = msgs.scrollHeight;
}

function removeTyping() {
  const t = document.getElementById('typing-indicator');
  if (t) t.remove();
}

async function sendMessage() {
  if (isLoading) return;
  const input = document.getElementById('user-input');
  const text = input.value.trim();
  if (!text) return;

  input.value = ''; input.style.height = 'auto';
  isLoading = true;
  document.getElementById('send-btn').disabled = true;

  addMessage('user', text);
  addTyping();

  try {
    // send only the new message and session_id
    // server owns history
    const prevLen = history.length;
    const res = await fetch(API_URL() + '/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session_id: currentSessionId }),
    });

    removeTyping();

    if (!res.ok) {
      const err = await res.text();
      addMessage('assistant', `Error ${res.status}: ${err}`);
    } else {
      const data = await res.json();
      currentSessionId = data.session_id;

      // Update session badge
      const badge = document.getElementById('session-badge');
      badge.style.display = 'block';
      badge.textContent = 'Active: #' + data.session_id;

      // fetch full updated history from server to find new tool results
      const sessionRes = await fetch(API_URL() + '/sessions/' + currentSessionId);
      const sessionData = await sessionRes.json();
      history = sessionData.messages;
      // new messages is everything after the user msg we already rendered
      const newMessages = history.slice(prevLen + 1);

      // collect every generate_synthetic_patients tool call result from this turn
      const batches = extractAllPatientBatches(newMessages);
      const totalPatients = batches.reduce((sum, b) => sum + b.patients.length, 0);

      if (batches.length > 0) {
        const bubble = addMessage('assistant', '', null);

        // summary indicator
        const ind = document.createElement('div');
        ind.className = 'tool-indicator';
        ind.style.animation = 'none';
        ind.textContent = batches.length === 1
          ? `Generated ${totalPatients} synthetic patient records`
          : `Generated ${totalPatients} synthetic patient records across ${batches.length} calls`;
        bubble.appendChild(ind);

        // one table per tool call, each with its own label
        batches.forEach((batch, i) => {
          if (batches.length > 1) {
            const label = document.createElement('p');
            label.style.cssText = 'font-size:12px;font-weight:500;color:#555;margin:10px 0 4px';
            label.textContent = `Call ${i + 1} - ${batch.patients.length} patient${batch.patients.length > 1 ? 's' : ''}` +
              (batch.label ? `: ${batch.label}` : '');
            bubble.appendChild(label);
          }
          bubble.appendChild(renderTable(batch.patients));
        });

        // claude's text response below the tables
        if (data.response) {
          const p = document.createElement('p');
          p.innerHTML = formatText(data.response);
          p.style.marginTop = '10px';
          bubble.appendChild(p);
        }
      } else {
        addMessage('assistant', data.response || 'No response.');
      }
    }
  } catch (e) {
    removeTyping();
    addMessage('assistant', `Connection error: ${e.message}. Is the API server running at ${API_URL()}?`);
  }

  isLoading = false;
  document.getElementById('send-btn').disabled = false;
}

// collect every successful generate_synthetic_patients result from a set of
// messages (typically just the new messages from this turn).
// Returns an array of { patients, label } objects - one per tool call.
function extractAllPatientBatches(messages) {
  const batches = [];

  // build a map of tool_use id to input so we can label each batch
  const toolInputs = {};
  for (const msg of messages) {
    if (msg.role === 'assistant' && Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block.type === 'tool_use' && block.name === 'generate_synthetic_patients') {
          // build a short human-readable label from the call's inputs
          const inp = block.input || {};
          const parts = [];
          if (inp.n_samples) parts.push(`n=${inp.n_samples}`);
          if (inp.careunit)  parts.push(inp.careunit);
          if (inp.ccs_group) parts.push(inp.ccs_group);
          if (inp.age)       parts.push(`age ${inp.age}`);
          toolInputs[block.id] = parts.join(', ');
        }
      }
    }
  }

  // collect each tool_result that has patient data
  for (const msg of messages) {
    if (msg.role === 'user' && Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block.type === 'tool_result' && block.content) {
          try {
            const parsed = JSON.parse(block.content);
            if (parsed.status === 'success' && parsed.patients && parsed.patients.length > 0) {
              batches.push({
                patients: parsed.patients,
                label: toolInputs[block.tool_use_id] || '',
              });
            }
          } catch {}
        }
      }
    }
  }

  return batches;
}

function newSession() {
  currentSessionId = null;
  history = [];
  document.getElementById('messages').innerHTML = '';
  document.getElementById('session-badge').style.display = 'none';
  document.getElementById('session-list').innerHTML = '';
  addMessage('assistant', 'Started a new session. What patient profile would you like to generate?');
}

async function loadSessions() {
  const container = document.getElementById('session-list');
  container.innerHTML = '<div style="font-size:11px;color:#aaa;padding:4px 0">Loading...</div>';
  try {
    const res = await fetch(API_URL() + '/sessions');
    const data = await res.json();
    if (!data.sessions.length) {
      container.innerHTML = '<div style="font-size:11px;color:#aaa;padding:4px 0">No saved sessions yet.</div>';
      return;
    }
    container.innerHTML = '';
    data.sessions
      .sort((a, b) => b.updated_at.localeCompare(a.updated_at))
      .forEach(s => {
        const btn = document.createElement('button');
        btn.className = 'preset-btn';
        btn.innerHTML = `<strong style="font-size:11px;color:#185fa5">#${s.short_id}</strong> <span style="float:right;color:#aaa">${s.turns} turns</span><span style="display:block;font-size:11px;margin-top:2px">${s.label}</span><span style="font-size:10px;color:#aaa">${s.updated_at}</span>`;
        btn.onclick = () => resumeSession(s.session_id);
        container.appendChild(btn);
      });
  } catch(e) {
    container.innerHTML = '<div style="font-size:11px;color:#e24b4a">Could not reach API.</div>';
  }
}

async function resumeSessionFromInput() {
  const input = document.getElementById('session-id-input');
  const val = input.value.trim();
  if (!val) return;
  input.value = '';
  await resumeSession(val);
}

async function resumeSession(session_id) {
  document.getElementById('session-list').innerHTML = '';
  try {
    const res = await fetch(API_URL() + '/sessions/' + session_id);
    if (!res.ok) { addMessage('assistant', 'Session not found.'); return; }
    const data = await res.json();
    currentSessionId = data.session_id;
    history = data.messages;

    // clear and re-render conversation
    const msgs = document.getElementById('messages');
    msgs.innerHTML = '';
    addMessage('assistant', `Resumed session #${data.short_id}: "${data.label}" (${data.turns} turns, last active ${data.updated_at})`);

    // replay each user/assistant text turn visually
    let i = 0;
    while (i < data.messages.length) {
      const msg = data.messages[i];
      if (msg.role === 'user' && typeof msg.content === 'string') {
        addMessage('user', msg.content);
      } else if (msg.role === 'assistant') {
        // check if next user message has tool results
        const nextMsg = data.messages[i + 1];
        const batches = nextMsg ? extractAllPatientBatches([nextMsg]) : [];
        const text = Array.isArray(msg.content)
          ? msg.content.filter(b => b.type === 'text').map(b => b.text).join('')
          : (msg.content || '');
        if (batches.length || text) {
          const bubble = addMessage('assistant', '', null);
          if (batches.length) {
            const ind = document.createElement('div');
            ind.className = 'tool-indicator'; ind.style.animation = 'none';
            ind.textContent = `${batches.reduce((s,b)=>s+b.patients.length,0)} synthetic patient records`;
            bubble.appendChild(ind);
            batches.forEach(b => bubble.appendChild(renderTable(b.patients)));
          }
          if (text) {
            const p = document.createElement('p');
            p.innerHTML = formatText(text);
            p.style.marginTop = '8px';
            bubble.appendChild(p);
          }
        }
      }
      i++;
    }

    // show session badge
    const badge = document.getElementById('session-badge');
    badge.style.display = 'block';
    badge.textContent = 'Active: #' + data.session_id;
  } catch(e) {
    addMessage('assistant', 'Failed to load session: ' + e.message);
  }
}