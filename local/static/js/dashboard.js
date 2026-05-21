/**
 * AI Video Agent — PRO Enterprise Operations Control JS
 * ──────────────────────────────────────────────────────────
 * Manages WebSocket state, dynamic tabs, audio streams, voice 
 * activity visualizers, and simulated dashboard metrics.
 */

// ── State ────────────────────────────────────────────────────────────────────
const state = {
  ws: null,
  sessionActive: false,
  micActive: false,
  audioCtx: null,
  micStream: null,
  micProcessor: null,
  isPlaying: false,
  isSpeaking: false,
  config: {},
  activeTab: 'incident-change',
  tpsInterval: null,
  outAudioCtx: null,
  nextStartTime: 0,
  speechMode: 'gemini_live',
};

// ── DOM Helpers ──────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const dom = {
  // Navigation Tabs
  navItems:        document.querySelectorAll('.nav-item'),
  tabPanels:       document.querySelectorAll('.tab-panel'),
  moduleTitle:     $('moduleTitle'),
  moduleDesc:      $('moduleDesc'),
  
  // Interactive Elements
  anomalyHeatmap:  $('anomalyHeatmap'),
  latencyTimeline: $('latencyTimeline'),
  tpsValue:        $('tpsValue'),

  // Avatar & Speech Components
  micBtn:          $('micBtn'),
  micLabel:        $('micLabel'),
  textInput:       $('textInput'),
  sendBtn:         $('sendBtn'),
  btnStartSession: $('btnStartSession'),
  btnStopSession:  $('btnStopSession'),
  conversationLog: $('conversationLog'),
  transcriptArea:  $('transcriptArea'),
  liveTranscript:  $('liveTranscript'),
  avatarScreen:    document.querySelector('.avatar-screen'),
  avatarImage:     $('avatarImage'),
  avatarVideo:     $('avatarVideo'),
  avatarGlow:      $('avatarGlow'),
  speakingWave:    $('speakingWave'),
  gpuStatusDot:    $('gpuStatusDot'),
  vramVal:         $('vramVal'),
  vramFill:        $('vramFill'),
  
  // Pilled Status
  pillDashboard:   $('pill-dashboard'),
  pillGemini:      $('pill-gemini'),
  pillCartesia:    $('pill-cartesia'),

  // Modals
  personaModal:      $('personaModal'),
  btnSwitchPersona:  $('btnSwitchPersona'),
  closePersonaModal: $('closePersonaModal'),
  speechModeSelect:  $('speechModeSelect'),
};

// Module Details mapping
const moduleDetails = {
  'incident-change': {
    title: 'Incident & Change Management',
    desc: 'Perform Root Cause Analysis (RCA), monitor active incidents, and coordinate remediation.'
  },
  'observability': {
    title: 'Observability & Monitoring',
    desc: 'Audit infrastructure logs, view anomaly detection alerts, and audit general system health.'
  },
  'hyper-transaction': {
    title: 'Hyper-Transaction Excellence',
    desc: 'Track UPI Transactions Per Second (TPS), processing latency, and EOD/BOD batches.'
  },
  'security-resilience': {
    title: 'Security & Resilience Network',
    desc: 'Assess system vulnerability posture, audit firewall IP security logs, and deploy emergency rollback.'
  },
  'performance-capacity': {
    title: 'Performance & Capacity Management',
    desc: 'Forecast workload demand metrics, review FinOps burn rates, and tune right-sizing.'
  },
  'predictive-intel': {
    title: 'Predictive Intelligence Engine',
    desc: 'Audit trend similarity correlation profiles and monitor system state trends.'
  }
};

// ── Tab Navigation ───────────────────────────────────────────────────────────
dom.navItems.forEach(item => {
  item.addEventListener('click', () => {
    const targetTab = item.getAttribute('data-tab');
    if (!targetTab) return;

    // Switch Nav Class
    dom.navItems.forEach(nav => nav.classList.remove('active'));
    item.classList.add('active');

    // Switch Panel Visibility
    dom.tabPanels.forEach(panel => {
      panel.classList.toggle('active', panel.id === `tab-${targetTab}`);
    });

    // Update Header Text
    const details = moduleDetails[targetTab];
    if (details) {
      dom.moduleTitle.textContent = details.title;
      dom.moduleDesc.textContent = details.desc;
    }

    state.activeTab = targetTab;

    // Trigger tab-specific loaders
    if (targetTab === 'incident-change') {
      startTelemetryLogs();
    } else if (targetTab === 'observability') {
      renderAnomalyHeatmap();
    } else if (targetTab === 'hyper-transaction') {
      startHyperTransactionStream();
      startTransactionLedgerStream();
    } else if (targetTab === 'security-resilience') {
      startFirewallLogStream();
    } else {
      clearInterval(state.tpsInterval);
      clearInterval(logStreamInterval);
      clearInterval(ledgerStreamInterval);
      clearInterval(firewallStreamInterval);
    }

    // Sync active tab module state to the backend
    send({ type: 'sync_module', module: targetTab });
  });
});

// ── Render Components ────────────────────────────────────────────────────────
function renderAnomalyHeatmap() {
  if (!dom.anomalyHeatmap) return;
  dom.anomalyHeatmap.innerHTML = '';
  
  // Create 48 blocks representing 48 hours anomaly logs
  for (let i = 0; i < 48; i++) {
    const block = document.createElement('div');
    block.className = 'heatmap-block';
    
    // Simulate random anomaly risk
    const val = Math.random();
    if (val > 0.9) {
      block.style.backgroundColor = 'rgba(244, 63, 94, 0.7)'; // Critical
      block.style.borderColor = 'var(--color-red)';
      block.setAttribute('data-severity', 'critical');
      block.setAttribute('title', 'Anomaly Peak Detected (Risk score: ' + val.toFixed(2) + ')');
    } else if (val > 0.7) {
      block.style.backgroundColor = 'rgba(255, 157, 108, 0.4)'; // Warning
      block.style.borderColor = 'var(--color-orange)';
      block.setAttribute('data-severity', 'warning');
      block.setAttribute('title', 'Moderate Drift Alert (Risk score: ' + val.toFixed(2) + ')');
    } else {
      block.style.backgroundColor = 'rgba(16, 185, 129, 0.15)'; // Secure
      block.style.borderColor = 'rgba(16, 185, 129, 0.3)';
      block.setAttribute('data-severity', 'secure');
      block.setAttribute('title', 'Normal operations');
    }
    dom.anomalyHeatmap.appendChild(block);
  }
}

window.filterAnomalies = (severity) => {
  // Toggle filter active classes
  document.querySelectorAll('#tab-observability .filter-btn').forEach(btn => {
    btn.classList.toggle('active', btn.id === `anomalyFilter${severity.charAt(0).toUpperCase() + severity.slice(1)}`);
  });

  // Filter blocks
  const blocks = document.querySelectorAll('.heatmap-block');
  blocks.forEach(block => {
    const sev = block.getAttribute('data-severity');
    if (severity === 'all') {
      block.style.opacity = '1';
      block.style.transform = 'scale(1)';
    } else {
      if (sev === severity) {
        block.style.opacity = '1';
        block.style.transform = 'scale(1.05)';
      } else {
        block.style.opacity = '0.15';
        block.style.transform = 'scale(0.9)';
      }
    }
  });
};


function startHyperTransactionStream() {
  if (!dom.latencyTimeline) return;
  dom.latencyTimeline.innerHTML = '';
  clearInterval(state.tpsInterval);

  // Initialize 40 latency bars
  for (let i = 0; i < 40; i++) {
    const bar = document.createElement('div');
    bar.className = 'timeline-bar';
    const heightPercent = Math.floor(Math.random() * 60) + 15;
    bar.style.height = `${heightPercent}%`;
    dom.latencyTimeline.appendChild(bar);
  }

  // Stream live latency updates
  state.tpsInterval = setInterval(() => {
    // Generate new bar height
    const heightPercent = Math.floor(Math.random() * 70) + 10;
    const bar = document.createElement('div');
    bar.className = 'timeline-bar';
    bar.style.height = `${heightPercent}%`;
    
    // Slide left
    if (dom.latencyTimeline.children.length >= 40) {
      dom.latencyTimeline.removeChild(dom.latencyTimeline.firstChild);
    }
    dom.latencyTimeline.appendChild(bar);

    // Dynamic TPS counter fluctuation
    if (dom.tpsValue) {
      const baseTps = 2400 + Math.floor(Math.random() * 150) - 75;
      dom.tpsValue.textContent = baseTps.toLocaleString();
    }
  }, 350);
}

// ── Incident Module 1 Enhancements ───────────────────────────────────────────
let logStreamInterval = null;

// Telemetry Mock Logger
function startTelemetryLogs() {
  const consoleEl = $('terminalLogs');
  if (!consoleEl) return;
  clearInterval(logStreamInterval);

  const systems = ['SYS', 'DB', 'K8S', 'GATEWAY', 'APP-PDB'];
  const actions = [
    'Resource footprint clean status: OK',
    'Shared pool latch contention checked',
    'Pod connection ping back: 0.8ms',
    'Connection pool size stable: 32/128',
    'SQL execution trace audit complete',
    'Garbage collection check resolved',
    'Flushed temp query caches'
  ];

  logStreamInterval = setInterval(() => {
    if (state.activeTab !== 'incident-change') {
      clearInterval(logStreamInterval);
      return;
    }
    const sys = systems[Math.floor(Math.random() * systems.length)];
    const act = actions[Math.floor(Math.random() * actions.length)];
    const line = document.createElement('div');
    line.className = `log-line ${sys === 'DB' ? 'database' : sys === 'SYS' ? 'system' : 'success'}`;
    line.textContent = `[${sys}] ${act}`;
    consoleEl.appendChild(line);
    
    // Auto-scroll
    consoleEl.scrollTop = consoleEl.scrollHeight;
    
    // Cap log lines inside terminal
    while (consoleEl.children.length > 20) {
      consoleEl.removeChild(consoleEl.firstChild);
    }
  }, 1800);
}

// Filter Runbooks function
window.filterRunbooks = (status) => {
  // Toggle active button class
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.classList.toggle('active', btn.id === `runbookFilter${status.charAt(0).toUpperCase() + status.slice(1)}`);
  });

  // Filter Items
  const items = document.querySelectorAll('.runbook-item');
  items.forEach(item => {
    const itemStatus = item.getAttribute('data-status');
    if (status === 'all') {
      item.style.display = 'block';
    } else {
      item.style.display = (itemStatus === status) ? 'block' : 'none';
    }
  });
};

// Auto-run remediation simulation
window.triggerDBARemediation = () => {
  const btn = $('btnRunRemediation');
  if (btn) btn.disabled = true;

  const consoleEl = $('terminalLogs');
  const addLog = (text, type = '') => {
    if (!consoleEl) return;
    const line = document.createElement('div');
    line.className = `log-line ${type}`;
    line.textContent = `[DBA-JOB] ${text}`;
    consoleEl.appendChild(line);
    consoleEl.scrollTop = consoleEl.scrollHeight;
  };

  addLog('Executing dynamic remediation job...', 'system');
  addMessage('ai', 'Triggering memory compaction and active shared pool reclamation on orcl_pdb1...');

  setTimeout(() => {
    addLog('FLUSH SHARED_POOL issued on session: active', 'database');
  }, 1000);

  setTimeout(() => {
    addLog('Heap compaction process completed. ORA-04031 state resolved.', 'success');
    addMessage('ai', 'Compaction successful. Memory pool reclaimed on instance orcl_pdb1.');
    
    // Update Runbook item status
    const rb = $('rb-ora-04031');
    if (rb) {
      rb.setAttribute('data-status', 'mitigated');
      rb.classList.remove('active');
      const actions = rb.querySelector('.runbook-actions');
      if (actions) {
        actions.innerHTML = '<span class="status-badge secure" style="padding: 4px 8px;">Compacted & Mitigated ✓</span>';
      }
    }

    // Update Timeline Step 4 to Finished
    const step4 = $('rcaTimelineStep4');
    if (step4) {
      step4.className = 'timeline-step finished';
      const timeSpan = step4.querySelector('.step-time');
      if (timeSpan) timeSpan.textContent = new Date().toLocaleTimeString('en-US', { hour12: false });
    }
  }, 2500);
};

window.askAvatarAboutIncident = () => {
  if (!state.sessionActive) {
    alert('Please connect the agent first.');
    return;
  }
  dom.textInput.value = "Explain the ORA-04031 Incident root cause and how to fix it.";
  sendText();
};

// ── Module 3: Hyper-Transaction Live Stream ──────────────────────────────────
let ledgerStreamInterval = null;
function startTransactionLedgerStream() {
  const ledgerEl = $('transactionLedgerBody');
  if (!ledgerEl) return;
  clearInterval(ledgerStreamInterval);

  const gateways = ['SBI', 'HDFC', 'ICICI', 'AXIS'];
  let txnCounter = 98721204;

  ledgerStreamInterval = setInterval(() => {
    if (state.activeTab !== 'hyper-transaction') {
      clearInterval(ledgerStreamInterval);
      return;
    }
    const gate = gateways[Math.floor(Math.random() * gateways.length)];
    const latency = Math.floor(Math.random() * 15) + 10;
    
    // Update live bank status labels
    const gateLabel = $(`gate-${gate.toLowerCase()}`);
    if (gateLabel) gateLabel.textContent = `${latency}ms ✓`;

    // Append to table
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>TXN-${txnCounter++}</td>
      <td>${gate}</td>
      <td>${latency}ms</td>
      <td><span class="status-badge secure">SUCCESS</span></td>
    `;
    ledgerEl.insertBefore(tr, ledgerEl.firstChild);

    // Limit table length
    if (ledgerEl.children.length > 8) {
      ledgerEl.removeChild(ledgerEl.lastChild);
    }
  }, 1200);
}

// ── Module 4: Security Firewall Audit & Kill Switch ──────────────────────────
let firewallStreamInterval = null;
function startFirewallLogStream() {
  const bodyEl = $('firewallAuditBody');
  if (!bodyEl) return;
  clearInterval(firewallStreamInterval);

  const ips = ['192.168.10.45', '10.231.42.112', '172.16.89.21', '192.168.12.19', '10.220.14.88'];
  const endpoints = ['/api/v1/auth/login', '/admin/config/db', '/api/v2/transactions', '/static/css/style.css', '/api/persona'];

  firewallStreamInterval = setInterval(() => {
    if (state.activeTab !== 'security-resilience') {
      clearInterval(firewallStreamInterval);
      return;
    }
    const ip = ips[Math.floor(Math.random() * ips.length)];
    const path = endpoints[Math.floor(Math.random() * endpoints.length)];
    const threat = Math.random();
    
    const isThreat = threat > 0.85;
    const badgeText = isThreat ? 'BLOCK' : (threat > 0.6 ? 'AUDIT' : 'ALLOW');
    const badgeClass = isThreat ? 'color: var(--color-red); background: rgba(244, 63, 94, 0.15);' : (threat > 0.6 ? 'color: var(--color-orange); background: rgba(255, 157, 108, 0.15);' : 'color: var(--color-green); background: rgba(16, 185, 129, 0.15);');
    const threatBadgeClass = isThreat ? 'warning' : (threat > 0.6 ? 'warning' : 'secure');

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${ip}</td>
      <td><code>${path}</code></td>
      <td><span class="status-badge ${threatBadgeClass}">${threat.toFixed(2)}</span></td>
      <td><span class="status-badge" style="${badgeClass}">${badgeText}</span></td>
    `;
    bodyEl.insertBefore(tr, bodyEl.firstChild);

    if (bodyEl.children.length > 8) {
      bodyEl.removeChild(bodyEl.lastChild);
    }
  }, 1600);
}

window.triggerEmergencyKill = () => {
  const confirmKill = confirm('Are you sure you want to trigger emergency rollback on Kubernetes core deployment clusters?');
  if (!confirmKill) return;

  const btn = $('btnKillswitch');
  const timerText = $('killswitchTimerText');
  const progressBar = $('killswitchProgressBar');
  const countdownContainer = $('killswitchCountdownContainer');

  if (btn) btn.disabled = true;
  if (countdownContainer) countdownContainer.style.display = 'block';

  let count = 5;
  progressBar.style.width = '100%';
  timerText.textContent = `ROLLING BACK IN: ${count}s`;

  addMessage('user', 'Trigger rollback!');
  addMessage('ai', 'Deploying Kill Switch rollback. Reverting active cluster deployments to stable tag: v2.4.1.');

  const interval = setInterval(() => {
    count--;
    progressBar.style.width = `${(count / 5) * 100}%`;
    timerText.textContent = `ROLLING BACK IN: ${count}s`;

    if (count <= 0) {
      clearInterval(interval);
      if (countdownContainer) countdownContainer.style.display = 'none';
      
      // Update states
      const label = $('killswitchLabel');
      const status = $('killswitchStatus');
      const stateVal = $('killSwitchStateVal');
      const threatVal = $('securityThreatLevelVal');

      if (label) label.textContent = 'ROLLBACK COMPLETE';
      if (status) status.textContent = 'Active version: v2.4.1';
      if (stateVal) {
        stateVal.textContent = 'ACTIVE';
        stateVal.className = 'metric-value';
        stateVal.style.color = 'var(--color-orange)';
      }
      if (threatVal) {
        threatVal.textContent = 'SAFE';
        threatVal.style.color = 'var(--color-green)';
      }

      addMessage('ai', 'Rollback sequence finished. Kubernetes deployment node group successfully reverted to v2.4.1.');
    }
  }, 1000);
};

// ── Module 5: Performance & Capacity Right-sizing ────────────────────────────
window.applyNodeDownscale = () => {
  const btn = $('btnApplyDownscale');
  if (btn) btn.disabled = true;

  addMessage('ai', 'Applying Oracle DBA node capacity right-sizing recommendation...');

  setTimeout(() => {
    const desc = $('opt-db-desc');
    const savings = $('opt-db-savings');
    const burnVal = $('finopsBurnVal');
    const nodesVal = $('activeNodesVal');

    if (desc) desc.innerHTML = 'Downscaled to 16 vCPUs. Savings locked.';
    if (savings) savings.innerHTML = 'Savings realized: <strong>$140/mo (Reclaimed)</strong>';
    if (burnVal) burnVal.textContent = '$288'; // reduced budget
    if (nodesVal) nodesVal.textContent = '3 / 12'; // downscaled 1 node

    addMessage('ai', 'Oracle DB capacity downscale successfully completed. Daily burn rate decreased to $288.');
  }, 1500);
};

// ── Module 6: Predictive Similarity Simulator ─────────────────────────────────
window.runPredictionSimulation = () => {
  const btn = $('btnRunPredictionSim');
  if (btn) btn.disabled = true;

  const matchVal = $('trendMatchVal');
  const riskVal = $('outageRiskVal');
  const step3 = $('flowStep3Desc');

  if (matchVal) matchVal.textContent = 'AUDITING...';
  addMessage('ai', 'Running ML Pattern Similarity search against historical outage profiles...');

  setTimeout(() => {
    if (matchVal) matchVal.textContent = '99.4%';
    if (riskVal) riskVal.textContent = '0.02%';
    if (step3) step3.textContent = 'Pattern matching complete. Drift resolved.';

    addMessage('ai', 'ML similarity search complete. Outage risk has dropped to 0.02% (drift resolved).');
    if (btn) btn.disabled = false;
  }, 2000);
};



// ── WebSocket Connection ──────────────────────────────────────────────────────
function connectWS() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${protocol}//${location.host}/ws/conversation`;
  state.ws = new WebSocket(url);

  state.ws.onopen = () => {
    console.log('✓ WebSocket connected');
    dom.pillDashboard.innerHTML = '<span class="dot online"></span> Dashboard';
  };

  state.ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      handleMessage(msg);
    } catch (e) {
      console.error('WS parse error:', e);
    }
  };

  state.ws.onclose = () => {
    console.log('WebSocket closed, reconnecting...');
    dom.pillDashboard.innerHTML = '<span class="dot offline"></span> Dashboard';
    setTimeout(connectWS, 3000);
  };
}

function send(obj) {
  if (state.ws?.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(obj));
  }
}

// ── WS Msg Dispatch ──────────────────────────────────────────────────────────
function handleMessage(msg) {
  switch (msg.type) {
    case 'status':
      updateServiceStatus(msg.services);
      break;

    case 'config':
      state.config = msg.data;
      break;

    case 'conversation_started':
      onConversationStarted();
      break;

    case 'conversation_stopped':
      onConversationStopped();
      break;

    case 'transcript':
      onTranscript(msg.text, msg.final);
      break;

    case 'ai_text':
      addMessage('ai', msg.text);
      // In web_speech mode, Cartesia TTS sends the audio via ai_audio.
      // Browser speechSynthesis is NOT used for output — Cartesia handles it.
      break;

    case 'ai_audio':
      // Play Cartesia MP3 in web_speech mode OR Gemini PCM in gemini_live mode
      playAudio(msg.audio, msg.format, msg.sampleRate);
      break;

    case 'video_frame':
      // Live JPEG frame from GPU — update avatar image directly
      if (msg.data) {
        const avatarImg = document.getElementById('avatarImage');
        if (avatarImg) {
          avatarImg.src = 'data:image/jpeg;base64,' + msg.data;
        }
      }
      break;

    case 'error':
      addMessage('ai', 'An error occurred: ' + msg.message);
      break;
  }
}

// ── Update Service Badges ────────────────────────────────────────────────────
function updateServiceStatus(services) {
  // Update Pills
  if (services.gemini) {
    dom.pillGemini.innerHTML = `<span class="dot ${services.gemini.status}"></span> Gemini Live`;
  }
  if (services.cartesia) {
    dom.pillCartesia.innerHTML = `<span class="dot ${services.cartesia.status}"></span> Cartesia TTS`;
  }

  // Update GPU Card Info
  if (services.gpu_server) {
    dom.gpuStatusDot.className = `status-dot ${services.gpu_server.status}`;
    if (services.gpu_server.status === 'online') {
      dom.vramVal.textContent = '22.0 / 48 GB Used';
      dom.vramFill.style.width = '45%';
      dom.vramFill.style.background = 'var(--color-green)';
    } else {
      dom.vramVal.textContent = '0.0 / 48 GB';
      dom.vramFill.style.width = '0%';
    }
  }
}

// ── Conversation Flow Controls ───────────────────────────────────────────────
function onConversationStarted() {
  state.sessionActive = true;
  dom.btnStartSession.disabled = true;
  dom.btnStopSession.disabled  = false;
  dom.micBtn.disabled = false;
  dom.micLabel.textContent = 'Speak now';
}

function onConversationStopped() {
  state.sessionActive = false;
  dom.btnStartSession.disabled = false;
  dom.btnStopSession.disabled  = true;
  dom.speechModeSelect.disabled = false;
  if (state.micActive) {
    if (state.speechMode === 'web_speech') {
      stopWebSpeechRecognition();
    } else {
      stopMic();
    }
  }
  dom.micLabel.textContent = 'Disconnected';
  
  // Clean up persistent output audio context
  if (state.outAudioCtx) {
    try {
      state.outAudioCtx.close();
    } catch (e) {}
    state.outAudioCtx = null;
  }
  state.nextStartTime = 0;
  if (window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
  setSpeaking(false);
}

dom.btnStartSession.addEventListener('click', () => {
  const modalities = state.speechMode === 'web_speech' ? ['TEXT'] : ['AUDIO'];
  send({ type: 'start_conversation', modalities });
  dom.micLabel.textContent = 'Connecting...';
  dom.speechModeSelect.disabled = true;
});

dom.btnStopSession.addEventListener('click', () => {
  send({ type: 'stop_conversation' });
});

// ── Microphone Controller ────────────────────────────────────────────────────
dom.micBtn.addEventListener('click', () => {
  if (!state.sessionActive) return;
  if (state.micActive) {
    if (state.speechMode === 'web_speech') {
      stopWebSpeechRecognition();
    } else {
      stopMic();
    }
  } else {
    if (state.speechMode === 'web_speech') {
      startWebSpeechRecognition();
    } else {
      startMic();
    }
  }
});

async function startMic() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: true,
        noiseSuppression: true,
      }
    });

    state.micStream = stream;
    state.micActive = true;

    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const source = state.audioCtx.createMediaStreamSource(stream);
    const processor = state.audioCtx.createScriptProcessor(4096, 1, 1);
    
    source.connect(processor);
    processor.connect(state.audioCtx.destination);

    processor.onaudioprocess = (e) => {
      if (!state.micActive) return;
      const float32 = e.inputBuffer.getChannelData(0);
      const int16 = float32ToInt16(float32);
      const b64 = arrayBufferToBase64(int16.buffer);
      send({ type: 'audio_chunk', data: b64 });
    };

    state.micProcessor = processor;
    dom.micBtn.classList.add('active');
    dom.micLabel.textContent = 'Listening...';
    dom.transcriptArea.style.display = 'block';
  } catch (err) {
    console.error('Mic Access Denied:', err);
  }
}

function stopMic() {
  state.micActive = false;
  if (state.micProcessor) {
    state.micProcessor.disconnect();
    state.micProcessor = null;
  }
  if (state.micStream) {
    state.micStream.getTracks().forEach(track => track.stop());
    state.micStream = null;
  }
  if (state.audioCtx) {
    state.audioCtx.close();
    state.audioCtx = null;
  }
  dom.micBtn.classList.remove('active');
  dom.micLabel.textContent = 'Click to speak';
  dom.transcriptArea.style.display = 'none';
}

function float32ToInt16(float32Array) {
  const int16 = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return int16;
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function base64ToArrayBuffer(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

// ── Transcripts & Chat ────────────────────────────────────────────────────────
function onTranscript(text, isFinal) {
  dom.liveTranscript.textContent = text;
  if (isFinal && text.trim()) {
    addMessage('user', text);
    dom.liveTranscript.textContent = '';
  }
}

function addMessage(role, text) {
  const welcome = dom.conversationLog.querySelector('.welcome-chat');
  if (welcome) welcome.remove();

  const div = document.createElement('div');
  div.className = `chat-bubble ${role}`;
  div.textContent = text;
  dom.conversationLog.appendChild(div);
  dom.conversationLog.scrollTop = dom.conversationLog.scrollHeight;
}

// ── Audio Output Playback & Speaking State ─────────────────────────────────────
async function playAudio(audioB64, format, sampleRate) {
  try {
    const buffer = base64ToArrayBuffer(audioB64);
    setSpeaking(true);

    const rate = sampleRate || 24000;
    
    // Lazy-initialize single persistent output context
    if (!state.outAudioCtx || state.outAudioCtx.state === 'closed') {
      state.outAudioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: rate });
      state.nextStartTime = state.outAudioCtx.currentTime;
    }
    const ctx = state.outAudioCtx;

    // Resume suspended context (safari/chrome security policy)
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }

    if (format === 'mp3') {
      const decoded = await ctx.decodeAudioData(buffer.slice(0));
      const source = ctx.createBufferSource();
      source.buffer = decoded;
      source.connect(ctx.destination);
      
      const startTime = Math.max(ctx.currentTime, state.nextStartTime);
      source.start(startTime);
      state.nextStartTime = startTime + decoded.duration;
      
      source.onended = () => {
        if (ctx.currentTime >= state.nextStartTime - 0.05) {
          setSpeaking(false);
        }
      };
    } else {
      const int16 = new Int16Array(buffer);
      const audioBuffer = ctx.createBuffer(1, int16.length, rate);
      const channel = audioBuffer.getChannelData(0);
      for (let i = 0; i < int16.length; i++) {
        channel[i] = int16[i] / 32768.0;
      }
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      
      const startTime = Math.max(ctx.currentTime, state.nextStartTime);
      source.start(startTime);
      state.nextStartTime = startTime + audioBuffer.duration;
      
      source.onended = () => {
        if (ctx.currentTime >= state.nextStartTime - 0.05) {
          setSpeaking(false);
        }
      };
    }
  } catch (e) {
    console.error('Audio playback error:', e);
    setSpeaking(false);
  }
}

function setSpeaking(active) {
  state.isSpeaking = active;
  if (active) {
    dom.avatarScreen.classList.add('speaking');
    dom.speakingWave.classList.add('active');
  } else {
    dom.avatarScreen.classList.remove('speaking');
    dom.speakingWave.classList.remove('active');
  }
}

// ── Web Speech API Helpers (Local STT & TTS) ──────────────────────────────────
let recognition = null;

function initSpeechRecognition() {
  if (recognition) return;
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    console.warn("Speech recognition not supported in this browser.");
    return;
  }
  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = 'en-US';

  recognition.onstart = () => {
    state.micActive = true;
    dom.micBtn.classList.add('active');
    dom.micLabel.textContent = 'Listening locally...';
    dom.transcriptArea.style.display = 'block';
  };

  recognition.onresult = (event) => {
    let interimTranscript = '';
    let finalTranscript = '';

    for (let i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        finalTranscript += event.results[i][0].transcript;
      } else {
        interimTranscript += event.results[i][0].transcript;
      }
    }

    const displayText = finalTranscript || interimTranscript;
    dom.liveTranscript.textContent = displayText;

    if (finalTranscript.trim()) {
      dom.textInput.value = finalTranscript;
    }
  };

  recognition.onerror = (event) => {
    console.error("Speech recognition error:", event.error);
    stopWebSpeechRecognition();
  };

  recognition.onend = () => {
    if (state.micActive) {
      const text = dom.textInput.value.trim();
      if (text) {
        sendText();
      }
      stopWebSpeechRecognition();
    }
  };
}

function startWebSpeechRecognition() {
  initSpeechRecognition();
  if (!recognition) {
    alert("Speech recognition is not supported in this browser. Please use Chrome/Safari.");
    return;
  }
  if (window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
  try {
    recognition.start();
  } catch (e) {
    console.error(e);
  }
}

function stopWebSpeechRecognition() {
  state.micActive = false;
  dom.micBtn.classList.remove('active');
  dom.micLabel.textContent = 'Click to speak';
  dom.transcriptArea.style.display = 'none';
  if (recognition) {
    try {
      recognition.stop();
    } catch (e) {}
  }
}

// ── Speech Queue (fixes Chrome 15s break + first-word-twice bugs) ────────────
let _speechBuffer = '';
let _speechTimer  = null;
let _speechQueue  = [];      // sentences waiting to be spoken
let _speechBusy   = false;   // true while an utterance is playing

// Accumulate streaming chunks; fire speakQueue 400ms after last chunk
function accumulateSpeech(chunk) {
  _speechBuffer += chunk;
  if (_speechTimer) clearTimeout(_speechTimer);
  _speechTimer = setTimeout(() => {
    const fullText = _speechBuffer.trim();
    _speechBuffer = '';
    _speechTimer  = null;
    if (fullText) queueSpeech(fullText);
  }, 400);
}

// Split text into sentences and push them onto the queue
function queueSpeech(text) {
  // Cancel any in-flight speech from the previous response
  if (window.speechSynthesis) window.speechSynthesis.cancel();
  _speechQueue = [];
  _speechBusy  = false;

  // Split on sentence boundaries; keep the delimiter attached
  const sentences = text
    .replace(/([.!?])\s+/g, '$1|')
    .split('|')
    .map(s => s.replace(/\*\*?/g, '').replace(/\[.*?\]/g, '').trim())
    .filter(Boolean);

  _speechQueue = sentences;
  // Small delay after cancel() — fixes Chrome 'first word repeated' bug
  setTimeout(speakNext, 50);
}

function speakNext() {
  if (!window.speechSynthesis || _speechQueue.length === 0) {
    _speechBusy = false;
    setSpeaking(false);
    return;
  }
  _speechBusy = true;
  const sentence = _speechQueue.shift();

  const utterance = new SpeechSynthesisUtterance(sentence);
  const voices = window.speechSynthesis.getVoices();
  const voice  = voices.find(v => v.lang.startsWith('en') && v.name.includes('Google'))
              || voices.find(v => v.lang.startsWith('en'))
              || voices[0];
  if (voice) utterance.voice = voice;
  utterance.rate  = 1.0;
  utterance.pitch = 1.0;

  utterance.onstart = () => setSpeaking(true);
  utterance.onend   = () => speakNext();   // chain next sentence
  utterance.onerror = (e) => {
    console.warn('TTS error:', e.error);
    speakNext();                            // skip errored sentence, continue
  };

  window.speechSynthesis.speak(utterance);
}

function speakWebSpeech(text) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();

  const cleanText = text.replace(/\*\*?/g, '').replace(/\[.*?\]/g, '').trim();
  if (!cleanText) return;

  const utterance = new SpeechSynthesisUtterance(cleanText);
  const voices = window.speechSynthesis.getVoices();
  const voice = voices.find(v => v.lang.startsWith('en') && v.name.includes('Google')) || 
                voices.find(v => v.lang.startsWith('en')) || 
                voices[0];
  if (voice) {
    utterance.voice = voice;
  }

  utterance.onstart = () => {
    setSpeaking(true);
  };
  utterance.onend = () => {
    setSpeaking(false);
  };
  utterance.onerror = () => {
    setSpeaking(false);
  };

  window.speechSynthesis.speak(utterance);
}

// ── Text Messaging ───────────────────────────────────────────────────────────
dom.sendBtn.addEventListener('click', sendText);
dom.textInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendText();
});

function sendText() {
  const text = dom.textInput.value.trim();
  if (!text) return;
  if (!state.sessionActive) {
    alert('Connect the session agent first.');
    return;
  }
  addMessage('user', text);
  send({ type: 'text_message', text });
  dom.textInput.value = '';
}

// ── Modals & Switch Persona ──────────────────────────────────────────────────
dom.btnSwitchPersona.addEventListener('click', () => {
  dom.personaModal.style.display = 'flex';
});
dom.closePersonaModal.addEventListener('click', () => {
  dom.personaModal.style.display = 'none';
});
$('btnUploadPersona').addEventListener('click', () => {
  $('personaUpload').click();
});
$('personaUpload').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    dom.avatarImage.src = ev.target.result;
    dom.personaModal.style.display = 'none';
  };
  reader.readAsDataURL(file);
});

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  connectWS();
  dom.micBtn.disabled = true;
  startTelemetryLogs(); // Start console feed for default tab (Incident & Change)
  
  // Sync speech mode from dropdown
  if (dom.speechModeSelect) {
    state.speechMode = dom.speechModeSelect.value;
    dom.speechModeSelect.addEventListener('change', (e) => {
      state.speechMode = e.target.value;
      console.log('Speech engine set to:', state.speechMode);
    });
  }
});
