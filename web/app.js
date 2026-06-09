// ClinicalAI Discharge Summary Agent frontend logic

// Dynamically resolve API base URL
const API = window.location.protocol === 'file:'
    ? 'http://localhost:8000'
    : '';

// Global state
const state = {
    patients:             {},   // patient_name -> { preview, full_length }
    patientTexts:         {},   // patient_name -> raw_text preview
    pipelineResults:      {},   // patient_name -> full pipeline result
    savedDrafts:          [],   // from /api/drafts
    savedTraces:          [],   // from /api/traces
    savedLearningData:    {},   // from /api/learning-curve
    currentSummaryPatient: null,
    currentTracePatient:   null,
    currentLearningPatient:null,
    currentSummaryTab:    'document',
    learningChart:        null,
};

async function loadSavedArtifactState() {
    const [draftsRes, tracesRes, learningRes] = await Promise.allSettled([
        apiGet('/api/drafts'),
        apiGet('/api/traces'),
        apiGet('/api/learning-curve'),
    ]);

    if (draftsRes.status === 'fulfilled') state.savedDrafts = draftsRes.value.drafts || [];
    if (tracesRes.status === 'fulfilled') state.savedTraces = tracesRes.value.traces || [];
    if (learningRes.status === 'fulfilled') state.savedLearningData = learningRes.value.learning_data || {};

    updatePatientSelectors();
    return {
        drafts: state.savedDrafts,
        traces: state.savedTraces,
        learningData: state.savedLearningData,
    };
}

const ICONS = {
    home: '<svg viewBox="0 0 24 24"><path d="M3 11.5 12 4l9 7.5"/><path d="M5 10.5V20h14v-9.5"/><path d="M9.5 20v-5h5v5"/></svg>',
    upload: '<svg viewBox="0 0 24 24"><path d="M4 16v1a3 3 0 0 0 3 3h10a3 3 0 0 0 3-3v-1"/><path d="m16 7-4-4-4 4"/><path d="M12 3v13"/></svg>',
    agent: '<svg viewBox="0 0 24 24"><rect x="5" y="7" width="14" height="11" rx="3"/><path d="M12 7V4"/><path d="M8.5 12h.01M15.5 12h.01"/><path d="M9 16h6"/></svg>',
    draft: '<svg viewBox="0 0 24 24"><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/><path d="M8 13h8M8 17h5"/></svg>',
    chart: '<svg viewBox="0 0 24 24"><path d="M4 19V5"/><path d="M4 19h16"/><path d="m7 15 3-4 3 2 4-7"/></svg>',
    trace: '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6"/><path d="m16 16 4 4"/><path d="M8.5 11h5M11 8.5v5"/></svg>',
    key: '<svg viewBox="0 0 24 24"><circle cx="7.5" cy="14.5" r="4.5"/><path d="m11 11 9-9"/><path d="m15 6 3 3"/><path d="m17 4 3 3"/></svg>',
    check: '<svg viewBox="0 0 24 24"><path d="m5 12 4 4L19 6"/></svg>',
    alert: '<svg viewBox="0 0 24 24"><path d="M12 3 2 21h20z"/><path d="M12 9v5"/><path d="M12 17h.01"/></svg>',
    meds: '<svg viewBox="0 0 24 24"><rect x="4" y="11" width="16" height="8" rx="4"/><path d="m10 11 4 8"/><rect x="7" y="3" width="10" height="8" rx="4"/></svg>',
    lab: '<svg viewBox="0 0 24 24"><path d="M9 3h6"/><path d="M10 3v6l-5 9a2 2 0 0 0 1.7 3h10.6A2 2 0 0 0 19 18l-5-9V3"/><path d="M7 15h10"/></svg>',
    diagnostic: '<svg viewBox="0 0 24 24"><path d="M3 12h4l3-7 4 14 3-7h4"/></svg>',
    flag: '<svg viewBox="0 0 24 24"><path d="M5 21V4"/><path d="M5 4s2-1 5-1 5 2 9 1v10c-4 1-6-1-9-1s-5 1-5 1"/></svg>',
    final: '<svg viewBox="0 0 24 24"><path d="m5 12 4 4L19 6"/><circle cx="12" cy="12" r="9"/></svg>',
    patient: '<svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.4 3.6-8 8-8s8 3.6 8 8"/></svg>',
};

function icon(name, className = 'ui-icon') {
    return `<span class="${className}" aria-hidden="true">${ICONS[name] || ICONS.draft}</span>`;
}

function decorateStaticUi() {
    const tabIcons = {
        dashboard: 'home',
        upload: 'upload',
        agent: 'agent',
        summary: 'draft',
        learning: 'chart',
        trace: 'trace',
    };

    Object.entries(tabIcons).forEach(([tab, iconName]) => {
        const btn = document.querySelector(`.nav-tab[data-tab="${tab}"]`);
        if (btn && !btn.querySelector('.ui-icon')) {
            btn.innerHTML = `${icon(iconName)}<span>${btn.textContent.trim()}</span>`;
        }
    });

    const buttonIcons = {
        'hero-upload-btn': 'upload',
        'hero-drafts-btn': 'draft',
        'api-save-btn': 'key',
        'btn-doc-view': 'draft',
        'btn-diff-view': 'trace',
        'btn-print': 'final',
        'btn-download': 'upload',
    };

    Object.entries(buttonIcons).forEach(([id, iconName]) => {
        const btn = document.getElementById(id);
        if (btn && !btn.querySelector('.ui-icon')) {
            btn.innerHTML = `${icon(iconName)}<span>${btn.textContent.trim()}</span>`;
        }
    });
}

// Tab navigation
function switchTab(tabId) {
    document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));

    const section = document.getElementById(`page-${tabId}`);
    const tab     = document.querySelector(`[data-tab="${tabId}"]`);
    if (section) section.classList.add('active');
    if (tab)     tab.classList.add('active');

    // Lazy-load on tab switch
    if (tabId === 'summary')   loadSavedDrafts();
    if (tabId === 'trace')     loadSavedTraces();
    if (tabId === 'learning')  loadLearningData();
    if (tabId === 'dashboard') refreshDashboard();
}

document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
});

// Summary sub-tabs
function switchSummaryTab(subTab) {
    state.currentSummaryTab = subTab;

    const btnDoc  = document.getElementById('btn-doc-view');
    const btnDiff = document.getElementById('btn-diff-view');
    if (btnDoc)  btnDoc.classList.toggle('btn-primary',  subTab === 'document');
    if (btnDoc)  btnDoc.classList.toggle('btn-outline',  subTab !== 'document');
    if (btnDiff) btnDiff.classList.toggle('btn-primary', subTab === 'diff');
    if (btnDiff) btnDiff.classList.toggle('btn-outline', subTab !== 'diff');

    document.getElementById('summary-document-view').style.display = subTab === 'document' ? 'block' : 'none';
    document.getElementById('summary-diff-view').style.display     = subTab === 'diff'     ? 'block' : 'none';
}

// Toast notifications
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast     = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const icons = { success: icon('check', 'toast-icon'), error: icon('alert', 'toast-icon'), info: icon('agent', 'toast-icon') };
    toast.innerHTML = `${icons[type] || icons.info}<span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('removing');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// API helpers
async function apiGet(url) {
    const res = await fetch(`${API}${url}`);
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'API request failed');
    }
    return res.json();
}

async function apiPost(url, body, isFormData = false) {
    const opts = { method: 'POST' };
    if (isFormData) {
        opts.body = body;
    } else {
        opts.headers = { 'Content-Type': 'application/json' };
        opts.body    = JSON.stringify(body);
    }
    const res = await fetch(`${API}${url}`, opts);
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'API request failed');
    }
    return res.json();
}

// Dashboard
async function refreshDashboard() {
    try {
        const { drafts, traces, learningData } = await loadSavedArtifactState();

        const numPatients = getCompleteArtifactPatients().length;
        let totalSteps = 0;
        let totalFlags = 0;

        drafts.forEach(d => {
            totalFlags += (d.data.clinical_safety_flags || []).length;
        });
        traces.forEach(t => {
            totalSteps += (t.data || []).length;
        });

        let bestScore = '--';
        const ld = learningData;
        if (ld && Object.keys(ld).length > 0) {
            let minLast = Infinity;
            Object.values(ld).forEach(arr => {
                if (arr.length > 0) {
                    const last = arr[arr.length - 1];
                    if (last < minLast) minLast = last;
                }
            });
            bestScore = minLast === 0 ? '0.0000' : minLast.toFixed(4);
        }

        animateStatValue('stat-patients-val', numPatients);
        animateStatValue('stat-steps-val',    totalSteps);
        animateStatValue('stat-flags-val',    totalFlags);
        document.getElementById('stat-learning-val').textContent = bestScore;

    } catch (e) {
        console.warn('Dashboard refresh error:', e);
    }
}

function animateStatValue(elementId, targetValue) {
    const el  = document.getElementById(elementId);
    let current  = 0;
    const step   = Math.max(1, Math.ceil(targetValue / 24));
    const interval = setInterval(() => {
        current += step;
        if (current >= targetValue) {
            current = targetValue;
            clearInterval(interval);
        }
        el.textContent = current;
    }, 30);
}

// PDF upload
const uploadZone = document.getElementById('upload-zone');
const fileInput  = document.getElementById('file-input');

uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) handleFileUpload(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFileUpload(fileInput.files[0]);
});

async function handleFileUpload(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        showToast('Please upload a PDF file.', 'error');
        return;
    }

    uploadZone.innerHTML = `
        <div class="upload-zone-icon">
            <div style="width:48px;height:48px;border:3px solid #DDE8F5;border-top-color:#1A6FBA;border-radius:50%;animation:spin 0.8s linear infinite;margin:0 auto;"></div>
        </div>
        <div class="upload-zone-text" style="margin-top:16px;">Parsing clinical records...</div>
        <div class="upload-zone-hint">Extracting patient data from <strong>${escapeHtml(file.name)}</strong></div>
    `;

    try {
        const formData = new FormData();
        formData.append('file', file);
        const result = await apiPost('/api/upload-pdf', formData, true);

        state.patients = result.patients;
        Object.keys(result.patients).forEach(name => {
            state.patientTexts[name] = result.patients[name].preview;
        });

        renderPatientCards();
        showToast(` Extracted ${Object.keys(result.patients).length} patient(s) successfully.`, 'success');
    } catch (e) {
        showToast(`Upload failed: ${e.message}`, 'error');
        resetUploadZone();
    }
}

function resetUploadZone() {
    uploadZone.innerHTML = `
        <div class="upload-zone-icon">${icon('upload', 'upload-svg')}</div>
        <div class="upload-zone-text">Drag &amp; drop a clinical PDF here, or <strong style="color:var(--blue)">click to browse</strong></div>
        <div class="upload-zone-hint" style="margin-top:8px;">Supports multi-patient PDF documents &middot; PDF format only</div>
    `;
}

function renderPatientCards() {
    const container = document.getElementById('patients-container');
    const grid      = document.getElementById('patients-grid');
    container.style.display = 'block';
    grid.innerHTML = '';
    resetUploadZone();

    Object.entries(state.patients).forEach(([name, info], idx) => {
        const card = document.createElement('div');
        card.className = 'card patient-card animate-in';
        card.style.animationDelay = `${idx * 0.12}s`;
        card.innerHTML = `
            <div class="patient-card-header">
                <span class="patient-name">${icon('patient')} ${escapeHtml(name)}</span>
                <span class="patient-badge">${(info.full_length / 1000).toFixed(1)}K chars</span>
            </div>
            <div class="patient-text-preview">${escapeHtml(info.preview)}</div>
            <div class="patient-actions">
                <button class="btn btn-primary btn-sm" id="run-btn-${idx}" onclick="runFullPipeline('${escapeAttr(name)}', ${idx})">
                    ${icon('agent')} Run Full Pipeline
                </button>
                <button class="btn btn-outline btn-sm" onclick="switchTab('agent')">
                    ${icon('trace')} Monitor
                </button>
            </div>
        `;
        grid.appendChild(card);
    });
}

// Full pipeline execution

// Agent monitor
function showAgentMonitor(patientName) {
    document.getElementById('agent-empty').style.display  = 'none';
    document.getElementById('agent-live').style.display   = 'block';
    document.getElementById('agent-patient-name').textContent   = `Processing: ${patientName}`;
    document.getElementById('agent-iteration-label').textContent = 'Running 3-iteration pipeline...';
    document.getElementById('agent-timeline').innerHTML = '';
    updateAgentProgress(0, 10);
    updateAgentStatus('Initializing', 'status-observing');
}

function updateAgentProgress(current, max) {
    const pct = Math.round((current / max) * 100);
    document.getElementById('agent-progress').style.width    = `${pct}%`;
    document.getElementById('agent-step-label').textContent  = `Step ${current} / ${max}`;
    document.getElementById('agent-percent-label').textContent = `${pct}%`;
}

function updateAgentStatus(label, className) {
    const badge   = document.getElementById('agent-status-badge');
    badge.className  = `status-badge ${className}`;
    badge.textContent = ` ${label}`;
}

async function animateAgentTimeline(patientName, trace, iteration) {
    const timeline = document.getElementById('agent-timeline');
    timeline.innerHTML = '';
    document.getElementById('agent-iteration-label').textContent = `Iteration ${iteration}  Final Aligned Run`;

    const totalSteps = trace.length;

    for (let i = 0; i < totalSteps; i++) {
        const step     = trace[i];
        const toolName = extractToolName(step.action_chosen);

        if (i < totalSteps - 1) {
            updateAgentStatus(
                step.action_chosen.includes('CALL_TOOL') ? 'Executing Tool' : 'Reasoning',
                step.action_chosen.includes('CALL_TOOL') ? 'status-executing' : 'status-reasoning'
            );
        } else {
            updateAgentStatus('Finalizing', 'status-finalizing');
        }

        updateAgentProgress(i + 1, totalSteps);

        const stepEl = document.createElement('div');
        stepEl.className = 'timeline-step';
        stepEl.style.animationDelay = `${i * 0.08}s`;
        stepEl.innerHTML = `
            <div class="step-dot"></div>
            <div class="card-static step-card">
                <div class="step-header">
                    <span class="step-number">${step.step_number}</span>
                    <span class="tool-pill ${getToolClass(toolName)}">${getToolIcon(toolName)} ${toolName}</span>
                </div>
                <div class="step-reasoning">${escapeHtml(step.reasoning)}</div>
                <div class="step-result">${escapeHtml(step.result)}</div>
                <div class="step-next"> Next: ${escapeHtml(step.next_decision)}</div>
            </div>
        `;
        timeline.appendChild(stepEl);
        await sleep(280);
    }

    updateAgentStatus('Complete ', 'status-complete');
    updateAgentProgress(totalSteps, totalSteps);
}

function extractToolName(action) {
    if (action.includes('MedicationReconciliation')) return 'MedicationReconciliation';
    if (action.includes('PendingResultsCheck'))      return 'PendingResultsCheck';
    if (action.includes('DiagnosticCheck'))          return 'DiagnosticCheck';
    if (action.includes('FlagContradiction'))        return 'FlagContradiction';
    if (action.includes('FINAL_DRAFT'))              return 'FINAL_DRAFT';
    return action;
}

function getToolClass(tool) {
    if (tool.includes('Medication'))                   return 'tool-medication';
    if (tool.includes('Pending') || tool.includes('Results')) return 'tool-labs';
    if (tool.includes('Diagnostic'))                   return 'tool-diagnostic';
    if (tool.includes('Flag') || tool.includes('Contradiction')) return 'tool-conflict';
    if (tool.includes('FINAL'))                        return 'tool-finalize';
    return 'tool-medication';
}

function getToolIcon(tool) {
    if (tool.includes('Medication')) return icon('meds', 'tool-icon');
    if (tool.includes('Pending') || tool.includes('Results')) return icon('lab', 'tool-icon');
    if (tool.includes('Diagnostic')) return icon('diagnostic', 'tool-icon');
    if (tool.includes('Flag') || tool.includes('Contradiction')) return icon('flag', 'tool-icon');
    if (tool.includes('FINAL')) return icon('final', 'tool-icon');
    if (tool.includes('LOCAL') || tool.includes('EXTRACTION')) return icon('agent', 'tool-icon');
    return icon('trace', 'tool-icon');
}

// Patient selectors
function updatePatientSelectors() {
    const completeNames = getCompleteArtifactPatients();
    const summaryNames = [...completeNames];

    renderPatientSelector('summary-patient-selector', summaryNames, (name) => {
        state.currentSummaryPatient = name;
        renderSummaryForPatient(name);
    });

    const learningNames = [...completeNames];
    renderPatientSelector('learning-patient-selector', learningNames, (name) => {
        state.currentLearningPatient = name;
        renderLearningForPatient(name);
    });

    const traceNames = [...completeNames];
    renderPatientSelector('trace-patient-selector', traceNames, (name) => {
        state.currentTracePatient = name;
        renderTraceForPatient(name);
    });
}

function getCompleteArtifactPatients() {
    const draftNames = new Set(state.savedDrafts.map(d => d.data && d.data.patient_name).filter(Boolean));
    const traceNames = new Set(state.savedTraces.map(t => t.filename.replace('_trace.json', '').replace(/_/g, ' ')));
    const learningNames = new Set(Object.keys(state.savedLearningData || {}));
    const pipelineNames = new Set(Object.keys(state.pipelineResults));
    const all = new Set([...draftNames, ...traceNames, ...learningNames, ...pipelineNames]);
    return [...all].filter(name => {
        if (pipelineNames.has(name)) return true;
        return draftNames.has(name) && traceNames.has(name) && learningNames.has(name);
    });
}

function renderPatientSelector(containerId, names, onClick) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    names.forEach(name => {
        const btn = document.createElement('button');
        btn.className  = 'patient-select-btn';
        btn.innerHTML = `${icon('patient')}<span>${escapeHtml(name)}</span>`;
        btn.onclick = () => {
            container.querySelectorAll('.patient-select-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            onClick(name);
        };
        container.appendChild(btn);
    });
}

// Discharge summary viewer
async function loadSavedDrafts() {
    try {
        await loadSavedArtifactState();
        const completeNames = getCompleteArtifactPatients();
        if (state.currentSummaryPatient && !completeNames.includes(state.currentSummaryPatient)) {
            state.currentSummaryPatient = null;
        }

        if (!state.currentSummaryPatient && completeNames.length > 0) {
            const firstName = completeNames[0];
            state.currentSummaryPatient = firstName;
            renderSummaryForPatient(firstName);
            highlightPatientBtn('summary-patient-selector', firstName);
        } else if (state.currentSummaryPatient) {
            renderSummaryForPatient(state.currentSummaryPatient);
            highlightPatientBtn('summary-patient-selector', state.currentSummaryPatient);
        } else {
            renderSummaryForPatient(null);
        }
    } catch (e) {
        console.warn('Failed to load drafts:', e);
    }
}

function renderSummaryForPatient(patientName) {
    let draft  = null;
    let edited = null;

    const pipelineResult = state.pipelineResults[patientName];
    if (pipelineResult) {
        const lastIter = pipelineResult.iterations[pipelineResult.iterations.length - 1];
        draft  = lastIter.draft;
        edited = lastIter.edited;
    } else {
        const saved = state.savedDrafts.find(d => d.data.patient_name === patientName);
        if (saved) draft = saved.data;
    }

    if (!draft) {
        document.getElementById('summary-empty').style.display   = 'block';
        document.getElementById('summary-content').style.display = 'none';
        document.getElementById('summary-actions').style.display = 'none';
        return;
    }

    document.getElementById('summary-empty').style.display   = 'none';
    document.getElementById('summary-content').style.display = 'block';
    document.getElementById('summary-actions').style.display = 'flex';

    renderDischargeSummary(draft);
    if (edited) renderDiffView(draft, edited);

    // Default to document view
    switchSummaryTab(state.currentSummaryTab);
}

// Discharge summary document renderer
function renderDischargeSummary(draft) {
    const el       = document.getElementById('summary-content');
    const flags    = draft.clinical_safety_flags  || [];
    const meds     = draft.discharge_medications  || [];
    const pending  = draft.pending_results        || [];
    const procedures = draft.procedures_performed || [];
    const secondary  = draft.secondary_diagnoses  || [];
    const allergies  = draft.allergies            || [];

    const conditionClass = getConditionClass(draft.discharge_condition);

    el.innerHTML = `
        <div class="discharge-paper" id="discharge-paper-print">

            <!-- Official Header -->
            <div class="discharge-header">
                <div style="display:flex; align-items:center; gap:14px;">
                    <div style="font-size:2rem;"></div>
                    <div>
                        <div class="discharge-hospital-name">ClinicalAI Medical Center</div>
                        <div class="discharge-doc-title">Clinical Discharge Summary  Confidential</div>
                    </div>
                </div>
                <div style="margin-top:16px; font-size:0.8rem; opacity:0.65; border-top:1px solid rgba(255,255,255,0.2); padding-top:12px;">
                    Generated by ClinicalAI Discharge Summary Agent  AI-assisted, clinician-reviewed
                </div>
            </div>

            <!-- Patient Info Strip -->
            <div class="discharge-patient-strip">
                <div class="discharge-patient-field">
                    <span class="discharge-patient-field-label">Patient Name</span>
                    <span class="discharge-patient-field-value">${escapeHtml(draft.patient_name || 'N/A')}</span>
                </div>
                <div class="discharge-patient-field">
                    <span class="discharge-patient-field-label">MRN</span>
                    <span class="discharge-patient-field-value">${escapeHtml(draft.medical_record_number || 'N/A')}</span>
                </div>
                <div class="discharge-patient-field">
                    <span class="discharge-patient-field-label">Age / Gender</span>
                    <span class="discharge-patient-field-value">${escapeHtml(draft.age_and_gender || 'N/A')}</span>
                </div>
                <div class="discharge-patient-field">
                    <span class="discharge-patient-field-label">Admission Date</span>
                    <span class="discharge-patient-field-value">${escapeHtml(draft.admission_date || 'N/A')}</span>
                </div>
                <div class="discharge-patient-field">
                    <span class="discharge-patient-field-label">Discharge Date</span>
                    <span class="discharge-patient-field-value">${escapeHtml(draft.discharge_date || 'N/A')}</span>
                </div>
                <div class="discharge-patient-field">
                    <span class="discharge-patient-field-label">Discharge Condition</span>
                    <span class="discharge-condition-badge" style="display:inline-flex; align-items:center; gap:6px; padding:4px 12px; border-radius:100px; font-weight:700; font-size:0.8rem; background:${conditionClass.bg}; color:${conditionClass.color};">
                        ${conditionClass.icon} ${escapeHtml(draft.discharge_condition || 'N/A')}
                    </span>
                </div>
            </div>

            <!-- Document Body -->
            <div class="discharge-body">

                <!-- 1. Principal Diagnosis -->
                <div class="discharge-section">
                    <div class="discharge-section-title">Principal Diagnosis</div>
                    <div class="discharge-section-body" style="font-weight:600; font-size:0.95rem; color:var(--text-primary);">
                        ${escapeHtml(draft.principal_diagnosis)}
                    </div>
                </div>

                ${secondary.length > 0 ? `
                <!-- 2. Secondary Diagnoses -->
                <div class="discharge-section">
                    <div class="discharge-section-title">Secondary Diagnoses</div>
                    <ul class="bullet-list">
                        ${secondary.map(d => `<li>${escapeHtml(d)}</li>`).join('')}
                    </ul>
                </div>
                ` : ''}

                <!-- 3. Hospital Course -->
                <div class="discharge-section">
                    <div class="discharge-section-title">Hospital Course</div>
                    <div class="discharge-section-body">${escapeHtml(draft.hospital_course)}</div>
                </div>

                ${procedures.length > 0 ? `
                <!-- 4. Procedures -->
                <div class="discharge-section">
                    <div class="discharge-section-title">Procedures Performed</div>
                    <ul class="bullet-list">
                        ${procedures.map(p => `<li>${escapeHtml(p)}</li>`).join('')}
                    </ul>
                </div>
                ` : ''}

                <!-- 5. Discharge Medications -->
                <div class="discharge-section">
                    <div class="discharge-section-title">Discharge Medications</div>
                    <div style="overflow-x:auto;">
                        <table class="meds-table">
                            <thead>
                                <tr>
                                    <th>Medication</th>
                                    <th>Dosage</th>
                                    <th>Frequency</th>
                                    <th>Status</th>
                                    <th>Reconciliation Note</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${meds.map(m => `
                                    <tr>
                                        <td style="font-weight:600; color:var(--text-primary);">${escapeHtml(m.name)}</td>
                                        <td>${escapeHtml(m.dosage)}</td>
                                        <td>${escapeHtml(m.frequency)}</td>
                                        <td><span class="status-pill status-${m.status}">${m.status}</span></td>
                                        <td style="font-size:0.78rem;">${escapeHtml(m.reconciliation_note)}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- 6. Allergies -->
                <div class="discharge-section">
                    <div class="discharge-section-title">Known Allergies</div>
                    <div class="discharge-section-body">
                        ${allergies.length > 0
                            ? allergies.map(a => `<span style="display:inline-block; padding:4px 12px; background:var(--red-light); color:var(--red); border-radius:100px; font-size:0.8rem; font-weight:600; margin:2px;">${escapeHtml(a)}</span>`).join('')
                            : '<span style="color:var(--text-muted); font-style:italic;">No known allergies documented</span>'
                        }
                    </div>
                </div>

                <!-- 7. Follow-up Instructions -->
                <div class="discharge-section">
                    <div class="discharge-section-title">Follow-up Instructions</div>
                    <div class="discharge-section-body">${escapeHtml(draft.follow_up_instructions)}</div>
                </div>

                ${pending.length > 0 ? `
                <!-- 8. Pending Results -->
                <div class="discharge-section">
                    <div class="discharge-section-title" style="color:var(--amber);"> Pending Results</div>
                    ${pending.map(p => `<div class="pending-box"> ${escapeHtml(p)}</div>`).join('')}
                </div>
                ` : ''}

                ${flags.length > 0 ? `
                <!-- 9. Clinical Safety Flags -->
                <div class="discharge-section">
                    <div class="discharge-section-title" style="color:var(--red);"> Clinical Safety Flags (${flags.length})</div>
                    <div class="flags-grid">
                        ${flags.map(f => `
                            <div class="flag-card flag-${f.category}">
                                <div class="flag-category">${formatFlagCategory(f.category)}</div>
                                <div class="flag-item">${escapeHtml(f.item_involved)}</div>
                                <div class="flag-desc">${escapeHtml(f.description)}</div>
                                <div class="flag-action">Action: ${escapeHtml(f.action_taken)}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
                ` : ''}

                <!-- Footer -->
                <div style="border-top:2px solid var(--border); padding-top:16px; margin-top:8px; display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;">
                    <div style="font-size:0.72rem; color:var(--text-muted);">
                        This document was generated by AI. Clinical decisions must be verified by a licensed clinician.
                    </div>
                    <div style="font-size:0.72rem; color:var(--text-muted);">
                        Generated: ${new Date().toLocaleDateString('en-IN', { year:'numeric', month:'long', day:'numeric' })}
                    </div>
                </div>

            </div><!-- /.discharge-body -->
        </div><!-- /.discharge-paper -->
    `;
}

function getConditionClass(condition) {
    const c = (condition || '').toLowerCase();
    if (c.includes('stable'))   return { bg: '#D1FAE5', color: '#059669', icon: icon('check') };
    if (c.includes('critical')) return { bg: '#FEE2E2', color: '#DC2626', icon: icon('alert') };
    if (c.includes('guarded'))  return { bg: '#FEF3C7', color: '#D97706', icon: icon('flag') };
    if (c.includes('fair'))     return { bg: '#E0F2FE', color: '#0284C7', icon: icon('diagnostic') };
    return { bg: '#EDE9FE', color: '#7C3AED', icon: icon('patient') };
}

function formatFlagCategory(cat) {
    const map = {
        'MISSING_DATA':          `${icon('alert')} Missing Data`,
        'MEDICATION_MISMATCH':   `${icon('meds')} Medication Mismatch`,
        'CONFLICTING_DIAGNOSES': `${icon('diagnostic')} Conflicting Diagnoses`,
        'PENDING_RESULT_WARNING':`${icon('lab')} Pending Result Warning`,
    };
    return map[cat] || cat;
}

// Print & PDF export
function printSummary() {
    window.print();
}

function downloadSummaryPDF() {
    // Use print-to-PDF via browser's print dialog with PDF destination
    showToast('Opening print dialog  select "Save as PDF" to download.', 'info');
    setTimeout(() => {
        window.print();
    }, 600);
}

// Diff view
function renderDiffView(draft, edited) {
    document.getElementById('diff-empty').style.display   = 'none';
    document.getElementById('diff-content').style.display = 'grid';

    const diffEl = document.getElementById('diff-content');

    const fields = [
        { label: 'Principal Diagnosis',    key: 'principal_diagnosis' },
        { label: 'Follow-up Instructions', key: 'follow_up_instructions' },
        { label: 'Discharge Condition',    key: 'discharge_condition' },
    ];

    let leftHtml  = '<div class="diff-panel card-static diff-ai"><div class="diff-panel-title"> AI Agent Draft</div><div class="diff-content">';
    let rightHtml = '<div class="diff-panel card-static diff-doctor"><div class="diff-panel-title"> Doctor-Edited Version</div><div class="diff-content">';

    fields.forEach(f => {
        const orig = draft[f.key]  || '';
        const edit = edited[f.key] || '';

        leftHtml  += `<div style="margin-bottom:18px;"><strong style="color:var(--text-primary);font-size:0.8rem;display:block;margin-bottom:4px;">${f.label}</strong>`;
        rightHtml += `<div style="margin-bottom:18px;"><strong style="color:var(--text-primary);font-size:0.8rem;display:block;margin-bottom:4px;">${f.label}</strong>`;

        if (orig === edit) {
            leftHtml  += `<span style="color:var(--text-secondary);">${escapeHtml(orig)}</span>`;
            rightHtml += `<span style="color:var(--text-secondary);">${escapeHtml(edit)}</span>`;
        } else {
            leftHtml  += `<span style="color:var(--text-secondary);">${escapeHtml(orig)}</span>`;
            const added = findAddedText(orig, edit);
            if (added.prefix) rightHtml += `<span class="diff-highlight">${escapeHtml(added.prefix)}</span>`;
            rightHtml += `<span style="color:var(--text-secondary);">${escapeHtml(added.shared)}</span>`;
            if (added.suffix) rightHtml += `<span class="diff-highlight">${escapeHtml(added.suffix)}</span>`;
        }

        leftHtml  += '</div>';
        rightHtml += '</div>';
    });

    leftHtml  += '</div></div>';
    rightHtml += '</div></div>';
    diffEl.innerHTML = leftHtml + rightHtml;
}

function findAddedText(original, edited) {
    let prefix = '', suffix = '', shared = original;
    if (edited.startsWith(original)) {
        suffix = edited.slice(original.length); shared = original;
    } else if (edited.endsWith(original)) {
        prefix = edited.slice(0, edited.length - original.length); shared = original;
    } else if (edited.includes(original)) {
        const idx = edited.indexOf(original);
        prefix = edited.slice(0, idx); shared = original;
        suffix = edited.slice(idx + original.length);
    } else {
        shared = edited;
    }
    return { prefix, shared, suffix };
}

// Learning panel
async function loadLearningData() {
    try {
        await loadSavedArtifactState();
    } catch (e) {
        state.savedLearningData = {};
        updatePatientSelectors();
    }

    const completeNames = getCompleteArtifactPatients();
    if (state.currentLearningPatient && !completeNames.includes(state.currentLearningPatient)) {
        state.currentLearningPatient = null;
    }

    if (state.currentLearningPatient) {
        renderLearningForPatient(state.currentLearningPatient);
        highlightPatientBtn('learning-patient-selector', state.currentLearningPatient);
    } else {
        if (completeNames.length === 0) {
            showLearningEmpty('Run the full pipeline to generate the draft, trace, and learning curve together.');
            return;
        }
        const first = completeNames[0];
        state.currentLearningPatient = first;
        renderLearningForPatient(first);
        highlightPatientBtn('learning-patient-selector', first);
        return;
    }
}

function renderLearningForPatient(patientName) {
    const result = state.pipelineResults[patientName];
    if (!result) {
        renderSavedLearningOverview(patientName);
        return;
    }

    document.getElementById('learning-empty').style.display   = 'none';
    document.getElementById('learning-content').style.display = 'block';

    // Iteration cards
    const grid = document.getElementById('iterations-grid');
    grid.innerHTML = '';
    const labels = ['Baseline Generation', 'Feedback-Injected', 'Fully Aligned'];

    result.iterations.forEach((iter, idx) => {
        const isZero = iter.edit_distance === 0;
        const card   = document.createElement('div');
        card.className = 'card iteration-card animate-in';
        card.style.animationDelay = `${idx * 0.12}s`;
        card.innerHTML = `
            <div class="iteration-number">Iteration ${iter.iteration}</div>
            <div class="iteration-label">${labels[idx] || ''}</div>
            <div class="iteration-score ${isZero ? 'score-zero' : 'score-high'}">${iter.edit_distance.toFixed(4)}</div>
            <div class="iteration-status" style="color:${isZero ? 'var(--teal)' : 'var(--red)'}">
                ${isZero ? ' Perfect Alignment' : ' Corrections Needed'}
            </div>
        `;
        grid.appendChild(card);
    });

    const finalDist = result.iterations[result.iterations.length - 1].edit_distance;
    document.getElementById('perfect-alignment').style.display = finalDist === 0 ? 'block' : 'none';

    // Chart
    const allData = {};
    Object.entries(state.pipelineResults).forEach(([name, res]) => {
        allData[name] = res.iterations.map(i => i.edit_distance);
    });
    renderLearningChartFromData(allData);

    // Rules
    const rulesList = document.getElementById('rules-list');
    rulesList.innerHTML = '';
    const rules = result.final_correction_memory || [];
    if (rules.length === 0) {
        rulesList.innerHTML = '<div class="empty-state" style="padding:20px;"><div class="empty-state-text">No correction rules extracted yet.</div></div>';
    } else {
        rules.forEach(rule => {
            const card = document.createElement('div');
            card.className = 'card-static rule-card';
            card.innerHTML = `<div class="rule-icon"></div><div class="rule-text">${escapeHtml(rule)}</div>`;
            rulesList.appendChild(card);
        });
    }
}

function renderSavedLearningOverview(patientName = null) {
    const data = state.savedLearningData || {};
    const entries = Object.entries(data).filter(([, distances]) => Array.isArray(distances) && distances.length > 0);
    if (!entries.length) {
        showLearningEmpty('Run the full pipeline to generate the draft, trace, and learning curve together.');
        return;
    }

    document.getElementById('learning-empty').style.display = 'none';
    document.getElementById('learning-content').style.display = 'block';
    document.getElementById('perfect-alignment').style.display = 'none';

    const selected = patientName && data[patientName] ? [[patientName, data[patientName]]] : entries;
    const grid = document.getElementById('iterations-grid');
    grid.innerHTML = '';
    selected.forEach(([name, distances], idx) => {
        const finalScore = distances[distances.length - 1];
        const card = document.createElement('div');
        card.className = 'card iteration-card animate-in';
        card.style.animationDelay = `${idx * 0.08}s`;
        card.innerHTML = `
            <div class="iteration-number">${escapeHtml(name)}</div>
            <div class="iteration-label">${distances.length} optimization run${distances.length === 1 ? '' : 's'}</div>
            <div class="iteration-score ${finalScore === 0 ? 'score-zero' : 'score-high'}">${finalScore.toFixed(4)}</div>
            <div class="iteration-status" style="color:${finalScore === 0 ? 'var(--teal)' : 'var(--text-muted)'}">
                ${finalScore === 0 ? 'Ready for review' : 'Review suggested'}
            </div>
        `;
        grid.appendChild(card);
    });

    renderLearningChartFromData(data);
    document.getElementById('rules-list').innerHTML = `
        <div class="empty-state mini-empty">
            <div class="empty-state-text">Saved learning curve loaded</div>
            <div class="empty-state-hint">Run a fresh pipeline in this browser session to view extracted correction rules.</div>
        </div>
    `;
}

function showLearningEmpty(message) {
    document.getElementById('learning-empty').style.display = 'flex';
    document.getElementById('learning-content').style.display = 'none';
    const hint = document.getElementById('learning-empty').querySelector('.empty-state-hint');
    if (hint) hint.textContent = message;
}

function renderLearningChartFromData(data) {
    const ctx = document.getElementById('learning-chart');
    if (!ctx) return;
    if (state.learningChart) state.learningChart.destroy();
    const validEntries = Object.entries(data || {}).filter(([, distances]) => Array.isArray(distances) && distances.length > 0);
    if (!validEntries.length) {
        renderFallbackLearningChart({});
        return;
    }
    if (typeof Chart === 'undefined') {
        renderFallbackLearningChart(Object.fromEntries(validEntries));
        return;
    }
    ctx.style.display = 'block';
    const fallback = document.getElementById('learning-chart-fallback');
    if (fallback) fallback.remove();

    const datasets = [];
    const colors   = ['#1A6FBA', '#0D9488', '#D97706', '#DC2626', '#7C3AED'];
    let maxLen = 0;

    validEntries.forEach(([name, distances], idx) => {
        if (distances.length > maxLen) maxLen = distances.length;
        datasets.push({
            label: `Patient: ${name}`,
            data:  distances,
            borderColor:     colors[idx % colors.length],
            backgroundColor: colors[idx % colors.length] + '18',
            borderWidth: 3,
            pointRadius: 7,
            pointBackgroundColor: colors[idx % colors.length],
            pointBorderColor: '#fff',
            pointBorderWidth: 2,
            tension: 0.35,
            fill: true,
        });
    });

    const labels = Array.from({ length: maxLen }, (_, i) => `Run ${i + 1}`);

    state.learningChart = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 1200, easing: 'easeInOutQuart' },
            scales: {
                x: {
                    grid:  { color: 'rgba(15,23,42,0.06)' },
                    ticks: { color: '#64748B', font: { family: 'Inter', size: 12 } },
                    title: { display: true, text: 'Optimization Iterations', color: '#64748B', font: { family: 'Inter', size: 12 } }
                },
                y: {
                    min: -0.05, max: 1.05,
                    grid:  { color: 'rgba(15,23,42,0.06)' },
                    ticks: { color: '#64748B', font: { family: 'Inter', size: 12 } },
                    title: { display: true, text: 'Normalized Edit Distance', color: '#64748B', font: { family: 'Inter', size: 12 } }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#334155', font: { family: 'Inter', size: 12 }, usePointStyle: true }
                },
                tooltip: {
                    backgroundColor: '#FFFFFF',
                    titleColor: '#0F172A',
                    bodyColor:  '#334155',
                    titleFont:  { family: 'Inter', weight: '700' },
                    bodyFont:   { family: 'Inter' },
                    borderColor: '#DDE8F5',
                    borderWidth: 1,
                    boxShadow:  '0 4px 16px rgba(15,23,42,0.12)',
                }
            }
        }
    });
}

function renderFallbackLearningChart(data) {
    const canvas = document.getElementById('learning-chart');
    if (!canvas) return;
    const container = canvas.parentElement;
    canvas.style.display = 'none';
    let fallback = document.getElementById('learning-chart-fallback');
    if (!fallback) {
        fallback = document.createElement('div');
        fallback.id = 'learning-chart-fallback';
        fallback.className = 'fallback-chart';
        container.appendChild(fallback);
    }

    const entries = Object.entries(data || {}).filter(([, values]) => Array.isArray(values) && values.length);
    if (!entries.length) {
        fallback.innerHTML = '<div class="empty-state mini-empty"><div class="empty-state-text">No learning curve yet</div><div class="empty-state-hint">Run the full pipeline to generate all outputs together.</div></div>';
        return;
    }

    const colors = ['#1A6FBA', '#0D9488', '#D97706', '#DC2626', '#7C3AED'];
    const width = 760;
    const height = 260;
    const pad = 36;
    const maxLen = Math.max(...entries.map(([, values]) => values.length), 1);
    const xFor = i => pad + (maxLen === 1 ? 0 : (i / (maxLen - 1)) * (width - pad * 2));
    const yFor = value => pad + (1 - Math.max(0, Math.min(1, value))) * (height - pad * 2);

    const series = entries.map(([name, values], idx) => {
        const color = colors[idx % colors.length];
        const points = values.map((v, i) => `${xFor(i)},${yFor(v)}`).join(' ');
        const dots = values.map((v, i) => `<circle cx="${xFor(i)}" cy="${yFor(v)}" r="5" fill="${color}" stroke="#fff" stroke-width="2"><title>${name}: ${v.toFixed(4)}</title></circle>`).join('');
        return `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>${dots}`;
    }).join('');

    const legend = entries.map(([name], idx) => `<span><i style="background:${colors[idx % colors.length]}"></i>${escapeHtml(name)}</span>`).join('');
    fallback.innerHTML = `
        <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Learning curve">
            <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#DDE8F5" stroke-width="2"/>
            <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#DDE8F5" stroke-width="2"/>
            <line x1="${pad}" y1="${yFor(0.5)}" x2="${width - pad}" y2="${yFor(0.5)}" stroke="#E8F2FD" stroke-width="1"/>
            ${series}
        </svg>
        <div class="fallback-chart-legend">${legend}</div>
    `;
}

// Trace explorer
async function loadSavedTraces() {
    try {
        await loadSavedArtifactState();
        const completeNames = getCompleteArtifactPatients();
        if (state.currentTracePatient && !completeNames.includes(state.currentTracePatient)) {
            state.currentTracePatient = null;
        }

        if (!state.currentTracePatient && completeNames.length > 0) {
            const firstName = completeNames[0];
            state.currentTracePatient = firstName;
            renderTraceFromSaved(firstName);
            highlightPatientBtn('trace-patient-selector', firstName);
        } else if (state.currentTracePatient) {
            renderTraceForPatient(state.currentTracePatient);
            highlightPatientBtn('trace-patient-selector', state.currentTracePatient);
        } else {
            renderTraceAccordion([]);
        }
    } catch (e) {
        console.warn('Failed to load traces:', e);
    }
}

function renderTraceForPatient(patientName) {
    const pipelineResult = state.pipelineResults[patientName];
    if (pipelineResult) {
        const lastIter = pipelineResult.iterations[pipelineResult.iterations.length - 1];
        renderTraceAccordion(lastIter.trace);
        return;
    }
    renderTraceFromSaved(patientName);
}

function renderTraceFromSaved(patientName) {
    const slug  = patientName.replace(/ /g, '_');
    const saved = state.savedTraces.find(t => t.filename.includes(slug));
    if (saved) renderTraceAccordion(saved.data);
}

function renderTraceAccordion(traceData) {
    const accordion = document.getElementById('trace-accordion');
    const emptyEl   = document.getElementById('trace-empty');

    if (!traceData || traceData.length === 0) {
        emptyEl.style.display = 'block';
        accordion.innerHTML   = '';
        return;
    }

    emptyEl.style.display = 'none';
    accordion.innerHTML   = '';

    traceData.forEach(step => {
        const item     = document.createElement('div');
        item.className = 'trace-item';
        item.dataset.tool    = extractToolName(step.action_chosen);
        item.dataset.content = JSON.stringify(step).toLowerCase();

        const toolName = extractToolName(step.action_chosen);
        item.innerHTML = `
            <div class="trace-item-header" onclick="this.parentElement.classList.toggle('open')">
                <span class="chevron"></span>
                <span class="step-number">${step.step_number}</span>
                <span class="tool-pill ${getToolClass(toolName)}">${getToolIcon(toolName)} ${toolName}</span>
                <span style="flex:1; font-size:0.8rem; color:var(--text-muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                    ${escapeHtml(step.reasoning.substring(0, 90))}${step.reasoning.length > 90 ? '...' : ''}
                </span>
            </div>
            <div class="trace-item-body">
                <div class="trace-json">${syntaxHighlightJson(JSON.stringify(step, null, 2))}</div>
            </div>
        `;
        accordion.appendChild(item);
    });
}

document.getElementById('trace-search').addEventListener('input', filterTraces);
document.getElementById('trace-filter').addEventListener('change', filterTraces);

function filterTraces() {
    const searchTerm = document.getElementById('trace-search').value.toLowerCase();
    const filterTool = document.getElementById('trace-filter').value;

    document.querySelectorAll('#trace-accordion .trace-item').forEach(item => {
        const matchSearch = !searchTerm || item.dataset.content.includes(searchTerm);
        const matchTool   = filterTool === 'all' || item.dataset.tool === filterTool;
        item.style.display = matchSearch && matchTool ? 'block' : 'none';
    });
}

// Utility functions
function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    const div       = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
}

function escapeAttr(str) {
    return String(str).replace(/'/g, "\\'");
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function syntaxHighlightJson(json) {
    return json
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"([^"]+)":/g, '<span class="json-key">"$1"</span>:')
        .replace(/: "([^"]*)"/g, ': <span class="json-string">"$1"</span>')
        .replace(/: (\d+\.?\d*)/g, ': <span class="json-number">$1</span>');
}

function highlightPatientBtn(containerId, patientName) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.querySelectorAll('.patient-select-btn').forEach(btn => {
        if (btn.textContent.includes(patientName)) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
}

// API key panel

function toggleApiKeyPanel() {
    const card = document.getElementById('api-key-card');
    if (card) card.classList.toggle('open');
}

function openApiKeyPanel() {
    const card = document.getElementById('api-key-card');
    if (card) card.classList.add('open');
}

function toggleKeyVisibility() {
    const input = document.getElementById('api-key-input');
    const btn   = document.getElementById('api-eye-btn');
    if (!input) return;
    if (input.type === 'password') {
        input.type = 'text';
        if (btn) btn.textContent = '';
    } else {
        input.type = 'password';
        if (btn) btn.textContent = '';
    }
}

async function saveApiKey() {
    const input = document.getElementById('api-key-input');
    const btn   = document.getElementById('api-save-btn');
    if (!input) return;

    const key = input.value.trim();
    if (!key || key.length < 8) {
        showToast('Please enter a valid API key (min 8 characters).', 'error');
        return;
    }

    btn.disabled  = true;
        btn.innerHTML = '<div class="spinner"></div><span>Saving...</span>';

    try {
        await apiPost('/api/set-api-key', { api_key: key });
        showToast(' API key saved! The agent will now use your provider for processing.', 'success');
        input.value = '';
        input.type  = 'password';
        await checkConfigStatus();

        // Collapse the panel after success
        const card = document.getElementById('api-key-card');
        if (card) card.classList.remove('open');
    } catch (e) {
        showToast(`Failed to save key: ${e.message}`, 'error');
    } finally {
        btn.disabled  = false;
        btn.innerHTML = `${icon('key')}<span>Save &amp; Apply</span>`;
    }
}

async function checkConfigStatus() {
    const badge    = document.getElementById('api-status-badge');
    const dot      = document.getElementById('api-status-dot');
    const text     = document.getElementById('api-status-text');
    const subtitle = document.getElementById('api-key-subtitle');
    const card     = document.getElementById('api-key-card');

    try {
        const cfg = await apiGet('/api/config-status');

        if (cfg.has_key) {
            badge.className    = 'api-status-badge active';
            text.textContent   = `${cfg.provider}  ${cfg.model}`;
            if (subtitle) subtitle.textContent = `Active: ${cfg.provider}  ${cfg.model}`;
            if (card)  card.classList.add('has-key');
        } else {
            badge.className    = 'api-status-badge inactive';
            text.textContent   = 'No key  local mode';
            if (subtitle) subtitle.textContent = 'Add an API key to use a cloud LLM (faster & more accurate)';
            if (card)  card.classList.remove('has-key');
        }
    } catch (e) {
        badge.className  = 'api-status-badge';
        text.textContent = 'Unavailable';
    }
}

// Allow pressing Enter in the API key field to save
document.addEventListener('DOMContentLoaded', () => {
    const keyInput = document.getElementById('api-key-input');
    if (keyInput) {
        keyInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') saveApiKey();
        });
    }
    decorateStaticUi();
});

// Processing status panel

// Full pipeline step descriptions  shown progressively while the request runs
const PIPELINE_STEPS = [
    { icon: icon('upload', 'step-icon'), msg: 'Parsing clinical text and structuring patient data from the PDF...' },
    { icon: icon('agent', 'step-icon'), msg: 'Initialising the ReAct agent and loading clinical reasoning modules...' },
    { icon: icon('trace', 'step-icon'), msg: 'Agent is observing the patient record and planning its first action...' },
    { icon: icon('meds', 'step-icon'), msg: 'Running medication reconciliation and cross-checking admission vs discharge meds...' },
    { icon: icon('meds', 'step-icon'), msg: 'Comparing prescribed dosages against documented clinical indications...' },
    { icon: icon('lab', 'step-icon'), msg: 'Checking for pending lab results, cultures, and unresolved diagnostic flags...' },
    { icon: icon('diagnostic', 'step-icon'), msg: 'Running diagnostic consistency checks against documented findings...' },
    { icon: icon('flag', 'step-icon'), msg: 'Scanning for clinical safety contradictions and flagging anomalies...' },
    { icon: icon('draft', 'step-icon'), msg: 'Agent is composing the first-pass discharge summary draft (Iteration 1/3)...' },
    { icon: icon('agent', 'step-icon'), msg: 'Applying simulated doctor review policy so the agent can self-correct...' },
    { icon: icon('chart', 'step-icon'), msg: 'Measuring edit distance between AI draft and reviewed version...' },
    { icon: icon('trace', 'step-icon'), msg: 'Extracting correction rules from clinician edits to improve the next iteration...' },
    { icon: icon('chart', 'step-icon'), msg: 'Injecting learned feedback rules and starting Iteration 2/3...' },
    { icon: icon('meds', 'step-icon'), msg: 'Re-running medication reconciliation with improved context...' },
    { icon: icon('draft', 'step-icon'), msg: 'Refining discharge summary with feedback-corrected reasoning (Iteration 2/3)...' },
    { icon: icon('chart', 'step-icon'), msg: 'Doctor review policy applied again and measuring improvement in alignment...' },
    { icon: icon('final', 'step-icon'), msg: 'Starting final alignment run, Iteration 3/3...' },
    { icon: icon('draft', 'step-icon'), msg: 'Agent generating final, fully-aligned discharge summary...' },
    { icon: icon('final', 'step-icon'), msg: 'Finalising structured output and saving draft, trace, and learning data...' },
    { icon: icon('check', 'step-icon'), msg: 'Pipeline complete! Your discharge summary is ready to review.' },
];

let _processingIntervalId  = null;
let _elapsedIntervalId     = null;
let _stepIndex             = 0;
let _startTime             = null;
let _completedSteps        = [];

function showProcessingPanel(patientName) {
    const panel = document.getElementById('processing-panel');
    if (!panel) return;

    // Reset state
    _stepIndex      = 0;
    _completedSteps = [];
    _startTime      = Date.now();

    document.getElementById('processing-title').textContent   = `Processing: ${patientName}`;
    document.getElementById('processing-patient').textContent = 'The AI agent is analysing the clinical record...';
    document.getElementById('processing-steps-log').innerHTML = '';
    setProcessingMessage(PIPELINE_STEPS[0].icon, PIPELINE_STEPS[0].msg);

    panel.style.display = 'block';
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Elapsed timer  updates every second
    _elapsedIntervalId = setInterval(() => {
        const secs = Math.floor((Date.now() - _startTime) / 1000);
        const mins = Math.floor(secs / 60);
        const s    = secs % 60;
        const el   = document.getElementById('processing-elapsed');
        if (el) el.textContent = mins > 0 ? `${mins}m ${s}s` : `${s}s`;
    }, 1000);

    // Step message cycler  advances every ~9 seconds through the steps
    _processingIntervalId = setInterval(() => {
        if (_stepIndex < PIPELINE_STEPS.length - 2) {
            // Mark current as done
            const prev = PIPELINE_STEPS[_stepIndex];
            addCompletedStep(prev.icon, prev.msg);

            _stepIndex++;
            const next = PIPELINE_STEPS[_stepIndex];
            setProcessingMessage(next.icon, next.msg);
        }
    }, 9000);
}

function hideProcessingPanel() {
    clearInterval(_processingIntervalId);
    clearInterval(_elapsedIntervalId);

    // Show final success step
    const last = PIPELINE_STEPS[PIPELINE_STEPS.length - 1];
    setProcessingMessage(last.icon, last.msg);
    addCompletedStep(last.icon, 'Pipeline complete  all 3 iterations finished.');

    // Hide panel after a short delay
    setTimeout(() => {
        const panel = document.getElementById('processing-panel');
        if (panel) panel.style.display = 'none';
    }, 2800);
}

function setProcessingMessage(icon, msg) {
    const el = document.getElementById('processing-step-message');
    if (!el) return;
    // Fade in new message
    el.style.opacity   = '0';
    el.style.transform = 'translateY(6px)';
    setTimeout(() => {
        el.innerHTML       = `<span style="font-size:1.3rem;flex-shrink:0;">${icon}</span> ${escapeHtml(msg)}`;
        el.style.opacity   = '1';
        el.style.transform = 'translateY(0)';
    }, 200);
}

function addCompletedStep(icon, msg) {
    const log = document.getElementById('processing-steps-log');
    if (!log) return;

    const entry = document.createElement('div');
    entry.className = 'processing-log-entry done';
    entry.innerHTML = `
        <div class="log-dot"></div>
        <span style="font-size:1rem;">${icon}</span>
        <span>${escapeHtml(msg.length > 72 ? msg.substring(0, 72) + '...' : msg)}</span>
        <span style="margin-left:auto; font-size:0.7rem; color:var(--green); font-weight:700;"></span>
    `;
    log.appendChild(entry);
    // Auto-scroll to bottom
    log.scrollTop = log.scrollHeight;
}

// Full pipeline shows the processing panel on upload, then animates the agent monitor on completion

async function runFullPipeline(patientName, btnIdx) {
    const btn = document.getElementById(`run-btn-${btnIdx}`);
    if (btn) {
        btn.disabled  = true;
        btn.innerHTML = '<div class="spinner"></div><span>Processing...</span>';
    }

    showToast(`Starting 3-iteration pipeline for ${patientName}...`, 'info');

    // Show processing panel on upload page instead of immediately switching to agent
    showProcessingPanel(patientName);

    // Also update agent monitor in background
    switchTab('upload');
    showAgentMonitor(patientName);
    // Switch back to upload so user sees the processing panel
    document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.getElementById('page-upload').classList.add('active');
    const uploadTab = document.querySelector('[data-tab="upload"]');
    if (uploadTab) uploadTab.classList.add('active');

    try {
        const result = await apiGet(`/api/run-full-pipeline?patient_name=${encodeURIComponent(patientName)}`);
        state.pipelineResults[patientName] = result;

        hideProcessingPanel();

        const finalIteration = result.iterations[result.iterations.length - 1];

        // Now animate the agent monitor
        await sleep(400);
        switchTab('agent');
        await animateAgentTimeline(patientName, finalIteration.trace, result.iterations.length);

        showToast(` Pipeline complete for ${patientName}! Edit distance: ${finalIteration.edit_distance.toFixed(4)}`, 'success');

        updatePatientSelectors();
        state.currentSummaryPatient  = patientName;
        state.currentLearningPatient = patientName;
        state.currentTracePatient    = patientName;

    } catch (e) {
        hideProcessingPanel();
        showToast(`Pipeline failed: ${e.message}`, 'error');
        updateAgentStatus('Error', 'status-executing');
    } finally {
        if (btn) {
            btn.disabled  = false;
            btn.innerHTML = `${icon('agent')}<span>Run Full Pipeline</span>`;
        }
    }
}

// Init on page load
document.addEventListener('DOMContentLoaded', () => {
    refreshDashboard();
    checkConfigStatus();     // Show API key status on load
    openApiKeyPanel();       // Start expanded so users notice it
});

