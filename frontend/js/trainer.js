/**
 * Trainer — Phase 1 + Phase 2
 *
 * Phase 1: Labeled dataset upload + summary visualization.
 * Phase 2: Training config, start, WebSocket live streaming,
 *          Plotly loss/accuracy charts in real-time.
 */

const Trainer = (() => {
    // ── State ──
    let datasetId = null;
    let datasetSummary = null;
    let jobId = null;
    let ws = null;           // WebSocket connection
    let totalEpochs = 30;
    let history = [];        // [{epoch, train_loss, val_loss, train_acc, val_acc}]
    let selectedPreset = 'auto';  // v0.8: selected training preset

    const CLASS_COLORS = [
        '#06b6d4', '#a78bfa', '#34d399', '#fbbf24', '#f87171',
        '#22d3ee', '#84cc16', '#e879f9', '#fb923c', '#2dd4bf',
    ];

    // ───────────────────────────────────────────────────────────
    // Initialisation
    // ───────────────────────────────────────────────────────────

    function init() {
        const dropZone = document.getElementById('train-drop-zone');
        const fileInput = document.getElementById('train-file-input');

        // Drag-and-drop
        dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', e => {
            e.preventDefault(); dropZone.classList.remove('dragover');
            if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

        // Buttons
        document.getElementById('train-reset-btn').addEventListener('click', resetTrainer);
        document.getElementById('train-next-btn').addEventListener('click', _showGuidedSection);
        document.getElementById('train-start-btn').addEventListener('click', _startTraining);

        _bindBrowseLink();
    }

    function openFilePicker() { document.getElementById('train-file-input').click(); }

    function _bindBrowseLink() {
        const link = document.querySelector('#train-drop-zone .browse-link');
        if (link) link.onclick = e => { e.stopPropagation(); document.getElementById('train-file-input').click(); };
    }

    // ───────────────────────────────────────────────────────────
    // Phase 1 — File Upload
    // ───────────────────────────────────────────────────────────

    async function handleFile(file) {
        const ext = file.name.split('.').pop().toLowerCase();
        if (!['csv', 'txt', 'zip'].includes(ext)) {
            _setStatus('train-upload-status', 'error', `Unsupported format ".${ext}". Use .csv or .zip`);
            return;
        }
        _setStatus('train-upload-status', 'loading', `<span class="spinner"></span>Parsing dataset…`);

        const form = new FormData();
        form.append('file', file);

        try {
            const aHeaders = window.Auth ? Auth.authHeaders() : {};
            const resp = await fetch(`${API_BASE}/train/upload`, { method: 'POST', body: form, headers: aHeaders });
            if (!resp.ok) { const err = await resp.json(); throw new Error(err.detail || 'Upload failed'); }
            const data = await resp.json();
            datasetId = data.dataset_id;
            datasetSummary = data;
            _setStatus('train-upload-status', 'success', `Dataset parsed: ${data.total_samples} samples, ${data.class_names.length} classes`);
            _showSummary(data);
        } catch (err) {
            _setStatus('train-upload-status', 'error', `Error: ${err.message}`);
        }
    }

    // ───────────────────────────────────────────────────────────
    // Phase 1 — Summary Rendering
    // ───────────────────────────────────────────────────────────

    function _showSummary(data) {
        document.getElementById('train-summary-section').classList.remove('hidden');
        _renderInfoBar(data);
        _renderSchemaReport(data);
        _renderDataPreview(data);
        _renderClassChart(data);
        _renderClassTable(data);
        document.getElementById('train-next-btn').disabled = false;
        document.getElementById('train-summary-section').scrollIntoView({ behavior: 'smooth' });
        // Self-Improving: refresh warm-start availability for this dataset shape
        _refreshWarmStartToggle(data);
    }

    async function _refreshWarmStartToggle(data) {
        const toggle = document.getElementById('cfg-warm-start');
        const hint = document.getElementById('warm-start-hint');
        if (!toggle) return;
        toggle.disabled = true;
        toggle.checked = false;
        try {
            const headers = window.Auth ? Auth.authHeaders() : {};
            const resp = await fetch(`${API_BASE}/models/history`, { headers });
            if (!resp.ok) return;
            const body = await resp.json();
            const ckpts = body.checkpoints || [];
            const nCh = data.n_channels || 1;
            const compatible = ckpts.find(c => (c.input_shape || {}).n_channels === nCh);
            if (compatible) {
                toggle.disabled = false;
                toggle.checked = !!compatible.is_active;
                if (hint) {
                    const isZh = (window.App && App.lang === 'zh');
                    hint.textContent = isZh
                        ? `已找到兼容 v${compatible.version}（验证准确率 ${(compatible.best_val_acc*100).toFixed(1)}%）`
                        : `Compatible v${compatible.version} found (val acc ${(compatible.best_val_acc*100).toFixed(1)}%)`;
                }
            }
        } catch (_) { /* offline / no auth → leave disabled */ }
    }

    function _renderInfoBar(data) {
        const bar = document.getElementById('train-info-bar');
        const chLabel = data.channel_detected
            ? `${data.n_channels} (auto-detected)`
            : `${data.n_channels}`;
        const items = [
            { label: 'Format',   value: data.format === 'zip_folder' ? 'ZIP (folder-per-class)' : 'CSV (labeled)' },
            { label: 'Classes',  value: data.class_names.length },
            { label: 'Samples',  value: data.total_samples.toLocaleString() },
            { label: 'Length',   value: `${data.signal_length} pts/ch` },
            { label: 'Channels', value: chLabel },
        ];
        bar.innerHTML = items.map(it =>
            `<div class="info-item"><span class="label">${it.label}:</span><span class="value">${it.value}</span></div>`
        ).join('');
    }

    function _renderSchemaReport(data) {
        const el = document.getElementById('train-schema-report');
        const isZh = App.lang === 'zh';

        const labelCol = data.label_column || 'label';
        const sigCols = data.signal_columns || [];
        const nCh = data.n_channels || 1;
        const chDetected = data.channel_detected;
        const totalCols = data.total_signal_cols || sigCols.length;
        const sigLen = data.signal_length;

        // Column role tags
        let tagsHTML = `<span class="schema-tag label">${labelCol} <span class="tag-role">${isZh ? '标签' : 'label'}</span></span>`;

        if (chDetected && nCh > 1) {
            const chMap = data.channel_map || {};
            for (const prefix of Object.keys(chMap).sort()) {
                const cols = chMap[prefix];
                const first = cols[0], last = cols[cols.length - 1];
                tagsHTML += `<span class="schema-tag channel">${first} ~ ${last} <span class="tag-role">${prefix}</span></span>`;
            }
        } else {
            // Show first few signal columns
            const show = sigCols.slice(0, 4);
            for (const col of show) {
                tagsHTML += `<span class="schema-tag signal">${col} <span class="tag-role">${isZh ? '信号' : 'signal'}</span></span>`;
            }
            if (sigCols.length > 4) {
                tagsHTML += `<span class="schema-tag signal">... +${sigCols.length - 4} ${isZh ? '列' : 'more'}</span>`;
            }
        }

        // Summary text
        let summaryLines = [];
        if (chDetected && nCh > 1) {
            summaryLines.push(isZh
                ? `<strong>${nCh}</strong> 个通道自动检测（列名前缀 ch{N}_），每通道 <strong>${sigLen}</strong> 个采样点`
                : `<strong>${nCh}</strong> channels auto-detected (column prefix ch{N}_), <strong>${sigLen}</strong> samples per channel`);
        } else {
            summaryLines.push(isZh
                ? `单通道信号，<strong>${totalCols}</strong> 个采样点（特征维度）`
                : `Single-channel signal, <strong>${totalCols}</strong> sample points (feature dimension)`);
        }
        summaryLines.push(isZh
            ? `标签列: <code>${labelCol}</code> → <strong>${data.class_names.length}</strong> 个类别，共 <strong>${data.total_samples}</strong> 个样本`
            : `Label column: <code>${labelCol}</code> → <strong>${data.class_names.length}</strong> classes, <strong>${data.total_samples}</strong> total samples`);
        summaryLines.push(isZh
            ? `CNN 输入维度: <code>(batch, ${nCh}, ${sigLen})</code>`
            : `CNN input shape: <code>(batch, ${nCh}, ${sigLen})</code>`);

        el.innerHTML = `
            <h4>${isZh ? '数据结构报告' : 'Data Schema Report'}</h4>
            <div class="schema-cols">${tagsHTML}</div>
            <div class="schema-summary-text">${summaryLines.join('<br>')}</div>
        `;
    }

    function _renderDataPreview(data) {
        const dp = data.data_preview;
        if (!dp || !dp.rows || !dp.rows.length) {
            document.getElementById('data-preview-details').style.display = 'none';
            return;
        }
        document.getElementById('data-preview-details').style.display = '';
        const cols = dp.columns;
        const headerHTML = cols.map(c => `<th>${c}</th>`).join('');
        const rowsHTML = dp.rows.map(row => {
            const cells = cols.map(c => `<td>${row[c] ?? ''}</td>`).join('');
            return `<tr>${cells}</tr>`;
        }).join('');
        document.getElementById('train-data-preview').innerHTML = `
            <table><thead><tr>${headerHTML}</tr></thead><tbody>${rowsHTML}</tbody></table>
        `;
    }

    function _renderClassChart(data) {
        const { class_names, class_counts } = data;
        const counts = class_names.map(c => class_counts[c] ?? 0);
        const colors = class_names.map((_, i) => CLASS_COLORS[i % CLASS_COLORS.length]);
        Plotly.newPlot('train-class-chart', [{
            type: 'bar', x: class_names, y: counts, marker: { color: colors },
            text: counts.map(String), textposition: 'outside',
            hovertemplate: '<b>%{x}</b><br>Samples: %{y}<extra></extra>',
        }], {
            margin: { t: 20, r: 10, b: 60, l: 50 }, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            font: { size: 12, color: '#e2e8f0' }, xaxis: { title: 'Class', tickangle: class_names.length > 6 ? -35 : 0, gridcolor: 'rgba(148,163,184,0.15)' },
            yaxis: { title: 'Sample Count', gridcolor: 'rgba(148,163,184,0.15)' }, bargap: 0.3,
        }, { responsive: true, displayModeBar: false });
    }

    function _renderClassTable(data) {
        const { class_names, class_counts, total_samples } = data;
        const colors = class_names.map((_, i) => CLASS_COLORS[i % CLASS_COLORS.length]);
        const rows = class_names.map((cls, i) => {
            const count = class_counts[cls] ?? 0;
            const pct = total_samples > 0 ? ((count / total_samples) * 100).toFixed(1) : '0.0';
            return `<tr><td><span class="class-dot" style="background:${colors[i]}"></span>${cls}</td>
                    <td>${count.toLocaleString()}</td>
                    <td><div class="confidence-bar"><div class="confidence-fill" style="width:${pct}%;background:${colors[i]}"></div></div>
                        <span style="font-size:0.8rem;color:var(--text-secondary)">${pct}%</span></td></tr>`;
        }).join('');
        document.getElementById('train-class-table').innerHTML = `
            <table class="train-table"><thead><tr><th>Class</th><th>Samples</th><th>Distribution</th></tr></thead>
            <tbody>${rows}</tbody></table>`;
    }

    // ───────────────────────────────────────────────────────────
    // v0.8 — Guided Mode Section
    // ───────────────────────────────────────────────────────────

    const _PRESET_META = {
        auto:     { icon: '⚡', badge: 'Recommended', badge_zh: '推荐' },
        fast:     { icon: '🔬', badge: null },
        thorough: { icon: '🏆', badge: 'Best Accuracy', badge_zh: '最高精度' },
        custom:   { icon: '⚙️', badge: null },
    };

    async function _showGuidedSection() {
        const sec = document.getElementById('train-guided-section');
        sec.classList.remove('hidden');

        // Render preset cards from server (or use cached presets)
        try {
            const resp = await fetch(`${API_BASE}/train/presets`);
            const data = await resp.json();
            _renderModeCards(data.presets);
        } catch (_) {
            _renderModeCards(null);
        }

        sec.scrollIntoView({ behavior: 'smooth' });

        // Kick off data quality assessment in background
        if (datasetId) _assessDataQuality(datasetId);
    }

    function _renderModeCards(presets) {
        const grid = document.getElementById('train-mode-cards');
        const isZh = App.lang === 'zh';
        const keys = ['auto', 'fast', 'thorough', 'custom'];

        if (!presets) {
            grid.innerHTML = '<p style="color:var(--text-secondary);font-size:0.85rem;">Could not load presets. Using Custom mode.</p>';
            _selectPreset('custom');
            _showConfigSection();
            return;
        }

        grid.innerHTML = keys.map(key => {
            const p = presets[key];
            const meta = _PRESET_META[key] || { icon: '◆', badge: null };
            const title = isZh ? p.label_zh : p.label_en;
            const desc = isZh ? p.description_zh : p.description_en;
            const time = isZh ? p.time_estimate_zh : p.time_estimate_en;
            const badge = meta.badge ? `<span class="mode-badge">${isZh && meta.badge_zh ? meta.badge_zh : meta.badge}</span>` : '';
            return `<div class="mode-card${key === selectedPreset ? ' selected' : ''}" data-preset="${key}" onclick="Trainer._selectPresetAndConfigure('${key}')">
                ${badge}
                <div class="mode-icon">${meta.icon}</div>
                <div class="mode-title">${title}</div>
                <div class="mode-desc">${desc}</div>
                <div class="mode-time">${time}</div>
            </div>`;
        }).join('');
    }

    function _selectPresetAndConfigure(preset) {
        _selectPreset(preset);
        // Highlight selected card
        document.querySelectorAll('.mode-card').forEach(c => {
            c.classList.toggle('selected', c.dataset.preset === preset);
        });
        // Show config section after brief delay for visual feedback
        setTimeout(() => _showConfigSection(preset), 150);
    }

    function _selectPreset(preset) {
        selectedPreset = preset;
    }

    async function _assessDataQuality(dsId) {
        const banner = document.getElementById('train-quality-banner');
        try {
            const aHeaders = window.Auth ? Auth.authHeaders() : {};
            const resp = await fetch(`${API_BASE}/train/assess?dataset_id=${dsId}`, {
                method: 'POST', headers: aHeaders,
            });
            if (!resp.ok) return;
            const data = await resp.json();
            _renderQualityBanner(banner, data);
        } catch (_) {
            // Silently skip quality assessment on error
        }
    }

    function _renderQualityBanner(banner, data) {
        const isZh = App.lang === 'zh';
        const score = data.quality_score;
        const issues = data.issues || [];

        let scoreClass = 'excellent';
        let scoreLabel = isZh ? '优秀' : 'Excellent';
        if (score < 40)      { scoreClass = 'error';   scoreLabel = isZh ? '需修复' : 'Needs Fix'; }
        else if (score < 65) { scoreClass = 'warning'; scoreLabel = isZh ? '有问题' : 'Issues'; }
        else if (score < 85) { scoreClass = 'good';    scoreLabel = isZh ? '良好' : 'Good'; }

        const scoreColor = scoreClass === 'excellent' ? 'var(--success)'
            : scoreClass === 'good' ? 'var(--accent)'
            : scoreClass === 'warning' ? 'var(--warning)' : 'var(--danger)';

        const issueIcons = { error: '✗', warning: '⚠', info: 'ℹ' };

        const issueHTML = issues.map(i => `
            <div class="quality-issue">
                <span class="issue-icon" style="color:${i.severity === 'error' ? 'var(--danger)' : i.severity === 'warning' ? 'var(--warning)' : 'var(--text-secondary)'}">${issueIcons[i.severity] || '•'}</span>
                <span>${isZh ? i.message_zh : i.message_en}</span>
            </div>`).join('');

        const noIssueText = isZh ? '数据质量良好，可以开始训练。' : 'Data quality looks good — ready to train.';

        banner.innerHTML = `
            <div class="quality-banner ${scoreClass}">
                <div class="quality-score-row">
                    <div class="quality-score-circle" style="background:${scoreColor}20;border:2px solid ${scoreColor};color:${scoreColor}">
                        ${score}
                    </div>
                    <div>
                        <strong>${isZh ? '数据质量评分' : 'Data Quality Score'}: ${score}/100 — ${scoreLabel}</strong>
                        <div style="font-size:0.8rem;color:var(--text-secondary);margin-top:0.2rem;">
                            ${data.sample_count} ${isZh ? '个样本' : 'samples'} · ${data.n_classes} ${isZh ? '个类别' : 'classes'}
                        </div>
                    </div>
                </div>
                ${issues.length > 0 ? issueHTML : `<div style="font-size:0.85rem;color:var(--text-secondary);padding-top:0.5rem;border-top:1px solid var(--border)">${noIssueText}</div>`}
            </div>`;
        banner.classList.remove('hidden');
    }

    // ───────────────────────────────────────────────────────────
    // Phase 2 — Config Section
    // ───────────────────────────────────────────────────────────

    function _showConfigSection(preset) {
        if (preset) selectedPreset = preset;
        document.getElementById('train-config-section').classList.remove('hidden');
        // Pre-populate channels from detected dataset info
        if (datasetSummary && datasetSummary.channel_detected) {
            document.getElementById('cfg-channels').value = datasetSummary.n_channels;
        } else {
            document.getElementById('cfg-channels').value = 0;
        }
        // Apply preset defaults to form (for non-custom presets, show read-only note)
        if (preset && preset !== 'custom') {
            // [epochs, lr, batch_size, auto_mode, lr_search]
            const presetDefaults = {
                auto:     [50,  0.001, 64, true,  false],
                fast:     [20,  0.001, 64, false, false],
                thorough: [100, 0.001, 32, true,  true],
            };
            const [ep, lr, bs, am, lrSearch] = presetDefaults[preset] || [30, 0.001, 64, false, false];
            document.getElementById('cfg-epochs').value = ep;
            document.getElementById('cfg-lr').value = lr;
            document.getElementById('cfg-batch').value = String(bs);
            const autoEl = document.getElementById('cfg-auto-mode');
            if (autoEl) { autoEl.checked = am; autoEl.dispatchEvent(new Event('change')); }
            const lrSearchEl = document.getElementById('cfg-lr-search');
            if (lrSearchEl) lrSearchEl.checked = !!lrSearch;
        }
        document.getElementById('train-config-section').scrollIntoView({ behavior: 'smooth' });
    }

    // ───────────────────────────────────────────────────────────
    // Phase 2 — Start Training
    // ───────────────────────────────────────────────────────────

    async function _startTraining() {
        if (!datasetId) return;

        const epochs = parseInt(document.getElementById('cfg-epochs').value) || 30;
        const lr = parseFloat(document.getElementById('cfg-lr').value) || 0.001;
        const batchSize = parseInt(document.getElementById('cfg-batch').value) || 64;
        const valSplit = parseFloat(document.getElementById('cfg-val-split').value) || 0.2;
        const nChannels = parseInt(document.getElementById('cfg-channels').value) || 0;

        totalEpochs = epochs;
        history = [];

        document.getElementById('train-start-btn').disabled = true;
        _setStatus('train-run-status', 'loading', '<span class="spinner"></span>Starting training…');

        try {
            const trainHeaders = { 'Content-Type': 'application/json', ...(window.Auth ? Auth.authHeaders() : {}) };
            const resp = await fetch(`${API_BASE}/train/start`, {
                method: 'POST',
                headers: trainHeaders,
                body: JSON.stringify({
                    dataset_id: datasetId,
                    preset: selectedPreset,
                    epochs, learning_rate: lr, batch_size: batchSize, val_split: valSplit,
                    n_channels: nChannels,
                    auto_mode: document.getElementById('cfg-auto-mode')?.checked || false,
                    early_stopping_patience: parseInt(document.getElementById('cfg-early-stop')?.value) || 10,
                    use_class_weights: document.getElementById('cfg-class-weights')?.checked !== false,
                    lr_search: document.getElementById('cfg-lr-search')?.checked || false,
                    warm_start: document.getElementById('cfg-warm-start')?.checked || false,
                }),
            });
            if (!resp.ok) { const err = await resp.json(); throw new Error(err.detail || 'Failed to start'); }
            const data = await resp.json();
            jobId = data.job_id;

            _setStatus('train-run-status', 'success', `Training started (job: ${jobId})`);
            _showDashboard();
            _connectWebSocket(jobId);
        } catch (err) {
            _setStatus('train-run-status', 'error', `Error: ${err.message}`);
            document.getElementById('train-start-btn').disabled = false;
        }
    }

    // ───────────────────────────────────────────────────────────
    // Phase 2 — Dashboard + Charts
    // ───────────────────────────────────────────────────────────

    function _showDashboard() {
        const sec = document.getElementById('train-dashboard-section');
        sec.classList.remove('hidden');
        document.getElementById('train-progress-bar').classList.remove('hidden');

        // Init empty charts
        const chartLayout = (yTitle) => ({
            margin: { t: 10, r: 10, b: 40, l: 50 },
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            font: { size: 11, color: '#e2e8f0' },
            xaxis: { title: 'Epoch', gridcolor: 'rgba(148,163,184,0.15)' },
            yaxis: { title: yTitle, gridcolor: 'rgba(148,163,184,0.15)' },
            legend: { orientation: 'h', y: 1.12 },
            showlegend: true,
        });

        Plotly.newPlot('train-loss-chart', [
            { x: [], y: [], name: 'Train Loss', mode: 'lines+markers', line: { color: '#06b6d4' } },
            { x: [], y: [], name: 'Val Loss', mode: 'lines+markers', line: { color: '#f472b6', dash: 'dash' } },
        ], chartLayout('Loss'), { responsive: true, displayModeBar: false });

        Plotly.newPlot('train-acc-chart', [
            { x: [], y: [], name: 'Train Acc', mode: 'lines+markers', line: { color: '#06b6d4' } },
            { x: [], y: [], name: 'Val Acc', mode: 'lines+markers', line: { color: '#f472b6', dash: 'dash' } },
        ], chartLayout('Accuracy'), { responsive: true, displayModeBar: false });

        document.getElementById('train-epoch-log').innerHTML = '';
        document.getElementById('train-complete-banner').classList.add('hidden');

        sec.scrollIntoView({ behavior: 'smooth' });
    }

    function _updateCharts(m) {
        // Extend Loss chart
        Plotly.extendTraces('train-loss-chart', {
            x: [[m.epoch], [m.epoch]],
            y: [[m.train_loss], [m.val_loss]],
        }, [0, 1]);

        // Extend Accuracy chart
        Plotly.extendTraces('train-acc-chart', {
            x: [[m.epoch], [m.epoch]],
            y: [[m.train_acc], [m.val_acc]],
        }, [0, 1]);

        // Progress bar
        const pct = Math.round((m.epoch / totalEpochs) * 100);
        document.getElementById('train-progress-fill').style.width = pct + '%';
        document.getElementById('train-progress-label').textContent = `${m.epoch}/${totalEpochs} (${pct}%)`;

        // Epoch log line
        const log = document.getElementById('train-epoch-log');
        const line = document.createElement('div');
        line.className = 'epoch-line';
        line.textContent = `Epoch ${m.epoch}: loss=${m.train_loss.toFixed(4)} val_loss=${m.val_loss.toFixed(4)} acc=${(m.train_acc*100).toFixed(1)}% val_acc=${(m.val_acc*100).toFixed(1)}%`;
        log.appendChild(line);
        log.scrollTop = log.scrollHeight;

        // v0.8: Update plain-language narration
        _updateNarration(m);
    }

    // ── v0.8: Plain-language training narration ──
    function _updateNarration(m) {
        const el = document.getElementById('train-narration');
        if (!el) return;
        const isZh = App.lang === 'zh';
        const parts = [];

        // Learning trend (need at least 3 epochs)
        if (history.length >= 3) {
            const prev = history[history.length - 3];
            const improving = m.val_loss < prev.val_loss - 0.001;
            const plateaued = Math.abs(m.val_loss - prev.val_loss) < 0.001;
            if (improving) {
                parts.push(isZh
                    ? `模型正在持续改善——验证损失从 ${prev.val_loss.toFixed(3)} 降至 ${m.val_loss.toFixed(3)}。`
                    : `Model is learning — val loss dropped from ${prev.val_loss.toFixed(3)} to ${m.val_loss.toFixed(3)} over the last 3 epochs.`);
            } else if (plateaued) {
                parts.push(isZh
                    ? `验证损失趋于平稳，若持续不变，早停将自动触发。`
                    : `Performance has plateaued — early stopping will trigger if this continues.`);
            }
        }

        // Accuracy status
        const valPct = (m.val_acc * 100).toFixed(1);
        if (m.val_acc >= 0.90) {
            parts.push(isZh ? `验证准确率 ${valPct}%，表现优秀。` : `Validation accuracy ${valPct}% — excellent performance.`);
        } else if (m.val_acc >= 0.75) {
            parts.push(isZh ? `验证准确率 ${valPct}%，结果良好。` : `Validation accuracy ${valPct}% — solid results.`);
        } else if (history.length > 5) {
            parts.push(isZh ? `验证准确率 ${valPct}%，仍在学习中。` : `Validation accuracy ${valPct}% — still learning.`);
        }

        // Overfitting warning
        const gap = m.train_acc - m.val_acc;
        if (gap > 0.15 && history.length > 5) {
            parts.push(isZh
                ? `注意：训练准确率（${(m.train_acc*100).toFixed(1)}%）远高于验证准确率（${valPct}%），存在过拟合迹象。`
                : `Note: training acc (${(m.train_acc*100).toFixed(1)}%) >> val acc (${valPct}%) — possible overfitting.`);
        }

        if (parts.length > 0) {
            el.textContent = parts.join(' ');
            el.classList.remove('hidden');
        }
    }

    // ───────────────────────────────────────────────────────────
    // Phase 2 — WebSocket
    // ───────────────────────────────────────────────────────────

    function _connectWebSocket(jobId) {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${protocol}//${location.host}/api/train/ws/${jobId}`;
        ws = new WebSocket(url);

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);

            if (msg.type === 'start') {
                console.log('Training started:', msg);
                totalEpochs = msg.total_epochs || totalEpochs;
            }

            if (msg.type === 'epoch') {
                history.push(msg);
                _updateCharts(msg);
            }

            // ── LR range test progress (only emitted when user opted in) ──
            if (msg.type === 'lr_search_start') {
                const isZh = (window.App && App.lang === 'zh');
                _setStatus('train-run-status', 'loading',
                    `<span class="spinner"></span>` + (isZh
                        ? `搜索最优学习率（0/${msg.total_iter || 100}）…`
                        : `Searching optimal learning rate (0/${msg.total_iter || 100})…`));
            }
            if (msg.type === 'lr_search_progress') {
                const isZh = (window.App && App.lang === 'zh');
                const lrStr = (typeof msg.lr === 'number') ? msg.lr.toExponential(2) : '?';
                _setStatus('train-run-status', 'loading',
                    `<span class="spinner"></span>` + (isZh
                        ? `搜索最优学习率（${msg.iter}/${msg.total}） · lr=${lrStr}`
                        : `Searching optimal learning rate (${msg.iter}/${msg.total}) · lr=${lrStr}`));
            }
            if (msg.type === 'lr_search_done') {
                const isZh = (window.App && App.lang === 'zh');
                const lrStr = (typeof msg.suggested_lr === 'number') ? msg.suggested_lr.toExponential(2) : '?';
                _setStatus('train-run-status', 'success', isZh
                    ? `已找到推荐学习率 lr=${lrStr}，开始训练…`
                    : `Suggested learning rate lr=${lrStr}, starting training…`);
            }

            if (msg.type === 'complete') {
                _onTrainingComplete(msg);
            }

            if (msg.type === 'error') {
                _setStatus('train-run-status', 'error', `Training error: ${msg.message}`);
                document.getElementById('train-start-btn').disabled = false;
            }
        };

        ws.onerror = () => {
            _setStatus('train-run-status', 'error', 'WebSocket connection failed');
            document.getElementById('train-start-btn').disabled = false;
        };

        ws.onclose = () => {
            ws = null;
        };
    }

    function _onTrainingComplete(msg) {
        const banner = document.getElementById('train-complete-banner');
        const bestAcc = (msg.best_val_acc * 100).toFixed(2);
        banner.innerHTML = `Training complete! Best validation accuracy: <strong>${bestAcc}%</strong> over ${msg.total_epochs} epochs. Loading results…`;
        banner.classList.remove('hidden');

        document.getElementById('train-progress-fill').style.width = '100%';
        document.getElementById('train-progress-label').textContent = `${msg.total_epochs}/${msg.total_epochs} (100%)`;

        document.getElementById('train-start-btn').disabled = false;
        document.getElementById('train-start-btn').textContent =
            App.lang === 'zh' ? '重新训练' : 'Re-train';

        // Phase 3: fetch post-training visualizations
        _loadPostTrainingResults();

        // Self-Improving: refresh My Models history (give backend a moment to persist)
        if (window.MyModels) {
            setTimeout(() => MyModels.refresh(), 1500);
        }
    }

    // ───────────────────────────────────────────────────────────
    // Phase 3 — Post-Training Visualizations
    // ───────────────────────────────────────────────────────────

    async function _loadPostTrainingResults() {
        if (!jobId) return;
        document.getElementById('train-results-section').classList.remove('hidden');

        // Fetch confusion matrix and t-SNE in parallel
        const [cmResp, tsneResp] = await Promise.allSettled([
            fetch(`${API_BASE}/train/${jobId}/confusion_matrix`).then(r => r.ok ? r.json() : Promise.reject(r)),
            fetch(`${API_BASE}/train/${jobId}/tsne`).then(r => r.ok ? r.json() : Promise.reject(r)),
        ]);

        if (cmResp.status === 'fulfilled') _renderConfusionMatrix(cmResp.value);
        if (tsneResp.status === 'fulfilled') _renderTSNE(tsneResp.value);

        document.getElementById('train-results-section').scrollIntoView({ behavior: 'smooth' });

        // Update banner
        const banner = document.getElementById('train-complete-banner');
        banner.innerHTML = banner.innerHTML.replace('Loading results…', '');

        // v0.8: Load result interpretation panel
        _loadInterpretPanel(jobId);

        // Phase 5.5: Grad-CAM attention heatmaps
        if (typeof GradCAM !== 'undefined') {
            GradCAM.fetchForTraining(jobId);
        }

        // Phase 6: Load publication-quality figures
        if (typeof Figures !== 'undefined') {
            Figures.loadPublicationFigures(jobId);
        }
    }

    async function _loadInterpretPanel(jid) {
        const sec = document.getElementById('train-interpret-section');
        const content = document.getElementById('train-interpret-content');
        if (!sec || !content) return;
        sec.classList.remove('hidden');

        try {
            const aHeaders = window.Auth ? Auth.authHeaders() : {};
            const resp = await fetch(`${API_BASE}/train/${jid}/interpret`, { headers: aHeaders });
            if (!resp.ok) { sec.classList.add('hidden'); return; }
            const d = await resp.json();
            _renderInterpretPanel(content, d);
        } catch (_) {
            sec.classList.add('hidden');
        }
    }

    function _renderInterpretPanel(el, d) {
        const isZh = App.lang === 'zh';
        const acc = (d.accuracy * 100).toFixed(2);
        const readinessBadge = `<span class="readiness-badge ${d.readiness}">${acc}% — ${d.readiness.charAt(0).toUpperCase() + d.readiness.slice(1)}</span>`;
        const nextSteps = (isZh ? d.next_steps_zh : d.next_steps_en) || [];

        el.innerHTML = `
            <div class="interpret-grid">
                <div class="interpret-card">
                    <h4 data-en="Publication Readiness" data-zh="发表就绪度">${isZh ? '发表就绪度' : 'Publication Readiness'}</h4>
                    ${readinessBadge}
                    <p>${isZh ? d.readiness_zh : d.readiness_en}</p>
                </div>
                <div class="interpret-card">
                    <h4 data-en="Training Dynamics" data-zh="训练动态">${isZh ? '训练动态' : 'Training Dynamics'}</h4>
                    <p>${isZh ? d.dynamics_zh : d.dynamics_en}</p>
                    <p style="margin-top:0.5rem;font-size:0.8rem;color:var(--text-secondary)">
                        ${isZh ? `使用了 ${d.epochs_used} 轮${d.early_stopped ? '（早停触发）' : ''}` : `${d.epochs_used} epochs used${d.early_stopped ? ' (early stopped)' : ''}`}
                    </p>
                </div>
                <div class="interpret-card">
                    <h4 data-en="Weakest Class" data-zh="最薄弱类别">${isZh ? '最薄弱类别' : 'Weakest Class'}</h4>
                    <p>${isZh ? d.worst_class_advice_zh : d.worst_class_advice_en}</p>
                </div>
                <div class="interpret-card">
                    <h4 data-en="Recommended Next Steps" data-zh="建议下一步">${isZh ? '建议下一步' : 'Recommended Next Steps'}</h4>
                    <ul class="next-steps-list">${nextSteps.map(s => `<li>${s}</li>`).join('')}</ul>
                </div>
            </div>`;
    }

    let _cmData = null;   // cached for toggle
    let _cmMode = 'count'; // 'count' | 'percent'

    function _renderConfusionMatrix(data) {
        _cmData = data;
        _cmMode = 'count';
        _drawCMChart();

        const { per_class, accuracy } = data;

        // Per-class metrics table
        const tableHTML = `
            <table class="train-table">
                <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
                <tbody>
                    ${per_class.map(c => `<tr>
                        <td>${c.class}</td>
                        <td>${(c.precision * 100).toFixed(1)}%</td>
                        <td>${(c.recall * 100).toFixed(1)}%</td>
                        <td>${(c.f1 * 100).toFixed(1)}%</td>
                        <td>${c.support}</td>
                    </tr>`).join('')}
                    <tr style="font-weight:600;border-top:2px solid var(--border)">
                        <td>Overall Accuracy</td>
                        <td colspan="4">${(accuracy * 100).toFixed(2)}%</td>
                    </tr>
                </tbody>
            </table>`;
        document.getElementById('train-metrics-table').innerHTML = tableHTML;
    }

    function _drawCMChart() {
        const { matrix, class_names } = _cmData;
        const n = class_names.length;
        const isPercent = _cmMode === 'percent';

        // Compute row-normalized percentage matrix
        const pctMatrix = matrix.map(row => {
            const sum = row.reduce((a, b) => a + b, 0);
            return row.map(v => sum > 0 ? (v / sum) * 100 : 0);
        });

        const zData = isPercent ? pctMatrix : matrix;
        const annotations = [];
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                const val = zData[i][j];
                const text = isPercent ? val.toFixed(1) + '%' : String(val);
                annotations.push({
                    x: class_names[j], y: class_names[i],
                    text,
                    showarrow: false,
                    font: { color: val > 0 ? 'white' : '#999', size: 13 },
                });
            }
        }

        Plotly.newPlot('train-cm-chart', [{
            z: zData,
            x: class_names,
            y: class_names,
            type: 'heatmap',
            colorscale: [[0, '#1e293b'], [0.5, '#6366f1'], [1, '#06b6d4']],
            showscale: false,
            hovertemplate: isPercent
                ? 'True: %{y}<br>Pred: %{x}<br>Ratio: %{z:.1f}%<extra></extra>'
                : 'True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>',
        }], {
            margin: { t: 20, r: 10, b: 60, l: 80 },
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            font: { size: 11, color: '#e2e8f0' },
            xaxis: { title: 'Predicted', side: 'bottom' },
            yaxis: { title: 'True', autorange: 'reversed' },
            annotations,
        }, { responsive: true, displayModeBar: false });
    }

    function toggleCMMode() {
        if (!_cmData) return;
        _cmMode = _cmMode === 'count' ? 'percent' : 'count';
        _drawCMChart();
        const btn = document.getElementById('cm-toggle-btn');
        if (btn) {
            btn.textContent = _cmMode === 'count' ? '% →' : '# →';
            btn.title = _cmMode === 'count' ? 'Switch to percentage' : 'Switch to count';
        }
    }

    function _renderTSNE(data) {
        const { x, y, labels, class_names } = data;

        // Group by class for color-coded scatter
        const traces = class_names.map((cls, i) => {
            const idx = labels.map((l, j) => l === cls ? j : -1).filter(j => j >= 0);
            return {
                x: idx.map(j => x[j]),
                y: idx.map(j => y[j]),
                mode: 'markers',
                type: 'scatter',
                name: cls,
                marker: { size: 7, color: CLASS_COLORS[i % CLASS_COLORS.length], opacity: 0.8 },
                hovertemplate: `${cls}<br>(%{x:.2f}, %{y:.2f})<extra></extra>`,
            };
        });

        Plotly.newPlot('train-tsne-chart', traces, {
            margin: { t: 10, r: 10, b: 40, l: 40 },
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            font: { size: 11, color: '#e2e8f0' },
            xaxis: { title: 't-SNE 1', zeroline: false, gridcolor: 'rgba(148,163,184,0.15)' },
            yaxis: { title: 't-SNE 2', zeroline: false, gridcolor: 'rgba(148,163,184,0.15)' },
            legend: { orientation: 'h', y: 1.12 },
            showlegend: true,
        }, { responsive: true, displayModeBar: false });
    }

    // ───────────────────────────────────────────────────────────
    // Helpers
    // ───────────────────────────────────────────────────────────

    function _setStatus(id, type, html) {
        const el = document.getElementById(id);
        el.className = `status ${type}`;
        el.innerHTML = html;
        el.classList.remove('hidden');
    }

    function resetTrainer() {
        datasetId = null; datasetSummary = null; jobId = null; history = [];
        selectedPreset = 'auto';
        if (ws) { ws.close(); ws = null; }
        ['train-summary-section', 'train-guided-section', 'train-config-section',
         'train-dashboard-section', 'train-results-section', 'train-interpret-section',
         'train-figures-section'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.classList.add('hidden');
        });
        const qb = document.getElementById('train-quality-banner');
        if (qb) { qb.innerHTML = ''; qb.classList.add('hidden'); }
        const nr = document.getElementById('train-narration');
        if (nr) { nr.textContent = ''; nr.classList.add('hidden'); }
        document.getElementById('train-upload-status').classList.add('hidden');
        document.getElementById('train-run-status')?.classList.add('hidden');
        document.getElementById('train-file-input').value = '';
        document.getElementById('train-next-btn').disabled = true;
        document.getElementById('train-start-btn').disabled = false;
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // ───────────────────────────────────────────────────────────
    // Phase 4 — Export Functions
    // ───────────────────────────────────────────────────────────

    function _downloadURL(url) {
        const a = document.createElement('a');
        a.href = url;
        a.download = '';
        document.body.appendChild(a);
        a.click();
        a.remove();
    }

    function exportModel()   { if (jobId) _downloadURL(`${API_BASE}/train/${jobId}/export/model`); }
    function exportHistory() { if (jobId) _downloadURL(`${API_BASE}/train/${jobId}/export/history`); }
    function exportCM()      { if (jobId) _downloadURL(`${API_BASE}/train/${jobId}/export/confusion_matrix_csv`); }
    function exportTSNE()    { if (jobId) _downloadURL(`${API_BASE}/train/${jobId}/export/tsne_csv`); }
    function exportReport()  { if (jobId) _downloadURL(`${API_BASE}/train/${jobId}/export/report`); }

    function getJobId() { return jobId; }

    /**
     * Adopt a dataset that has already been registered server-side
     * (e.g. by the prep router) and render the standard training summary.
     */
    async function loadDatasetById(id) {
        try {
            const aHeaders = window.Auth ? Auth.authHeaders() : {};
            const resp = await fetch(`${API_BASE}/train/dataset/${id}`, { headers: aHeaders });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
                throw new Error(err.detail || `HTTP ${resp.status}`);
            }
            const data = await resp.json();
            datasetId = data.dataset_id || id;
            datasetSummary = data;
            _setStatus('train-upload-status', 'success',
                `Dataset loaded: ${data.total_samples} samples, ${data.class_names.length} classes`);
            _showSummary(data);
            return true;
        } catch (err) {
            _setStatus('train-upload-status', 'error', `Load failed: ${err.message}`);
            return false;
        }
    }

    // Public
    return { init, openFilePicker, handleFile, reset: resetTrainer, _bindBrowseLink,
             exportModel, exportHistory, exportCM, exportTSNE, exportReport, toggleCMMode,
             getJobId, loadDatasetById, _selectPresetAndConfigure };
})();

window.Trainer = Trainer;
