/**
 * Data Preparation module — Mode A/B/C.
 *
 * Lets users window raw recordings into the training-ready CSV format
 * via /api/prep/inspect → /api/prep/segment → /api/prep/{id}/promote
 * (or download). Mirrors the patterns used in trainer.js.
 */

const Prep = (() => {
    let mode = null;             // 'A' | 'B' | 'C'
    let pendingFile = null;      // raw File object
    let inspectInfo = null;      // { kind, files, columns, ... } from /prep/inspect
    let preparedDatasetId = null; // returned from /prep/segment
    let preparedFilename = null;

    const PALETTE = ['#2563eb','#7c3aed','#059669','#d97706','#dc2626',
                     '#0891b2','#65a30d','#c026d3','#ea580c','#0f766e'];

    function init() {
        _bindDropZone();
        _bindFileInput();
        // Seed Mode B with one empty interval row so users see the editor
        _renderIntervalRow({ start: '', end: '', label: '' });
    }

    function onShow() {
        _bindDropZone();
        _bindFileInput();
    }

    function _bindDropZone() {
        const dz = document.getElementById('prep-drop-zone');
        if (!dz || dz.dataset.bound === '1') return;
        dz.dataset.bound = '1';
        dz.addEventListener('click', () => document.getElementById('prep-file-input').click());
        dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('dragover'); });
        dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
        dz.addEventListener('drop', (e) => {
            e.preventDefault(); dz.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        });
    }

    function _bindFileInput() {
        const inp = document.getElementById('prep-file-input');
        if (!inp || inp.dataset.bound === '1') return;
        inp.dataset.bound = '1';
        inp.addEventListener('change', (e) => {
            if (e.target.files.length) handleFile(e.target.files[0]);
        });
    }

    function selectMode(m) {
        mode = m;
        document.querySelectorAll('.prep-mode-card').forEach(c => {
            c.classList.toggle('selected', c.dataset.prepMode === m);
        });
        document.getElementById('prep-step-upload').classList.remove('hidden');
        // Reset downstream UI when mode flips
        document.getElementById('prep-step-config').classList.add('hidden');
        document.getElementById('prep-step-preview').classList.add('hidden');
        document.getElementById('prep-actions').classList.add('hidden');
        const hint = document.getElementById('prep-upload-hint');
        if (hint) {
            const map = {
                A: '(.zip with class folders)', B: '(.csv long recording)', C: '(.zip of CSVs)',
            };
            hint.textContent = map[m] || '';
        }
        _toggleModeBlocks();
    }

    function _toggleModeBlocks() {
        document.getElementById('prep-mode-b-fields').classList.toggle('hidden', mode !== 'B');
        document.getElementById('prep-mode-c-fields').classList.toggle('hidden', mode !== 'C');
    }

    async function handleFile(file) {
        const isZh = (window.App && App.lang === 'zh');
        if (!mode) {
            _setStatus('prep-upload-status', 'error',
                isZh ? '请先选择输入模式（步骤 1）。' : 'Choose a mode first.');
            return;
        }
        const ext = (file.name.split('.').pop() || '').toLowerCase();
        const wantZip = mode === 'A' || mode === 'C';
        const UNSUPPORTED_ARCHIVES = ['rar', '7z', 'tar', 'gz', 'tgz', 'bz2', 'xz'];
        if (wantZip && UNSUPPORTED_ARCHIVES.includes(ext)) {
            _setStatus('prep-upload-status', 'error', isZh
                ? `不支持 .${ext} 格式（仅支持 .zip）。请用 WinRAR / 7-Zip 重新打包为 .zip 后再上传。`
                : `.${ext} archives are not supported — only .zip is. Please re-package the folder as a .zip (use WinRAR / 7-Zip → Add to ".zip").`);
            return;
        }
        if (wantZip && ext !== 'zip') {
            _setStatus('prep-upload-status', 'error', isZh
                ? `模式 ${mode} 需要 .zip 文件（当前为 .${ext}）。`
                : `Mode ${mode} expects a .zip file (got .${ext}).`);
            return;
        }
        if (mode === 'B' && !(ext === 'csv' || ext === 'txt')) {
            _setStatus('prep-upload-status', 'error', isZh
                ? `模式 B 需要 .csv 文件（当前为 .${ext}）。`
                : `Mode B expects a .csv file (got .${ext}).`);
            return;
        }
        pendingFile = file;
        preparedDatasetId = null;
        _setStatus('prep-upload-status', 'loading', `Inspecting ${file.name}…`);

        try {
            const fd = new FormData();
            fd.append('file', file);
            const aHeaders = window.Auth ? Auth.authHeaders() : {};
            const resp = await fetch(`${API_BASE}/prep/inspect`, { method: 'POST', body: fd, headers: aHeaders });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
                throw new Error(err.detail || `HTTP ${resp.status}`);
            }
            inspectInfo = await resp.json();
            _setStatus('prep-upload-status', 'success',
                `${inspectInfo.kind.toUpperCase()} · ${inspectInfo.files.length} file(s) · first file: ${inspectInfo.row_count} rows × ${inspectInfo.columns.length} cols`);
            _prefillForm();
            document.getElementById('prep-step-config').classList.remove('hidden');
        } catch (err) {
            _setStatus('prep-upload-status', 'error', `Inspect failed: ${err.message}`);
        }
    }

    function _prefillForm() {
        if (inspectInfo.suggested_sampling_rate) {
            document.getElementById('prep-sr').value = inspectInfo.suggested_sampling_rate;
        }
        document.getElementById('prep-sigcol').value = inspectInfo.suggested_signal_col || 0;

        if (mode === 'C') _renderFileMap();
    }

    function _renderFileMap() {
        const tbody = document.getElementById('prep-filemap-body');
        if (!inspectInfo || !inspectInfo.files) return;
        tbody.innerHTML = inspectInfo.files.map((f, i) => `
            <tr>
                <td style="padding:0.4rem;font-family:monospace;font-size:0.85rem;">${_escape(f.path)}</td>
                <td style="padding:0.4rem;">${f.rows}</td>
                <td style="padding:0.4rem;">
                    <input type="text" class="prep-filelabel" data-path="${_escape(f.path)}"
                           placeholder="ClassName" style="width:100%;">
                </td>
            </tr>`).join('');
    }

    // ── Mode B intervals editor ─────────────────────────────────
    function addInterval() {
        _renderIntervalRow({ start: '', end: '', label: '' });
    }

    function _renderIntervalRow({ start, end, label }) {
        const tbody = document.getElementById('prep-intervals-body');
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="padding:0.4rem;"><input type="number" class="prep-iv-start" value="${start}" step="0.1" style="width:90%;"></td>
            <td style="padding:0.4rem;"><input type="number" class="prep-iv-end"   value="${end}"   step="0.1" style="width:90%;"></td>
            <td style="padding:0.4rem;"><input type="text"   class="prep-iv-label" value="${label}" style="width:90%;" placeholder="ClassName"></td>
            <td style="padding:0.4rem;"><button class="btn" type="button" onclick="this.closest('tr').remove()">×</button></td>`;
        tbody.appendChild(tr);
    }

    function _collectIntervals() {
        return Array.from(document.querySelectorAll('#prep-intervals-body tr')).map(tr => ({
            start_sec: parseFloat(tr.querySelector('.prep-iv-start').value),
            end_sec:   parseFloat(tr.querySelector('.prep-iv-end').value),
            label:     tr.querySelector('.prep-iv-label').value.trim(),
        })).filter(iv => iv.label && !Number.isNaN(iv.start_sec) && !Number.isNaN(iv.end_sec));
    }

    function _collectFileMap() {
        const map = {};
        document.querySelectorAll('.prep-filelabel').forEach(inp => {
            const v = inp.value.trim();
            if (v) map[inp.dataset.path] = v;
        });
        return map;
    }

    // ── Run segmentation ────────────────────────────────────────
    async function runSegmentation() {
        if (!pendingFile || !mode) {
            _setStatus('prep-run-status', 'error', 'Upload a file first.');
            return;
        }
        const config = {
            mode,
            sampling_rate:      parseFloat(document.getElementById('prep-sr').value),
            segment_length_sec: parseFloat(document.getElementById('prep-seglen').value),
            overlap_ratio:      parseFloat(document.getElementById('prep-overlap').value) || 0,
            signal_col_index:   parseInt(document.getElementById('prep-sigcol').value, 10) || 0,
        };
        if (mode === 'B') {
            config.intervals = _collectIntervals();
            if (!config.intervals.length) {
                _setStatus('prep-run-status', 'error', 'Add at least one labeled interval.');
                return;
            }
        }
        if (mode === 'C') {
            config.file_label_map = _collectFileMap();
            if (!Object.keys(config.file_label_map).length) {
                _setStatus('prep-run-status', 'error', 'Assign at least one file a label.');
                return;
            }
        }

        const btn = document.getElementById('prep-run-btn');
        btn.disabled = true;
        _setStatus('prep-run-status', 'loading', 'Generating training CSV…');

        try {
            const fd = new FormData();
            fd.append('file', pendingFile);
            fd.append('config', JSON.stringify(config));
            const aHeaders = window.Auth ? Auth.authHeaders() : {};
            const resp = await fetch(`${API_BASE}/prep/segment`, { method: 'POST', body: fd, headers: aHeaders });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
                throw new Error(typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail));
            }
            const data = await resp.json();
            preparedDatasetId = data.dataset_id;
            preparedFilename = data.filename;
            _setStatus('prep-run-status', 'success',
                `Generated ${data.row_count} samples across ${data.summary.class_names.length} classes.`);
            _renderPreview(data);
            document.getElementById('prep-step-preview').classList.remove('hidden');
            document.getElementById('prep-actions').classList.remove('hidden');
            document.getElementById('prep-step-preview').scrollIntoView({ behavior: 'smooth' });
        } catch (err) {
            _setStatus('prep-run-status', 'error', `Failed: ${err.message}`);
        } finally {
            btn.disabled = false;
        }
    }

    function _renderPreview(data) {
        const s = data.summary;
        const bar = document.getElementById('prep-summary-bar');
        bar.innerHTML = `
            <strong>${data.row_count.toLocaleString()}</strong> samples ·
            <strong>${s.class_names.length}</strong> classes ·
            segment length <strong>${s.signal_length}</strong> samples ·
            channels <strong>${s.n_channels}</strong>`;

        // Class distribution chart
        const counts = s.class_names.map(c => s.class_counts[c] || 0);
        const colors = s.class_names.map((_, i) => PALETTE[i % PALETTE.length]);
        if (window.Plotly) {
            Plotly.newPlot('prep-class-chart', [{
                type: 'bar', x: s.class_names, y: counts, marker: { color: colors },
                text: counts.map(String), textposition: 'outside',
            }], {
                margin: { t: 20, r: 10, b: 60, l: 50 },
                paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                font: { size: 12, color: '#e2e8f0' },
                xaxis: { title: 'Class', gridcolor: 'rgba(148,163,184,0.15)' },
                yaxis: { title: 'Samples', gridcolor: 'rgba(148,163,184,0.15)' },
                bargap: 0.3,
            }, { responsive: true, displayModeBar: false });
        }

        // Class table
        const total = data.row_count || 1;
        const rows = s.class_names.map((cls, i) => {
            const c = s.class_counts[cls] || 0;
            const pct = ((c / total) * 100).toFixed(1);
            return `<tr><td><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${colors[i]};margin-right:6px;"></span>${_escape(cls)}</td>
                    <td>${c}</td><td>${pct}%</td></tr>`;
        }).join('');
        document.getElementById('prep-class-table').innerHTML = `
            <table class="train-table" style="width:100%;"><thead><tr><th>Class</th><th>Samples</th><th>Share</th></tr></thead>
            <tbody>${rows}</tbody></table>`;

        // Data preview (first 10 rows × first 6 signal cols + label)
        const previewCols = ['s1','s2','s3','s4','s5','s6'].filter(c => data.preview_rows[0] && c in data.preview_rows[0]);
        const tableRows = data.preview_rows.map(r => `<tr>${
            previewCols.map(c => `<td>${(typeof r[c] === 'number') ? r[c].toFixed(4) : _escape(String(r[c] ?? ''))}</td>`).join('')
        }<td>...</td><td>${_escape(String(r.label ?? ''))}</td></tr>`).join('');
        document.getElementById('prep-data-preview').innerHTML = `
            <table class="train-table" style="width:100%;font-size:0.85rem;">
                <thead><tr>${previewCols.map(c => `<th>${c}</th>`).join('')}<th>…</th><th>label</th></tr></thead>
                <tbody>${tableRows}</tbody></table>`;
    }

    // ── Final actions ───────────────────────────────────────────
    async function useForTraining(btn) {
        if (!preparedDatasetId) return;
        const original = btn ? btn.innerHTML : null;
        if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> ...'; }
        try {
            const aHeaders = window.Auth ? Auth.authHeaders() : {};
            const resp = await fetch(`${API_BASE}/prep/${preparedDatasetId}/promote`,
                                     { method: 'POST', headers: aHeaders });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
                throw new Error(err.detail || `HTTP ${resp.status}`);
            }
            // Switch to Train mode and load dataset there
            App.switchMode('train');
            if (window.Trainer && Trainer.loadDatasetById) {
                await Trainer.loadDatasetById(preparedDatasetId);
            }
            App.toast('Dataset ready in Train tab.');
        } catch (err) {
            App.toast(`Promote failed: ${err.message}`, 'error');
        } finally {
            if (btn) { btn.disabled = false; btn.innerHTML = original; }
        }
    }

    function downloadCsv() {
        if (!preparedDatasetId) return;
        const url = `${API_BASE}/prep/${preparedDatasetId}/download`;
        // Simple anchor download (no auth header needed for the download link)
        const a = document.createElement('a');
        a.href = url;
        a.download = preparedFilename || 'biospark_prepped.csv';
        document.body.appendChild(a);
        a.click();
        a.remove();
    }

    // ── Helpers ─────────────────────────────────────────────────
    function _setStatus(id, kind, msg) {
        const el = document.getElementById(id);
        if (!el) return;
        el.classList.remove('hidden', 'success', 'error', 'loading');
        el.classList.add(kind);
        el.className = `status ${kind}`;
        el.innerHTML = (kind === 'loading' ? '<span class="spinner"></span>' : '') + _escape(msg);
    }

    function _escape(s) {
        return String(s).replace(/[&<>"']/g, c => ({
            '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
        }[c]));
    }

    return {
        init, onShow, selectMode, handleFile,
        addInterval, runSegmentation,
        useForTraining, downloadCsv,
    };
})();

window.Prep = Prep;
