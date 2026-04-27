/**
 * My Models — Self-Improving model history viewer.
 *
 * Lists every TrainingRun + ModelCheckpoint for the current user (or
 * sentinel id 0 in single-user mode), shows a sparkline of best_val_acc
 * over version, and exposes Activate / Delete actions.
 */
(function () {
    let lastData = null;

    function init() {
        const refreshBtn = document.getElementById('my-models-refresh');
        if (refreshBtn) refreshBtn.addEventListener('click', refresh);
    }

    function onShow() {
        refresh();
    }

    async function refresh() {
        const container = document.getElementById('my-models-list');
        if (!container) return;
        const isZh = (window.App && App.lang === 'zh');
        container.innerHTML = `<p style="color:var(--text-secondary)">${
            isZh ? '正在加载历史…' : 'Loading history…'}</p>`;

        try {
            const headers = window.Auth ? Auth.authHeaders() : {};
            const resp = await fetch(`${API_BASE}/models/history`, { headers });
            if (!resp.ok) throw new Error('Failed to load history');
            const data = await resp.json();
            lastData = data;
            _render(data);
        } catch (err) {
            container.innerHTML = `<p style="color:var(--error)">${err.message}</p>`;
        }
    }

    function _render(data) {
        const isZh = (window.App && App.lang === 'zh');
        const container = document.getElementById('my-models-list');
        const ckpts = data.checkpoints || [];
        const runs = data.runs || [];

        if (!ckpts.length) {
            container.innerHTML = `<p style="color:var(--text-secondary)">${
                isZh ? '尚无训练历史。完成一次训练后这里会显示版本演化。'
                     : 'No training history yet. Train once and your model versions will appear here.'}</p>`;
            _renderChart([]);
            return;
        }

        const runsByCkpt = {};
        for (const r of runs) {
            // Find the checkpoint produced by this run (1:1)
            const ck = ckpts.find(c => c.training_run_id === r.id);
            if (ck) runsByCkpt[ck.id] = r;
        }

        // Sort versions ascending for the sparkline; descending for the table
        const ascending = [...ckpts].sort((a, b) => a.version - b.version);
        _renderChart(ascending);

        const rowsHTML = ckpts.map(c => {
            const run = runsByCkpt[c.id];
            const accPct = ((c.best_val_acc || 0) * 100).toFixed(2);
            const shape = c.input_shape || {};
            const shapeStr = `${shape.n_channels || 1}×${shape.signal_length || '?'}`;
            const created = c.created_at ? new Date(c.created_at).toLocaleString() : '—';
            const warmFrom = run && run.warm_started_from_id;

            const activateBtn = c.is_active
                ? `<span class="badge active">${isZh ? '当前激活' : 'Active'}</span>`
                : `<button class="btn small" data-act="activate" data-id="${c.id}">${
                    isZh ? '设为激活' : 'Activate'}</button>`;
            const deleteBtn = `<button class="btn small danger" data-act="delete" data-id="${c.id}">${
                isZh ? '删除' : 'Delete'}</button>`;
            const warmTag = warmFrom
                ? `<small style="color:var(--text-secondary)">${isZh ? '热启动自' : 'warm-started from'} #${warmFrom}</small>`
                : '';

            return `<tr>
                <td><strong>v${c.version}</strong> ${warmTag}</td>
                <td>${accPct}%</td>
                <td>${c.n_classes}</td>
                <td><code>${shapeStr}</code></td>
                <td>${created}</td>
                <td style="white-space:nowrap;display:flex;gap:0.4rem;">${activateBtn}${deleteBtn}</td>
            </tr>`;
        }).join('');

        container.innerHTML = `
            <table style="width:100%;border-collapse:collapse;">
                <thead><tr>
                    <th>${isZh ? '版本' : 'Version'}</th>
                    <th>${isZh ? '验证准确率' : 'Val Acc'}</th>
                    <th>${isZh ? '类别数' : 'Classes'}</th>
                    <th>${isZh ? '输入形状' : 'Input Shape'}</th>
                    <th>${isZh ? '创建时间' : 'Created'}</th>
                    <th>${isZh ? '操作' : 'Actions'}</th>
                </tr></thead>
                <tbody>${rowsHTML}</tbody>
            </table>
        `;

        container.querySelectorAll('button[data-act]').forEach(btn => {
            btn.addEventListener('click', () => _onAction(btn.dataset.act, parseInt(btn.dataset.id)));
        });
    }

    function _renderChart(ascending) {
        const el = document.getElementById('my-models-chart');
        if (!el || typeof Plotly === 'undefined') return;
        if (!ascending.length) { el.innerHTML = ''; return; }
        const xs = ascending.map(c => `v${c.version}`);
        const ys = ascending.map(c => (c.best_val_acc || 0) * 100);
        Plotly.newPlot(el, [{
            x: xs, y: ys, mode: 'lines+markers',
            line: { color: '#06b6d4' }, marker: { size: 8, color: '#06b6d4' },
            text: ys.map(v => v.toFixed(1) + '%'),
            hovertemplate: '%{x}: %{y:.2f}%<extra></extra>',
        }], {
            margin: { t: 20, r: 10, b: 40, l: 50 },
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            font: { size: 11, color: '#e2e8f0' },
            xaxis: { title: 'Version', gridcolor: 'rgba(148,163,184,0.15)' },
            yaxis: { title: 'Val Acc (%)', gridcolor: 'rgba(148,163,184,0.15)' },
        }, { responsive: true, displayModeBar: false });
    }

    async function _onAction(act, id) {
        const isZh = (window.App && App.lang === 'zh');
        try {
            const headers = window.Auth ? Auth.authHeaders() : {};
            if (act === 'activate') {
                const resp = await fetch(`${API_BASE}/models/${id}/activate`, { method: 'POST', headers });
                if (!resp.ok) throw new Error(await resp.text());
                if (window.App) App.toast(isZh ? '已激活该版本' : 'Activated', 'info');
            } else if (act === 'delete') {
                if (!confirm(isZh ? '确认删除该 checkpoint?' : 'Delete this checkpoint?')) return;
                const resp = await fetch(`${API_BASE}/models/${id}`, { method: 'DELETE', headers });
                if (!resp.ok) throw new Error(await resp.text());
                if (window.App) App.toast(isZh ? '已删除' : 'Deleted', 'info');
            }
            await refresh();
        } catch (err) {
            if (window.App) App.toast(`Error: ${err.message}`, 'error');
        }
    }

    window.MyModels = { init, onShow, refresh };
})();
