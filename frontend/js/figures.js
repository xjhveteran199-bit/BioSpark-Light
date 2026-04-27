/**
 * Figures — Publication-quality figure preview & download.
 *
 * Shows server-rendered matplotlib figures (PNG/SVG) in a T6 panel
 * with style selection (Nature / IEEE / Science) and download buttons.
 */

const Figures = (() => {
    let _jobId = null;
    let _style = 'nature';

    function init(jobId) {
        _jobId = jobId;
    }

    function setStyle(style) {
        _style = style;
        // Update active button
        document.querySelectorAll('.fig-style-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.style === style);
        });
    }

    async function showFigurePreview(containerId, figType, params) {
        const el = document.getElementById(containerId);
        if (!el || !_jobId) return;
        el.innerHTML = '<div class="fig-loading"><span class="spinner"></span></div>';

        const qs = new URLSearchParams({ style: _style, fmt: 'png', ...(params || {}) });
        const url = `${API_BASE}/train/${_jobId}/figures/${figType}?${qs}`;

        try {
            const resp = await fetch(url);
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
                throw new Error(err.detail || `HTTP ${resp.status}`);
            }
            const blob = await resp.blob();
            const objUrl = URL.createObjectURL(blob);
            el.innerHTML = `<img src="${objUrl}" alt="${figType}" style="max-width:100%;border-radius:6px;">`;
        } catch (e) {
            el.innerHTML = `<span class="status error" style="font-size:0.85rem;">Failed: ${e.message}</span>`;
        }
    }

    function downloadFigure(figType, fmt, params) {
        if (!_jobId) return;
        const qs = new URLSearchParams({ style: _style, fmt: fmt || 'png', ...(params || {}) });
        const url = `${API_BASE}/train/${_jobId}/figures/${figType}?${qs}`;
        const a = document.createElement('a');
        a.href = url;
        a.download = `biospark_${figType}_${_jobId}.${fmt || 'png'}`;
        document.body.appendChild(a);
        a.click();
        a.remove();
    }

    async function downloadAllFigures(triggerBtn) {
        if (!_jobId) return;
        const btn = triggerBtn || (event && event.currentTarget) || null;
        const originalLabel = btn ? btn.innerHTML : null;
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> ...';
        }
        const url = `${API_BASE}/train/${_jobId}/figures/all.zip?style=${_style}`;
        try {
            const resp = await fetch(url);
            if (!resp.ok) {
                let detail = `HTTP ${resp.status}`;
                try {
                    const body = await resp.json();
                    detail = body.detail || detail;
                } catch (_) {
                    try { detail = await resp.text() || detail; } catch (_e) { /* ignore */ }
                }
                throw new Error(detail);
            }
            const blob = await resp.blob();
            if (!blob || blob.size === 0) throw new Error('Empty zip received from server.');
            const objUrl = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = objUrl;
            a.download = `biospark_figures_${_jobId}.zip`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            setTimeout(() => URL.revokeObjectURL(objUrl), 30000);
        } catch (e) {
            console.error('Download all figures failed:', e);
            const msg = `Download failed: ${e.message}`;
            if (window.App && typeof window.App.toast === 'function') {
                window.App.toast(msg, 'error');
            } else {
                alert(msg);
            }
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = originalLabel;
            }
        }
    }

    async function loadPublicationFigures(jobId) {
        _jobId = jobId;
        const sec = document.getElementById('train-figures-section');
        if (sec) sec.classList.remove('hidden');

        await Promise.allSettled([
            showFigurePreview('fig-training-curves', 'training_curves'),
            showFigurePreview('fig-confusion-matrix', 'confusion_matrix', { mode: 'both' }),
            showFigurePreview('fig-tsne', 'tsne'),
            showFigurePreview('fig-per-class', 'per_class_metrics'),
            showFigurePreview('fig-architecture', 'architecture'),
        ]);
    }

    async function refreshAll() {
        if (_jobId) await loadPublicationFigures(_jobId);
    }

    return {
        init,
        setStyle,
        loadPublicationFigures,
        downloadFigure,
        downloadAllFigures,
        refreshAll,
    };
})();

window.Figures = Figures;
