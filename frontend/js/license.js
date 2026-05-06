/**
 * BioSpark-Light — License module
 *
 * Handles:
 *   - Fetching /api/license/status on startup
 *   - Rendering the license banner below the tab bar
 *   - Showing the paywall modal when trial is exhausted
 *   - Activating a license key via /api/license/activate
 *   - Intercepting HTTP 402 responses from the training API
 */

const License = {
    _status: null,

    // ---------------------------------------------------------------- init

    async init() {
        this._intercept402();
        await this.refresh();
    },

    // --------------------------------------------------- status & banner

    async refresh() {
        try {
            const r = await window._bslFetch(`${window.API_BASE}/license/status`);
            if (!r.ok) return;
            this._status = await r.json();
            this._renderBanner();
        } catch (_) {
            // Server not yet ready on very first paint — silently ignore.
        }
    },

    _renderBanner() {
        const banner = document.getElementById('license-banner');
        if (!banner) return;
        const s = this._status;
        if (!s) { banner.classList.add('hidden'); return; }

        const zh = window.App && App.lang === 'zh';

        if (s.status === 'licensed') {
            const exp = s.expiry || '—';
            banner.className = 'license-banner license-banner--licensed';
            banner.innerHTML = zh
                ? `✅ 已授权 &nbsp;·&nbsp; ${s.email || ''} &nbsp;·&nbsp; 有效期至 ${exp}`
                : `✅ Licensed &nbsp;·&nbsp; ${s.email || ''} &nbsp;·&nbsp; expires ${exp}`;
        } else if (s.status === 'trial') {
            const left = Math.max(0, s.runs_limit - s.runs_used);
            banner.className = 'license-banner license-banner--trial';
            banner.innerHTML = zh
                ? `🔓 试用模式 &nbsp;·&nbsp; 还剩 <strong>${left}</strong> 次训练机会`
                  + `&emsp;<a class="license-banner-link" href="#" onclick="License.showPaywall(event)">`
                  + `已有激活码？</a>`
                : `🔓 Trial &nbsp;·&nbsp; <strong>${left}</strong> training run${left !== 1 ? 's' : ''} remaining`
                  + `&emsp;<a class="license-banner-link" href="#" onclick="License.showPaywall(event)">`
                  + `Have a key?</a>`;
        } else {
            // expired
            banner.className = 'license-banner license-banner--expired';
            banner.innerHTML = zh
                ? `⚠️ 试用次数已用完 &emsp;<a class="license-banner-link" href="#" onclick="License.showPaywall(event)">激活授权码</a>`
                : `⚠️ Trial ended &emsp;<a class="license-banner-link" href="#" onclick="License.showPaywall(event)">Activate license</a>`;
        }
        banner.classList.remove('hidden');
    },

    // ---------------------------------------------------- paywall modal

    showPaywall(e) {
        if (e && e.preventDefault) e.preventDefault();
        const modal = document.getElementById('license-modal');
        if (modal) {
            modal.classList.remove('hidden');
            // Clear previous state
            const inp = document.getElementById('license-key-input');
            const msg = document.getElementById('license-activate-msg');
            if (inp) inp.value = '';
            if (msg) { msg.textContent = ''; msg.className = 'license-activate-msg hidden'; }
        }
        if (window.App) App.applyLang();
    },

    closeModal() {
        document.getElementById('license-modal')?.classList.add('hidden');
    },

    handleBackdropClick(e) {
        if (e.target && e.target.id === 'license-modal') this.closeModal();
    },

    // --------------------------------------------------- key activation

    async activate() {
        const inp = document.getElementById('license-key-input');
        const msg = document.getElementById('license-activate-msg');
        const key = inp?.value?.trim() || '';

        if (!key) {
            this._setMsg(msg, '请输入激活码 / Please enter a license key.', 'error');
            return;
        }

        this._setMsg(msg, '验证中… / Verifying…', 'info');

        try {
            const r = await window._bslFetch(`${window.API_BASE}/license/activate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ key }),
            });
            const data = await r.json();
            if (r.ok && data.success) {
                this._setMsg(msg, data.message, 'success');
                await this.refresh();
                setTimeout(() => this.closeModal(), 1800);
            } else {
                const errMsg = data.detail || data.message || '激活失败 / Activation failed.';
                this._setMsg(msg, errMsg, 'error');
            }
        } catch (_) {
            this._setMsg(msg, '网络错误，请重试 / Network error, please retry.', 'error');
        }
    },

    _setMsg(el, text, kind) {
        if (!el) return;
        el.textContent = text;
        el.className = `license-activate-msg license-msg-${kind}`;
    },

    // ---------------------------------------- global 402 interceptor

    _intercept402() {
        // Wrap the native fetch so any 402 from /api/training/start
        // automatically triggers the paywall modal.
        const orig = window.fetch.bind(window);
        window._bslFetch = orig; // preserve for License's own calls

        window.fetch = async function (...args) {
            const response = await orig(...args);
            if (response.status === 402) {
                try {
                    const clone = response.clone();
                    const body = await clone.json();
                    if (body?.detail?.paywall && window.License) {
                        License.showPaywall();
                    }
                } catch (_) { /* ignore parse errors */ }
            }
            return response;
        };
    },
};

window.License = License;
