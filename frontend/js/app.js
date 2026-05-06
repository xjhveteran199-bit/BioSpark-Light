/**
 * BioSpark-Light — slim app controller.
 *
 * Just three concerns:
 *   1. Mode switch between #prep-mode / #train-mode / #my-models-mode
 *   2. i18n via data-en / data-zh attributes
 *   3. Boot the Trainer / Prep / MyModels modules
 *
 * No auth, no analyze flow, no model registry, no streaming.
 */

const API_BASE = `${window.location.origin}/api`;
window.API_BASE = API_BASE;

const App = {
    lang: 'en',
    mode: 'prep', // 'prep' | 'train' | 'my-models'

    init() {
        if (window.Trainer) Trainer.init();
        if (window.Prep) Prep.init();
        if (window.MyModels) MyModels.init();
        if (window.Donate) Donate.init();
        if (window.License) License.init();

        // Detect browser language; remember user's manual choice in localStorage
        const stored = localStorage.getItem('biospark-lang');
        if (stored === 'en' || stored === 'zh') {
            this.lang = stored;
        } else if (navigator.language && navigator.language.startsWith('zh')) {
            this.lang = 'zh';
        }
        this.applyLang();

        // Default mode = prep (first thing a new user does)
        this.switchMode('prep');
    },

    switchMode(mode) {
        this.mode = mode;
        const map = {
            prep: 'prep-mode',
            train: 'train-mode',
            'my-models': 'my-models-mode',
        };
        Object.entries(map).forEach(([m, id]) => {
            const el = document.getElementById(id);
            if (el) el.classList.toggle('hidden', m !== mode);
        });
        document.querySelectorAll('.mode-tab').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });

        // Trainer's drop-zone browse link needs re-binding after innerHTML toggles
        if (mode === 'train' && window.Trainer && Trainer._bindBrowseLink) {
            Trainer._bindBrowseLink();
        }
        if (mode === 'prep' && window.Prep && Prep.onShow) Prep.onShow();
        if (mode === 'my-models' && window.MyModels && MyModels.onShow) MyModels.onShow();
    },

    toggleLang() {
        this.lang = this.lang === 'en' ? 'zh' : 'en';
        localStorage.setItem('biospark-lang', this.lang);
        this.applyLang();
        if (window.License) License._renderBanner();
    },

    applyLang() {
        const lang = this.lang;
        document.querySelectorAll('[data-en][data-zh]').forEach(el => {
            const text = el.getAttribute(`data-${lang}`);
            if (text) el.innerHTML = text;
        });
        // Train drop-zone browse link is part of innerHTML — re-bind
        const browseLink = document.querySelector('#train-drop-zone .browse-link');
        if (browseLink && window.Trainer && Trainer._bindBrowseLink) {
            Trainer._bindBrowseLink();
        }
    },

    /** Toast notifier — used by trainer.js / prep.js / figures.js */
    toast(msg, kind) {
        try {
            const bar = document.createElement('div');
            bar.textContent = msg;
            bar.style.cssText = 'position:fixed;bottom:24px;right:24px;background:#1e293b;color:#fff;'
                + 'padding:0.75rem 1rem;border-radius:8px;font-size:0.9rem;max-width:380px;'
                + 'box-shadow:0 4px 12px rgba(0,0,0,0.2);z-index:9999;'
                + (kind === 'error' ? 'border-left:4px solid #dc2626;' : 'border-left:4px solid #2563eb;');
            document.body.appendChild(bar);
            setTimeout(() => bar.remove(), 5000);
        } catch (_) {
            alert(msg);
        }
    },
};

document.addEventListener('DOMContentLoaded', () => App.init());
window.App = App;
