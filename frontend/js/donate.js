/**
 * BioSpark-Light — donate / support modal.
 *
 * Behavior:
 *   - Opens automatically on first launch (or any launch where the
 *     user hasn't ticked "don't show on next launch")
 *   - Always reachable from the footer "☕ Support the author" button
 *   - "Don't show" preference is per-version, so a new release re-prompts
 *     once (gives us a chance to refresh the message / QR / perks)
 *   - Closes on Esc or clicking the dim backdrop
 *
 * State key: localStorage["biospark-donate-dismissed-v0.2"]
 *   value "1" = user opted out for this version
 */

const DONATE_VERSION = '0.2';
const DONATE_KEY = `biospark-donate-dismissed-v${DONATE_VERSION}`;

const Donate = {
    _bound: false,

    /** Called from App.init() — decides whether to auto-pop on launch. */
    init() {
        this._bindKeys();
        // Slight delay so the page renders + i18n applies first; feels less jarring
        const dismissed = localStorage.getItem(DONATE_KEY) === '1';
        if (!dismissed) {
            setTimeout(() => this.open(), 600);
        }
    },

    open() {
        const modal = document.getElementById('donate-modal');
        if (!modal) return;
        // Re-apply current language to the modal's data-en/data-zh nodes
        if (window.App && App.applyLang) App.applyLang();
        // Sync the "don't show" checkbox to current preference
        const cb = document.getElementById('donate-dont-show');
        if (cb) cb.checked = localStorage.getItem(DONATE_KEY) === '1';
        modal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    },

    close() {
        const modal = document.getElementById('donate-modal');
        if (!modal) return;
        // Persist the "don't show" preference
        const cb = document.getElementById('donate-dont-show');
        if (cb) {
            if (cb.checked) localStorage.setItem(DONATE_KEY, '1');
            else            localStorage.removeItem(DONATE_KEY);
        }
        modal.classList.add('hidden');
        document.body.style.overflow = '';
    },

    /** Dismiss only when the dim backdrop (not the card) is clicked. */
    handleBackdropClick(evt) {
        if (evt && evt.target && evt.target.id === 'donate-modal') {
            this.close();
        }
    },

    _bindKeys() {
        if (this._bound) return;
        this._bound = true;
        document.addEventListener('keydown', (evt) => {
            if (evt.key === 'Escape') {
                const modal = document.getElementById('donate-modal');
                if (modal && !modal.classList.contains('hidden')) {
                    this.close();
                }
            }
        });
    },
};

window.Donate = Donate;
