document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const postText = document.getElementById('postText');
    const resultsContent = document.getElementById('resultsContent');
    const resultsWrapper = document.getElementById('resultsWrapper');
    const emptyState = document.getElementById('emptyState');
    const errorState = document.getElementById('errorState');
    const btnText = document.querySelector('.btn-text');
    const btnLoader = document.querySelector('.btn-loader');
    const charCount = document.getElementById('charCount');
    const explainSection = document.getElementById('explainSection');
    const explainLoading = document.getElementById('explainLoading');
    const explainResults = document.getElementById('explainResults');
    const expandBtn = document.getElementById('expandBtn');
    const resultsPanel = document.querySelector('.results-panel');
    const iconExpand = document.querySelector('.icon-expand');
    const iconCollapse = document.querySelector('.icon-collapse');

    // Expand/Collapse logic
    if (expandBtn) {
        expandBtn.addEventListener('click', () => {
            resultsPanel.classList.toggle('expanded');
            iconExpand.classList.toggle('hidden');
            iconCollapse.classList.toggle('hidden');
        });
    }

    // Character counter
    postText.addEventListener('input', () => {
        const len = postText.value.length;
        charCount.textContent = `${len} char${len !== 1 ? 's' : ''}`;
        
        // Reset states if user starts typing again after an error
        if (len === 0) {
            emptyState.classList.remove('hidden');
            errorState.classList.add('hidden');
            resultsContent.classList.add('hidden');
        }
    });

    analyzeBtn.addEventListener('click', async () => {
        const text = postText.value.trim();
        
        // Validation Heuristics
        const words = text.split(/\s+/).filter(w => w.length > 0);
        const meaningfulChars = text.replace(/[^a-zA-Z]/g, '');
        
        // Count typical code/log symbols and numbers
        const codeSymbolCount = (text.match(/[{}\[\]\\\/_=<>\*\&^%$#@0-9]/g) || []).length;
        const codeRatio = text.length > 0 ? codeSymbolCount / text.length : 0;
        
        // Gibberish Detection (Keyboard mashing)
        let hasGibberish = false;
        if (/[bcdfghjklmnpqrstvwxz]{6,}/i.test(text)) {
            hasGibberish = true;
        }
        for (let word of words) {
            if (word.length > 25 && !word.startsWith('http')) {
                hasGibberish = true;
                break;
            }
        }
        
        const isInvalid = !text || meaningfulChars.length < 5 || words.length < 3 || codeRatio > 0.15 || hasGibberish;
        
        if (isInvalid) {
            postText.style.animation = 'shake 0.4s ease';
            setTimeout(() => { postText.style.animation = ''; }, 500);
            
            emptyState.classList.add('hidden');
            resultsContent.classList.add('hidden');
            errorState.classList.remove('hidden');
            return;
        }

        const modelType = document.querySelector('input[name="modelType"]:checked').value;

        // Loading state
        analyzeBtn.disabled = true;
        btnText.style.display = 'none';
        btnLoader.style.display = 'flex';
        emptyState.classList.add('hidden');
        errorState.classList.add('hidden');
        resultsContent.classList.add('hidden');
        resultsWrapper.innerHTML = '';
        explainSection.classList.add('hidden');
        explainLoading.classList.remove('hidden');
        explainResults.classList.add('hidden');
        explainResults.innerHTML = '';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, model: modelType })
            });

            if (!response.ok) throw new Error('Server error');

            const data = await response.json();

            data.results.forEach((res, index) => {
                const confPercent = res.confidence !== null ? (res.confidence * 100).toFixed(1) : null;
                const dotClass = res.model_name.includes('Deep') ? 'dot-deep' : 'dot-traditional';
                const badgeClass = res.prediction.replace(/\s+/g, '-');

                let confidenceHTML = '';
                if (confPercent !== null) {
                    confidenceHTML = `
                        <div class="confidence-value-lg">
                            ${confPercent}%
                            <small>confidence</small>
                        </div>
                    `;
                }

                let meterHTML = '';
                if (confPercent !== null) {
                    meterHTML = `
                        <div class="meter-track">
                            <div class="meter-fill" id="meter-${index}" data-target="${confPercent}"></div>
                        </div>
                    `;
                }

                const card = document.createElement('div');
                card.className = 'result-card';
                card.style.animationDelay = `${index * 0.12}s`;
                card.innerHTML = `
                    <div class="result-card-header">
                        <span class="dot ${dotClass}"></span>
                        ${res.model_name}
                    </div>
                    <div class="result-main">
                        <div class="result-badge badge-${badgeClass}">
                            ${res.prediction}
                        </div>
                        ${confidenceHTML}
                    </div>
                    ${meterHTML}
                `;

                resultsWrapper.appendChild(card);

                // Animate meter
                if (confPercent !== null) {
                    setTimeout(() => {
                        const meter = document.getElementById(`meter-${index}`);
                        if (meter) meter.style.width = `${meter.dataset.target}%`;
                    }, 300 + index * 150);
                }
            });

            emptyState.classList.add('hidden');
            resultsContent.classList.remove('hidden');

            // ---- Fire async explainability request ----
            explainSection.classList.remove('hidden');
            explainLoading.classList.remove('hidden');
            explainResults.classList.add('hidden');

            fetchExplanations(text, modelType);

        } catch (error) {
            console.error('Error:', error);
            alert('Analysis failed. Make sure the server is running.');
        } finally {
            analyzeBtn.disabled = false;
            btnText.style.display = 'flex';
            btnLoader.style.display = 'none';
        }
    });

    // ================================================================
    // Explainability: fetch and render SHAP + LIME charts
    // ================================================================
    async function fetchExplanations(text, modelType) {
        try {
            const response = await fetch('/explain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, model: modelType })
            });

            if (!response.ok) throw new Error('Explain endpoint failed');

            const data = await response.json();

            explainLoading.classList.add('hidden');
            explainResults.classList.remove('hidden');
            explainResults.innerHTML = '';

            data.explanations.forEach((exp, idx) => {
                const modelBlock = document.createElement('div');
                modelBlock.className = 'explain-model-block';
                modelBlock.style.animationDelay = `${idx * 0.15}s`;

                const dotClass = exp.model_name.includes('Deep') ? 'dot-deep' : 'dot-traditional';

                modelBlock.innerHTML = `
                    <div class="explain-model-title">
                        <span class="dot ${dotClass}"></span>
                        ${exp.model_name} — <strong>${exp.predicted_class}</strong>
                    </div>
                    <div class="explain-tabs" data-block="${idx}">
                        <button class="explain-tab active" data-tab="lime" data-block="${idx}">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                                <path d="M12 20V10"/>
                                <path d="M18 20V4"/>
                                <path d="M6 20v-4"/>
                            </svg>
                            LIME
                        </button>
                        <button class="explain-tab" data-tab="shap" data-block="${idx}">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                                <path d="M3 3v18h18"/>
                                <path d="m19 9-5 5-4-4-3 3"/>
                            </svg>
                            SHAP
                        </button>
                    </div>
                    <div class="explain-chart-container" id="explain-chart-${idx}">
                        <div class="explain-chart-pane active" data-pane="lime" data-block="${idx}">
                            ${renderBarChart(exp.lime, 'lime')}
                        </div>
                        <div class="explain-chart-pane" data-pane="shap" data-block="${idx}">
                            ${renderBarChart(exp.shap, 'shap')}
                        </div>
                    </div>
                `;

                explainResults.appendChild(modelBlock);
            });

            // Attach tab click handlers using event delegation
            explainResults.addEventListener('click', (e) => {
                const tab = e.target.closest('.explain-tab');
                if (!tab) return;

                const blockIdx = tab.dataset.block;
                const tabName = tab.dataset.tab;

                // Update active tab
                document.querySelectorAll(`.explain-tab[data-block="${blockIdx}"]`).forEach(t => t.classList.remove('active'));
                tab.classList.add('active');

                // Show matching pane
                document.querySelectorAll(`.explain-chart-pane[data-block="${blockIdx}"]`).forEach(p => p.classList.remove('active'));
                const targetPane = document.querySelector(`.explain-chart-pane[data-pane="${tabName}"][data-block="${blockIdx}"]`);
                if (targetPane) {
                    targetPane.classList.add('active');

                    // Re-trigger bar animations
                    setTimeout(() => {
                        targetPane.querySelectorAll('.explain-bar-fill').forEach(bar => {
                            const target = bar.dataset.width;
                            bar.style.width = '0%';
                            requestAnimationFrame(() => {
                                requestAnimationFrame(() => {
                                    bar.style.width = target + '%';
                                });
                            });
                        });
                    }, 50);
                }
            });

            // Animate bars on first render
            setTimeout(() => {
                document.querySelectorAll('.explain-chart-pane.active .explain-bar-fill').forEach(bar => {
                    bar.style.width = bar.dataset.width + '%';
                });
            }, 200);

        } catch (error) {
            console.error('Explainability error:', error);
            explainLoading.classList.add('hidden');
            explainResults.classList.remove('hidden');
            explainResults.innerHTML = `
                <div class="explain-error">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="12" y1="8" x2="12" y2="12"/>
                        <line x1="12" y1="16" x2="12.01" y2="16"/>
                    </svg>
                    <span>Could not generate explanations. Try again or check server logs.</span>
                </div>
            `;
        }
    }

    // ================================================================
    // Render a horizontal bar chart from word-importance pairs
    // ================================================================
    function renderBarChart(pairs, type) {
        if (!pairs || pairs.length === 0) {
            return `<div class="explain-empty">No feature importance data available.</div>`;
        }

        const maxAbsVal = Math.max(...pairs.map(p => Math.abs(p[1])), 0.001);

        let html = '<div class="explain-bar-chart">';

        pairs.forEach(([word, value], i) => {
            const isPositive = value >= 0;
            const widthPercent = Math.min((Math.abs(value) / maxAbsVal) * 100, 100);
            const barClass = isPositive ? 'bar-positive' : 'bar-negative';
            const displayValue = value >= 0 ? `+${value.toFixed(4)}` : value.toFixed(4);

            html += `
                <div class="explain-bar-row" style="animation-delay: ${i * 0.04}s">
                    <span class="explain-bar-word" title="${word}">${escapeHtml(word)}</span>
                    <div class="explain-bar-track">
                        <div class="explain-bar-fill ${barClass}" data-width="${widthPercent.toFixed(1)}" style="width: 0%"></div>
                    </div>
                    <span class="explain-bar-value ${barClass}">${displayValue}</span>
                </div>
            `;
        });

        html += '</div>';
        return html;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});
