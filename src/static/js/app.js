/**
 * Smartphone Addiction Analytical Platform
 * Interactive Client Application (app.js)
 *
 * Handles:
 * - 12 behavioral input controls + decision threshold slider with live readouts (PAR-1)
 * - 35ms debounced live REST inference against /api/predict
 * - Live latency telemetry display
 * - 4 granular ratio diagnostic cards with strict target indicators
 * - Dynamic clinical recommendations list with correct spelling ('ADDICTION DETECTED' / 'HEALTHY')
 * - Inline 24h physiological limit warning alert (AC-1.7)
 * - Accessible 4-tab keyboard navigation (diagnostic, cohorts, whatif, batch)
 * - Calibrated fallback heuristic matching ADR-0007
 */

(function () {
    'use strict';

    // -------------------------------------------------------------------------
    // Application State
    // -------------------------------------------------------------------------
    const state = {
        profile: {
            age: 25,
            gender: 'Male',
            stress_level: 'Medium',
            academic_work_impact: 'Yes',
            daily_screen_time_hours: 7.5,
            social_media_hours: 3.5,
            gaming_hours: 1.2,
            work_study_hours: 2.5,
            weekend_screen_time: 9.5,
            sleep_hours: 6.8,
            notifications_per_day: 140,
            app_opens_per_day: 100,
            threshold: 0.50
        },
        cohort: {
            dimension: 'age_bracket',
            filter_stress: null
        },
        whatIf: {
            delta_recreation: -1.5,
            delta_sleep: 1.0,
            delta_opens: -30
        },
        batch: {
            downloadToken: null
        },
        lastProbability: 0.505,
        debounceTimer: null,
        cohortCache: new Map(),
        screenSleepMatrixCache: null,
        benchmarkOverlayCache: new Map()
    };

    // -------------------------------------------------------------------------
    // DOM Element References
    // -------------------------------------------------------------------------
    const dom = {
        // Telemetry
        telemetryModel: document.getElementById('telemetry-model'),
        telemetryLatency: document.getElementById('telemetry-latency'),

        // Warning
        boundaryWarning: document.getElementById('boundary-warning'),
        excessHoursMsg: document.getElementById('excess-hours-msg'),

        // Tab Navigation
        tabNav: document.querySelector('.nav-tab-container'),
        tabButtons: document.querySelectorAll('.tab-button'),
        tabPanels: document.querySelectorAll('.tab-content-panel'),

        // Inputs & Readouts
        inputAge: document.getElementById('input-age'),
        valAge: document.getElementById('val-age'),
        inputDailyScreen: document.getElementById('input-daily-screen'),
        valDailyScreen: document.getElementById('val-daily-screen'),
        inputSocialMedia: document.getElementById('input-social-media'),
        valSocialMedia: document.getElementById('val-social-media'),
        inputGaming: document.getElementById('input-gaming'),
        valGaming: document.getElementById('val-gaming'),
        inputWorkStudy: document.getElementById('input-work-study'),
        valWorkStudy: document.getElementById('val-work-study'),
        inputWeekendScreen: document.getElementById('input-weekend-screen'),
        valWeekendScreen: document.getElementById('val-weekend-screen'),
        inputSleep: document.getElementById('input-sleep'),
        valSleep: document.getElementById('val-sleep'),
        inputNotifications: document.getElementById('input-notifications'),
        valNotifications: document.getElementById('val-notifications'),
        inputAppOpens: document.getElementById('input-app-opens'),
        valAppOpens: document.getElementById('val-app-opens'),
        inputThreshold: document.getElementById('input-threshold'),
        valThreshold: document.getElementById('val-threshold'),

        // Assessment Output
        gaugeArc: document.getElementById('gauge-arc'),
        gaugeProbNum: document.getElementById('gauge-probability-num'),
        statusPill: document.getElementById('status-pill'),
        badgeStatus: document.getElementById('badge-status'),
        classificationSummary: document.getElementById('classification-summary'),
        metricScreenSleep: document.getElementById('metric-screen-sleep'),
        targetScreenSleep: document.getElementById('target-screen-sleep'),
        metricRecShare: document.getElementById('metric-rec-share'),
        targetRecShare: document.getElementById('target-rec-share'),
        metricUnlockMins: document.getElementById('metric-unlock-mins'),
        targetUnlockMins: document.getElementById('target-unlock-mins'),
        metricWeekendSurge: document.getElementById('metric-weekend-surge'),
        targetWeekendSurge: document.getElementById('target-weekend-surge'),
        interventionList: document.getElementById('intervention-list'),

        // Cohort Analytics
        cohortTotalRecords: document.getElementById('cohort-total-records'),
        cohortOverallPrevalence: document.getElementById('cohort-overall-prevalence'),
        cohortMeanScreen: document.getElementById('cohort-mean-screen'),
        cohortMeanSleep: document.getElementById('cohort-mean-sleep'),
        cohortCardsContainer: document.getElementById('cohort-cards-container'),
        heatmapGrid: document.getElementById('heatmap-grid'),
        valQuantileScreen: document.getElementById('val-quantile-screen'),
        barQuantileScreen: document.getElementById('bar-quantile-screen'),
        valUserScreenEcho: document.getElementById('val-user-screen-echo'),
        valQuantileSleep: document.getElementById('val-quantile-sleep'),
        barQuantileSleep: document.getElementById('bar-quantile-sleep'),
        valUserSleepEcho: document.getElementById('val-user-sleep-echo'),
        valQuantileOpens: document.getElementById('val-quantile-opens'),
        barQuantileOpens: document.getElementById('bar-quantile-opens'),
        valUserOpensEcho: document.getElementById('val-user-opens-echo'),
        valQuantileNotifs: document.getElementById('val-quantile-notifs'),
        barQuantileNotifs: document.getElementById('bar-quantile-notifs'),
        valUserNotifsEcho: document.getElementById('val-user-notifs-echo'),

        // What-If
        leverRecreation: document.getElementById('lever-recreation-reduce'),
        valLeverRecreation: document.getElementById('val-lever-recreation'),
        leverSleep: document.getElementById('lever-sleep-extend'),
        valLeverSleep: document.getElementById('val-lever-sleep'),
        leverOpens: document.getElementById('lever-opens-batch'),
        valLeverOpens: document.getElementById('val-lever-opens'),
        simBaselineProb: document.getElementById('sim-baseline-prob'),
        simBaselineBadge: document.getElementById('sim-baseline-badge'),
        simCounterfactualProb: document.getElementById('sim-counterfactual-prob'),
        simCounterfactualBadge: document.getElementById('sim-counterfactual-badge'),
        simDeltaBadge: document.getElementById('sim-delta-badge'),
        simBaseSS: document.getElementById('sim-base-ss'),
        simNewSS: document.getElementById('sim-new-ss'),
        simBaseRec: document.getElementById('sim-base-rec'),
        simNewRec: document.getElementById('sim-new-rec'),
        simBaseScreen: document.getElementById('sim-base-screen'),
        simNewScreen: document.getElementById('sim-new-screen'),
        simBaseSleep: document.getElementById('sim-base-sleep'),
        simNewSleep: document.getElementById('sim-new-sleep'),
        simInterventionsList: document.getElementById('sim-interventions-list'),
        btnGenerateOptimal: document.getElementById('btn-generate-optimal'),
        optimalTargetCard: document.getElementById('optimal-target-card'),
        optimalTargetText: document.getElementById('optimal-target-text'),
        btnApplyOptimal: document.getElementById('btn-apply-optimal'),

        // Batch Diagnostics
        batchDropzone: document.getElementById('batch-dropzone'),
        batchFileInput: document.getElementById('batch-file-input'),
        btnLoadDemoBatch: document.getElementById('btn-load-demo-batch'),
        batchLoading: document.getElementById('batch-loading'),
        batchError: document.getElementById('batch-error'),
        batchErrorMsg: document.getElementById('batch-error-msg'),
        batchResultsView: document.getElementById('batch-results-view'),
        batchTotalRecords: document.getElementById('batch-total-records'),
        batchAddictionRate: document.getElementById('batch-addiction-rate'),
        batchHighRiskRate: document.getElementById('batch-high-risk-rate'),
        batchMeanSS: document.getElementById('batch-mean-ss'),
        batchTableBody: document.getElementById('batch-table-body'),
        btnExportCsv: document.getElementById('btn-export-csv'),

        // Toast Container
        toastContainer: document.getElementById('toast-container')
    };

    // -------------------------------------------------------------------------
    // Toast Notification Manager (Accessible Live Alerts & Offline Feedback)
    // -------------------------------------------------------------------------
    const toastManager = (function () {
        let lastToastTime = 0;
        let lastToastMessage = '';
        const THROTTLE_MS = 3500;

        function showToast(title, message, type = 'warning', duration = 4000) {
            const now = Date.now();
            if (message === lastToastMessage && now - lastToastTime < THROTTLE_MS && type !== 'danger') {
                return; // Throttles repeat toasts from rapid slider inputs
            }
            lastToastTime = now;
            lastToastMessage = message;

            const container = dom.toastContainer || document.getElementById('toast-container');
            if (!container) return;

            const toast = document.createElement('div');
            toast.className = 'toast toast-' + type;
            toast.setAttribute('role', type === 'danger' ? 'alert' : 'status');

            const iconMap = {
                warning: '!',
                danger: '✕',
                info: 'ℹ',
                success: '✓'
            };
            const icon = iconMap[type] || 'ℹ';

            toast.innerHTML =
                '<span class="toast-icon" aria-hidden="true">' + icon + '</span>' +
                '<div class="toast-content">' +
                    '<div class="toast-title">' + escapeHtml(title) + '</div>' +
                    '<div class="toast-message">' + escapeHtml(message) + '</div>' +
                '</div>' +
                '<button type="button" class="toast-close" aria-label="Dismiss notification">&times;</button>';

            const closeBtn = toast.querySelector('.toast-close');
            let timer = null;

            function dismiss() {
                if (timer) clearTimeout(timer);
                toast.classList.add('toast-dismissing');
                setTimeout(function () {
                    if (toast.parentNode) {
                        toast.parentNode.removeChild(toast);
                    }
                }, 200);
            }

            if (closeBtn) {
                closeBtn.addEventListener('click', dismiss);
            }

            if (duration > 0) {
                timer = setTimeout(dismiss, duration);
            }

            container.appendChild(toast);
        }

        function escapeHtml(str) {
            return String(str)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;');
        }

        return { showToast };
    })();

    // -------------------------------------------------------------------------
    // Initialization
    // -------------------------------------------------------------------------
    function init() {
        bindNavigation();
        bindInputControls();
        bindCohortControls();
        bindWhatIfControls();
        bindBatchControls();
        bindA11yKeyboard();
        bindNetworkMonitoring();
        checkSystemHealth();

        // Initial evaluation
        checkPhysiologicalBoundary();
        triggerDiagnosticPrediction();
        loadCohortAnalytics();
        loadHeatmapMatrix();
        updateBenchmarkOverlay();
        runWhatIfSimulation();
    }

    // -------------------------------------------------------------------------
    // Accessible Radiogroup Keyboard & Network Monitor Listeners
    // -------------------------------------------------------------------------
    function bindA11yKeyboard() {
        document.querySelectorAll('.segmented-control[role="radiogroup"]').forEach(function (group) {
            const items = Array.from(group.querySelectorAll('.segment-btn[role="radio"]'));
            items.forEach(function (item, idx) {
                item.addEventListener('keydown', function (e) {
                    let targetIdx = null;
                    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                        targetIdx = (idx + 1) % items.length;
                    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                        targetIdx = (idx - 1 + items.length) % items.length;
                    }
                    if (targetIdx !== null) {
                        e.preventDefault();
                        items[targetIdx].click();
                        items[targetIdx].focus();
                    }
                });
            });
        });
    }

    function bindNetworkMonitoring() {
        window.addEventListener('offline', function () {
            toastManager.showToast('Network Disconnected', 'Working offline. Local fallback heuristic will evaluate inputs.', 'warning');
        });
        window.addEventListener('online', function () {
            toastManager.showToast('Connection Restored', 'Reconnected to analytical backend server.', 'info');
            triggerDiagnosticPrediction();
        });
    }

    // -------------------------------------------------------------------------
    // Tab Navigation & Keyboard Handling
    // -------------------------------------------------------------------------
    function bindNavigation() {
        const buttons = Array.from(dom.tabButtons);

        function activateTab(index) {
            if (index < 0 || index >= buttons.length) return;
            const btn = buttons[index];
            const targetTabId = btn.getAttribute('data-tab');

            buttons.forEach((b, i) => {
                const isActive = (i === index);
                b.classList.toggle('active', isActive);
                b.setAttribute('aria-selected', isActive ? 'true' : 'false');
                b.setAttribute('tabindex', isActive ? '0' : '-1');
            });

            dom.tabPanels.forEach(panel => {
                panel.classList.remove('active');
            });

            const targetPanel = document.getElementById(targetTabId);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }

            btn.focus();

            // Refresh tab-specific dynamic content
            if (targetTabId === 'tab-cohorts') {
                loadCohortAnalytics();
                loadHeatmapMatrix();
                updateBenchmarkOverlay();
            } else if (targetTabId === 'tab-whatif') {
                runWhatIfSimulation();
            }
        }

        buttons.forEach((btn, idx) => {
            btn.addEventListener('click', () => activateTab(idx));

            btn.addEventListener('keydown', (e) => {
                let targetIdx = null;
                if (e.key === 'ArrowRight') {
                    targetIdx = (idx + 1) % buttons.length;
                } else if (e.key === 'ArrowLeft') {
                    targetIdx = (idx - 1 + buttons.length) % buttons.length;
                } else if (e.key === 'Home') {
                    targetIdx = 0;
                } else if (e.key === 'End') {
                    targetIdx = buttons.length - 1;
                }

                if (targetIdx !== null) {
                    e.preventDefault();
                    activateTab(targetIdx);
                }
            });
        });
    }

    // -------------------------------------------------------------------------
    // Input Controls & Live Readout Synchronizers (PAR-1)
    // -------------------------------------------------------------------------
    function bindInputControls() {
        const sliderConfigs = [
            {
                input: dom.inputAge,
                readout: dom.valAge,
                key: 'age',
                format: function (v) { return String(Math.round(v)); }
            },
            {
                input: dom.inputDailyScreen,
                readout: dom.valDailyScreen,
                key: 'daily_screen_time_hours',
                format: function (v) { return parseFloat(v).toFixed(1) + 'h'; }
            },
            {
                input: dom.inputSocialMedia,
                readout: dom.valSocialMedia,
                key: 'social_media_hours',
                format: function (v) { return parseFloat(v).toFixed(1) + 'h'; }
            },
            {
                input: dom.inputGaming,
                readout: dom.valGaming,
                key: 'gaming_hours',
                format: function (v) { return parseFloat(v).toFixed(1) + 'h'; }
            },
            {
                input: dom.inputWorkStudy,
                readout: dom.valWorkStudy,
                key: 'work_study_hours',
                format: function (v) { return parseFloat(v).toFixed(1) + 'h'; }
            },
            {
                input: dom.inputWeekendScreen,
                readout: dom.valWeekendScreen,
                key: 'weekend_screen_time',
                format: function (v) { return parseFloat(v).toFixed(1) + 'h'; }
            },
            {
                input: dom.inputSleep,
                readout: dom.valSleep,
                key: 'sleep_hours',
                format: function (v) { return parseFloat(v).toFixed(1) + 'h'; }
            },
            {
                input: dom.inputNotifications,
                readout: dom.valNotifications,
                key: 'notifications_per_day',
                format: function (v) { return String(Math.round(v)); }
            },
            {
                input: dom.inputAppOpens,
                readout: dom.valAppOpens,
                key: 'app_opens_per_day',
                format: function (v) { return String(Math.round(v)); }
            },
            {
                input: dom.inputThreshold,
                readout: dom.valThreshold,
                key: 'threshold',
                format: function (v) { return parseFloat(v).toFixed(2); }
            }
        ];

        sliderConfigs.forEach(function (cfg) {
            if (!cfg.input) return;

            cfg.input.addEventListener('input', function (e) {
                const rawVal = parseFloat(e.target.value);
                const isInt = (cfg.key === 'age' || cfg.key === 'notifications_per_day' || cfg.key === 'app_opens_per_day');
                state.profile[cfg.key] = isInt ? Math.round(rawVal) : rawVal;

                cfg.input.setAttribute('aria-valuenow', rawVal);

                if (cfg.readout) {
                    cfg.readout.textContent = cfg.format(rawVal);
                }

                checkPhysiologicalBoundary();

                // If threshold slider changed, instantly re-evaluate risk gauge readout and status pill (PAR-1, PAR-3)
                if (cfg.key === 'threshold') {
                    const currentProbability = state.lastProbability;
                    const newThreshold = state.profile.threshold;
                    const isAddicted = currentProbability >= newThreshold;
                    const tier = currentProbability >= 0.70 ? 'HIGH' : (currentProbability >= 0.40 ? 'MODERATE' : 'LOW');
                    const computedStatusLabel = isAddicted
                        ? 'ADDICTION DETECTED • ' + tier + ' RISK'
                        : 'HEALTHY PATTERN • ' + tier + ' RISK';
                    const classification = isAddicted ? 'ADDICTION DETECTED' : 'HEALTHY';

                    // Immediately update #badge-status text
                    if (dom.badgeStatus) {
                        dom.badgeStatus.textContent = computedStatusLabel;
                    }

                    // Immediately flip #status-pill styling between high/mod/low without waiting for API response
                    if (dom.statusPill) {
                        let pillClass = 'status-pill-low';
                        if (isAddicted) {
                            pillClass = 'status-pill-high';
                        } else if (tier === 'MODERATE') {
                            pillClass = 'status-pill-mod';
                        } else {
                            pillClass = 'status-pill-low';
                        }
                        dom.statusPill.className = 'status-pill ' + pillClass;
                    }

                    if (typeof window.renderRiskGauge === 'function') {
                        window.renderRiskGauge(currentProbability, newThreshold, computedStatusLabel, classification);
                    }
                }

                schedulePrediction();
            });
        });

        // Segmented controls (gender, stress, impact)
        document.querySelectorAll('.segment-btn[data-group]').forEach(function (btn) {
            btn.addEventListener('click', function () {
                const group = btn.getAttribute('data-group');
                const val = btn.getAttribute('data-value');

                document.querySelectorAll('.segment-btn[data-group="' + group + '"]').forEach(function (b) {
                    b.classList.remove('active');
                    b.setAttribute('aria-checked', 'false');
                });

                btn.classList.add('active');
                btn.setAttribute('aria-checked', 'true');

                if (group === 'gender') state.profile.gender = val;
                if (group === 'stress') state.profile.stress_level = val;
                if (group === 'impact') state.profile.academic_work_impact = val;

                schedulePrediction();
            });
        });
    }

    // -------------------------------------------------------------------------
    // Physiological Boundary Check (AC-1.7: 24h limit warning banner)
    // -------------------------------------------------------------------------
    function checkPhysiologicalBoundary() {
        if (!dom.boundaryWarning) return;

        const screen = state.profile.daily_screen_time_hours;
        const work = state.profile.work_study_hours;
        const sleep = state.profile.sleep_hours;
        const total = screen + work + sleep;

        if (total > 24.0 || (screen + sleep > 24.0)) {
            dom.boundaryWarning.classList.add('active');
            if (dom.excessHoursMsg) {
                dom.excessHoursMsg.textContent =
                    'Sum of daily screen (' + screen.toFixed(1) + 'h), work/study (' +
                    work.toFixed(1) + 'h), and sleep (' + sleep.toFixed(1) +
                    'h) equals ' + total.toFixed(1) +
                    'h, exceeding the 24.0h/day physiological limit. Please verify daily digital budget.';
            }
        } else {
            dom.boundaryWarning.classList.remove('active');
        }
    }

    // -------------------------------------------------------------------------
    // Debounced Live Inference (35ms debounce calling POST /api/predict)
    // -------------------------------------------------------------------------
    function schedulePrediction() {
        if (state.debounceTimer) {
            clearTimeout(state.debounceTimer);
        }
        state.debounceTimer = setTimeout(function () {
            triggerDiagnosticPrediction();
        }, 35);
    }

    async function triggerDiagnosticPrediction() {
        const t0 = performance.now();
        try {
            const resp = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(state.profile)
            });

            if (!resp.ok) {
                throw new Error('HTTP ' + resp.status);
            }

            const data = await resp.json();
            const measuredLatency = Math.max(0.1, performance.now() - t0);
            const displayLatency = data.latency_ms !== undefined ? data.latency_ms : measuredLatency;

            if (dom.telemetryLatency) {
                dom.telemetryLatency.textContent = displayLatency.toFixed(1) + ' ms';
            }

            state.lastProbability = data.probability;
            renderDiagnosticOutput(data);
        } catch (err) {
            // Graceful client fallback matching ADR-0007 centered logit
            toastManager.showToast('Offline Mode Active', 'Prediction API unreachable. Calibrated local baseline model active.', 'warning');
            const fallbackData = computeClientFallback(state.profile);
            state.lastProbability = fallbackData.probability;
            renderDiagnosticOutput(fallbackData);
        }
    }

    // -------------------------------------------------------------------------
    // Render Assessment Output (Arc Gauge, 4 Ratio Tiles, Interventions)
    // -------------------------------------------------------------------------
    function renderDiagnosticOutput(data) {
        const prob = data.probability;
        const thresh = data.decision_threshold !== undefined ? data.decision_threshold : state.profile.threshold;
        const statusLabel = data.status_label;
        const classification = data.classification;

        // 1. Render SVG Risk Gauge via charts.js
        if (typeof window.renderRiskGauge === 'function') {
            window.renderRiskGauge(prob, thresh, statusLabel, classification);
        }

        // Update #badge-status and #status-pill using data.status_label
        if (statusLabel) {
            if (dom.badgeStatus) {
                dom.badgeStatus.textContent = statusLabel;
            }
            if (dom.statusPill) {
                const isAddicted = prob >= thresh || statusLabel.indexOf('ADDICTION DETECTED') !== -1;
                const tier = prob >= 0.70 ? 'HIGH' : (prob >= 0.40 ? 'MODERATE' : 'LOW');
                let pillClass = 'status-pill-low';
                if (isAddicted || statusLabel.indexOf('ADDICTION DETECTED') !== -1) {
                    pillClass = 'status-pill-high';
                } else if (tier === 'MODERATE' || statusLabel.indexOf('MODERATE RISK') !== -1) {
                    pillClass = 'status-pill-mod';
                } else {
                    pillClass = 'status-pill-low';
                }
                dom.statusPill.className = 'status-pill ' + pillClass;
            }
        }

        // 2. Render 4 Granular Ratio Diagnostic Cards with Target Indicators
        const ratios = data.ratios || {};
        const screenToSleep = ratios.screen_to_sleep_ratio !== undefined ? ratios.screen_to_sleep_ratio : (ratios.screen_to_sleep || 0);
        const recShare = ratios.recreational_share !== undefined ? ratios.recreational_share : (ratios.recreational_to_screen || 0);
        const avgUnlock = ratios.avg_unlock_minutes !== undefined ? ratios.avg_unlock_minutes : 0;
        const weekendSurge = ratios.weekend_surge_hours !== undefined ? ratios.weekend_surge_hours : (ratios.weekend_diff || 0);

        // Ratio 1: Screen / Sleep Ratio (Target < 1.0x)
        if (dom.metricScreenSleep && dom.targetScreenSleep) {
            dom.metricScreenSleep.textContent = screenToSleep.toFixed(2) + 'x';
            if (screenToSleep <= 1.0) {
                dom.targetScreenSleep.className = 'metric-card-target target-met';
                dom.targetScreenSleep.innerHTML = '<span>Target Met (&lt; 1.0x)</span>';
            } else {
                dom.targetScreenSleep.className = 'metric-card-target target-breached';
                dom.targetScreenSleep.innerHTML = '<span>Target Exceeded (&gt; 1.0x)</span>';
            }
        }

        // Ratio 2: Recreational Share (Target < 50%)
        if (dom.metricRecShare && dom.targetRecShare) {
            dom.metricRecShare.textContent = (recShare * 100).toFixed(1) + '%';
            if (recShare <= 0.50) {
                dom.targetRecShare.className = 'metric-card-target target-met';
                dom.targetRecShare.innerHTML = '<span>Target Met (&lt; 50%)</span>';
            } else {
                dom.targetRecShare.className = 'metric-card-target target-breached';
                dom.targetRecShare.innerHTML = '<span>Target Exceeded (&gt; 50%)</span>';
            }
        }

        // Ratio 3: Avg Session Length (Target > 5.0 min per SPEC Journey 1)
        if (dom.metricUnlockMins && dom.targetUnlockMins) {
            dom.metricUnlockMins.textContent = avgUnlock.toFixed(1) + ' min';
            if (avgUnlock >= 5.0) {
                dom.targetUnlockMins.className = 'metric-card-target target-met';
                dom.targetUnlockMins.innerHTML = '<span>Target Met (&gt; 5.0 min)</span>';
            } else {
                dom.targetUnlockMins.className = 'metric-card-target target-breached';
                dom.targetUnlockMins.innerHTML = '<span>Target Fragmented (&lt; 5.0 min)</span>';
            }
        }

        // Ratio 4: Weekend Surge (Target < 2.0h per SPEC Journey 1)
        if (dom.metricWeekendSurge && dom.targetWeekendSurge) {
            const surgeSign = weekendSurge >= 0 ? '+' : '';
            dom.metricWeekendSurge.textContent = surgeSign + weekendSurge.toFixed(1) + 'h';
            if (weekendSurge <= 2.0) {
                dom.targetWeekendSurge.className = 'metric-card-target target-met';
                dom.targetWeekendSurge.innerHTML = '<span>Target Met (&lt; 2.0h)</span>';
            } else {
                dom.targetWeekendSurge.className = 'metric-card-target target-breached';
                dom.targetWeekendSurge.innerHTML = '<span>Surge Detected (&gt; 2.0h)</span>';
            }
        }

        // 3. Dynamic Clinical Interventions (AC-1.5, PAR-5, PAR-6)
        if (dom.interventionList) {
            dom.interventionList.innerHTML = '';
            const recs = data.interventions || [];

            if (recs.length > 0) {
                recs.forEach(function (rec) {
                    const item = document.createElement('div');
                    item.className = 'intervention-item';
                    item.textContent = rec;
                    dom.interventionList.appendChild(item);
                });
            } else {
                const item = document.createElement('div');
                item.className = 'intervention-item optimal';
                item.textContent = 'Balanced behavioral hygiene detected. Maintain current digital boundaries.';
                dom.interventionList.appendChild(item);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Client-Side Fallback Calculator (ADR-0007 centered logit calibration)
    // -------------------------------------------------------------------------
    function computeClientFallback(p) {
        const screen = p.daily_screen_time_hours;
        const sleep = p.sleep_hours;
        const soc = p.social_media_hours;
        const game = p.gaming_hours;
        const opens = p.app_opens_per_day;
        const weekend = p.weekend_screen_time;

        const eps = 1e-5;
        const s_to_sl = screen / (sleep + eps);
        const rec_to_s = (soc + game) / (screen + eps);
        const avg_unlock = (screen * 60.0) / (opens + eps);
        const weekend_diff = weekend - screen;

        const stressMap = { Low: 0, Medium: 1, High: 2 };
        const impactMap = { No: 0, Yes: 1 };

        // Centered logit model: yields ~50.5% on standard population median baseline
        const logit = 0.02 +
            0.35 * (screen - 7.5) +
            0.40 * (soc - 3.5) +
            0.25 * (game - 1.2) +
            0.18 * (weekend - 9.5) -
            0.35 * (sleep - 6.8) +
            0.008 * (p.notifications_per_day - 140) +
            0.010 * (opens - 100) +
            0.22 * (stressMap[p.stress_level] !== undefined ? stressMap[p.stress_level] - 1 : 0) +
            0.30 * (impactMap[p.academic_work_impact] !== undefined ? impactMap[p.academic_work_impact] - 1 : 0);

        const prob = Math.min(0.98, Math.max(0.02, 1.0 / (1.0 + Math.exp(-logit))));
        const isAddicted = prob >= p.threshold;

        // PAR-6: Correct spelling 'ADDICTION DETECTED' / 'HEALTHY'
        const classification = isAddicted ? 'ADDICTION DETECTED' : 'HEALTHY';
        const tier = prob >= 0.70 ? 'High' : (prob >= 0.40 ? 'Moderate' : 'Healthy');
        const tierUpper = prob >= 0.70 ? 'HIGH' : (prob >= 0.40 ? 'MODERATE' : 'LOW');
        const computedStatusLabel = isAddicted
            ? `ADDICTION DETECTED • ${tierUpper} RISK`
            : `HEALTHY PATTERN • ${tierUpper} RISK`;

        const interventions = [];
        if (s_to_sl > 1.2) {
            interventions.push('Sleep Protection: Screen time exceeds sleep duration. Enforce a 60-minute pre-bed digital curfew.');
        }
        if (rec_to_s > 0.60) {
            interventions.push('Recreation Audit: Over 60% of device time is social media/gaming. Establish focused non-digital hobbies.');
        }
        if (opens > 120 || p.notifications_per_day > 180) {
            interventions.push('Notification Hygiene: High daily unlock frequency (>120 opens/day) or alert volume (>180/day). Batch non-essential notifications into designated hourly slots.');
        }
        if (weekend_diff > 2.5) {
            interventions.push('Weekend Disconnect: Surge of >2.5 hours on weekends suggests compensatory bingeing. Schedule structured offline weekend activities.');
        }

        return {
            probability: prob,
            prediction: isAddicted ? 1 : 0,
            classification: classification,
            severity: tier,
            status_label: computedStatusLabel,
            ratios: {
                screen_to_sleep_ratio: s_to_sl,
                recreational_share: rec_to_s,
                avg_unlock_minutes: avg_unlock,
                weekend_surge_hours: weekend_diff,
                total_accounted_hours: screen + p.work_study_hours + sleep,
                exceeds_daily_budget: (screen + p.work_study_hours + sleep) > 24.0
            },
            interventions: interventions,
            latency_ms: 0.50,
            decision_threshold: p.threshold
        };
    }

    // -------------------------------------------------------------------------
    // TAB 2: Population Cohort Analytics (M2 Support)
    // -------------------------------------------------------------------------
    function bindCohortControls() {
        document.querySelectorAll('.segment-btn[data-cohort-dim]').forEach(function (btn) {
            btn.addEventListener('click', function () {
                document.querySelectorAll('.segment-btn[data-cohort-dim]').forEach(function (b) {
                    b.classList.remove('active');
                    b.setAttribute('aria-checked', 'false');
                });
                btn.classList.add('active');
                btn.setAttribute('aria-checked', 'true');
                state.cohort.dimension = btn.getAttribute('data-cohort-dim');
                loadCohortAnalytics();
            });
        });

        document.querySelectorAll('.segment-btn[data-stress-filter]').forEach(function (btn) {
            btn.addEventListener('click', function () {
                document.querySelectorAll('.segment-btn[data-stress-filter]').forEach(function (b) {
                    b.classList.remove('active');
                    b.setAttribute('aria-checked', 'false');
                });
                btn.classList.add('active');
                btn.setAttribute('aria-checked', 'true');
                const val = btn.getAttribute('data-stress-filter');
                state.cohort.filter_stress = val === 'All' ? null : val;
                loadCohortAnalytics();
            });
        });
    }

    async function loadCohortAnalytics() {
        if (!dom.cohortCardsContainer) return;
        const dim = state.cohort.dimension || 'age_bracket';
        const stress = state.cohort.filter_stress;
        const cacheKey = dim + '__' + (stress || 'all');

        // Instant sub-2ms render from memory cache
        if (state.cohortCache.has(cacheKey)) {
            const cached = state.cohortCache.get(cacheKey);
            applyCohortAnalytics(cached);
            return;
        }

        try {
            let url = '/api/analytics/cohorts?dimension=' + encodeURIComponent(dim);
            if (stress) {
                url += '&filter_stress=' + encodeURIComponent(stress);
            }

            const resp = await fetch(url);
            if (!resp.ok) return;
            const data = await resp.json();
            state.cohortCache.set(cacheKey, data);
            applyCohortAnalytics(data);
        } catch (err) {
            // Non-blocking fallback
        }
    }

    function applyCohortAnalytics(data) {
        if (data.total_records !== undefined && dom.cohortTotalRecords) {
            dom.cohortTotalRecords.textContent = Number(data.total_records).toLocaleString();
        }

        const cohorts = data.cohorts || (Array.isArray(data) ? data : []);

        if (typeof window.renderCohortCards === 'function') {
            window.renderCohortCards('cohort-cards-container', cohorts);
        } else if (window.AppCharts && typeof window.AppCharts.renderCohortCards === 'function') {
            window.AppCharts.renderCohortCards('cohort-cards-container', cohorts);
        }
    }

    async function loadHeatmapMatrix() {
        if (!dom.heatmapGrid) return;

        // Instant sub-2ms render from memory cache
        if (state.screenSleepMatrixCache !== null) {
            applyHeatmapMatrix(state.screenSleepMatrixCache);
            return;
        }

        try {
            const resp = await fetch('/api/analytics/distributions/screen-sleep-matrix');
            if (!resp.ok) return;
            const data = await resp.json();
            state.screenSleepMatrixCache = data;
            applyHeatmapMatrix(data);
        } catch (err) {
            // Non-blocking fallback
        }
    }

    function applyHeatmapMatrix(data) {
        if (typeof window.renderScreenSleepHeatmap === 'function') {
            window.renderScreenSleepHeatmap('heatmap-grid', data);
        } else if (window.AppCharts && typeof window.AppCharts.renderScreenSleepHeatmap === 'function') {
            window.AppCharts.renderScreenSleepHeatmap('heatmap-grid', data);
        }
    }

    async function updateBenchmarkOverlay() {
        // Echo current profile inputs to subtext labels
        if (dom.valUserScreenEcho) {
            dom.valUserScreenEcho.textContent = state.profile.daily_screen_time_hours.toFixed(1) + 'h';
        }
        if (dom.valUserSleepEcho) {
            dom.valUserSleepEcho.textContent = state.profile.sleep_hours.toFixed(1) + 'h';
        }
        if (dom.valUserOpensEcho) {
            dom.valUserOpensEcho.textContent = String(Math.round(state.profile.app_opens_per_day));
        }
        if (dom.valUserNotifsEcho) {
            dom.valUserNotifsEcho.textContent = String(Math.round(state.profile.notifications_per_day));
        }

        const cacheKey = state.profile.daily_screen_time_hours.toFixed(1) + '__' +
            state.profile.sleep_hours.toFixed(1) + '__' +
            Math.round(state.profile.app_opens_per_day) + '__' +
            Math.round(state.profile.notifications_per_day);

        // Instant sub-2ms lookup from memory cache
        if (state.benchmarkOverlayCache.has(cacheKey)) {
            const cached = state.benchmarkOverlayCache.get(cacheKey);
            applyBenchmarkOverlay(cached);
            return;
        }

        try {
            const payload = {
                daily_screen_time_hours: state.profile.daily_screen_time_hours,
                sleep_hours: state.profile.sleep_hours,
                app_opens_per_day: state.profile.app_opens_per_day,
                notifications_per_day: state.profile.notifications_per_day
            };

            const resp = await fetch('/api/analytics/distributions/benchmark-overlay', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!resp.ok) return;
            const data = await resp.json();
            state.benchmarkOverlayCache.set(cacheKey, data);
            applyBenchmarkOverlay(data);
        } catch (err) {
            // Local fallback percentile estimates
            const screenPct = Math.min(99, Math.max(1, Math.round((state.profile.daily_screen_time_hours / 14.0) * 100)));
            const sleepPct = Math.min(99, Math.max(1, Math.round((state.profile.sleep_hours / 12.0) * 100)));
            const opensPct = Math.min(99, Math.max(1, Math.round((state.profile.app_opens_per_day / 200.0) * 100)));
            const notifsPct = Math.min(99, Math.max(1, Math.round((state.profile.notifications_per_day / 250.0) * 100)));
            const fallbackData = {
                screen_time_percentile: screenPct,
                sleep_hours_percentile: sleepPct,
                sleep_duration_percentile: sleepPct,
                app_opens_percentile: opensPct,
                notifications_percentile: notifsPct,
                screen_time_label: screenPct + 'th Percentile in Screen Time',
                sleep_hours_label: sleepPct + 'th Percentile in Sleep Duration',
                population_mean_screen: 7.64,
                population_mean_sleep: 6.80
            };
            applyBenchmarkOverlay(fallbackData);
        }
    }

    function applyBenchmarkOverlay(data) {
        if (typeof window.renderQuantileOverlays === 'function') {
            window.renderQuantileOverlays(data);
        } else if (window.AppCharts && typeof window.AppCharts.renderQuantileOverlays === 'function') {
            window.AppCharts.renderQuantileOverlays(data);
        }
    }

    // -------------------------------------------------------------------------
    // TAB 3: What-If Simulation & Optimizer (M3 Support)
    // -------------------------------------------------------------------------
    function bindWhatIfControls() {
        if (!dom.leverRecreation || !dom.leverSleep || !dom.leverOpens) return;

        dom.leverRecreation.addEventListener('input', function (e) {
            state.whatIf.delta_recreation = parseFloat(e.target.value);
            if (dom.valLeverRecreation) {
                dom.valLeverRecreation.textContent = state.whatIf.delta_recreation.toFixed(2) + ' hrs';
            }
            dom.leverRecreation.setAttribute('aria-valuenow', state.whatIf.delta_recreation);
            runWhatIfSimulation();
        });

        dom.leverSleep.addEventListener('input', function (e) {
            state.whatIf.delta_sleep = parseFloat(e.target.value);
            if (dom.valLeverSleep) {
                dom.valLeverSleep.textContent = '+' + state.whatIf.delta_sleep.toFixed(2) + ' hrs';
            }
            dom.leverSleep.setAttribute('aria-valuenow', state.whatIf.delta_sleep);
            runWhatIfSimulation();
        });

        dom.leverOpens.addEventListener('input', function (e) {
            state.whatIf.delta_opens = parseInt(e.target.value, 10);
            if (dom.valLeverOpens) {
                dom.valLeverOpens.textContent = state.whatIf.delta_opens + ' opens';
            }
            dom.leverOpens.setAttribute('aria-valuenow', state.whatIf.delta_opens);
            runWhatIfSimulation();
        });

        if (dom.btnGenerateOptimal) {
            dom.btnGenerateOptimal.addEventListener('click', generateOptimalTarget);
        }

        if (dom.btnApplyOptimal) {
            dom.btnApplyOptimal.addEventListener('click', applyOptimalTargetToSliders);
        }
    }

    async function runWhatIfSimulation() {
        if (!dom.simBaselineProb || !dom.simCounterfactualProb) return;

        try {
            const reqPayload = {
                baseline_profile: state.profile,
                delta_daily_screen_time_hours: state.whatIf.delta_recreation,
                delta_social_media_hours: Math.round(state.whatIf.delta_recreation * 0.7 * 100) / 100,
                delta_gaming_hours: Math.round(state.whatIf.delta_recreation * 0.3 * 100) / 100,
                delta_sleep_hours: state.whatIf.delta_sleep,
                delta_app_opens_per_day: state.whatIf.delta_opens
            };

            const resp = await fetch('/api/analytics/what-if', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(reqPayload)
            });

            if (!resp.ok) {
                throw new Error('What-If simulation request failed');
            }
            const data = await resp.json();
            renderWhatIfOutput(data);
        } catch (err) {
            toastManager.showToast('Local Simulation Active', 'What-If calculation fell back to calibrated baseline model.', 'info');
            // Local simulation fallback conforming to ADR-0007
            const baseProb = state.lastProbability;
            const delta = (state.whatIf.delta_recreation * 0.04) - (state.whatIf.delta_sleep * 0.05) + (state.whatIf.delta_opens * 0.001);
            const simProb = Math.min(0.98, Math.max(0.02, baseProb + delta));
            const simScreen = Math.max(0.5, state.profile.daily_screen_time_hours + state.whatIf.delta_recreation);
            const simSleep = Math.min(18.0, Math.max(1.0, state.profile.sleep_hours + state.whatIf.delta_sleep));
            const simRecHours = Math.max(0.0, (state.profile.social_media_hours + state.profile.gaming_hours + state.whatIf.delta_recreation));

            renderWhatIfOutput({
                baseline_probability: baseProb,
                simulated_probability: simProb,
                risk_delta: simProb - baseProb,
                baseline_classification: baseProb >= (state.profile.threshold || 0.5) ? 'ADDICTION DETECTED' : 'HEALTHY',
                simulated_classification: simProb >= (state.profile.threshold || 0.5) ? 'ADDICTION DETECTED' : 'HEALTHY',
                baseline_ratios: {
                    screen_to_sleep: state.profile.daily_screen_time_hours / (state.profile.sleep_hours + 1e-5),
                    recreational_to_screen: (state.profile.social_media_hours + state.profile.gaming_hours) / (state.profile.daily_screen_time_hours + 1e-5)
                },
                simulated_ratios: {
                    screen_to_sleep: simScreen / (simSleep + 1e-5),
                    recreational_to_screen: simRecHours / (simScreen + 1e-5)
                },
                simulated_profile: {
                    daily_screen_time_hours: simScreen,
                    sleep_hours: simSleep
                },
                interventions: []
            });
        }
    }

    function renderWhatIfOutput(data) {
        if (!dom.simBaselineProb || !dom.simCounterfactualProb) return;

        const basePct = (data.baseline_probability * 100).toFixed(1);
        const simPct = (data.simulated_probability * 100).toFixed(1);
        const deltaPct = Math.abs(data.risk_delta * 100).toFixed(1);
        const threshold = state.profile.threshold || 0.50;

        dom.simBaselineProb.textContent = basePct + '%';
        if (dom.simBaselineBadge) {
            const isBaseAddicted = data.baseline_classification
                ? data.baseline_classification.toUpperCase().includes('ADDICTION')
                : data.baseline_probability >= threshold;
            dom.simBaselineBadge.className = 'badge ' + (isBaseAddicted ? 'badge-danger' : 'badge-low');
            dom.simBaselineBadge.textContent = isBaseAddicted ? 'At-Risk' : 'Healthy';
        }

        dom.simCounterfactualProb.textContent = simPct + '%';
        const isSimAddicted = data.simulated_classification
            ? data.simulated_classification.toUpperCase().includes('ADDICTION')
            : data.simulated_probability >= threshold;
        dom.simCounterfactualProb.style.color = isSimAddicted ? 'var(--danger)' : 'var(--low-risk)';

        if (dom.simCounterfactualBadge) {
            dom.simCounterfactualBadge.className = 'badge ' + (isSimAddicted ? 'badge-danger' : 'badge-low');
            dom.simCounterfactualBadge.textContent = isSimAddicted ? 'At-Risk' : 'Healthy';
        }

        if (dom.simDeltaBadge) {
            if (data.risk_delta < -0.0005) {
                dom.simDeltaBadge.className = 'delta-badge delta-reduction';
                dom.simDeltaBadge.textContent = '-' + deltaPct + '% Risk Reduction';
            } else if (data.risk_delta > 0.0005) {
                dom.simDeltaBadge.className = 'delta-badge delta-increase';
                dom.simDeltaBadge.textContent = '+' + deltaPct + '% Risk Increase';
            } else {
                dom.simDeltaBadge.className = 'delta-badge delta-reduction';
                dom.simDeltaBadge.textContent = '0.0% Risk Delta';
            }
        }

        if (data.baseline_ratios && data.simulated_ratios) {
            const baseSS = data.baseline_ratios.screen_to_sleep ?? data.baseline_ratios.screen_to_sleep_ratio;
            const simSS = data.simulated_ratios.screen_to_sleep ?? data.simulated_ratios.screen_to_sleep_ratio;
            const baseRec = data.baseline_ratios.recreational_share ?? data.baseline_ratios.recreational_to_screen;
            const simRec = data.simulated_ratios.recreational_share ?? data.simulated_ratios.recreational_to_screen;

            if (dom.simBaseSS && baseSS != null) dom.simBaseSS.textContent = Number(baseSS).toFixed(2) + 'x';
            if (dom.simNewSS && simSS != null) {
                dom.simNewSS.textContent = Number(simSS).toFixed(2) + 'x';
                dom.simNewSS.style.color = simSS < 1.0 ? 'var(--low-risk)' : (simSS <= 1.2 ? 'var(--amber)' : 'var(--danger)');
            }
            if (dom.simBaseRec && baseRec != null) dom.simBaseRec.textContent = (Number(baseRec) * 100).toFixed(1) + '%';
            if (dom.simNewRec && simRec != null) {
                dom.simNewRec.textContent = (Number(simRec) * 100).toFixed(1) + '%';
                dom.simNewRec.style.color = simRec < 0.5 ? 'var(--low-risk)' : (simRec <= 0.6 ? 'var(--amber)' : 'var(--danger)');
            }
        }

        if (dom.simBaseScreen) dom.simBaseScreen.textContent = state.profile.daily_screen_time_hours.toFixed(1) + 'h';
        if (dom.simNewScreen) {
            const newScreen = data.simulated_profile && data.simulated_profile.daily_screen_time_hours != null
                ? data.simulated_profile.daily_screen_time_hours
                : Math.max(0.0, state.profile.daily_screen_time_hours + state.whatIf.delta_recreation);
            dom.simNewScreen.textContent = Number(newScreen).toFixed(1) + 'h';
        }

        if (dom.simBaseSleep) dom.simBaseSleep.textContent = state.profile.sleep_hours.toFixed(1) + 'h';
        if (dom.simNewSleep) {
            const newSleep = data.simulated_profile && data.simulated_profile.sleep_hours != null
                ? data.simulated_profile.sleep_hours
                : Math.min(18.0, Math.max(1.0, state.profile.sleep_hours + state.whatIf.delta_sleep));
            dom.simNewSleep.textContent = Number(newSleep).toFixed(1) + 'h';
        }

        if (dom.simInterventionsList) {
            dom.simInterventionsList.innerHTML = '';
            const recs = data.interventions || data.simulated_interventions || [];
            if (recs.length > 0) {
                recs.forEach(function (rec) {
                    const item = document.createElement('div');
                    item.className = 'intervention-item';
                    item.textContent = rec;
                    dom.simInterventionsList.appendChild(item);
                });
            } else {
                const item = document.createElement('div');
                item.className = 'intervention-item optimal';
                item.textContent = 'All behavioral hygiene parameters met under counterfactual model.';
                dom.simInterventionsList.appendChild(item);
            }
        }
    }

    let optimalPrescription = null;

    async function generateOptimalTarget() {
        if (!dom.optimalTargetCard || !dom.optimalTargetText) return;
        try {
            const resp = await fetch('/api/analytics/what-if/optimize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ baseline_profile: state.profile })
            });

            if (!resp.ok) {
                throw new Error('Failed to compute optimal habit target');
            }
            const data = await resp.json();
            optimalPrescription = data;

            dom.optimalTargetCard.style.display = 'block';
            let pathwayText = '';
            if (Array.isArray(data.recommended_pathway) && data.recommended_pathway.length > 0) {
                pathwayText = data.recommended_pathway.join(' • ');
            } else if (data.summary) {
                pathwayText = data.summary;
            } else {
                pathwayText = 'Recommended adjustment: Reduce daily screen time by ' +
                    Number(data.target_screen_time_reduction_hours || 0).toFixed(1) + 'h, increase sleep by ' +
                    Number(data.target_sleep_increase_hours || 0).toFixed(1) + 'h, and batch ' +
                    (data.target_app_opens_reduction || 0) + ' app opens.';
            }
            dom.optimalTargetText.textContent = pathwayText;
        } catch (err) {
            toastManager.showToast('Offline Target Solver', 'Applied calibrated lifestyle recommendation offline.', 'info');
            optimalPrescription = {
                target_screen_time_reduction_hours: 1.5,
                target_sleep_increase_hours: 1.0,
                target_app_opens_reduction: 30,
                summary: 'Recommended habit adjustments: Reduce recreational screen time by 1.5h, increase sleep by 1.0h, and batch notifications.'
            };
            dom.optimalTargetCard.style.display = 'block';
            dom.optimalTargetText.textContent = optimalPrescription.summary;
        }
    }

    function applyOptimalTargetToSliders() {
        if (!optimalPrescription) return;

        const screenRed = optimalPrescription.target_screen_time_reduction_hours !== undefined
            ? optimalPrescription.target_screen_time_reduction_hours
            : (optimalPrescription.target_screen_reduction || 0);
        const sleepInc = optimalPrescription.target_sleep_increase_hours !== undefined
            ? optimalPrescription.target_sleep_increase_hours
            : (optimalPrescription.target_sleep_increase || 0);
        const opensRed = optimalPrescription.target_app_opens_reduction !== undefined
            ? optimalPrescription.target_app_opens_reduction
            : (optimalPrescription.target_opens_reduction || 0);

        state.whatIf.delta_recreation = -Math.abs(screenRed);
        state.whatIf.delta_sleep = Math.abs(sleepInc);
        state.whatIf.delta_opens = -Math.abs(opensRed);

        if (dom.leverRecreation && dom.valLeverRecreation) {
            dom.leverRecreation.value = state.whatIf.delta_recreation;
            dom.leverRecreation.setAttribute('aria-valuenow', state.whatIf.delta_recreation);
            dom.valLeverRecreation.textContent = state.whatIf.delta_recreation.toFixed(2) + ' hrs';
        }

        if (dom.leverSleep && dom.valLeverSleep) {
            dom.leverSleep.value = state.whatIf.delta_sleep;
            dom.leverSleep.setAttribute('aria-valuenow', state.whatIf.delta_sleep);
            dom.valLeverSleep.textContent = '+' + state.whatIf.delta_sleep.toFixed(2) + ' hrs';
        }

        if (dom.leverOpens && dom.valLeverOpens) {
            dom.leverOpens.value = state.whatIf.delta_opens;
            dom.leverOpens.setAttribute('aria-valuenow', state.whatIf.delta_opens);
            dom.valLeverOpens.textContent = state.whatIf.delta_opens + ' opens';
        }

        runWhatIfSimulation();
    }

    // -------------------------------------------------------------------------
    // TAB 4: Batch CSV Diagnostics (M3 Support)
    // -------------------------------------------------------------------------
    function bindBatchControls() {
        if (!dom.batchDropzone || !dom.batchFileInput) return;

        dom.batchDropzone.addEventListener('click', function () {
            dom.batchFileInput.click();
        });

        dom.batchDropzone.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                dom.batchFileInput.click();
            }
        });

        dom.batchDropzone.addEventListener('dragover', function (e) {
            e.preventDefault();
            dom.batchDropzone.classList.add('drag-over');
        });

        dom.batchDropzone.addEventListener('dragleave', function () {
            dom.batchDropzone.classList.remove('drag-over');
        });

        dom.batchDropzone.addEventListener('drop', function (e) {
            e.preventDefault();
            dom.batchDropzone.classList.remove('drag-over');
            if (e.dataTransfer && e.dataTransfer.files.length > 0) {
                handleBatchUpload(e.dataTransfer.files[0]);
            }
        });

        dom.batchFileInput.addEventListener('change', function (e) {
            if (e.target.files && e.target.files.length > 0) {
                handleBatchUpload(e.target.files[0]);
            }
        });

        if (dom.btnLoadDemoBatch) {
            dom.btnLoadDemoBatch.addEventListener('click', function (e) {
                e.stopPropagation();
                generateAndSubmitDemoBatch();
            });
        }

        if (dom.btnExportCsv) {
            dom.btnExportCsv.addEventListener('click', function () {
                if (state.batch.downloadToken) {
                    window.location.href = '/api/predict/batch/export?token=' + encodeURIComponent(state.batch.downloadToken);
                } else {
                    showBatchError('No batch scoring results available to export. Please upload or score a dataset first.');
                }
            });
        }
    }

    async function handleBatchUpload(file) {
        if (!file || (!file.name.toLowerCase().endsWith('.csv') && file.type !== 'text/csv')) {
            showBatchError('Please select a valid CSV file format.');
            return;
        }

        hideBatchError();
        if (dom.batchLoading) dom.batchLoading.style.display = 'block';
        if (dom.batchResultsView) dom.batchResultsView.style.display = 'none';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const url = '/api/predict/batch' + (state.profile.threshold ? '?threshold=' + encodeURIComponent(state.profile.threshold) : '');
            const resp = await fetch(url, {
                method: 'POST',
                body: formData
            });

            if (dom.batchLoading) dom.batchLoading.style.display = 'none';

            if (!resp.ok) {
                const errData = await resp.json().catch(function () { return {}; });
                let errorMsg = 'Batch processing failed';
                if (resp.status === 413) {
                    errorMsg = errData.detail || 'Batch upload exceeds maximum limit of 10,000 rows.';
                } else if (resp.status === 422) {
                    if (typeof errData.detail === 'string') {
                        errorMsg = errData.detail;
                    } else if (Array.isArray(errData.detail)) {
                        errorMsg = errData.detail.map(function (e) {
                            return (e.loc ? e.loc.join('.') + ': ' : '') + (e.msg || 'Validation error');
                        }).join('; ');
                    } else {
                        errorMsg = 'Validation error: Missing or invalid CSV headers/data.';
                    }
                } else {
                    errorMsg = errData.detail || ('Batch request failed with HTTP ' + resp.status);
                }
                showBatchError(errorMsg);
                return;
            }

            const data = await resp.json();
            renderBatchResults(data);
        } catch (err) {
            if (dom.batchLoading) dom.batchLoading.style.display = 'none';
            showBatchError(err.message || 'Error communicating with batch prediction service.');
        }
    }

    function generateAndSubmitDemoBatch() {
        const headers = [
            'age', 'gender', 'stress_level', 'academic_work_impact',
            'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
            'work_study_hours', 'weekend_screen_time', 'sleep_hours',
            'notifications_per_day', 'app_opens_per_day'
        ];

        const demoProfiles = [
            { age: 24, gender: 'Female', stress: 'High', impact: 'Yes', screen: 9.5, social: 4.8, gaming: 1.5, work: 2.2, weekend: 12.0, sleep: 5.2, notifs: 195, opens: 140 },
            { age: 22, gender: 'Male', stress: 'High', impact: 'Yes', screen: 10.2, social: 3.5, gaming: 4.2, work: 1.8, weekend: 13.5, sleep: 4.8, notifs: 210, opens: 165 },
            { age: 28, gender: 'Female', stress: 'Low', impact: 'No', screen: 4.2, social: 1.5, gaming: 0.2, work: 3.5, weekend: 5.5, sleep: 7.8, notifs: 65, opens: 45 },
            { age: 26, gender: 'Male', stress: 'Medium', impact: 'No', screen: 5.8, social: 2.2, gaming: 1.0, work: 3.0, weekend: 7.0, sleep: 7.2, notifs: 95, opens: 70 },
            { age: 21, gender: 'Other', stress: 'High', impact: 'Yes', screen: 8.8, social: 4.0, gaming: 2.0, work: 2.0, weekend: 11.0, sleep: 5.5, notifs: 175, opens: 125 },
            { age: 31, gender: 'Female', stress: 'Low', impact: 'No', screen: 3.8, social: 1.2, gaming: 0.0, work: 4.0, weekend: 4.5, sleep: 8.0, notifs: 50, opens: 38 },
            { age: 23, gender: 'Male', stress: 'High', impact: 'Yes', screen: 11.0, social: 5.0, gaming: 3.5, work: 1.5, weekend: 14.0, sleep: 4.5, notifs: 240, opens: 180 },
            { age: 29, gender: 'Female', stress: 'Medium', impact: 'Yes', screen: 7.2, social: 3.2, gaming: 0.8, work: 3.0, weekend: 8.5, sleep: 6.5, notifs: 130, opens: 90 },
            { age: 25, gender: 'Male', stress: 'Low', impact: 'No', screen: 4.5, social: 1.8, gaming: 0.5, work: 3.5, weekend: 6.0, sleep: 7.5, notifs: 75, opens: 55 },
            { age: 20, gender: 'Female', stress: 'High', impact: 'Yes', screen: 9.0, social: 5.2, gaming: 1.0, work: 2.0, weekend: 11.5, sleep: 5.0, notifs: 185, opens: 135 },
            { age: 33, gender: 'Male', stress: 'Low', impact: 'No', screen: 3.5, social: 1.0, gaming: 0.0, work: 4.5, weekend: 4.0, sleep: 8.2, notifs: 40, opens: 30 },
            { age: 27, gender: 'Female', stress: 'Medium', impact: 'No', screen: 6.2, social: 2.8, gaming: 0.5, work: 3.2, weekend: 7.5, sleep: 6.8, notifs: 110, opens: 80 },
            { age: 22, gender: 'Male', stress: 'High', impact: 'Yes', screen: 8.5, social: 3.8, gaming: 2.5, work: 1.8, weekend: 10.5, sleep: 5.4, notifs: 160, opens: 115 },
            { age: 30, gender: 'Other', stress: 'Medium', impact: 'No', screen: 5.0, social: 2.0, gaming: 0.5, work: 3.8, weekend: 6.5, sleep: 7.4, notifs: 85, opens: 60 },
            { age: 24, gender: 'Female', stress: 'High', impact: 'Yes', screen: 10.5, social: 6.0, gaming: 1.2, work: 2.0, weekend: 13.0, sleep: 4.6, notifs: 225, opens: 170 },
            { age: 34, gender: 'Male', stress: 'Low', impact: 'No', screen: 4.0, social: 1.2, gaming: 0.2, work: 4.0, weekend: 5.0, sleep: 7.6, notifs: 60, opens: 42 },
            { age: 19, gender: 'Male', stress: 'High', impact: 'Yes', screen: 9.8, social: 4.2, gaming: 3.2, work: 1.2, weekend: 12.5, sleep: 5.0, notifs: 200, opens: 150 },
            { age: 27, gender: 'Female', stress: 'Low', impact: 'No', screen: 4.8, social: 2.0, gaming: 0.3, work: 3.5, weekend: 5.8, sleep: 7.7, notifs: 70, opens: 50 },
            { age: 25, gender: 'Other', stress: 'Medium', impact: 'Yes', screen: 7.6, social: 3.6, gaming: 1.4, work: 2.5, weekend: 9.5, sleep: 6.2, notifs: 145, opens: 105 },
            { age: 23, gender: 'Female', stress: 'High', impact: 'Yes', screen: 8.9, social: 4.5, gaming: 1.5, work: 2.2, weekend: 11.2, sleep: 5.3, notifs: 180, opens: 130 }
        ];

        const csvLines = [headers.join(',')];
        demoProfiles.forEach(function (p) {
            csvLines.push([
                p.age, p.gender, p.stress, p.impact,
                p.screen, p.social, p.gaming, p.work,
                p.weekend, p.sleep, p.notifs, p.opens
            ].join(','));
        });

        const blob = new Blob([csvLines.join('\n')], { type: 'text/csv' });
        const file = new File([blob], 'demo_cohort_20_records.csv', { type: 'text/csv' });
        handleBatchUpload(file);
    }

    function showBatchError(msg) {
        if (dom.batchError) {
            dom.batchError.style.display = 'flex';
        }
        if (dom.batchErrorMsg) {
            dom.batchErrorMsg.textContent = msg;
        }
    }

    function hideBatchError() {
        if (dom.batchError) {
            dom.batchError.style.display = 'none';
        }
    }

    function renderBatchResults(data) {
        state.batch.downloadToken = data.download_token;

        if (dom.batchTotalRecords) dom.batchTotalRecords.textContent = data.total_records != null ? data.total_records : '0';
        if (dom.batchAddictionRate) dom.batchAddictionRate.textContent = (data.addiction_prevalence_pct != null ? Number(data.addiction_prevalence_pct).toFixed(1) : '0.0') + '%';
        if (dom.batchHighRiskRate) dom.batchHighRiskRate.textContent = (data.high_risk_pct != null ? Number(data.high_risk_pct).toFixed(1) : '0.0') + '%';
        if (dom.batchMeanSS) dom.batchMeanSS.textContent = (data.mean_screen_to_sleep != null ? Number(data.mean_screen_to_sleep).toFixed(2) : '0.00') + 'x';

        const rows = data.preview_rows || data.sample_records || [];
        if (dom.batchTableBody) {
            dom.batchTableBody.innerHTML = '';
            if (rows.length === 0) {
                const tr = document.createElement('tr');
                tr.innerHTML = '<td colspan="8" style="text-align: center; color: var(--text-3); padding: 20px;">No preview records available.</td>';
                dom.batchTableBody.appendChild(tr);
            } else {
                rows.slice(0, 15).forEach(function (r, idx) {
                    const tr = document.createElement('tr');
                    const isAddicted = r.classification === 'ADDICTION DETECTED' || r.prediction === 1;
                    const prob = r.predicted_probability != null ? r.predicted_probability : (r.probability || 0.5);
                    const riskPct = (Number(prob) * 100).toFixed(1);
                    const screenTime = r.daily_screen_time_hours != null ? Number(r.daily_screen_time_hours).toFixed(1) + 'h' : '--';
                    const sleepTime = r.sleep_hours != null ? Number(r.sleep_hours).toFixed(1) + 'h' : '--';
                    const age = r.age != null ? r.age : '--';
                    const gender = r.gender || '--';
                    const primaryAdvisory = r.primary_intervention || 'Balanced Routine';

                    tr.innerHTML =
                        '<td>' + (idx + 1) + '</td>' +
                        '<td>' + age + '</td>' +
                        '<td>' + gender + '</td>' +
                        '<td>' + screenTime + '</td>' +
                        '<td>' + sleepTime + '</td>' +
                        '<td><strong>' + riskPct + '%</strong></td>' +
                        '<td>' +
                            '<span class="badge ' + (isAddicted ? 'badge-danger' : 'badge-low') + '">' +
                                (isAddicted ? 'Addiction Detected' : 'Healthy Pattern') +
                            '</span>' +
                        '</td>' +
                        '<td style="font-size: 0.8rem; color: var(--text-2);">' + primaryAdvisory + '</td>';
                    dom.batchTableBody.appendChild(tr);
                });
            }
        }

        if (dom.batchResultsView) {
            dom.batchResultsView.style.display = 'block';
        }
    }

    // -------------------------------------------------------------------------
    // Health & System Telemetry Probe
    // -------------------------------------------------------------------------
    async function checkSystemHealth() {
        try {
            const resp = await fetch('/health');
            if (resp.ok) {
                const data = await resp.json();
                if (data.model_loaded && dom.telemetryModel) {
                    dom.telemetryModel.textContent = 'MODEL: ' + String(data.model_family).toUpperCase() + ' FOLD ' + data.fold;
                }
            }
        } catch (err) {
            // Non-blocking telemetry warning
        }
    }

    // Initialize on DOM Ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
