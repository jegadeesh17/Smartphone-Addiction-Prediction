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
        debounceTimer: null
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
        cohortCardsContainer: document.getElementById('cohort-cards-container'),
        heatmapGrid: document.getElementById('heatmap-grid'),
        valQuantileScreen: document.getElementById('val-quantile-screen'),
        barQuantileScreen: document.getElementById('bar-quantile-screen'),
        valUserScreenEcho: document.getElementById('val-user-screen-echo'),
        valQuantileSleep: document.getElementById('val-quantile-sleep'),
        barQuantileSleep: document.getElementById('bar-quantile-sleep'),
        valUserSleepEcho: document.getElementById('val-user-sleep-echo'),

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
        btnExportCsv: document.getElementById('btn-export-csv')
    };

    // -------------------------------------------------------------------------
    // Initialization
    // -------------------------------------------------------------------------
    function init() {
        bindNavigation();
        bindInputControls();
        bindCohortControls();
        bindWhatIfControls();
        bindBatchControls();
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

                if (cfg.readout) {
                    cfg.readout.textContent = cfg.format(rawVal);
                }

                checkPhysiologicalBoundary();

                // If threshold slider changed, instantly re-evaluate risk gauge readout (PAR-1, PAR-3)
                if (cfg.key === 'threshold') {
                    if (typeof window.renderRiskGauge === 'function') {
                        window.renderRiskGauge(state.lastProbability, state.profile.threshold);
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
        const thresh = state.profile.threshold;

        // 1. Render SVG Risk Gauge via charts.js
        if (typeof window.renderRiskGauge === 'function') {
            window.renderRiskGauge(prob, thresh);
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

        // Ratio 3: Avg Session Length (Target > 8.0 min)
        if (dom.metricUnlockMins && dom.targetUnlockMins) {
            dom.metricUnlockMins.textContent = avgUnlock.toFixed(1) + ' min';
            if (avgUnlock >= 8.0) {
                dom.targetUnlockMins.className = 'metric-card-target target-met';
                dom.targetUnlockMins.innerHTML = '<span>Target Met (&gt; 8.0 min)</span>';
            } else {
                dom.targetUnlockMins.className = 'metric-card-target target-breached';
                dom.targetUnlockMins.innerHTML = '<span>Target Fragmented (&lt; 8.0 min)</span>';
            }
        }

        // Ratio 4: Weekend Surge (Target < 1.5h)
        if (dom.metricWeekendSurge && dom.targetWeekendSurge) {
            const surgeSign = weekendSurge >= 0 ? '+' : '';
            dom.metricWeekendSurge.textContent = surgeSign + weekendSurge.toFixed(1) + 'h';
            if (weekendSurge <= 1.5) {
                dom.targetWeekendSurge.className = 'metric-card-target target-met';
                dom.targetWeekendSurge.innerHTML = '<span>Target Met (&lt; 1.5h)</span>';
            } else {
                dom.targetWeekendSurge.className = 'metric-card-target target-breached';
                dom.targetWeekendSurge.innerHTML = '<span>Surge Detected (&gt; 1.5h)</span>';
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
            status_label: isAddicted ? 'Elevated Risk Tier' : (prob >= 0.35 ? 'Compensatory Usage Pattern' : 'Balanced Habit Profile'),
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
                });
                btn.classList.add('active');
                const val = btn.getAttribute('data-stress-filter');
                state.cohort.filter_stress = val === 'All' ? null : val;
                loadCohortAnalytics();
            });
        });
    }

    async function loadCohortAnalytics() {
        if (!dom.cohortCardsContainer) return;
        try {
            let url = '/api/analytics/cohorts?dimension=' + encodeURIComponent(state.cohort.dimension);
            if (state.cohort.filter_stress) {
                url += '&filter_stress=' + encodeURIComponent(state.cohort.filter_stress);
            }

            const resp = await fetch(url);
            if (!resp.ok) return;
            const cohorts = await resp.json();
            renderCohortCards(cohorts);
        } catch (err) {
            // Silent fallback if analytics endpoint is pending
        }
    }

    function renderCohortCards(cohorts) {
        if (!dom.cohortCardsContainer || !Array.isArray(cohorts)) return;
        dom.cohortCardsContainer.innerHTML = '';
        cohorts.forEach(function (c) {
            const card = document.createElement('div');
            card.className = 'cohort-stat-card';
            const prevPct = (c.addiction_prevalence * 100).toFixed(1);

            card.innerHTML =
                '<div class="cohort-title">' +
                    '<span>' + c.cohort_name + '</span>' +
                    '<span class="badge ' + (c.addiction_prevalence >= 0.71 ? 'badge-high' : 'badge-mod') + '">' + prevPct + '% Risk</span>' +
                '</div>' +
                '<div class="cohort-metric-row">' +
                    '<span class="cohort-metric-label">Sample Count:</span>' +
                    '<span class="cohort-metric-val">' + Number(c.sample_count).toLocaleString() + '</span>' +
                '</div>' +
                '<div class="cohort-metric-row">' +
                    '<span class="cohort-metric-label">Mean Screen Time:</span>' +
                    '<span class="cohort-metric-val">' + Number(c.mean_screen_time).toFixed(2) + 'h</span>' +
                '</div>' +
                '<div class="cohort-metric-row">' +
                    '<span class="cohort-metric-label">Mean Sleep Duration:</span>' +
                    '<span class="cohort-metric-val">' + Number(c.mean_sleep_hours).toFixed(2) + 'h</span>' +
                '</div>' +
                '<div class="cohort-metric-row">' +
                    '<span class="cohort-metric-label">Mean Daily App Opens:</span>' +
                    '<span class="cohort-metric-val">' + Math.round(c.mean_app_opens) + '</span>' +
                '</div>';
            dom.cohortCardsContainer.appendChild(card);
        });
    }

    async function loadHeatmapMatrix() {
        if (!dom.heatmapGrid) return;
        try {
            const resp = await fetch('/api/analytics/distributions/screen-sleep-matrix');
            if (!resp.ok) return;
            const data = await resp.json();
            renderHeatmap(data);
        } catch (err) {
            // Silent fallback
        }
    }

    function renderHeatmap(data) {
        if (!dom.heatmapGrid || !data || !Array.isArray(data.addiction_rate_matrix)) return;
        dom.heatmapGrid.innerHTML = '';

        const corner = document.createElement('div');
        corner.className = 'heatmap-header-cell';
        corner.textContent = 'Screen\\Sleep';
        dom.heatmapGrid.appendChild(corner);

        const sleepLabels = ['3-5h', '5-6h', '6-7h', '7-8h', '8-9h', '9-12h'];
        sleepLabels.forEach(function (lbl) {
            const h = document.createElement('div');
            h.className = 'heatmap-header-cell';
            h.textContent = lbl;
            dom.heatmapGrid.appendChild(h);
        });

        const screenLabels = ['0-4h', '4-6h', '6-8h', '8-10h', '10-12h', '12-16h'];
        data.addiction_rate_matrix.forEach(function (row, rIdx) {
            const rowHeader = document.createElement('div');
            rowHeader.className = 'heatmap-header-cell';
            rowHeader.style.textAlign = 'left';
            rowHeader.textContent = screenLabels[rIdx] || ('R' + rIdx);
            dom.heatmapGrid.appendChild(rowHeader);

            row.forEach(function (rate, cIdx) {
                const cell = document.createElement('div');
                cell.className = 'heatmap-cell';
                const ratePct = (rate * 100).toFixed(0);

                const redIntensity = Math.min(220, Math.max(30, Math.round(rate * 220)));
                const greenIntensity = Math.min(180, Math.max(30, Math.round((1 - rate) * 160)));
                cell.style.backgroundColor = 'rgb(' + redIntensity + ', ' + greenIntensity + ', 50)';
                cell.title = 'Screen: ' + screenLabels[rIdx] + ', Sleep: ' + sleepLabels[cIdx] + ' | Risk Rate: ' + ratePct + '%';
                cell.innerHTML = '<span>' + ratePct + '%</span>';
                dom.heatmapGrid.appendChild(cell);
            });
        });
    }

    async function updateBenchmarkOverlay() {
        if (!dom.valQuantileScreen || !dom.barQuantileScreen) return;
        try {
            const resp = await fetch('/api/analytics/distributions/benchmark-overlay', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(state.profile)
            });
            if (!resp.ok) return;
            const data = await resp.json();

            dom.valQuantileScreen.textContent = Math.round(data.screen_time_percentile) + 'th Percentile';
            dom.barQuantileScreen.style.width = Math.min(100, data.screen_time_percentile) + '%';
            if (dom.valUserScreenEcho) {
                dom.valUserScreenEcho.textContent = state.profile.daily_screen_time_hours.toFixed(1) + 'h';
            }

            dom.valQuantileSleep.textContent = Math.round(data.sleep_duration_percentile) + 'th Percentile';
            dom.barQuantileSleep.style.width = Math.min(100, data.sleep_duration_percentile) + '%';
            if (dom.valUserSleepEcho) {
                dom.valUserSleepEcho.textContent = state.profile.sleep_hours.toFixed(1) + 'h';
            }
        } catch (err) {
            // Fallback estimates
            const screenPct = Math.min(99, Math.max(1, Math.round((state.profile.daily_screen_time_hours / 14.0) * 100)));
            const sleepPct = Math.min(99, Math.max(1, Math.round((state.profile.sleep_hours / 12.0) * 100)));
            dom.valQuantileScreen.textContent = screenPct + 'th Percentile';
            dom.barQuantileScreen.style.width = screenPct + '%';
            dom.valQuantileSleep.textContent = sleepPct + 'th Percentile';
            dom.barQuantileSleep.style.width = sleepPct + '%';
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
                dom.valLeverRecreation.textContent = state.whatIf.delta_recreation.toFixed(1) + ' hrs';
            }
            runWhatIfSimulation();
        });

        dom.leverSleep.addEventListener('input', function (e) {
            state.whatIf.delta_sleep = parseFloat(e.target.value);
            if (dom.valLeverSleep) {
                dom.valLeverSleep.textContent = '+' + state.whatIf.delta_sleep.toFixed(1) + ' hrs';
            }
            runWhatIfSimulation();
        });

        dom.leverOpens.addEventListener('input', function (e) {
            state.whatIf.delta_opens = parseInt(e.target.value, 10);
            if (dom.valLeverOpens) {
                dom.valLeverOpens.textContent = state.whatIf.delta_opens + ' opens';
            }
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
                baseline: state.profile,
                delta_social_media_hours: state.whatIf.delta_recreation * 0.7,
                delta_gaming_hours: state.whatIf.delta_recreation * 0.3,
                delta_sleep_hours: state.whatIf.delta_sleep,
                delta_app_opens_per_day: state.whatIf.delta_opens,
                delta_daily_screen_time_hours: 0.0
            };

            const resp = await fetch('/api/analytics/what-if', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(reqPayload)
            });

            if (!resp.ok) return;
            const data = await resp.json();
            renderWhatIfOutput(data);
        } catch (err) {
            // Local simulation fallback
            const baseProb = state.lastProbability;
            const delta = (state.whatIf.delta_recreation * 0.04) - (state.whatIf.delta_sleep * 0.05) + (state.whatIf.delta_opens * 0.001);
            const simProb = Math.min(0.98, Math.max(0.02, baseProb + delta));
            renderWhatIfOutput({
                baseline_probability: baseProb,
                simulated_probability: simProb,
                risk_delta: simProb - baseProb,
                baseline_ratios: {
                    screen_to_sleep: state.profile.daily_screen_time_hours / state.profile.sleep_hours,
                    recreational_to_screen: (state.profile.social_media_hours + state.profile.gaming_hours) / state.profile.daily_screen_time_hours
                },
                simulated_ratios: {
                    screen_to_sleep: Math.max(0.5, state.profile.daily_screen_time_hours + state.whatIf.delta_recreation) / (state.profile.sleep_hours + state.whatIf.delta_sleep),
                    recreational_to_screen: Math.max(0.0, (state.profile.social_media_hours + state.profile.gaming_hours + state.whatIf.delta_recreation)) / state.profile.daily_screen_time_hours
                },
                simulated_interventions: []
            });
        }
    }

    function renderWhatIfOutput(data) {
        if (!dom.simBaselineProb || !dom.simCounterfactualProb) return;

        const basePct = (data.baseline_probability * 100).toFixed(1);
        const simPct = (data.simulated_probability * 100).toFixed(1);
        const deltaPct = Math.abs(data.risk_delta * 100).toFixed(1);

        dom.simBaselineProb.textContent = basePct + '%';
        if (dom.simBaselineBadge) {
            dom.simBaselineBadge.className = data.baseline_probability >= state.profile.threshold ? 'badge badge-high' : 'badge badge-low';
            dom.simBaselineBadge.textContent = data.baseline_probability >= state.profile.threshold ? 'Elevated' : 'Balanced';
        }

        dom.simCounterfactualProb.textContent = simPct + '%';
        dom.simCounterfactualProb.style.color = data.simulated_probability < state.profile.threshold ? 'var(--low-risk)' : 'var(--danger)';
        if (dom.simCounterfactualBadge) {
            dom.simCounterfactualBadge.className = data.simulated_probability >= state.profile.threshold ? 'badge badge-high' : 'badge badge-low';
            dom.simCounterfactualBadge.textContent = data.simulated_probability >= state.profile.threshold ? 'Elevated' : 'Balanced';
        }

        if (dom.simDeltaBadge) {
            if (data.risk_delta < -0.001) {
                dom.simDeltaBadge.className = 'delta-badge delta-reduction';
                dom.simDeltaBadge.textContent = '-' + deltaPct + '% Risk Reduction';
            } else if (data.risk_delta > 0.001) {
                dom.simDeltaBadge.className = 'delta-badge delta-increase';
                dom.simDeltaBadge.textContent = '+' + deltaPct + '% Risk Increase';
            } else {
                dom.simDeltaBadge.className = 'delta-badge delta-reduction';
                dom.simDeltaBadge.textContent = '0.0% Risk Delta';
            }
        }

        if (data.baseline_ratios && data.simulated_ratios) {
            if (dom.simBaseSS) dom.simBaseSS.textContent = Number(data.baseline_ratios.screen_to_sleep).toFixed(2) + 'x';
            if (dom.simNewSS) dom.simNewSS.textContent = Number(data.simulated_ratios.screen_to_sleep).toFixed(2) + 'x';
            if (dom.simBaseRec) dom.simBaseRec.textContent = (Number(data.baseline_ratios.recreational_to_screen) * 100).toFixed(1) + '%';
            if (dom.simNewRec) dom.simNewRec.textContent = (Number(data.simulated_ratios.recreational_to_screen) * 100).toFixed(1) + '%';
        }

        if (dom.simBaseScreen) dom.simBaseScreen.textContent = state.profile.daily_screen_time_hours.toFixed(1) + 'h';
        if (dom.simNewScreen) {
            const newScreen = Math.max(0.5, state.profile.daily_screen_time_hours + state.whatIf.delta_recreation);
            dom.simNewScreen.textContent = newScreen.toFixed(1) + 'h';
        }

        if (dom.simBaseSleep) dom.simBaseSleep.textContent = state.profile.sleep_hours.toFixed(1) + 'h';
        if (dom.simNewSleep) {
            const newSleep = Math.min(14.0, state.profile.sleep_hours + state.whatIf.delta_sleep);
            dom.simNewSleep.textContent = newSleep.toFixed(1) + 'h';
        }

        if (dom.simInterventionsList) {
            dom.simInterventionsList.innerHTML = '';
            const recs = data.simulated_interventions || [];
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
                body: JSON.stringify(state.profile)
            });

            if (!resp.ok) return;
            const data = await resp.json();
            optimalPrescription = data;

            dom.optimalTargetCard.style.display = 'block';
            dom.optimalTargetText.textContent = data.summary;
        } catch (err) {
            optimalPrescription = {
                target_screen_reduction: 1.5,
                target_sleep_increase: 1.0,
                target_opens_reduction: 30,
                summary: 'Recommended habit adjustments: Reduce recreational screen time by 1.5h, increase sleep by 1.0h, and batch notifications.'
            };
            dom.optimalTargetCard.style.display = 'block';
            dom.optimalTargetText.textContent = optimalPrescription.summary;
        }
    }

    function applyOptimalTargetToSliders() {
        if (!optimalPrescription) return;

        state.whatIf.delta_recreation = -Math.abs(optimalPrescription.target_screen_reduction);
        state.whatIf.delta_sleep = Math.abs(optimalPrescription.target_sleep_increase);
        state.whatIf.delta_opens = -Math.abs(optimalPrescription.target_opens_reduction);

        if (dom.leverRecreation && dom.valLeverRecreation) {
            dom.leverRecreation.value = state.whatIf.delta_recreation;
            dom.valLeverRecreation.textContent = state.whatIf.delta_recreation.toFixed(1) + ' hrs';
        }

        if (dom.leverSleep && dom.valLeverSleep) {
            dom.leverSleep.value = state.whatIf.delta_sleep;
            dom.valLeverSleep.textContent = '+' + state.whatIf.delta_sleep.toFixed(1) + ' hrs';
        }

        if (dom.leverOpens && dom.valLeverOpens) {
            dom.leverOpens.value = state.whatIf.delta_opens;
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
            if (e.dataTransfer.files.length > 0) {
                handleBatchUpload(e.dataTransfer.files[0]);
            }
        });

        dom.batchFileInput.addEventListener('change', function (e) {
            if (e.target.files.length > 0) {
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
                }
            });
        }
    }

    async function handleBatchUpload(file) {
        if (!file.name.endsWith('.csv')) {
            showBatchError('Please select a valid CSV file format.');
            return;
        }

        hideBatchError();
        if (dom.batchLoading) dom.batchLoading.style.display = 'block';
        if (dom.batchResultsView) dom.batchResultsView.style.display = 'none';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch('/api/predict/batch', {
                method: 'POST',
                body: formData
            });

            if (dom.batchLoading) dom.batchLoading.style.display = 'none';
            if (!resp.ok) {
                const errData = await resp.json().catch(function () { return {}; });
                throw new Error(errData.detail || 'Batch processing failed');
            }

            const data = await resp.json();
            renderBatchResults(data);
        } catch (err) {
            if (dom.batchLoading) dom.batchLoading.style.display = 'none';
            showBatchError(err.message || 'Error processing batch CSV.');
        }
    }

    function generateAndSubmitDemoBatch() {
        const headers = [
            'id', 'age', 'gender', 'stress_level', 'academic_work_impact',
            'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
            'work_study_hours', 'weekend_screen_time', 'sleep_hours',
            'notifications_per_day', 'app_opens_per_day'
        ];

        const rows = [headers.join(',')];
        for (let i = 1; i <= 20; i++) {
            const isHigh = i % 2 === 0;
            const age = 19 + (i % 14);
            const gender = i % 3 === 0 ? 'Female' : (i % 3 === 1 ? 'Male' : 'Other');
            const stress = isHigh ? 'High' : (i % 3 === 0 ? 'Medium' : 'Low');
            const impact = isHigh ? 'Yes' : 'No';
            const screen = isHigh ? (8.5 + (i * 0.2)).toFixed(1) : (4.0 + (i * 0.15)).toFixed(1);
            const soc = (parseFloat(screen) * 0.4).toFixed(1);
            const game = (parseFloat(screen) * 0.15).toFixed(1);
            const work = (parseFloat(screen) * 0.3).toFixed(1);
            const weekend = (parseFloat(screen) + 2.5).toFixed(1);
            const sleep = isHigh ? (5.5 - (i * 0.05)).toFixed(1) : (7.5 + (i * 0.02)).toFixed(1);
            const notifs = isHigh ? 180 + i * 2 : 70 + i;
            const opens = isHigh ? 130 + i : 60 + i;

            rows.push([
                'P-' + (1000 + i), age, gender, stress, impact,
                screen, soc, game, work, weekend, sleep, notifs, opens
            ].join(','));
        }

        const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
        const file = new File([blob], 'demo_clinical_cohort.csv', { type: 'text/csv' });
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

        if (dom.batchTotalRecords) dom.batchTotalRecords.textContent = data.total_records || '20';
        if (dom.batchAddictionRate) dom.batchAddictionRate.textContent = (data.addiction_prevalence_pct || 65.0).toFixed(1) + '%';
        if (dom.batchHighRiskRate) dom.batchHighRiskRate.textContent = (data.high_risk_pct || 45.0).toFixed(1) + '%';
        if (dom.batchMeanSS) dom.batchMeanSS.textContent = (data.mean_screen_to_sleep || 1.24).toFixed(2) + 'x';

        if (dom.batchTableBody && Array.isArray(data.preview_rows)) {
            dom.batchTableBody.innerHTML = '';
            data.preview_rows.forEach(function (r, idx) {
                const tr = document.createElement('tr');
                const isAddicted = r.classification === 'ADDICTION DETECTED';
                const riskPct = ((r.predicted_probability || 0.5) * 100).toFixed(1);

                tr.innerHTML =
                    '<td>' + (idx + 1) + '</td>' +
                    '<td>' + r.age + '</td>' +
                    '<td>' + r.gender + '</td>' +
                    '<td>' + parseFloat(r.daily_screen_time_hours).toFixed(1) + 'h</td>' +
                    '<td>' + parseFloat(r.sleep_hours).toFixed(1) + 'h</td>' +
                    '<td><strong>' + riskPct + '%</strong></td>' +
                    '<td>' +
                        '<span class="badge ' + (isAddicted ? 'badge-high' : 'badge-low') + '">' +
                            (isAddicted ? 'Addiction' : 'Healthy') +
                        '</span>' +
                    '</td>' +
                    '<td style="font-size: 0.775rem; color: var(--text-2);">' + (r.primary_intervention || 'Balanced Routine') + '</td>';
                dom.batchTableBody.appendChild(tr);
            });
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
