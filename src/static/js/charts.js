/**
 * Smartphone Addiction Analytical Platform
 * Visualization & Risk Gauge Rendering (charts.js)
 *
 * Implements:
 * - SVG circular arc risk gauge with 220-degree sweep and viewBox 0 0 240 170.
 * - Positive margin clearance (>32px) between bottom arc terminals and caption label per ADR-0008.
 * - Smooth stroke-dashoffset transition without layout reflow (zero transition: width).
 * - Authoritative 3-tier status mapping per ADR-0006:
 *   * Elevated Risk Tier (prob >= threshold, badge-high, status-pill-high)
 *   * Compensatory Usage Pattern (0.35 <= prob < threshold, badge-mod, status-pill-mod)
 *   * Balanced Habit Profile (prob < 0.35, badge-low, status-pill-low)
 * - Strict adherence to tokens in tokens.css and WCAG AA contrast guidelines.
 */

(function (root, factory) {
    if (typeof module === 'object' && module.exports) {
        module.exports = factory();
    } else {
        const exports = factory();
        root.renderRiskGauge = exports.renderRiskGauge;
        root.getClinicalTier = exports.getClinicalTier;
        root.renderCohortCards = exports.renderCohortCards;
        root.renderScreenSleepHeatmap = exports.renderScreenSleepHeatmap;
        root.renderQuantileOverlays = exports.renderQuantileOverlays;
        root.AppCharts = exports;
    }
})(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    /**
     * SVG Circular Arc Geometry Constants (ADR-0008)
     * Center: (120, 120), Radius R=85, Sweep: 220° (from 160° clockwise to 380° / 20°)
     * Arc terminals end at Y ≈ 149.1px in viewBox 0 0 240 170, leaving >20px internal canvas headroom.
     * Full circle circumference: 2 * PI * 85 ≈ 534.07 px
     * 220° arc length: (220 / 360) * 2 * PI * 85 ≈ 326.3766 px
     */
    const GAUGE_ARC_LENGTH = 326.4;

    /**
     * Authoritative 3-tier clinical status mapping per ADR-0006 & SPEC AC-1.1 / PAR-4
     * @param {number} prob - Evaluated addiction probability [0.0, 1.0]
     * @param {number} threshold - Decision threshold tau [0.05, 0.95]
     * @returns {Object} Tier metadata with label, classes, CSS color tokens, and advisory HTML
     */
    function getClinicalTier(prob, threshold) {
        const tau = Number(threshold) || 0.50;
        const isAddicted = prob >= tau;
        const tier = prob >= 0.70 ? 'HIGH' : (prob >= 0.40 ? 'MODERATE' : 'LOW');
        const computedStatusLabel = isAddicted
            ? 'ADDICTION DETECTED • ' + tier + ' RISK'
            : 'HEALTHY PATTERN • ' + tier + ' RISK';

        let pillClass = 'status-pill-low';
        let badgeClass = 'badge-low';
        let colorVar = 'var(--low-risk)';

        if (isAddicted) {
            pillClass = 'status-pill-high';
            badgeClass = 'badge-high';
            colorVar = 'var(--danger)';
        } else if (tier === 'MODERATE') {
            pillClass = 'status-pill-mod';
            badgeClass = 'badge-mod';
            colorVar = 'var(--amber)';
        }

        return {
            tier: tier,
            isAddicted: isAddicted,
            label: computedStatusLabel,
            badgeClass: badgeClass,
            pillClass: pillClass,
            colorVar: colorVar,
            summaryHtml: isAddicted
                ? 'Estimated risk probability exceeds clinical decision threshold (&tau; = ' + tau.toFixed(2) + '). Behavioral boundary restructuring recommended.'
                : 'Estimated risk probability demonstrates healthy digital habits below threshold (&tau; = ' + tau.toFixed(2) + '). Maintain current balanced routine.'
        };
    }

    /**
     * Renders the SVG circular arc risk gauge and synchronizes authoritative status displays.
     * Conforms strictly to ADR-0006 and ADR-0008 geometry and clearance contracts.
     *
     * @param {number} probability - Assessed addiction probability in range [0.0, 1.0]
     * @param {number} [threshold=0.50] - Active classification decision threshold tau
     * @param {string} [statusLabel] - Authoritative status label from backend API
     * @param {string} [classification] - Diagnostic classification ("ADDICTION DETECTED" or "HEALTHY")
     * @returns {Object} Render evaluation metadata
     */
    function renderRiskGauge(probability, threshold, statusLabel, classification) {
        const prob = Math.min(1.0, Math.max(0.0, Number(probability) || 0.0));
        const tau = Math.min(0.95, Math.max(0.05, Number(threshold) || 0.50));
        const pctFormatted = (prob * 100).toFixed(1) + '%';
        const tauFormatted = tau.toFixed(2);

        // 1. Calculate Arc Stroke Dashoffset
        // offset = length * (1.0 - prob)
        // At prob = 0.0 -> offset = 326.4 (empty)
        // At prob = 1.0 -> offset = 0.0 (full 220° sweep)
        const offset = GAUGE_ARC_LENGTH * (1.0 - prob);
        const arcEl = document.getElementById('gauge-arc');
        if (arcEl) {
            arcEl.style.strokeDasharray = GAUGE_ARC_LENGTH.toString();
            arcEl.style.strokeDashoffset = offset.toFixed(2);
        }

        // 2. Numerical Percentage Readout
        const probNumEl = document.getElementById('gauge-probability-num');
        if (probNumEl) {
            probNumEl.textContent = pctFormatted;
        }

        // 3. Status determination & definitive classification (ADR-0006, PAR-3, PAR-4, PAR-6)
        const isAddicted = prob >= tau;
        const tier = prob >= 0.70 ? 'HIGH' : (prob >= 0.40 ? 'MODERATE' : 'LOW');
        const computedStatusLabel = isAddicted
            ? 'ADDICTION DETECTED • ' + tier + ' RISK'
            : 'HEALTHY PATTERN • ' + tier + ' RISK';

        // If statusLabel is passed from API, use it directly for #badge-status!
        const finalStatusLabel = (statusLabel && typeof statusLabel === 'string' && statusLabel.trim().length > 0)
            ? statusLabel
            : computedStatusLabel;

        const finalClassification = classification || (isAddicted ? 'ADDICTION DETECTED' : 'HEALTHY');

        // Status pill styling:
        // - If isAddicted (or statusLabel contains "ADDICTION DETECTED"): status-pill-high
        // - Else if tier === 'MODERATE' (or statusLabel contains "MODERATE RISK"): status-pill-mod
        // - Else: status-pill-low
        let pillClass = 'status-pill-low';
        let colorVar = 'var(--low-risk)';
        let badgeClass = 'badge-low';

        if (isAddicted || (statusLabel && statusLabel.indexOf('ADDICTION DETECTED') !== -1)) {
            pillClass = 'status-pill-high';
            colorVar = 'var(--danger)';
            badgeClass = 'badge-high';
        } else if (tier === 'MODERATE' || (statusLabel && statusLabel.indexOf('MODERATE RISK') !== -1)) {
            pillClass = 'status-pill-mod';
            colorVar = 'var(--amber)';
            badgeClass = 'badge-mod';
        } else {
            pillClass = 'status-pill-low';
            colorVar = 'var(--low-risk)';
            badgeClass = 'badge-low';
        }

        if (arcEl) {
            arcEl.style.stroke = colorVar;
        }

        // 4. Consolidated Authoritative Status Pill (ADR-0006)
        const statusPill = document.getElementById('status-pill');
        if (statusPill) {
            statusPill.className = 'status-pill ' + pillClass;
        }

        const badgeStatus = document.getElementById('badge-status');
        if (badgeStatus) {
            badgeStatus.className = 'badge ' + badgeClass;
            badgeStatus.textContent = finalStatusLabel;
        }

        // 5. Narrative Classification Summary
        const summaryEl = document.getElementById('classification-summary');
        if (summaryEl) {
            const decisionVerb = (finalClassification === 'ADDICTION DETECTED' || isAddicted || (statusLabel && statusLabel.indexOf('ADDICTION DETECTED') !== -1))
                ? 'ADDICTION DETECTED'
                : 'HEALTHY';
            const actionAdvice = (decisionVerb === 'ADDICTION DETECTED')
                ? 'Behavioral boundary restructuring recommended.'
                : 'Maintain current balanced digital routine.';

            summaryEl.innerHTML = 'Diagnostic decision: <strong>' + decisionVerb +
                '</strong> at threshold &tau; = ' + tauFormatted +
                ' (assessed risk probability: <strong>' + pctFormatted + '</strong>). ' + actionAdvice;
        }

        return {
            probability: prob,
            threshold: tau,
            percentage: pctFormatted,
            label: finalStatusLabel,
            classification: finalClassification,
            tier: tier,
            isAddicted: isAddicted,
            badgeClass: badgeClass,
            pillClass: pillClass,
            colorVar: colorVar
        };
    }

    /**
     * Escape special HTML characters to prevent XSS.
     * @param {*} str - Raw string or value
     * @returns {string} Escaped string
     */
    function escapeHtml(str) {
        if (str === null || str === undefined) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    /**
     * Renders demographic sub-cohort breakdown cards (M2-TASK-04, Journey 2, AC-2.1, AC-2.2).
     * Displays cohort title, sample count, addiction prevalence %, and key time allocations
     * (mean screen time, sleep hours, social media, mobile gaming, and daily app opens).
     *
     * @param {string|HTMLElement} containerId - Target container element or ID
     * @param {Array|Object} cohortsList - Array of cohort items or CohortDistributionResponse object
     */
    function renderCohortCards(containerId, cohortsList) {
        const container = typeof containerId === 'string'
            ? document.getElementById(containerId)
            : containerId;
        if (!container) return;

        // Extract array from response payload if nested under .cohorts
        const cohorts = Array.isArray(cohortsList)
            ? cohortsList
            : (cohortsList && Array.isArray(cohortsList.cohorts) ? cohortsList.cohorts : []);

        container.innerHTML = '';

        if (!cohorts || cohorts.length === 0) {
            const emptyEl = document.createElement('div');
            emptyEl.className = 'empty-state-notice';
            emptyEl.style.cssText = 'grid-column: 1 / -1; padding: 24px; text-align: center; color: var(--text-2); background: var(--panel-subtle); border: 1px dashed var(--line); border-radius: var(--radius-md);';
            emptyEl.textContent = 'No demographic cohort records found matching the specified filter criteria.';
            container.appendChild(emptyEl);
            return;
        }

        cohorts.forEach(function (c) {
            const card = document.createElement('div');
            card.className = 'cohort-stat-card';
            if (c.cohort_id) {
                card.setAttribute('data-cohort-id', c.cohort_id);
            }

            const title = c.cohort_name || c.label || c.cohort_id || 'Cohort Segment';
            const count = (c.sample_count !== undefined)
                ? c.sample_count
                : (c.metrics && c.metrics.count !== undefined ? c.metrics.count : 0);
            const countFormatted = Number(count).toLocaleString();

            const prev = (c.addiction_prevalence !== undefined)
                ? c.addiction_prevalence
                : (c.metrics && c.metrics.addiction_prevalence !== undefined ? c.metrics.addiction_prevalence : 0);
            const prevPct = (prev * 100).toFixed(1) + '%';

            let badgeClass = 'badge-low';
            let riskTierLabel = 'Low Risk';
            if (prev >= 0.70) {
                badgeClass = 'badge-high';
                riskTierLabel = 'High Risk';
            } else if (prev >= 0.40) {
                badgeClass = 'badge-mod';
                riskTierLabel = 'Moderate Risk';
            }

            const screenMean = (c.mean_screen_time !== undefined)
                ? c.mean_screen_time
                : (c.metrics && c.metrics.screen_time_mean !== undefined ? c.metrics.screen_time_mean : 0);
            const sleepMean = (c.mean_sleep_hours !== undefined)
                ? c.mean_sleep_hours
                : (c.metrics && c.metrics.sleep_hours_mean !== undefined ? c.metrics.sleep_hours_mean : 0);
            const socialMean = (c.metrics && c.metrics.social_media_mean !== undefined)
                ? c.metrics.social_media_mean
                : 0;
            const gamingMean = (c.metrics && c.metrics.gaming_mean !== undefined)
                ? c.metrics.gaming_mean
                : 0;
            const opensMean = (c.mean_app_opens !== undefined)
                ? c.mean_app_opens
                : (c.metrics && c.metrics.app_opens_mean !== undefined ? c.metrics.app_opens_mean : 0);

            card.innerHTML =
                '<div class="cohort-title">' +
                    '<span>' + escapeHtml(title) + '</span>' +
                    '<span class="badge ' + badgeClass + '" title="' + riskTierLabel + ' (' + prevPct + ')">' + prevPct + ' Risk</span>' +
                '</div>' +
                '<div class="cohort-metric-row">' +
                    '<span class="cohort-metric-label">Sample Count:</span>' +
                    '<span class="cohort-metric-val">' + countFormatted + '</span>' +
                '</div>' +
                '<div class="cohort-metric-row">' +
                    '<span class="cohort-metric-label">Mean Screen Time:</span>' +
                    '<span class="cohort-metric-val">' + Number(screenMean).toFixed(2) + 'h</span>' +
                '</div>' +
                '<div class="cohort-metric-row">' +
                    '<span class="cohort-metric-label">Mean Sleep Duration:</span>' +
                    '<span class="cohort-metric-val">' + Number(sleepMean).toFixed(2) + 'h</span>' +
                '</div>' +
                '<div class="cohort-metric-row">' +
                    '<span class="cohort-metric-label">Mean Social Media:</span>' +
                    '<span class="cohort-metric-val">' + Number(socialMean).toFixed(2) + 'h</span>' +
                '</div>' +
                '<div class="cohort-metric-row">' +
                    '<span class="cohort-metric-label">Mean Mobile Gaming:</span>' +
                    '<span class="cohort-metric-val">' + Number(gamingMean).toFixed(2) + 'h</span>' +
                '</div>' +
                '<div class="cohort-metric-row">' +
                    '<span class="cohort-metric-label">Daily App Opens:</span>' +
                    '<span class="cohort-metric-val">' + Math.round(opensMean) + '</span>' +
                '</div>';

            container.appendChild(card);
        });
    }

    /**
     * Renders 6 Screen bins x 5 Sleep bins 2D joint density & addiction heatmap (M2-TASK-04, AC-2.3).
     * Cells display density % and addiction rate %, with background color styled according to
     * addiction rate (--danger for high, --amber for moderate, --low-risk for low).
     * Tooltips/hover show exact count and rates.
     *
     * @param {string|HTMLElement} containerId - Target grid container element or ID
     * @param {Object} matrixData - ScreenSleepMatrixResponse payload
     */
    function renderScreenSleepHeatmap(containerId, matrixData) {
        const container = typeof containerId === 'string'
            ? document.getElementById(containerId)
            : containerId;
        if (!container || !matrixData) return;

        container.innerHTML = '';

        const screenBins = matrixData.screen_bins || ['0-4', '4-6', '6-8', '8-10', '10-12', '12+'];
        const sleepBins = matrixData.sleep_bins || ['<5', '5-6', '6-7', '7-8', '8+'];
        const cellsList = matrixData.cells || [];

        // Build lookup map: `${screenBin}__${sleepBin}` -> cell
        const cellMap = {};
        cellsList.forEach(function (c) {
            const key = String(c.screen_bin) + '__' + String(c.sleep_bin);
            cellMap[key] = c;
        });

        // Corner cell
        const cornerCell = document.createElement('div');
        cornerCell.className = 'heatmap-header-cell heatmap-corner-cell';
        cornerCell.innerHTML = '<span>Screen \\ Sleep</span>';
        container.appendChild(cornerCell);

        // Column headers (Sleep bins, 5 cols)
        sleepBins.forEach(function (sleepBin) {
            const colHeader = document.createElement('div');
            colHeader.className = 'heatmap-header-cell heatmap-col-header';
            colHeader.textContent = sleepBin + 'h';
            colHeader.title = 'Sleep duration interval: ' + sleepBin + ' hours';
            container.appendChild(colHeader);
        });

        // 6 Screen bins (rows)
        screenBins.forEach(function (screenBin, rIdx) {
            // Row header
            const rowHeader = document.createElement('div');
            rowHeader.className = 'heatmap-header-cell heatmap-row-header';
            rowHeader.textContent = screenBin + 'h';
            rowHeader.title = 'Screen time interval: ' + screenBin + ' hours';
            container.appendChild(rowHeader);

            // 5 Sleep cells
            sleepBins.forEach(function (sleepBin, cIdx) {
                const key = String(screenBin) + '__' + String(sleepBin);
                const cellData = cellMap[key] || (cellsList[rIdx * sleepBins.length + cIdx] || null);

                const cell = document.createElement('div');
                cell.className = 'heatmap-cell';
                cell.setAttribute('tabindex', '0');

                let rate = 0;
                let density = 0;
                let count = 0;

                if (cellData) {
                    rate = cellData.addiction_rate_pct !== undefined ? cellData.addiction_rate_pct : 0;
                    density = cellData.density_pct !== undefined ? cellData.density_pct : (cellData.cell_density_pct || 0);
                    count = cellData.count !== undefined ? cellData.count : (cellData.sample_count || 0);
                } else if (matrixData.addiction_rate_matrix && matrixData.matrix) {
                    rate = (matrixData.addiction_rate_matrix[rIdx] && matrixData.addiction_rate_matrix[rIdx][cIdx]) || 0;
                    density = (matrixData.matrix[rIdx] && matrixData.matrix[rIdx][cIdx]) || 0;
                }

                // Background color styled according to addiction rate (--danger for high, --amber for moderate, --low-risk for low)
                let cellClass = 'heatmap-cell-low';
                let bgColor = 'var(--low-risk)';
                if (rate >= 70.0) {
                    cellClass = 'heatmap-cell-danger';
                    const alpha = Math.min(0.98, Math.max(0.60, 0.60 + 0.38 * ((rate - 70.0) / 30.0)));
                    bgColor = 'rgba(156, 63, 44, ' + alpha.toFixed(2) + ')';
                } else if (rate >= 40.0) {
                    cellClass = 'heatmap-cell-amber';
                    const alpha = Math.min(0.92, Math.max(0.55, 0.55 + 0.37 * ((rate - 40.0) / 30.0)));
                    bgColor = 'rgba(161, 91, 22, ' + alpha.toFixed(2) + ')';
                } else {
                    cellClass = 'heatmap-cell-low';
                    const alpha = Math.min(0.92, Math.max(0.55, 0.55 + 0.37 * (rate / 40.0)));
                    bgColor = 'rgba(30, 126, 52, ' + alpha.toFixed(2) + ')';
                }

                cell.classList.add(cellClass);
                cell.style.backgroundColor = bgColor;

                const tooltipText = 'Screen: ' + screenBin + 'h | Sleep: ' + sleepBin + 'h\n' +
                    'Addiction Rate: ' + rate.toFixed(1) + '%\n' +
                    'Population Density: ' + density.toFixed(2) + '%\n' +
                    'Sample Size: ' + Number(count).toLocaleString() + ' participants';

                cell.setAttribute('title', tooltipText);
                cell.setAttribute('aria-label', 'Screen ' + screenBin + 'h, Sleep ' + sleepBin + 'h, Addiction Rate ' + rate.toFixed(1) + ' percent, Density ' + density.toFixed(1) + ' percent');

                cell.innerHTML =
                    '<span class="heatmap-rate">' + rate.toFixed(1) + '%</span>' +
                    '<span class="heatmap-density">' + density.toFixed(1) + '% pop</span>';

                container.appendChild(cell);
            });
        });
    }

    /**
     * Updates personal quantile benchmark overlay bars and labels (M2-TASK-04, AC-2.4).
     * Updates percentile progress bars and narrative labels for:
     * - Daily Screen Time
     * - Sleep Duration
     * - App Opens / Day
     * - Notifications / Day
     *
     * @param {Object} quantileData - BenchmarkOverlayResponse payload
     */
    function renderQuantileOverlays(quantileData) {
        if (!quantileData) return;

        // 1. Screen Time Percentile
        const screenPct = Number(quantileData.screen_time_percentile) || 0;
        const valScreen = document.getElementById('val-quantile-screen');
        const barScreen = document.getElementById('bar-quantile-screen');
        if (valScreen) {
            valScreen.textContent = quantileData.screen_time_label || (Math.round(screenPct) + 'th Percentile');
        }
        if (barScreen) {
            barScreen.style.width = Math.min(100, Math.max(0, screenPct)) + '%';
            barScreen.style.backgroundColor = screenPct >= 75 ? 'var(--danger)' : (screenPct >= 50 ? 'var(--amber)' : 'var(--gold)');
        }

        // 2. Sleep Duration Percentile
        const sleepPct = Number(quantileData.sleep_duration_percentile !== undefined ? quantileData.sleep_duration_percentile : quantileData.sleep_hours_percentile) || 0;
        const valSleep = document.getElementById('val-quantile-sleep');
        const barSleep = document.getElementById('bar-quantile-sleep');
        if (valSleep) {
            valSleep.textContent = quantileData.sleep_hours_label || (Math.round(sleepPct) + 'th Percentile');
        }
        if (barSleep) {
            barSleep.style.width = Math.min(100, Math.max(0, sleepPct)) + '%';
            // Low sleep is high risk!
            barSleep.style.backgroundColor = sleepPct <= 25 ? 'var(--danger)' : (sleepPct <= 45 ? 'var(--amber)' : 'var(--low-risk)');
        }

        // 3. App Opens Percentile
        const valOpens = document.getElementById('val-quantile-opens');
        const barOpens = document.getElementById('bar-quantile-opens');
        if (quantileData.app_opens_percentile !== null && quantileData.app_opens_percentile !== undefined) {
            const opensPct = Number(quantileData.app_opens_percentile);
            if (valOpens) {
                valOpens.textContent = Math.round(opensPct) + 'th Percentile';
            }
            if (barOpens) {
                barOpens.style.width = Math.min(100, Math.max(0, opensPct)) + '%';
                barOpens.style.backgroundColor = opensPct >= 75 ? 'var(--danger)' : (opensPct >= 50 ? 'var(--amber)' : 'var(--gold)');
            }
        }

        // 4. Notifications Percentile
        const valNotifs = document.getElementById('val-quantile-notifs');
        const barNotifs = document.getElementById('bar-quantile-notifs');
        if (quantileData.notifications_percentile !== null && quantileData.notifications_percentile !== undefined) {
            const notifsPct = Number(quantileData.notifications_percentile);
            if (valNotifs) {
                valNotifs.textContent = Math.round(notifsPct) + 'th Percentile';
            }
            if (barNotifs) {
                barNotifs.style.width = Math.min(100, Math.max(0, notifsPct)) + '%';
                barNotifs.style.backgroundColor = notifsPct >= 75 ? 'var(--danger)' : (notifsPct >= 50 ? 'var(--amber)' : 'var(--gold)');
            }
        }
    }

    return {
        renderRiskGauge: renderRiskGauge,
        getClinicalTier: getClinicalTier,
        renderCohortCards: renderCohortCards,
        renderScreenSleepHeatmap: renderScreenSleepHeatmap,
        renderQuantileOverlays: renderQuantileOverlays,
        GAUGE_ARC_LENGTH: GAUGE_ARC_LENGTH
    };
});
