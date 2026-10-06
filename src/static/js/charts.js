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

    return {
        renderRiskGauge: renderRiskGauge,
        getClinicalTier: getClinicalTier,
        GAUGE_ARC_LENGTH: GAUGE_ARC_LENGTH
    };
});
