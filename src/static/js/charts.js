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
     * Authoritative 3-tier clinical status mapping per ADR-0006
     * @param {number} prob - Evaluated addiction probability [0.0, 1.0]
     * @param {number} threshold - Decision threshold tau [0.05, 0.95]
     * @returns {Object} Tier metadata with label, classes, CSS color tokens, and advisory HTML
     */
    function getClinicalTier(prob, threshold) {
        const tau = Number(threshold) || 0.50;

        if (prob >= tau) {
            return {
                label: 'Elevated Risk Tier',
                badgeClass: 'badge-high',
                pillClass: 'status-pill-high',
                colorVar: 'var(--danger)',
                summaryHtml: 'Estimated risk probability exceeds clinical decision threshold (&tau; = ' +
                    tau.toFixed(2) + '). Behavioral boundary restructuring recommended.'
            };
        } else if (prob >= 0.35) {
            return {
                label: 'Compensatory Usage Pattern',
                badgeClass: 'badge-mod',
                pillClass: 'status-pill-mod',
                colorVar: 'var(--amber)',
                summaryHtml: 'Estimated risk probability reflects moderate recreational usage below threshold (&tau; = ' +
                    tau.toFixed(2) + '). Targeted routine adjustments advised.'
            };
        } else {
            return {
                label: 'Balanced Habit Profile',
                badgeClass: 'badge-low',
                pillClass: 'status-pill-low',
                colorVar: 'var(--low-risk)',
                summaryHtml: 'Estimated risk probability demonstrates balanced behavioral equilibrium below threshold (&tau; = ' +
                    tau.toFixed(2) + '). Healthy digital routine maintained.'
            };
        }
    }

    /**
     * Renders the SVG circular arc risk gauge and synchronizes authoritative status displays.
     * Conforms strictly to ADR-0006 and ADR-0008 geometry and clearance contracts.
     *
     * @param {number} probability - Assessed addiction probability in range [0.0, 1.0]
     * @param {number} [threshold=0.50] - Active classification decision threshold tau
     * @returns {Object} Render evaluation metadata
     */
    function renderRiskGauge(probability, threshold) {
        const prob = Math.min(1.0, Math.max(0.0, Number(probability) || 0.0));
        const tau = Math.min(0.95, Math.max(0.05, Number(threshold) || 0.50));
        const pctFormatted = (prob * 100).toFixed(1) + '%';

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

        // 3. 3-Tier Clinical Spectrum Determination
        const tier = getClinicalTier(prob, tau);

        if (arcEl) {
            arcEl.style.stroke = tier.colorVar;
        }

        // 4. Consolidated Authoritative Status Pill (ADR-0006)
        const statusPill = document.getElementById('status-pill');
        if (statusPill) {
            // Apply single authoritative status pill classes
            statusPill.className = 'status-pill ' + tier.pillClass;
        }

        const badgeStatus = document.getElementById('badge-status');
        if (badgeStatus) {
            badgeStatus.className = 'badge ' + tier.badgeClass;
            badgeStatus.textContent = tier.label;
        }

        // 5. Narrative Classification Summary
        const summaryEl = document.getElementById('classification-summary');
        if (summaryEl) {
            summaryEl.innerHTML = 'Estimated risk probability <strong>' + pctFormatted + '</strong>: ' + tier.summaryHtml;
        }

        return {
            probability: prob,
            threshold: tau,
            percentage: pctFormatted,
            label: tier.label,
            badgeClass: tier.badgeClass,
            pillClass: tier.pillClass,
            colorVar: tier.colorVar
        };
    }

    return {
        renderRiskGauge: renderRiskGauge,
        getClinicalTier: getClinicalTier,
        GAUGE_ARC_LENGTH: GAUGE_ARC_LENGTH
    };
});
