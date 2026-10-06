# Design System & UI Specification

Design tokens, typography, component geometry, and interaction standards for the Smartphone Addiction Analytical Platform, directly inspired by high-craft editorial analytical tools.

---

## 1. Design Tokens & Color Palette

```css
:root {
    /* Canvas & Surfaces */
    --bg: #F6F3EA;             /* Warm editorial parchment background */
    --panel: #FFFDF7;          /* Card surface container */
    --panel-subtle: #F9F7F0;   /* Secondary recessed surface */
    --field-bg: #F8F6EF;       /* Form input background */
    
    /* Borders & Dividers */
    --line: rgba(150, 113, 45, 0.25);      /* Warm bronze hairline border */
    --line-soft: rgba(22, 32, 47, 0.08);   /* Subtle slate divider */
    --track: #DAD4C2;                      /* Slider track fill */
    
    /* Brand Accents */
    --gold: #9C6F28;           /* Primary brand accent / highlight */
    --gold-hover: #7E5920;     /* Interactive button hover */
    --ink-on-gold: #15202F;    /* Contrast text on gold buttons */
    
    /* Typography Inks */
    --text-1: #16202F;         /* High-contrast slate heading ink */
    --text-2: #55617A;         /* Secondary descriptive text */
    --text-3: #7A8296;         /* Subtle caption & metadata ink */
    
    /* Status & Severity */
    --low-risk: #1E7E34;       /* Healthy green status */
    --low-risk-bg: #EAF7EE;    /* Soft green pill background */
    --amber: #A15B16;          /* Moderate/at-risk amber status */
    --amber-bg: #FDF4E7;       /* Soft amber pill background */
    --danger: #9C3F2C;         /* High-risk crimson status */
    --danger-bg: #FDF0ED;      /* Soft danger pill background */
}
```

---

## 2. Typography

- **Headers & Display**: `'Source Serif 4', Georgia, serif`
  - Page Title: `1.85rem`, weight 600, letter-spacing `-0.02em`
  - Section Headers: `1.25rem`, weight 600
  - Card Titles: `1.05rem`, weight 600
- **Body, Controls & Data**: `'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif`
  - Body Text: `0.925rem`, weight 400, line-height 1.5
  - Metric Numbers: `1.65rem`, weight 700, tabular-nums
  - Form Labels & Slider Values: `0.85rem`, weight 500
  - Badges & Telemetry: `0.75rem`, weight 600, uppercase, letter-spacing `0.05em`

---

## 3. Layout & Component Geometry

### Viewports & Responsiveness
- **Desktop Grid (1440px / 1200px)**:
  - Header: Sticky top bar with brand mark, live status indicator, and tab navigation.
  - Two-Column Workspace:
    - Left Column (`55%` width): Profile controls (Demographics, Screen Time Allocation, Physiological Signals, Decision Threshold).
    - Right Column (`45%` width): Live Risk Gauge, Status Banner, Granular Metric Tiles, and Intervention Actions.
- **Mobile Viewport (390px)**:
  - Single column fluid stack.
  - Live Risk summary pinned or immediately visible above inputs.
  - Large touch targets (`min-height: 44px`) for sliders and segmented pills.

### Component Styling
- **Card Panels (`.panel`)**:
  - `background: var(--panel)`
  - `border: 1px solid var(--line)`
  - `border-radius: 8px`
  - `padding: 24px`
  - `box-shadow: 0 1px 3px rgba(22, 32, 47, 0.04)`
- **Metric Cards (`.metric-card`)**:
  - `background: var(--panel-subtle)`
  - `border: 1px solid var(--line-soft)`
  - `border-radius: 6px`
  - `padding: 16px`
- **Pills & Status Badges (`.badge`)**:
  - `border-radius: 999px`
  - `padding: 4px 10px`
  - Font size `0.75rem`

---

## 4. Interaction & Motion Standards
- Slider dragging triggers an immediate local value readout and debounced API request (`35ms`).
- Gauge needle or progress arc animates smoothly with CSS `transition: stroke-dashoffset 0.25s cubic-bezier(0.4, 0, 0.2, 1)`.
- Zero page reloads or layout shift on interaction.

---

## 5. Prototype Views & Accessibility Invariants

### Interactive Views
1. **Individual Diagnostic View (`#tab-diagnostic`)**:
   - Primary 55/45 two-column layout.
   - Demographics, Digital Time Budget, and Habit Intensity slider controls with immediate value readouts.
   - Physiological boundary banner alerting if daily accounted hours exceed 24.0h.
   - Reactive SVG circular arc risk gauge with dynamic severity color mapping (`--low-risk`, `--amber`, `--danger`).
   - Granular diagnostic ratio tiles (`Screen / Sleep`, `Recreational Share`, `Avg Session Length`, `Weekend Surge`) with target breach indicators.
   - Actionable clinical intervention advisories (Sleep Protection, Recreation Audit, Notification Hygiene, Weekend Disconnect).
   - Multi-model 5-fold CV benchmark performance table.
2. **Population Cohort Analytics Explorer (`#tab-cohorts`)**:
   - Demographic slicing by Age Bracket, Gender, Stress Level, and Academic Impact across 691,369 Kaggle records.
   - Dynamic cohort summary cards with addiction prevalence % and behavioral averages.
   - 2D Joint Density Heatmap showing Screen Time vs Sleep Duration addiction risk concentrations.
   - Personal Quantile Benchmark Overlay mapping individual habits against population percentiles.
3. **What-If Scenario Simulator (`#tab-whatif`)**:
   - Interactive habit counterfactual levers (Recreation Reduction, Sleep Extension, Open Batching).
   - Real-time probability delta badge and status transition indicator (`ADDICTION DETECTED` ➔ `HEALTHY`).
   - Minimal intervention solver button calculating optimal lifestyle adjustment pathway.
4. **Batch Population Diagnostics (`#tab-batch`)**:
   - Drag & drop CSV dropzone with 10,000-row validation and one-click demo cohort loader.
   - Aggregate cohort summary strip and searchable diagnostic preview table.
   - Full diagnostic CSV export with computed probabilities, classifications, and clinical advisories.

### Accessibility Standards
- **Keyboard Reachability**: All controls (sliders, tab navigation, segmented pills, buttons, dropzone) have visible `:focus-visible` styling (`outline: 2px solid var(--gold); outline-offset: 2px;`).
- **Semantic Labels**: Every slider has an explicit `<label for="...">`, segmented buttons use `role="radiogroup"` and `aria-checked`, tabs use `role="tablist"` and `aria-selected`.
- **Motion Reduction**: `@media (prefers-reduced-motion: reduce)` resets CSS transitions and gauge animations.
- **Viewport Fluidity**: Clean responsive stack tested down to 390px mobile viewport with zero horizontal overflow and $\ge 44\text{px}$ touch target heights.


