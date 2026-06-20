# Episode Creation Update - Treatment-Line-Based Approach

## Date
2025-11-11

## Motivation
Previous imaging-based episode creation was too complex and created discrepancies:
- GPT-5: Good accuracy but creates better (more nuanced) episodes
- GPT-4.1: Fast but creates too many non-sensical episodes

The imaging-focus required extensive reasoning about imaging clusters, disease states, and context windows.

## New Simplified Approach

Episodes now follow **natural clinical phases** based on treatment lines:

### Episode Types
1. **Baseline** - Background information predating oncological disease
   - Pre-existing conditions, smoking history, comorbidities
   - Includes events from ANY date if they describe baseline state
   - Anchor: one day before first oncological event

2. **Diagnosis** - Initial diagnostic workup
   - Symptoms → specialist visit → all diagnostics (imaging, biopsies, pathology)
   - Ends when treatment decision is made
   - Anchor: specialist visit OR first diagnostic procedure

3. **Treatment Lines** - Each line of therapy
   - Includes systemic therapy, surgery, radiation as planned
   - Contains ALL events during line: imaging, labs, symptoms, adverse effects
   - **Critical**: Event that ENDS the line stays IN that line (not next episode)
   - Ends when: progression, toxicity, completion, no effect

4. **Post-Oncological** - End of active treatment
   - Hospice, comfort care only, DNR/DNI changes

### Key Simplifications
- Imaging is **context within treatment lines**, not episode boundaries
- Treatment lines are natural decision points
- Less reasoning required about imaging clusters and disease states
- Clearer episode boundaries based on treatment strategy changes

## Expected Benefits
- **Faster**: Less complex reasoning required
- **More consistent**: Clear rules for episode boundaries
- **Clinically intuitive**: Matches how oncologists think about patient journey
- **Better cross-model consistency**: Simpler logic should reduce GPT-4.1 vs GPT-5 discrepancies

## Next Steps
- [ ] Create test cohort (5-10 patients)
- [ ] Run new episode creation on test cohort
- [ ] Compare speed and accuracy vs previous imaging-based approach
- [ ] Validate episode quality and clinical coherence
