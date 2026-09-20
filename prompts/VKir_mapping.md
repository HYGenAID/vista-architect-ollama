# VISTA Architect prompts: oncology -> vascular surgery (vkir)

File names are unchanged so the loader does not break. What follows is what changed
*inside* the prompts and therefore needs a matching change in the pipeline code.

## Placeholder renames

| Old | New | Files |
|---|---|---|
| `{TUMOR_TYPE}` | `{VASCULAR_SUBTYPE}` | patient_info, patient_info_and_note, summary, timeline, chat_system |
| `{ct_date_vector}` | unchanged variable name, now documented as `imaging_date_vector` (CTA/MRA/DSA/duplex) | patient_info_and_note |

All other placeholders (`{xml_chunk}`, `{timeline_context}`, `{episodes_context}`,
`{deterministic_context}`, `{source_text}`, `{current_timeline}`, `{current_summary}`,
`{is_final_chunk}`, `{index_patient_*}`, `{candidates_*}`, `{n_candidates}`) are unchanged.

Note: the original `timeline.txt` as pasted contained backslash-escaped placeholders
(`{TUMOR\_TYPE}`, `{current\_timeline}`, `{xml\_chunk}`). Those look like markdown
escaping artefacts and would raise on `str.format`. They are written clean here.

## JSON section key renames

| Old | New |
|---|---|
| `TUMOR INFORMATION` | `LIMB AND VASCULAR STATUS` |
| `cancer_history` (summary.txt) | `vascular_history` |

Anything reading `patient_info["TUMOR INFORMATION"]` or `summary["cancer_history"]`
must be updated.

## Field renames within the schema

| Old field | New field |
|---|---|
| `ecog_performance_status` | `functional_status` (+ new `ambulatory_status`) |
| `therapy_toxicities` | `procedural_complications` |
| `tnm_staging` | `wifi_classification` (+ new `clinical_stage` for Rutherford/Fontaine) |
| `histology` | `anatomical_pattern` (segments + GLASS stage + runoff) |
| `metastasis_status` | `tissue_loss_status` |
| `lymph_node_involvement` | `foot_infection_status` |
| `driver_mutations` | `haemodynamics` (ABI, ankle/toe pressure, TBI, TcPO2, waveform) |
| - | `risk_markers` (HbA1c, LDL, eGFR, CRP, Hb, albumin, thrombophilia) |
| `radiation_therapy` | `amputation_history` |
| `surgical_candidate` | `revascularisation_candidate` (+ `strategy` sub-key) |
| `date_of_last_ct` | `date_of_last_vascular_imaging` |
| - | `index_limb`, `secondary_prevention`, `non_vascular_surgery`, `wound_images` |

Timeline events gained a required `laterality` field (`left`/`right`/`bilateral`/`unknown`)
and a new event type `wound`. Dedup keys are now
`(date, type, laterality)` and, for imaging, `(date, modality, site, laterality, finding)`.

## Coverage of the clinician-requested variable set

| Requested (FI) | Where it lands |
|---|---|
| Syntymäaika, sukupuoli, tupakointi | `PATIENT DEMOGRAPHICS.date_of_birth / sex / smoking_history` |
| Sairaudet ICD-10, verisuoni- ja syöpädiagnoosit lihavoituina | `previous_conditions` with `**bold**` markers |
| LDL, kolesterolilääkitys, hyytymishäiriöt, verenohennus | `secondary_prevention` + `risk_markers` |
| Ei-verisuonikirurgiset leikkaukset | `TREATMENTS.non_vascular_surgery` |
| Verisuonikirurgiset toimenpiteet | `TREATMENTS.previous` + `amputation_history` |
| Aiemmat leikkauskomplikaatiot | `procedural_complications` |
| Toimintakyky (ECOG-vastaava) | `functional_status` + `ambulatory_status` |
| Verisuonikirurginen oirehistoria | `clinical_stage` + timeline `symptom` events |
| Haavakuvat | `wound_images` + timeline `wound` events |
| ABI, varvaspaine | `haemodynamics` + timeline `examination` roll-up |
| Kuvantamiset, WIfI | `anatomical_pattern`, `wifi_classification`, timeline `imaging` events |

## Guideline anchors used

- Conte MS, Bradbury AW, Kolh P, et al. Global Vascular Guidelines on the Management of
  Chronic Limb-Threatening Ischemia. J Vasc Surg. 2019;69(6S):3S-125S.
- Nordanstig J, Behrendt CA, Baumgartner I, et al. ESVS 2024 Clinical Practice Guidelines
  on the Management of Asymptomatic Lower Limb Peripheral Arterial Disease and Intermittent
  Claudication. Eur J Vasc Endovasc Surg. 2024;67(1):9-96.
- Mills JL, Conte MS, Armstrong DG, et al. The Society for Vascular Surgery Lower Extremity
  Threatened Limb Classification System: Risk Stratification Based on Wound, Ischemia, and
  foot Infection (WIfI). J Vasc Surg. 2014;59(1):220-234.e2.
  https://doi.org/10.1016/j.jvs.2013.08.003
