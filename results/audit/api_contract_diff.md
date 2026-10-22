# Frontend vs Backend API Contract Diff Report

**File Checked**: `../FrontendSkinDerma/milk10k-frontend/src/api/types.ts`  
*(Note: Direct relative path `../milk10k-frontend/src/api/types.ts` is absent; the actual active frontend workspace is located at `../FrontendSkinDerma/milk10k-frontend/src/api/types.ts`)*  
**Date**: 2026-10-22  
**API Model Version**: `exp2c-panderm-vit-base-v1`  

---

## 1. Request Contract Diff (`PredictRequest`)

| Field | Frontend Definition (`types.ts`) | Backend Requirement (`/api/predict`) | Status | Required Change in Frontend |
|---|---|---|---|---|
| `dermoscopic_image` | `File \| undefined` (Optional) | `UploadFile` (Required) | **BREAKING DIFF** | Make `dermoscopic_image: File` mandatory in form submission. |
| `clinical_image` | `File` (Required) | `UploadFile \| None` (Optional) | **BREAKING DIFF** | Make `clinical_image?: File` optional; backend ignores it for this arm. |
| `skin_tone` | `number \| undefined` (Optional) | `int \| None` (Optional) | **MATCH** | No change required. Echoed back in response. |

---

## 2. Response Contract Diff (`PredictResponse`)

| Field | Frontend Type (`types.ts`) | Backend Type (`PredictResponse`) | Status | Required Change in Frontend |
|---|---|---|---|---|
| `predicted_class` | `string` | `str` | **MATCH** | None. |
| `confidence` | `number` | `float` (in [0, 1]) | **MATCH** | None. |
| `malignant_probability` | `number` | `float` (in [0, 1]) | **MATCH** | None. Sum of probabilities for {AKIEC, BCC, MAL_OTH, MEL, SCCKA}. |
| `per_class_probs` | `Record<string, number>` | `Dict[str, float]` (11 classes) | **MATCH** | None. Probabilities sum to 1.0 within 1e-4. |
| `concept_scores` | `{ ulceration_crust: number; ... }` (Required Object) | `Optional[Dict[str, float]]` (`null`) | **BREAKING DIFF** | Change type to `concept_scores: { ... } \| null`. Add null-check guard in UI so UI displays "N/A (Concept bottleneck model pending Phase 5)" instead of crashing. |
| `explainability_overlay` | `string` | `Optional[str]` (`data:image/png;base64,... \| null`) | **DIFF** | Change type to `explainability_overlay: string \| null`. |
| `model_version` | `string` | `str` | **MATCH** | None. Returns `"exp2c-panderm-vit-base-v1"`. |
| `disclaimer` | *Not present in frontend type* | `str` | **NEW FIELD** | Add `disclaimer: string` to frontend interface and display prominently in prediction banner. |
| `inference_ms` | *Not present in frontend type* | `float` | **NEW FIELD** | Add `inference_ms?: number` to display server inference latency badge. |
| `input_note` | *Not present in frontend type* | `str` | **NEW FIELD** | Add `input_note?: string` to display tooltip/badge informing user that clinical image/skin tone are ignored. |
| `skin_tone` | *Not present in frontend type* | `Optional[int]` | **NEW FIELD** | Add `skin_tone?: number \| null`. |

---

## 3. Recommended Updated TypeScript Definition for Frontend

```typescript
// Proposed src/api/types.ts update for ../FrontendSkinDerma/milk10k-frontend/
export interface PredictRequest {
  dermoscopic_image: File; // REQUIRED: Primary diagnostic modality for Exp-2c PanDerm
  clinical_image?: File;   // OPTIONAL: Validated if provided, but ignored by single-modality ViT
  skin_tone?: number;      // OPTIONAL: Monk skin tone bucket (1..10), echoed back
}

export interface PredictResponse {
  predicted_class: string;
  confidence: number;
  malignant_probability: number;
  per_class_probs: Record<string, number>;
  concept_scores: {
    ulceration_crust: number;
    hair: number;
    vasculature: number;
    erythema: number;
    pigmented: number;
    gel_fluid: number;
    skin_markings: number;
  } | null; // Null in Phase 6-lite (honest presentation: never fabricated)
  explainability_overlay: string | null; // Base64 data URI (RGBA PNG)
  model_version: string;
  disclaimer: string;
  inference_ms: number;
  input_note: string;
  skin_tone?: number | null;
}
```
