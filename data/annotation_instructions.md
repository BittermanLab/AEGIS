# Annotation Guidelines for irAEs Labeling

These instructions are designed to help you annotate each document with labels that provide information about the presence of immune-related Adverse Events (irAEs) and their descriptive attributes. **Do not use information from other notes on the same patient.**

---

## 1. Overview

- **Objective:**  
  Label each document with the presence of irAEs and associated attributes based solely on the current note.

- **Key Considerations:**  
  - The language used to describe irAEs may vary; use your best judgment.  
  - Do **not** incorporate details from other patient notes.  
  - Not every sign or symptom overlapping with an irAE should be labeled—only label events where the irAE is in the differential diagnosis.

---

## 2. Priority irAEs

For this project, you will only label the following **6 priority irAE disorders**:

- **Myocarditis**
- **Dermatitis**
- **Thyroiditis**
- **Hepatitis**
- **Colitis**
- **Pneumonitis**

> **Note:** These irAEs are umbrella terms for multiple CTCAE terms. For example, "immune-related cough" should be labeled as pneumonitis. (See Appendix I: CTCAE term mappings for details.)

Additionally, all notes were selected because they were written within 12 months of an immunotherapy infusion—this timeline can help guide your determination.

---

## 3. Annotation Objectives

- **Focus on Medical Events:**  
  Annotate events where the irAE is included in the differential diagnosis. This should represent clusters of signs and symptoms that, taken together, suggest the disorder.

- **Exclude Isolated Symptoms:**  
  Do **not** label isolated signs/symptoms if they are clearly explained by another diagnosis. For example:  
  - **Pneumonitis:** Do not label if shortness of breath is clearly due to COPD or infectious pneumonia.  
  - **Myocarditis:** Do not label if chest pain is clearly due to coronary artery disease.

- **Labeling Ambiguous Presentations:**  
  If the irAE is in the differential—even if deemed unlikely or possible—label it unless a clear alternative diagnosis is stated in the note.

---

## 4. Labeling Schema

Each document should be annotated using two sets of labels: **Current** and **Past**. Each label is appended by an abbreviation for the irAE type (e.g., `Pneum` for pneumonitis, `Col` for colitis, etc.).

### A. Current Labels

1. **CurrentType\_***  
   - **Description:** Indicates if the patient is currently experiencing the irAE.  
   - **ValueSet:**  
     - `1` = yes  
     - **Blank** = no

2. **CurrentGrade\_***  
   - **Description:** If `CurrentType` is yes, this indicates the current CTCAE grade of the irAE.  
   - **ValueSet:**  
     - `1` = grade 1  
     - `2` = grade 2  
     - `3` = grade 3  
     - `4` = grade 4  
     - `5` = grade 5

3. **CurrentAttr\_***  
   - **Description:** Indicates the attribution of the current irAE.  
   - **ValueSet:**  
     - `1` = IO  
     - **Blank** = not IO

4. **CurrentCert\_***  
   - **Description:** If `CurrentType` (and `CurrentAttr`) is yes, this indicates the explicit certainty that the event is due to immunotherapy.  
   - **ValueSet:**  
     - `1` = unlikely  
     - `2` = Possible  
     - `3` = Likely  
     - `4` = Certain  
     - **Blank** = no explicit certainty described

### B. Past Labels

1. **PastType\_***  
   - **Description:** Indicates if the patient had the irAE in the past.  
   - **ValueSet:**  
     - `1` = yes  
     - **Blank** = no

2. **PastMaxGrade\_***  
   - **Description:** If `PastType` is yes, this indicates the maximum CTCAE grade experienced in the past (excluding the current grade).  
   - **ValueSet:**  
     - `1` = grade 1  
     - `2` = grade 2  
     - `3` = grade 3  
     - `4` = grade 4  
     - `5` = grade 5

3. **PastAttr\_***  
   - **Description:** Indicates the attribution for the past irAE.  
   - **ValueSet:**  
     - `1` = IO  
     - **Blank** = not IO

4. **PastCert\_***  
   - **Description:** If `PastType` (and `PastAttr`) is yes, this indicates the explicit certainty that the past event was due to immunotherapy.  
   - **ValueSet:**  
     - `1` = unlikely  
     - `2` = Possible  
     - `3` = Likely  
     - `4` = Certain  
     - **Blank** = no explicit certainty described

---

## 5. Temporal Considerations

When annotating, consider the timeline of the irAE presentation:

### A. Past, Now Resolved irAEs

- **Definition:**  
  An irAE that started in the past and is now completely resolved.
  
- **Annotation:**  
  Use **only** the **Past*** labels.

- **Example:**  
  A note stating “stage I lung adenocarcinoma (previously on pembro but on hold 2/2 pneumonitis)” should be annotated with Past labels only.

### B. Ongoing irAEs

- **Definition:**  
  An irAE that started in the past and is still ongoing. It may have changed in severity over time, resolved and recurred, or become chronic.
  
- **Annotation:**  
  Use **both** **Current*** and **Past*** labels.
  
- **Note:**  
  Even if the patient is on immune suppressants and asymptomatic, the irAE is considered present (Current).

### C. New irAE Presentations

- **Definition:**  
  The patient is presenting with the irAE for the first time.
  
- **Annotation:**  
  Use **only** the **Current*** labels.

### Additional Timeline Instructions

- **Current Labels:**  
  Reflect what is occurring at the time the note is written. For example, if a patient had a past grade 3 irAE that has improved to grade 2, annotate `CurrentGrade = 2`.

- **Past Labels:**  
  Reflect the historical maximum grade or attributes (excluding the current grade). For example, if the maximum past grade was 2 and the current grade is 3, annotate `PastMaxGrade = 2`.

- **Ambiguous Language:**  
  Phrases like “s/p immunotherapy, course complicated by pneumonitis” should be labeled as a **Past** event unless other parts of the note clearly indicate that the irAE is ongoing.

---

## 6. Handling Multiple Episodes

- **Multiple Episodes:**  
  If a patient experienced multiple episodes of the same irAE with different levels of certainty or attribution:
  - Use your best judgment.
  - If a clear attribution and certainty are provided for a past event and the current event is described as a recurrence with less detail, favor the detailed prior event’s certainty and attribution.

---

## 7. Annotation Examples

### **Example 1**

**Text:**  
> "The patient previously had immunotherapy-related colitis treated with steroids. The patient now presents with recurrence of the colitis."

**Annotations for Colitis (`Col`):**

- **Current Labels:**
  - `CurrentType_col = yes`
  - `CurrentGrade_col = 1` *(max inferable grade based on the text)*
  - `CurrentAttr_col = yes`
  - `CurrentCert_col = yes` *(clearly described as a recurrence)*

- **Past Labels:**
  - `PastType_col = yes`
  - `PastMaxGrade_col = 2`
  - `PastAttr_col = yes`
  - `PastCert_col = yes`

---

### **Example 2**

**Text:**  
> "The patient previously had immunotherapy-related colitis treated with steroids. The patient now presents with possible recurrence of colitis.”

**Annotations for Colitis (`Col`):**

- **Current Labels:**
  - `CurrentType_col = yes`
  - `CurrentGrade_col = 1` *(max inferable grade)*
  - `CurrentAttr_col = yes`
  - `CurrentCert_col = [blank]` *(no explicit certainty for the current event)*

- **Past Labels:**
  - `PastType_col = yes`
  - `PastMaxGrade_col = 2`
  - `PastAttr_col = yes`
  - `PastCert_col = yes`

---

### **Example 3**

**Text:**  
> “Previously admitted for pembro-related pneumonitis… now is admitted for dyspnea due to pneumonia vs inflammatory process such as pneumonitis (2/2 ?too quick discontinuation of steroid taper for prior pneumonitis.)”

**Annotations for Pneumonitis (`Pneum`):**

- **Current Labels:**
  - `CurrentType_Pneum = 1`
  - `CurrentGrade_Pneum = 3` *(hospital admission implies a grade 3 event)*
  - `CurrentAttr_Pneum = 1`
  - `CurrentCert_Pneum = 2` *(labeled as possible due to “vs” language)*

- **Past Labels:**
  - `PastType_Pneum = 1`
  - `PastMaxGrade_Pneum = 3`
  - `PastAttr_Pneum = 1`
  - `PastCert_Pneum = 4`

---

## 8. Final Reminders

- **Use Best Judgment:**  
  Natural language descriptions may vary; annotate based on the overall context provided in the note.

- **Do Not Cross-Reference:**  
  Only use information from the current note.

- **Ambiguous Cases:**  
  If unsure whether to label as Current or Past, lean toward Past unless evidence clearly indicates an ongoing event.

- **Documentation:**  
  Refer back to these guidelines whenever you have questions during the annotation process.
