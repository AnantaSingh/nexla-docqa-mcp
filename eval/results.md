# Evaluation Results

Total gold questions: **55**. Judge: Claude (LLM-as-judge), temperature 0.

## Accuracy by question type (scoped to source document)

| Type | N | Correct | Partial | Incorrect | Accuracy (C) | C+P |
|---|---|---|---|---|---|---|
| text-only | 10 | 7 | 1 | 2 | 70% | 80% |
| multimodal-t | 21 | 16 | 1 | 4 | 76% | 81% |
| multimodal-f | 12 | 9 | 1 | 2 | 75% | 83% |
| meta-data | 7 | 4 | 2 | 1 | 57% | 86% |
| **answerable total** | 50 | 36 | 5 | 9 | **72%** | **82%** |

## Abstention on unanswerable questions (anti-hallucination)

- Correctly abstained (scoped): **5/5** (100%)
- Correctly abstained (unscoped, all docs): **5/5** (100%)

## Multi-document routing (unscoped: correct report cited among all 5)

- Correct source document cited: **44/50** (88%)

## Per-question detail

| Type | Company | Verdict | Routed | Question |
|---|---|---|---|---|
| text-only | Costco | PARTIAL | ✓ | What is the future minimum payment for finance leases in 2025? |
| text-only | Costco | CORRECT | ✓ | Did the company declare cash dividends in 2022? |
| multimodal-t | Costco | INCORRECT | ✓ | What was the total revenue for the Company in the United States in 2022? |
| multimodal-t | Costco | CORRECT | ✓ | How many employees did the company have in Canada in 2022? |
| multimodal-f | Costco | CORRECT | ✓ | Who has the longest tenure as an executive officer at Costco? |
| multimodal-t | Costco | CORRECT | ✓ | What was the gross margin percentage for the year 2020? |
| multimodal-t | Costco | CORRECT | ✓ | What was the total average sales per warehouse for the company in the fiscal year 2013? |
| meta-data | Costco | PARTIAL | ✓ | On which page does the document report the information about executive officers? |
| unanswerable | Costco | ABSTAINED | — | What was the income revenues for the company in Macau? |
| text-only | Accenture | CORRECT | ✓ | What initiative was launched to assist clients in sustainable cloud migration? |
| text-only | Accenture | CORRECT | ✓ | Has the company set a goal to achieve net-zero emissions by 2025? |
| multimodal-t | Accenture | INCORRECT | ✗ | By how much did the net assets of the Europe market increase from fiscal 2018 to fiscal 2020? |
| multimodal-t | Accenture | INCORRECT | ✓ | What are the total expected benefit payments for U.S. Pension Plans for the year 2023? |
| multimodal-t | Accenture | PARTIAL | ✓ | What was the percentage increase in total revenues of the company from fiscal 2019 to fiscal 2020? |
| multimodal-t | Accenture | CORRECT | ✓ | What was the company's operating income for Europe in 2020? |
| multimodal-t | Accenture | CORRECT | ✓ | What was the company's operating income for the fiscal year 2018? |
| multimodal-f | Accenture | CORRECT | ✓ | Which geographic market of the company contributed the most to the fiscal 2020 revenue for the company? |
| meta-data | Accenture | CORRECT | ✗ | How many pages are there in the document? |
| meta-data | Accenture | PARTIAL | ✗ | What is the primary message conveyed on page 11 in the document? |
| unanswerable | Accenture | ABSTAINED | — | Did IBM's total revenue increase from 2019 to 2020? |
| text-only | McDonald's | CORRECT | ✓ | How did the occupancy and other operating expenses for company-operated restaurants change over the three years? |
| text-only | McDonald's | CORRECT | ✓ | What was the total amount of McDonald's Corporation's unrecognized tax benefits at the end of 2020? |
| multimodal-t | McDonald's | CORRECT | ✓ | How much did the International Operated Markets contribute to the total franchised revenues in 2019? |
| multimodal-f | McDonald's | INCORRECT | ✓ | How many company-operated restaurants were there in the U.S. at the end of 2018? |
| multimodal-t | McDonald's | CORRECT | ✗ | What was the total revenue for the company in 2019? |
| meta-data | McDonald's | CORRECT | ✗ | How many pages in total does this report have? |
| meta-data | McDonald's | CORRECT | ✓ | How many times does the report mention "franchised margins"? |
| unanswerable | McDonald's | ABSTAINED | — | How many stores does the company open in Shanghai? |
| text-only | Philip | INCORRECT | ✓ | On which page does the report provide Mine Safety Disclosures? |
| text-only | Philip | CORRECT | ✓ | Did PMI have higher net revenues in the European Union in 2020 compared to 2018? |
| multimodal-t | Philip | CORRECT | ✓ | How much of the change in PMI's Net Revenues was attributable to currency effects? |
| multimodal-t | Philip | CORRECT | ✓ | What was PMI's net revenue from combustible products in the European Union for 2020? |
| multimodal-t | Philip | CORRECT | ✓ | What was the shipment volume of Marlboro cigarettes in 2020 according to PMI's report? |
| multimodal-t | Philip | CORRECT | ✓ | Which brand in PMI experienced the highest percentage increase in shipment volume from 2019 to 2020? |
| multimodal-t | Philip | CORRECT | ✓ | What was PMI's net revenue for the European Union in 2020? |
| multimodal-t | Philip | CORRECT | ✓ | What was the total provision for PMI's income taxes in 2020? |
| multimodal-f | Philip | CORRECT | ✓ | What product category generated the highest net revenue for the company in 2020? |
| multimodal-f | Philip | CORRECT | ✓ | How did PMI's cumulative total shareholder return compare to the S&P 500 Index at the end of the 2020? |
| meta-data | Philip | INCORRECT | ✗ | What is the most common abbreviation in the document? |
| unanswerable | Philip | ABSTAINED | — | On which page does are the photos of the company's board members displayed? |
| text-only | Toyota | INCORRECT | ✓ | What was the year-on-year percentage change in Toyota's operating income for fiscal 2021? |
| text-only | Toyota | CORRECT | ✓ | What is the energy efficiency target for Toyota's compact SUV in the bZ series? |
| multimodal-t | Toyota | CORRECT | ✓ | What is the percentage of women hired globally by Toyota Motor Corporation? |
| multimodal-t | Toyota | INCORRECT | ✓ | How many members listed have a board service length of 3 years, and what are their names? |
| multimodal-t | Toyota | CORRECT | ✓ | What is the percentage of shares held by the largest shareholder listed in the report? |
| multimodal-t | Toyota | CORRECT | ✓ | What is the total number of common shares held by the top three shareholders combined for Toyota Motor Corporation? |
| multimodal-f | Toyota | CORRECT | ✓ | What is the primary goal of the Fleet Management System of Toyota? |
| multimodal-f | Toyota | PARTIAL | ✓ | How does the Fleet Management System of Toyota respond when there is an increase in waiting customers? |
| multimodal-f | Toyota | INCORRECT | ✓ | What is the difference in the rate of the engine being off during driving between HEVs and PHEVs? |
| multimodal-f | Toyota | CORRECT | ✓ | What is the largest category of shareholders shown in the company's ownership breakdown? |
| multimodal-f | Toyota | CORRECT | ✓ | What is the role of the Board of Directors in Toyota's Corporate Governance structure? |
| multimodal-f | Toyota | CORRECT | ✓ | What is the target percentage for the reduction in cost of a single battery? |
| multimodal-f | Toyota | CORRECT | ✓ | How does the evolution of battery control models contribute to the development of Toyota's next-generation BEVs? |
| meta-data | Toyota | CORRECT | ✓ | On which page does the document detail the Toyota Production System (TPS)? |
| unanswerable | Toyota | ABSTAINED | — | What percentage of the executive committee's nationality is American? |
