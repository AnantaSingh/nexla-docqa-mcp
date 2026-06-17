# Example Interaction Log

Verbatim outputs from the `query_documents` MCP tool against the indexed corpus (Toyota,
Costco, McDonald's, Accenture, Philip Morris). Each example shows the answer plus the
**source attribution** (company, fiscal year, page) returned in `citations`.

---

### 1. Single-document factual lookup (table) — with source attribution

**Q:** "What was Costco's total revenue in fiscal 2022?"

**Answer** (`answer_found: true`):
> Costco's total revenue in fiscal 2022 was **$226,954 million**.

**Citation:** Costco Wholesale Corporation (FY2022), **p.40**, `table`
> *"…CONSOLIDATED STATEMENTS OF INCOME … 2022 2021 2020 REVENUE Net sales $ 222,730 $ 192,052 $ 163,220 … Total revenue 226,954 195,929 166,761"*

This is the exact figure from the consolidated income statement — the layout-aware parser keeps
the `Total revenue` row intact with its three fiscal-year values, and the page is cited precisely.

---

### 2. Multi-document comparison — context pulled from two reports

**Q:** "Compare the total revenue of Costco and McDonald's in their latest reported fiscal years."

**Answer** (`answer_found: true`):
> In their latest reported fiscal years, **Costco** (FY2022) reported total revenue of
> **$226,954 million**, while **McDonald's** (FY2020) reported total revenues of
> **$19,208 million**. Costco's total revenue was significantly higher than McDonald's by
> approximately $207,746 million.

**Citations:**
- Costco Wholesale Corporation (FY2022), **p.40**, `table`
- McDonald's Corporation (FY2020), **p.18**, `table`

The system retrieved and attributed figures from **two different documents** in a single answer,
and correctly tagged each with its company and fiscal year.

---

### 3. Document-scoped question (`document="Accenture"`)

**Q:** "Has Accenture set a goal to achieve net-zero emissions by 2025?"

**Answer** (`answer_found: true`):
> Yes, Accenture has set a goal to achieve net-zero emissions by 2025. According to the FY2020
> Form 10-K, Accenture announced three additional, industry-leading environmental goals: to
> achieve net-zero emissions, move to zero waste and plan for water risk by 2025.

**Citation:** Accenture plc (FY2020), **p.7**, `text`
> *"…we are building on our earlier goals for emissions reduction and recently announced three
> additional, industry-leading goals: to achieve net-zero emissions, move to zero waste…"*

---

### 4. Unanswerable question — correct abstention (no hallucination)

**Q:** "How many stores does the company open in Shanghai?" *(scoped to McDonald's)*

**Answer** (`answer_found: false`):
> The provided sources do not contain information about how many stores McDonald's opens in Shanghai.

**Citations:** *(none)*

The report contains no such figure, so the system **declines to answer** rather than inventing one.
This behavior is verified across all 5 `unanswerable` gold questions (100% correct abstention; see
[`eval/results.md`](../eval/results.md)).
