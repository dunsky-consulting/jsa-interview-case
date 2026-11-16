# Prep kit: reference sheet

## The library (30 public documents)
Utility applications, regulator decisions and orders, strategic plans, guidelines and research reports.
One document is in French. Page numbers in citations are physical PDF pages.

| Title | Type | Jurisdiction | Date |
|---|---|---|---|
| ATCO Electric 2023 Cost-of-Service Review Application | application | Alberta | 2021-11-15 |
| Distribution Point-of-Delivery Interconnection Process Guideline - Standards of Service (AESO-AUC-2019JUL18-002 Attachment 1) | guideline | Alberta | 2015-05-04 |
| Distribution Deficiency Report (DDR) Guideline | guideline | Alberta | 2021-03-31 |
| Distribution System Inquiry Final Report | research | Alberta | 2021-02-19 |
| Decision 26615-D01-2022 - ATCO Electric Ltd. and FortisAlberta Inc. 2023 Cost-of-Service Review | decision | Alberta | 2022-07-28 |
| Decision 26617-D02-2022 - ENMAX and EPCOR 2023 Cost-of-Service Review - Reasons for Approval of Negotiated Settlements | decision | Alberta | 2022-07-28 |
| ENMAX Power Corporation 2023 Distribution Cost of Service Application | application | Alberta | 2022-01-17 |
| Proceeding 26617 - Appendix E - Demand Side Management Supplemental Information | application | Alberta | 2022-01-17 |
| Proceeding 26617 - Appendix I - Advanced Metering Infrastructure Project Business Case | application | Alberta | 2022-01-17 |
| EPCOR 2023 Phase I Distribution Tariff Application - Blackline | application | Alberta | 2022-01-17 |
| FortisAlberta Inc. 2023 Cost of Service Application | application | Alberta | 2021-11-15 |
| FortisAlberta 2023 COS Application - Appendix B - Software - Major Applications | application | Alberta | 2021-11-15 |
| FortisAlberta 2023 COS Application - Appendix D - Advanced Metering Infrastructure Business Case (Redacted) | application | Alberta | 2021-11-15 |
| FortisAlberta 2023 COS Application - Appendix G - Demand Side Management | application | Alberta | 2021-11-15 |
| SMUD 5-Year Strategic Plan 2020-2024 | plan | California | 2020-01-01 |
| Order No. 34281 - Dismissing the Smart Grid Foundation Project Application and Providing Guidance for a Grid Modernization Strategy | order | Hawaii | 2017-01-04 |
| Customer Energy Resources for Hawai'i - A Customer-First CER Strategy | plan | Hawaii | 2021-05-01 |
| Modernizing Hawai'i's Grid For Our Customers - Grid Modernization Strategy | plan | Hawaii | 2017-08-29 |
| Grid Modernization Phase 1 Implementation Plan (Application) | application | Hawaii | 2018-06-21 |
| Grid Modernization Phase 2 Implementation Plan (Updated Application - ADMS and Field Devices) | application | Hawaii | 2021-03-31 |
| Consumers Energy Grid Modernization Roadmap Assessment | research | Michigan | 2021-04-01 |
| Central Hudson Distributed System Implementation Plan (DSIP) | plan | New York | 2020-06-30 |
| A Modernized Grid - The Platform for a Clean Energy Future | research | Oregon |  |
| Hydro-Quebec Strategic Plan 2022-2026 | plan | Quebec | 2022-01-01 |
| Hydro-Quebec Plan strategique 2022-2026 | plan | Quebec | 2022-01-01 |
| Maximizing Distributed Energy Resource Value Through Grid Modernization | research | United States |  |
| A Playbook for Modernizing the Distribution Grid - Volume 1 - Goals, Principles and Plan Evaluation Checklist | guideline | United States |  |
| Load Forecasting in Electric Utility Integrated Resource Planning (LBNL-1006395) | research | United States | 2016-10-01 |
| Load Forecasting with Climate Variability for Transmission and Distribution System Planning | research | United States | 2021-11-01 |
| Grid Modernization: Policy, Market Trends and Directions Forward | research | United States | 2019-05-21 |

## Metadata stored per document
`doc_id, title, publisher, utility, jurisdiction, doc_type (application|decision|plan|guideline|research|order),
proceeding, date, language, superseded_by, access (public|internal), source_path`

## Tools (Unity Catalog functions)
```sql
search_reports(
  query STRING,                 -- natural-language query
  k INT DEFAULT 3,              -- chunks to return (1-50)
  utility STRING DEFAULT NULL,  -- substring match on utility
  doc_type STRING DEFAULT NULL, -- exact match
  jurisdiction STRING DEFAULT NULL
) RETURNS TABLE (chunk_id, doc_id, page, title, publisher, utility, jurisdiction, doc_type,
                 proceeding, date, language, superseded_by, access, score, text)

get_report_metadata(doc_id STRING DEFAULT NULL)   -- NULL lists the library
  RETURNS TABLE (doc_id, title, publisher, utility, jurisdiction, doc_type, proceeding, date,
                 language, superseded_by, access, source_path)
```

## How answers are scored
| Scorer | Kind |
|---|---|
| retrieval hit: an expected document was retrieved | deterministic |
| correct not-found: declines when the library lacks the answer, answers when it has it | deterministic |
| citation present: `[doc_id, p. N]` pointing at a real document | deterministic |
| expectations_guidelines: the answer meets the question's guidelines in the eval set | LLM judge |
| retrieval groundedness: the answer is supported by the retrieved passages | LLM judge |
| relevance to query: the answer addresses the question asked | LLM judge |
| tool call efficiency: no redundant or looping tool calls | LLM judge |
| safety: no harmful content | LLM judge |
| latency, tokens, steps | deterministic |

The judges run on a separate endpoint (`jsa-judge`, a different model from the agent's). A question passes when
correct not-found and expectations_guidelines both pass; the other scorers are diagnostics.
