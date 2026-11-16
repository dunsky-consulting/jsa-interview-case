-- Rendered by jsa.tools.render_tool_sql: brace placeholders are filled from Settings.
-- vector_search() needs a constant num_results and has no filter argument, so we over-fetch
-- 50 neighbours and filter/rank afterwards. Parameters are qualified with the function name
-- (search_reports.utility) because they share names with index columns.
CREATE OR REPLACE FUNCTION {catalog}.{schema}.search_reports(
  query STRING COMMENT 'Natural-language search query.',
  k INT DEFAULT 3 COMMENT 'Number of chunks to return (1-50).',
  utility STRING DEFAULT NULL COMMENT 'Optional case-insensitive substring match on the utility column.',
  doc_type STRING DEFAULT NULL COMMENT 'Optional exact match: application, decision, plan, guideline, research or order.',
  jurisdiction STRING DEFAULT NULL COMMENT 'Optional exact match, e.g. Alberta, Hawaii, Quebec.'
)
RETURNS TABLE (
  chunk_id STRING,
  doc_id STRING,
  page INT,
  title STRING,
  publisher STRING,
  utility STRING,
  jurisdiction STRING,
  doc_type STRING,
  proceeding STRING,
  date STRING,
  language STRING,
  superseded_by STRING,
  access STRING,
  score DOUBLE,
  text STRING
)
COMMENT 'Semantic search over the report library. Returns the top-k matching chunks with document metadata.'
RETURN
  SELECT chunk_id, doc_id, page, title, publisher, utility, jurisdiction, doc_type, proceeding, date,
         language, superseded_by, access, score, text
  FROM (
    SELECT *, search_score AS score, row_number() OVER (ORDER BY search_score DESC) AS rank
    FROM vector_search(
      index => '{index_name}',
      query_vector => ai_query('{embedding_endpoint}', concat('query: ', search_reports.query), returnType => 'ARRAY<FLOAT>'),
      num_results => 50
    )
    WHERE (search_reports.utility IS NULL OR lower(utility) LIKE concat('%', lower(search_reports.utility), '%'))
      AND (search_reports.doc_type IS NULL OR doc_type = search_reports.doc_type)
      AND (search_reports.jurisdiction IS NULL OR jurisdiction = search_reports.jurisdiction)
  )
  -- Parameters are not resolvable inside QUALIFY, hence the subquery.
  WHERE rank <= least(greatest(coalesce(search_reports.k, 3), 1), 50)
  ORDER BY rank
