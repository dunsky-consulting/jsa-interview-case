-- Rendered by jsa.tools.render_tool_sql: brace placeholders are filled from Settings.
-- Pass NULL to list every document in the library.
CREATE OR REPLACE FUNCTION {catalog}.{schema}.get_report_metadata(
  doc_id STRING DEFAULT NULL COMMENT 'Document id as returned by search_reports; NULL lists all documents.'
)
RETURNS TABLE (
  doc_id STRING,
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
  source_path STRING
)
COMMENT 'Metadata for one report (or the whole library): publisher, utility, type, proceeding, date, supersession, access level.'
RETURN
  SELECT doc_id, title, publisher, utility, jurisdiction, doc_type, proceeding, date, language,
         superseded_by, access, source_path
  FROM {documents_table}
  WHERE get_report_metadata.doc_id IS NULL OR doc_id = get_report_metadata.doc_id
