output "cloud_run_service_uri" {
  description = "Deployed URL of the Work-to-Blog ADK Agent Cloud Run service."
  value       = google_cloud_run_v2_service.blog_writer_agent_service.uri
}

output "agent_service_account_email" {
  description = "Least-privilege Service Account email used by the agent runtime."
  value       = google_service_account.blog_agent_runtime_sa.email
}

output "vertex_search_data_store_id" {
  description = "Provisioned Vertex AI Search Data Store ID for persistent knowledge retrieval."
  value       = google_discovery_engine_data_store.technical_blog_knowledge_store.data_store_id
}

output "cms_secret_manager_id" {
  description = "Secret Manager resource ID storing the CMS publishing token."
  value       = google_secret_manager_secret.cms_publishing_token.secret_id
}
