//! API capabilities of the GLM-only engine.
pub fn json(cuda: bool) -> String {
    format!(
        r#"{{
  "schema_version":1,
  "cuda":{cuda},
  "requires_cuda_for_inference":true,
  "models":[{{"id":"glm53","kind":"chat","variants":["fp8"],"context_length":131072,"resident":true,"elastic_vram":true}}],
  "chat":{{"models":["glm53"],"stream":true,"tools":true,"disconnect_cancels":true,
    "glm53":{{"context_length":131072,"max_output_default":32768,"parallel_tool_calls":true,"tool_choice":["auto"],"reasoning_content":true,
      "thinking":{{"enable_thinking":true,"chat_template_kwargs":true,"reasoning_effort":["low","high","max","none"],"reasoning_budget":true}},
      "sampling":["temperature","top_p","top_k","min_p","repetition_penalty","seed","stop"],"usage_in_final_chunk":true,"prefix_reuse":true}}}},
  "gpu":{{"status":"/v1/gpu","elastic_vram":true}},
  "jobs":{{"image":{{"models":[]}},"video":{{"models":[]}}}}
}}"#
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn only_glm_is_advertised() {
        for cuda in [false, true] {
            let v = crate::json::parse(&super::json(cuda)).unwrap();
            let models = v.get("models").unwrap().array().unwrap();
            assert_eq!(models.len(), 1);
            assert_eq!(models[0].get("id").unwrap().string().unwrap(), "glm53");
            assert_eq!(v.get("cuda").unwrap().boolean().unwrap(), cuda);
        }
    }
}
