//! Tool calling optimizer for VeloLLM proxy.
//!
//! This module enhances the reliability of tool calling by:
//! - Fixing malformed JSON in tool call arguments
//! - Deduplicating tool calls
//! - Validating arguments against JSON Schema
//! - Supporting only Mistral and Llama models (reliable tool calling)
//!
//! References:
//! - Ollama Tool Calling: https://docs.ollama.com/capabilities/tool-calling
//! - Mistral Function Calling: https://docs.mistral.ai/capabilities/function_calling
//! - Llama 3.1/3.2: https://www.llama.com/docs/model-cards-and-prompt-formats/

use crate::types::openai::{Tool, ToolCall};
use jsonschema::Validator;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors that can occur during tool optimization
#[derive(Debug, Error)]
pub enum ToolError {
    /// JSON is invalid and cannot be repaired
    #[error("Invalid JSON that cannot be repaired: {0}")]
    InvalidJson(String),

    /// JSON Schema validation failed
    #[error("Validation failed for tool '{tool}': {message}")]
    ValidationFailed { tool: String, message: String },

    /// Schema compilation error
    #[error("Failed to compile schema for tool '{tool}': {message}")]
    SchemaCompilationError { tool: String, message: String },
}

/// Models that reliably support tool calling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportedModel {
    /// Mistral models (mistral:7b, mistral-small, mistral-nemo, etc.)
    Mistral,
    /// Llama 3.1+ models (llama3.1:8b, llama3.2:3b, etc.)
    Llama,
}

impl SupportedModel {
    /// Check if a model name corresponds to a supported model
    pub fn from_model_name(model: &str) -> Option<Self> {
        let model_lower = model.to_lowercase();

        // Check for Mistral models
        if model_lower.contains("mistral") {
            return Some(Self::Mistral);
        }

        // Check for Llama 3.1+ models (supports tool calling)
        if model_lower.contains("llama3.1")
            || model_lower.contains("llama3.2")
            || model_lower.contains("llama3.3")
            || model_lower.contains("llama-3.1")
            || model_lower.contains("llama-3.2")
            || model_lower.contains("llama-3.3")
            || model_lower.contains("llama4")
            || model_lower.contains("llama-4")
        {
            return Some(Self::Llama);
        }

        // Older Llama 3 might work but less reliably
        if model_lower.contains("llama3") || model_lower.contains("llama-3") {
            // Only versions 3.1 and above are officially supported
            // but we allow llama3 as a fallback with a warning
            return Some(Self::Llama);
        }

        None
    }

    /// Get model family name
    pub fn family(&self) -> &'static str {
        match self {
            Self::Mistral => "mistral",
            Self::Llama => "llama",
        }
    }
}

/// Tool optimizer for enhancing tool calling reliability
pub struct ToolOptimizer {
    /// Compiled JSON Schema validators for each registered tool
    validators: HashMap<String, Validator>,
    /// Tool definitions for reference
    tools: HashMap<String, Tool>,
    /// Statistics
    stats: ToolOptimizerStats,
}

/// Statistics for tool optimization
#[derive(Debug, Default, Clone)]
pub struct ToolOptimizerStats {
    /// Number of tool calls processed
    pub calls_processed: u64,
    /// Number of JSON repairs performed
    pub json_repairs: u64,
    /// Number of duplicate calls removed
    pub duplicates_removed: u64,
    /// Number of validation failures
    pub validation_failures: u64,
    /// Number of successful validations
    pub validation_successes: u64,
}

impl Default for ToolOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolOptimizer {
    /// Create a new tool optimizer
    pub fn new() -> Self {
        Self {
            validators: HashMap::new(),
            tools: HashMap::new(),
            stats: ToolOptimizerStats::default(),
        }
    }

    /// Get optimization statistics
    pub fn stats(&self) -> &ToolOptimizerStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = ToolOptimizerStats::default();
    }

    /// Check if a model supports tool calling
    pub fn is_model_supported(model: &str) -> bool {
        SupportedModel::from_model_name(model).is_some()
    }

    /// Get the supported model type from a model name
    pub fn get_model_type(model: &str) -> Option<SupportedModel> {
        SupportedModel::from_model_name(model)
    }

    /// Register tools and compile their JSON schemas for validation
    pub fn register_tools(&mut self, tools: &[Tool]) -> Result<(), ToolError> {
        for tool in tools {
            let name = &tool.function.name;
            let schema = &tool.function.parameters;

            // Compile the JSON schema
            match Validator::new(schema) {
                Ok(validator) => {
                    debug!(tool = %name, "Registered tool schema");
                    self.validators.insert(name.clone(), validator);
                    self.tools.insert(name.clone(), tool.clone());
                }
                Err(e) => {
                    warn!(tool = %name, error = %e, "Failed to compile schema, validation disabled");
                    // Store the tool but without validation
                    self.tools.insert(name.clone(), tool.clone());
                }
            }
        }

        info!(count = tools.len(), "Registered tools for optimization");
        Ok(())
    }

    /// Clear all registered tools
    pub fn clear_tools(&mut self) {
        self.validators.clear();
        self.tools.clear();
    }

    /// Process and optimize tool calls from model response
    ///
    /// This performs:
    /// 1. JSON repair for malformed arguments
    /// 2. Deduplication of identical tool calls
    /// 3. Validation against JSON Schema (if available)
    pub fn process_tool_calls(&mut self, calls: Vec<ToolCall>) -> Vec<ToolCall> {
        if calls.is_empty() {
            return calls;
        }

        let original_count = calls.len();
        info!(count = original_count, "Processing tool calls");

        let mut result = Vec::with_capacity(calls.len());
        let mut seen_signatures = HashSet::new();

        for mut call in calls {
            self.stats.calls_processed += 1;

            // 1. Fix JSON arguments
            match self.fix_json(&call.function.arguments) {
                Ok(fixed) => {
                    if fixed != call.function.arguments {
                        debug!(
                            tool = %call.function.name,
                            original = %call.function.arguments,
                            fixed = %fixed,
                            "Repaired JSON arguments"
                        );
                        self.stats.json_repairs += 1;
                        call.function.arguments = fixed;
                    }
                }
                Err(e) => {
                    warn!(
                        tool = %call.function.name,
                        error = %e,
                        args = %call.function.arguments,
                        "Failed to repair JSON, keeping original"
                    );
                }
            }

            // 2. Deduplicate by signature (function name + arguments)
            let signature = format!("{}:{}", call.function.name, call.function.arguments);
            if seen_signatures.contains(&signature) {
                debug!(
                    tool = %call.function.name,
                    "Removing duplicate tool call"
                );
                self.stats.duplicates_removed += 1;
                continue;
            }
            seen_signatures.insert(signature);

            // 3. Validate against schema
            if let Err(e) = self.validate_call(&call) {
                warn!(
                    tool = %call.function.name,
                    error = %e,
                    "Validation failed, including call anyway"
                );
                self.stats.validation_failures += 1;
            } else {
                self.stats.validation_successes += 1;
            }

            result.push(call);
        }

        info!(
            original = original_count,
            processed = result.len(),
            repairs = self.stats.json_repairs,
            duplicates = self.stats.duplicates_removed,
            "Tool calls processed"
        );

        result
    }

    /// Fix common JSON issues in tool call arguments
    ///
    /// Common issues fixed:
    /// - Markdown code blocks (```json ... ```)
    /// - Trailing commas in objects/arrays
    /// - Single quotes instead of double quotes
    /// - Unquoted property names (simple cases)
    /// - Mixed content with JSON embedded
    pub fn fix_json(&self, raw: &str) -> Result<String, ToolError> {
        let trimmed = raw.trim();

        // Empty or null case
        if trimmed.is_empty() || trimmed == "null" {
            return Ok("{}".to_string());
        }

        // 1. Try parsing as-is first
        if serde_json::from_str::<Value>(trimmed).is_ok() {
            return Ok(trimmed.to_string());
        }

        // 2. Remove markdown code blocks
        let mut fixed = trimmed.to_string();
        if fixed.starts_with("```json") || fixed.starts_with("```JSON") {
            fixed = fixed
                .trim_start_matches("```json")
                .trim_start_matches("```JSON")
                .to_string();
        }
        if fixed.starts_with("```") {
            fixed = fixed.trim_start_matches("```").to_string();
        }
        if fixed.ends_with("```") {
            fixed = fixed.trim_end_matches("```").to_string();
        }
        fixed = fixed.trim().to_string();

        // Check if fixed
        if serde_json::from_str::<Value>(&fixed).is_ok() {
            return Ok(fixed);
        }

        // 3. Fix trailing commas (common LLM mistake)
        // Replace ,} with } and ,] with ]
        let re_trailing_comma_obj = regex::Regex::new(r",\s*\}").unwrap();
        let re_trailing_comma_arr = regex::Regex::new(r",\s*\]").unwrap();
        fixed = re_trailing_comma_obj.replace_all(&fixed, "}").to_string();
        fixed = re_trailing_comma_arr.replace_all(&fixed, "]").to_string();

        if serde_json::from_str::<Value>(&fixed).is_ok() {
            return Ok(fixed);
        }

        // 4. Replace single quotes with double quotes (careful with strings)
        // This is a simple heuristic, not perfect
        if fixed.contains('\'') && !fixed.contains('"') {
            fixed = fixed.replace('\'', "\"");
            if serde_json::from_str::<Value>(&fixed).is_ok() {
                return Ok(fixed);
            }
        }

        // 5. Try to extract JSON object from mixed content
        if let Some(extracted) = self.extract_json_object(&fixed) {
            if serde_json::from_str::<Value>(&extracted).is_ok() {
                return Ok(extracted);
            }
        }

        // 6. Try to extract JSON array from mixed content
        if let Some(extracted) = self.extract_json_array(&fixed) {
            if serde_json::from_str::<Value>(&extracted).is_ok() {
                return Ok(extracted);
            }
        }

        // If all else fails, return the best attempt
        Err(ToolError::InvalidJson(raw.to_string()))
    }

    /// Extract a JSON object from mixed text content
    fn extract_json_object(&self, text: &str) -> Option<String> {
        let start = text.find('{')?;
        let mut depth = 0;
        let mut in_string = false;
        let mut escape_next = false;

        for (i, c) in text[start..].char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match c {
                '\\' if in_string => escape_next = true,
                '"' => in_string = !in_string,
                '{' if !in_string => depth += 1,
                '}' if !in_string => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(text[start..=start + i].to_string());
                    }
                }
                _ => {}
            }
        }

        None
    }

    /// Extract a JSON array from mixed text content
    fn extract_json_array(&self, text: &str) -> Option<String> {
        let start = text.find('[')?;
        let mut depth = 0;
        let mut in_string = false;
        let mut escape_next = false;

        for (i, c) in text[start..].char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match c {
                '\\' if in_string => escape_next = true,
                '"' => in_string = !in_string,
                '[' if !in_string => depth += 1,
                ']' if !in_string => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(text[start..=start + i].to_string());
                    }
                }
                _ => {}
            }
        }

        None
    }

    /// Validate tool call arguments against JSON Schema
    pub fn validate_call(&self, call: &ToolCall) -> Result<(), ToolError> {
        // Parse arguments
        let args: Value = serde_json::from_str(&call.function.arguments).map_err(|e| {
            ToolError::ValidationFailed {
                tool: call.function.name.clone(),
                message: format!("Invalid JSON: {}", e),
            }
        })?;

        // Check if we have a validator for this tool
        if let Some(validator) = self.validators.get(&call.function.name) {
            // Collect validation errors
            let errors: Vec<String> = validator
                .iter_errors(&args)
                .map(|e| e.to_string())
                .collect();

            if !errors.is_empty() {
                return Err(ToolError::ValidationFailed {
                    tool: call.function.name.clone(),
                    message: errors.join("; "),
                });
            }
        }

        Ok(())
    }

    /// Validate arguments against a specific tool's schema
    pub fn validate_arguments(&self, tool_name: &str, args: &Value) -> Result<(), ToolError> {
        if let Some(validator) = self.validators.get(tool_name) {
            // Collect validation errors
            let errors: Vec<String> = validator.iter_errors(args).map(|e| e.to_string()).collect();

            if !errors.is_empty() {
                return Err(ToolError::ValidationFailed {
                    tool: tool_name.to_string(),
                    message: errors.join("; "),
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::openai::{FunctionCall, FunctionDef};

    fn create_weather_tool() -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: "get_weather".to_string(),
                description: Some("Get weather for a location".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }),
            },
        }
    }

    fn create_tool_call(name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: format!("call_{}", uuid::Uuid::new_v4()),
            call_type: "function".to_string(),
            function: FunctionCall { name: name.to_string(), arguments: args.to_string() },
        }
    }

    #[test]
    fn test_supported_models() {
        // Mistral models
        assert!(ToolOptimizer::is_model_supported("mistral:7b"));
        assert!(ToolOptimizer::is_model_supported("mistral-small:24b"));
        assert!(ToolOptimizer::is_model_supported("mistral-nemo"));
        assert_eq!(ToolOptimizer::get_model_type("mistral:7b"), Some(SupportedModel::Mistral));

        // Llama models
        assert!(ToolOptimizer::is_model_supported("llama3.1:8b"));
        assert!(ToolOptimizer::is_model_supported("llama3.2:3b"));
        assert!(ToolOptimizer::is_model_supported("llama3.3:70b"));
        assert_eq!(ToolOptimizer::get_model_type("llama3.1:8b"), Some(SupportedModel::Llama));

        // Unsupported models
        assert!(!ToolOptimizer::is_model_supported("qwen:7b"));
        assert!(!ToolOptimizer::is_model_supported("qwen2.5:32b"));
        assert!(!ToolOptimizer::is_model_supported("phi3:mini"));
        assert!(!ToolOptimizer::is_model_supported("gemma2:9b"));
        assert_eq!(ToolOptimizer::get_model_type("qwen:7b"), None);
    }

    #[test]
    fn test_json_fix_valid() {
        let optimizer = ToolOptimizer::new();

        // Already valid JSON
        let result = optimizer.fix_json(r#"{"location": "Paris"}"#);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), r#"{"location": "Paris"}"#);
    }

    #[test]
    fn test_json_fix_markdown() {
        let optimizer = ToolOptimizer::new();

        // JSON wrapped in markdown code blocks
        let result = optimizer.fix_json(
            r#"```json
{"location": "Paris"}
```"#,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), r#"{"location": "Paris"}"#);
    }

    #[test]
    fn test_json_fix_trailing_comma() {
        let optimizer = ToolOptimizer::new();

        // Trailing comma in object
        let result = optimizer.fix_json(r#"{"location": "Paris",}"#);
        assert!(result.is_ok());
        let parsed: Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(parsed["location"], "Paris");

        // Trailing comma in array
        let result = optimizer.fix_json(r#"["a", "b",]"#);
        assert!(result.is_ok());
    }

    #[test]
    fn test_json_fix_extract_from_text() {
        let optimizer = ToolOptimizer::new();

        // JSON embedded in text
        let result = optimizer.fix_json(r#"Here is the result: {"location": "Paris"} - done"#);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), r#"{"location": "Paris"}"#);
    }

    #[test]
    fn test_json_fix_single_quotes() {
        let optimizer = ToolOptimizer::new();

        // Single quotes (Python style)
        let result = optimizer.fix_json(r#"{'location': 'Paris'}"#);
        assert!(result.is_ok());
        let parsed: Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(parsed["location"], "Paris");
    }

    #[test]
    fn test_deduplication() {
        let mut optimizer = ToolOptimizer::new();

        let calls = vec![
            create_tool_call("get_weather", r#"{"location": "Paris"}"#),
            create_tool_call("get_weather", r#"{"location": "Paris"}"#), // Duplicate
            create_tool_call("get_weather", r#"{"location": "London"}"#),
        ];

        let result = optimizer.process_tool_calls(calls);

        assert_eq!(result.len(), 2);
        assert_eq!(optimizer.stats().duplicates_removed, 1);
    }

    #[test]
    fn test_validation_success() {
        let mut optimizer = ToolOptimizer::new();
        optimizer.register_tools(&[create_weather_tool()]).unwrap();

        let call = create_tool_call("get_weather", r#"{"location": "Paris"}"#);
        let result = optimizer.validate_call(&call);

        assert!(result.is_ok());
    }

    #[test]
    fn test_validation_failure_missing_required() {
        let mut optimizer = ToolOptimizer::new();
        optimizer.register_tools(&[create_weather_tool()]).unwrap();

        // Missing required "location" field
        let call = create_tool_call("get_weather", r#"{"unit": "celsius"}"#);
        let result = optimizer.validate_call(&call);

        assert!(result.is_err());
        if let Err(ToolError::ValidationFailed { tool, message }) = result {
            assert_eq!(tool, "get_weather");
            assert!(message.contains("location") || message.contains("required"));
        }
    }

    #[test]
    fn test_validation_failure_wrong_type() {
        let mut optimizer = ToolOptimizer::new();
        optimizer.register_tools(&[create_weather_tool()]).unwrap();

        // Wrong type for "location" (number instead of string)
        let call = create_tool_call("get_weather", r#"{"location": 123}"#);
        let result = optimizer.validate_call(&call);

        assert!(result.is_err());
    }

    #[test]
    fn test_process_with_json_repair() {
        let mut optimizer = ToolOptimizer::new();
        optimizer.register_tools(&[create_weather_tool()]).unwrap();

        // Tool call with malformed JSON (trailing comma)
        let calls = vec![create_tool_call("get_weather", r#"{"location": "Paris",}"#)];

        let result = optimizer.process_tool_calls(calls);

        assert_eq!(result.len(), 1);
        assert_eq!(optimizer.stats().json_repairs, 1);

        // Verify the JSON was fixed
        let fixed_args: Value = serde_json::from_str(&result[0].function.arguments).unwrap();
        assert_eq!(fixed_args["location"], "Paris");
    }

    #[test]
    fn test_empty_and_null_args() {
        let optimizer = ToolOptimizer::new();

        assert_eq!(optimizer.fix_json("").unwrap(), "{}");
        assert_eq!(optimizer.fix_json("null").unwrap(), "{}");
        assert_eq!(optimizer.fix_json("  ").unwrap(), "{}");
    }

    #[test]
    fn test_nested_json_extraction() {
        let optimizer = ToolOptimizer::new();

        let result = optimizer.fix_json(
            r#"Let me call the function with these args: {"user": {"name": "John", "age": 30}} end"#,
        );
        assert!(result.is_ok());
        let parsed: Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(parsed["user"]["name"], "John");
    }

    #[test]
    fn test_stats_tracking() {
        let mut optimizer = ToolOptimizer::new();
        optimizer.register_tools(&[create_weather_tool()]).unwrap();

        let calls = vec![
            create_tool_call("get_weather", r#"{"location": "Paris",}"#), // Needs repair
            create_tool_call("get_weather", r#"{"location": "Paris"}"#),  // Duplicate after repair
            create_tool_call("get_weather", r#"{"location": "London"}"#), // Valid
        ];

        let _ = optimizer.process_tool_calls(calls);

        assert_eq!(optimizer.stats().calls_processed, 3);
        assert_eq!(optimizer.stats().json_repairs, 1);
        assert_eq!(optimizer.stats().duplicates_removed, 1);
    }
}
