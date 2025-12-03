//! llama.cpp timing output parser
//!
//! Parses performance metrics from llama.cpp inference output.

use once_cell::sync::Lazy;
use regex::Regex;

/// Parser for llama.cpp timing output
pub struct LlamaCppParser;

/// Performance metrics extracted from llama.cpp output
#[derive(Debug, Default, Clone, PartialEq)]
pub struct LlamaMetrics {
    /// Time to load the model in milliseconds
    pub load_time_ms: Option<f64>,
    /// Time spent sampling in milliseconds
    pub sample_time_ms: Option<f64>,
    /// Prompt evaluation time in milliseconds (time to first token)
    pub prompt_eval_time_ms: Option<f64>,
    /// Number of tokens in the prompt
    pub prompt_tokens: Option<u32>,
    /// Prompt processing speed (tokens per second)
    pub prompt_tokens_per_second: Option<f64>,
    /// Token generation time in milliseconds
    pub eval_time_ms: Option<f64>,
    /// Number of tokens generated
    pub eval_tokens: Option<u32>,
    /// Token generation speed (tokens per second)
    pub tokens_per_second: Option<f64>,
    /// Total inference time in milliseconds
    pub total_time_ms: Option<f64>,
    /// Total number of tokens processed
    pub total_tokens: Option<u32>,
}

impl LlamaCppParser {
    /// Parse all timing metrics from llama.cpp output
    ///
    /// Expected format:
    /// ```text
    /// llama_print_timings:        load time =     123.45 ms
    /// llama_print_timings:      sample time =      10.23 ms /    50 runs   (    0.20 ms per token,  4882.81 tokens per second)
    /// llama_print_timings: prompt eval time =     234.56 ms /    10 tokens (   23.46 ms per token,    42.63 tokens per second)
    /// llama_print_timings:        eval time =    2000.00 ms /    49 runs   (   40.82 ms per token,    24.50 tokens per second)
    /// llama_print_timings:       total time =    2368.24 ms /    59 tokens
    /// ```
    pub fn parse(output: &str) -> LlamaMetrics {
        let mut metrics = LlamaMetrics::default();

        for line in output.lines() {
            if !line.contains("llama_print_timings") && !line.contains("llama_perf_") {
                continue;
            }

            if line.contains("load time") {
                metrics.load_time_ms = Self::extract_time_ms(line);
            } else if line.contains("sample time") {
                metrics.sample_time_ms = Self::extract_time_ms(line);
            } else if line.contains("prompt eval time") {
                metrics.prompt_eval_time_ms = Self::extract_time_ms(line);
                metrics.prompt_tokens = Self::extract_token_count(line);
                metrics.prompt_tokens_per_second = Self::extract_tokens_per_second(line);
            } else if line.contains("eval time") {
                metrics.eval_time_ms = Self::extract_time_ms(line);
                metrics.eval_tokens = Self::extract_run_count(line);
                metrics.tokens_per_second = Self::extract_tokens_per_second(line);
            } else if line.contains("total time") {
                metrics.total_time_ms = Self::extract_time_ms(line);
                metrics.total_tokens = Self::extract_token_count(line);
            }
        }

        metrics
    }

    /// Check if the output contains valid timing information
    pub fn has_timing_info(output: &str) -> bool {
        output.contains("llama_print_timings") || output.contains("llama_perf_")
    }

    /// Extract time in milliseconds from a timing line
    ///
    /// Handles formats like:
    /// - "load time =     123.45 ms"
    /// - "eval time =    2000.00 ms /"
    pub fn extract_time_ms(line: &str) -> Option<f64> {
        // Pattern: "= <number> ms" with flexible spacing
        static TIME_REGEX: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"=\s+(\d+(?:\.\d+)?)\s*ms").expect("Invalid time regex"));

        TIME_REGEX
            .captures(line)
            .and_then(|caps| caps.get(1))
            .and_then(|m| m.as_str().parse().ok())
    }

    /// Extract tokens per second from a timing line
    ///
    /// Handles formats like:
    /// - "(   40.82 ms per token,    24.50 tokens per second)"
    /// - "(0.20 ms per token, 4882.81 tokens per second)"
    pub fn extract_tokens_per_second(line: &str) -> Option<f64> {
        // Pattern: "<number> tokens per second"
        static TPS_REGEX: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"(\d+(?:\.\d+)?)\s+tokens per second")
                .expect("Invalid tokens per second regex")
        });

        TPS_REGEX
            .captures(line)
            .and_then(|caps| caps.get(1))
            .and_then(|m| m.as_str().parse().ok())
    }

    /// Extract token count from a timing line
    ///
    /// Handles formats like:
    /// - "/ 10 tokens"
    /// - "/    59 tokens"
    pub fn extract_token_count(line: &str) -> Option<u32> {
        // Pattern: "/ <number> tokens"
        static TOKEN_REGEX: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"/\s+(\d+)\s+tokens").expect("Invalid token count regex"));

        TOKEN_REGEX
            .captures(line)
            .and_then(|caps| caps.get(1))
            .and_then(|m| m.as_str().parse().ok())
    }

    /// Extract run count from eval time line
    ///
    /// Handles formats like:
    /// - "/ 49 runs"
    pub fn extract_run_count(line: &str) -> Option<u32> {
        // Pattern: "/ <number> runs"
        static RUN_REGEX: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"/\s+(\d+)\s+runs").expect("Invalid run count regex"));

        RUN_REGEX
            .captures(line)
            .and_then(|caps| caps.get(1))
            .and_then(|m| m.as_str().parse().ok())
    }

    /// Extract milliseconds per token from a timing line
    ///
    /// Handles formats like:
    /// - "(   40.82 ms per token,"
    pub fn extract_ms_per_token(line: &str) -> Option<f64> {
        static MS_TOKEN_REGEX: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"\(\s*(\d+(?:\.\d+)?)\s+ms per token").expect("Invalid ms per token regex")
        });

        MS_TOKEN_REGEX
            .captures(line)
            .and_then(|caps| caps.get(1))
            .and_then(|m| m.as_str().parse().ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Test fixtures =====

    const LLAMA_OUTPUT_FULL: &str = "\
llama_print_timings:        load time =     123.45 ms
llama_print_timings:      sample time =      10.23 ms /    50 runs   (    0.20 ms per token,  4882.81 tokens per second)
llama_print_timings: prompt eval time =     234.56 ms /    10 tokens (   23.46 ms per token,    42.63 tokens per second)
llama_print_timings:        eval time =    2000.00 ms /    49 runs   (   40.82 ms per token,    24.50 tokens per second)
llama_print_timings:       total time =    2368.24 ms /    59 tokens
";

    const LLAMA_OUTPUT_MINIMAL: &str = "\
llama_print_timings:        eval time =    1500.00 ms /    30 runs   (   50.00 ms per token,    20.00 tokens per second)
";

    const LLAMA_OUTPUT_NEW_FORMAT: &str = "\
llama_perf_context_print:        load time =     500.00 ms
llama_perf_context_print: prompt eval time =     150.00 ms /    20 tokens (    7.50 ms per token,   133.33 tokens per second)
llama_perf_context_print:        eval time =    3000.00 ms /   100 runs   (   30.00 ms per token,    33.33 tokens per second)
llama_perf_context_print:       total time =    3650.00 ms /   120 tokens
";

    const LLAMA_OUTPUT_WITH_NOISE: &str = "\
Loading model...
Model loaded successfully.
Processing prompt...

llama_print_timings:        load time =     100.00 ms
llama_print_timings:        eval time =    1000.00 ms /    50 runs   (   20.00 ms per token,    50.00 tokens per second)
llama_print_timings:       total time =    1100.00 ms /    50 tokens

Generation complete.
";

    // ===== Full parsing tests =====

    #[test]
    fn test_parse_full_output() {
        let metrics = LlamaCppParser::parse(LLAMA_OUTPUT_FULL);

        assert_eq!(metrics.load_time_ms, Some(123.45));
        assert_eq!(metrics.sample_time_ms, Some(10.23));
        assert_eq!(metrics.prompt_eval_time_ms, Some(234.56));
        assert_eq!(metrics.prompt_tokens, Some(10));
        assert_eq!(metrics.prompt_tokens_per_second, Some(42.63));
        assert_eq!(metrics.eval_time_ms, Some(2000.00));
        assert_eq!(metrics.eval_tokens, Some(49));
        assert_eq!(metrics.tokens_per_second, Some(24.50));
        assert_eq!(metrics.total_time_ms, Some(2368.24));
        assert_eq!(metrics.total_tokens, Some(59));
    }

    #[test]
    fn test_parse_minimal_output() {
        let metrics = LlamaCppParser::parse(LLAMA_OUTPUT_MINIMAL);

        assert_eq!(metrics.load_time_ms, None);
        assert_eq!(metrics.prompt_eval_time_ms, None);
        assert_eq!(metrics.eval_time_ms, Some(1500.00));
        assert_eq!(metrics.tokens_per_second, Some(20.00));
    }

    #[test]
    fn test_parse_new_format() {
        let metrics = LlamaCppParser::parse(LLAMA_OUTPUT_NEW_FORMAT);

        assert_eq!(metrics.load_time_ms, Some(500.00));
        assert_eq!(metrics.prompt_eval_time_ms, Some(150.00));
        assert_eq!(metrics.prompt_tokens, Some(20));
        assert_eq!(metrics.tokens_per_second, Some(33.33));
        assert_eq!(metrics.total_time_ms, Some(3650.00));
    }

    #[test]
    fn test_parse_with_noise() {
        let metrics = LlamaCppParser::parse(LLAMA_OUTPUT_WITH_NOISE);

        assert_eq!(metrics.load_time_ms, Some(100.00));
        assert_eq!(metrics.eval_time_ms, Some(1000.00));
        assert_eq!(metrics.tokens_per_second, Some(50.00));
        assert_eq!(metrics.total_time_ms, Some(1100.00));
    }

    #[test]
    fn test_parse_empty() {
        let metrics = LlamaCppParser::parse("");
        assert_eq!(metrics, LlamaMetrics::default());
    }

    #[test]
    fn test_parse_no_timing() {
        let metrics = LlamaCppParser::parse("Just some random output\nNo timing here");
        assert_eq!(metrics, LlamaMetrics::default());
    }

    // ===== Individual extractor tests =====

    #[test]
    fn test_extract_time_ms() {
        assert_eq!(LlamaCppParser::extract_time_ms("load time =     123.45 ms"), Some(123.45));
        assert_eq!(LlamaCppParser::extract_time_ms("eval time =    2000.00 ms /"), Some(2000.00));
        assert_eq!(LlamaCppParser::extract_time_ms("time = 1000 ms"), Some(1000.0));
    }

    #[test]
    fn test_extract_tokens_per_second() {
        assert_eq!(
            LlamaCppParser::extract_tokens_per_second(
                "(   40.82 ms per token,    24.50 tokens per second)"
            ),
            Some(24.50)
        );
        assert_eq!(
            LlamaCppParser::extract_tokens_per_second(
                "(0.20 ms per token, 4882.81 tokens per second)"
            ),
            Some(4882.81)
        );
    }

    #[test]
    fn test_extract_token_count() {
        assert_eq!(LlamaCppParser::extract_token_count("/    10 tokens ("), Some(10));
        assert_eq!(LlamaCppParser::extract_token_count("/ 59 tokens"), Some(59));
    }

    #[test]
    fn test_extract_run_count() {
        assert_eq!(LlamaCppParser::extract_run_count("/    49 runs   ("), Some(49));
        assert_eq!(LlamaCppParser::extract_run_count("/ 30 runs"), Some(30));
    }

    #[test]
    fn test_extract_ms_per_token() {
        assert_eq!(LlamaCppParser::extract_ms_per_token("(   40.82 ms per token,"), Some(40.82));
        assert_eq!(LlamaCppParser::extract_ms_per_token("(0.20 ms per token,"), Some(0.20));
    }

    #[test]
    fn test_has_timing_info() {
        assert!(LlamaCppParser::has_timing_info(LLAMA_OUTPUT_FULL));
        assert!(LlamaCppParser::has_timing_info(LLAMA_OUTPUT_NEW_FORMAT));
        assert!(!LlamaCppParser::has_timing_info("No timing here"));
    }
}
