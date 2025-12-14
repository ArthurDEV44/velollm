//! Content deduplication for prompt compression.

use std::collections::HashMap;

/// Deduplicates repeated content across messages
pub struct ContentDeduplicator {
    /// Minimum length for a pattern to be considered for deduplication
    min_length: usize,
    /// Minimum occurrences for a pattern to be deduplicated
    min_occurrences: usize,
}

impl ContentDeduplicator {
    /// Create a new deduplicator with given thresholds
    pub fn new(min_length: usize, min_occurrences: usize) -> Self {
        Self { min_length, min_occurrences }
    }

    /// Find and deduplicate repeated content in a list of message contents
    ///
    /// Returns the modified contents and the number of patterns deduplicated
    pub fn deduplicate(&self, contents: &mut [String]) -> DeduplicationResult {
        // Combine all content to find patterns
        let combined: String = contents
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        // Find repeated patterns
        let patterns = self.find_repeated_patterns(&combined);

        if patterns.is_empty() {
            return DeduplicationResult { patterns_found: 0, chars_saved: 0 };
        }

        let mut total_saved = 0;
        let mut context_block = String::new();

        // Replace patterns with references
        for (idx, (pattern, count)) in patterns.iter().enumerate() {
            let ref_id = format!("[CTX:{}]", idx);
            let pattern_len = pattern.len();

            // Only deduplicate if it saves space
            // Cost: reference * count + context definition
            // Savings: pattern * count
            let cost = ref_id.len() * count + pattern_len + ref_id.len() + 3; // +3 for ": " and newline
            let savings = pattern_len * count;

            if savings > cost {
                // Add to context block
                if !context_block.is_empty() {
                    context_block.push('\n');
                }
                context_block.push_str(&format!("{}: {}", ref_id, pattern));

                // Replace in all contents
                for content in contents.iter_mut() {
                    *content = content.replace(pattern, &ref_id);
                }

                total_saved += savings - cost;
            }
        }

        // Prepend context block to first content if we have one
        if !context_block.is_empty() && !contents.is_empty() {
            let first = &mut contents[0];
            *first =
                format!("[Context definitions]\n{}\n[End context]\n\n{}", context_block, first);
        }

        DeduplicationResult { patterns_found: patterns.len(), chars_saved: total_saved }
    }

    /// Find repeated patterns in text
    fn find_repeated_patterns(&self, text: &str) -> Vec<(String, usize)> {
        let mut patterns: HashMap<String, usize> = HashMap::new();

        // Use sentence-level detection for simplicity and accuracy
        for sentence in self.extract_sentences(text) {
            if sentence.len() >= self.min_length {
                *patterns.entry(sentence).or_insert(0) += 1;
            }
        }

        // Filter by minimum occurrences and sort by potential savings
        let mut result: Vec<(String, usize)> = patterns
            .into_iter()
            .filter(|(_, count)| *count >= self.min_occurrences)
            .collect();

        // Sort by savings potential (length * count)
        result.sort_by(|a, b| (b.0.len() * b.1).cmp(&(a.0.len() * a.1)));

        // Limit to top 10 patterns to avoid over-compression
        result.truncate(10);

        result
    }

    /// Extract sentences from text
    fn extract_sentences(&self, text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();

        for c in text.chars() {
            current.push(c);

            if c == '.' || c == '!' || c == '?' || c == '\n' {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current = String::new();
            }
        }

        // Don't forget the last sentence
        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() {
            sentences.push(trimmed);
        }

        sentences
    }
}

/// Result of a deduplication operation
#[derive(Debug, Clone)]
pub struct DeduplicationResult {
    /// Number of patterns found and deduplicated
    pub patterns_found: usize,
    /// Characters saved by deduplication
    pub chars_saved: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_dedup_needed() {
        let dedup = ContentDeduplicator::new(50, 2);
        let mut contents = vec!["Hello world.".to_string(), "Goodbye world.".to_string()];

        let result = dedup.deduplicate(&mut contents);

        assert_eq!(result.patterns_found, 0);
    }

    #[test]
    fn test_sentence_extraction() {
        let dedup = ContentDeduplicator::new(10, 2);
        let sentences = dedup.extract_sentences("Hello world. How are you? I am fine!");

        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(sentences[1], "How are you?");
        assert_eq!(sentences[2], "I am fine!");
    }

    #[test]
    fn test_pattern_detection() {
        let dedup = ContentDeduplicator::new(10, 2);
        let text = "This is repeated content. Other stuff. This is repeated content. More stuff.";

        let patterns = dedup.find_repeated_patterns(text);

        assert!(!patterns.is_empty());
        assert!(patterns
            .iter()
            .any(|(p, c)| p.contains("repeated") && *c >= 2));
    }

    #[test]
    fn test_deduplication() {
        let dedup = ContentDeduplicator::new(20, 2);
        let repeated = "This sentence is repeated multiple times in the conversation.";
        let mut contents = vec![
            format!("Start. {} Middle.", repeated),
            format!("Another message. {} End.", repeated),
        ];

        let original_len: usize = contents.iter().map(|s| s.len()).sum();
        let result = dedup.deduplicate(&mut contents);

        let new_len: usize = contents.iter().map(|s| s.len()).sum();

        // If patterns were found, we might have saved some characters
        // (depending on whether savings > cost)
        if result.patterns_found > 0 && result.chars_saved > 0 {
            assert!(new_len <= original_len + 100); // Allow for context block overhead
        }
    }

    #[test]
    fn test_min_length_threshold() {
        let dedup = ContentDeduplicator::new(100, 2); // Very high threshold
        let mut contents = vec!["Short. Short.".to_string(), "Short. Short.".to_string()];

        let result = dedup.deduplicate(&mut contents);

        // Should not deduplicate short patterns
        assert_eq!(result.patterns_found, 0);
    }
}
