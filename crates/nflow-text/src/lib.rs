use jpreprocess::*;
use nflow_core::Sentence;
use anyhow::Result;

/// 日本語テキスト解析器
pub struct Analyzer {
    jpreprocess: JPreprocess<DefaultTokenizer>,
}

impl Analyzer {
    /// 辞書をロードして解析器を初期化する
    pub fn new() -> Result<Self> {
        let config = JPreprocessConfig {
             dictionary: SystemDictionaryConfig::Bundled(jpreprocess::kind::JPreprocessDictionaryKind::NaistJdic),
             user_dictionary: None,
        };
        let jpreprocess = JPreprocess::from_config(config)?;
        Ok(Self { jpreprocess })
    }

    /// テキストを解析し、音素列（Sentence）のリストに変換する
    pub fn analyze(&self, text: &str) -> Result<Vec<Sentence>> {
        let parts = self.split_sentences(text);
        let mut sentences = Vec::new();
        for part in parts {
            if let Some(s) = self.analyze_part(&part)? {
                sentences.push(s);
            }
        }
        Ok(sentences)
    }

    /// テキスト全体を一文として解析し、一つの Sentence を返す
    pub fn analyze_single(&self, text: &str) -> Result<Option<Sentence>> {
        self.analyze_part(text)
    }

    fn analyze_part(&self, text: &str) -> Result<Option<Sentence>> {
        if text.trim().is_empty() {
            return Ok(None);
        }

        let labels = self.jpreprocess.extract_fullcontext(text)?;
        let mut phonemes = Vec::new();
        let mut accents = Vec::new();

        for label in labels {
            if let Some((phoneme, accent)) = self.parse_label(&label.to_string()) {
                phonemes.push(phoneme);
                accents.push(accent);
            }
        }

        if phonemes.is_empty() {
            Ok(None)
        } else {
            Ok(Some(Sentence::new(text.to_string(), phonemes, accents, 0)))
        }
    }

    /// 日本語の文境界で分割する
    fn split_sentences(&self, text: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = String::new();
        let separators = ['。', '！', '？', '!', '?', '\n'];

        for c in text.chars() {
            current.push(c);
            if separators.contains(&c) {
                result.push(current.trim().to_string());
                current = String::new();
            }
        }
        if !current.trim().is_empty() {
            result.push(current.trim().to_string());
        }
        result
    }

    /// フルコンテキストラベルを解析して (音素, アクセント) を返す
    /// アクセント: 0=低, 1=高 (簡易的な実装)
    fn parse_label(&self, label: &str) -> Option<(String, i32)> {
        // 音素 (p3) の抽出: ^p2-p3+p4
        let p_start = label.find('-')? + 1;
        let p_end = label.find('+')?;
        let phoneme = label[p_start..p_end].to_string();

        if phoneme == "sil" || phoneme == "pau" {
            return Some((phoneme, 0));
        }

        // OpenJTalk ラベルの /F:f1_f2#f3... を解析
        // f1: アクセント句内でのモーラ位置 (1-indexed)
        // f3: アクセント核の位置 (0 は平板型)
        let f_part = label.split('/').find(|s| s.starts_with('F'))?;
        let f_content = &f_part[2..]; // "f1_f2#f3..."
        let pieces: Vec<&str> = f_content.split(&['_', '#', '@', '|']).collect();
        
        let f1: i32 = pieces.get(0)?.parse().ok()?;
        let f3: i32 = pieces.get(2)?.parse().ok()?;

        // 東京式アクセントの簡易基本ルール:
        // 1. 平板型 (f3==0): 1回目低、2回目以降高
        // 2. 頭高型 (f3==1): 1回目高、2回目以降低
        // 3. 中高・尾高型 (f3>1): 1回目低、2回目からf3回目まで高、以降低
        let accent = if f3 == 0 {
            if f1 == 1 { 0 } else { 1 }
        } else if f3 == 1 {
            if f1 == 1 { 1 } else { 0 }
        } else {
            if f1 > 1 && f1 <= f3 { 1 } else { 0 }
        };

        Some((phoneme, accent))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_with_accents() -> Result<()> {
        let analyzer = Analyzer::new()?;
        let text = "こんにちは！";
        let results = analyzer.analyze(text)?;
        
        assert_eq!(results.len(), 1);
        assert!(results[0].text.contains("こんにちは！"));
        
        // 音素が抽出されていること
        assert!(!results[0].phonemes.is_empty());
        // アクセント情報が含まれていること（すべて 0 ではない可能性がある）
        // ※「こんにちは」の場合、通常はアクセント核があるため 1 が含まれるはず
        let has_accent = results[0].accents.iter().any(|&a| a == 1);
        assert!(has_accent, "Accent information should contain some 1s for 'こんにちは'");
        
        Ok(())
    }

    #[test]
    fn test_split_sentences() -> Result<()> {
        let analyzer = Analyzer::new()?;
        let text = "こんにちは！元気ですか？今日は、いい天気ですね。";
        let results = analyzer.analyze(text)?;
        
        assert!(results.len() >= 3);
        assert!(results[0].text.contains("こんにちは！"));
        assert!(results[1].text.contains("元気ですか？"));
        assert!(results[2].text.contains("今日は、いい天気ですね。"));
        Ok(())
    }
}
