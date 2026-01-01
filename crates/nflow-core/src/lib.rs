use serde::{Deserialize, Serialize};

/// 1文の解析情報を表す構造体 (NDJSON スキーマ)
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Sentence {
    /// 原文
    pub text: String,
    /// 音素列
    pub phonemes: Vec<String>,
    /// アクセント情報のリスト (0: 低, 1: 高 など、実装に合わせて定義予定)
    pub accents: Vec<i32>,
    /// 話者・スタイルID
    pub style_id: u32,
    /// 対応する音声ファイル名 (学習時のみ使用)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wav_filename: Option<String>,
}

impl Sentence {
    pub fn new(text: String, phonemes: Vec<String>, accents: Vec<i32>, style_id: u32) -> Self {
        Self {
            text,
            phonemes,
            accents,
            style_id,
            wav_filename: None,
        }
    }

    pub fn with_wav(mut self, filename: String) -> Self {
        self.wav_filename = Some(filename);
        self
    }
}

/// 音素とIDの相互変換を行うトークナイザー
pub struct PhonemeTokenizer {
    phoneme_to_id: std::collections::HashMap<String, u32>,
    id_to_phoneme: Vec<String>,
}

impl PhonemeTokenizer {
    /// デフォルトの日本語音素セットで初期化
    pub fn new() -> Self {
        let mut phonemes = vec![
            "pad".to_string(), "sil".to_string(), "pau".to_string(),
            "a".to_string(), "i".to_string(), "u".to_string(), "e".to_string(), "o".to_string(),
            "ka".to_string(), "ki".to_string(), "ku".to_string(), "ke".to_string(), "ko".to_string(),
            "sa".to_string(), "shi".to_string(), "su".to_string(), "se".to_string(), "so".to_string(),
            "ta".to_string(), "chi".to_string(), "tsu".to_string(), "te".to_string(), "to".to_string(),
            "na".to_string(), "ni".to_string(), "nu".to_string(), "ne".to_string(), "no".to_string(),
            "ha".to_string(), "hi".to_string(), "fu".to_string(), "he".to_string(), "ho".to_string(),
            "ma".to_string(), "mi".to_string(), "mu".to_string(), "me".to_string(), "mo".to_string(),
            "ya".to_string(), "yu".to_string(), "yo".to_string(),
            "ra".to_string(), "ri".to_string(), "ru".to_string(), "re".to_string(), "ro".to_string(),
            "wa".to_string(), "wo".to_string(), "n".to_string(),
            // 子音・特殊音素
            "k".to_string(), "s".to_string(), "t".to_string(), "n".to_string(), "h".to_string(),
            "m".to_string(), "y".to_string(), "r".to_string(), "w".to_string(), "g".to_string(),
            "z".to_string(), "d".to_string(), "b".to_string(), "p".to_string(), "ky".to_string(),
            "gy".to_string(), "sh".to_string(), "ch".to_string(), "py".to_string(), "by".to_string(),
            "ny".to_string(), "hy".to_string(), "my".to_string(), "ry".to_string(), "ts".to_string(),
            "f".to_string(), "v".to_string(), "N".to_string(), "cl".to_string(),
        ];
        // 重複を除去してソート
        phonemes.sort();
        phonemes.dedup();

        let mut phoneme_to_id = std::collections::HashMap::new();
        for (i, p) in phonemes.iter().enumerate() {
            phoneme_to_id.insert(p.clone(), i as u32);
        }

        Self {
            phoneme_to_id,
            id_to_phoneme: phonemes,
        }
    }

    pub fn encode(&self, phonemes: &[String]) -> Vec<u32> {
        phonemes.iter()
            .map(|p| *self.phoneme_to_id.get(p).unwrap_or(&0))
            .collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_phoneme.len()
    }
}

impl Default for PhonemeTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_serialization() {
        let sentence = Sentence::new(
            "テスト。".to_string(),
            vec!["t".to_string(), "e".to_string(), "s".to_string(), "u".to_string(), "t".to_string(), "o".to_string()],
            vec![0, 1, 0, 0, 0, 0],
            0,
        );

        // シリアライズ (JSON)
        let json = serde_json::to_string(&sentence).unwrap();
        
        // デシリアライズ
        let deserialized: Sentence = serde_json::from_str(&json).unwrap();

        assert_eq!(sentence, deserialized);
        assert!(json.contains("\"text\":\"テスト。\""));
    }
}
