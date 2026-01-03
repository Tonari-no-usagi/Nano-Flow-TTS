pub mod vocoder;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use nflow_core::{PhonemeTokenizer, Sentence};
use nflow_model::NanoFlowModel;
use vocoder::GriffinLim;

/// 音声合成エンジン
pub struct Synthesizer {
    device: Device,
    model: NanoFlowModel,
    tokenizer: PhonemeTokenizer,
    vocoder: GriffinLim,
}

impl Synthesizer {
    pub fn new() -> Result<Self> {
        let device = Device::Cpu;
        let tokenizer = PhonemeTokenizer::new();
        let model_path = std::path::Path::new("models/checkpoint.safetensors");

        let vm = if model_path.exists() {
            eprintln!("Loading model from {:?}", model_path);
            match candle_core::safetensors::load(model_path, &device) {
                Ok(tensors) => VarBuilder::from_tensors(tensors, DType::F32, &device),
                Err(e) => {
                    eprintln!("Error loading model: {:?}", e);
                    eprintln!("--- IMPORTANT ---");
                    eprintln!("Architecture has changed. Please delete 'models/checkpoint.safetensors'");
                    eprintln!("and start training from scratch (without --resume).");
                    eprintln!("-----------------");
                    return Err(e.into());
                }
            }
        } else {
            eprintln!("No model found. Initializing with random weights.");
            VarBuilder::from_varmap(&candle_nn::VarMap::new(), DType::F32, &device)
        };

        let model = NanoFlowModel::new(
            tokenizer.vocab_size(),
            256, // enc_dim
            12,  // enc_layers
            80,  // dec_output_dim
            12,  // dec_layers
            vm,
        )?;

        let vocoder = GriffinLim::new(
            1024,   // n_fft
            256,    // hop_length
            1024,   // win_length
            80,     // n_mels
            22050.0,// sample_rate
            &device,
        )?;

        Ok(Self { device, model, tokenizer, vocoder })
    }

    /// 1文の解析データから音声波形を生成する
    pub fn synthesize(&self, sentence: &Sentence) -> Result<Vec<f32>> {
        let seq_len = sentence.phonemes.len();
        if seq_len == 0 { return Ok(vec![]); }

        // トークナイズ: String -> ID
        let phoneme_ids = self.tokenizer.encode(&sentence.phonemes);
        let accent_ids: Vec<u32> = sentence.accents.iter().map(|&a| a as u32).collect();

        let frames_per_phoneme = 12; // 1音素あたり約100ms弱 (256 hop_length * 12 / 22050Hz)
        let total_frames = seq_len * frames_per_phoneme;
        
        // ターゲットデバイスでの生成（階層化 Flow により 4ステップ程度でも実用品質）
        let (p_expanded, a_expanded) = {
            let mut p_vec = Vec::with_capacity(total_frames);
            let mut a_vec = Vec::with_capacity(total_frames);
            for i in 0..total_frames {
                let idx = (i / frames_per_phoneme).min(seq_len - 1);
                p_vec.push(phoneme_ids[idx]);
                a_vec.push(accent_ids[idx]);
            }
            (
                Tensor::from_vec(p_vec, (1, total_frames), &self.device)?,
                Tensor::from_vec(a_vec, (1, total_frames), &self.device)?
            )
        };

        let steps = 25; // 4から25に引き上げ。RTX 2070なら十分高速に動作します。
        let output = self.model.generate(&p_expanded, &a_expanded, steps, &self.device)?;
        // output: [1, total_frames, 80]
        
        // Griffin-Lim による波形生成
        let waveform = self.vocoder.decode(&output, 32)?; // 32 iterations
        
        Ok(waveform)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesize_basic() -> Result<()> {
        let synth = Synthesizer::new()?;
        let sentence = Sentence::new(
            "テスト。".to_string(),
            vec!["t".to_string(), "e".to_string()],
            vec![0, 0],
            0,
        );
        
        let waveform = synth.synthesize(&sentence)?;
        assert!(waveform.len() > 0);
        Ok(())
    }
}
