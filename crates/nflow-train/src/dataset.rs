use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use nflow_core::{PhonemeTokenizer, Sentence};
use rustfft::{num_complex::Complex, FftPlanner};
use std::path::Path;

/// 音声データとメタデータのペアを管理するデータセット
pub struct Dataset {
    items: Vec<DatasetItem>,
    tokenizer: PhonemeTokenizer,
    device: Device,
    mel_filter: Tensor,
}



#[derive(Clone)]
struct DatasetItem {
    sentence: Sentence,
    wav_path: std::path::PathBuf,
    phoneme_len: usize,
}

impl Dataset {
    pub fn load_from_dir<P: AsRef<Path>>(dir: P, tokenizer: PhonemeTokenizer, device: &Device, max_frames: Option<usize>) -> Result<Self> {
        let mut items = Vec::new();
        // datasets/*.jsonl を探す
        let pattern = dir.as_ref().join("*.jsonl");
        for entry in glob::glob(pattern.to_str().context("Invalid glob pattern")?)? {
            let path = entry?;
            let content = std::fs::read_to_string(&path)?;
            for line in content.lines() {
                if line.trim().is_empty() { continue; }
                let sentence: Sentence = serde_json::from_str(line)?;
                // meta.jsonl に wav_filename があればそれを使用、なければ以前と同様に text.wav を探す
                let wav_path = if let Some(ref filename) = sentence.wav_filename {
                    let direct_path = dir.as_ref().join(filename);
                    let wavs_path = dir.as_ref().join("wavs").join(filename);
                    if wavs_path.exists() {
                        wavs_path
                    } else {
                        direct_path
                    }
                } else {
                    dir.as_ref().join(format!("{}.wav", sentence.text))
                };
                let phoneme_len = sentence.phonemes.len();
                
                // If max_frames is set, we need to check the mel length.
                // However, we don't want to compute mel for all items during loading.
                // If cache exists, we can check it. If not, we might have to skip this check or do it lazily.
                // For now, let's assume we filter by phoneme_len as a heuristic, 
                // but a better way is to filter during batch making or collate if we don't have mel length.
                // Wait, if we have preprocessed mel, we can check its size.
                
                let mut skip = false;
                if let Some(max_f) = max_frames {
                    let mel_cache_path = wav_path.with_extension("wav.mel.safetensors");
                    if mel_cache_path.exists() {
                        // Check cache metadata without loading full tensor if possible, 
                        // but safetensors load is relatively fast for just header.
                        if let Ok(tensors) = candle_core::safetensors::load(&mel_cache_path, device) {
                            if let Some(mel) = tensors.get("mel") {
                                if mel.dim(0).unwrap_or(0) > max_f {
                                    skip = true;
                                }
                            }
                        }
                    }
                }

                if !skip {
                    items.push(DatasetItem { sentence, wav_path, phoneme_len });
                }
            }
        }
        let mel_filter = Self::create_mel_filter(1024, 80, 22050.0, device);
        Ok(Self { items, tokenizer, device: device.clone(), mel_filter })
    }

    /// バッチのインデックスリストを作成（Lazy Loading用）
    pub fn make_batches(&self, batch_size: usize) -> Vec<Vec<usize>> {
        let mut indices: Vec<usize> = (0..self.items.len()).collect();
        // Sort by phoneme length for bucketing
        indices.sort_by_key(|&i| self.items[i].phoneme_len);

        indices.chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// 指定されたインデックスのバッチを実際に読み込んで処理する（Lazy Loading）
    pub fn collate_batch(&self, indices: &[usize]) -> Result<(Tensor, Tensor, Tensor, Vec<usize>)> {
        let mut phonemes_vec = Vec::new();
        let mut accents_vec = Vec::new();
        let mut mels_vec = Vec::new();
        let mut lengths = Vec::new();

        for &idx in indices {
            let item = &self.items[idx];
            
            // Try to load cached mel spectrogram
            let mel_cache_path = item.wav_path.with_extension("wav.mel.safetensors");
            let mel = if mel_cache_path.exists() {
                let tensors = candle_core::safetensors::load(&mel_cache_path, &self.device)?;
                tensors.get("mel").cloned().context("Missing 'mel' tensor in cache")?
            } else {
                self.compute_mel(&item.wav_path)?
            };

            let p_ids = self.tokenizer.encode(&item.sentence.phonemes);
            
            let a_ids: Vec<u32> = item.sentence.accents.iter().map(|&a| a as u32).collect();
            
            phonemes_vec.push(Tensor::from_vec(p_ids, (item.phoneme_len,), &self.device)?);
            accents_vec.push(Tensor::from_vec(a_ids, (item.phoneme_len,), &self.device)?);
            mels_vec.push(mel);
            lengths.push(item.phoneme_len);
        }

        let padded_phonemes = self.pad_sequence(&phonemes_vec, 0)?;
        let padded_accents = self.pad_sequence(&accents_vec, 0)?;
        let padded_mels = self.pad_mels(&mels_vec)?;

        // Tensors are already contiguous from padding methods, so we can stack directly
        let phonemes = Tensor::stack(&padded_phonemes, 0)?;
        let accents = Tensor::stack(&padded_accents, 0)?;
        let mels = Tensor::stack(&padded_mels, 0)?;

        Ok((phonemes, accents, mels, lengths))
    }

    fn pad_sequence(&self, tensors: &[Tensor], pad_val: u32) -> Result<Vec<Tensor>> {
        if tensors.is_empty() { return Ok(Vec::new()); }
        let max_len = tensors.iter()
            .map(|t| t.dim(0))
            .collect::<candle_core::error::Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        let mut padded = Vec::new();
        for t in tensors {
            let len = t.dim(0)?;
            if len < max_len {
                let diff = max_len - len;
                let pad = Tensor::full(pad_val, (diff,), &self.device)?;
                padded.push(Tensor::cat(&[t, &pad], 0)?.contiguous()?);
            } else {
                padded.push(t.clone());
            }
        }
        Ok(padded)
    }

    fn pad_mels(&self, tensors: &[Tensor]) -> Result<Vec<Tensor>> {
        if tensors.is_empty() { return Ok(Vec::new()); }
        let max_len = tensors.iter()
            .map(|t| t.dim(0))
            .collect::<candle_core::error::Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        let mut padded = Vec::new(); // 
        for t in tensors {
            let len = t.dim(0)?;
            if len < max_len {
                let diff = max_len - len;
                // Pad with specific value? usually good to pad with min value or 0 if normalized.
                // Pad with specific value.
                // After fixed normalization, 0.0 might represent silence or mean.
                // Let's use a value that represents negative infinity in log-mel, 
                // but for training stability, a small value like -2.0 (representing log-mel -10) is often used.
                // For now, 0.0 is a reasonable default.
                let pad = Tensor::full(-2.0f32, (diff, 80), &self.device)?;
                padded.push(Tensor::cat(&[t, &pad], 0)?.contiguous()?);
            } else {
                padded.push(t.clone());
            }
        }
        Ok(padded)
    }

    /// (Legacy) Single item getter - kept for compatibility if needed, but not used in batched training
    pub fn get_item(&self, index: usize) -> Result<(Tensor, Tensor, Tensor)> {
        let _item = &self.items[index];
        // Use collate_batch with single index
        let (phonemes, accents, mels, _lengths) = self.collate_batch(&[index])?;
        Ok((phonemes, accents, mels))
    }

    pub fn compute_mel(&self, wav_path: &Path) -> Result<Tensor> {
        let mut reader = hound::WavReader::open(wav_path)
            .with_context(|| format!("Failed to open WAV file at {:?}", wav_path))?;
        let spec = reader.spec();
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
                reader.samples::<i32>().map(|s| s.unwrap() as f32 / max_val).collect()
            }
            hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        };

        // STFT パラメータ (22050Hz 想定)
        let n_fft = 1024;
        let hop_length = 256;
        let win_length = 1024;
        
        let stft = self.stft(&samples, n_fft, hop_length, win_length)?;
        
        // [frames, bin] * [bin, 80] -> [frames, 80]
        let mel = stft.matmul(&self.mel_filter)?;
        
        // ログスケール変換
        let mel = (mel + 1e-5)?.log()?;

        // 固定正規化 (Global Scaling)
        // インスタンス正規化は音量の絶対値を消してしまうため、TTSでは避けるべきです。
        // ここでは log10(magnitude) に近い値になるようスケールを調整します。
        // log(1e-5) は約 -11.5 なので、+5 して 5 で割ることで、おおよそ -1.3 〜 1.0 の範囲に収めます。
        let mel = (mel.affine(1.0 / 5.0, 1.0))?; 
        
        Ok(mel)
    }

    fn stft(&self, samples: &[f32], n_fft: usize, hop_length: usize, win_length: usize) -> Result<Tensor> {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);
        
        // Hann窓の作成
        let window: Vec<f32> = (0..win_length)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (win_length - 1) as f32).cos()))
            .collect();

        let mut frames = Vec::new();
        let num_frames = (samples.len() as f32 / hop_length as f32).floor() as usize;

        for i in 0..num_frames {
            let start = i * hop_length;
            let mut input: Vec<Complex<f32>> = vec![Complex::default(); n_fft];
            for j in 0..win_length {
                if start + j < samples.len() {
                    let w = window[j];
                    input[j] = Complex::new(samples[start + j] * w, 0.0);
                }
            }
            fft.process(&mut input);
            
            let mut mag_frame = Vec::new();
            for j in 0..(n_fft / 2 + 1) {
                mag_frame.push(input[j].norm());
            }
            frames.push(mag_frame);
        }

        let flat_frames: Vec<f32> = frames.into_iter().flatten().collect();
        Tensor::from_vec(flat_frames, (num_frames, n_fft / 2 + 1), &self.device).map_err(Into::into)
    }

    fn create_mel_filter(n_fft: usize, n_mels: usize, sample_rate: f32, device: &Device) -> Tensor {
        let f_min = 0.0;
        let f_max = sample_rate / 2.0;
        
        let mel_min = 1127.0f32 * (1.0f32 + f_min / 700.0f32).ln();
        let mel_max = 1127.0f32 * (1.0f32 + f_max / 700.0f32).ln();
        
        let mut mel_points = Vec::new();
        for i in 0..(n_mels + 2) {
            let m = mel_min + (mel_max - mel_min) * (i as f32 / (n_mels + 1) as f32);
            let f = 700.0f32 * ((m / 1127.0f32).exp() - 1.0f32);
            mel_points.push((n_fft as f32 + 1.0) * f / sample_rate);
        }

        let mut filter = vec![0f32; (n_fft / 2 + 1) * n_mels];
        for m in 0..n_mels {
            let f_prev = mel_points[m];
            let f_curr = mel_points[m + 1];
            let f_next = mel_points[m + 2];
            
            for f in 0..(n_fft / 2 + 1) {
                let f_hz = f as f32;
                let weight = if f_hz > f_prev && f_hz < f_curr {
                    (f_hz - f_prev) / (f_curr - f_prev)
                } else if f_hz >= f_curr && f_hz < f_next {
                    (f_next - f_hz) / (f_next - f_curr)
                } else {
                    0.0
                };
                filter[f * n_mels + m] = weight;
            }
        }

        Tensor::from_vec(filter, (n_fft / 2 + 1, n_mels), device).unwrap()
    }
    
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn get_wav_path(&self, index: usize) -> Result<std::path::PathBuf> {
        Ok(self.items.get(index).context("Index out of bounds")?.wav_path.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nflow_core::PhonemeTokenizer;
    use std::collections::HashMap;

    #[test]
    fn test_compute_mel_normalization() -> Result<()> {
        let wav_path = std::env::temp_dir().join("test_norm.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 22050,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&wav_path, spec)?;
        for i in 0..22050 {
            let t = i as f32 / 22050.0;
            let amp = (t * 440.0 * 2.0 * std::f32::consts::PI).sin();
            writer.write_sample((amp * 30000.0) as i16)?;
        }
        writer.finalize()?;

        // PhonemeTokenizerの初期化（テスト用）
        let tokenizer = PhonemeTokenizer::new();
        let device = Device::Cpu;
        let ds = Dataset {
             items: vec![],
             tokenizer,
             device: device.clone(),
             mel_filter: Dataset::create_mel_filter(1024, 80, 22050.0, &device),
        };

        let mel = ds.compute_mel(&wav_path)?;
        
        let mean = mel.mean_all()?;
        let std = (mel.broadcast_sub(&mean)?.sqr()?.mean_all()? + 1e-5)?.sqrt()?;
        
        let m = mean.to_scalar::<f32>()?;
        let s = std.to_scalar::<f32>()?;
        
        println!("Mean: {}, Std: {}", m, s);
        
        // 平均がほぼ0、分散がほぼ1であることを確認
        assert!(m.abs() < 1e-3, "Mean should be close to 0, got {}", m);
        assert!((s - 1.0).abs() < 1e-3, "Std should be close to 1, got {}", s);

        // クリーンアップ
        let _ = std::fs::remove_file(wav_path);
        
        Ok(())
    }

    #[test]
    fn test_batch_padding() -> Result<()> {
        let device = Device::Cpu;
        let tokenizer = PhonemeTokenizer::new();
        let ds = Dataset {
             items: vec![], // dummy
             tokenizer,
             device: device.clone(),
             mel_filter: Dataset::create_mel_filter(1024, 80, 22050.0, &device),
        };

        // Create dummy tensors of different lengths
        let t1 = Tensor::ones((10, 80), DType::F32, &device)?;
        let t2 = Tensor::ones((5, 80), DType::F32, &device)?;
        let t3 = Tensor::ones((12, 80), DType::F32, &device)?;

        let padded = ds.pad_mels(&[t1, t2, t3])?;
        
        // Max len should be 12
        assert_eq!(padded.len(), 3);
        assert_eq!(padded[0].dims(), &[12, 80]);
        assert_eq!(padded[1].dims(), &[12, 80]);
        assert_eq!(padded[2].dims(), &[12, 80]);

        // Validate padding value (0.0)
        let v = padded[1].get(6)?.get(0)?.to_scalar::<f32>()?;
        assert_eq!(v, 0.0);

        // Verify all padded tensors are contiguous
        for (i, t) in padded.iter().enumerate() {
            assert!(t.is_contiguous(), "Padded tensor {} should be contiguous", i);
        }

        // Verify stacking works without errors
        let stacked = Tensor::stack(&padded, 0)?;
        assert!(stacked.is_contiguous(), "Stacked tensor should be contiguous");

        Ok(())
    }

    #[test]
    fn test_mel_cache_loading() -> Result<()> {
        let device = Device::Cpu;
        let tokenizer = PhonemeTokenizer::new();
        
        // Create a temporary wav file match
        let temp_dir = tempfile::tempdir()?;
        let wav_path = temp_dir.path().join("test.wav");
        let mel_cache_path = temp_dir.path().join("test.wav.mel.safetensors");
        
        // Create dummy mel tensor and save as safetensors
        let dummy_mel = Tensor::ones((10, 80), DType::F32, &device)?;
        let mut map = HashMap::new();
        map.insert("mel".to_string(), dummy_mel.clone());
        candle_core::safetensors::save(&map, &mel_cache_path)?;
        
        // Create dummy WAV (empty header is fine for exists() check, but ComputeMel might fail if we actually call it)
        std::fs::write(&wav_path, vec![0; 44])?;

        let ds = Dataset {
            items: vec![DatasetItem {
                sentence: Sentence { 
                    text: "test".to_string(), 
                    phonemes: vec!["a".to_string()], 
                    accents: vec![0],
                    style_id: 0,
                    wav_filename: Some("test.wav".to_string()),
                },
                wav_path: wav_path.clone(),
                phoneme_len: 1,
            }],
            tokenizer,
            device: device.clone(),
            mel_filter: Dataset::create_mel_filter(1024, 80, 22050.0, &device),
        };
        
        // Load batch
        let (_p, _a, mels, _l) = ds.collate_batch(&[0])?;
        
        // Verify loaded mel matches dummy_mel (within padding if any)
        // collate_batch returns [batch, max_len, 80]
        let loaded_mel = mels.get(0)?;
        assert_eq!(loaded_mel.dims(), &[10, 80]);
        
        // Check content
        let val = loaded_mel.get(0)?.get(0)?.to_scalar::<f32>()?;
        assert_eq!(val, 1.0);
        Ok(())
    }

    #[test]
    fn test_load_with_max_frames() -> Result<()> {
        let device = Device::Cpu;
        let tokenizer = PhonemeTokenizer::new();
        let temp_dir = tempfile::tempdir()?;
        
        // 1. 小さいメル（5フレーム）
        let _wav1 = temp_dir.path().join("small.wav");
        let mel1_path = temp_dir.path().join("small.wav.mel.safetensors");
        let mel1 = Tensor::ones((5, 80), DType::F32, &device)?;
        let mut map1 = HashMap::new();
        map1.insert("mel".to_string(), mel1);
        candle_core::safetensors::save(&map1, &mel1_path)?;
        
        // 2. 大きいメル（20フレーム）
        let _wav2 = temp_dir.path().join("large.wav");
        let mel2_path = temp_dir.path().join("large.wav.mel.safetensors");
        let mel2 = Tensor::ones((20, 80), DType::F32, &device)?;
        let mut map2 = HashMap::new();
        map2.insert("mel".to_string(), mel2);
        candle_core::safetensors::save(&map2, &mel2_path)?;

        // JSONL作成
        let jsonl_path = temp_dir.path().join("test.jsonl");
        let content = format!(
            "{}\n{}",
            serde_json::to_string(&Sentence { text: "small".to_string(), phonemes: vec!["a".to_string()], accents: vec![0], style_id: 0, wav_filename: Some("small.wav".to_string()) })?,
            serde_json::to_string(&Sentence { text: "large".to_string(), phonemes: vec!["a".to_string()], accents: vec![0], style_id: 0, wav_filename: Some("large.wav".to_string()) })?
        );
        std::fs::write(&jsonl_path, content)?;

        // max_frames = 10 でロード -> small のみが残るはず
        let ds = Dataset::load_from_dir(temp_dir.path(), tokenizer, &device, Some(10))?;
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.items[0].sentence.text, "small");

        Ok(())
    }
}
