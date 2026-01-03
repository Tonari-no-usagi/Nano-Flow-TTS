use anyhow::Result;
use candle_core::{Device, Tensor};
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::Arc;

pub struct GriffinLim {
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    n_mels: usize,
    sample_rate: f32,
    mel_filter_inv: Tensor,
}

impl GriffinLim {
    pub fn new(n_fft: usize, hop_length: usize, win_length: usize, n_mels: usize, sample_rate: f32, device: &Device) -> Result<Self> {
        // メルスケーリング用の逆行列を近似的に作成
        let mel_filter = create_mel_filter(n_fft, n_mels, sample_rate, device);
        // 疑似逆行列 (A^T A)^-1 A^T は重いので、一旦簡易的に転置で近似するか、
        // あるいは個別のチャンネルの重みを正規化したものを使用
        let mel_filter_inv = mel_filter.t()?; 
        
        Ok(Self {
            n_fft,
            hop_length,
            win_length,
            n_mels,
            sample_rate,
            mel_filter_inv,
        })
    }

    /// メルスペクトログラム [frames, 80] を波形に変換
    pub fn decode(&self, mel: &Tensor, iterations: usize) -> Result<Vec<f32>> {
        // 1. Inverse Normalization -> Log-Mel -> Linear Magnitude
        // mel: [1, seq_len, 80] -> [seq_len, 80]
        let mel = mel.squeeze(0)?;
        // 逆正規化: (x - 1.0) * 5.0
        let mel = mel.affine(5.0, -5.0)?; 
        let mag = mel.exp()?;
        let linear_mag = mag.matmul(&self.mel_filter_inv)?; // [seq_len, 513]
        let linear_mag_vec = linear_mag.to_vec2::<f32>()?;
        
        let frames = linear_mag_vec.len();
        let bins = self.n_fft / 2 + 1;
        
        // 2. Griffin-Lim
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.n_fft);
        let ifft = planner.plan_fft_inverse(self.n_fft);
        
        let window: Vec<f32> = (0..self.win_length)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (self.win_length - 1) as f32).cos()))
            .collect();

        // 初期位相はランダム
        let mut rng = rand::rng();
        use rand::Rng;
        let mut stft = vec![vec![Complex::default(); bins]; frames];
        for f in 0..frames {
            for b in 0..bins {
                let angle = rng.random_range(0.0..2.0 * std::f32::consts::PI);
                stft[f][b] = Complex::from_polar(linear_mag_vec[f][b], angle);
            }
        }

        let mut waveform = Vec::new();
        for _ in 0..iterations {
            // ISTFT
            waveform = self.istft(&stft, frames);
            
            // STFT (Phase update)
            if iterations > 1 {
                stft = self.stft(&waveform, &linear_mag_vec, &fft, &window);
            }
        }

        Ok(waveform)
    }

    fn istft(&self, stft: &[Vec<Complex<f32>>], frames: usize) -> Vec<f32> {
        let mut waveform = vec![0.0; frames * self.hop_length + self.win_length];
        let mut window_sum = vec![0.0; waveform.len()];
        
        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(self.n_fft);
        
        let window: Vec<f32> = (0..self.win_length)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (self.win_length - 1) as f32).cos()))
            .collect();

        for i in 0..frames {
            let mut full_frame = vec![Complex::default(); self.n_fft];
            for j in 0..(self.n_fft / 2 + 1) {
                full_frame[j] = stft[i][j];
                if j > 0 && j < self.n_fft / 2 {
                    full_frame[self.n_fft - j] = stft[i][j].conj();
                }
            }
            
            ifft.process(&mut full_frame);
            
            let start = i * self.hop_length;
            for j in 0..self.win_length {
                if start + j < waveform.len() {
                    waveform[start + j] += full_frame[j].re * window[j] / self.n_fft as f32;
                    window_sum[start + j] += window[j] * window[j];
                }
            }
        }

        for i in 0..waveform.len() {
            if window_sum[i] > 1e-10 {
                waveform[i] /= window_sum[i];
            }
        }
        
        waveform
    }

    fn stft(&self, waveform: &[f32], target_mag: &[Vec<f32>], fft: &Arc<dyn rustfft::Fft<f32>>, window: &[f32]) -> Vec<Vec<Complex<f32>>> {
        let frames = target_mag.len();
        let bins = self.n_fft / 2 + 1;
        let mut stft = vec![vec![Complex::default(); bins]; frames];

        for i in 0..frames {
            let start = i * self.hop_length;
            let mut input = vec![Complex::default(); self.n_fft];
            for j in 0..self.win_length {
                if start + j < waveform.len() {
                    input[j] = Complex::new(waveform[start + j] * window[j], 0.0);
                }
            }
            fft.process(&mut input);
            
            for j in 0..bins {
                let mag = input[j].norm();
                if mag > 1e-10 {
                    stft[i][j] = input[j] * (target_mag[i][j] / mag);
                } else {
                    stft[i][j] = Complex::from_polar(target_mag[i][j], 0.0);
                }
            }
        }
        stft
    }
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
