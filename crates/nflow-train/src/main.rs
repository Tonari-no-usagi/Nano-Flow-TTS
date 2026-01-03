use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use clap::{Parser, Subcommand};
use nflow_core::PhonemeTokenizer;
use nflow_model::NanoFlowModel;

mod dataset;
mod prepare;
mod preprocess;

use dataset::Dataset;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// モデルの学習を実行します
    Train(TrainArgs),
    /// データセットの前処理（CSV -> JSONL変換）を実行します
    Prepare(prepare::PrepareArgs),
    /// データセットの事前計算（メルスペクトログラム生成）を実行します
    Preprocess(preprocess::PreprocessArgs),
}

#[derive(Parser)]
pub struct TrainArgs {
    /// 学習エポック数
    #[arg(long, default_value_t = 100)]
    pub epochs: usize,

    /// 既存のチェックポイントから再開する
    #[arg(long)]
    pub resume: bool,

    /// 学習時間の制限（分単位、0は無制限）
    #[arg(long, default_value_t = 0)]
    pub limit_mins: u64,

    /// 明示的にCPUを使用する
    #[arg(long)]
    pub cpu: bool,

    /// 進捗（Loss）を表示するステップの間隔
    #[arg(long, default_value_t = 100)]
    pub log_interval: usize,

    /// モデルを保存するステップの間隔
    #[arg(long, default_value_t = 500)]
    pub save_interval: usize,

    /// バッチサイズ
    #[arg(long, default_value_t = 16)]
    pub batch_size: usize,

    /// 学習に使用する最大フレーム数（メルスペクトログラムの長さ）
    #[arg(long, default_value_t = 1500)]
    pub max_frames: usize,

    /// 学習率 (Max)
    #[arg(long, default_value_t = 2e-4)]
    pub learning_rate: f64,

    /// モデルの次元 (Encoder/Decoder)
    #[arg(long, default_value_t = 256)]
    pub hidden_dim: usize,

    /// エンコーダーのレイヤー数
    #[arg(long, default_value_t = 12)]
    pub enc_layers: usize,

    /// デコーダーのレイヤー数
    #[arg(long, default_value_t = 12)]
    pub dec_layers: usize,

    /// tのサンプリング最小値 (0.0 - 1.0)
    #[arg(long, default_value_t = 0.0)]
    pub t_min: f32,

    /// tのサンプリング最大値 (0.0 - 1.0)
    #[arg(long, default_value_t = 1.0)]
    pub t_max: f32,
}

struct LRScheduler {
    warmup_steps: usize,
    max_steps: usize,
    max_lr: f64,
}

impl LRScheduler {
    fn new(warmup_steps: usize, max_steps: usize, max_lr: f64) -> Self {
        Self { warmup_steps, max_steps, max_lr }
    }

    fn get_lr(&self, step: usize) -> f64 {
        let min_lr = 1e-5; // 学習が完全に止まらないように最小値を設定
        if step < self.warmup_steps {
            (self.max_lr * (step as f64 / self.warmup_steps as f64)).max(min_lr)
        } else {
            let progress = (step - self.warmup_steps) as f64 / (self.max_steps - self.warmup_steps) as f64;
            let cos_val = 0.5 * (1.0 + (progress * std::f32::consts::PI as f64).cos());
            (self.max_lr * cos_val).max(min_lr)
        }
    }
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Train(args) => train(args),
        Commands::Prepare(args) => prepare::run(args),
        Commands::Preprocess(args) => preprocess::run(args),
    };

    if let Err(e) = result {
        eprintln!("\n[ERROR] Command failed with error:");
        eprintln!("{:?}", e);
        std::process::exit(1);
    }
}

fn train(args: TrainArgs) -> Result<()> {
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0).unwrap_or_else(|_| {
            println!("Warning: CUDA device not found, falling back to CPU.");
            Device::Cpu
        })
    };
    let start_time = std::time::Instant::now();
    let tokenizer = PhonemeTokenizer::new();

    // モデルパラメータの設定
    let vocab_size = tokenizer.vocab_size();
    let enc_dim = args.hidden_dim;
    let enc_layers = args.enc_layers;
    let dec_output_dim = 80;
    let dec_layers = args.dec_layers;

    // 変数管理とモデルの初期化
    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = NanoFlowModel::new(
        vocab_size,
        enc_dim,
        enc_layers,
        dec_output_dim,
        dec_layers,
        vs,
    )?;

    // モデルの保存パス
    let model_dir = std::path::Path::new("models");
    if !model_dir.exists() {
        std::fs::create_dir_all(model_dir)?;
    }
    let model_path = model_dir.join("checkpoint.safetensors");

    // チェックポイントのロード (Resume機能)
    if args.resume {
        if model_path.exists() {
            println!("Loading checkpoint from {:?}", model_path);
            varmap.load(&model_path)?;
        } else {
            println!("Warning: --resume specified but {:?} not found. Starting from scratch.", model_path);
        }
    }

    // データセットの読み込み
    let dataset = Dataset::load_from_dir("datasets", tokenizer, &device, Some(args.max_frames))?;
    if dataset.len() == 0 {
        println!("No data found in 'datasets' directory. Please place .wav and .jsonl files.");
        return Ok(());
    }

    let total_steps = args.epochs * (dataset.len() / args.batch_size + 1);
    let warmup_steps = total_steps / 10; // 10% warmup
    let scheduler = LRScheduler::new(warmup_steps, total_steps, args.learning_rate);
    
    // オプティマイザの初期化 (初期LRは0)
    let mut opt = candle_nn::AdamW::new_lr(varmap.all_vars(), 0.0)?;
    let mut global_step = 0;

    println!("Starting training loop with {} items (Total ~{} steps)...", dataset.len(), total_steps);
    
    for epoch in 1..=args.epochs {
        println!("\n--- Epoch {}/{} started ---", epoch, args.epochs);
        let mut total_loss = 0.0;
        let mut steps_in_epoch = 0;
        
        // インデックスバッチを作成（メモリ効率的）
        let batch_indices = dataset.make_batches(args.batch_size);
        let num_batches = batch_indices.len();

        for (i, indices) in batch_indices.iter().enumerate() {
            global_step += 1;
            
            // Scheduling
            let lr = scheduler.get_lr(global_step);
            opt.set_learning_rate(lr);

            // Lazy Loading: このバッチのデータだけを読み込む
            let (phonemes, accents, mels, lengths) = dataset.collate_batch(indices)
                .with_context(|| format!("Failed to load batch at Step {}", global_step))?;
            
            if mels.dim(0)? == 0 || mels.dim(1)? == 0 {
                println!("Warning: Skipping empty batch at Step {}", global_step);
                continue;
            }
            steps_in_epoch += 1;
            
            let b_sz = phonemes.dim(0)?;
            let max_mel_len = mels.dim(1)?;
            let mut aligned_phonemes_vec = Vec::new();
            let mut aligned_accents_vec = Vec::new();
            
            let p_cpu = phonemes.to_device(&Device::Cpu)?;
            let a_cpu = accents.to_device(&Device::Cpu)?;
            
            for b in 0..b_sz {
                let p_len = lengths[b]; // original phoneme length
                
                let cur_p = p_cpu.get(b)?;
                let cur_a = a_cpu.get(b)?;
                
                let mut p_indices = Vec::with_capacity(max_mel_len);
                let mut a_indices = Vec::with_capacity(max_mel_len);
                
                // 準動的アライメント: 境界線を少しぼかす
                // Jitter (揺らぎ) を加えることで、モデルに前後関係の自由度を与える
                let jitter_range = 0.5f32; // 前後0.5音素分くらいぼかす

                for j in 0..max_mel_len {
                    let base_idx = j as f32 * p_len as f32 / max_mel_len as f32;
                    // 乱数による揺らぎ
                    let offset: f32 = (rand::random::<f32>() - 0.5) * jitter_range;
                    let idx = (base_idx + offset).max(0.0) as usize;
                    let idx = idx.min(p_len - 1);
                    
                    p_indices.push(cur_p.get(idx)?.to_scalar::<u32>()?);
                    a_indices.push(cur_a.get(idx)?.to_scalar::<u32>()?);
                }
                aligned_phonemes_vec.push(Tensor::from_vec(p_indices, (max_mel_len,), &device)?);
                aligned_accents_vec.push(Tensor::from_vec(a_indices, (max_mel_len,), &device)?);
            }
            
            // Tensor::from_vec creates contiguous tensors, so we can stack directly
            let phonemes_aligned = Tensor::stack(&aligned_phonemes_vec, 0)?;
            let accents_aligned = Tensor::stack(&aligned_accents_vec, 0)?;
            
            let mel_batch = &mels; // [batch, max_len, 80]

            // Flow Matching
            let x_0 = Tensor::randn(0f32, 1f32, mel_batch.shape(), &device)?;
            
            // tのサンプリング範囲を制限
            let t_val: f32 = if args.t_min >= args.t_max {
                args.t_min.clamp(0.0, 1.0)
            } else {
                let r: f32 = rand::random();
                args.t_min + r * (args.t_max - args.t_min)
            };
            let t = Tensor::from_slice(&[t_val], (1,), &device)?;
            
            let x_t = {
                let t1 = (mel_batch.affine(t_val as f64, 0.0))?;
                let t2 = (x_0.affine((1.0 - t_val) as f64, 0.0))?;
                (t1 + t2)?.contiguous()?  // Ensure contiguity before passing to model
            };
            let target_v = (mel_batch.sub(&x_0))?;
            let (v_coarse, v_fine) = model.forward_train(&phonemes_aligned, &accents_aligned, &x_t, &t, &mel_batch)?;
            
            // ターゲットの分割: [batch, seq, 80] -> 16 + 64
            let target_coarse = target_v.narrow(2, 0, 16)?.contiguous()?;
            let target_fine = target_v.narrow(2, 16, 64)?.contiguous()?;

            let loss_coarse = candle_nn::loss::mse(&v_coarse, &target_coarse)?;
            let loss_fine = candle_nn::loss::mse(&v_fine, &target_fine)?;
            let loss = (loss_coarse + loss_fine)?;
            
            opt.backward_step(&loss)?;
            let loss_val = loss.to_scalar::<f32>()?;
            total_loss += loss_val;

            if global_step % args.log_interval == 0 {
                let elapsed = start_time.elapsed();
                println!(
                    "Epoch: {}, Step: {}/{}, LR: {:.2e}, Loss: {:.6}, Elapsed: {}s",
                    epoch,
                    i + 1,
                    num_batches,
                    lr,
                    loss_val,
                    elapsed.as_secs()
                );
            }

            // 定期保存
            if global_step % args.save_interval == 0 {
                varmap.save(&model_path)?;
                println!("Step {} (Epoch {}): Intermediate checkpoint saved.", global_step, epoch);
            }

            // 時間制限
            if args.limit_mins > 0 {
                let elapsed = start_time.elapsed();
                if elapsed.as_secs() >= args.limit_mins * 60 {
                    println!("Time limit reached. Saving...");
                    varmap.save(&model_path)?;
                    return Ok(());
                }
            }
        }

        
        if steps_in_epoch > 0 {
            println!("\n--- Epoch {} completed. Avg Loss: {:.6} ---", epoch, total_loss / steps_in_epoch as f32);
        } else {
            println!("\n--- Epoch {} completed with no steps. ---", epoch);
        }
    }

    println!("Training with NanoFlowModel completed successfully.");

    // モデルの保存
    varmap.save(&model_path)?;
    println!("Model saved to {:?}", model_path);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_training_step() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = NanoFlowModel::new(100, 32, 1, 80, 1, vs)?;
        let mut opt = candle_nn::AdamW::new_lr(varmap.all_vars(), 1e-3)?;

        let b_sz = 2;
        let seq_len = 5;
        let phonemes = Tensor::zeros((b_sz, seq_len), DType::U32, &device)?;
        let accents = Tensor::zeros((b_sz, seq_len), DType::U32, &device)?;
        let x_t = Tensor::zeros((b_sz, seq_len, 80), DType::F32, &device)?;
        let t = Tensor::zeros((1,), DType::F32, &device)?;
        let mel_batch = Tensor::zeros((b_sz, seq_len, 80), DType::F32, &device)?;
        let target_v = Tensor::zeros((b_sz, seq_len, 80), DType::F32, &device)?;

        let (v_coarse, v_fine) = model.forward_train(&phonemes, &accents, &x_t, &t, &mel_batch)?;
        let target_coarse = target_v.narrow(2, 0, 16)?;
        let target_fine = target_v.narrow(2, 16, 64)?;
        
        let loss = (candle_nn::loss::mse(&v_coarse, &target_coarse)? + candle_nn::loss::mse(&v_fine, &target_fine)?)?;
        opt.backward_step(&loss)?;

        assert!(loss.to_scalar::<f32>()? >= 0.0);
        Ok(())
    }
}
