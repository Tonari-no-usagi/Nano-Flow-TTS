use anyhow::Result;
use candle_core::Device;
use clap::Parser;
use nflow_core::PhonemeTokenizer;
use std::collections::HashMap;
use crate::dataset::Dataset;

#[derive(Parser, Debug)]
pub struct PreprocessArgs {
    /// データセットディレクトリ
    #[arg(long, default_value = "datasets")]
    pub dir: std::path::PathBuf,

    /// 明示的にCPUを使用する
    #[arg(long)]
    pub cpu: bool,

    /// 既に存在するファイルを上書きする
    #[arg(long)]
    pub overwrite: bool,
}

pub fn run(args: PreprocessArgs) -> Result<()> {
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0).unwrap_or_else(|_| {
            println!("Warning: CUDA device not found, falling back to CPU for preprocessing.");
            Device::Cpu
        })
    };

    let tokenizer = PhonemeTokenizer::new();
    println!("Loading dataset from {:?}...", args.dir);
    let dataset = Dataset::load_from_dir(&args.dir, tokenizer, &device, None)?;
    
    let total = dataset.len();
    println!("Found {} items. Starting mel-spectrogram pre-computation...", total);

    // Datasetの中身にアクセスするために暫定的に公開されている情報を利用するか、
    // あるいはDatasetに反復処理用のメソッドを追加する。
    // ここでは colate_batch を1件ずつ呼ぶのが簡単。
    
    for i in 0..total {
        // インデックスからWAVパスを取得する必要がある。
        // Dataset::get_item は非効率（collate_batchを呼ぶ）なため、
        // Datasetの実装にWAVパスを取得するメソッドを追加するか、
        // 内部構造を辿る必要がある。
        // 今回はシンプルに Dataset に path を取得するメソッドを追加することにする。
        let wav_path = dataset.get_wav_path(i)?;
        let cache_path = wav_path.with_extension("wav.mel.safetensors");

        if cache_path.exists() && !args.overwrite {
            if i % 100 == 0 {
                print!("\rSkipping {}/{} (already exists)...", i + 1, total);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            continue;
        }

        let mel = dataset.compute_mel(&wav_path)?;
        
        let mut map = HashMap::new();
        map.insert("mel".to_string(), mel);
        candle_core::safetensors::save(&map, &cache_path)?;

        if (i + 1) % 10 == 0 || i + 1 == total {
            print!("\rProcessed {}/{}...", i + 1, total);
            std::io::Write::flush(&mut std::io::stdout())?;
        }
    }

    println!("\nPre-computation completed.");
    Ok(())
}
