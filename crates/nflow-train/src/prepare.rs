use anyhow::{Context, Result};
use clap::Parser;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use nflow_text::Analyzer;

#[derive(Parser, Debug)]
pub struct PrepareArgs {
    /// 入力メタデータファイル (CSV/TSV)
    #[arg(long)]
    input: PathBuf,

    /// 出力ファイル (NDJSON)
    #[arg(long, default_value = "datasets/train.jsonl")]
    output: PathBuf,

    /// 区切り文字 (comma, tab, pipe)
    #[arg(long, default_value = "pipe")]
    delimiter: String,

    /// テキストが含まれるカラムのインデックス (0始まり)
    #[arg(long, default_value_t = 1)]
    text_col: usize,

    /// IDが含まれるカラムのインデックス (0始まり)
    #[arg(long, default_value_t = 0)]
    id_col: usize,

    /// ヘッダーがあるかどうか
    #[arg(long)]
    header: bool,
}

pub fn run(args: PrepareArgs) -> Result<()> {
    println!("Preparing dataset from {:?}...", args.input);
    
    // Analyzerの初期化 (辞書のロードなどはここで行われる)
    let analyzer = Analyzer::new().context("Failed to initialize nflow-text Analyzer")?;

    // CSVリーダーの設定
    let delimiter_byte = match args.delimiter.as_str() {
        "comma" => b',',
        "tab" => b'\t',
        "pipe" => b'|',
        _ => return Err(anyhow::anyhow!("Unsupported delimiter. Use 'comma', 'tab', or 'pipe'.")),
    };

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(delimiter_byte)
        .has_headers(args.header)
        .flexible(true) // 行ごとのカラム数が異なっても許容（簡易的）
        .from_path(&args.input)?;

    // 出力ファイルの準備
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(&args.output)?);

    let mut count = 0;
    for result in rdr.records() {
        let record = result?;
        
        let id = record.get(args.id_col)
            .context(format!("Missing ID column at index {}", args.id_col))?;
        let text = record.get(args.text_col)
            .context(format!("Missing text column at index {}", args.text_col))?;

        // テキスト解析
        // Analyzerは複数の文を返す可能性があるが、学習データセットとしては1行1文を想定することが多い。
        // ここでは、入力テキスト全体を解析し、出てきた全てのSentenceに対して処理を行う。
        // ただし、IDは元のファイルのものを使いたいので、複数文に分割された場合は _1, _2 などを付与するか、
        // あるいは Kokoro のように1行1文が保証されていることを期待するか。
        // ここではシンプルに、分割された場合はIDにサフィックスをつける戦略をとる。

        if let Some(sentence) = analyzer.analyze_single(text)? {
            let wav_filename = format!("{}.wav", id);
            let sentence = sentence.with_wav(wav_filename);

            let json = serde_json::to_string(&sentence)?;
            writeln!(writer, "{}", json)?;
            count += 1;
        }

        if count % 100 == 0 {
            print!("\rProcessed {} sentences...", count);
            std::io::stdout().flush()?;
        }
    }

    println!("\nDone. Saved {} sentences to {:?}", count, args.output);
    Ok(())
}
