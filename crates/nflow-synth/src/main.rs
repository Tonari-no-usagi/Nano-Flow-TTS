use std::io::{self, BufRead, BufWriter, Write};
use anyhow::Result;
use clap::Parser;
use nflow_core::Sentence;
use nflow_synth::Synthesizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// 出力フォーマット (f32, i16)
    #[arg(short, long, default_value = "f32")]
    format: String,

    /// WAVファイルとして保存する場合のパス (省略時は stdout に raw PCM)
    #[arg(short, long)]
    output: Option<String>,

    /// サンプリングレート
    #[arg(short, long, default_value_t = 22050)]
    sample_rate: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let synth = Synthesizer::new()?;
    let stdin = io::stdin();

    // WAVライターの初期化（パスが指定された場合）
    let mut wav_writer = if let Some(path) = &args.output {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: args.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        Some(hound::WavWriter::create(path, spec)?)
    } else {
        None
    };

    let mut stdout = BufWriter::new(io::stdout());

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }

        // BOM (Byte Order Mark) を除去する (Windows PowerShell対策)
        let line = line.trim_start_matches('\u{feff}');

        let sentence: Sentence = serde_json::from_str(&line)?;
        let waveform = synth.synthesize(&sentence)?;

        if let Some(ref mut writer) = wav_writer {
            // WAVファイルに書き込み (i16固定)
            for &sample in &waveform {
                let s = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                writer.write_sample(s)?;
            }
        } else {
            // stdout に Raw PCM 出力
            match args.format.as_str() {
                "i16" => {
                    let pcm_i16: Vec<i16> = waveform
                        .iter()
                        .map(|&x| (x.clamp(-1.0, 1.0) * 32767.0) as i16)
                        .collect();
                    let bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            pcm_i16.as_ptr() as *const u8,
                            pcm_i16.len() * std::mem::size_of::<i16>(),
                        )
                    };
                    stdout.write_all(bytes)?;
                }
                _ => {
                    let bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            waveform.as_ptr() as *const u8,
                            waveform.len() * std::mem::size_of::<f32>(),
                        )
                    };
                    stdout.write_all(bytes)?;
                }
            }
            stdout.flush()?;
        }
    }

    if let Some(writer) = wav_writer {
        writer.finalize()?;
    }

    Ok(())
}
