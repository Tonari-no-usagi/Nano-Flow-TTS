use anyhow::{Context, Result};
use clap::Parser;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::Deserialize;
use std::fs::{self, File};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    /// Output directory for the dataset
    #[arg(short, long, default_value = "datasets")]
    output: String,
}

#[derive(Deserialize, Debug)]
struct IndexEntry {
    id: String,
    archive_url: String,
    sizes: String,
}

// Archive.org Metadata API Structures
#[derive(Deserialize, Debug)]
struct ArchiveMetadataResponse {
    server: Option<String>,
    dir: Option<String>,
    files: Vec<ArchiveFile>,
}

#[derive(Deserialize, Debug)]
struct ArchiveFile {
    name: String,
    format: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let output_str = cli.output;
    let output_path = Path::new(&output_str);

    if !output_path.exists() {
        fs::create_dir_all(output_path)?;
    }
    
    // Canonicalize path for clearer output, if possible
    let abs_path = fs::canonicalize(output_path).unwrap_or(output_path.to_path_buf());
    println!("Starting Kokoro Speech Dataset downloader...");
    println!("Target Directory: {:?}", abs_path);

    // 1. Download Release Zip (Metadata)
    println!("Step 1/4: Downloading Metadata (Github)...");
    let client = Client::builder()
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        .build()?;

    let release_url = "https://github.com/kaiidams/Kokoro-Speech-Dataset/releases/download/1.3/kokoro-speech-v1_3.zip";
    let zip_data = download_file_to_memory(&client, release_url).await?;
    
    println!("Step 2/4: Extracting Metadata...");
    extract_zip_memory(&zip_data, &abs_path)?;

    // 2. Read index.json (Prioritize root, then search subdirs)
    let direct_index = abs_path.join("index.json");
    let index_path = if direct_index.exists() {
        direct_index
    } else {
        // Try to find it in subdirectories
        let mut found = None;
        if let Ok(entries) = fs::read_dir(&abs_path) {
            for entry in entries.flatten() {
                let p = entry.path().join("index.json");
                if p.exists() {
                    found = Some(p);
                    break;
                }
                // Also check data/index.json
                let p2 = entry.path().join("data").join("index.json");
                if p2.exists() {
                    found = Some(p2);
                    break;
                }
            }
        }
        // Fallback: check datasets/data/index.json directly
        if found.is_none() {
             let p = abs_path.join("data").join("index.json");
             if p.exists() { found = Some(p); }
        }
        
        found.ok_or_else(|| anyhow::anyhow!("index.json not found in extracted data. searched in {:?}", abs_path))?
    };
    
    println!("Found index.json at {:?}", index_path);
    // Adjusted root for data: where index.json is, or its parent?
    // If index.json is in datasets/index.json, then data is likely in datasets/
    // If index.json is in datasets/foo/data/index.json, then data root is datasets/foo/
    let data_root = if index_path.parent() == Some(&abs_path) {
        abs_path.clone()
    } else {
        // Assume standard structure: ROOT/data/index.json -> ROOT
        if index_path.parent().unwrap().ends_with("data") {
            index_path.parent().unwrap().parent().unwrap().to_path_buf()
        } else {
            // e.g. ROOT/index.json -> ROOT
            index_path.parent().unwrap().to_path_buf()
        }
    };

    let index_file = File::open(&index_path)?;
    let index: Vec<IndexEntry> = serde_json::from_reader(index_file)?;

    // 3. Download and Process Audio
    println!("Step 3/4: Downloading Audio Archives...");
    


    for entry in &index {
        let entry_dir = data_root.join("data").join(&entry.id);
        if !entry_dir.exists() {
             fs::create_dir_all(&entry_dir)?;
        }
        
        println!("Processing: {}", entry.id);
        println!(" - Attempting Zip download: {}", entry.archive_url);

        match download_file_to_memory(&client, &entry.archive_url).await {
            Ok(archive_data) => {
                println!(" - Extracting zip...");
                match extract_zip_memory(&archive_data, &data_root.join("data").join(&entry.id)) {
                    Ok(_) => println!(" - Zip extracted successfully."),
                    Err(e) => {
                         println!(" ! Zip extraction failed: {}. Trying fallback...", e);
                         download_individual_files(&client, &entry.id, &data_root.join("data").join(&entry.id), &entry.archive_url).await?;
                    }
                }
            },
            Err(e) => {
                println!(" ! Zip download failed: {}. Trying fallback (Individual Files)...", e);
                download_individual_files(&client, &entry.id, &data_root.join("data").join(&entry.id), &entry.archive_url).await?;
            }
        }
    }

    // 4. Extract Clips
    println!("Step 4/4: Cutting Audio Segments...");
    process_audio_cutting(&data_root, &index)?;

    println!("Done! Dataset is ready at {:?}", data_root);
    Ok(())
}

async fn download_individual_files(client: &Client, _kokoro_id: &str, target_dir: &Path, archive_url: &str) -> Result<()> {
    // Extract real Archive ID from URL
    // URL format: http://www.archive.org/download/<ID>/<FILE>
    // e.g. http://www.archive.org/download//meian_1403_librivox/meian_1403_librivox_64kb_mp3.zip
    
    let parts: Vec<&str> = archive_url.split("/download/").collect();
    if parts.len() < 2 {
        anyhow::bail!("Could not parse Archive ID from URL: {}", archive_url);
    }
    let after_download = parts[1];
    // Remove leading slashes if any (e.g. //meian...)
    let clean_path = after_download.trim_start_matches('/');
    // The ID is the first component
    let real_id = clean_path.split('/').next().unwrap_or("");
    
    if real_id.is_empty() {
        anyhow::bail!("Parsed empty Archive ID from URL: {}", archive_url);
    }

    // 1. Query Metadata
    let meta_url = format!("https://archive.org/metadata/{}", real_id);
    println!(" - Querying Metadata for ID '{}': {}", real_id, meta_url);
    
    let meta_res = client.get(&meta_url).send().await?;
    if !meta_res.status().is_success() {
        anyhow::bail!("Metadata query failed: {}", meta_res.status());
    }
    
    let meta: ArchiveMetadataResponse = meta_res.json().await?;
    
    let server = meta.server.ok_or_else(|| anyhow::anyhow!("Metadata missing 'server' field"))?;
    let dir = meta.dir.ok_or_else(|| anyhow::anyhow!("Metadata missing 'dir' field"))?;

    let base_url = format!("https://{}{}", server, dir); // e.g. https://ia600108.us.archive.org/22/items/id
    
    println!(" - Found server: {}", base_url);

    // 2. Filter MP3 files
    let mp3_files: Vec<&ArchiveFile> = meta.files.iter().filter(|f| {
        if let Some(fmt) = &f.format {
            // Prefer 64Kbps MP3 as in original zip, or VBR MP3
            fmt.contains("MP3")
        } else {
            false
        }
    }).collect();
    
    if mp3_files.is_empty() {
        anyhow::bail!("No MP3 files found in metadata for {}", real_id);
    }

    println!(" - Found {} MP3 files. Downloading...", mp3_files.len());
    
    for file in mp3_files {
        let file_url = format!("{}/{}", base_url, file.name);
         // handle special chars in url if needed, but mostly safe
        let file_path = target_dir.join(&file.name);
        
        if file_path.exists() {
            println!("   - Skipping existing: {}", file.name);
            continue;
        }

        println!("   - Downloading: {}", file.name);
        // Simple download
        let mut resp = client.get(&file_url).send().await?;
        if !resp.status().is_success() {
            println!("     ! Failed to download {}: {}", file.name, resp.status());
            continue; 
        }
        
        let mut out = File::create(&file_path)?;
        while let Some(chunk) = resp.chunk().await? {
            out.write_all(&chunk)?;
        }
    }

    Ok(())
}

async fn download_file_to_memory(client: &Client, url: &str) -> Result<Vec<u8>> {
    // Sanitize URL: force https and remove double slashes
    let url = url.replace("http://", "https://").replace(".org/download//", ".org/download/");
    
    // println!("Requesting: {}", url);
    let res = client.get(&url).send().await?;
    
    let status = res.status();
    if !status.is_success() {
        anyhow::bail!("Status: {}", status);
    }
    
    let headers = res.headers().clone();
    if let Some(ct) = headers.get(reqwest::header::CONTENT_TYPE) {
        if ct.to_str().unwrap_or("").contains("text/html") {
             anyhow::bail!("Download returned HTML instead of binary.");
        }
    }

    let total_size = res.content_length().unwrap_or(0);
    
    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
        .progress_chars("#>-"));

    let mut stream = res.bytes_stream();
    let mut data = Vec::with_capacity(total_size as usize);

    while let Some(item) = stream.next().await {
        let chunk = item?;
        data.write_all(&chunk)?;
        pb.inc(chunk.len() as u64);
    }
    pb.finish_with_message("Downloaded");
    
    if data.len() < 100 {
        if let Ok(s) = std::str::from_utf8(&data) {
             if s.contains("DOCTYPE") || s.contains("html") {
                 anyhow::bail!("Downloaded content looks like HTML error page");
             }
        }
    }

    Ok(data)
}

fn extract_zip_memory(data: &[u8], target_dir: &Path) -> Result<()> {
    let reader = Cursor::new(data);
    let mut zip = zip::ZipArchive::new(reader)?;

    for i in 0..zip.len() {
        let mut file = zip.by_index(i)?;
        let mut file_name = file.name().to_string();
        
        // Sanitize: some zip files coming from macos might have weird paths or __MACOSX
        if file_name.contains("__MACOSX") || file_name.starts_with(".") { continue; }

        let outpath = target_dir.join(&file_name);

        if file.name().ends_with('/') {
            fs::create_dir_all(&outpath)?;
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(p)?;
                }
            }
            let mut outfile = File::create(&outpath)?;
            std::io::copy(&mut file, &mut outfile)?;
        }
    }
    Ok(())
}

fn process_audio_cutting(base_dir: &Path, index: &[IndexEntry]) -> Result<()> {
    let data_dir = base_dir.join("data");
    let wavs_dir = base_dir.join("wavs");
    fs::create_dir_all(&wavs_dir)?;

    let mut csv_file = File::create(base_dir.join("metadata.csv"))?;

    for entry in index {
        let id_ = &entry.id;
        let mut metadata_path = data_dir.join(format!("{}.metadata.txt", id_));
        if !metadata_path.exists() {
            // Try parent dir if not in data/
            if let Some(p) = data_dir.parent() {
                metadata_path = p.join(format!("{}.metadata.txt", id_));
            }
        }
        let audio_dir = data_dir.join(id_);

        if !metadata_path.exists() {
            println!("Metadata file not found: {:?}", metadata_path);
            continue;
        }

        let content = fs::read_to_string(&metadata_path)?;
        
        let mut current_audio_file = String::new();
        let mut current_samples: Vec<f32> = Vec::new();
        let mut current_sample_rate = 0;

        for line in content.lines() {
            let parts: Vec<&str> = line.split('|').collect();
            if parts.len() < 6 { continue; }
            
            let seg_id = parts[0];
            let audio_file_name = parts[1];
            let start: usize = parts[2].parse()?;
            let end: usize = parts[3].parse()?;
            let text = parts[4];
            let reading = parts[5];

            if audio_file_name != current_audio_file {
                 let audio_path = audio_dir.join(audio_file_name);
                 
                 // Fallback strategies for filename
                 let load_target = if audio_path.exists() {
                     Some(audio_path)
                 } else {
                     // Try .mp3 replacement
                     let mp3_name = audio_file_name.replace(".wav", ".mp3");
                     let mp3_path = audio_dir.join(&mp3_name);
                     
                     if mp3_path.exists() {
                         Some(mp3_path)
                     } else {
                         // Try searching recursively or fuzzy match if needed?
                         // For now assume standard structure
                         None
                     }
                 };

                 if let Some(path) = load_target {
                     // println!("Loading audio: {:?}", path);
                     // only print if new file
                     if let Ok((samples, sr)) = decode_audio(&path) {
                         current_samples = samples;
                         current_sample_rate = sr;
                         current_audio_file = audio_file_name.to_string();
                     } else {
                         println!("Failed to decode: {:?}", path);
                         continue;
                     }
                 } else {
                     println!("Audio file missing: {} (looked in {:?})", audio_file_name, audio_dir);
                     continue; 
                 }
            }

            if end > current_samples.len() {
                continue;
            }
            
            let segment = &current_samples[start..end];
            let out_wav_path = wavs_dir.join(format!("{}.wav", seg_id));
            save_wav(&out_wav_path, segment, current_sample_rate)?;

            writeln!(csv_file, "{}|{}|{}", seg_id, text, reading)?;
        }
    }
    
    Ok(())
}

fn decode_audio(path: &Path) -> Result<(Vec<f32>, u32)> {
    let src = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension() {
        if let Some(ext_str) = ext.to_str() {
             hint.with_extension(ext_str);
        }
    }

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)
        .context(format!("Failed to probe format for {:?}", path))?;
    let mut format = probed.format;
    let track = format.default_track().context("No default track")?;
    
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    
    let track_id = track.id;
    let mut samples: Vec<f32> = Vec::new();
    let mut sample_rate = 0;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(_)) => break, // End of stream
            Err(e) => return Err(e.into()),
        };

        if packet.track_id() != track_id { continue; }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                if sample_rate == 0 {
                    sample_rate = decoded.spec().rate;
                }
                
                let channels = decoded.spec().channels.count();
                let mut sample_buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
                sample_buf.copy_interleaved_ref(decoded);
                
                let buf_samples = sample_buf.samples();
                
                if channels == 1 {
                    samples.extend_from_slice(buf_samples);
                } else {
                    for i in (0..buf_samples.len()).step_by(channels) {
                        samples.push(buf_samples[i]);
                    }
                }
            }
            Err(symphonia::core::errors::Error::IoError(_)) => break,
            Err(e) => return Err(e.into()),
        }
    }

    Ok((samples, sample_rate))
}

fn save_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    
    let mut writer = hound::WavWriter::create(path, spec)?;
    let max_val = i16::MAX as f32;
    
    for &sample in samples {
        let amp = (sample * max_val).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        writer.write_sample(amp)?;
    }
    
    writer.finalize()?;
    Ok(())
}
