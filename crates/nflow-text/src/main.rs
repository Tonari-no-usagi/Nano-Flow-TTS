use std::io::{self, BufRead, Write};
use anyhow::Result;
use nflow_text::Analyzer;

fn main() -> Result<()> {
    let analyzer = Analyzer::new()?;
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let text = line?;
        if text.trim().is_empty() { continue; }
        
        let sentences = analyzer.analyze(&text)?;
        
        for sentence in sentences {
            let json = serde_json::to_string(&sentence).unwrap();
            // 明示的に stdout に書き込み
            writeln!(stdout, "{}", json)?;
            stdout.flush()?;
        }
    }
    Ok(())
}
