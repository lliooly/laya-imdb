//! Streaming original-tokenizer inference. Input text must already follow the frozen cleaning rule.
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    env,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    time::Instant,
};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Deserialize)]
struct Review {
    id: String,
    text: String,
    label: u8,
}

#[derive(Serialize)]
struct Encoded<'a> {
    id: &'a str,
    label: u8,
    ids: &'a [u32],
    mask: &'a [u32],
    offsets: &'a [(usize, usize)],
}

fn encode_batch(
    tokenizer: &Tokenizer,
    rows: &[Review],
    writer: &mut BufWriter<File>,
) -> Result<()> {
    // Rust's ordinary encode uses byte offsets. Python exposes Unicode character offsets.
    let encoded = tokenizer
        .encode_batch_char_offsets(rows.iter().map(|r| r.text.as_str()).collect(), true)?;
    for (row, encoding) in rows.iter().zip(&encoded) {
        serde_json::to_writer(
            &mut *writer,
            &Encoded {
                id: &row.id,
                label: row.label,
                ids: encoding.get_ids(),
                mask: encoding.get_attention_mask(),
                offsets: encoding.get_offsets(),
            },
        )?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() == 1 && args[0] == "--version" {
        println!("imdb-tokenizer 0.1.0 tokenizers=0.22.2 offsets=unicode-codepoints");
        return Ok(());
    }
    if !args.len().is_multiple_of(2) {
        return Err(
            "Expected --tokenizer PATH --input PATH --output PATH --length N [--batch-size N]"
                .into(),
        );
    }
    let mut options = HashMap::new();
    for pair in args.chunks(2) {
        if ![
            "--tokenizer",
            "--input",
            "--output",
            "--length",
            "--batch-size",
        ]
        .contains(&pair[0].as_str())
        {
            return Err(format!("Unknown option {}", pair[0]).into());
        }
        if options.insert(pair[0].as_str(), pair[1].as_str()).is_some() {
            return Err("Duplicate option".into());
        }
    }
    let required = |key| {
        options
            .get(key)
            .copied()
            .ok_or_else(|| format!("Missing {key}"))
    };
    let length: usize = required("--length")?.parse()?;
    let batch_size: usize = options.get("--batch-size").unwrap_or(&"256").parse()?;
    if length < 3 || batch_size == 0 {
        return Err("length >= 3 and batch-size > 0 required".into());
    }
    let began = Instant::now();
    let mut tokenizer = Tokenizer::from_file(required("--tokenizer")?)?;
    if tokenizer.token_to_id("[PAD]") != Some(50283) {
        return Err("Expected pinned ModernBERT pad ID 50283".into());
    }
    tokenizer.with_truncation(Some(TruncationParams {
        max_length: length,
        ..Default::default()
    }))?;
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(length),
        pad_id: 50283,
        pad_token: "[PAD]".into(),
        ..Default::default()
    }));
    let input = required("--input")?;
    let output = required("--output")?;
    if std::path::Path::new(input).canonicalize()?
        == std::path::Path::new(output)
            .canonicalize()
            .unwrap_or_default()
    {
        return Err("Input and output must differ".into());
    }
    let reader = BufReader::new(File::open(input)?);
    // Refuse overwrites; the Python wrapper owns atomic publication of cache generations.
    let mut writer = BufWriter::new(File::options().write(true).create_new(true).open(output)?);
    let mut rows = Vec::with_capacity(batch_size);
    let mut count = 0;
    for (line_number, line) in reader.lines().enumerate() {
        let row: Review =
            serde_json::from_str(&line?).map_err(|e| format!("Line {}: {e}", line_number + 1))?;
        if row.label > 1 {
            return Err(format!("Invalid label at line {}", line_number + 1).into());
        }
        rows.push(row);
        count += 1;
        if rows.len() == batch_size {
            encode_batch(&tokenizer, &rows, &mut writer)?;
            rows.clear();
        }
    }
    if !rows.is_empty() {
        encode_batch(&tokenizer, &rows, &mut writer)?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    eprintln!(
        "{}",
        serde_json::json!({"rows": count, "length": length,
        "elapsed_seconds": began.elapsed().as_secs_f64(), "offset_unit": "unicode_codepoints"})
    );
    Ok(())
}
