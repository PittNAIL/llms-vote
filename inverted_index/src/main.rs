use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};

use clap::Parser;
use indicatif::ParallelProgressIterator;
use log::info;
use rayon::prelude::*;
use serde::Serialize;

const RARE_DISEASE_COUNT: usize = 6_304;
const DISCHARGE_MIMIC_IV_COUNT: usize = 331_794;

type NoteID = String;
type NoteText = String;
type RareDisease = String;

#[derive(Parser)]
struct Cli {
    /// The path to the file containing rare diseases
    rare_disease_data: std::path::PathBuf,
    /// The path to MIMIC-IV discharge summaries
    mimic_iv_discharge_data: std::path::PathBuf,
    /// The write path for the inverted index
    out_json_path: std::path::PathBuf,
}

// NOTE: Needed for `rayon`
#[derive(Clone, Serialize)]
struct Holder {
    term: RareDisease,
    note_ids: HashSet<NoteID>,
}

fn main() -> Result<(), anyhow::Error> {
    env_logger::init();

    let args = Cli::parse();

    info!("Loading rare diseases...");
    let file = File::open(args.rare_disease_data)?;
    let buf = BufReader::new(file);
    let mut rare_diseases: Vec<Holder> = buf
        .lines()
        .map(|l| l.expect("Could not parse line"))
        .map(|l| Holder {
            term: l.to_lowercase(),
            note_ids: HashSet::new(),
        })
        .collect();
    assert_eq!(rare_diseases.len(), RARE_DISEASE_COUNT);
    info!("Loading finished!");

    info!("Loading and parsing MIMIC-IV discharge CSV data...");
    let csv_reader = csv::Reader::from_path(args.mimic_iv_discharge_data)?;
    let mut mimic_data: Vec<(NoteID, NoteText)> = Vec::new();
    for result in csv_reader.into_deserialize() {
        let record: HashMap<String, String> = result?;
        mimic_data.push((record["note_id"].to_lowercase(), record["text"].to_owned()))
    }
    assert_eq!(mimic_data.len(), DISCHARGE_MIMIC_IV_COUNT);
    info!("Loading and parsing finished!");

    info!("Populating inverted index...");
    let len = rare_diseases.len() as u64;
    rare_diseases
        .par_iter_mut()
        .progress_count(len)
        .for_each(|entry| {
            for (note_id, text) in &mimic_data {
                if text.contains(&entry.term) {
                    (*entry).note_ids.insert(note_id.to_owned());
                }
            }
        });

    let mut filtered_rare_diseases: Vec<Holder> = Vec::new();
    filtered_rare_diseases.par_extend(
        rare_diseases
            .into_par_iter()
            .progress_count(len)
            .filter(|entry| !entry.note_ids.is_empty()),
    );
    info!("Inverted index has been populated!");

    info!("Writing to JSON...");
    let json_file = File::create(args.out_json_path)?;
    serde_json::to_writer_pretty(&json_file, &filtered_rare_diseases)?;
    info!("Finished writing to JSON!");

    Ok(())
}
