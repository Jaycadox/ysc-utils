use crate::disassemble::{DisassembledScript, GlobalReference};
use anyhow::{Context, Error, Result};
use onig::*;
use std::collections::HashMap;
use std::path::Path;

fn extract_numbers(input: &str) -> Vec<u32> {
    let mut input = input.to_string();
    if !input.starts_with("Global") {
        input = format!("Global_{input}");
    }

    let re = Regex::new(r"(_(\d+))|(\*(\d+))").unwrap();
    let mut numbers = Vec::new();

    for capture in re.captures_iter(&input) {
        for number in capture.iter().flatten() {
            if let Ok(num) = number.to_string().parse() {
                numbers.push(num);
            }
        }
    }
    numbers
}

pub fn update_global(
    old_script_path: impl AsRef<Path>,
    new_script_path: impl AsRef<Path>,
    input: &str,
) -> Result<String, Error> {
    let indexes = extract_numbers(input);
    let global_reference = GlobalReference::new(indexes[0], indexes[1..].to_vec());

    let old_script = DisassembledScript::from_ysc_file(old_script_path)
        .context("Failed to parse old ysc file")?;
    let new_script = DisassembledScript::from_ysc_file(new_script_path)
        .context("Failed to parse new ysc file")?;

    let res = old_script
        .find_global_references(&global_reference)
        .context("Failed to find tokens in old script")?;

    let sig = old_script.generate_signatures(&res);
    let results = new_script
        .find_from_signatures(&sig)
        .context("Failed to find tokens in new script")?;

    let candidates = results
        .iter()
        .map(|f| new_script.get_pretty(f.0))
        .collect::<Vec<_>>();
    let final_candidate = find_most_occurring_string(&candidates)
        .context("Could not find most occurring candidate")?;
    Ok(final_candidate.to_string())
}

fn find_most_occurring_string(strings: &[String]) -> Option<&String> {
    strings
        .iter()
        .filter(|f| f.starts_with("Global_"))
        .fold(HashMap::new(), |mut occurrences, string| {
            *occurrences.entry(string).or_insert(0) += 1;
            occurrences
        })
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(&string, _)| string)
}
