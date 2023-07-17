use crate::ast::AstError;
use crate::disassemble::{InstructionList, GlobalReference, DisassembleError};
use regex::Regex;
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

/// High level errors for the library
#[derive(Error, Debug)]
pub enum YscUtilError {
    /// Error pertaining to AST
    #[error("ast error")]
    AstError(#[from] AstError),
    /// Error pertaining to disassembly
    #[error("disassemble error")]
    DisassembleError(#[from] DisassembleError),
    /// Error pertaining to this module
    #[error("shared error")]
    SharedError(#[from] SharedError),
}

/// Errors pertaining to this module
#[derive(Error, Debug)]
pub enum SharedError {
    /// Attempted to find global reference, but couldn't
    #[error("failed to find tokens in old script")]
    TokensNotFound,

    /// List of global canidates's mode could not be determined
    #[error("failed to find most occuring value out of signature hits")]
    ModeFailed,
}

fn extract_numbers(input: &str) -> Vec<u32> {
    let mut input = input.to_string();
    if !input.starts_with("Global") {
        input = format!("Global_{input}");
    }

    let re = Regex::new(r"(_(\d+))|(\*(\d+))").unwrap();
    let mut numbers = Vec::new();

    for capture in re.captures_iter(&input) {
        for number in capture.iter().flatten() {
            if let Ok(num) = number.as_str().to_string().parse() {
                numbers.push(num);
            }
        }
    }
    numbers
}

/// Attempt to find a new version of a global (and optional offsets/array indexes) given the path to a new and old script
pub fn update_global(
    old_script_path: impl AsRef<Path>,
    new_script_path: impl AsRef<Path>,
    input: &str,
) -> Result<String, YscUtilError> {
    let indexes = extract_numbers(input);
    let global_reference = GlobalReference::new(indexes[0], indexes[1..].to_vec());

    let old_script = InstructionList::from_ysc_file(old_script_path)?;
    let new_script = InstructionList::from_ysc_file(new_script_path)?;

    let res = old_script
        .find_global_references(&global_reference)
        .ok_or(SharedError::TokensNotFound)?;

    let sig = old_script.generate_signatures(&res);
    let results = new_script
        .find_from_signatures(&sig)
        .ok_or(SharedError::TokensNotFound)?;

    let candidates = results
        .iter()
        .map(|f| new_script.get_pretty(f.0))
        .collect::<Vec<_>>();
    let final_candidate = find_most_occurring_string(&candidates).ok_or(SharedError::ModeFailed)?;
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
