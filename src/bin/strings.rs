use anyhow::{Context, Error, Result};
use ysc_utils::ysc::YSCScript;

fn main() {
    if let Err(err) = start() {
        println!("Error: {err}")
    }
}

fn start() -> Result<(), Error> {
    let args = std::env::args().skip(1).collect::<Vec<String>>();

    if args.is_empty() || args.len() != 1 {
        println!("Usage: strings %ysc_script%");
        return Ok(());
    }
    let script =
        YSCScript::from_ysc_file(&args[0]).context("Failed to read/parse/disassemble ysc file")?;

    for (_i, string) in script.strings.iter() {
        println!("{string}");
    }

    Ok(())
}
