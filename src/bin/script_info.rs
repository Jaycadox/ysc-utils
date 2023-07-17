use anyhow::{Error, Result};
use ysc_utils::ysc::*;

fn main() {
    match run() {
        Ok(()) => {}
        Err(error) => {
            println!("Error: {error}");
        }
    }
}

fn run() -> Result<(), Error> {
    let args = std::env::args().skip(1).collect::<Vec<String>>();

    if args.is_empty() || args.len() != 1 {
        println!("Usage: script_info %ysc_script%");
        println!("Example: script_info freemode.ysc.full");
        return Ok(());
    }

    let ysc = YSCScript::from_ysc_file(&args[0])?;

    println!("Script name: {}", ysc.name);
    println!("String table size: {}", ysc.strings.len());
    println!("Code size (bytes): {}", ysc.code.len());
    println!("Native table size: {}", ysc.native_table.len());
    println!("No. code tables: {}", ysc.code_table_offsets.len());
    println!("No. string tables: {}", ysc.string_table_offsets.len());

    Ok(())
}
