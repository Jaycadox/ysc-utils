use anyhow::{Context, Error, Result};
use onig::*;
use ysc_utils::disassemble::*;

fn extract_numbers(input: &str) -> Vec<u32> {
    let re = Regex::new(r"(_(\d+))|(\*(\d+))").unwrap();
    let mut numbers = Vec::new();

    for capture in re.captures_iter(input) {
        for sc in capture.iter() {
            if let Some(number) = sc {
                if let Ok(num) = number.to_string().parse() {
                    numbers.push(num);
                }
            }
        }
    }
    numbers
}

fn main() {
    match run() {
        Ok(()) => {}
        Err(error) => {
            println!("Error: {error}");
        }
    }
}

fn run() -> Result<(), Error> {
    let mut args = std::env::args().skip(1).collect::<Vec<String>>();

    if args.is_empty() || args.len() != 3 {
        println!("Usage: global_updater %old_ysc_script% %new_ysc_script% %tokens%");
        println!("Example: global_updater freemode_old.ysc.full freemode_new.ysc.full \"Global_262145[pLocal /*123*/].f_456\"");
        return Ok(());
    }

    let old_script_path = args.remove(0);
    let new_script_path = args.remove(0);
    let input = extract_numbers(&args.remove(0));

    let global_reference = GlobalReference::new(input[0], (&input[1..]).to_vec());

    let old_script = DisassembledScript::from_ysc_file(old_script_path)
        .context("Failed to parse old ysc file")?;
    let new_script = DisassembledScript::from_ysc_file(new_script_path)
        .context("Failed to parse new ysc file")?;

    let res = old_script
        .find_global_reference(&global_reference)
        .context("Failed to find tokens in old script")?;

    let sig = old_script.generate_signature(res);
    let result = new_script
        .find_from_signature(&sig)
        .context("Failed to find tokens in new script")?;

    println!("{}", new_script.get_pretty(result.0));
    if result.1 {
        println!("High confidence.")
    } else {
        println!("Low confidence.")
    }

    Ok(())
}