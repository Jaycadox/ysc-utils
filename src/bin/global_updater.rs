use ysc_utils::shared;

fn main() {
    let mut args = std::env::args().skip(1).collect::<Vec<String>>();

    if args.is_empty() || args.len() != 3 {
        println!("Usage: global_updater %old_ysc_script% %new_ysc_script% %tokens%");
        println!("Example: global_updater freemode_old.ysc.full freemode_new.ysc.full \"Global_262145[pLocal /*123*/].f_456\"");
        return;
    }

    let old_script_path = args.remove(0);
    let new_script_path = args.remove(0);
    let input = args.remove(0);

    match shared::update_global(old_script_path, new_script_path, &input) {
        Ok(str) => {
            println!("{str}");
        }
        Err(error) => {
            println!("Error: {error}");
        }
    }
}
