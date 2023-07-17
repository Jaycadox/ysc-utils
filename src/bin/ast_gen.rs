use ysc_utils::{ast::AstGenerator, ysc::YSCScript};

fn main() {
    let mut args = std::env::args().skip(1).collect::<Vec<String>>();

    if args.is_empty() {
        println!("Usage    : ast_gen %ysc_script% %function number/index%");
        println!("Example  : ast_gen freemode.ysc.full");
        println!("Example 2: ast_gen freemode.ysc.full func_305");
        return;
    }

    let function_index: Option<i32> = if args.len() == 2 {
        args.pop()
            .map(|func| func.replace("func_", "").parse().unwrap())
    } else {
        None
    };

    let script = YSCScript::from_ysc_file(&args.pop().expect("No script file in input"))
        .unwrap();

    let ast_gen = AstGenerator::try_from(script).unwrap();
    match function_index {
        Some(index) => {
            println!(
                "{}",
                match ast_gen.generate_function(index as usize) {
                    Ok(res) => format!("{}", res),
                    Err(e) => format!("Error: {e}"),
                }
            );
        }
        _ => {
            let mut num_pass = 0;
            println!("Starting...");
            let then = std::time::Instant::now();
            let funcs = (0..ast_gen.get_functions().len()).collect::<Vec<_>>();
            for _ in 0..1 {
                funcs.iter().for_each(|i| {
                    if let Ok(_func) = ast_gen.generate_function(*i) {
                        num_pass += 1;
                    }
                });
            }

            let now = std::time::Instant::now();
            let time = now.duration_since(then).as_millis();
            println!(
                "Result: {num_pass}/{} in {}ms ({}ms/func)",
                ast_gen.get_functions().len(),
                time,
                time as f32 / num_pass as f32
            );
        }
    }
}
