use crate::ysc::YSC;

mod ysc;

fn main() {
    let script_path = std::env::args().nth(1).unwrap();
    println!("{script_path}");
    let src = std::fs::read(script_path).unwrap();
    let mut ysc = YSC::new(&src).unwrap();
    println!("{:#?}", ysc.header);
}