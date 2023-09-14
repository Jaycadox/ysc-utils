use anyhow::{Error, Result};
use std::fmt::Formatter;
use std::io::Write;
use ysc_utils::disassemble::{Disassembler, Instruction};
use ysc_utils::ysc::YSCScript;

fn main() {
    if let Err(err) = start() {
        println!("Error: {err}")
    }
}

fn start() -> Result<(), Error> {
    let mut args = std::env::args().skip(1).collect::<Vec<String>>();
    let old_format = !args.is_empty() && args[0] == "--old";
    if old_format {
        args.remove(0);
    }

    if args.is_empty() || args.len() > 2 {
        println!("Usage: basic_disassemble %ysc_script% (optional: function number/index)");
        println!("Example  : basic_disasemble freemode.ysc.full");
        println!("Example 2: basic_disasemble --old freemode.ysc.full func_305");
        return Ok(());
    }

    let specified_function_index_str = args.get(1);
    let mut func_index: Option<usize> = None;

    let script =
        YSCScript::from_ysc_file(&args[0])?;

    let mut wr = Box::new(std::io::BufWriter::new(std::io::stdout()));

    if let Some(index) = specified_function_index_str {
        let val: usize = index.replace("func_", "").parse()?;
        func_index = Some(val + 1);
    }
    let mut disasm = Disassembler::new(&script);
    disasm.old_format = old_format;

    let script = disasm.disassemble(func_index)?;

    for inst in script.instructions {
        let mut padding = "".to_owned();

        if !matches!(inst, Instruction::Enter { .. }) {
            padding = "    ".to_owned();
        }

        let dop = DisassembledOpcode::from(inst);
        let out = format!("{padding}{dop}\n");
        let _ = wr.write(out.as_ref())?;
    }

    wr.flush()?;

    Ok(())
}

struct DisassembledOpcode {
    name: &'static str,
    operands: Option<Vec<i128>>,
    comment: Option<String>,
}

impl From<Instruction> for DisassembledOpcode {
    fn from(value: Instruction) -> Self {
        Self {
            name: DisassembledOpcode::get_name(&value),
            operands: DisassembledOpcode::get_operands(&value),
            comment: DisassembledOpcode::get_comment(&value),
        }
    }
}

impl DisassembledOpcode {
    fn get_comment(op: &Instruction) -> Option<String> {
        match op {
            Instruction::String { value, .. } => Some(value.to_string()),
            Instruction::Enter { index, .. } if index.is_some() => {
                Some(format!("func_{}", index.unwrap()))
            }
            Instruction::Call { func_index, .. } if func_index.is_some() => {
                Some(format!("func_{}", func_index.unwrap()))
            }
            Instruction::Native { native_hash, .. } => Some(format!("0x{:0X}", native_hash)),
            _ => None,
        }
    }

    fn get_operands(op: &Instruction) -> Option<Vec<i128>> {
        match op {
            Instruction::PushConstU8 { one } => Some(vec![*one as i128]),
            Instruction::PushConstU8U8 { one, two } => Some(vec![*one as i128, *two as i128]),
            Instruction::PushConstU8U8U8 { two, one, three } => {
                Some(vec![*one as i128, *two as i128, *three as i128])
            }
            Instruction::PushConstU32 { one } => Some(vec![*one as i128]),
            Instruction::PushConstF { one } => Some(vec![*one as i128]),
            Instruction::Native {
                native_table_index,
                num_args,
                num_returns,
                ..
            } => Some(vec![
                *num_args as i128,
                *num_returns as i128,
                *native_table_index as i128,
            ]),
            Instruction::Enter {
                arg_count,
                stack_variables,
                skip,
                ..
            } => Some(vec![
                *arg_count as i128,
                *stack_variables as i128,
                *skip as i128,
            ]),
            Instruction::Leave {
                arg_count,
                return_address_index,
            } => Some(vec![*arg_count as i128, *return_address_index as i128]),
            Instruction::ArrayU8 { size } => Some(vec![*size as i128]),
            Instruction::ArrayU8Load { size } => Some(vec![*size as i128]),
            Instruction::ArrayU8Store { size } => Some(vec![*size as i128]),
            Instruction::LocalU8 { frame_index } => Some(vec![*frame_index as i128]),
            Instruction::LocalU8Load { frame_index } => Some(vec![*frame_index as i128]),
            Instruction::LocalU8Store { frame_index } => Some(vec![*frame_index as i128]),
            Instruction::StaticU8 { static_var_index } => Some(vec![*static_var_index as i128]),
            Instruction::StaticU8Load { static_var_index } => Some(vec![*static_var_index as i128]),
            Instruction::StaticU8Store { static_var_index } => Some(vec![*static_var_index as i128]),
            Instruction::IaddU8 { num } => Some(vec![*num as i128]),
            Instruction::ImulU8 { num } => Some(vec![*num as i128]),
            Instruction::IoffsetU8 { offset } => Some(vec![*offset as i128]),
            Instruction::IoffsetU8Load { offset } => Some(vec![*offset as i128]),
            Instruction::IoffsetU8Store { offset } => Some(vec![*offset as i128]),
            Instruction::PushConstS16 { num } => Some(vec![*num as i128]),
            Instruction::IaddS16 { num } => Some(vec![*num as i128]),
            Instruction::ImulS16 { num } => Some(vec![*num as i128]),
            Instruction::IoffsetS16 { offset } => Some(vec![*offset as i128]),
            Instruction::IoffsetS16Load { offset } => Some(vec![*offset as i128]),
            Instruction::IoffsetS16Store { offset } => Some(vec![*offset as i128]),
            Instruction::ArrayU16 { size } => Some(vec![*size as i128]),
            Instruction::ArrayU16Load { size } => Some(vec![*size as i128]),
            Instruction::ArrayU16Store { size } => Some(vec![*size as i128]),
            Instruction::LocalU16 { frame_index } => Some(vec![*frame_index as i128]),
            Instruction::LocalU16Load { frame_index } => Some(vec![*frame_index as i128]),
            Instruction::LocalU16Store { frame_index } => Some(vec![*frame_index as i128]),
            Instruction::StaticU16 { static_var_index } => Some(vec![*static_var_index as i128]),
            Instruction::StaticU16Load { static_var_index } => Some(vec![*static_var_index as i128]),
            Instruction::StaticU16Store { static_var_index } => Some(vec![*static_var_index as i128]),
            Instruction::GlobalU16 { index } => Some(vec![*index as i128]),
            Instruction::GlobalU16Load { index } => Some(vec![*index as i128]),
            Instruction::GlobalU16Store { index } => Some(vec![*index as i128]),
            Instruction::J { offset } => Some(vec![*offset as i128]),
            Instruction::Jz { offset } => Some(vec![*offset as i128]),
            Instruction::IEqJz { offset } => Some(vec![*offset as i128]),
            Instruction::INeJz { offset } => Some(vec![*offset as i128]),
            Instruction::IGtJz { offset } => Some(vec![*offset as i128]),
            Instruction::IGeJz { offset } => Some(vec![*offset as i128]),
            Instruction::ILtJz { offset } => Some(vec![*offset as i128]),
            Instruction::ILeJz { offset } => Some(vec![*offset as i128]),
            Instruction::Call { location, .. } => Some(vec![*location as i128]),
            Instruction::LocalU24 { frame_index } => Some(vec![*frame_index as i128]),
            Instruction::LocalU24Load { frame_index } => Some(vec![*frame_index as i128]),
            Instruction::LocalU24Store { frame_index } => Some(vec![*frame_index as i128]),
            Instruction::GlobalU24 { index } => Some(vec![*index as i128]),
            Instruction::GlobalU24Load { index } => Some(vec![*index as i128]),
            Instruction::GlobalU24Store { index } => Some(vec![*index as i128]),
            Instruction::PushConstU24 { num } => Some(vec![*num as i128]),
            Instruction::Switch { num_of_entries, .. } => Some(vec![*num_of_entries as i128]),
            Instruction::String { index, .. } => Some(vec![*index as i128]),
            Instruction::TextLabelAssignString { size } => Some(vec![*size as i128]),
            Instruction::TextLabelAssignInt { size } => Some(vec![*size as i128]),
            Instruction::TextLabelAppendString { size } => Some(vec![*size as i128]),
            Instruction::TextLabelAppendInt { size } => Some(vec![*size as i128]),
            _ => None,
        }
    }

    fn get_name(op: &Instruction) -> &'static str {
        match op {
            Instruction::Nop => "NOP",
            Instruction::Iadd => "IADD",
            Instruction::Isub => "ISUB",
            Instruction::Imul => "IMUL",
            Instruction::Idiv => "IDIV",
            Instruction::Imod => "IMOD",
            Instruction::Inot => "INOT",
            Instruction::Ineg => "INEG",
            Instruction::Ieq => "IEQ",
            Instruction::Ine => "INE",
            Instruction::Igt => "IGT",
            Instruction::Ige => "IGE",
            Instruction::Ilt => "ILT",
            Instruction::Ile => "ILE",
            Instruction::Fadd => "FADD",
            Instruction::Fsub => "FSUB",
            Instruction::Fmul => "FMUL",
            Instruction::Fdiv => "FDIV",
            Instruction::Fmod => "FMOD",
            Instruction::Fneg => "FNEG",
            Instruction::Feq => "FEQ",
            Instruction::Fne => "FNE",
            Instruction::Fgt => "FGT",
            Instruction::Fge => "FGE",
            Instruction::Flt => "FLT",
            Instruction::Fle => "FLE",
            Instruction::Vadd => "VADD",
            Instruction::Vsub => "VSUB",
            Instruction::Vmul => "VMUL",
            Instruction::Vdiv => "VDIV",
            Instruction::Vneg => "VNEG",
            Instruction::Iand => "IAND",
            Instruction::Ior => "IOR",
            Instruction::Ixor => "IXOR",
            Instruction::I2f => "I2F",
            Instruction::F2i => "F2I",
            Instruction::F2v => "F2V",
            Instruction::PushConstU8 { .. } => "PUSH_CONST_U8",
            Instruction::PushConstU8U8 { .. } => "PUSH_CONST_U8_U8",
            Instruction::PushConstU8U8U8 { .. } => "PUSH_CONST_U8_U8",
            Instruction::PushConstU32 { .. } => "PUSH_CONST_U32",
            Instruction::PushConstF { .. } => "PUSH_CONST_F",
            Instruction::Dup => "DUP",
            Instruction::Drop => "DROP",
            Instruction::Native { .. } => "NATIVE",
            Instruction::Enter { .. } => "ENTER",
            Instruction::Leave { .. } => "LEAVE",
            Instruction::Load => "LOAD",
            Instruction::Store => "STORE",
            Instruction::StoreRev => "STORE_REV",
            Instruction::LoadN => "LOADN",
            Instruction::StoreN => "STOREN",
            Instruction::ArrayU8 { .. } => "ARRAY_U8",
            Instruction::ArrayU8Load { .. } => "ARRAY_U8_LOAD",
            Instruction::ArrayU8Store { .. } => "ARRAY_U8_STORE",
            Instruction::LocalU8 { .. } => "LOCAL_U8",
            Instruction::LocalU8Load { .. } => "LOCAL_U8_LOAD",
            Instruction::LocalU8Store { .. } => "LOAD_U8_STORE",
            Instruction::StaticU8 { .. } => "STATIC_U8",
            Instruction::StaticU8Load { .. } => "STATIC_U8_LOAD",
            Instruction::StaticU8Store { .. } => "STATIC_U8_STORE",
            Instruction::IaddU8 { .. } => "IADD_U8",
            Instruction::ImulU8 { .. } => "IMUL_U8",
            Instruction::Ioffset => "IOFFSET",
            Instruction::IoffsetU8 { .. } => "IOFFSET_U8",
            Instruction::IoffsetU8Load { .. } => "IOFFSET_U8_LOAD",
            Instruction::IoffsetU8Store { .. } => "IOFFSET_U8_STORE",
            Instruction::PushConstS16 { .. } => "PUSH_CONST_S16",
            Instruction::IaddS16 { .. } => "IADD_S16",
            Instruction::ImulS16 { .. } => "IMUL_S16",
            Instruction::IoffsetS16 { .. } => "IOFFSET_S16",
            Instruction::IoffsetS16Load { .. } => "IOFFSET_S16_LOAD",
            Instruction::IoffsetS16Store { .. } => "IOFFSET_S16_STORE",
            Instruction::ArrayU16 { .. } => "ARRAY_U16",
            Instruction::ArrayU16Load { .. } => "ARRAY_U16_LOAD",
            Instruction::ArrayU16Store { .. } => "ARRAY_U16_STORE",
            Instruction::LocalU16 { .. } => "LOCAL_U16",
            Instruction::LocalU16Load { .. } => "LOCAL_U16_LOAD",
            Instruction::LocalU16Store { .. } => "LOCAL_U16_STORE",
            Instruction::StaticU16 { .. } => "STATIC_U16",
            Instruction::StaticU16Load { .. } => "STATIC_U16_LOAD",
            Instruction::StaticU16Store { .. } => "STATIC_U16_STORE",
            Instruction::GlobalU16 { .. } => "GLOBAL_U16",
            Instruction::GlobalU16Load { .. } => "GLOBAL_U16_STORE",
            Instruction::GlobalU16Store { .. } => "GLOBAL_U16_STORE",
            Instruction::J { .. } => "J",
            Instruction::Jz { .. } => "JZ",
            Instruction::IEqJz { .. } => "IEQ_JZ",
            Instruction::INeJz { .. } => "INE_JZ",
            Instruction::IGtJz { .. } => "IGT_JZ",
            Instruction::IGeJz { .. } => "IGE_JZ",
            Instruction::ILtJz { .. } => "ILT_JZ",
            Instruction::ILeJz { .. } => "ILE_JZ",
            Instruction::Call { .. } => "CALL",
            Instruction::LocalU24 { .. } => "LOCAL_U24",
            Instruction::LocalU24Load { .. } => "LOCAL_U24_LOAD",
            Instruction::LocalU24Store { .. } => "LOCAL_U24_STORE",
            Instruction::GlobalU24 { .. } => "GLOBAL_U24",
            Instruction::GlobalU24Load { .. } => "GLOBAL_U24_LOAD",
            Instruction::GlobalU24Store { .. } => "GLOBAL_U24_STORE",
            Instruction::PushConstU24 { .. } => "PUSH_CONST_U24",
            Instruction::Switch { .. } => "SWITCH",
            Instruction::String { .. } => "STRING",
            Instruction::StringHash => "STRING_HASH",
            Instruction::TextLabelAssignString { .. } => "TEXT_LABEL_ASSIGN_STRING",
            Instruction::TextLabelAssignInt { .. } => "TEXT_LABEL_ASSIGN_INT",
            Instruction::TextLabelAppendString { .. } => "TEXT_LABEL_APPEND_STRING",
            Instruction::TextLabelAppendInt { .. } => "TEXT_LABEL_APPEND_INT",
            Instruction::TextLabelCopy => "TEXT_LABEL_COPY",
            Instruction::Catch => "CATCH",
            Instruction::Throw => "THROW",
            Instruction::CallIndirect => "CALL_INDIRECT",
            Instruction::PushConstM1 => "PUSH_CONST_M1",
            Instruction::PushConst0 => "PUSH_CONST_0",
            Instruction::PushConst1 => "PUSH_CONST_1",
            Instruction::PushConst2 => "PUSH_CONST_2",
            Instruction::PushConst3 => "PUSH_CONST_3",
            Instruction::PushConst4 => "PUSH_CONST_4",
            Instruction::PushConst5 => "PUSH_CONST_5",
            Instruction::PushConst6 => "PUSH_CONST_6",
            Instruction::PushConst7 => "PUSH_CONST_7",
            Instruction::PushConstFM1 => "PUSH_CONSTF_M1",
            Instruction::PushConstF0 => "PUSH_CONSTF_0",
            Instruction::PushConstF1 => "PUSH_CONSTF_1",
            Instruction::PushConstF2 => "PUSH_CONSTF_2",
            Instruction::PushConstF3 => "PUSH_CONSTF_3",
            Instruction::PushConstF4 => "PUSH_CONSTF_4",
            Instruction::PushConstF5 => "PUSH_CONSTF_5",
            Instruction::PushConstF6 => "PUSH_CONSTF_6",
            Instruction::PushConstF7 => "PUSH_CONSTF_7",
            Instruction::IsBitSet => "IS_BIT_SET",
        }
    }
}

impl std::fmt::Display for DisassembledOpcode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut operands = "".to_owned();
        let mut comment = "".to_owned();

        if let Some(ops) = &self.operands {
            operands = ops
                .iter()
                .map(|num| num.to_string())
                .collect::<Vec<String>>()
                .join(" ");
        }

        if let Some(c) = &self.comment {
            comment = format!("/* {c} */");
        }
        write!(f, "{: <16} {} {}", self.name, operands, comment)
    }
}
