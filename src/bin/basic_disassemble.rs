use anyhow::{Context, Error, Result};
use std::fmt::Formatter;
use std::io::Write;
use ysc_utils::disassemble::{DisassembledScript, Opcode};

fn main() {
    if let Err(err) = start() {
        println!("Error: {err}")
    }
}

fn start() -> Result<(), Error> {
    let args = std::env::args().skip(1).collect::<Vec<String>>();

    if args.is_empty() || args.len() > 2 {
        println!("Usage: basic_disassemble %ysc_script% (optional: function number/index)");
        println!("Example  : basic_disasemble freemode.ysc.full");
        println!("Example 2: basic_disasemble freemode.ysc.full func_305");
        return Ok(());
    }

    let specified_function_index_str = args.get(1);
    let mut specified_function_index: i32 = -1;

    let script = DisassembledScript::from_ysc_file(&args[0])
        .context("Failed to read/parse/disassemble ysc file")?;

    let mut wr = Box::new(std::io::BufWriter::new(std::io::stdout()));

    if let Some(index) = specified_function_index_str {
        specified_function_index = index.replace("func_", "").parse()?;
        specified_function_index += 1;
    }

    let instructions = script.instructions;

    let mut func_index = 0;
    for inst in instructions {
        let mut padding = "".to_owned();
        let mut function_comment = "".to_owned();

        match &inst {
            Opcode::Enter { .. } => {
                if func_index == specified_function_index {
                    break;
                }

                function_comment = format!(" /* func_{func_index} */");
                func_index += 1;
            }
            _ => {
                padding = "    ".to_owned();
            }
        }

        if specified_function_index == -1 || func_index == specified_function_index {
            let dop = DisassembledOpcode::from(inst);
            let out = format!("{padding}{dop}{function_comment}\n");
            let _ = wr.write(out.as_ref())?;
        }
    }

    wr.flush()?;

    Ok(())
}

struct DisassembledOpcode {
    name: &'static str,
    operands: Option<Vec<usize>>,
    comment: Option<String>,
}

impl From<Opcode> for DisassembledOpcode {
    fn from(value: Opcode) -> Self {
        Self {
            name: DisassembledOpcode::get_name(&value),
            operands: DisassembledOpcode::get_operands(&value),
            comment: DisassembledOpcode::get_comment(&value),
        }
    }
}

impl DisassembledOpcode {
    fn get_comment(op: &Opcode) -> Option<String> {
        match op {
            Opcode::String { value, .. } => Some(value.to_string()),
            _ => None,
        }
    }

    fn get_operands(op: &Opcode) -> Option<Vec<usize>> {
        match op {
            Opcode::PushConstU8 { one } => Some(vec![*one as usize]),
            Opcode::PushConstU8U8 { one, two } => Some(vec![*one as usize, *two as usize]),
            Opcode::PushConstU8U8U8 { two, one, three } => {
                Some(vec![*one as usize, *two as usize, *three as usize])
            }
            Opcode::PushConstU32 { one } => Some(vec![*one as usize]),
            Opcode::PushConstF { one } => Some(vec![*one as usize]),
            Opcode::Native {
                native_hash,
                native_table_index,
                num_args,
                num_returns,
            } => Some(vec![
                *native_hash as usize,
                *native_table_index as usize,
                *num_args as usize,
                *num_returns as usize,
            ]),
            Opcode::Enter {
                arg_count,
                stack_variables,
                skip,
            } => Some(vec![
                *arg_count as usize,
                *stack_variables as usize,
                *skip as usize,
            ]),
            Opcode::Leave {
                arg_count,
                return_address_index,
            } => Some(vec![*arg_count as usize, *return_address_index as usize]),
            Opcode::ArrayU8 { size } => Some(vec![*size as usize]),
            Opcode::ArrayU8Load { size } => Some(vec![*size as usize]),
            Opcode::ArrayU8Store { size } => Some(vec![*size as usize]),
            Opcode::LocalU8 { frame_index } => Some(vec![*frame_index as usize]),
            Opcode::LocalU8Load { frame_index } => Some(vec![*frame_index as usize]),
            Opcode::LocalU8Store { frame_index } => Some(vec![*frame_index as usize]),
            Opcode::StaticU8 { static_var_index } => Some(vec![*static_var_index as usize]),
            Opcode::StaticU8Load { static_var_index } => Some(vec![*static_var_index as usize]),
            Opcode::StaticU8Store { static_var_index } => Some(vec![*static_var_index as usize]),
            Opcode::IaddU8 { num } => Some(vec![*num as usize]),
            Opcode::ImulU8 { num } => Some(vec![*num as usize]),
            Opcode::IoffsetU8 { offset } => Some(vec![*offset as usize]),
            Opcode::IoffsetU8Load { offset } => Some(vec![*offset as usize]),
            Opcode::IoffsetU8Store { offset } => Some(vec![*offset as usize]),
            Opcode::PushConstS16 { num } => Some(vec![*num as usize]),
            Opcode::IaddS16 { num } => Some(vec![*num as usize]),
            Opcode::ImulS16 { num } => Some(vec![*num as usize]),
            Opcode::IoffsetS16 { offset } => Some(vec![*offset as usize]),
            Opcode::IoffsetS16Load { offset } => Some(vec![*offset as usize]),
            Opcode::IoffsetS16Store { offset } => Some(vec![*offset as usize]),
            Opcode::ArrayU16 { size } => Some(vec![*size as usize]),
            Opcode::ArrayU16Load { size } => Some(vec![*size as usize]),
            Opcode::ArrayU16Store { size } => Some(vec![*size as usize]),
            Opcode::LocalU16 { frame_index } => Some(vec![*frame_index as usize]),
            Opcode::LocalU16Load { frame_index } => Some(vec![*frame_index as usize]),
            Opcode::LocalU16Store { frame_index } => Some(vec![*frame_index as usize]),
            Opcode::StaticU16 { static_var_index } => Some(vec![*static_var_index as usize]),
            Opcode::StaticU16Load { static_var_index } => Some(vec![*static_var_index as usize]),
            Opcode::StaticU16Store { static_var_index } => Some(vec![*static_var_index as usize]),
            Opcode::GlobalU16 { index } => Some(vec![*index as usize]),
            Opcode::GlobalU16Load { index } => Some(vec![*index as usize]),
            Opcode::GlobalU16Store { index } => Some(vec![*index as usize]),
            Opcode::J { offset } => Some(vec![*offset as usize]),
            Opcode::Jz { offset } => Some(vec![*offset as usize]),
            Opcode::IEqJz { offset } => Some(vec![*offset as usize]),
            Opcode::INeJz { offset } => Some(vec![*offset as usize]),
            Opcode::IGtJz { offset } => Some(vec![*offset as usize]),
            Opcode::IGeJz { offset } => Some(vec![*offset as usize]),
            Opcode::ILtJz { offset } => Some(vec![*offset as usize]),
            Opcode::ILeJz { offset } => Some(vec![*offset as usize]),
            Opcode::Call { location } => Some(vec![*location as usize]),
            Opcode::LocalU24 { frame_index } => Some(vec![*frame_index as usize]),
            Opcode::LocalU24Load { frame_index } => Some(vec![*frame_index as usize]),
            Opcode::LocalU24Store { frame_index } => Some(vec![*frame_index as usize]),
            Opcode::GlobalU24 { index } => Some(vec![*index as usize]),
            Opcode::GlobalU24Load { index } => Some(vec![*index as usize]),
            Opcode::GlobalU24Store { index } => Some(vec![*index as usize]),
            Opcode::PushConstU24 { num } => Some(vec![*num as usize]),
            Opcode::Switch { num_of_entries, .. } => Some(vec![*num_of_entries as usize]),
            Opcode::String { index, .. } => Some(vec![*index]),
            Opcode::TextLabelAssignString { size } => Some(vec![*size as usize]),
            Opcode::TextLabelAssignInt { size } => Some(vec![*size as usize]),
            Opcode::TextLabelAppendString { size } => Some(vec![*size as usize]),
            Opcode::TextLabelAppendInt { size } => Some(vec![*size as usize]),
            _ => None,
        }
    }

    fn get_name(op: &Opcode) -> &'static str {
        match op {
            Opcode::Nop => "NOP",
            Opcode::Iadd => "IADD",
            Opcode::Isub => "ISUB",
            Opcode::Imul => "IMUL",
            Opcode::Idiv => "IDIV",
            Opcode::Imod => "IMOD",
            Opcode::Inot => "INOT",
            Opcode::Ineg => "INEG",
            Opcode::Ieq => "IEQ",
            Opcode::Ine => "INE",
            Opcode::Igt => "IGT",
            Opcode::Ige => "IGE",
            Opcode::Ilt => "ILT",
            Opcode::Ile => "ILE",
            Opcode::Fadd => "FADD",
            Opcode::Fsub => "FSUB",
            Opcode::Fmul => "FMUL",
            Opcode::Fdiv => "FDIV",
            Opcode::Fmod => "FMOD",
            Opcode::Fneg => "FNEG",
            Opcode::Feq => "FEQ",
            Opcode::Fne => "FNE",
            Opcode::Fgt => "FGT",
            Opcode::Fge => "FGE",
            Opcode::Flt => "FLT",
            Opcode::Fle => "FLE",
            Opcode::Vadd => "VADD",
            Opcode::Vsub => "VSUB",
            Opcode::Vmul => "VMUL",
            Opcode::Vdiv => "VDIV",
            Opcode::Vneg => "VNEG",
            Opcode::Iand => "IAND",
            Opcode::Ior => "IOR",
            Opcode::Ixor => "IXOR",
            Opcode::I2f => "I2F",
            Opcode::F2i => "F2I",
            Opcode::F2v => "F2V",
            Opcode::PushConstU8 { .. } => "PUSH_CONST_U8",
            Opcode::PushConstU8U8 { .. } => "PUSH_CONST_U8_U8",
            Opcode::PushConstU8U8U8 { .. } => "PUSH_CONST_U8_U8",
            Opcode::PushConstU32 { .. } => "PUSH_CONST_U32",
            Opcode::PushConstF { .. } => "PUSH_CONST_F",
            Opcode::Dup => "DUP",
            Opcode::Drop => "DROP",
            Opcode::Native { .. } => "NATIVE",
            Opcode::Enter { .. } => "ENTER",
            Opcode::Leave { .. } => "LEAVE",
            Opcode::Load => "LOAD",
            Opcode::Store => "STORE",
            Opcode::StoreRev => "STORE_REV",
            Opcode::LoadN => "LOADN",
            Opcode::StoreN => "STOREN",
            Opcode::ArrayU8 { .. } => "ARRAY_U8",
            Opcode::ArrayU8Load { .. } => "ARRAY_U8_LOAD",
            Opcode::ArrayU8Store { .. } => "ARRAY_U8_STORE",
            Opcode::LocalU8 { .. } => "LOCAL_U8",
            Opcode::LocalU8Load { .. } => "LOCAL_U8_LOAD",
            Opcode::LocalU8Store { .. } => "LOAD_U8_STORE",
            Opcode::StaticU8 { .. } => "STATIC_U8",
            Opcode::StaticU8Load { .. } => "STATIC_U8_LOAD",
            Opcode::StaticU8Store { .. } => "STATIC_U8_STORE",
            Opcode::IaddU8 { .. } => "IADD_U8",
            Opcode::ImulU8 { .. } => "IMUL_U8",
            Opcode::Ioffset => "IOFFSET",
            Opcode::IoffsetU8 { .. } => "IOFFSET_U8",
            Opcode::IoffsetU8Load { .. } => "IOFFSET_U8_LOAD",
            Opcode::IoffsetU8Store { .. } => "IOFFSET_U8_STORE",
            Opcode::PushConstS16 { .. } => "PUSH_CONST_S16",
            Opcode::IaddS16 { .. } => "IADD_S16",
            Opcode::ImulS16 { .. } => "IMUL_S16",
            Opcode::IoffsetS16 { .. } => "IOFFSET_S16",
            Opcode::IoffsetS16Load { .. } => "IOFFSET_S16_LOAD",
            Opcode::IoffsetS16Store { .. } => "IOFFSET_S16_STORE",
            Opcode::ArrayU16 { .. } => "ARRAY_U16",
            Opcode::ArrayU16Load { .. } => "ARRAY_U16_LOAD",
            Opcode::ArrayU16Store { .. } => "ARRAY_U16_STORE",
            Opcode::LocalU16 { .. } => "LOCAL_U16",
            Opcode::LocalU16Load { .. } => "LOCAL_U16_LOAD",
            Opcode::LocalU16Store { .. } => "LOCAL_U16_STORE",
            Opcode::StaticU16 { .. } => "STATIC_U16",
            Opcode::StaticU16Load { .. } => "STATIC_U16_LOAD",
            Opcode::StaticU16Store { .. } => "STATIC_U16_STORE",
            Opcode::GlobalU16 { .. } => "GLOBAL_U16",
            Opcode::GlobalU16Load { .. } => "GLOBAL_U16_STORE",
            Opcode::GlobalU16Store { .. } => "GLOBAL_U16_STORE",
            Opcode::J { .. } => "J",
            Opcode::Jz { .. } => "JZ",
            Opcode::IEqJz { .. } => "IEQ_JZ",
            Opcode::INeJz { .. } => "INE_JZ",
            Opcode::IGtJz { .. } => "IGT_JZ",
            Opcode::IGeJz { .. } => "IGE_JZ",
            Opcode::ILtJz { .. } => "ILT_JZ",
            Opcode::ILeJz { .. } => "ILE_JZ",
            Opcode::Call { .. } => "CALL",
            Opcode::LocalU24 { .. } => "LOCAL_U24",
            Opcode::LocalU24Load { .. } => "LOCAL_U24_LOAD",
            Opcode::LocalU24Store { .. } => "LOCAL_U24_STORE",
            Opcode::GlobalU24 { .. } => "GLOBAL_U24",
            Opcode::GlobalU24Load { .. } => "GLOBAL_U24_STORE",
            Opcode::GlobalU24Store { .. } => "GLOBAL_U24_STORE",
            Opcode::PushConstU24 { .. } => "PUSH_CONST_U24",
            Opcode::Switch { .. } => "SWITCH",
            Opcode::String { .. } => "STRING",
            Opcode::StringHash => "STRING_HASH",
            Opcode::TextLabelAssignString { .. } => "TEXT_LABEL_ASSIGN_STRING",
            Opcode::TextLabelAssignInt { .. } => "TEXT_LABEL_ASSIGN_INT",
            Opcode::TextLabelAppendString { .. } => "TEXT_LABEL_APPEND_STRING",
            Opcode::TextLabelAppendInt { .. } => "TEXT_LABEL_APPEND_INT",
            Opcode::TextLabelCopy => "TEXT_LABEL_COPY",
            Opcode::Catch => "CATCH",
            Opcode::Throw => "THROW",
            Opcode::CallIndirect => "CALL_INDIRECT",
            Opcode::PushConstM1 => "PUSH_CONST_M1",
            Opcode::PushConst0 => "PUSH_CONST_0",
            Opcode::PushConst1 => "PUSH_CONST_1",
            Opcode::PushConst2 => "PUSH_CONST_2",
            Opcode::PushConst3 => "PUSH_CONST_3",
            Opcode::PushConst4 => "PUSH_CONST_4",
            Opcode::PushConst5 => "PUSH_CONST_5",
            Opcode::PushConst6 => "PUSH_CONST_6",
            Opcode::PushConst7 => "PUSH_CONST_7",
            Opcode::PushConstFM1 => "PUSH_CONSTF_M1",
            Opcode::PushConstF0 => "PUSH_CONSTF_0",
            Opcode::PushConstF1 => "PUSH_CONSTF_1",
            Opcode::PushConstF2 => "PUSH_CONSTF_2",
            Opcode::PushConstF3 => "PUSH_CONSTF_3",
            Opcode::PushConstF4 => "PUSH_CONSTF_4",
            Opcode::PushConstF5 => "PUSH_CONSTF_5",
            Opcode::PushConstF6 => "PUSH_CONSTF_6",
            Opcode::PushConstF7 => "PUSH_CONSTF_7",
            Opcode::IsBitSet => "IS_BIT_SET",
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
