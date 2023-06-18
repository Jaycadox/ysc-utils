use byteorder::{LittleEndian, BigEndian, ReadBytesExt};
use anyhow::{Error, Result};
use std::io::{Cursor, Seek, SeekFrom};
use std::path::Path;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use enum_index_derive::*;
use enum_index::*;
use crate::YSC;
use crate::ysc::YSCScript;

pub struct GlobalReference {
    index: u32,
    offsets: Vec<u32>
}

impl GlobalReference {
    pub fn new(index: u32, offsets: Vec<u32>) -> Self {
        Self { index, offsets }
    }
}

#[derive(Debug)]
pub struct GlobalSignature {
    enter_offset: Option<i32>,
    previous_opcodes: Option<Vec<u8>>,
    size: usize
}

pub struct DisassembledScript {
    pub instructions: Vec<Opcode>
}

const PREV_OPCODE_SIG_LENGTH: usize = 15;

impl DisassembledScript {
    pub fn new(instructions: Vec<Opcode>) -> Self {
        Self { instructions }
    }

    pub fn from_ysc_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let src = std::fs::read(path)?;
        let ysc = YSC::new(&src)?.get_script()?;
        let script = Disassembler::new(&ysc).disassemble()?;

        Ok(script)
    }

    pub fn get_pretty(&self, ops: &[Opcode]) -> String {
        let mut string_buf = "".to_owned();
        for op in ops {
            if let Some(global_index) = DisassembledScript::get_global_index(op) {
                string_buf += &format!("Global_{global_index}");
            } else if let Some(offset) = DisassembledScript::get_global_offset_value(op) {
                if format!("{op:?}").to_lowercase().contains("array") { /* hacky way to detect arrays */
                    string_buf += &format!("[? /*{offset}*/]");
                } else {
                    string_buf += &format!(".f_{offset}");
                }
            }
        }

        string_buf
    }

    pub fn find_from_signature(&self, signature: &GlobalSignature) -> Option<(&[Opcode], bool)> {
        let mut prev_opcode_answer = 0;
        let mut enter_answer = false;

        if let Some(prev_opcodes) = &signature.previous_opcodes {
            let mut current_prev_index = 0;
            let mut i = 0;
            'search: for op in &self.instructions {
                let val = op.enum_index() as u8;
                if val != prev_opcodes[current_prev_index] {
                    current_prev_index = 0;
                }
                if val == prev_opcodes[current_prev_index] {
                    current_prev_index += 1;
                    if current_prev_index == PREV_OPCODE_SIG_LENGTH {
                        prev_opcode_answer = i + 1;
                        break 'search;
                    }
                }
                i += 1;
            }

            if let Some(offset) = &signature.enter_offset {
                match &self.instructions[prev_opcode_answer - *offset as usize] {
                    Opcode::Enter { .. } => {
                        enter_answer = true;
                    }
                    _ => {}
                }
            }

            return Some((&self.instructions[prev_opcode_answer..(prev_opcode_answer+signature.size)], enter_answer));
        }

        None
    }

    pub fn generate_signature(&self, index_and_size: (usize, usize)) -> GlobalSignature {
        let (index, size) = index_and_size;

        let mut prev_opcode_sig = None;
        if index > PREV_OPCODE_SIG_LENGTH {
            let mut opcodes = Vec::new();
            for opcode in &self.instructions[(index - PREV_OPCODE_SIG_LENGTH)..index] {
                opcodes.push(opcode.enum_index() as u8);
            }
            prev_opcode_sig = Some(opcodes);
        }

        let mut enter_offset = None;
        for offset in 1..=4096 {
            if offset >= index {
                break;
            }
            let opcode = &self.instructions[index - offset];
            match opcode {
                Opcode::Enter { .. } => {
                    enter_offset = Some(offset as i32);
                    break;
                }
                _ => {}
            }
        }

        GlobalSignature {
            previous_opcodes: prev_opcode_sig,
            enter_offset,
            size
        }
    }

    fn get_global_index(op: &Opcode) -> Option<u32> {
        match op {
            Opcode::GlobalU16 { index } => Some(*index as u32),
            Opcode::GlobalU16Store { index } => Some(*index as u32),
            Opcode::GlobalU16Load { index } => Some(*index as u32),
            Opcode::GlobalU24 { index } => Some(*index as u32),
            Opcode::GlobalU24Store { index } => Some(*index as u32),
            Opcode::GlobalU24Load { index } => Some(*index as u32),
            _ => None
        }
    }

    fn get_global_offset_value(op: &Opcode) -> Option<u32> {
        match op {
            Opcode::IoffsetU8 { offset } => Some(*offset as u32),
            Opcode::IoffsetU8Load { offset } => Some(*offset as u32),
            Opcode::IoffsetU8Store { offset } => Some(*offset as u32),
            Opcode::IoffsetS16 { offset } => Some(*offset as u32),
            Opcode::IoffsetS16Load { offset } => Some(*offset as u32),
            Opcode::IoffsetS16Store{ offset } => Some(*offset as u32),
            Opcode::PushConstU24{ num } => Some(*num as u32),
            Opcode::ArrayU8 { size } => Some(*size as u32),
            Opcode::ArrayU8Store { size } => Some(*size as u32),
            Opcode::ArrayU8Load { size } => Some(*size as u32),
            Opcode::ArrayU16 { size } => Some(*size as u32),
            Opcode::ArrayU16Store { size } => Some(*size as u32),
            Opcode::ArrayU16Load { size } => Some(*size as u32),
            _ => None
        }
    }

    pub fn find_global_reference(&self, global_ref: &GlobalReference) -> Option<(usize, usize)> {
        let mut i = 0;
        let len = self.instructions.len();
        let mut in_global = false;

        let mut offset_index = 0;

        while i < len {
            if let Some(global_value) = DisassembledScript::get_global_index(&self.instructions[i]) {
                in_global = global_value == global_ref.index;
            }
            else if let Some(offset) = DisassembledScript::get_global_offset_value(&self.instructions[i]) {
                if in_global {
                    if global_ref.offsets[offset_index] == offset {
                        offset_index += 1;
                        if offset_index == global_ref.offsets.len() {
                            return Some((i - global_ref.offsets.len(), 1 + global_ref.offsets.len()));
                        }
                    } else {
                        in_global = false;
                        offset_index = 0;
                    }
                }
            } else {
                in_global = false;
                offset_index = 0;
            }



            if !in_global {
                offset_index = 0;
            }
            i += 1;
        }

        None
    }
}

pub struct Disassembler<'a> {
    script: &'a YSCScript,
    current_stack_top: u32
}

impl<'a> Disassembler<'a> {
    pub fn new(script: &'a YSCScript) -> Self {
        Self { script, current_stack_top: 0 }
    }
    pub fn disassemble(&mut self) -> Result<DisassembledScript, Error> {
        let mut cursor = Cursor::new(&self.script.code[..]);
        cursor.set_position(0);

        let mut opcodes = Vec::new();
        while cursor.position() < self.script.code.len() as u64 {
            self.current_stack_top = 0;
            let inst = self.disassemble_opcode(&mut cursor)?;
            opcodes.push(inst);
        }

        Ok(DisassembledScript::new(opcodes))
    }

    fn disassemble_opcode(&mut self, cursor: &mut Cursor<&[u8]>) -> Result<Opcode, Error> {
        let value = RawOpcode::try_from(cursor.read_u8()?)?;
        match value {
            RawOpcode::Nop => Ok(Opcode::Nop),
            RawOpcode::Iadd => Ok(Opcode::Iadd),
            RawOpcode::Isub => Ok(Opcode::Isub),
            RawOpcode::Imul => Ok(Opcode::Imul),
            RawOpcode::Idiv => Ok(Opcode::Idiv),
            RawOpcode::Imod => Ok(Opcode::Imod),
            RawOpcode::Inot => Ok(Opcode::Inot),
            RawOpcode::Ineg => Ok(Opcode::Ineg),
            RawOpcode::Ieq => Ok(Opcode::Ieq),
            RawOpcode::Ine => Ok(Opcode::Ine),
            RawOpcode::Igt => Ok(Opcode::Igt),
            RawOpcode::Ige => Ok(Opcode::Ige),
            RawOpcode::Ilt => Ok(Opcode::Ilt),
            RawOpcode::Ile => Ok(Opcode::Ile),
            RawOpcode::Fadd => Ok(Opcode::Fadd),
            RawOpcode::Fsub => Ok(Opcode::Fsub),
            RawOpcode::Fmul => Ok(Opcode::Fmul),
            RawOpcode::Fdiv => Ok(Opcode::Fdiv),
            RawOpcode::Fmod => Ok(Opcode::Fmod),
            RawOpcode::Fneg => Ok(Opcode::Fneg),
            RawOpcode::Feq => Ok(Opcode::Feq),
            RawOpcode::Fne => Ok(Opcode::Fne),
            RawOpcode::Fgt => Ok(Opcode::Fgt),
            RawOpcode::Fge => Ok(Opcode::Fge),
            RawOpcode::Flt => Ok(Opcode::Flt),
            RawOpcode::Fle => Ok(Opcode::Fle),
            RawOpcode::Vadd => Ok(Opcode::Vadd),
            RawOpcode::Vsub => Ok(Opcode::Vsub),
            RawOpcode::Vmul => Ok(Opcode::Vmul),
            RawOpcode::Vdiv => Ok(Opcode::Vdiv),
            RawOpcode::Vneg => Ok(Opcode::Vneg),
            RawOpcode::Iand => Ok(Opcode::Iand),
            RawOpcode::Ior => Ok(Opcode::Ior),
            RawOpcode::Ixor => Ok(Opcode::Ixor),
            RawOpcode::I2F => Ok(Opcode::I2f),
            RawOpcode::F2I => Ok(Opcode::F2i),
            RawOpcode::F2V => Ok(Opcode::F2v),
            RawOpcode::PushConstU8 => {
                self.current_stack_top = cursor.read_u8()? as u32;
                Ok(Opcode::PushConstU8 { one: self.current_stack_top as u8 })
            },
            RawOpcode::PushConstU8U8 => Ok(Opcode::PushConstU8U8 { one: cursor.read_u8()?, two: cursor.read_u8()? }),
            RawOpcode::PushConstU8U8U8 => Ok(Opcode::PushConstU8U8U8 { one: cursor.read_u8()?, two: cursor.read_u8()?, three: cursor.read_u8()? }),
            RawOpcode::PushConstU32 => {
                self.current_stack_top = cursor.read_u32::<LittleEndian>()?;
                Ok(Opcode::PushConstU32 { one: self.current_stack_top })
            },
            RawOpcode::PushConstF => Ok(Opcode::PushConstF { one: cursor.read_f32::<LittleEndian>()? }),
            RawOpcode::Dup => Ok(Opcode::Dup),
            RawOpcode::Drop => Ok(Opcode::Drop),
            RawOpcode::Native => {
                let packed_args_and_returns = cursor.read_u8()?;
                let num_args = packed_args_and_returns >> 2;
                let num_returns = packed_args_and_returns & 0b11;
                let native_table_index = cursor.read_u16::<BigEndian>()?;
                Ok(
                    Opcode::Native {
                        num_args,
                        num_returns,
                        native_table_index,
                        native_hash: self.script.native_table[native_table_index as usize]
                    }
                )
            }
            RawOpcode::Enter => {
                let arg_count = cursor.read_u8()?;
                let stack_variables = cursor.read_u16::<LittleEndian>()?;
                let skip = cursor.read_u8()?;
                cursor.seek(SeekFrom::Current(skip as i64))?;
                Ok(
                    Opcode::Enter {
                        arg_count,
                        stack_variables,
                        skip
                    }
                )
            }
            RawOpcode::Leave => Ok(Opcode::Leave { arg_count: cursor.read_u8()?, return_address_index: cursor.read_u8()? }),
            RawOpcode::Load => Ok(Opcode::Load),
            RawOpcode::Store => Ok(Opcode::Store),
            RawOpcode::StoreRev => Ok(Opcode::StoreRev),
            RawOpcode::LoadN => Ok(Opcode::LoadN),
            RawOpcode::StoreN => Ok(Opcode::StoreN),
            RawOpcode::ArrayU8 => Ok(Opcode::ArrayU8 { size: cursor.read_u8()? }),
            RawOpcode::ArrayU8Load => Ok(Opcode::ArrayU8Load { size: cursor.read_u8()? }),
            RawOpcode::ArrayU8Store => Ok(Opcode::ArrayU8Store { size: cursor.read_u8()? }),
            RawOpcode::LocalU8 => Ok(Opcode::LocalU8 { frame_index: cursor.read_u8()? }),
            RawOpcode::LocalU8Load => Ok(Opcode::LocalU8Load { frame_index: cursor.read_u8()? }),
            RawOpcode::LocalU8Store => Ok(Opcode::LocalU8Store { frame_index: cursor.read_u8()? }),
            RawOpcode::StaticU8 => Ok(Opcode::StaticU8 { static_var_index: cursor.read_u8()? }),
            RawOpcode::StaticU8Load => Ok(Opcode::StaticU8Load { static_var_index: cursor.read_u8()? }),
            RawOpcode::StaticU8Store => Ok(Opcode::StaticU8Store { static_var_index: cursor.read_u8()? }),
            RawOpcode::IaddU8 => Ok(Opcode::IaddU8 { num: cursor.read_u8()? }),
            RawOpcode::ImulU8 => Ok(Opcode::ImulU8 { num: cursor.read_u8()? }),
            RawOpcode::Ioffset => Ok(Opcode::Ioffset),
            RawOpcode::IoffsetU8 => Ok(Opcode::IoffsetU8 { offset: cursor.read_u8()? }),
            RawOpcode::IoffsetU8Load => Ok(Opcode::IoffsetU8Load { offset: cursor.read_u8()? }),
            RawOpcode::IoffsetU8Store => Ok(Opcode::IoffsetU8Store { offset: cursor.read_u8()? }),
            RawOpcode::PushConstS16 => Ok(Opcode::PushConstS16 { num: cursor.read_i16::<LittleEndian>()? }),
            RawOpcode::IaddS16 => Ok(Opcode::IaddS16 { num: cursor.read_i16::<LittleEndian>()? }),
            RawOpcode::ImulS16 => Ok(Opcode::ImulS16 { num: cursor.read_i16::<LittleEndian>()? }),
            RawOpcode::IoffsetS16 => Ok(Opcode::IoffsetS16 { offset: cursor.read_i16::<LittleEndian>()? }),
            RawOpcode::IoffsetS16Load => Ok(Opcode::IoffsetS16Load { offset: cursor.read_i16::<LittleEndian>()? }),
            RawOpcode::IoffsetS16Store => Ok(Opcode::IoffsetS16Store { offset: cursor.read_i16::<LittleEndian>()? }),
            RawOpcode::ArrayU16 => Ok(Opcode::ArrayU16 { size: cursor.read_u16::<LittleEndian>()? }),
            RawOpcode::ArrayU16Load => Ok(Opcode::ArrayU16Load { size: cursor.read_u16::<LittleEndian>()? }),
            RawOpcode::ArrayU16Store => Ok(Opcode::ArrayU16Store { size: cursor.read_u16::<LittleEndian>()? }),
            RawOpcode::LocalU16 => Ok(Opcode::LocalU16 { frame_index: cursor.read_u16::<LittleEndian>()? }),
            RawOpcode::LocalU16Load => Ok(Opcode::LocalU16Load { frame_index: cursor.read_u16::<LittleEndian>()? }),
            RawOpcode::LocalU16Store => Ok(Opcode::LocalU16Store { frame_index: cursor.read_u16::<LittleEndian>()? }),
            RawOpcode::StaticU16 => Ok(Opcode::StaticU16 { static_var_index: cursor.read_u16::<LittleEndian>()? }),
            RawOpcode::StaticU16Load => Ok(Opcode::StaticU16Load { static_var_index: cursor.read_u16::<LittleEndian>()? }),
            RawOpcode::StaticU16Store => Ok(Opcode::StaticU16Store { static_var_index: cursor.read_u16::<LittleEndian>()? }),
            RawOpcode::GlobalU16 => Ok(Opcode::GlobalU16 { index: cursor.read_u16::<LittleEndian>()? }),
            RawOpcode::GlobalU16Load => Ok(Opcode::GlobalU16Load { index: cursor.read_u16::<LittleEndian>()? }),
            RawOpcode::GlobalU16Store => Ok(Opcode::GlobalU16Store { index: cursor.read_u16::<LittleEndian>()? }),
            RawOpcode::J => Ok(Opcode::J { offset: cursor.read_i16::<LittleEndian>()? }),
            RawOpcode::Jz => Ok(Opcode::Jz { offset: cursor.read_i16::<LittleEndian>()? }),
            RawOpcode::IeqJz => Ok(Opcode::IEqJz { offset: cursor.read_i16::<LittleEndian>()? }),
            RawOpcode::IneJz => Ok(Opcode::INeJz { offset: cursor.read_i16::<LittleEndian>()? }),
            RawOpcode::IgtJz => Ok(Opcode::IGtJz { offset: cursor.read_i16::<LittleEndian>()? }),
            RawOpcode::IgeJz => Ok(Opcode::IGeJz { offset: cursor.read_i16::<LittleEndian>()? }),
            RawOpcode::IltJz => Ok(Opcode::ILtJz { offset: cursor.read_i16::<LittleEndian>()? }),
            RawOpcode::IleJz => Ok(Opcode::ILeJz { offset: cursor.read_i16::<LittleEndian>()? }),
            RawOpcode::Call => Ok(Opcode::Call { location: cursor.read_u24::<LittleEndian>()? }),
            RawOpcode::LocalU24 => Ok(Opcode::LocalU24 { frame_index: cursor.read_u24::<LittleEndian>()? }),
            RawOpcode::LocalU24Load => Ok(Opcode::LocalU24Load { frame_index: cursor.read_u24::<LittleEndian>()? }),
            RawOpcode::LocalU24Store => Ok(Opcode::LocalU24Store { frame_index: cursor.read_u24::<LittleEndian>()? }),
            RawOpcode::GlobalU24 => Ok(Opcode::GlobalU24 { index: cursor.read_u24::<LittleEndian>()? }),
            RawOpcode::GlobalU24Load => Ok(Opcode::GlobalU24Load { index: cursor.read_u24::<LittleEndian>()? }),
            RawOpcode::GlobalU24Store => Ok(Opcode::GlobalU24Store { index: cursor.read_u24::<LittleEndian>()? }),
            RawOpcode::PushConstU24 => {
                self.current_stack_top = cursor.read_u24::<LittleEndian>()?;
                Ok(Opcode::PushConstU24 { num: self.current_stack_top })
            },
            RawOpcode::Switch => {
                let num_entries = cursor.read_u8()?;
                let mut entries = Vec::new();
                for _ in 0..num_entries {

                    let entry = SwitchEntry {
                        index_id: cursor.read_u32::<LittleEndian>()?,
                        jump_offset: cursor.read_u16::<LittleEndian>()?
                    };

                    // Suppress unused warnings
                    let _ = entry.index_id;
                    let _ = entry.jump_offset;

                    entries.push(entry);
                }

                Ok(
                    Opcode::Switch {
                        num_of_entries: num_entries,
                        entries
                    }
                )
            }
            RawOpcode::String => Ok(Opcode::String { value: self.script.strings[self.current_stack_top as usize].to_string() } ),
            RawOpcode::StringHash => Ok(Opcode::StringHash),
            RawOpcode::TextLabelAssignString => Ok(Opcode::TextLabelAssignString { size: cursor.read_u8()? }),
            RawOpcode::TextLabelAssignInt => Ok(Opcode::TextLabelAssignInt { size: cursor.read_u8()? }),
            RawOpcode::TextLabelAppendString => Ok(Opcode::TextLabelAppendString { size: cursor.read_u8()? }),
            RawOpcode::TextLabelAppendInt => Ok(Opcode::TextLabelAppendInt { size: cursor.read_u8()? }),
            RawOpcode::TextLabelCopy => Ok(Opcode::TextLabelCopy),
            RawOpcode::Catch => Ok(Opcode::Catch),
            RawOpcode::Throw => Ok(Opcode::Throw),
            RawOpcode::CallIndirect => Ok(Opcode::CallIndirect),
            RawOpcode::PushConstM1 => Ok(Opcode::PushConstM1),
            RawOpcode::PushConst0 => { self.current_stack_top = 0; Ok(Opcode::PushConst0) },
            RawOpcode::PushConst1 => { self.current_stack_top = 1; Ok(Opcode::PushConst1) },
            RawOpcode::PushConst2 => { self.current_stack_top = 2; Ok(Opcode::PushConst2) },
            RawOpcode::PushConst3 => { self.current_stack_top = 3; Ok(Opcode::PushConst3) },
            RawOpcode::PushConst4 => { self.current_stack_top = 4; Ok(Opcode::PushConst4) },
            RawOpcode::PushConst5 => { self.current_stack_top = 5; Ok(Opcode::PushConst5) },
            RawOpcode::PushConst6 => { self.current_stack_top = 6; Ok(Opcode::PushConst6) },
            RawOpcode::PushConst7 => { self.current_stack_top = 7; Ok(Opcode::PushConst7) },
            RawOpcode::PushConstFm1 => Ok(Opcode::PushConstFM1),
            RawOpcode::PushConstF0 => Ok(Opcode::PushConstF0),
            RawOpcode::PushConstF1 => Ok(Opcode::PushConstF1),
            RawOpcode::PushConstF2 => Ok(Opcode::PushConstF2),
            RawOpcode::PushConstF3 => Ok(Opcode::PushConstF3),
            RawOpcode::PushConstF4 => Ok(Opcode::PushConstF4),
            RawOpcode::PushConstF5 => Ok(Opcode::PushConstF5),
            RawOpcode::PushConstF6 => Ok(Opcode::PushConstF6),
            RawOpcode::PushConstF7 => Ok(Opcode::PushConstF7),
            RawOpcode::IsBitSet => Ok(Opcode::IsBitSet),
        }
    }
}

#[derive(Debug, EnumIndex)]
pub enum Opcode {
    Nop,
    Iadd,
    Isub,
    Imul,
    Idiv,
    Imod,
    Inot,
    Ineg,
    Ieq,
    Ine,
    Igt,
    Ige,
    Ilt,
    Ile,
    Fadd,
    Fsub,
    Fmul,
    Fdiv,
    Fmod,
    Fneg,
    Feq,
    Fne,
    Fgt,
    Fge,
    Flt,
    Fle,
    Vadd,
    Vsub,
    Vmul,
    Vdiv,
    Vneg,
    Iand,
    Ior,
    Ixor,
    I2f,
    F2i,
    F2v,
    PushConstU8 { one: u8 },
    PushConstU8U8 { one: u8, two: u8 },
    PushConstU8U8U8 { one: u8, two: u8, three: u8 },
    PushConstU32 { one: u32 },
    PushConstF { one: f32 },
    Dup,
    Drop,
    Native { num_args: u8, num_returns: u8, native_table_index: u16, native_hash: u64 },
    Enter { arg_count: u8, stack_variables: u16, skip: u8, },
    Leave { arg_count: u8, return_address_index: u8 },
    Load,
    Store,
    StoreRev,
    LoadN,
    StoreN,
    ArrayU8 { size: u8 },
    ArrayU8Load { size: u8 },
    ArrayU8Store { size: u8 },
    LocalU8 { frame_index: u8 },
    LocalU8Load { frame_index: u8 },
    LocalU8Store { frame_index: u8 },
    StaticU8 { static_var_index: u8 },
    StaticU8Load { static_var_index: u8 },
    StaticU8Store { static_var_index: u8 },
    IaddU8 { num: u8 },
    ImulU8 { num: u8 },
    Ioffset,
    IoffsetU8 { offset: u8 },
    IoffsetU8Load { offset: u8 },
    IoffsetU8Store { offset: u8 },
    PushConstS16 { num: i16 },
    IaddS16 { num: i16 },
    ImulS16 { num: i16 },
    IoffsetS16 { offset: i16 },
    IoffsetS16Load { offset: i16 },
    IoffsetS16Store { offset: i16 },
    ArrayU16 { size: u16 },
    ArrayU16Load { size: u16 },
    ArrayU16Store { size: u16 },
    LocalU16 { frame_index: u16 },
    LocalU16Load { frame_index: u16 },
    LocalU16Store { frame_index: u16 },
    StaticU16 { static_var_index: u16 },
    StaticU16Load { static_var_index: u16 },
    StaticU16Store { static_var_index: u16 },
    GlobalU16 { index: u16 },
    GlobalU16Load { index: u16 },
    GlobalU16Store { index: u16 },
    J { offset: i16 },
    Jz { offset: i16 },
    IEqJz { offset: i16 },
    INeJz { offset: i16 },
    IGtJz { offset: i16 },
    IGeJz { offset: i16 },
    ILtJz { offset: i16 },
    ILeJz { offset: i16 },
    Call { location: u32 },
    LocalU24 { frame_index: u32 },
    LocalU24Load { frame_index: u32 },
    LocalU24Store { frame_index: u32 },
    GlobalU24 { index: u32 },
    GlobalU24Load { index: u32 },
    GlobalU24Store { index: u32 },
    PushConstU24 { num: u32 },
    Switch { num_of_entries: u8, entries: Vec<SwitchEntry> },
    String { value: String },
    StringHash,
    TextLabelAssignString { size: u8 },
    TextLabelAssignInt { size: u8 },
    TextLabelAppendString { size: u8 },
    TextLabelAppendInt { size: u8 },
    TextLabelCopy,
    Catch,
    Throw,
    CallIndirect,
    PushConstM1,
    PushConst0,
    PushConst1,
    PushConst2,
    PushConst3,
    PushConst4,
    PushConst5,
    PushConst6,
    PushConst7,
    PushConstFM1,
    PushConstF0,
    PushConstF1,
    PushConstF2,
    PushConstF3,
    PushConstF4,
    PushConstF5,
    PushConstF6,
    PushConstF7,
    IsBitSet
}

#[derive(Debug)]
pub struct SwitchEntry { index_id: u32, jump_offset: u16 }

#[repr(u8)]
#[derive(Debug, Eq, PartialEq, TryFromPrimitive, IntoPrimitive)]
enum RawOpcode {
    Nop = 0,
    Iadd,
    Isub,
    Imul,
    Idiv,
    Imod,
    Inot,
    Ineg,
    Ieq,
    Ine,
    Igt,
    Ige,
    Ilt,
    Ile,
    Fadd,
    Fsub,
    Fmul,
    Fdiv,
    Fmod,
    Fneg,
    Feq,
    Fne,
    Fgt,
    Fge,
    Flt,
    Fle,
    Vadd,
    Vsub,
    Vmul,
    Vdiv,
    Vneg,
    Iand,
    Ior,
    Ixor,
    I2F,
    F2I,
    F2V,
    PushConstU8,
    PushConstU8U8,
    PushConstU8U8U8,
    PushConstU32,
    PushConstF,
    Dup,
    Drop,
    Native,
    Enter,
    Leave,
    Load,
    Store,
    StoreRev,
    LoadN,
    StoreN,
    ArrayU8,
    ArrayU8Load,
    ArrayU8Store,
    LocalU8,
    LocalU8Load,
    LocalU8Store,
    StaticU8,
    StaticU8Load,
    StaticU8Store,
    IaddU8,
    ImulU8,
    Ioffset,
    IoffsetU8,
    IoffsetU8Load,
    IoffsetU8Store,
    PushConstS16,
    IaddS16,
    ImulS16,
    IoffsetS16,
    IoffsetS16Load,
    IoffsetS16Store,
    ArrayU16,
    ArrayU16Load,
    ArrayU16Store,
    LocalU16,
    LocalU16Load,
    LocalU16Store,
    StaticU16,
    StaticU16Load,
    StaticU16Store,
    GlobalU16,
    GlobalU16Load,
    GlobalU16Store,
    J,
    Jz,
    IeqJz,
    IneJz,
    IgtJz,
    IgeJz,
    IltJz,
    IleJz,
    Call,
    LocalU24,
    LocalU24Load,
    LocalU24Store,
    GlobalU24,
    GlobalU24Load,
    GlobalU24Store,
    PushConstU24,
    Switch,
    String,
    StringHash,
    TextLabelAssignString,
    TextLabelAssignInt,
    TextLabelAppendString,
    TextLabelAppendInt,
    TextLabelCopy,
    Catch,
    Throw,
    CallIndirect,
    PushConstM1,
    PushConst0,
    PushConst1,
    PushConst2,
    PushConst3,
    PushConst4,
    PushConst5,
    PushConst6,
    PushConst7,
    PushConstFm1,
    PushConstF0,
    PushConstF1,
    PushConstF2,
    PushConstF3,
    PushConstF4,
    PushConstF5,
    PushConstF6,
    PushConstF7,
    IsBitSet
}

