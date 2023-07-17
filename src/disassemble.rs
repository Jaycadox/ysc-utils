use crate::ysc::*;
use thiserror::Error;
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use enum_index::*;
use enum_index_derive::*;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use std::cmp::{max, min};
use std::collections::HashMap;
use std::io::{Cursor, Seek, SeekFrom};
use std::path::Path;

/// Generic errors pertaining to disassembly
#[derive(Error, Debug)]
pub enum DisassembleError {
    /// Most commonly this includes "file not found" errors
    #[error("io error")]
    IOError(#[from] std::io::Error),

    /// An byte we read does not seem to have a corresponding opcode
    #[error("bad opcode")]
    BadOpcode(#[from] ::num_enum::TryFromPrimitiveError<RawOpcode>),

    /// Generic ysc error
    #[error("ysc error")]
    YscError(#[from] YscError),
}

/// A global reference with one or more offsets
pub struct GlobalReference {
    index: u32,
    offsets: Vec<u32>,
}

impl GlobalReference {
    /// Create a global reference given an index and one or more offsets
    pub fn new(index: u32, offsets: Vec<u32>) -> Self {
        Self { index, offsets }
    }
}

/// Information needed to find a global reference across different versions of the same script
#[derive(Debug)]
pub struct GlobalSignature {
    enter_offset: Option<i32>,
    previous_opcodes: Option<Vec<u8>>,
    size: usize,
    hint: usize,
}

/// List of script instructions, used for global matching
pub struct InstructionList {
    /// Vector of ScriptVM instructions
    pub instructions: Vec<Instruction>,
}

const PREV_OPCODE_SIG_LENGTH: usize = 15;

impl InstructionList {
    /// Create a ScriptInstructionList given a list of instructions
    pub fn new(instructions: Vec<Instruction>) -> Self {
        Self { instructions }
    }

    /// Create a ScriptInstructionList given a path to a ysc file
    pub fn from_ysc_file(path: impl AsRef<Path>) -> Result<Self, DisassembleError> {
        let src = std::fs::read(path)?;
        let ysc = YSCReader::new(&src)?.get_script()?;
        let script = Disassembler::new(&ysc).disassemble(None)?;

        Ok(script)
    }
    /// Poor mans way getting a C like string from a global reference with optional offsets and arrays
    pub fn get_pretty(&self, ops: &[Instruction]) -> String {
        let mut string_buf = "".to_owned();
        for op in ops {
            if let Some(global_index) = InstructionList::get_global_index(op) {
                string_buf += &format!("Global_{global_index}");
            } else if let Some(offset) = InstructionList::get_global_offset_value(op) {
                if format!("{op:?}").to_lowercase().contains("array") {
                    /* hacky way to detect arrays */
                    string_buf += &format!("[? /*{offset}*/]");
                } else {
                    string_buf += &format!(".f_{offset}");
                }
            }
        }

        string_buf
    }

    fn find_from_signature_given_instructions<'a>(
        signature: &GlobalSignature,
        instructions: &'a [Instruction],
    ) -> Option<(&'a [Instruction], bool)> {
        let mut prev_opcode_answer = 0;
        let mut enter_answer = false;

        if let Some(prev_opcodes) = &signature.previous_opcodes {
            let mut current_prev_index = 0;
            'search: for (i, op) in instructions.iter().enumerate() {
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
            }

            if let Some(offset) = &signature.enter_offset {
                if *offset < prev_opcode_answer as i32 {
                    if let Instruction::Enter { .. } =
                        &instructions[prev_opcode_answer - *offset as usize]
                    {
                        enter_answer = true;
                    }
                }
            }

            return Some((
                &instructions[prev_opcode_answer..(prev_opcode_answer + signature.size)],
                enter_answer,
            ));
        }

        None
    }

    /// Try to find a global reference in a script given a Vec of GlobalSignatures
    /// If the bool is true, there is a higher chance that the reference is correct
    pub fn find_from_signatures(
        &self,
        signatures: &Vec<GlobalSignature>,
    ) -> Option<Vec<(&[Instruction], bool)>> {
        let mut results = Vec::new();
        for sig in signatures {
            if let Some(res) = self.find_from_signature(sig) {
                results.push(res);
            }
        }

        if results.is_empty() {
            None
        } else {
            Some(results)
        }
    }

    /// Try to find a global reference in a script given a GlobalSignatures
    /// If the bool is true, there is a higher chance that the reference is correct
    pub fn find_from_signature(&self, signature: &GlobalSignature) -> Option<(&[Instruction], bool)> {
        return if let Some(res) = InstructionList::find_from_signature_given_instructions(
            signature,
            &self.instructions[(max(signature.hint as i64 - 512, 0) as usize)
                ..min(signature.hint + 512, self.instructions.len() - 1)],
        ) {
            Some(res)
        } else {
            InstructionList::find_from_signature_given_instructions(
                signature,
                &self.instructions,
            )
        };
    }

    /// Generate signatures given a global index and the number of instructions after which specify the offset
    pub fn generate_signatures(
        &self,
        indexes_and_sizes: &[(usize, usize)],
    ) -> Vec<GlobalSignature> {
        indexes_and_sizes
            .iter()
            .map(|f| self.generate_signature(f))
            .collect()
    }

    /// Generate a signature given a global index and the number of instructions after which specify the offset
    pub fn generate_signature(&self, index_and_size: &(usize, usize)) -> GlobalSignature {
        let (index, size) = *index_and_size;

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
            if let Instruction::Enter { .. } = opcode {
                enter_offset = Some(offset as i32);
                break;
            }
        }

        GlobalSignature {
            previous_opcodes: prev_opcode_sig,
            enter_offset,
            size,
            hint: index,
        }
    }

    fn get_global_index(op: &Instruction) -> Option<u32> {
        match op {
            Instruction::GlobalU16 { index } => Some(*index as u32),
            Instruction::GlobalU16Store { index } => Some(*index as u32),
            Instruction::GlobalU16Load { index } => Some(*index as u32),
            Instruction::GlobalU24 { index } => Some(*index),
            Instruction::GlobalU24Store { index } => Some(*index),
            Instruction::GlobalU24Load { index } => Some(*index),
            _ => None,
        }
    }

    fn get_global_offset_value(op: &Instruction) -> Option<u32> {
        match op {
            Instruction::IoffsetU8 { offset } => Some(*offset as u32),
            Instruction::IoffsetU8Load { offset } => Some(*offset as u32),
            Instruction::IoffsetU8Store { offset } => Some(*offset as u32),
            Instruction::IoffsetS16 { offset } => Some(*offset as u32),
            Instruction::IoffsetS16Load { offset } => Some(*offset as u32),
            Instruction::IoffsetS16Store { offset } => Some(*offset as u32),
            Instruction::PushConstU24 { num } => Some(*num),
            Instruction::ArrayU8 { size } => Some(*size as u32),
            Instruction::ArrayU8Store { size } => Some(*size as u32),
            Instruction::ArrayU8Load { size } => Some(*size as u32),
            Instruction::ArrayU16 { size } => Some(*size as u32),
            Instruction::ArrayU16Store { size } => Some(*size as u32),
            Instruction::ArrayU16Load { size } => Some(*size as u32),
            _ => None,
        }
    }

    /// Find multiple locations and sizes of GlobalReferences in a script from the beginning
    pub fn find_global_references(
        &self,
        global_ref: &GlobalReference,
    ) -> Option<Vec<(usize, usize)>> {
        let mut current_offset = 0;
        const NUM_OF_REFERENCES: usize = 7;
        let mut references = Vec::with_capacity(NUM_OF_REFERENCES);
        for _ in 0..NUM_OF_REFERENCES {
            if let Some(reference) = self.find_global_reference_from(global_ref, current_offset) {
                references.push(reference);
                current_offset = reference.0 + reference.1 + 1;
            } else {
                break;
            }
        }

        if references.is_empty() {
            None
        } else {
            Some(references)
        }
    }

    /// Find locations and sizes of GlobalReferences in a script from the beginning
    pub fn find_global_reference(&self, global_ref: &GlobalReference) -> Option<(usize, usize)> {
        self.find_global_reference_from(global_ref, 0)
    }

    /// Find locations and sizes of GlobalReferences in a script given the starting index
    pub fn find_global_reference_from(
        &self,
        global_ref: &GlobalReference,
        from: usize,
    ) -> Option<(usize, usize)> {
        let mut i = from;
        let len = self.instructions.len();
        let mut in_global = false;
        let empty = global_ref.offsets.is_empty();
        let mut offset_index = 0;

        while i < len {
            if let Some(global_value) = InstructionList::get_global_index(&self.instructions[i])
            {
                in_global = global_value == global_ref.index;
                if empty && in_global {
                    return Some((i, 1));
                }
            } else if let Some(offset) =
                InstructionList::get_global_offset_value(&self.instructions[i])
            {
                if in_global && !empty {
                    if global_ref.offsets[offset_index] == offset {
                        offset_index += 1;
                        if offset_index == global_ref.offsets.len() {
                            return Some((
                                i - global_ref.offsets.len(),
                                1 + global_ref.offsets.len(),
                            ));
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

            i += 1;
        }

        None
    }
}

/// ScriptVM disassembler
pub struct Disassembler<'a> {
    script: &'a YSCScript,
    current_stack_top: i64,
}

impl<'a> Disassembler<'a> {
    /// Creates a disassembler given a borrowed YSCScript
    /// Disassembler needs to access the code blocks, string table, native table, etc.. of the script, but won't mutate it
    pub fn new(script: &'a YSCScript) -> Self {
        Self {
            script,
            current_stack_top: 0,
        }
    }

    /// Disassembles a function, or an entire script (if func_index is None)
    pub fn disassemble(&mut self, func_index: Option<usize>) -> Result<InstructionList, DisassembleError> {
        let mut cursor = Cursor::new(&self.script.code[..]);
        cursor.set_position(0);

        let mut current_func_index = 0;

        let mut opcodes = Vec::new();
        let mut function_table = HashMap::<usize, usize>::new();

        let has_func_index = func_index.is_some();

        let mut inst;

        while cursor.position() < self.script.code.len() as u64 {
            let raw = RawOpcode::try_from(cursor.read_u8()?)?;

            let mut is_enter_opcode = false;
            if let RawOpcode::Enter = &raw {
                function_table.insert(
                    (cursor.position() - 1) as usize, /* remove 1 byte because opcode was read */
                    current_func_index,
                );

                if has_func_index && current_func_index > func_index.unwrap() {
                    break;
                }
                is_enter_opcode = true;
            }

            inst = self.disassemble_opcode(&raw, &mut cursor)?;

            if is_enter_opcode {
                if let Instruction::Enter { index, .. } = &mut inst {
                    *index = Some(current_func_index);
                }
                current_func_index += 1;
            }

            if !has_func_index || current_func_index == func_index.unwrap() {
                opcodes.push(inst);
            }
        }

        for opcode in &mut opcodes {
            if let Instruction::Call {
                func_index,
                location,
            } = opcode
            {
                let index = function_table.get(&(*location as usize)).copied();
                *func_index = index;
            }
        }

        Ok(InstructionList::new(opcodes))
    }

    fn disassemble_opcode(
        &mut self,
        raw_op: &RawOpcode,
        cursor: &mut Cursor<&[u8]>,
    ) -> Result<Instruction, DisassembleError> {
        match raw_op {
            RawOpcode::Nop => Ok(Instruction::Nop),
            RawOpcode::Iadd => Ok(Instruction::Iadd),
            RawOpcode::Isub => Ok(Instruction::Isub),
            RawOpcode::Imul => Ok(Instruction::Imul),
            RawOpcode::Idiv => Ok(Instruction::Idiv),
            RawOpcode::Imod => Ok(Instruction::Imod),
            RawOpcode::Inot => Ok(Instruction::Inot),
            RawOpcode::Ineg => Ok(Instruction::Ineg),
            RawOpcode::Ieq => Ok(Instruction::Ieq),
            RawOpcode::Ine => Ok(Instruction::Ine),
            RawOpcode::Igt => Ok(Instruction::Igt),
            RawOpcode::Ige => Ok(Instruction::Ige),
            RawOpcode::Ilt => Ok(Instruction::Ilt),
            RawOpcode::Ile => Ok(Instruction::Ile),
            RawOpcode::Fadd => Ok(Instruction::Fadd),
            RawOpcode::Fsub => Ok(Instruction::Fsub),
            RawOpcode::Fmul => Ok(Instruction::Fmul),
            RawOpcode::Fdiv => Ok(Instruction::Fdiv),
            RawOpcode::Fmod => Ok(Instruction::Fmod),
            RawOpcode::Fneg => Ok(Instruction::Fneg),
            RawOpcode::Feq => Ok(Instruction::Feq),
            RawOpcode::Fne => Ok(Instruction::Fne),
            RawOpcode::Fgt => Ok(Instruction::Fgt),
            RawOpcode::Fge => Ok(Instruction::Fge),
            RawOpcode::Flt => Ok(Instruction::Flt),
            RawOpcode::Fle => Ok(Instruction::Fle),
            RawOpcode::Vadd => Ok(Instruction::Vadd),
            RawOpcode::Vsub => Ok(Instruction::Vsub),
            RawOpcode::Vmul => Ok(Instruction::Vmul),
            RawOpcode::Vdiv => Ok(Instruction::Vdiv),
            RawOpcode::Vneg => Ok(Instruction::Vneg),
            RawOpcode::Iand => Ok(Instruction::Iand),
            RawOpcode::Ior => Ok(Instruction::Ior),
            RawOpcode::Ixor => Ok(Instruction::Ixor),
            RawOpcode::I2F => Ok(Instruction::I2f),
            RawOpcode::F2I => Ok(Instruction::F2i),
            RawOpcode::F2V => Ok(Instruction::F2v),
            RawOpcode::PushConstU8 => {
                self.current_stack_top = cursor.read_u8()? as i64;
                Ok(Instruction::PushConstU8 {
                    one: self.current_stack_top as u8,
                })
            }
            RawOpcode::PushConstU8U8 => Ok(Instruction::PushConstU8U8 {
                one: cursor.read_u8()?,
                two: cursor.read_u8()?,
            }),
            RawOpcode::PushConstU8U8U8 => Ok(Instruction::PushConstU8U8U8 {
                one: cursor.read_u8()?,
                two: cursor.read_u8()?,
                three: cursor.read_u8()?,
            }),
            RawOpcode::PushConstU32 => {
                self.current_stack_top = cursor.read_u32::<LittleEndian>()? as i64;
                Ok(Instruction::PushConstU32 {
                    one: self.current_stack_top as u32,
                })
            }
            RawOpcode::PushConstF => Ok(Instruction::PushConstF {
                one: cursor.read_f32::<LittleEndian>()?,
            }),
            RawOpcode::Dup => Ok(Instruction::Dup),
            RawOpcode::Drop => Ok(Instruction::Drop),
            RawOpcode::Native => {
                let packed_args_and_returns = cursor.read_u8()?;
                let num_args = packed_args_and_returns >> 2;
                let num_returns = packed_args_and_returns & 0b11;
                let native_table_index = cursor.read_u16::<BigEndian>()?;
                Ok(Instruction::Native {
                    num_args,
                    num_returns,
                    native_table_index,
                    native_hash: self.script.native_table[native_table_index as usize],
                })
            }
            RawOpcode::Enter => {
                let arg_count = cursor.read_u8()?;
                let stack_variables = cursor.read_u16::<LittleEndian>()?;
                let skip = cursor.read_u8()?;
                cursor.seek(SeekFrom::Current(skip as i64))?;
                Ok(Instruction::Enter {
                    arg_count,
                    stack_variables,
                    skip,
                    index: None,
                })
            }
            RawOpcode::Leave => Ok(Instruction::Leave {
                arg_count: cursor.read_u8()?,
                return_address_index: cursor.read_u8()?,
            }),
            RawOpcode::Load => Ok(Instruction::Load),
            RawOpcode::Store => Ok(Instruction::Store),
            RawOpcode::StoreRev => Ok(Instruction::StoreRev),
            RawOpcode::LoadN => Ok(Instruction::LoadN),
            RawOpcode::StoreN => Ok(Instruction::StoreN),
            RawOpcode::ArrayU8 => Ok(Instruction::ArrayU8 {
                size: cursor.read_u8()?,
            }),
            RawOpcode::ArrayU8Load => Ok(Instruction::ArrayU8Load {
                size: cursor.read_u8()?,
            }),
            RawOpcode::ArrayU8Store => Ok(Instruction::ArrayU8Store {
                size: cursor.read_u8()?,
            }),
            RawOpcode::LocalU8 => Ok(Instruction::LocalU8 {
                frame_index: cursor.read_u8()?,
            }),
            RawOpcode::LocalU8Load => Ok(Instruction::LocalU8Load {
                frame_index: cursor.read_u8()?,
            }),
            RawOpcode::LocalU8Store => Ok(Instruction::LocalU8Store {
                frame_index: cursor.read_u8()?,
            }),
            RawOpcode::StaticU8 => Ok(Instruction::StaticU8 {
                static_var_index: cursor.read_u8()?,
            }),
            RawOpcode::StaticU8Load => Ok(Instruction::StaticU8Load {
                static_var_index: cursor.read_u8()?,
            }),
            RawOpcode::StaticU8Store => Ok(Instruction::StaticU8Store {
                static_var_index: cursor.read_u8()?,
            }),
            RawOpcode::IaddU8 => Ok(Instruction::IaddU8 {
                num: cursor.read_u8()?,
            }),
            RawOpcode::ImulU8 => Ok(Instruction::ImulU8 {
                num: cursor.read_u8()?,
            }),
            RawOpcode::Ioffset => Ok(Instruction::Ioffset),
            RawOpcode::IoffsetU8 => Ok(Instruction::IoffsetU8 {
                offset: cursor.read_u8()?,
            }),
            RawOpcode::IoffsetU8Load => Ok(Instruction::IoffsetU8Load {
                offset: cursor.read_u8()?,
            }),
            RawOpcode::IoffsetU8Store => Ok(Instruction::IoffsetU8Store {
                offset: cursor.read_u8()?,
            }),
            RawOpcode::PushConstS16 => {
                self.current_stack_top = cursor.read_i16::<LittleEndian>()? as i64;
                Ok(Instruction::PushConstS16 {
                    num: self.current_stack_top as i16,
                })
            }
            RawOpcode::IaddS16 => Ok(Instruction::IaddS16 {
                num: cursor.read_i16::<LittleEndian>()?,
            }),
            RawOpcode::ImulS16 => Ok(Instruction::ImulS16 {
                num: cursor.read_i16::<LittleEndian>()?,
            }),
            RawOpcode::IoffsetS16 => Ok(Instruction::IoffsetS16 {
                offset: cursor.read_i16::<LittleEndian>()?,
            }),
            RawOpcode::IoffsetS16Load => Ok(Instruction::IoffsetS16Load {
                offset: cursor.read_i16::<LittleEndian>()?,
            }),
            RawOpcode::IoffsetS16Store => Ok(Instruction::IoffsetS16Store {
                offset: cursor.read_i16::<LittleEndian>()?,
            }),
            RawOpcode::ArrayU16 => Ok(Instruction::ArrayU16 {
                size: cursor.read_u16::<LittleEndian>()?,
            }),
            RawOpcode::ArrayU16Load => Ok(Instruction::ArrayU16Load {
                size: cursor.read_u16::<LittleEndian>()?,
            }),
            RawOpcode::ArrayU16Store => Ok(Instruction::ArrayU16Store {
                size: cursor.read_u16::<LittleEndian>()?,
            }),
            RawOpcode::LocalU16 => Ok(Instruction::LocalU16 {
                frame_index: cursor.read_u16::<LittleEndian>()?,
            }),
            RawOpcode::LocalU16Load => Ok(Instruction::LocalU16Load {
                frame_index: cursor.read_u16::<LittleEndian>()?,
            }),
            RawOpcode::LocalU16Store => Ok(Instruction::LocalU16Store {
                frame_index: cursor.read_u16::<LittleEndian>()?,
            }),
            RawOpcode::StaticU16 => Ok(Instruction::StaticU16 {
                static_var_index: cursor.read_u16::<LittleEndian>()?,
            }),
            RawOpcode::StaticU16Load => Ok(Instruction::StaticU16Load {
                static_var_index: cursor.read_u16::<LittleEndian>()?,
            }),
            RawOpcode::StaticU16Store => Ok(Instruction::StaticU16Store {
                static_var_index: cursor.read_u16::<LittleEndian>()?,
            }),
            RawOpcode::GlobalU16 => Ok(Instruction::GlobalU16 {
                index: cursor.read_u16::<LittleEndian>()?,
            }),
            RawOpcode::GlobalU16Load => Ok(Instruction::GlobalU16Load {
                index: cursor.read_u16::<LittleEndian>()?,
            }),
            RawOpcode::GlobalU16Store => Ok(Instruction::GlobalU16Store {
                index: cursor.read_u16::<LittleEndian>()?,
            }),
            RawOpcode::J => Ok(Instruction::J {
                offset: cursor.read_i16::<LittleEndian>()?,
            }),
            RawOpcode::Jz => Ok(Instruction::Jz {
                offset: cursor.read_i16::<LittleEndian>()?,
            }),
            RawOpcode::IeqJz => Ok(Instruction::IEqJz {
                offset: cursor.read_i16::<LittleEndian>()?,
            }),
            RawOpcode::IneJz => Ok(Instruction::INeJz {
                offset: cursor.read_i16::<LittleEndian>()?,
            }),
            RawOpcode::IgtJz => Ok(Instruction::IGtJz {
                offset: cursor.read_i16::<LittleEndian>()?,
            }),
            RawOpcode::IgeJz => Ok(Instruction::IGeJz {
                offset: cursor.read_i16::<LittleEndian>()?,
            }),
            RawOpcode::IltJz => Ok(Instruction::ILtJz {
                offset: cursor.read_i16::<LittleEndian>()?,
            }),
            RawOpcode::IleJz => Ok(Instruction::ILeJz {
                offset: cursor.read_i16::<LittleEndian>()?,
            }),
            RawOpcode::Call => Ok(Instruction::Call {
                location: cursor.read_u24::<LittleEndian>()?,
                func_index: None,
            }),
            RawOpcode::LocalU24 => Ok(Instruction::LocalU24 {
                frame_index: cursor.read_u24::<LittleEndian>()?,
            }),
            RawOpcode::LocalU24Load => Ok(Instruction::LocalU24Load {
                frame_index: cursor.read_u24::<LittleEndian>()?,
            }),
            RawOpcode::LocalU24Store => Ok(Instruction::LocalU24Store {
                frame_index: cursor.read_u24::<LittleEndian>()?,
            }),
            RawOpcode::GlobalU24 => Ok(Instruction::GlobalU24 {
                index: cursor.read_u24::<LittleEndian>()?,
            }),
            RawOpcode::GlobalU24Load => Ok(Instruction::GlobalU24Load {
                index: cursor.read_u24::<LittleEndian>()?,
            }),
            RawOpcode::GlobalU24Store => Ok(Instruction::GlobalU24Store {
                index: cursor.read_u24::<LittleEndian>()?,
            }),
            RawOpcode::PushConstU24 => {
                self.current_stack_top = cursor.read_u24::<LittleEndian>()? as i64;
                Ok(Instruction::PushConstU24 {
                    num: self.current_stack_top as u32,
                })
            }
            RawOpcode::Switch => {
                let num_entries = cursor.read_u8()?;
                let mut entries = Vec::new();
                for _ in 0..num_entries {
                    let entry = SwitchEntry {
                        index_id: cursor.read_u32::<LittleEndian>()?,
                        jump_offset: cursor.read_u16::<LittleEndian>()?,
                    };

                    // Suppress unused warnings
                    let _ = entry.index_id;
                    let _ = entry.jump_offset;

                    entries.push(entry);
                }

                Ok(Instruction::Switch {
                    num_of_entries: num_entries,
                    entries,
                })
            }
            RawOpcode::String => {
                let key = &(self.current_stack_top as usize);
                if self.script.strings.contains_key(key) {
                    Ok(Instruction::String {
                        index: *key,
                        value: self.script.strings[key].to_string(),
                    })
                } else {
                    Ok(Instruction::String {
                        index: *key,
                        value: self
                            .script
                            .get_string_with_index(*key)
                            .unwrap_or("!!! [disassembler] Unable to find string !!!".to_owned()),
                    })
                }
            }
            RawOpcode::StringHash => Ok(Instruction::StringHash),
            RawOpcode::TextLabelAssignString => Ok(Instruction::TextLabelAssignString {
                size: cursor.read_u8()?,
            }),
            RawOpcode::TextLabelAssignInt => Ok(Instruction::TextLabelAssignInt {
                size: cursor.read_u8()?,
            }),
            RawOpcode::TextLabelAppendString => Ok(Instruction::TextLabelAppendString {
                size: cursor.read_u8()?,
            }),
            RawOpcode::TextLabelAppendInt => Ok(Instruction::TextLabelAppendInt {
                size: cursor.read_u8()?,
            }),
            RawOpcode::TextLabelCopy => Ok(Instruction::TextLabelCopy),
            RawOpcode::Catch => Ok(Instruction::Catch),
            RawOpcode::Throw => Ok(Instruction::Throw),
            RawOpcode::CallIndirect => Ok(Instruction::CallIndirect),
            RawOpcode::PushConstM1 => Ok(Instruction::PushConstM1),
            RawOpcode::PushConst0 => {
                self.current_stack_top = 0;
                Ok(Instruction::PushConst0)
            }
            RawOpcode::PushConst1 => {
                self.current_stack_top = 1;
                Ok(Instruction::PushConst1)
            }
            RawOpcode::PushConst2 => {
                self.current_stack_top = 2;
                Ok(Instruction::PushConst2)
            }
            RawOpcode::PushConst3 => {
                self.current_stack_top = 3;
                Ok(Instruction::PushConst3)
            }
            RawOpcode::PushConst4 => {
                self.current_stack_top = 4;
                Ok(Instruction::PushConst4)
            }
            RawOpcode::PushConst5 => {
                self.current_stack_top = 5;
                Ok(Instruction::PushConst5)
            }
            RawOpcode::PushConst6 => {
                self.current_stack_top = 6;
                Ok(Instruction::PushConst6)
            }
            RawOpcode::PushConst7 => {
                self.current_stack_top = 7;
                Ok(Instruction::PushConst7)
            }
            RawOpcode::PushConstFm1 => Ok(Instruction::PushConstFM1),
            RawOpcode::PushConstF0 => Ok(Instruction::PushConstF0),
            RawOpcode::PushConstF1 => Ok(Instruction::PushConstF1),
            RawOpcode::PushConstF2 => Ok(Instruction::PushConstF2),
            RawOpcode::PushConstF3 => Ok(Instruction::PushConstF3),
            RawOpcode::PushConstF4 => Ok(Instruction::PushConstF4),
            RawOpcode::PushConstF5 => Ok(Instruction::PushConstF5),
            RawOpcode::PushConstF6 => Ok(Instruction::PushConstF6),
            RawOpcode::PushConstF7 => Ok(Instruction::PushConstF7),
            RawOpcode::IsBitSet => Ok(Instruction::IsBitSet),
        }
    }
}

#[allow(missing_docs)]
#[derive(Debug, EnumIndex, Clone)]
pub enum Instruction {
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
    PushConstU8 {
        one: u8,
    },
    PushConstU8U8 {
        one: u8,
        two: u8,
    },
    PushConstU8U8U8 {
        one: u8,
        two: u8,
        three: u8,
    },
    PushConstU32 {
        one: u32,
    },
    PushConstF {
        one: f32,
    },
    Dup,
    Drop,
    Native {
        num_args: u8,
        num_returns: u8,
        native_table_index: u16,
        native_hash: u64,
    },
    Enter {
        arg_count: u8,
        stack_variables: u16,
        skip: u8,
        index: Option<usize>,
    },
    Leave {
        arg_count: u8,
        return_address_index: u8,
    },
    Load,
    Store,
    StoreRev,
    LoadN,
    StoreN,
    ArrayU8 {
        size: u8,
    },
    ArrayU8Load {
        size: u8,
    },
    ArrayU8Store {
        size: u8,
    },
    LocalU8 {
        frame_index: u8,
    },
    LocalU8Load {
        frame_index: u8,
    },
    LocalU8Store {
        frame_index: u8,
    },
    StaticU8 {
        static_var_index: u8,
    },
    StaticU8Load {
        static_var_index: u8,
    },
    StaticU8Store {
        static_var_index: u8,
    },
    IaddU8 {
        num: u8,
    },
    ImulU8 {
        num: u8,
    },
    Ioffset,
    IoffsetU8 {
        offset: u8,
    },
    IoffsetU8Load {
        offset: u8,
    },
    IoffsetU8Store {
        offset: u8,
    },
    PushConstS16 {
        num: i16,
    },
    IaddS16 {
        num: i16,
    },
    ImulS16 {
        num: i16,
    },
    IoffsetS16 {
        offset: i16,
    },
    IoffsetS16Load {
        offset: i16,
    },
    IoffsetS16Store {
        offset: i16,
    },
    ArrayU16 {
        size: u16,
    },
    ArrayU16Load {
        size: u16,
    },
    ArrayU16Store {
        size: u16,
    },
    LocalU16 {
        frame_index: u16,
    },
    LocalU16Load {
        frame_index: u16,
    },
    LocalU16Store {
        frame_index: u16,
    },
    StaticU16 {
        static_var_index: u16,
    },
    StaticU16Load {
        static_var_index: u16,
    },
    StaticU16Store {
        static_var_index: u16,
    },
    GlobalU16 {
        index: u16,
    },
    GlobalU16Load {
        index: u16,
    },
    GlobalU16Store {
        index: u16,
    },
    J {
        offset: i16,
    },
    Jz {
        offset: i16,
    },
    IEqJz {
        offset: i16,
    },
    INeJz {
        offset: i16,
    },
    IGtJz {
        offset: i16,
    },
    IGeJz {
        offset: i16,
    },
    ILtJz {
        offset: i16,
    },
    ILeJz {
        offset: i16,
    },
    Call {
        location: u32,
        func_index: Option<usize>,
    },
    LocalU24 {
        frame_index: u32,
    },
    LocalU24Load {
        frame_index: u32,
    },
    LocalU24Store {
        frame_index: u32,
    },
    GlobalU24 {
        index: u32,
    },
    GlobalU24Load {
        index: u32,
    },
    GlobalU24Store {
        index: u32,
    },
    PushConstU24 {
        num: u32,
    },
    Switch {
        num_of_entries: u8,
        entries: Vec<SwitchEntry>,
    },
    String {
        index: usize,
        value: String,
    },
    StringHash,
    TextLabelAssignString {
        size: u8,
    },
    TextLabelAssignInt {
        size: u8,
    },
    TextLabelAppendString {
        size: u8,
    },
    TextLabelAppendInt {
        size: u8,
    },
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
    IsBitSet,
}

impl Instruction {
    /// Returns the offset of a jump instrution, if it is one
    pub fn get_jump_offset(&self) -> Option<i16> {
        match &self {
            Instruction::Jz { offset } => Some(*offset),
            Instruction::ILtJz { offset } => Some(*offset),
            Instruction::ILeJz { offset } => Some(*offset),
            Instruction::INeJz { offset } => Some(*offset),
            Instruction::IEqJz { offset } => Some(*offset),
            Instruction::IGtJz { offset } => Some(*offset),
            Instruction::IGeJz { offset } => Some(*offset),
            Instruction::J { offset } => Some(*offset),
            _ => None,
        }
    }

    /// Returns the size (in bytes) of the opcode
    pub fn get_size(&self) -> usize {
        match &self {
            Instruction::Nop => 1,
            Instruction::Iadd => 1,
            Instruction::Isub => 1,
            Instruction::Imul => 1,
            Instruction::Idiv => 1,
            Instruction::Imod => 1,
            Instruction::Inot => 1,
            Instruction::Ineg => 1,
            Instruction::Ieq => 1,
            Instruction::Ine => 1,
            Instruction::Igt => 1,
            Instruction::Ige => 1,
            Instruction::Ilt => 1,
            Instruction::Ile => 1,
            Instruction::Fadd => 1,
            Instruction::Fsub => 1,
            Instruction::Fmul => 1,
            Instruction::Fdiv => 1,
            Instruction::Fmod => 1,
            Instruction::Fneg => 1,
            Instruction::Feq => 1,
            Instruction::Fne => 1,
            Instruction::Fgt => 1,
            Instruction::Fge => 1,
            Instruction::Flt => 1,
            Instruction::Fle => 1,
            Instruction::Vadd => 1,
            Instruction::Vsub => 1,
            Instruction::Vmul => 1,
            Instruction::Vdiv => 1,
            Instruction::Vneg => 1,
            Instruction::Iand => 1,
            Instruction::Ior => 1,
            Instruction::Ixor => 1,
            Instruction::I2f => 1,
            Instruction::F2i => 1,
            Instruction::F2v => 1,
            Instruction::PushConstU8 { .. } => 2,
            Instruction::PushConstU8U8 { .. } => 3,
            Instruction::PushConstU8U8U8 { .. } => 4,
            Instruction::PushConstU32 { .. } => 5,
            Instruction::PushConstF { .. } => 5,
            Instruction::Dup => 1,
            Instruction::Drop => 1,
            Instruction::Native { .. } => 4,
            Instruction::Enter { skip, .. } => 5 + *skip as usize,
            Instruction::Leave { .. } => 3,
            Instruction::Load => 1,
            Instruction::Store => 1,
            Instruction::StoreRev => 1,
            Instruction::LoadN => 1,
            Instruction::StoreN => 1,
            Instruction::ArrayU8 { .. } => 2,
            Instruction::ArrayU8Load { .. } => 2,
            Instruction::ArrayU8Store { .. } => 2,
            Instruction::LocalU8 { .. } => 2,
            Instruction::LocalU8Load { .. } => 2,
            Instruction::LocalU8Store { .. } => 2,
            Instruction::StaticU8 { .. } => 2,
            Instruction::StaticU8Load { .. } => 2,
            Instruction::StaticU8Store { .. } => 2,
            Instruction::IaddU8 { .. } => 2,
            Instruction::ImulU8 { .. } => 2,
            Instruction::Ioffset => 1,
            Instruction::IoffsetU8 { .. } => 2,
            Instruction::IoffsetU8Load { .. } => 2,
            Instruction::IoffsetU8Store { .. } => 2,
            Instruction::PushConstS16 { .. } => 3,
            Instruction::IaddS16 { .. } => 3,
            Instruction::ImulS16 { .. } => 3,
            Instruction::IoffsetS16 { .. } => 3,
            Instruction::IoffsetS16Load { .. } => 3,
            Instruction::IoffsetS16Store { .. } => 3,
            Instruction::ArrayU16 { .. } => 3,
            Instruction::ArrayU16Load { .. } => 3,
            Instruction::ArrayU16Store { .. } => 3,
            Instruction::LocalU16 { .. } => 3,
            Instruction::LocalU16Load { .. } => 3,
            Instruction::LocalU16Store { .. } => 3,
            Instruction::StaticU16 { .. } => 3,
            Instruction::StaticU16Load { .. } => 3,
            Instruction::StaticU16Store { .. } => 3,
            Instruction::GlobalU16 { .. } => 3,
            Instruction::GlobalU16Load { .. } => 3,
            Instruction::GlobalU16Store { .. } => 3,
            Instruction::J { .. } => 3,
            Instruction::Jz { .. } => 3,
            Instruction::IEqJz { .. } => 3,
            Instruction::INeJz { .. } => 3,
            Instruction::IGtJz { .. } => 3,
            Instruction::IGeJz { .. } => 3,
            Instruction::ILtJz { .. } => 3,
            Instruction::ILeJz { .. } => 3,
            Instruction::Call { .. } => 4,
            Instruction::LocalU24 { .. } => 4,
            Instruction::LocalU24Load { .. } => 4,
            Instruction::LocalU24Store { .. } => 4,
            Instruction::GlobalU24 { .. } => 4,
            Instruction::GlobalU24Load { .. } => 4,
            Instruction::GlobalU24Store { .. } => 4,
            Instruction::PushConstU24 { .. } => 4,
            Instruction::Switch { num_of_entries, .. } => *num_of_entries as usize * 6 + 2,
            Instruction::String { .. } => 1,
            Instruction::StringHash => 1,
            Instruction::TextLabelAssignString { .. } => 2,
            Instruction::TextLabelAssignInt { .. } => 2,
            Instruction::TextLabelAppendString { .. } => 2,
            Instruction::TextLabelAppendInt { .. } => 2,
            Instruction::TextLabelCopy => 1,
            Instruction::Catch => 1,
            Instruction::Throw => 1,
            Instruction::CallIndirect => 1,
            Instruction::PushConstM1 => 1,
            Instruction::PushConst0 => 1,
            Instruction::PushConst1 => 1,
            Instruction::PushConst2 => 1,
            Instruction::PushConst3 => 1,
            Instruction::PushConst4 => 1,
            Instruction::PushConst5 => 1,
            Instruction::PushConst6 => 1,
            Instruction::PushConst7 => 1,
            Instruction::PushConstFM1 => 1,
            Instruction::PushConstF0 => 1,
            Instruction::PushConstF1 => 1,
            Instruction::PushConstF2 => 1,
            Instruction::PushConstF3 => 1,
            Instruction::PushConstF4 => 1,
            Instruction::PushConstF5 => 1,
            Instruction::PushConstF6 => 1,
            Instruction::PushConstF7 => 1,
            Instruction::IsBitSet => 1,
        }
    }

    /// Returns the number of items the opcode attempts to push to the stack
    /// Some opcodes (like LoadN) make this not guaranteed to be known
    /// This does not include the amount of items the instruction pops from the stack as the purpose is to be able to quickly iterate through a function backwards, to see the number of returns
    pub fn get_stack_size(&self) -> Option<isize> {
        let val = match &self {
            Instruction::Nop => 0,
            Instruction::Iadd => 1,
            Instruction::Isub => 1,
            Instruction::Imul => 1,
            Instruction::Idiv => 1,
            Instruction::Imod => 1,
            Instruction::Inot => 1,
            Instruction::Ineg => 1,
            Instruction::Ieq => 1,
            Instruction::Ine => 1,
            Instruction::Igt => 1,
            Instruction::Ige => 1,
            Instruction::Ilt => 1,
            Instruction::Ile => 1,
            Instruction::Fadd => 1,
            Instruction::Fsub => 1,
            Instruction::Fmul => 1,
            Instruction::Fdiv => 1,
            Instruction::Fmod => 1,
            Instruction::Fneg => 1,
            Instruction::Feq => 1,
            Instruction::Fne => 1,
            Instruction::Fgt => 1,
            Instruction::Fge => 1,
            Instruction::Flt => 1,
            Instruction::Fle => 1,
            Instruction::Vadd => 3,
            Instruction::Vsub => 3,
            Instruction::Vmul => 3,
            Instruction::Vdiv => 3,
            Instruction::Vneg => 3,
            Instruction::Iand => 1,
            Instruction::Ior => 1,
            Instruction::Ixor => 1,
            Instruction::I2f => 1,
            Instruction::F2i => 1,
            Instruction::F2v => 1,
            Instruction::PushConstU8 { .. } => 1,
            Instruction::PushConstU8U8 { .. } => 2,
            Instruction::PushConstU8U8U8 { .. } => 3,
            Instruction::PushConstU32 { .. } => 1,
            Instruction::PushConstF { .. } => 1,
            Instruction::Dup => 1,
            Instruction::Drop => 0,
            Instruction::Native { num_returns, .. } => *num_returns as isize,
            Instruction::Enter { .. } => 0,
            Instruction::Leave { .. } => 0,
            Instruction::Load => 1,
            Instruction::Store => 1,
            Instruction::StoreRev => 1,
            Instruction::LoadN => 1,
            Instruction::StoreN => 1,
            Instruction::ArrayU8 { .. } => 1,
            Instruction::ArrayU8Load { .. } => 1,
            Instruction::ArrayU8Store { .. } => 0,
            Instruction::LocalU8 { .. } => 1,
            Instruction::LocalU8Load { .. } => 1,
            Instruction::LocalU8Store { .. } => 0,
            Instruction::StaticU8 { .. } => 1,
            Instruction::StaticU8Load { .. } => 1,
            Instruction::StaticU8Store { .. } => 0,
            Instruction::IaddU8 { .. } => 1,
            Instruction::ImulU8 { .. } => 1,
            Instruction::Ioffset => 1,
            Instruction::IoffsetU8 { .. } => 1,
            Instruction::IoffsetU8Load { .. } => 1,
            Instruction::IoffsetU8Store { .. } => 0,
            Instruction::PushConstS16 { .. } => 1,
            Instruction::IaddS16 { .. } => 1,
            Instruction::ImulS16 { .. } => 1,
            Instruction::IoffsetS16 { .. } => 1,
            Instruction::IoffsetS16Load { .. } => 1,
            Instruction::IoffsetS16Store { .. } => 0,
            Instruction::ArrayU16 { .. } => 1,
            Instruction::ArrayU16Load { .. } => 1,
            Instruction::ArrayU16Store { .. } => 0,
            Instruction::LocalU16 { .. } => 1,
            Instruction::LocalU16Load { .. } => 1,
            Instruction::LocalU16Store { .. } => 0,
            Instruction::StaticU16 { .. } => 1,
            Instruction::StaticU16Load { .. } => 1,
            Instruction::StaticU16Store { .. } => 0,
            Instruction::GlobalU16 { .. } => 1,
            Instruction::GlobalU16Load { .. } => 1,
            Instruction::GlobalU16Store { .. } => 0,
            Instruction::J { .. } => -1,
            Instruction::Jz { .. } => -1,
            Instruction::IEqJz { .. } => -1,
            Instruction::INeJz { .. } => -1,
            Instruction::IGtJz { .. } => -1,
            Instruction::IGeJz { .. } => -1,
            Instruction::ILtJz { .. } => -1,
            Instruction::ILeJz { .. } => -1,
            Instruction::Call { .. } => -1,
            Instruction::LocalU24 { .. } => 1,
            Instruction::LocalU24Load { .. } => 1,
            Instruction::LocalU24Store { .. } => 0,
            Instruction::GlobalU24 { .. } => 1,
            Instruction::GlobalU24Load { .. } => 1,
            Instruction::GlobalU24Store { .. } => 0,
            Instruction::PushConstU24 { .. } => 1,
            Instruction::Switch { .. } => -1,
            Instruction::String { .. } => 1,
            Instruction::StringHash => 1,
            Instruction::TextLabelAssignString { .. } => 1,
            Instruction::TextLabelAssignInt { .. } => 1,
            Instruction::TextLabelAppendString { .. } => 1,
            Instruction::TextLabelAppendInt { .. } => 1,
            Instruction::TextLabelCopy => 1,
            Instruction::Catch => -1,
            Instruction::Throw => -1,
            Instruction::CallIndirect => -1,
            Instruction::PushConstM1 => 1,
            Instruction::PushConst0 => 1,
            Instruction::PushConst1 => 1,
            Instruction::PushConst2 => 1,
            Instruction::PushConst3 => 1,
            Instruction::PushConst4 => 1,
            Instruction::PushConst5 => 1,
            Instruction::PushConst6 => 1,
            Instruction::PushConst7 => 1,
            Instruction::PushConstFM1 => 1,
            Instruction::PushConstF0 => 1,
            Instruction::PushConstF1 => 1,
            Instruction::PushConstF2 => 1,
            Instruction::PushConstF3 => 1,
            Instruction::PushConstF4 => 1,
            Instruction::PushConstF5 => 1,
            Instruction::PushConstF6 => 1,
            Instruction::PushConstF7 => 1,
            Instruction::IsBitSet => 1,
        };

        if val == -1 {
            None
        } else {
            Some(val)
        }
    }
}

/// Information about a Switch instruction
#[derive(Debug, Clone)]
pub struct SwitchEntry {
    index_id: u32,
    jump_offset: u16,
}

#[allow(missing_docs)]
#[repr(u8)]
#[derive(Debug, Eq, PartialEq, TryFromPrimitive, IntoPrimitive)]
pub enum RawOpcode {
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
    IsBitSet,
}
