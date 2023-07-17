use bitbuffer::{BitReadBuffer, BitReadStream, LittleEndian, BitError};
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

/// Errors pertaining to reading YSC files
#[derive(Error, Debug)]
pub enum YscError {
    /// Generic deserialization error, most likely invalid format
    #[error("error while reading script file")]
    InvalidOpcodeFormat(#[from] BitError),

    /// Generic io error (ex: file not found)
    #[error("io error")]
    IoError(#[from] std::io::Error),
}

/// YSC header information
#[derive(Default, Debug)]
pub struct YSCHeader {
    /// Not entirely sure
    pub page_base: i32,
    /// Not entirely sure
    pub page_map_ptr: i32,
    /// Start of code blocks
    pub code_block_base_ptr: i32,
    /// Some sort of information pertaining to the version/signature of globals
    pub globals_signature: i32,
    /// Total length of the code
    pub code_size: i32,
    /// Number of parameters for the script
    pub parameter_count: i32,
    /// Number of static/script local variabes
    pub static_count: i32,
    /// Number of script global variables
    pub global_count: i32,
    /// Number of natives at natives_ptr
    pub natives_count: i32,
    /// Location of static variables
    pub statics_ptr: i32,
    /// Location of global variables
    pub globals_ptr: i32,
    /// Location of natives
    pub natives_ptr: i32,
    /// unk
    pub unk_1: i32,
    /// unk
    pub unk_2: i32,
    /// JOAAT hash of the scriot name
    pub script_name_hash: i32,
    /// unk
    pub unk_3: i32,
    /// Location of the script name
    pub script_name_ptr: i32,
    /// Location of the string table
    pub string_blocks_base_ptr: i32,
    /// Total size of strin table
    pub string_size: i32,
    /// unk
    pub unk_4: i32,
    /// unk
    pub unk_5: i32,
    /// unk
    pub unk_6: i32,
}

impl YSCHeader {
    fn new(stream: &mut BitReadStream<'_, LittleEndian>) -> Result<Self, YscError> {
        stream.set_pos(0)?;
        let mut header = YSCHeader::default();

        let read_int = |stream: &mut BitReadStream<'_, LittleEndian>| -> Result<i32, YscError> {
            let res = stream.read_int::<i32>(32)?;
            Ok(res)
        };

        let read_int_large = |stream: &mut BitReadStream<'_, LittleEndian>| -> Result<i32, YscError> {
            let res = stream.read_int::<i32>(32)?;
            stream.skip_bits(32)?;
            Ok(res)
        };

        let read_ptr = |stream: &mut BitReadStream<'_, LittleEndian>| -> Result<i32, YscError> {
            let res = stream.read_int::<i32>(32)? & 0xFFFFFF;
            stream.skip_bits(32)?;
            Ok(res)
        };
        header.page_base = read_int_large(stream)?;
        header.page_map_ptr = read_ptr(stream)?;
        header.code_block_base_ptr = read_ptr(stream)?;
        header.globals_signature = read_int(stream)?;
        header.code_size = read_int(stream)?;
        header.parameter_count = read_int(stream)?;
        header.static_count = read_int(stream)?;
        header.global_count = read_int(stream)?;
        header.natives_count = read_int(stream)?;
        header.statics_ptr = read_ptr(stream)?;
        header.globals_ptr = read_ptr(stream)?;
        header.natives_ptr = read_ptr(stream)?;
        header.unk_1 = read_int_large(stream)?;
        header.unk_2 = read_int_large(stream)?;
        header.script_name_hash = read_int(stream)?;
        header.unk_3 = read_int(stream)?;
        header.script_name_ptr = read_ptr(stream)?;
        header.string_blocks_base_ptr = read_ptr(stream)?;
        header.string_size = read_int_large(stream)?;
        header.unk_4 = read_int_large(stream)?;
        header.unk_5 = read_int_large(stream)?;
        header.unk_6 = read_int_large(stream)?;
        Ok(header)
    }
}

/// On-demand information about a YSC file
#[derive(Debug)]
pub struct YSCReader<'a> {
    stream: BitReadStream<'a, LittleEndian>,
    /// Script header
    pub header: YSCHeader,
}

/// YSC script with data gathered in a useful and structured way
#[derive(Debug)]
pub struct YSCScript {
    /// Plaintext name of script
    pub name: String,
    /// Vec of native hashes
    pub native_table: Vec<u64>,
    /// Vec of string table offsets
    pub string_table_offsets: Vec<u64>,
    /// Vec of code table offsets
    pub code_table_offsets: Vec<u64>,
    /// HashMap with the string's location and value
    pub strings: HashMap<usize, String>,
    /// Large Vec of the entire string table as bytes
    /// Used for if a String instruction indexes into the middle of a string
    pub string_table: Vec<u8>,
    /// Vec of the code bytes of the script
    pub code: Vec<u8>,
}

impl YSCScript {
    /// Create a YSCSCript from a path
    pub fn from_ysc_file(path: impl AsRef<Path>) -> Result<Self, YscError> {
        let src = std::fs::read(path)?;
        let ysc = YSCReader::new(&src)?.get_script()?;
        Ok(ysc)
    }

    /// Attempt to get a string from a certain index, this allows you to index into the middle of strings
    pub fn get_string_with_index(&self, index: usize) -> Option<String> {
        let table = &self.string_table;
        let mut i = index;
        let max = table.len();

        let mut string_buf: Vec<u8> = Vec::with_capacity(100);

        if i >= max {
            return None;
        }

        while i < max {
            while i < max {
                let b = table[i];
                match b {
                    0 => {
                        let converted_str = String::from_utf8_lossy(&string_buf)
                            .replace('\\', "\\\\")
                            .replace('\"', "\\\"");
                        return Some(converted_str);
                    }
                    10 => string_buf.extend_from_slice(&[92, 110]), // newline \n
                    13 => string_buf.extend_from_slice(&[92, 114]), // cartridge return \r
                    34 => string_buf.extend_from_slice(&[92, 34]),  // quotation \"
                    _ => string_buf.push(b),
                }
                i += 1;
            }
        }
        None
    }
}

impl<'a> TryFrom<YSCReader<'a>> for YSCScript {
    type Error = YscError;
    fn try_from(mut value: YSCReader<'a>) -> Result<Self, Self::Error> {
        value.get_script()
    }
}

impl<'a> YSCReader<'a> {
    /// Create a YSCReader given the bytes to the YSC file
    pub fn new(data: &'a [u8]) -> Result<Self, YscError> {
        let buffer = BitReadBuffer::new(data, LittleEndian);
        let mut stream = BitReadStream::new(buffer);
        let header = YSCHeader::new(&mut stream)?;
        Ok(YSCReader { stream, header })
    }

    /// Create a YSCSCript from a YSCReader
    pub fn get_script(&mut self) -> Result<YSCScript, YscError> {
        let string_table = self.get_string_table()?;
        Ok(YSCScript {
            name: self.get_script_name()?,
            native_table: self.get_native_table()?,
            string_table_offsets: self.get_string_table_offsets()?,
            code_table_offsets: self.get_code_table_offsets()?,
            strings: self.get_strings(&string_table)?,
            string_table,
            code: self.get_code()?,
        })
    }

    /// Read current script name
    pub fn get_script_name(&mut self) -> Result<String, YscError> {
        self.stream
            .set_pos(self.header.script_name_ptr as usize * 8)?;

        Ok(self.stream.read_string(None)?.to_string())
    }

    /// Read native table
    pub fn get_native_table(&mut self) -> Result<Vec<u64>, YscError> {
        self.stream.set_pos(self.header.natives_ptr as usize * 8)?;
        let mut natives = vec![];
        for i in 0..self.header.natives_count {
            let encrypted_hash: u64 = self.stream.read_int(64)?;
            let hash = YSCReader::native_hash_rotate(encrypted_hash, (self.header.code_size + i) as u32);
            natives.push(hash);
        }
        Ok(natives)
    }

    /// Read string table offsets
    pub fn get_string_table_offsets(&mut self) -> Result<Vec<u64>, YscError> {
        self.stream
            .set_pos((self.header.string_blocks_base_ptr * 8) as usize)?;
        let string_block_count = (self.header.string_size + 0x3FFF).overflowing_shr(14).0;
        let mut string_tables = Vec::with_capacity(string_block_count as usize);
        for _ in 0..string_block_count {
            string_tables.push(self.stream.read_int::<u64>(32)? & 0xFFFFFF);
            self.stream.skip_bits(32)?;
        }

        Ok(string_tables)
    }

    /// Read code
    pub fn get_code(&mut self) -> Result<Vec<u8>, YscError> {
        let mut code = Vec::new();
        let code_table_offsets = self.get_code_table_offsets()?;

        for (i, offset) in code_table_offsets.iter().enumerate() {
            let table_size = if (i + 1) * 0x4000 >= self.header.code_size as usize {
                self.header.code_size % 0x4000
            } else {
                0x4000
            } as usize;

            let mut current_table: Vec<u8> = Vec::with_capacity(table_size);
            self.stream.set_pos((*offset * 8) as usize)?;
            current_table.extend_from_slice(&self.stream.read_bytes(table_size)?);
            code.append(&mut current_table);
        }

        Ok(code)
    }

    /// Read entire string table
    pub fn get_string_table(&mut self) -> Result<Vec<u8>, YscError> {
        let string_block_count = (self.header.string_size + 0x3FFF).overflowing_shr(14).0;
        let string_table_offsets = self.get_string_table_offsets()?;
        let mut table: Vec<u8> = vec![0; self.header.string_size as usize];

        let mut i = 0;
        let mut off = 0;

        while i < string_block_count {
            let table_size = if (i + 1) * 0x4000 >= self.header.string_size {
                self.header.string_size % 0x4000
            } else {
                0x4000
            };

            self.stream
                .set_pos((string_table_offsets[i as usize] * 8) as usize)?;
            let table_bytes = self.stream.read_bytes(table_size as usize)?.to_vec();
            table[off..off + table_bytes.len()].copy_from_slice(&table_bytes);
            i += 1;
            off += 0x4000;
        }
        Ok(table)
    }

    /// Read strings and get their location
    pub fn get_strings(&mut self, table: &Vec<u8>) -> Result<HashMap<usize, String>, YscError> {
        let mut strings = HashMap::new();
        let mut string_buf: Vec<u8> = Vec::with_capacity(100);

        let mut i = 0;
        let mut index;
        let max = table.len();

        while i < max {
            index = i;
            'inner: while i < max {
                let b = table[i];
                match b {
                    0 => {
                        i += 1;
                        break 'inner;
                    }
                    10 => string_buf.extend_from_slice(&[92, 110]), // newline \n
                    13 => string_buf.extend_from_slice(&[92, 114]), // cartridge return \r
                    34 => string_buf.extend_from_slice(&[92, 34]),  // quotation \"
                    _ => string_buf.push(b),
                }
                i += 1;
            }
            let converted_str = String::from_utf8_lossy(&string_buf)
                .replace('\\', "\\\\")
                .replace('\"', "\\\"");
            strings.insert(index, converted_str.to_string());
            string_buf.clear();
        }

        Ok(strings)
    }

    /// Read offsets to code table blocks
    pub fn get_code_table_offsets(&mut self) -> Result<Vec<u64>, YscError> {
        self.stream
            .set_pos((self.header.code_block_base_ptr * 8) as usize)?;
        let code_block_count = (self.header.code_size + 0x3FFF).overflowing_shr(14).0;
        let mut code_tables = Vec::with_capacity(code_block_count as usize);
        for _i in 0..code_block_count {
            code_tables.push(self.stream.read_int::<u64>(32)? & 0xFFFFFF);
            self.stream.skip_bits(32)?;
        }

        Ok(code_tables)
    }

    //pub fn get_string_table(&mut self) -> Result<Vec<String>, Error> {
    //    let string_table_offsets = self.get_string_table_offsets()?;
    //    let string_size = self.header.string_size;
    //    let mut table = Vec::with_capacity(string_size as usize);
    //
    //}

    fn native_hash_rotate(hash: u64, mut rotate: u32) -> u64 {
        rotate %= 64;
        (hash.overflowing_shl(rotate)).0 | (hash.overflowing_shr(64 - rotate).0)
    }
}
