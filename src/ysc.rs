use std::error::Error;
use bitbuffer::{BitReadBuffer, BitReadStream, LittleEndian};

#[derive(Default, Debug)]
pub struct YSCHeader {
    pub page_base: i32,
    pub page_map_ptr: i32,
    pub code_block_base_ptr: i32,
    pub globals_signature: i32,
    pub code_size: i32,
    pub parameter_count: i32,
    pub static_count: i32,
    pub global_count: i32,
    pub natives_count: i32,
    pub statics_ptr: i32,
    pub globals_ptr: i32,
    pub natives_ptr: i32,
    pub unk_1: i32,
    pub unk_2: i32,
    pub script_name_hash: i32,
    pub unk_3: i32,
    pub script_name_ptr: i32,
    pub string_blocks_base_ptr: i32,
    pub string_size: i32,
    pub unk_4: i32,
    pub unk_5: i32,
    pub unk_6: i32,
}

impl YSCHeader {
    fn new(stream: &mut BitReadStream<'_, LittleEndian>) -> Result<Self, Box<dyn Error>> {
        stream.set_pos(0)?;
        let mut header = YSCHeader::default();

        let read_int = |stream: &mut BitReadStream<'_, LittleEndian>| -> Result<i32, Box<dyn Error>> {
            let res = stream.read_int::<i32>(32)?;
            Ok(res)
        };

        let read_int_large = |stream: &mut BitReadStream<'_, LittleEndian>| -> Result<i32, Box<dyn Error>> {
            let res = stream.read_int::<i32>(32)?;
            stream.skip_bits(32)?;
            Ok(res)
        };

        let read_ptr = |stream: &mut BitReadStream<'_, LittleEndian>| -> Result<i32, Box<dyn Error>> {
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

#[derive(Debug)]
pub struct YSC<'a> {
    stream: BitReadStream<'a, LittleEndian>,
    pub header: YSCHeader
}

impl<'a> YSC<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self, Box<dyn Error>> {
        let buffer = BitReadBuffer::new(&data, LittleEndian);
        let mut stream = BitReadStream::new(buffer);
        Ok(YSC {
            header: YSCHeader::new(&mut stream)?,
            stream
        })
    }
    pub fn get_script_name(&mut self) -> Result<String, Box<dyn Error>> {
        self.stream.set_pos(self.header.script_name_ptr as usize * 8)?;

        Ok(self.stream.read_string(None)?.to_string())
    }
    pub fn get_native_table(&mut self) -> Result<Vec<u64>, Box<dyn Error>> {
        self.stream.set_pos(self.header.natives_ptr as usize * 8)?;
        let mut natives = vec![];
        for i in 0..self.header.natives_count {
            let encrypted_hash: u64 = self.stream.read_int(64)?;
            let hash = YSC::native_hash_rotate(encrypted_hash,
                                               (self.header.code_size + i) as u32);
            natives.push(hash);
        }
        Ok(natives)
    }

    pub fn get_string_table_offsets(&mut self) -> Result<Vec<u64>, Box<dyn Error>> {
        self.stream.set_pos((self.header.string_blocks_base_ptr * 8) as usize)?;
        let string_block_count = (self.header.string_size + 0x3FFF).overflowing_shr(14).0;
        let mut string_tables = Vec::with_capacity(string_block_count as usize);
        for i in 0..string_block_count {
            string_tables.push(self.stream.read_int::<u64>(32)? & 0xFFFFFF);
            self.stream.skip_bits(32)?;
        }

        Ok(string_tables)
    }

    pub fn get_code_table_offsets(&mut self) -> Result<Vec<u64>, Box<dyn Error>> {
        self.stream.set_pos((self.header.code_block_base_ptr * 8) as usize)?;
        let code_block_count = (self.header.code_size + 0x3FFF).overflowing_shr(14).0;
        let mut code_tables = Vec::with_capacity(code_block_count as usize);
        for i in 0..code_block_count {
            code_tables.push(self.stream.read_int::<u64>(32)? & 0xFFFFFF);
            self.stream.skip_bits(32)?;
        }

        Ok(code_tables)
    }

    //pub fn get_string_table(&mut self) -> Result<Vec<String>, Box<dyn Error>> {
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