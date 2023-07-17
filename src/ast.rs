use crate::disassemble::{Disassembler, Instruction, DisassembleError};
use crate::ysc::YSCScript;
use std::fmt::{Display, Formatter};
use thiserror::Error;

/// Generic stack error types returned by AstGenerator
#[derive(Error, Debug)]
pub enum AstStackError {
    /// During decompilation, items might be requested from the stack, and the stack might not contain enough items
    #[error("no more items are in the stack. stack is of size {size:?} and {requested:?} items were requested")]
    StackEnded {
        /// Total size of the stack
        size: Option<usize>,
        /// However many items were requested from the stack
        requested: Option<usize>
    },
    /// Items on the stack can have a "size" on the stack which is greater than one.
    /// You can push a Vec3 to the stack, it'll be one stack item, but it'll have the size of 3
    /// Under certain conditions, trying to aquire less elements then the size of the current element can result in an error
    #[error("attempt to pop item off the stack that does not meet proper stack boundaries ({remaining} remaining)")]
    StackMisaligned {
        /// How many items are left in the stack
        remaining: usize,
    }
}

/// Generic error types returned by AstGenerator
#[derive(Error, Debug)]
pub enum AstError {
    /// Errors pertaining to the stack
    #[error("stack error")]
    StackError(#[from] AstStackError),

    /// Errors pertaining to the disassembly of a script
    #[error("disassemble error")]
    DisassembleError(#[from] DisassembleError),

    /// Attempt to perform actions on a function index which doesn't exist
    #[error("specified function index {index} is larger than count {count}")]
    BadFunctionIndex {
        /// Specified function index
        index: usize,

        /// Number of functions
        count: usize
    },

    /// Blocks can be thought of the group of expressions within if/else/while etc.. blocks
    /// If a child block inside another block references code outside of its parent block, this could trigger
    /// Note that specific hotfixes are applied for very common cases where this might happen
    #[error("invalid block size")]
    InvalidBlockSize,

    /// The disassembler runs a two-stage pass on the code
    /// - It first generates a list of all the functions and their locations
    /// - Then it finds CALL instructions and determines which exact function they're calling
    /// This can fail if the disassembler didn't get to disassemble the entire script and you call a later function
    #[error("call instruction tried to call a non-existent function")]
    NoCallIndex,

    /// If we must disassemble a function to determine how many arguments it returns, and that function calls its-self recursively too many times, we have to fail
    #[error("function {index} repeatedly calls its-self")]
    RecursionLimit {
        /// The function that attempted to be called
        index: usize
    },
    /// Certain instructions push or pop an undetermined number of arguments from the stack, this makes attempts at static decompilation hard
    #[error("instruction {opcode:#?} called with non-const size")]
    DynamicStackSize {
        /// Opcode that is dynamic
        opcode: Instruction
    },
    /// AST generator found an opcode that it didn't support
    #[error("unsupported opcode {opcode:#?}")]
    UnsupportedOpcode {
        /// Opcode that is unsupported
        opcode: Instruction
    }
}

/// A function that has not been decompiled, contains just enough information to be decompiled
#[derive(Debug, Clone)]
pub struct ProtoFunction {
    num_args: u8,
    index: usize,
    instructions: Vec<Instruction>,
}

impl ProtoFunction {
    fn get_func(&self) -> Function {
        Function::new(&self.instructions, self.index, AstStack::new(), 0, Vars::default())
    }
}

/// A fully decompiled function
#[derive(Debug)]
pub struct AstFunction {
    body: Ast,
    num_args: u8,
    num_returns: u8,
    index: usize,
    vars: Vars,
}

#[derive(Debug, Clone)]
enum Ast {
    Store {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    Memcpy {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
        size: u32,
    },
    Reference {
        val: Box<Ast>,
    },
    Return {
        var: Option<Box<Ast>>,
    },
    Offset {
        var: Box<Ast>,
        offset: u32,
    },
    Global {
        index: u32,
    },
    Static {
        index: u32,
    },
    ConstInt {
        val: i32,
    },
    ConstFloat {
        val: f32,
    },
    FloatSub {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    FloatDiv {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    FloatAdd {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    IntAdd {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    IntDiv {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    IntMod {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    IntSub {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    IntMul {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    FloatMul {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    IntLessThan {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    IntGreaterThan {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    IntLessThanOrEq {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    IntGreaterThanOrEq {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    FloatLessThanOrEq {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    FloatGreaterThanOrEq {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    FloatLessThan {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    FloatGreaterThan {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    FloatEqual {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    StatementList {
        list: Vec<Ast>,
        stack_size: usize,
    },
    StackVariableList {
        list: Vec<Ast>,
    },
    Local {
        index: u8,
        local_var_index: Option<u32>,
    },
    IsBitSet {
        val: Box<Ast>,
        bit: Box<Ast>,
    },
    Native {
        args_list: Vec<Ast>,
        num_returns: u8,
        hash: usize,
    },
    IntegerNotEqual {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    IntegerEqual {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    IntegerAnd {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    IntegerOr {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    String {
        index: Box<Ast>,
        value: Option<String>,
    },
    LoadN {
        size: u32,
        address: Box<Ast>,
    }, // todo: make this support non-const int sizes (no idea how though)
    Dereference {
        val: Box<Ast>,
    },
    Temporary {
        index: u16,
        field: Option<u16>,
    },
    Not {
        val: Box<Ast>,
    },
    If {
        condition: Box<Ast>,
        body: Box<Ast>,
        else_: Option<Box<Ast>>,
    },
    While {
        condition: Box<Ast>,
        body: Box<Ast>,
    },
    Call {
        index: u32,
        args: Vec<Ast>,
        num_returns: u8,
    },
    Array {
        var: Box<Ast>,
        at: Box<Ast>,
        size: u32,
    },
}

impl Display for AstFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let return_type = match self.num_returns {
            0 => "void".to_owned(),
            1 => "var".to_owned(),
            _ => format!("var[{}]", self.num_returns),
        };
        let args = (0..self.num_args)
            .map(|i| format!("var arg{i}"))
            .collect::<Vec<_>>()
            .join(", ");

        let local_vars = if self.vars.local == 0 && self.vars.temp == 0 {
            "".to_owned()
        } else {
            let mut vars_str = "".to_owned();
            if self.vars.local != 0 {
                let local_vars = format!(
                    "\n\tvar {};",
                    (0..self.vars.local)
                        .map(|x| format!("local{x}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                vars_str.push_str(&local_vars);
            }
            if self.vars.temp != 0 {
                let temp_vars = format!(
                    "\n\tvar {};\n",
                    (0..self.vars.temp)
                        .map(|x| format!("temp_{x}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                vars_str.push_str(&temp_vars);
            }
            vars_str
        };

        write!(
            f,
            "{} func_{}({}) {{{}\n{}\n}}",
            return_type, self.index, args, local_vars, self.body
        )
    }
}

impl Ast {
    fn get_stack_size(&self) -> usize {
        return match self {
            Ast::Store { .. } => 0,
            Ast::Memcpy { .. } => 0,
            Ast::Return { .. } => 0,
            Ast::Offset { .. } => 1,
            Ast::Global { .. } => 1,
            Ast::ConstInt { .. } => 1,
            Ast::StatementList { stack_size, .. } => *stack_size,
            Ast::StackVariableList { list } => list.iter().map(|x| x.get_stack_size()).sum(),
            Ast::Local { .. } => 1,
            Ast::IsBitSet { .. } => 1,
            Ast::Native { num_returns, .. } => *num_returns as usize,
            Ast::IntegerNotEqual { .. } => 1,
            Ast::IntegerEqual { .. } => 1,
            Ast::FloatEqual { .. } => 1,
            Ast::String { .. } => 1,
            Ast::LoadN { size, .. } => *size as usize,
            Ast::Temporary { .. } => 1,
            Ast::If { .. } => 0,
            Ast::While { .. } => 0,
            Ast::Not { .. } => 1,
            Ast::Dereference { .. } => 1,
            Ast::ConstFloat { .. } => 1,
            Ast::Call { num_returns, .. } => *num_returns as usize,
            Ast::FloatSub { .. } => 1,
            Ast::FloatDiv { .. } => 1,
            Ast::IntDiv { .. } => 1,
            Ast::IntMod { .. } => 1,
            Ast::FloatAdd { .. } => 1,
            Ast::Static { .. } => 1,
            Ast::Reference { .. } => 1,
            Ast::IntAdd { .. } => 1,
            Ast::IntSub { .. } => 1,
            Ast::IntLessThan { .. } => 1,
            Ast::IntGreaterThan { .. } => 1,
            Ast::IntLessThanOrEq { .. } => 1,
            Ast::IntGreaterThanOrEq { .. } => 1,
            Ast::IntegerOr { .. } => 1,
            Ast::IntegerAnd { .. } => 1,
            Ast::Array { .. } => 1,
            Ast::FloatLessThan { .. } => 1,
            Ast::FloatGreaterThan { .. } => 1,
            Ast::FloatLessThanOrEq { .. } => 1,
            Ast::FloatGreaterThanOrEq { .. } => 1,
            Ast::IntMul { .. } => 1,
            Ast::FloatMul { .. } => 1,
        };
    }
}

impl Display for Ast {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let line = match self {
            Ast::Store { lhs, rhs } => {
                format!("{} = {};", lhs, rhs)
            }
            Ast::Memcpy { lhs, rhs, size } => {
                format!("memcpy({}, {}, {} * 4);", lhs, rhs, size)
            }
            Ast::Return { var } => {
                let mut returns = "".to_owned();
                if let Some(ret) = var {
                    returns = format!(" {ret}");
                }

                format!("return{returns};")
            }
            Ast::Offset { var, offset } => match &**var {
                Ast::Reference { val } => format!("{val}.f_{offset}"),
                _ => format!("{var}->f_{offset}"),
            },
            Ast::ConstInt { val } => format!("{val}"),
            Ast::StatementList { list, .. } => {
                let mut lines = vec![];
                for ast_token in list {
                    for line in format!("{ast_token}").lines() {
                        lines.push(format!("\t{line}"));
                    }

                    if matches!(ast_token, Ast::Native { .. }) {
                        // Natives can be statements and expressions
                        let len = lines.len();
                        lines[len - 1].push(';');
                    }
                }
                lines.join("\n")
            }
            Ast::StackVariableList { list } => {
                if list.is_empty() {
                    "".to_owned()
                } else if list.len() == 1 {
                    format!("{}", list[0])
                } else {
                    format!(
                        "{{ {} }}",
                        list.iter()
                            .map(|ast| format!("{}", ast))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            }
            Ast::Local {
                index,
                local_var_index,
            } => match local_var_index {
                Some(loc) => format!("local{loc}"),
                _ => format!("arg{index}"),
            },
            Ast::Global { index } => format!("Global_{index}"),
            Ast::Static { index } => format!("Static_{index}"),
            Ast::IsBitSet { bit, val } => format!("IS_BIT_SET({}, {})", bit, val),
            Ast::Native {
                hash, args_list, ..
            } => {
                format!(
                    "0x{hash:X}({})",
                    args_list
                        .iter()
                        .map(|x| format!("{x}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            Ast::IntegerNotEqual { lhs, rhs } => {
                format!("{} != {}", lhs, rhs)
            }
            Ast::IntegerEqual { lhs, rhs } => {
                format!("{} == {}", lhs, rhs)
            }
            Ast::FloatEqual { lhs, rhs } => {
                format!("{} == {}", lhs, rhs)
            }
            Ast::String { index, value } => {
                if let Some(str_val) = value {
                    format!("\"{str_val}\"")
                } else {
                    format!("_STRING({index})")
                }
            }
            Ast::LoadN { address, .. } => format!("{address}"),
            Ast::Temporary { index, field } if field.is_none() => format!("temp_{index}"),
            Ast::Temporary { index, field } => format!("temp_{index}->f_{}", field.unwrap()),
            Ast::If {
                condition,
                body,
                else_,
            } => {
                if else_.is_none() {
                    format!("if ({condition}) {{\n{body}\n}}")
                } else {
                    format!(
                        "if ({condition}) {{\n{body}\n}} else {{\n{}\n}}",
                        else_.as_ref().unwrap()
                    )
                }
            }
            Ast::While { condition, body } => format!("while ({condition}) {{\n{body}\n}}"),
            Ast::Not { val } => match &**val {
                Ast::IntegerEqual { lhs, rhs } => {
                    format!("({lhs} != {rhs})")
                }
                Ast::IntegerNotEqual { lhs, rhs } => {
                    format!("({lhs} == {rhs})")
                }
                Ast::Not { val } => format!("{val}"),
                _ => format!("!({val})"),
            },
            Ast::Dereference { val } => match &**val {
                Ast::Reference { val } => format!("{val}"),
                _ => format!("*{val}"),
            },
            Ast::ConstFloat { val } => format!("{val}f"),
            Ast::Call { index, args, .. } => format!(
                "func_{index}({})",
                args.iter()
                    .map(|x| format!("{x}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Ast::FloatSub { lhs, rhs } => format!("({lhs} - {rhs})"),
            Ast::FloatDiv { lhs, rhs } => format!("({lhs} / {rhs})"),
            Ast::IntDiv { lhs, rhs } => format!("({lhs} / {rhs})"),
            Ast::IntMod { lhs, rhs } => format!("({lhs} % {rhs})"),
            Ast::FloatAdd { lhs, rhs } => format!("({lhs} + {rhs})"),
            Ast::Reference { val } => format!("&{val}"),
            Ast::IntAdd { lhs, rhs } => format!("({lhs} + {rhs})"),
            Ast::IntSub { lhs, rhs } => format!("({lhs} + {rhs})"),
            Ast::IntLessThan { lhs, rhs } => format!("({lhs} < {rhs})"),
            Ast::IntGreaterThan { lhs, rhs } => format!("({lhs} > {rhs})"),
            Ast::IntLessThanOrEq { lhs, rhs } => format!("({lhs} <= {rhs})"),
            Ast::IntGreaterThanOrEq { lhs, rhs } => format!("({lhs} >= {rhs})"),
            Ast::IntegerOr { lhs, rhs } => format!("({lhs} || {rhs})"),
            Ast::IntegerAnd { lhs, rhs } => format!("({lhs} && {rhs})"),
            Ast::Array { var, at, size } => format!("{var}[{at} /*{size}*/]"),
            Ast::FloatLessThan { lhs, rhs } => format!("({lhs} < {rhs})"),
            Ast::FloatGreaterThan { lhs, rhs } => format!("({lhs} > {rhs})"),
            Ast::FloatLessThanOrEq { lhs, rhs } => format!("({lhs} <= {rhs})"),
            Ast::FloatGreaterThanOrEq { lhs, rhs } => format!("({lhs} >= {rhs})"),
            Ast::IntMul { lhs, rhs } => format!("({lhs} * {rhs})"),
            Ast::FloatMul { lhs, rhs } => format!("({lhs} * {rhs})"),
        };

        write!(f, "{}", line)
    }
}

#[derive(Clone)]
struct AstStack {
    stack: Vec<Ast>,
}

impl AstStack {
    fn new() -> Self {
        Self {
            stack: Vec::with_capacity(8),
        }
    }

    fn push(&mut self, ast: Ast) {
        if ast.get_stack_size() != 0 {
            self.stack.push(ast);
        }
    }

    fn pop(
        &mut self,
        statements: &mut Vec<Ast>,
        temp_vars_count: &mut u8,
        size: u32,
    ) -> Result<Vec<Ast>, AstStackError> {
        if size == 0 {
            return Ok(vec![]);
        }
        let og_size = size;
        let total_stack_size = self.stack.len();
        let mut size_remaining: i64 = size as i64;
        let mut items: Vec<Ast> = Vec::with_capacity(size as usize);

        for i in (0..self.stack.len()).rev() {
            let item = &self.stack[i];
            let size = item.get_stack_size() as i64;
            size_remaining -= size;
            if size != 0 {
                items.push(
                    self.stack
                        .pop()
                        .ok_or(AstStackError::StackEnded { size: Some(total_stack_size), requested: Some(og_size as usize) })?,
                );
            }
            match size_remaining {
                size_remaining if size_remaining < 0 => {
                    return Err(AstStackError::StackMisaligned { remaining: size_remaining as usize });
                }
                size_remaining if size_remaining == 0 => {
                    return if items.len() == 1 && og_size == 1 && items[0].get_stack_size() == 1 {
                        Ok(items)
                    } else {
                        if items.len() == 1 && items[0].get_stack_size() == og_size as usize {
                            return Ok(items);
                        }
                        let mut single_sized_items = vec![];
                        for item in items {
                            let size = item.get_stack_size();
                            if size == 1 {
                                single_sized_items.push(item);
                            } else {
                                statements.push(Ast::Store {
                                    lhs: Box::new(Ast::Temporary {
                                        index: *temp_vars_count as u16,
                                        field: None,
                                    }),
                                    rhs: Box::new(item),
                                });
                                for i in (0..size).rev() {
                                    single_sized_items.push(Ast::Temporary {
                                        index: *temp_vars_count as u16,
                                        field: Some(i as u16),
                                    })
                                }

                                *temp_vars_count += 1;
                            }
                        }
                        if single_sized_items.len() != og_size as usize {
                            panic!(
                                "pop field refs could not get stack pop to match user requested size"
                            );
                        }
                        Ok(single_sized_items)
                    };
                }
                _ => {}
            }
        }

        Err(AstStackError::StackEnded { size: Some(og_size as usize), requested: Some(total_stack_size) })
    }

    fn len(&self) -> usize {
        self.stack.iter().map(|x| x.get_stack_size()).sum()
    }

    fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }
}

#[derive(Default, Debug, Clone)]
struct Vars {
    local: u8,
    temp: u8,
}

struct Function<'a> {
    instructions: &'a [Instruction],
    function_index: usize,
    stack: AstStack,
    depth: usize,
    vars: Vars,
}

impl<'a> Function<'a> {
    fn new(
        instructions: &'a [Instruction],
        function_index: usize,
        stack: AstStack,
        depth: usize,
        vars: Vars,
    ) -> Self {
        Self {
            instructions,
            function_index,
            stack,
            depth,
            vars,
        }
    }
}

/// Creates Functions and turns them into AstFunctions
/// Can be created from a YSCScript 
pub struct AstGenerator {
    functions: Vec<ProtoFunction>,
}

impl AstGenerator {
    /// Try to generate AST for a function given it's index
    pub fn generate_function(&self, index: usize) -> Result<AstFunction, AstError> {
        self.generate_function_with_stack(index, AstStack::new(), 0)
    }

    /// Get a list of all the ProtoFunctions that AstGenerator can access
    pub fn get_functions(&self) -> &Vec<ProtoFunction> {
        &self.functions
    }

    /// Get a list of all the ProtoFunctions that AstGenerator can access
    pub fn get_functions_mut(&mut self) -> &mut Vec<ProtoFunction> {
        &mut self.functions
    }

    fn generate_function_with_stack(
        &self,
        index: usize,
        stack: AstStack,
        depth: usize,
    ) -> Result<AstFunction, AstError> {
    
        if index >= self.functions.len() {
            return Err(AstError::BadFunctionIndex { index, count: self.functions.len() });
        }

        let ast_func = self.generate_ast_from_function(&self.functions[index], stack, depth)?;
        Ok(ast_func)
    }

    fn generate_ast_from_function(
        &self,
        function: &ProtoFunction,
        stack: AstStack,
        depth: usize,
    ) -> Result<AstFunction, AstError> {
        let (body, num_returns, vars) = self.generate_ast(&mut Function::new(
            &function.instructions[1..],
            function.index,
            stack,
            depth,
            Vars::default(),
        ))?;

        let ast_func = AstFunction {
            body,
            num_args: function.num_args,
            num_returns,
            index: function.index,
            vars,
        };

        Ok(ast_func)
    }

    fn generate_conditional_block<'b>(
        index: &mut usize,
        offset_remaining: i16,
        instructions: &'b [Instruction],
    ) -> Option<&'b [Instruction]> {
        *index += 1;
        let og_index = *index;

        fn find_offset(
            mut offset_remaining: i16,
            instructions: &[Instruction],
            mut index: usize,
        ) -> Option<usize> {
            while offset_remaining > 0 {
                if instructions.len() == index {
                    return None;
                }
                let inst = &instructions[index];
                offset_remaining -= inst.get_size() as i16; // todo: this might be too small when factoring large switches
                match inst {
                    Instruction::Jz { offset } => {
                        offset_remaining = std::cmp::max(*offset, offset_remaining);
                    }
                    Instruction::ILtJz { offset } => {
                        offset_remaining = std::cmp::max(*offset, offset_remaining);
                    }
                    Instruction::ILeJz { offset } => {
                        offset_remaining = std::cmp::max(*offset, offset_remaining);
                    }
                    Instruction::INeJz { offset } => {
                        offset_remaining = std::cmp::max(*offset, offset_remaining);
                    }
                    Instruction::IEqJz { offset } => {
                        offset_remaining = std::cmp::max(*offset, offset_remaining);
                    }
                    Instruction::IGtJz { offset } => {
                        offset_remaining = std::cmp::max(*offset, offset_remaining);
                    }
                    Instruction::IGeJz { offset } => {
                        offset_remaining = std::cmp::max(*offset, offset_remaining);
                    }
                    Instruction::J { offset } => {
                        offset_remaining = std::cmp::max(*offset, offset_remaining);
                    }
                    _ => {}
                }

                if offset_remaining != 0 {
                    index += 1;
                }
            }
            Some(index)
        }

        *index = find_offset(offset_remaining, instructions, *index)?;

        if *index < instructions.len() {
            Some(&instructions[og_index..=*index])
        } else {
            None
        }
    }

    fn generate_if(
        &self,
        condition: Box<Ast>,
        offset_remaining: i16,
        index: &mut usize,
        statements: &mut Vec<Ast>,
        func: &mut Function,
    ) -> Result<(), AstError> {
        if offset_remaining == 0 {
            return Ok(());
        }

        let mut instructions_in_block =
            AstGenerator::generate_conditional_block(index, offset_remaining, func.instructions)
                .ok_or(AstError::InvalidBlockSize)?;
        let mut is_while = false;
        let else_ = if !instructions_in_block.is_empty() {
            let inst = instructions_in_block.last().unwrap();
            if let Instruction::J { offset } = inst {
                if *offset > 0 {
                    instructions_in_block =
                        &instructions_in_block[..(instructions_in_block.len() - 1)];
                    let offset = *offset;
                    let else_block =
                        AstGenerator::generate_conditional_block(index, offset, func.instructions)
                            .ok_or(AstError::InvalidBlockSize)?;

                    Some(Box::new(
                        self.generate_ast(&mut Function::new(
                            else_block,
                            func.function_index,
                            func.stack.clone(),
                            func.depth,
                            func.vars.clone(),
                        ))?
                        .0,
                    ))
                } else if *offset != 0 {
                    instructions_in_block =
                        &instructions_in_block[..(instructions_in_block.len() - 1)];
                    is_while = true;
                    None
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        let if_ = if is_while {
            Ast::While {
                condition,
                body: Box::new(
                    self.generate_ast(&mut Function::new(
                        instructions_in_block,
                        func.function_index,
                        func.stack.clone(),
                        func.depth,
                        func.vars.clone(),
                    ))?
                    .0,
                ),
            }
        } else {
            Ast::If {
                condition,
                body: Box::new(
                    self.generate_ast(&mut Function::new(
                        instructions_in_block,
                        func.function_index,
                        func.stack.clone(),
                        func.depth,
                        func.vars.clone(),
                    ))?
                    .0,
                ),
                else_,
            }
        };
        statements.push(if_);
        Ok(())
    }

    fn generate_ast(&self, func: &mut Function) -> Result<(Ast, u8, Vars), AstError> {
        let mut index = 0;
        let mut statements = Vec::with_capacity(8);
        let mut returns = 0;

        fn err() -> AstError {
            AstError::StackError(AstStackError::StackEnded { size: None, requested: None })
        }

        while index < func.instructions.len() {
            let inst = &func.instructions[index];

            // let list = Ast::StatementList {
            //     list: statements.clone(),
            //     stack_size: 0,
            // };
            // println!("INST: {inst:?} func_{function_index}");
            // println!("STACK: {:?} / {}", stack.stack, stack.len());
            // if stack.len() != 0 {
            //     println!("STACK_TOP: {}", stack.stack.last().unwrap());
            // }
            // println!("ITER:\n{list}\n\n");

            match inst {
                Instruction::PushConstM1 => {
                    func.stack.push(Ast::ConstInt { val: -1 });
                }
                Instruction::PushConst0 => {
                    func.stack.push(Ast::ConstInt { val: 0 });
                }
                Instruction::PushConst1 => {
                    func.stack.push(Ast::ConstInt { val: 1 });
                }
                Instruction::PushConst2 => {
                    func.stack.push(Ast::ConstInt { val: 2 });
                }
                Instruction::PushConst3 => {
                    func.stack.push(Ast::ConstInt { val: 3 });
                }
                Instruction::PushConst4 => {
                    func.stack.push(Ast::ConstInt { val: 4 });
                }
                Instruction::PushConst5 => {
                    func.stack.push(Ast::ConstInt { val: 5 });
                }
                Instruction::PushConst6 => {
                    func.stack.push(Ast::ConstInt { val: 6 });
                }
                Instruction::PushConst7 => {
                    func.stack.push(Ast::ConstInt { val: 7 });
                }
                Instruction::PushConstU8 { one } => {
                    func.stack.push(Ast::ConstInt { val: *one as i32 });
                }
                Instruction::PushConstS16 { num } => {
                    func.stack.push(Ast::ConstInt { val: *num as i32 });
                }
                Instruction::PushConstF { one } => {
                    func.stack.push(Ast::ConstFloat { val: *one });
                }
                Instruction::PushConstF0 => {
                    func.stack.push(Ast::ConstFloat { val: 0.0 });
                }
                Instruction::PushConstF1 => {
                    func.stack.push(Ast::ConstFloat { val: 1.0 });
                }
                Instruction::PushConstF2 => {
                    func.stack.push(Ast::ConstFloat { val: 2.0 });
                }
                Instruction::PushConstF3 => {
                    func.stack.push(Ast::ConstFloat { val: 3.0 });
                }
                Instruction::PushConstF4 => {
                    func.stack.push(Ast::ConstFloat { val: 4.0 });
                }
                Instruction::PushConstF5 => {
                    func.stack.push(Ast::ConstFloat { val: 5.0 });
                }
                Instruction::PushConstF6 => {
                    func.stack.push(Ast::ConstFloat { val: 6.0 });
                }
                Instruction::PushConstF7 => {
                    func.stack.push(Ast::ConstFloat { val: 7.0 });
                }
                Instruction::PushConstFM1 => {
                    func.stack.push(Ast::ConstFloat { val: -1.0 });
                }
                Instruction::PushConstU32 { one } => {
                    func.stack.push(Ast::ConstInt { val: *one as i32 });
                }
                Instruction::PushConstU24 { num } => {
                    func.stack.push(Ast::ConstInt { val: *num as i32 });
                }
                Instruction::Throw => {
                    func.stack.push(Ast::ConstInt { val: 0 });
                }
                Instruction::Nop => {}
                Instruction::Ilt => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::IntLessThan {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Instruction::Ile => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::IntLessThanOrEq {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Instruction::Igt => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::IntGreaterThan {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Instruction::Ige => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::IntGreaterThanOrEq {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Instruction::Flt => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::FloatLessThan {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Instruction::Fgt => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::FloatGreaterThan {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Instruction::Fle => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::FloatLessThanOrEq {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Instruction::Fge => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::FloatGreaterThanOrEq {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Instruction::Ior => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);

                    func.stack.push(Ast::IntegerOr {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Instruction::Iand => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::IntegerAnd {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Instruction::Iadd => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::IntAdd {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Instruction::Imul => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::IntMul {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Instruction::ImulU8 { num } => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 1)?;
                    let lhs = args.pop().ok_or(err())?;
                    func.stack.push(Ast::IntMul {
                        lhs: Box::new(lhs),
                        rhs: Box::new(Ast::ConstInt { val: *num as i32 }),
                    });
                }
                Instruction::ImulS16 { num } => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 1)?;
                    let lhs = args.pop().ok_or(err())?;
                    func.stack.push(Ast::IntMul {
                        lhs: Box::new(lhs),
                        rhs: Box::new(Ast::ConstInt { val: *num as i32 }),
                    });
                }
                Instruction::Fmul => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::FloatMul {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Instruction::IaddS16 { num } => {
                    let arg = func
                        .stack
                        .pop(&mut statements, &mut func.vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;
                    func.stack.push(Ast::IntAdd {
                        lhs: Box::new(arg),
                        rhs: Box::new(Ast::ConstInt { val: *num as i32 }),
                    });
                }
                Instruction::IaddU8 { num } => {
                    let arg = func
                        .stack
                        .pop(&mut statements, &mut func.vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;
                    func.stack.push(Ast::IntAdd {
                        lhs: Box::new(arg),
                        rhs: Box::new(Ast::ConstInt { val: *num as i32 }),
                    });
                }
                Instruction::Isub => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::IntSub {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Instruction::Fsub => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::FloatSub {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Instruction::Idiv => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::IntDiv {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Instruction::Imod => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::IntMod {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Instruction::Fdiv => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::FloatDiv {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Instruction::Fadd => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::FloatAdd {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Instruction::StaticU16Load { static_var_index } => {
                    func.stack.push(Ast::Static {
                        index: *static_var_index as u32,
                    });
                }
                Instruction::StaticU16 { static_var_index } => {
                    func.stack.push(Ast::Reference {
                        val: Box::new(Ast::Static {
                            index: *static_var_index as u32,
                        }),
                    });
                }
                Instruction::StaticU16Store { static_var_index } => {
                    let arg = func
                        .stack
                        .pop(&mut statements, &mut func.vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;
                    statements.push(Ast::Store {
                        lhs: Box::new(Ast::Static { index: *static_var_index as u32 }),
                        rhs: Box::new(arg),
                    });
                }
                Instruction::GlobalU16 { index } => {
                    func.stack.push(Ast::Reference {
                        val: Box::new(Ast::Global {
                            index: *index as u32,
                        }),
                    });
                }
                Instruction::GlobalU24 { index } => {
                    func.stack.push(Ast::Reference {
                        val: Box::new(Ast::Global { index: *index }),
                    });
                }
                Instruction::ArrayU8 { size } => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (index, ptr) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    let ptr = match ptr {
                        Ast::Reference { val } => val,
                        _ => Box::new(ptr),
                    };
                    func.stack.push(Ast::Reference {
                        val: Box::new(Ast::Array {
                            var: ptr,
                            at: Box::new(index),
                            size: *size as u32,
                        }),
                    })
                }
                Instruction::ArrayU8Load { size } => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (index, ptr) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    let ptr = match ptr {
                        Ast::Reference { val } => val,
                        _ => Box::new(ptr),
                    };
                    func.stack.push(Ast::Array {
                        var: ptr,
                        at: Box::new(index),
                        size: *size as u32,
                    })
                }
                Instruction::ArrayU16 { size } => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (index, ptr) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    let ptr = match ptr {
                        Ast::Reference { val } => val,
                        _ => Box::new(ptr),
                    };
                    func.stack.push(Ast::Reference {
                        val: Box::new(Ast::Array {
                            var: ptr,
                            at: Box::new(index),
                            size: *size as u32,
                        }),
                    })
                }
                Instruction::ArrayU16Load { size } => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (index, ptr) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    let ptr = match ptr {
                        Ast::Reference { val } => val,
                        _ => Box::new(ptr),
                    };
                    func.stack.push(Ast::Array {
                        var: ptr,
                        at: Box::new(index),
                        size: *size as u32,
                    })
                }

                Instruction::LocalU8 { frame_index } => {
                    self.register_local_var(
                        func.function_index,
                        *frame_index,
                        &mut func.vars.local,
                    );
                    let num_args = self.functions[func.function_index].num_args;

                    let local_var_index = if *frame_index > num_args {
                        Some(*frame_index as u32 - num_args as u32 - 1)
                    } else {
                        None
                    };
                    func.stack.push(Ast::Reference {
                        val: Box::new(Ast::Local {
                            index: *frame_index,
                            local_var_index,
                        }),
                    });
                }
                Instruction::LocalU8Load { frame_index } => {
                    self.register_local_var(
                        func.function_index,
                        *frame_index,
                        &mut func.vars.local,
                    );
                    let num_args = self.functions[func.function_index].num_args;

                    let local_var_index = if *frame_index > num_args {
                        Some(*frame_index as u32 - num_args as u32 - 1)
                    } else {
                        None
                    };

                    func.stack.push(Ast::Local {
                        index: *frame_index,
                        local_var_index,
                    });
                }
                Instruction::LocalU8Store { frame_index } => {
                    self.register_local_var(
                        func.function_index,
                        *frame_index,
                        &mut func.vars.local,
                    );
                    let num_args = self.functions[func.function_index].num_args;

                    let local_var_index = if *frame_index > num_args {
                        Some(*frame_index as u32 - num_args as u32 - 1)
                    } else {
                        None
                    };
                    let arg = func
                        .stack
                        .pop(&mut statements, &mut func.vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;

                    statements.push(Ast::Store {
                        lhs: Box::new(Ast::Local {
                            index: *frame_index,
                            local_var_index,
                        }),
                        rhs: Box::new(arg),
                    });
                }
                Instruction::GlobalU24Store { index } => {
                    let arg = func
                        .stack
                        .pop(&mut statements, &mut func.vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;
                    statements.push(Ast::Store {
                        lhs: Box::new(Ast::Global { index: *index }),
                        rhs: Box::new(arg),
                    });
                }
                Instruction::GlobalU16Store { index } => {
                    let arg = func
                        .stack
                        .pop(&mut statements, &mut func.vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;
                    statements.push(Ast::Store {
                        lhs: Box::new(Ast::Global {
                            index: *index as u32,
                        }),
                        rhs: Box::new(arg),
                    });
                }
                Instruction::GlobalU16Load { index } => {
                    func.stack.push(Ast::Global {
                        index: *index as u32,
                    });
                }
                Instruction::GlobalU24Load { index } => {
                    func.stack.push(Ast::Global { index: *index });
                }
                Instruction::IoffsetU8Store { offset } => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let rhs = args.pop().ok_or(err())?;
                    let var = args.pop().ok_or(err())?;

                    statements.push(Ast::Store {
                        lhs: Box::new(Ast::Offset {
                            var: Box::new(var),
                            offset: *offset as u32,
                        }),
                        rhs: Box::new(rhs),
                    });
                }
                Instruction::IoffsetS16 { offset } => {
                    let arg = func
                        .stack
                        .pop(&mut statements, &mut func.vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;

                    func.stack.push(Ast::Reference {
                        val: Box::new(Ast::Offset {
                            var: Box::new(arg),
                            offset: *offset as u32,
                        }),
                    });
                }
                Instruction::IoffsetU8 { offset } => {
                    let arg = func
                        .stack
                        .pop(&mut statements, &mut func.vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;

                    func.stack.push(Ast::Reference {
                        val: Box::new(Ast::Offset {
                            var: Box::new(arg),
                            offset: *offset as u32,
                        }),
                    });
                }
                Instruction::IoffsetU8Load { offset } => {
                    let arg = func
                        .stack
                        .pop(&mut statements, &mut func.vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;

                    //let arg = match arg {
                    //    Ast::Reference { val } => val,
                    //    _ => Box::new(arg)
                    //};

                    func.stack.push(Ast::Offset {
                        var: Box::new(arg),
                        offset: *offset as u32,
                    });
                }
                Instruction::IoffsetS16Load { offset } => {
                    let arg = func
                        .stack
                        .pop(&mut statements, &mut func.vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;

                    func.stack.push(Ast::Offset {
                        var: Box::new(arg),
                        offset: *offset as u32,
                    });
                }
                Instruction::IoffsetS16Store { offset } => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let rhs = args.pop().ok_or(err())?;
                    let var = args.pop().ok_or(err())?;

                    statements.push(Ast::Store {
                        lhs: Box::new(Ast::Offset {
                            var: Box::new(var),
                            offset: *offset as u32,
                        }),
                        rhs: Box::new(rhs),
                    });
                }
                Instruction::Leave { .. } => {
                    if func.stack.is_empty() {
                        statements.push(Ast::Return { var: None });
                    } else {
                        let len = func.stack.len();
                        let mut items =
                            func
                                .stack
                                .pop(&mut statements, &mut func.vars.temp, len as u32)?;

                        if items.len() == 1 {
                            statements.push(Ast::Return {
                                var: Some(Box::new(items.pop().ok_or(err())?)),
                            });
                        } else {
                            statements.push(Ast::Return {
                                var: Some(Box::new(Ast::StackVariableList { list: items })),
                            });
                        }
                        returns = len as u8;
                    }
                    break;
                }
                Instruction::IsBitSet => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (bit, val) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::IsBitSet {
                        val: Box::new(val),
                        bit: Box::new(bit),
                    });
                }
                Instruction::Native {
                    native_hash,
                    num_args,
                    num_returns,
                    ..
                } => {
                    let mut args_list =
                        func
                            .stack
                            .pop(&mut statements, &mut func.vars.temp, *num_args as u32)?;
                    args_list.reverse();
                    let native = Ast::Native {
                        num_returns: *num_returns,
                        args_list,
                        hash: *native_hash as usize,
                    };

                    if *num_returns == 0 {
                        statements.push(native);
                    } else {
                        func.stack.push(native);
                    }
                }
                Instruction::Call { func_index, .. } => {
                    let index = func_index.ok_or(AstError::NoCallIndex)?;
                    if func.depth > 24 {
                        return Err(AstError::RecursionLimit { index: func_index.unwrap() });
                    }
                    let num_args = self.functions[index].num_args;
                    let mut args_list =
                        func
                            .stack
                            .pop(&mut statements, &mut func.vars.temp, num_args as u32)?;
                    args_list.reverse();
                    func.depth += 1;
                    let num_returns =
                        match self.get_number_of_returns(&self.functions[index].get_func()) {
                            Ok(res) => {
                                func.depth -= 1;
                                Ok(res)
                            }
                            Err(e) => {
                                func.depth -= 1;
                                Err(e)
                            }
                        }?;

                    let call = Ast::Call {
                        num_returns: num_returns as u8,
                        args: args_list,
                        index: func_index.unwrap() as u32,
                    };

                    if num_returns == 0 {
                        statements.push(call);
                    } else {
                        func.stack.push(call);
                    }
                }
                Instruction::Drop => {
                    let ast = func
                        .stack
                        .pop(&mut statements, &mut func.vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;
                    if ast.get_stack_size() == 1 {
                        // very janky way of detecting if this is a statement or a temporary value
                        if format!("{ast}").ends_with(';') {
                            statements.push(ast);
                        }
                    }
                }
                Instruction::Dup => {
                    // Fixes Rockstar's jank conditionals
                    if func.instructions.len() - index > 2 {
                        if matches!(func.instructions[index + 1], Instruction::Inot)
                            && matches!(func.instructions[index + 2], Instruction::Jz { .. })
                        {
                            index += 2;
                        }

                        if matches!(func.instructions[index + 1], Instruction::Jz { .. }) {
                            index += 1;
                        }
                    } else {
                        let ast = func
                            .stack
                            .pop(&mut statements, &mut func.vars.temp, 1)?
                            .pop()
                            .ok_or(err())?;
                        func.stack.push(ast.clone());
                        func.stack.push(ast);
                    }
                }
                Instruction::Ine => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::IntegerNotEqual {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    })
                }
                Instruction::Feq => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::FloatEqual {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    })
                }
                Instruction::Ieq => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    func.stack.push(Ast::IntegerEqual {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    })
                }
                Instruction::String { value, .. } => {
                    let string_index = func
                        .stack
                        .pop(&mut statements, &mut func.vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;
                    func.stack.push(Ast::String {
                        index: Box::new(string_index),
                        value: Some(value.clone()),
                    })
                }
                Instruction::LoadN => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (size, address) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    let size = if let Ast::ConstInt { val } = size {
                        val
                    } else {
                        return Err(AstError::DynamicStackSize { opcode: inst.clone() });
                    };

                    func.stack.push(Ast::LoadN {
                        address: Box::new(address),
                        size: size as u32,
                    })
                }
                Instruction::StoreN => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (size, lhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    let size = if let Ast::ConstInt { val } = size {
                        val
                    } else {
                        return Err(AstError::DynamicStackSize { opcode: inst.clone() });
                    };

                    let mut stack_items =
                        func
                            .stack
                            .pop(&mut statements, &mut func.vars.temp, size as u32)?;
                    if stack_items.len() == 1 {
                        statements.push(Ast::Memcpy {
                            lhs: Box::new(lhs),
                            rhs: Box::new(stack_items.pop().ok_or(err())?),
                            size: size as u32,
                        });
                    } else {
                        let lhs = Ast::Dereference { val: Box::new(lhs) };
                        for i in 0..stack_items.len() {
                            let rhs = Box::new(stack_items.remove(0)); // todo: improve speed
                            statements.push(Ast::Store {
                                lhs: Box::new(Ast::Offset {
                                    var: Box::new(lhs.clone()),
                                    offset: i as u32,
                                }),
                                rhs,
                            })
                        }
                    }
                }
                Instruction::Store => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (rhs, lhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    let lhs = Ast::Dereference { val: Box::new(lhs) };

                    statements.push(Ast::Store {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    })
                }
                Instruction::StoreRev => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    let lhs = Ast::Dereference { val: Box::new(lhs) };
                    func.stack.push(lhs.clone());
                    statements.push(Ast::Store {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    })
                }
                Instruction::Inot => {
                    let arg = func
                        .stack
                        .pop(&mut statements, &mut func.vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;
                    func.stack.push(Ast::Not { val: Box::new(arg) });
                }
                Instruction::Load => {
                    let arg = func
                        .stack
                        .pop(&mut statements, &mut func.vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;
                    func.stack.push(Ast::Dereference { val: Box::new(arg) });
                }
                Instruction::J { offset } if *offset == 0 => {}
                Instruction::Jz { offset } if *offset >= 0 => {
                    let condition = Box::new(
                        func
                            .stack
                            .pop(&mut statements, &mut func.vars.temp, 1)?
                            .pop()
                            .ok_or(err())?,
                    );
                    self.generate_if(condition, *offset, &mut index, &mut statements, func)?;
                }
                Instruction::IEqJz { offset } if *offset >= 0 => {
                    let mut args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    let condition = Box::new(Ast::IntegerEqual {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                    self.generate_if(condition, *offset, &mut index, &mut statements, func)?;
                }
                Instruction::INeJz { offset } if *offset >= 0 => {
                    let args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let condition = Box::new(Ast::IntegerNotEqual {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone()),
                    });
                    self.generate_if(condition, *offset, &mut index, &mut statements, func)?;
                }
                Instruction::IGtJz { offset } if *offset >= 0 => {
                    let args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let condition = Box::new(Ast::IntGreaterThan {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone()),
                    });
                    self.generate_if(condition, *offset, &mut index, &mut statements, func)?;
                }
                Instruction::IGeJz { offset } if *offset >= 0 => {
                    let args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let condition = Box::new(Ast::IntGreaterThanOrEq {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone()),
                    });
                    self.generate_if(condition, *offset, &mut index, &mut statements, func)?;
                }
                Instruction::ILtJz { offset } if *offset >= 0 => {
                    let args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let condition = Box::new(Ast::IntLessThan {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone()),
                    });
                    self.generate_if(condition, *offset, &mut index, &mut statements, func)?;
                }
                Instruction::ILeJz { offset } if *offset >= 0 => {
                    let args = func.stack.pop(&mut statements, &mut func.vars.temp, 2)?;
                    let condition = Box::new(Ast::IntLessThanOrEq {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone()),
                    });
                    self.generate_if(condition, *offset, &mut index, &mut statements, func)?;
                }
                _ => {
                    return Err(AstError::UnsupportedOpcode { opcode: inst.clone() });
                }
            }

            index += 1;
        }

        Ok((
            Ast::StatementList {
                list: statements,
                stack_size: func.stack.len(),
            },
            returns,
            func.vars.clone(),
        ))
    }

    fn get_number_of_returns(&self, callee_func: &Function) -> Result<usize, AstError> {
        let inst_iter = callee_func.instructions.iter().rev();

        for inst in inst_iter {
            if !matches!(&inst, Instruction::Leave { .. }) {
                if let Some(size) = inst.get_stack_size() {
                    return Ok(size as usize);
                }
            }
        }

        self.generate_function_with_stack(callee_func.function_index, AstStack::new(), 0).map(|x| x.num_returns as usize)
    }

    fn register_local_var(&self, func_index: usize, local_index: u8, local_vars: &mut u8) {
        let func = &self.functions[func_index];
        if local_index > func.num_args {
            *local_vars += 1;
        }
    }

    fn extract_functions(instructions: Vec<Instruction>) -> Vec<ProtoFunction> {
        let mut instruction_start_index = 0;

        let mut last_arg_count = 0;

        let mut functions = vec![];
        let mut last_index = 0;

        for (instruction_end_index, (i, inst)) in instructions.iter().enumerate().enumerate() {
            if let Instruction::Enter {
                arg_count, index, ..
            } = inst
            {
                last_index = index.unwrap() - 1;
                if instruction_end_index != 0 {
                    functions.push(ProtoFunction {
                        index: index.unwrap() - 1,
                        num_args: last_arg_count,
                        instructions: instructions[instruction_start_index..instruction_end_index]
                            .to_vec(),
                    })
                }
                last_arg_count = *arg_count;
                instruction_start_index = i;
            }
        }

        functions.push(ProtoFunction {
            index: last_index,
            num_args: last_arg_count,
            instructions: instructions[instruction_start_index..].to_vec(),
        });

        functions
    }
}

impl TryFrom<YSCScript> for AstGenerator {
    type Error = AstError;
    fn try_from(value: YSCScript) -> std::result::Result<Self, Self::Error> {
        let instructions = Disassembler::new(&value).disassemble(None)?.instructions;
        let ast_gen = Self {
            functions: AstGenerator::extract_functions(instructions),
        };

        Ok(ast_gen)
    }
}
