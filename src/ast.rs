// CURRENT ISSUE: if statements with multiple conditionals

use crate::disassemble::{Disassembler, Opcode};
use crate::ysc::YSCScript;
use anyhow::{anyhow, Context, Error, Result};
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex};

pub fn test() {
    let mut args = std::env::args().skip(1).collect::<Vec<String>>();

    if args.is_empty() {
        println!("Usage    : ast_gen %ysc_script% %function number/index%");
        println!("Example  : ast_gen freemode.ysc.full");
        println!("Example 2: ast_gen freemode.ysc.full func_305");
        return;
    }

    let function_index: Option<i32> = if args.len() == 2 {
        match args.pop() {
            Some(func) => Some(func.replace("func_", "").parse().unwrap()),
            _ => None,
        }
    } else {
        None
    };

    let script = YSCScript::from_ysc_file(&args.pop().expect("No script file in input"))
        .context("Failed to read/parse/disassemble ysc file")
        .unwrap();

    let ast_gen = Arc::new(AstGenerator::try_from(script).unwrap());
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
            let num_pass = AtomicU32::new(0);
            println!("Starting...");
            let then = std::time::Instant::now();
            let funcs = (0..ast_gen.functions.len()).collect::<Vec<_>>();
            for _ in 0..1 {
                funcs.iter().for_each(|i| {
                    if let Ok(_func) = ast_gen.generate_function(*i) {
                        //println!("done: {}", _func.index);
                        num_pass.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                    } else {
                        //println!("failed: {}", i);
                    }
                });
            }

            let now = std::time::Instant::now();
            let time = now.duration_since(then).as_millis();
            let num_pass: u32 = num_pass.into_inner();
            println!(
                "Result: {num_pass}/{} in {}ms ({}ms/func)",
                ast_gen.functions.len(),
                time,
                time as f32 / num_pass as f32
            );
        }
    }
}

#[derive(Debug, Clone)]
struct Function {
    num_args: u8,
    index: usize,
    instructions: Vec<Opcode>,
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

#[derive(Debug)]
pub struct AstFunction {
    body: Ast,
    num_args: u8,
    num_returns: u8,
    index: usize,
    vars: Vars
}

#[derive(Clone)]
pub struct AstStack {
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
    ) -> Result<Vec<Ast>, Error> {
        if size == 0 {
            return Ok(vec![]);
        }
        let og_size = size;
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
                        .ok_or(anyhow!("Cannot pop stack further"))?,
                );
            }
            match size_remaining {
                size_remaining if size_remaining < 0 => {
                    return Err(anyhow!("(Stack overflow). make `Sized` (to reduce a large elements size) and `StackFieldReference` (to reference a single element in a large element) AST type"));
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

        Err(anyhow!("empty stack"))
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
    temp: u8
}

pub struct AstGenerator {
    functions: Arc<Vec<Function>>,
    lifted_functions: Arc<Mutex<HashMap<usize, Arc<AstFunction>>>>,
    active_functions: Arc<Mutex<HashSet<usize>>>,
}

impl AstGenerator {
    pub fn generate_function(&self, index: usize) -> Result<Arc<AstFunction>, Error> {
        self.generate_function_with_stack(index, &mut AstStack::new(), 0)
    }

    pub fn generate_function_with_stack(
        &self,
        index: usize,
        stack: &mut AstStack,
        depth: usize,
    ) -> Result<Arc<AstFunction>, Error> {
        let map = self.lifted_functions.lock().unwrap();
        if map.contains_key(&index) {
            return Ok(Arc::clone(map.get(&index).unwrap()));
        }

        drop(map);

        if index >= self.functions.len() {
            return Err(anyhow!(
                "Specified function index is larger than function count"
            ));
        }
        let func = &self.functions[index];

        let ast_func = Arc::new(self.generate_ast_from_function(func, stack, depth)?);
        let mut map = self.lifted_functions.lock().unwrap();
        map.entry(index).or_insert_with(|| Arc::clone(&ast_func));
        Ok(Arc::clone(&ast_func))
    }

    fn generate_ast_from_function(
        &self,
        function: &Function,
        stack: &mut AstStack,
        depth: usize,
    ) -> Result<AstFunction, Error> {
        self.active_functions.lock().unwrap().insert(function.index);

        let (body, num_returns, vars) = match self.generate_ast(
            &function.instructions[1..],
            function.index,
            stack,
            depth,
            &mut Vars::default()
        ) {
            Ok(res) => {
                self.active_functions
                    .lock()
                    .unwrap()
                    .remove(&function.index);
                Ok(res)
            }
            Err(e) => {
                self.active_functions
                    .lock()
                    .unwrap()
                    .remove(&function.index);
                Err(e)
            }
        }?;

        let ast_func = AstFunction {
            body,
            num_args: function.num_args,
            num_returns,
            index: function.index,
            vars
        };

        Ok(ast_func)
    }

    fn generate_conditional_block<'a, 'b>(
        index: &'a mut usize,
        mut offset_remaining: i16,
        instructions: &'b [Opcode],
    ) -> Option<&'b [Opcode]> {
        *index += 1;
        let og_index = *index;

        fn find_offset<'a, 'b>(
            mut offset_remaining: i16,
            instructions: &'b [Opcode],
            mut index: usize,
        ) -> Option<usize> {
            while offset_remaining > 0 {
                if instructions.len() == index {
                    return None;
                }
                let inst = &instructions[index];
                offset_remaining -= inst.get_size() as i16; // todo: this might be too small when factoring large switches
                match inst {
                    Opcode::Jz { offset } => {
                        offset_remaining = std::cmp::max(*offset, offset_remaining);
                    }
                    Opcode::ILtJz { offset } => {
                        offset_remaining = std::cmp::max(*offset, offset_remaining);
                    }
                    Opcode::ILeJz { offset } => {
                        offset_remaining = std::cmp::max(*offset, offset_remaining);
                    }
                    Opcode::INeJz { offset } => {
                        offset_remaining = std::cmp::max(*offset, offset_remaining);
                    }
                    Opcode::IEqJz { offset } => {
                        offset_remaining = std::cmp::max(*offset, offset_remaining);
                    }
                    Opcode::IGtJz { offset } => {
                        offset_remaining = std::cmp::max(*offset, offset_remaining);
                    }
                    Opcode::IGeJz { offset } => {
                        offset_remaining = std::cmp::max(*offset, offset_remaining);
                    }
                    Opcode::J { offset } => {
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

        let instructions_in_block;
        if *index < instructions.len() {
            instructions_in_block = &instructions[og_index..=*index];
        } else {
            return None;
        }

        Some(instructions_in_block)
    }

    fn generate_if(
        &self,
        condition: Box<Ast>,
        offset_remaining: i16,
        index: &mut usize,
        instructions: &[Opcode],
        statements: &mut Vec<Ast>,
        function_index: usize,
        stack: &mut AstStack,
        depth: usize,
        vars: &mut Vars
    ) -> Result<()> {
        if offset_remaining == 0 {
            return Ok(());
        }

        let mut instructions_in_block = AstGenerator::generate_conditional_block(
            index,
            offset_remaining,
            instructions,
        )
        .context("Invalid block size: {}")?;
        let mut is_while = false;
        let else_ = if !instructions_in_block.is_empty() {
            let inst = instructions_in_block.last().unwrap();
            if let Opcode::J { offset } = inst {
                if *offset > 0 {
                    instructions_in_block =
                        &instructions_in_block[..(instructions_in_block.len() - 1)];
                    let offset = *offset;
                    let else_block = AstGenerator::generate_conditional_block(
                        index,
                        offset,
                        instructions,
                    )
                    .ok_or(anyhow!("Invalid block size"))?;

                    Some(Box::new(
                        self.generate_ast(
                            else_block,
                            function_index,
                            stack,
                            depth,
                            vars
                        )?
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
                condition: condition,
                body: Box::new(
                    self.generate_ast(
                        instructions_in_block,
                        function_index,
                        stack,
                        depth,
                        vars
                    )?
                    .0,
                ),
            }
        } else {
            Ast::If {
                condition,
                body: Box::new(
                    self.generate_ast(
                        instructions_in_block,
                        function_index,
                        stack,
                        depth,
                        vars
                    )?
                    .0,
                ),
                else_,
            }
        };
        statements.push(if_);
        anyhow::Ok(())
    }

    fn generate_ast(
        &self,
        instructions: &[Opcode],
        function_index: usize,
        stack: &mut AstStack,
        mut depth: usize,
        vars: &mut Vars
    ) -> Result<(Ast, u8, Vars), Error> {
        let mut index = 0;
        let mut statements = Vec::with_capacity(8);
        let mut returns = 0;

        fn err() -> anyhow::Error {
            anyhow!("Invalid stack item")
        }

        while index < instructions.len() {
            let inst = &instructions[index];

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
                Opcode::PushConstM1 => {
                    stack.push(Ast::ConstInt { val: -1 });
                }
                Opcode::PushConst0 => {
                    stack.push(Ast::ConstInt { val: 0 });
                }
                Opcode::PushConst1 => {
                    stack.push(Ast::ConstInt { val: 1 });
                }
                Opcode::PushConst2 => {
                    stack.push(Ast::ConstInt { val: 2 });
                }
                Opcode::PushConst3 => {
                    stack.push(Ast::ConstInt { val: 3 });
                }
                Opcode::PushConst4 => {
                    stack.push(Ast::ConstInt { val: 4 });
                }
                Opcode::PushConst5 => {
                    stack.push(Ast::ConstInt { val: 5 });
                }
                Opcode::PushConst6 => {
                    stack.push(Ast::ConstInt { val: 6 });
                }
                Opcode::PushConst7 => {
                    stack.push(Ast::ConstInt { val: 7 });
                }
                Opcode::PushConstU8 { one } => {
                    stack.push(Ast::ConstInt { val: *one as i32 });
                }
                Opcode::PushConstS16 { num } => {
                    stack.push(Ast::ConstInt { val: *num as i32 });
                }
                Opcode::PushConstF { one } => {
                    stack.push(Ast::ConstFloat { val: *one });
                }
                Opcode::PushConstF0 => {
                    stack.push(Ast::ConstFloat { val: 0.0 });
                }
                Opcode::PushConstF1 => {
                    stack.push(Ast::ConstFloat { val: 1.0 });
                }
                Opcode::PushConstF2 => {
                    stack.push(Ast::ConstFloat { val: 2.0 });
                }
                Opcode::PushConstF3 => {
                    stack.push(Ast::ConstFloat { val: 3.0 });
                }
                Opcode::PushConstF4 => {
                    stack.push(Ast::ConstFloat { val: 4.0 });
                }
                Opcode::PushConstF5 => {
                    stack.push(Ast::ConstFloat { val: 5.0 });
                }
                Opcode::PushConstF6 => {
                    stack.push(Ast::ConstFloat { val: 6.0 });
                }
                Opcode::PushConstF7 => {
                    stack.push(Ast::ConstFloat { val: 7.0 });
                }
                Opcode::PushConstFM1 => {
                    stack.push(Ast::ConstFloat { val: -1.0 });
                }
                Opcode::PushConstU32 { one } => {
                    stack.push(Ast::ConstInt { val: *one as i32 });
                }
                Opcode::PushConstU24 { num } => {
                    stack.push(Ast::ConstInt { val: *num as i32 });
                }
                Opcode::Throw => {
                    stack.push(Ast::ConstInt { val: 0 });
                }
                Opcode::Nop => {}
                Opcode::Ilt => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    stack.push(Ast::IntLessThan {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Opcode::Ile => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    stack.push(Ast::IntLessThanOrEq {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Opcode::Igt => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    stack.push(Ast::IntGreaterThan {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Opcode::Ige => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    stack.push(Ast::IntGreaterThanOrEq {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Opcode::Flt => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    stack.push(Ast::FloatLessThan {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Opcode::Fgt => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    stack.push(Ast::FloatGreaterThan {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Opcode::Fle => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    stack.push(Ast::FloatLessThanOrEq {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Opcode::Fge => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    stack.push(Ast::FloatGreaterThanOrEq {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Opcode::Ior => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);

                    stack.push(Ast::IntegerOr {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Opcode::Iand => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    stack.push(Ast::IntegerAnd {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Opcode::Iadd => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    stack.push(Ast::IntAdd {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Opcode::Imul => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    stack.push(Ast::IntMul {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Opcode::ImulU8 { num } => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 1)?;
                    let lhs = args.pop().ok_or(err())?;
                    stack.push(Ast::IntMul {
                        lhs: Box::new(lhs),
                        rhs: Box::new(Ast::ConstInt { val: *num as i32 }),
                    });
                }
                Opcode::ImulS16 { num } => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 1)?;
                    let lhs = args.pop().ok_or(err())?;
                    stack.push(Ast::IntMul {
                        lhs: Box::new(lhs),
                        rhs: Box::new(Ast::ConstInt { val: *num as i32 }),
                    });
                }
                Opcode::Fmul => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    stack.push(Ast::FloatMul {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Opcode::IaddS16 { num } => {
                    let arg = stack
                        .pop(&mut statements, &mut vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;
                    stack.push(Ast::IntAdd {
                        lhs: Box::new(arg),
                        rhs: Box::new(Ast::ConstInt { val: *num as i32 }),
                    });
                }
                Opcode::IaddU8 { num } => {
                    let arg = stack
                        .pop(&mut statements, &mut vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;
                    stack.push(Ast::IntAdd {
                        lhs: Box::new(arg),
                        rhs: Box::new(Ast::ConstInt { val: *num as i32 }),
                    });
                }
                Opcode::Isub => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    stack.push(Ast::IntSub {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Opcode::Fsub => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    stack.push(Ast::FloatSub {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Opcode::Idiv => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    stack.push(Ast::IntDiv {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Opcode::Imod => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    stack.push(Ast::IntMod {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Opcode::Fdiv => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    stack.push(Ast::FloatDiv {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Opcode::Fadd => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    stack.push(Ast::FloatAdd {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                }
                Opcode::StaticU16 { static_var_index } => {
                    stack.push(Ast::Reference {
                        val: Box::new(Ast::Static {
                            index: *static_var_index as u32,
                        }),
                    });
                }
                Opcode::GlobalU16 { index } => {
                    stack.push(Ast::Reference {
                        val: Box::new(Ast::Global {
                            index: *index as u32,
                        }),
                    });
                }
                Opcode::GlobalU24 { index } => {
                    stack.push(Ast::Reference {
                        val: Box::new(Ast::Global { index: *index }),
                    });
                }
                Opcode::ArrayU8 { size } => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (index, ptr) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    let ptr = match ptr {
                        Ast::Reference { val } => val,
                        _ => Box::new(ptr),
                    };
                    stack.push(Ast::Reference {
                        val: Box::new(Ast::Array {
                            var: ptr,
                            at: Box::new(index),
                            size: *size as u32,
                        }),
                    })
                }
                Opcode::ArrayU8Load { size } => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (index, ptr) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    let ptr = match ptr {
                        Ast::Reference { val } => val,
                        _ => Box::new(ptr),
                    };
                    stack.push(Ast::Array {
                        var: ptr,
                        at: Box::new(index),
                        size: *size as u32,
                    })
                }
                Opcode::ArrayU16 { size } => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (index, ptr) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    let ptr = match ptr {
                        Ast::Reference { val } => val,
                        _ => Box::new(ptr),
                    };
                    stack.push(Ast::Reference {
                        val: Box::new(Ast::Array {
                            var: ptr,
                            at: Box::new(index),
                            size: *size as u32,
                        }),
                    })
                }
                Opcode::ArrayU16Load { size } => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (index, ptr) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    let ptr = match ptr {
                        Ast::Reference { val } => val,
                        _ => Box::new(ptr),
                    };
                    stack.push(Ast::Array {
                        var: ptr,
                        at: Box::new(index),
                        size: *size as u32,
                    })
                }

                Opcode::LocalU8 { frame_index } => {
                    self.register_local_var(function_index, *frame_index, &mut vars.local);
                    let num_args = self.functions[function_index].num_args;

                    let local_var_index = if *frame_index > num_args {
                        Some(*frame_index as u32 - num_args as u32 - 1)
                    } else {
                        None
                    };
                    stack.push(Ast::Reference {
                        val: Box::new(Ast::Local {
                            index: *frame_index,
                            local_var_index,
                        }),
                    });
                }
                Opcode::LocalU8Load { frame_index } => {
                    self.register_local_var(function_index, *frame_index, &mut vars.local);
                    let num_args = self.functions[function_index].num_args;

                    let local_var_index = if *frame_index > num_args {
                        Some(*frame_index as u32 - num_args as u32 - 1)
                    } else {
                        None
                    };

                    stack.push(Ast::Local {
                        index: *frame_index,
                        local_var_index,
                    });
                }
                Opcode::LocalU8Store { frame_index } => {
                    self.register_local_var(function_index, *frame_index, &mut vars.local);
                    let num_args = self.functions[function_index].num_args;

                    let local_var_index = if *frame_index > num_args {
                        Some(*frame_index as u32 - num_args as u32 - 1)
                    } else {
                        None
                    };
                    let arg = stack
                        .pop(&mut statements, &mut vars.temp, 1)?
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
                Opcode::GlobalU24Store { index } => {
                    let arg = stack
                        .pop(&mut statements, &mut vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;
                    statements.push(Ast::Store {
                        lhs: Box::new(Ast::Global { index: *index }),
                        rhs: Box::new(arg),
                    });
                }
                Opcode::GlobalU16Store { index } => {
                    let arg = stack
                        .pop(&mut statements, &mut vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;
                    statements.push(Ast::Store {
                        lhs: Box::new(Ast::Global {
                            index: *index as u32,
                        }),
                        rhs: Box::new(arg),
                    });
                }
                Opcode::GlobalU16Load { index } => {
                    stack.push(Ast::Global {
                        index: *index as u32,
                    });
                }
                Opcode::GlobalU24Load { index } => {
                    stack.push(Ast::Global { index: *index });
                }
                Opcode::IoffsetU8Store { offset } => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
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
                Opcode::IoffsetS16 { offset } => {
                    let arg = stack
                        .pop(&mut statements, &mut vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;

                    stack.push(Ast::Reference {
                        val: Box::new(Ast::Offset {
                            var: Box::new(arg),
                            offset: *offset as u32,
                        }),
                    });
                }
                Opcode::IoffsetU8 { offset } => {
                    let arg = stack
                        .pop(&mut statements, &mut vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;

                    stack.push(Ast::Reference {
                        val: Box::new(Ast::Offset {
                            var: Box::new(arg),
                            offset: *offset as u32,
                        }),
                    });
                }
                Opcode::IoffsetU8Load { offset } => {
                    let arg = stack
                        .pop(&mut statements, &mut vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;

                    //let arg = match arg {
                    //    Ast::Reference { val } => val,
                    //    _ => Box::new(arg)
                    //};

                    stack.push(Ast::Offset {
                        var: Box::new(arg),
                        offset: *offset as u32,
                    });
                }
                Opcode::IoffsetS16Load { offset } => {
                    let arg = stack
                        .pop(&mut statements, &mut vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;

                    stack.push(Ast::Offset {
                        var: Box::new(arg),
                        offset: *offset as u32,
                    });
                }
                Opcode::IoffsetS16Store { offset } => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
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
                Opcode::Leave { .. } => {
                    if stack.is_empty() {
                        statements.push(Ast::Return { var: None });
                    } else {
                        let len = stack.len();
                        let mut items = stack.pop(&mut statements, &mut vars.temp, len as u32)?;

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
                Opcode::IsBitSet => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (bit, val) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    stack.push(Ast::IsBitSet {
                        val: Box::new(val),
                        bit: Box::new(bit),
                    });
                }
                Opcode::Native {
                    native_hash,
                    num_args,
                    num_returns,
                    ..
                } => {
                    let mut args_list = stack.pop(&mut statements, &mut vars.temp, *num_args as u32)?;
                    args_list.reverse();
                    let native = Ast::Native {
                        num_returns: *num_returns,
                        args_list,
                        hash: *native_hash as usize,
                    };

                    if *num_returns == 0 {
                        statements.push(native);
                    } else {
                        stack.push(native);
                    }
                }
                Opcode::Call { func_index, .. } => {
                    let index = func_index.ok_or(anyhow!("Call did not have valid func index"))?;
                    if depth > 128 || self.active_functions.lock().unwrap().contains(&index) {
                        return Err(anyhow!("Function recursively calls itself"));
                    }
                    let num_args = self.functions[index].num_args;
                    let mut args_list = stack.pop(&mut statements, &mut vars.temp, num_args as u32)?;
                    args_list.reverse();
                    depth += 1;
                    let mut new_stack = AstStack::new();

                    let num_returns =
                        match self.generate_function_with_stack(index, &mut new_stack, depth) {
                            Ok(res) => {
                                depth -= 1;
                                Ok(res.num_returns)
                            }
                            Err(e) => {
                                depth -= 1;
                                Err(e)
                            }
                        }?;

                    let call = Ast::Call {
                        num_returns,
                        args: args_list,
                        index: func_index.unwrap() as u32,
                    };

                    if num_returns == 0 {
                        statements.push(call);
                    } else {
                        stack.push(call);
                    }
                }
                Opcode::Drop => {
                    let ast = stack
                        .pop(&mut statements, &mut vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;
                    if ast.get_stack_size() == 1 {
                        // very janky way of detecting if this is a statement or a temporary value
                        if format!("{ast}").ends_with(';') {
                            statements.push(ast);
                        }
                    }
                }
                Opcode::Dup => {
                    // Fixes Rockstar's jank conditionals
                    if instructions.len() - index > 2 {
                        if matches!(instructions[index + 1], Opcode::Inot) {
                            if matches!(instructions[index + 2], Opcode::Jz { .. }) {
                                index += 2;
                            }
                        }

                        if matches!(instructions[index + 1], Opcode::Jz { .. }) {
                            index += 1;
                        }
                    } else {
                        let ast = stack
                            .pop(&mut statements, &mut vars.temp, 1)?
                            .pop()
                            .ok_or(err())?;
                        stack.push(ast.clone());
                        stack.push(ast);
                    }
                }
                Opcode::Ine => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    stack.push(Ast::IntegerNotEqual {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    })
                }
                Opcode::Ieq => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    stack.push(Ast::IntegerEqual {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    })
                }
                Opcode::String { value, .. } => {
                    let string_index = stack
                        .pop(&mut statements, &mut vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;
                    stack.push(Ast::String {
                        index: Box::new(string_index),
                        value: Some(value.clone()),
                    })
                }
                Opcode::LoadN => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (size, address) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    let size = if let Ast::ConstInt { val } = size {
                        val
                    } else {
                        return Err(anyhow!("LoadN called with non-const size."));
                    };

                    stack.push(Ast::LoadN {
                        address: Box::new(address),
                        size: size as u32,
                    })
                }
                Opcode::StoreN => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (size, lhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    let size = if let Ast::ConstInt { val } = size {
                        val
                    } else {
                        return Err(anyhow!("StoreN called with non-const size."));
                    };

                    let mut stack_items = stack.pop(&mut statements, &mut vars.temp, size as u32)?;
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
                Opcode::Store => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (rhs, lhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    let lhs = Ast::Dereference { val: Box::new(lhs) };

                    statements.push(Ast::Store {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    })
                }
                Opcode::StoreRev => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    let lhs = Ast::Dereference { val: Box::new(lhs) };
                    stack.push(lhs.clone());
                    statements.push(Ast::Store {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    })
                }
                Opcode::Inot => {
                    let arg = stack
                        .pop(&mut statements, &mut vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;
                    stack.push(Ast::Not { val: Box::new(arg) });
                }
                Opcode::Load => {
                    let arg = stack
                        .pop(&mut statements, &mut vars.temp, 1)?
                        .pop()
                        .ok_or(err())?;
                    stack.push(Ast::Dereference { val: Box::new(arg) });
                }
                Opcode::J { offset } if *offset == 0 => {}
                Opcode::Jz { offset } if *offset >= 0 => {
                    let condition = Box::new(stack
                        .pop(&mut statements, &mut vars.temp, 1)?
                        .pop()
                        .ok_or(err())?);
                    self.generate_if(
                        condition,
                        *offset,
                        &mut index,
                        instructions,
                        &mut statements,
                        function_index,
                        stack,
                        depth,
                        vars
                    )?;
                }
                Opcode::IEqJz { offset } if *offset >= 0 => {
                    let mut args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let (lhs, rhs) = (args.pop().ok_or(err())?, args.pop().ok_or(err())?);
                    let condition = Box::new(Ast::IntegerEqual {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    });
                    self.generate_if(
                        condition,
                        *offset,
                        &mut index,
                        instructions,
                        &mut statements,
                        function_index,
                        stack,
                        depth,
                        vars
                    )?;
                }
                Opcode::INeJz { offset } if *offset >= 0 => {
                    let args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let condition = Box::new(Ast::IntegerNotEqual {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone()),
                    });
                    self.generate_if(
                        condition,
                        *offset,
                        &mut index,
                        instructions,
                        &mut statements,
                        function_index,
                        stack,
                        depth,
                        vars
                    )?;
                }
                Opcode::IGtJz { offset } if *offset >= 0 => {
                    let args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let condition = Box::new(Ast::IntGreaterThan {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone()),
                    });
                    self.generate_if(
                        condition,
                        *offset,
                        &mut index,
                        instructions,
                        &mut statements,
                        function_index,
                        stack,
                        depth,
                        vars
                    )?;
                }
                Opcode::ILtJz { offset } if *offset >= 0 => {
                    let args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let condition = Box::new(Ast::IntLessThan {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone()),
                    });
                    self.generate_if(
                        condition,
                        *offset,
                        &mut index,
                        instructions,
                        &mut statements,
                        function_index,
                        stack,
                        depth,
                        vars
                    )?;
                }
                Opcode::ILeJz { offset } if *offset >= 0 => {
                    let args = stack.pop(&mut statements, &mut vars.temp, 2)?;
                    let condition = Box::new(Ast::IntLessThanOrEq {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone()),
                    });
                    self.generate_if(
                        condition,
                        *offset,
                        &mut index,
                        instructions,
                        &mut statements,
                        function_index,
                        stack,
                        depth,
                        vars
                    )?;
                }
                _ => {
                    return Err(anyhow!("unsupported opcode: {inst:?}"));
                }
            }

            index += 1;
        }

        Ok((
            Ast::StatementList {
                list: statements,
                stack_size: stack.len(),
            },
            returns,
            vars.clone()
        ))
    }

    fn register_local_var(&self, func_index: usize, local_index: u8, local_vars: &mut u8) {
        let func = &self.functions[func_index];
        if local_index > func.num_args {
            *local_vars += 1;
        }
    }

    fn get_functions(instructions: Vec<Opcode>) -> Vec<Function> {
        let mut instruction_start_index = 0;

        let mut last_arg_count = 0;

        let mut functions = vec![];
        let mut last_index = 0;

        for (instruction_end_index, (i, inst)) in instructions.iter().enumerate().enumerate() {
            if let Opcode::Enter {
                arg_count, index, ..
            } = inst
            {
                last_index = index.unwrap() - 1;
                if instruction_end_index != 0 {
                    functions.push(Function {
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

        functions.push(Function {
            index: last_index,
            num_args: last_arg_count,
            instructions: instructions[instruction_start_index..].to_vec(),
        });

        functions
    }
}

impl TryFrom<YSCScript> for AstGenerator {
    type Error = Error;
    fn try_from(value: YSCScript) -> std::result::Result<Self, Self::Error> {
        let instructions = Disassembler::new(&value).disassemble(None)?.instructions;
        let ast_gen = Self {
            functions: Arc::new(AstGenerator::get_functions(instructions)),
            lifted_functions: Arc::new(Mutex::new(HashMap::new())),
            active_functions: Arc::new(Mutex::new(HashSet::new())),
        };

        Ok(ast_gen)
    }
}
