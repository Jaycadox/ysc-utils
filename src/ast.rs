use crate::disassemble::{Disassembler, Opcode};
use crate::ysc::YSCScript;
use anyhow::{anyhow, Context, Error, Result};
use std::fmt::{Display, Formatter};
use std::collections::{HashMap, HashSet};

pub fn test() {
    let args = std::env::args().skip(1).collect::<Vec<String>>();

    if args.is_empty() || args.len() != 2 {
        println!("Usage    : ast_gen %ysc_script% %function number/index%");
        println!("Example  : ast_gen freemode.ysc.full");
        println!("Example 2: ast_gen freemode.ysc.full func_305");
        return;
    }

    let function_index: i32 = args[1].replace("func_", "").parse().unwrap();

    let script = YSCScript::from_ysc_file(&args[0])
        .context("Failed to read/parse/disassemble ysc file")
        .unwrap();

    let mut ast_gen = AstGenerator::try_from(script).unwrap();
    let func_ast = ast_gen.generate_function(function_index as usize).unwrap();

    println!("{}", func_ast);
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
    Reference {
        val: Box<Ast>
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
    ConstInt {
        val: i32,
    },
    ConstFloat {
        val: f32
    },
    FloatSub {
        lhs: Box<Ast>,
        rhs: Box<Ast>
    },
    FloatAdd {
        lhs: Box<Ast>,
        rhs: Box<Ast>
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
        bit: Box<Ast>
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
    String {
        index: Box<Ast>,
        value: Option<String>,
    },
    LoadN {
        size: u32,
        address: Box<Ast>,
    }, // todo: make this support non-const int sizes (no idea how though)
    Dereference {
        val: Box<Ast>
    },
    Temporary {
        index: u16,
        field: Option<u16>
    },
    Not {
        val: Box<Ast>
    },
    If {
        condition: Box<Ast>,
        body: Box<Ast>
    },
    Call {
        index: u32,
        args: Vec<Ast>,
        num_returns: u8
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

        let local_vars = if self.local_vars.is_empty() {
            "".to_owned()
        } else {
            let local_vars = format!(
                "\n\tvar {};",
                self.local_vars
                    .iter()
                    .map(|x| format!("local{x}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            let temp_vars = format!(
                "\n\tvar {};\n",
                (0..self.temp_vars)
                    .map(|x| format!("temp_{x}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );

            format!("{local_vars}{temp_vars}")
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
            Ast::Not { .. } => 1,
            Ast::Dereference { .. } => 1,
            Ast::ConstFloat { .. } => 1,
            Ast::Call { num_returns, .. } => *num_returns as usize,
            Ast::FloatSub { .. } => 1,
            Ast::FloatAdd { .. } => 1,
            Ast::Reference { .. } => 1,
        };
    }
}

impl Display for Ast {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let line = match self {
            Ast::Store { lhs, rhs } => {
                format!("{} = {};", lhs, rhs)
            }
            Ast::Return { var } => {
                let mut returns = "".to_owned();
                if let Some(ret) = var {
                    returns = format!(" {ret}");
                }

                format!("return{returns};")
            }
            Ast::Offset { var, offset } => {
                match &**var {
                    Ast::Reference { val } => format!("{val}.f_{offset}"),
                    _ => format!("{var}.f_{offset}")
                }
            }
            Ast::ConstInt { val } => format!("{val}"),
            Ast::StatementList { list, .. } => {
                let mut lines = vec![];
                for ast_token in list {
                    for line in format!("{ast_token}").lines() {
                        lines.push(format!("\t{line}"));
                    }

                    if matches!(ast_token, Ast::Native { .. }) { // Natives can be statements and expressions
                        let len = lines.len();
                        lines[len - 1].push(';');
                    }
                }
                lines.join("\n")
            }
            Ast::StackVariableList { list } => {
                if list.is_empty() {
                    format!("")
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
            Ast::IsBitSet { bit, val } => format!(
                "IS_BIT_SET({}, {})",
                bit,
                val
            ),
            Ast::Native {
                hash,
                args_list,
                ..
            } => {
                format!("0x{hash:X}({})", args_list.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(", "))
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
            Ast::Temporary { index, field } => format!("temp_{index}.f_{}", field.unwrap()),
            Ast::If { condition, body } => format!("if ({condition}) {{\n{body}\n}}"),
            Ast::Not { val } => format!("!{val}"),
            Ast::Dereference { val } => format!("*{val}"),
            Ast::ConstFloat { val } => format!("{val}f"),
            Ast::Call { index, args, .. } => format!("func_{index}({})", args.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(", ")),
            Ast::FloatSub { lhs, rhs } => format!("({lhs} - {rhs})"),
            Ast::FloatAdd { lhs, rhs } => format!("({lhs} + {rhs})"),
            Ast::Reference { val } => format!("&{val}"),
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
    local_vars: HashSet<u8>,
    temp_vars: u16
}

#[derive(Clone)]
struct AstStack {
    stack: Vec<Ast>,
}

impl AstStack {
    fn new() -> Self {
        Self { stack: vec![] }
    }

    fn push(&mut self, ast: Ast) {
        if ast.get_stack_size() != 0 {
            self.stack.push(ast);
        }
    }

    fn pop(&mut self, statements: &mut Vec<Ast>, temp_vars_count: &mut u16, size: u32) -> Result<Vec<Ast>, Error> {
        if size == 0 {
            return Ok(vec![]);
        }
        let og_size = size;
        let mut size_remaining: i64 = size as i64;
        let mut items: Vec<Ast> = vec![];

        for item in self.stack.clone().iter().rev() {
            let size = item.get_stack_size() as i64;
            size_remaining -= size;
            if size != 0 {
                items.push(self.stack.pop().ok_or(anyhow!("Cannot pop stack further"))?);
            }

            if size_remaining < 0 {
                return Err(anyhow!("(Stack overflow). make `Sized` (to reduce a large elements size) and `StackFieldReference` (to reference a single element in a large element) AST type"));
            } else if size_remaining == 0 {
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
                                lhs: Box::new(Ast::Temporary { index: *temp_vars_count, field: None }),
                                rhs: Box::new(item)
                            });
                            for i in (0..size).rev() {
                                single_sized_items.push(Ast::Temporary { index: *temp_vars_count, field: Some(i as u16) })
                            }

                            *temp_vars_count += 1;
                        }
                    }
                    if single_sized_items.len() != og_size as usize {
                        panic!("pop field refs could not get stack pop to match user requested size");
                    }
                    Ok(single_sized_items)
                };
            }
        }

        Err(anyhow!("empty stack"))
    }

    fn len(&self) -> usize {
        self.stack.iter().map(|x| x.get_stack_size()).sum()
    }

    fn clear(&mut self) {
        self.stack.clear()
    }

    fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }
}

pub struct AstGenerator {
    functions: Vec<Function>,
    lifted_functions: HashMap<usize, AstFunction>,
    stack: AstStack,
    function_local_vars: HashMap<usize, HashSet<u8>>,
    function_temp_vars: HashMap<usize, u16>,
}

impl AstGenerator {
    pub fn generate_function(&mut self, index: usize) -> Result<&AstFunction, Error> {
        if self.lifted_functions.contains_key(&index) {
            return Ok(self.lifted_functions.get(&index).unwrap());
        }

        if index >= self.functions.len() {
            return Err(anyhow!(
                "Specified function index is larger than function count"
            ));
        }
        let func = self.functions[index].clone();
    
        let ast_func = self.generate_ast_from_function(func)?;
        self.lifted_functions.insert(index, ast_func);
        Ok(self.lifted_functions.get(&index).unwrap())
    }

    fn generate_ast_from_function(&mut self, function: Function) -> Result<AstFunction, Error> {
        self.stack.clear();
        let (body, num_returns) = self
            .generate_ast(&function.instructions[1..], function.index)?;

        let mut local_vars = HashSet::new();

        if let Some(set) = self.function_local_vars.get(&function.index) {
            local_vars = set.clone();
        }

        let num_temp_vars = match self.function_temp_vars.get(&function.index) {
            Some(&count) => count,
            _ => 0
        };

        let ast_func = AstFunction {
            body,
            num_args: function.num_args,
            num_returns,
            index: function.index,
            local_vars,
            temp_vars: num_temp_vars
        };

        Ok(ast_func)
    }

    fn generate_ast(&mut self, instructions: &[Opcode], function_index: usize) -> Result<(Ast, u8), Error> {
        let mut index = 0;
        let mut statements = vec![];
        let mut temp_vars = 0;
        let mut block_sizes = vec![];
        let mut returns = 0;

        while index < instructions.len() {
            let inst = &instructions[index];
            // let list = Ast::StatementList { list: statements.clone(), stack_size: 0 };
            // println!("INST: {inst:?}");
            // println!("STACK: {:?}", self.stack.stack);
            // println!("ITER:\n{list:#?}\n\n");
            
            match inst {
                Opcode::PushConstM1 => {
                    self.stack.push(Ast::ConstInt { val: -1 });
                }
                Opcode::PushConst0 => {
                    self.stack.push(Ast::ConstInt { val: 0 });
                }
                Opcode::PushConst1 => {
                    self.stack.push(Ast::ConstInt { val: 1 });
                }
                Opcode::PushConst2 => {
                    self.stack.push(Ast::ConstInt { val: 2 });
                }
                Opcode::PushConst3 => {
                    self.stack.push(Ast::ConstInt { val: 3 });
                }
                Opcode::PushConst4 => {
                    self.stack.push(Ast::ConstInt { val: 4 });
                }
                Opcode::PushConst5 => {
                    self.stack.push(Ast::ConstInt { val: 5 });
                }
                Opcode::PushConst6 => {
                    self.stack.push(Ast::ConstInt { val: 6 });
                }
                Opcode::PushConst7 => {
                    self.stack.push(Ast::ConstInt { val: 7 });
                }
                Opcode::PushConstU8 { one } => {
                    self.stack.push(Ast::ConstInt { val: *one as i32 });
                }
                Opcode::PushConstS16 { num } => {
                    self.stack.push(Ast::ConstInt { val: *num as i32 });
                }
                Opcode::PushConstF { one } => {
                    self.stack.push(Ast::ConstFloat {
                        val: *one,
                    });
                }
                Opcode::PushConstU32 { one } => {
                    self.stack.push(Ast::ConstInt {
                        val: *one as i32,
                    });
                }
                Opcode::Fadd => {
                    let args = self.stack.pop(&mut statements, &mut temp_vars, 2)?;
                    self.stack.push(Ast::FloatAdd { lhs: Box::new(args[1].clone()), rhs: Box::new(args[0].clone()) });
                }
                Opcode::Fsub => {
                    let args = self.stack.pop(&mut statements, &mut temp_vars, 2)?;
                    self.stack.push(Ast::FloatSub { lhs: Box::new(args[1].clone()), rhs: Box::new(args[0].clone()) });
                }
                Opcode::GlobalU16 { index } => {
                    self.stack.push(Ast::Reference { val: Box::new(Ast::Global {
                        index: *index as u32,
                    })});
                }
                Opcode::GlobalU24 { index } => {
                    self.stack.push(Ast::Reference { val: Box::new(Ast::Global {
                        index: *index as u32,
                    })});
                }
                Opcode::LocalU8 { frame_index } => {
                    self.register_local_var(function_index, *frame_index);
                    let num_args = self.functions[function_index].num_args;

                    let local_var_index = if *frame_index > num_args {
                        Some(*frame_index as u32 - num_args as u32 - 1 as u32)
                    } else {
                        None
                    };
                    self.stack.push(Ast::Reference { val: Box::new(Ast::Local {
                        index: *frame_index,
                        local_var_index,
                    })});
                }
                Opcode::LocalU8Load { frame_index } => {
                    self.register_local_var(function_index, *frame_index);
                    let num_args = self.functions[function_index].num_args;

                    let local_var_index = if *frame_index > num_args {
                        Some(*frame_index as u32 - num_args as u32 - 1 as u32)
                    } else {
                        None
                    };

                    self.stack.push(Ast::Local {
                        index: *frame_index as u8,
                        local_var_index,
                    });
                }
                Opcode::LocalU8Store { frame_index } => {
                    self.register_local_var(function_index, *frame_index);
                    let num_args = self.functions[function_index].num_args;

                    let local_var_index = if *frame_index > num_args {
                        Some(*frame_index as u32 - num_args as u32 - 1 as u32)
                    } else {
                        None
                    };
                    let arg = self.stack.pop(&mut statements, &mut temp_vars, 1)?[0].clone();
                    statements.push(Ast::Store {
                        lhs: Box::new(Ast::Local { index: *frame_index, local_var_index }),
                        rhs: Box::new(arg)
                    });
                }
                Opcode::GlobalU24Load { index } => {
                    self.stack.push(Ast::Global {
                        index: *index as u32,
                    });
                }
                Opcode::IoffsetU8Store { offset } => {
                    let args = self.stack.pop(&mut statements, &mut temp_vars, 2)?;
                    statements.push(Ast::Store {
                        lhs: Box::new(Ast::Offset { var: Box::new(args[0].clone()), offset: *offset as u32 }),
                        rhs: Box::new(args[1].clone())
                    });
                }
                Opcode::IoffsetU8Load { offset } => {
                    let arg = self.stack.pop(&mut statements, &mut temp_vars, 1)?[0].clone();
                    self.stack.push(Ast::Offset { var: Box::new(arg), offset: *offset as u32 });
                }
                Opcode::IoffsetS16Load { offset } => {
                    let arg = self.stack.pop(&mut statements, &mut temp_vars, 1)?[0].clone();
                    self.stack.push(Ast::Offset { var: Box::new(arg), offset: *offset as u32 });
                }
                Opcode::IoffsetS16Store { offset } => {
                    let args = self.stack.pop(&mut statements, &mut temp_vars, 2)?;
                    statements.push(Ast::Store {
                        lhs: Box::new(Ast::Offset { var: Box::new(args[0].clone()), offset: *offset as u32 }),
                        rhs: Box::new(args[1].clone())
                    });
                }
                Opcode::Leave { .. } => {
                    if self.stack.is_empty() {
                        statements.push(Ast::Return { var: None });
                    } else {
                        let len = self.stack.len();
                        let items = self.stack.clone().pop(&mut statements, &mut temp_vars, len as u32)?;
                        if items.is_empty() {
                            panic!("empty return when stack wasn't empty");
                        } else if items.len() == 1 {
                            statements.push(Ast::Return { var: Some(Box::new(items[0].clone())) });
                        } else {
                            statements.push(Ast::Return { var: Some(Box::new(Ast::StackVariableList { list: items } ))});
                        }
                        returns = self.stack.len() as u8;
                        self.stack.clear();
                        
                    }
                }
                Opcode::IsBitSet => {
                    let args = Box::new(self.stack.pop(&mut statements, &mut temp_vars,2)?);

                    self.stack.push(Ast::IsBitSet {
                        val: Box::new(args[0].clone()),
                        bit: Box::new(args[1].clone()),
                    });
                }
                Opcode::Native {
                    native_hash,
                    num_args,
                    num_returns,
                    ..
                } => {
                    let mut args_list = self.stack.pop(&mut statements, &mut temp_vars, *num_args as u32)?;
                    args_list.reverse();
                    let native = Ast::Native {
                        num_returns: *num_returns,
                        args_list,
                        hash: *native_hash as usize,
                    };

                    if *num_returns == 0 {
                        statements.push(native);
                    } else {
                        self.stack.push(native);
                    }
                }
                Opcode::Call { func_index, .. } => {
                    let index = func_index.ok_or(anyhow!("Call did not have valid func index"))? - 1;

                    let num_args = self.functions[func_index.unwrap()].num_args;
                    let mut args_list = self.stack.pop(&mut statements, &mut temp_vars, num_args as u32)?;
                    args_list.reverse();
                    let num_returns = self.generate_function(index)?.num_returns;

                    let call = Ast::Call {
                        num_returns,
                        args: args_list,
                        index: func_index.unwrap() as u32
                    };

                    if num_returns == 0 {
                        statements.push(call);
                    } else {
                        self.stack.push(call);
                    }
                }
                Opcode::Drop => {
                    let ast = self.stack.pop(&mut statements, &mut temp_vars, 1)?[0].clone();
                    if ast.get_stack_size() == 1 {
                        statements.push(ast);
                    }
                }
                Opcode::Ine => {
                    let args = self.stack.pop(&mut statements, &mut temp_vars, 2)?;
                    self.stack.push(Ast::IntegerNotEqual {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone())
                    })
                }
                Opcode::Ieq => {
                    let args = self.stack.pop(&mut statements, &mut temp_vars, 2)?;
                    self.stack.push(Ast::IntegerEqual {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone())
                    })
                }
                Opcode::String { value, .. } => {
                    let string_index = self.stack.pop(&mut statements, &mut temp_vars, 1)?.pop().unwrap();
                    self.stack.push(Ast::String {
                        index: Box::new(string_index),
                        value: Some(value.clone()),
                    })
                }
                Opcode::LoadN => {
                    let args = self.stack.pop(&mut statements, &mut temp_vars, 2)?;
                    let size;
                    if let Ast::ConstInt { val } = args[1] {
                        size = val;
                    } else {
                        panic!("LoadN called with non-const size.")
                    }

                    self.stack.push(Ast::LoadN {
                        address: Box::new(args[0].clone()),
                        size: size as u32,
                    })
                }
                Opcode::StoreN => {
                    let args = self.stack.pop(&mut statements, &mut temp_vars, 2)?;
                    let lhs = args[0].clone() ;
                    let size;
                    if let Ast::ConstInt { val } = args[1] {
                        size = val;
                    } else {
                        panic!("StoreN called with non-const size.")
                    }
                    let mut stack_items = self.stack.pop(&mut statements, &mut temp_vars, size as u32)?;
                    if stack_items.len() == 1 {
                        statements.push(Ast::Store { lhs: Box::new(lhs), rhs: Box::new(stack_items[0].clone()) });
                    } else {
                        let lhs = Ast::Dereference { val: Box::new(lhs) } ;
                        for i in 0..stack_items.len() {
                            let rhs = Box::new(stack_items.remove(0)); // todo: improve speed
                            statements.push(
                                Ast::Store { lhs: Box::new(Ast::Offset { var: Box::new(lhs.clone()), offset: i as u32 }), rhs }
                            )
                        }
                    }
                }
                Opcode::Store => {
                    let args = self.stack.pop(&mut statements, &mut temp_vars, 2)?;
                    let lhs = Ast::Dereference { val: Box::new(args[0].clone()) } ;
                    let rhs = args[1].clone();

                    statements.push(
                        Ast::Store { lhs: Box::new(lhs), rhs: Box::new(rhs) }
                    )
                }
                Opcode::Inot => {
                    let arg = self.stack.pop(&mut statements, &mut temp_vars, 1)?[0].clone();
                    self.stack.push(Ast::Not { val: Box::new(arg) });
                }
                Opcode::Load =>{
                    let arg = self.stack.pop(&mut statements, &mut temp_vars, 1)?[0].clone();
                    self.stack.push(Ast::Dereference { val: Box::new(arg) });
                }
                Opcode::Jz { offset } if *offset > 0 => {
                    let condition = self.stack.pop(&mut statements, &mut temp_vars, 1)?[0].clone();

                    let mut offset_remaining = *offset;
                    let mut instructions_in_block = vec![];
                    index += 1;

                    while offset_remaining > 0 {
                        let inst = instructions[index].clone();
                        offset_remaining -= inst.get_size() as i16; // todo: this might be too small when factoring large switches
                        instructions_in_block.push(inst);
                        if offset_remaining != 0 {
                            index += 1;
                        }
                    }

                    statements.push(Ast::If { condition: Box::new(condition), body: Box::new(self.generate_ast(&instructions_in_block[..], function_index)?.0) });
                }
                Opcode::IEqJz { offset } if *offset > 0 => {
                    let args = self.stack.pop(&mut statements, &mut temp_vars, 2)?;
                    let condition = Box::new(
                        Ast::IntegerEqual { lhs: Box::new(args[1].clone()), rhs: Box::new(args[0].clone()) }
                    );

                    let mut offset_remaining = *offset;
                    let mut instructions_in_block = vec![];
                    index += 1;

                    while offset_remaining > 0 {
                        let inst = instructions[index].clone();
                        offset_remaining -= inst.get_size() as i16; // todo: this might be too small when factoring large switches
                        instructions_in_block.push(inst);
                        if offset_remaining != 0 {
                            index += 1;
                        }
                    }

                    statements.push(Ast::If { condition, body: Box::new(self.generate_ast(&instructions_in_block[..], function_index)?.0) });
                }
                Opcode::INeJz { offset } if *offset > 0 => {
                    let args = self.stack.pop(&mut statements, &mut temp_vars, 2)?;
                    let condition = Box::new(
                        Ast::IntegerNotEqual { lhs: Box::new(args[0].clone()), rhs: Box::new(args[1].clone()) }
                    );

                    let mut offset_remaining = *offset;
                    let mut instructions_in_block = vec![];
                    index += 1;

                    while offset_remaining > 0 {
                        let inst = instructions[index].clone();
                        offset_remaining -= inst.get_size() as i16; // todo: this might be too small when factoring large switches
                        instructions_in_block.push(inst);
                        if offset_remaining != 0 {
                            index += 1;
                        }
                    }

                    statements.push(Ast::If { condition, body: Box::new(self.generate_ast(&instructions_in_block[..], function_index)?.0) });
                }
                _ => {
                    panic!("unsupported opcode: {inst:?}");
                }
            }
            block_sizes.push(inst.get_size());
            index += 1;
        }
        
        if !self.function_temp_vars.contains_key(&function_index) {
            self.function_temp_vars.insert(function_index, temp_vars);
        } else {
            *self.function_temp_vars.get_mut(&function_index).unwrap() += temp_vars;
        }

        Ok((Ast::StatementList {
            list: statements,
            stack_size: self.stack.len(),
        }, returns))
    }

    fn register_local_var(&mut self, func_index: usize, local_index: u8) {
        let func = &self.functions[func_index];
        if local_index > func.num_args {
            let arg_index = local_index - func.num_args - 1;

            if let Some(set) = self.function_local_vars.get_mut(&func_index) {
                set.insert(arg_index);
            } else {
                self.function_local_vars.insert(func_index, HashSet::new());
                self.function_local_vars
                    .get_mut(&func_index)
                    .unwrap()
                    .insert(arg_index);
            }
        }
    }

    fn get_functions(instructions: Vec<Opcode>) -> Vec<Function> {
        let mut instruction_start_index = 0;
        let mut instruction_end_index = 0;

        let mut last_arg_count = 0;

        let mut functions = vec![];

        for (i, inst) in instructions.iter().enumerate() {
            if let Opcode::Enter {
                arg_count, index, ..
            } = inst
            {
                if instruction_end_index != 0 {
                    functions.push(Function {
                        index: (index.unwrap() - 1) as usize,
                        num_args: last_arg_count,
                        instructions: (&instructions
                            [instruction_start_index..instruction_end_index])
                            .to_vec(),
                    })
                }
                last_arg_count = *arg_count;
                instruction_start_index = i;
            }

            instruction_end_index += 1;
        }

        functions
    }
}

impl TryFrom<YSCScript> for AstGenerator {
    type Error = Error;
    fn try_from(value: YSCScript) -> std::result::Result<Self, Self::Error> {
        let instructions = Disassembler::new(&value).disassemble(None)?.instructions;
        let ast_gen = Self {
            functions: AstGenerator::get_functions(instructions),
            lifted_functions: HashMap::new(),
            stack: AstStack::new(),
            function_local_vars: HashMap::new(),
            function_temp_vars: HashMap::new()
        };

        Ok(ast_gen)
    }
}
