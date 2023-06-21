use crate::disassemble::{Disassembler, Opcode};
use crate::ysc::YSCScript;
use anyhow::{anyhow, Context, Error, Result};
use eframe::egui::Key::P;
use eframe::epaint::ahash::{HashMap, HashMapExt, HashSet, HashSetExt};
use std::cell::Cell;
use std::fmt::{Display, Formatter};
use std::rc::Rc;

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
        stack_var_list: Box<Ast>,
    },
    Return {
        var: Option<Box<Ast>>,
    },
    Offset {
        var: Box<Ast>,
        offset: u32,
    },
    GlobalAddress {
        index: u32,
    },
    LocalAddress {
        index: u32,
        local_var_index: Option<u32>,
    },
    Global {
        index: u32,
    },
    ConstInt {
        val: u32,
    },
    StatementList {
        list: Vec<Ast>,
        stack_size: usize,
    },
    StackVariableList {
        list: Vec<Ast>,
    },
    LocalVariable {
        index: u8,
        local_var_index: Option<u32>,
    },
    IsBitSet {
        stack_var_list: Box<Ast>,
    },
    Native {
        args_list: Box<Ast>,
        num_returns: u8,
        hash: usize,
    },
    IntegerNotEqual {
        args_list: Box<Ast>,
    },
    IntegerEqual {
        args_list: Box<Ast>,
    },
    String {
        index: Box<Ast>,
        value: Option<String>,
    },
    LoadN {
        size: u32,
        address: Box<Ast>,
    }, // todo: make this support non-const int sizes (no idea how though)
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
            format!(
                "\n\tvar {};\n",
                self.local_vars
                    .iter()
                    .map(|x| format!("local{x}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
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
            Ast::GlobalAddress { .. } => 1,
            Ast::Global { .. } => 1,
            Ast::ConstInt { .. } => 1,
            Ast::StatementList { stack_size, .. } => *stack_size,
            Ast::StackVariableList { list } => list.iter().map(|x| x.get_stack_size()).sum(),
            Ast::LocalVariable { .. } => 1,
            Ast::IsBitSet { .. } => 1,
            Ast::Native { num_returns, .. } => *num_returns as usize,
            Ast::IntegerNotEqual { .. } => 1,
            Ast::IntegerEqual { .. } => 1,
            Ast::String { .. } => 1,
            Ast::LocalAddress { .. } => 1,
            Ast::LoadN { size, .. } => *size as usize,
        };
    }

    fn reversed(&self) -> Self {
        return match &self {
            Ast::StackVariableList { list } => {
                let mut l = list.clone();
                l.reverse();
                Ast::StackVariableList { list: l }
            }
            _ => self.clone()
        }
    }

    fn at(&self, index: usize) -> &Ast {
        // todo: similar thing to .pop(), in-case stack isn't aligned
        return if let Ast::StackVariableList { list } = &self {
            &list[index]
        } else if index != 0 {
            panic!("attempted to get stack variable index for non stack variable list")
        } else {
            &self
        };
    }
}

impl Display for Ast {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let line = match self {
            Ast::Store { stack_var_list } => {
                format!("{} = {};", stack_var_list.at(0), stack_var_list.at(1))
            }
            Ast::Return { var } => {
                let mut returns = "".to_owned();
                if let Some(ret) = var {
                    returns = format!(" {ret}");
                }

                format!("return{returns};")
            }
            Ast::Offset { var, offset } => {
                if let Ast::GlobalAddress { index } = *var.clone() {
                    format!("Global_{index}.f_{offset}")
                } else {
                    format!("*({var} + {offset})")
                }
            }
            Ast::GlobalAddress { index } => format!("&Global_{index}"),
            Ast::LocalAddress {
                index,
                local_var_index,
            } => match local_var_index {
                Some(loc) => format!("&local{loc}"),
                _ => format!("&arg{index}"),
            },
            Ast::ConstInt { val } => format!("{val}"),
            Ast::StatementList { list, stack_size } => {
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
                        "{}",
                        list.iter()
                            .map(|ast| format!("{}", ast))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            }
            Ast::LocalVariable {
                index,
                local_var_index,
            } => match local_var_index {
                Some(loc) => format!("local{loc}"),
                _ => format!("arg{index}"),
            },
            Ast::Global { index } => format!("Global_{index}"),
            Ast::IsBitSet { stack_var_list } => format!(
                "IS_BIT_SET({}, {})",
                stack_var_list.at(1),
                stack_var_list.at(0)
            ),
            Ast::Native {
                hash,
                args_list,
                ..
            } => {
                format!("0x{hash:X}({})", args_list.reversed())
            }
            Ast::IntegerNotEqual { args_list } => {
                format!("{} != {}", args_list.at(1), args_list.at(0))
            }
            Ast::IntegerEqual { args_list } => {
                format!("{} == {}", args_list.at(1), args_list.at(0))
            }
            Ast::String { index, value } => {
                if let Some(str_val) = value {
                    format!("\"{str_val}\"")
                } else {
                    format!("_STRING({index})")
                }
            }
            Ast::LoadN { size, address } => format!("*{address}"),
        };

        write!(f, "{}", line)
    }
}

enum StackVariable {
    I32,
    F32,
    S16,
    U8,
}

#[derive(Debug)]
pub struct AstFunction {
    body: Ast,
    num_args: u8,
    num_returns: u8,
    index: usize,
    local_vars: HashSet<u8>,
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

    fn pop(&mut self, size: u32) -> Option<Ast> {
        if size == 0 {
            return Some(Ast::StackVariableList { list: vec![] });
        }

        let mut size_remaining: i64 = size as i64;
        let mut items: Vec<Ast> = vec![];

        for item in self.stack.clone().iter().rev() {
            let size = item.get_stack_size() as i64;
            size_remaining -= size;
            items.push(self.stack.pop()?);

            if size_remaining < 0 {
                // We've overshot our stack
                todo!("(Stack overflow). make `Sized` (to reduce a large elements size) and `StackFieldReference` (to reference a single element in a large element) AST type");
            } else if size_remaining == 0 {
                return if items.len() == 1 {
                    Some(items[0].clone())
                } else {
                    Some(Ast::StackVariableList { list: items })
                };
            }
        }

        None
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
    script: YSCScript,
    stack: AstStack,
    function_local_vars: HashMap<usize, HashSet<u8>>,
}

impl AstGenerator {
    pub fn generate_function(&mut self, index: usize) -> Result<AstFunction, Error> {
        if index >= self.functions.len() {
            return Err(anyhow!(
                "Specified function index is larger than function count"
            ));
        }
        let func = self.functions[index].clone();
        self.generate_ast_from_function(func)
    }

    fn generate_ast_from_function(&mut self, function: Function) -> Result<AstFunction, Error> {
        let mut i = 0;

        self.stack.clear();
        let body = self
            .generate_ast(&function.instructions[1..], function.index)
            .ok_or(anyhow!("Error generating function AST"))?;

        let mut local_vars = HashSet::new();

        if let Some(set) = self.function_local_vars.get(&function.index) {
            local_vars = set.clone();
        }

        let ast_func = AstFunction {
            body,
            num_args: function.num_args,
            num_returns: self.stack.len() as u8,
            index: function.index,
            local_vars,
        };

        Ok(ast_func)
    }

    fn generate_ast(&mut self, instructions: &[Opcode], function_index: usize) -> Option<Ast> {
        let mut index = 0;
        let mut statements = vec![];

        while index < instructions.len() {
            let inst = &instructions[index];
            match inst {
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
                    self.stack.push(Ast::ConstInt { val: *one as u32 });
                }
                Opcode::PushConstS16 { num } => {
                    self.stack.push(Ast::ConstInt { val: *num as u32 });
                }
                Opcode::GlobalU16 { index } => {
                    self.stack.push(Ast::GlobalAddress {
                        index: *index as u32,
                    });
                }
                Opcode::GlobalU24 { index } => {
                    self.stack.push(Ast::GlobalAddress {
                        index: *index as u32,
                    });
                }
                Opcode::LocalU8 { frame_index } => {
                    self.register_local_var(function_index, *frame_index);
                    let num_args = self.functions[function_index].num_args;

                    let local_var_index = if *frame_index > num_args {
                        Some(*frame_index as u32 - num_args as u32 - 1 as u32)
                    } else {
                        None
                    };

                    self.stack.push(Ast::LocalAddress {
                        index: *frame_index as u32,
                        local_var_index,
                    });
                }
                Opcode::GlobalU24Load { index } => {
                    self.stack.push(Ast::Global {
                        index: *index as u32,
                    });
                }
                Opcode::IoffsetU8Store { offset } => {
                    let lhs = Box::new(self.stack.pop(1)?);
                    let rhs = Box::new(self.stack.pop(1)?);

                    statements.push(Ast::Store {
                        stack_var_list: Box::new(Ast::StackVariableList {
                            list: vec![
                                Ast::Offset {
                                    var: lhs,
                                    offset: *offset as u32,
                                },
                                *rhs,
                            ],
                        }),
                    });
                }
                Opcode::IoffsetS16Store { offset } => {
                    let lhs = Box::new(self.stack.pop(1)?);
                    let rhs = Box::new(self.stack.pop(1)?);

                    statements.push(Ast::Store {
                        stack_var_list: Box::new(Ast::StackVariableList {
                            list: vec![
                                Ast::Offset {
                                    var: lhs,
                                    offset: *offset as u32,
                                },
                                *rhs,
                            ],
                        }),
                    });
                }
                Opcode::Leave { .. } => {
                    statements.push(Ast::Return {
                        var: if self.stack.is_empty() {
                            None
                        } else {
                            let len = self.stack.len();
                            Some(Box::new(self.stack.clone().pop(len as u32)?))
                        },
                    });
                }
                Opcode::LocalU8Load { frame_index } => {
                    self.register_local_var(function_index, *frame_index);

                    let num_args = self.functions[function_index].num_args;

                    let local_var_index = if *frame_index > num_args {
                        Some(*frame_index as u32 - num_args as u32 - 1 as u32)
                    } else {
                        None
                    };

                    self.stack.push(Ast::LocalVariable {
                        index: *frame_index,
                        local_var_index,
                    });
                }
                Opcode::IsBitSet => {
                    let stack_list = Box::new(self.stack.pop(2)?);

                    self.stack.push(Ast::IsBitSet {
                        stack_var_list: stack_list,
                    })
                }
                Opcode::Native {
                    native_hash,
                    num_args,
                    num_returns,
                    ..
                } => {
                    let args_list = Box::new(self.stack.pop(*num_args as u32)?);
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
                Opcode::Ine => {
                    let args = self.stack.pop(2)?;
                    self.stack.push(Ast::IntegerNotEqual {
                        args_list: Box::new(args),
                    })
                }
                Opcode::Ieq => {
                    let args = self.stack.pop(2)?;
                    self.stack.push(Ast::IntegerEqual {
                        args_list: Box::new(args),
                    })
                }
                Opcode::String { value, .. } => {
                    let string_index = self.stack.pop(1)?;
                    self.stack.push(Ast::String {
                        index: Box::new(string_index),
                        value: Some(value.clone()),
                    })
                }
                Opcode::LoadN => {
                    let ptr = self.stack.pop(1)?;
                    let size;
                    if let Ast::ConstInt { val } = &self.stack.pop(1)? {
                        size = *val;
                    } else {
                        panic!("LoadN called with non-const size.")
                    }

                    self.stack.push(Ast::LoadN {
                        address: Box::new(ptr),
                        size,
                    })
                }
                _ => {
                    panic!("unsupported opcode: {inst:?}");
                }
            }

            index += 1;
        }

        Some(Ast::StatementList {
            list: statements,
            stack_size: self.stack.len(),
        })
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
            script: value,
            stack: AstStack::new(),
            function_local_vars: HashMap::new(),
        };

        Ok(ast_gen)
    }
}
