use crate::disassemble::{Disassembler, Opcode};
use crate::ysc::YSCScript;
use anyhow::{anyhow, Context, Error, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex};

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

    let ast_gen = Arc::new(AstGenerator::try_from(script).unwrap());
    if false {
        let num_pass = AtomicU32::new(0);
        println!("Starting...");
        let then = std::time::Instant::now();
        let funcs = (0..ast_gen.functions.len()).collect::<Vec<_>>();
        funcs.iter().for_each(|i| {
            if let Ok(_func) = ast_gen.generate_function(*i) {
                num_pass.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            }
        });
        let now = std::time::Instant::now();
        let time = now.duration_since(then).as_millis();
        let num_pass: u32 = num_pass.into_inner();
        println!(
            "Result: {num_pass}/{} in {}ms ({}ms/func)",
            ast_gen.functions.len(),
            time,
            time as f32 / num_pass as f32
        );
    } else {
        println!(
            "{}",
            ast_gen.generate_function(function_index as usize).unwrap()
        );
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
    FloatAdd {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    IntAdd {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },
    IntSub {
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

        let local_vars = if self.local_vars == 0 && self.temp_vars == 0 {
            "".to_owned()
        } else {
            let mut vars_str = "".to_owned();
            if self.local_vars != 0 {
                let local_vars = format!(
                    "\n\tvar {};",
                    (0..self.local_vars)
                        .map(|x| format!("local{x}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                vars_str.push_str(&local_vars);
            }
            if self.temp_vars != 0 {
                let temp_vars = format!(
                    "\n\tvar {};\n",
                    (0..self.temp_vars)
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
            Ast::IntAdd { .. } => 1,
            Ast::IntSub { .. } => 1,
            Ast::IntLessThan { .. } => 1,
            Ast::IntGreaterThan { .. } => 1,
            Ast::IntegerOr { .. } => 1,
            Ast::Array { .. } => 1,
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
            Ast::Offset { var, offset } => format!("{var}.f_{offset}"),
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
            Ast::Temporary { index, field } => format!("temp_{index}.f_{}", field.unwrap()),
            Ast::If { condition, body } => format!("if ({condition}) {{\n{body}\n}}"),
            Ast::Not { val } => format!("!{val}"),
            Ast::Dereference { val } => format!("*{val}"),
            Ast::ConstFloat { val } => format!("{val}f"),
            Ast::Call { index, args, .. } => format!(
                "func_{index}({})",
                args.iter()
                    .map(|x| format!("{x}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Ast::FloatSub { lhs, rhs } => format!("({lhs} - {rhs})"),
            Ast::FloatAdd { lhs, rhs } => format!("({lhs} + {rhs})"),
            Ast::Reference { val } => format!("&{val}"),
            Ast::IntAdd { lhs, rhs } => format!("({lhs} + {rhs})"),
            Ast::IntSub { lhs, rhs } => format!("({lhs} + {rhs})"),
            Ast::IntLessThan { lhs, rhs } => format!("({lhs} < {rhs})"),
            Ast::IntGreaterThan { lhs, rhs } => format!("({lhs} > {rhs})"),
            Ast::IntegerOr { lhs, rhs } => format!("({lhs} || {rhs})"),
            Ast::Array { var, at, size } => format!("{var}[{at} /*{size}*/]"),
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
    local_vars: u8,
    temp_vars: u8,
}

#[derive(Clone)]
pub struct AstStack {
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

        for item in self.stack.clone().iter().rev() {
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

pub struct AstGenerator {
    functions: Arc<Vec<Function>>,
    lifted_functions: Arc<Mutex<HashMap<usize, Arc<AstFunction>>>>,
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
        let func = self.functions[index].clone();

        let ast_func = Arc::new(self.generate_ast_from_function(func, stack, depth)?);
        let mut map = self.lifted_functions.lock().unwrap();
        map.entry(index).or_insert_with(|| Arc::clone(&ast_func));
        Ok(Arc::clone(&ast_func))
    }

    fn generate_ast_from_function(
        &self,
        function: Function,
        stack: &mut AstStack,
        depth: usize,
    ) -> Result<AstFunction, Error> {
        let (body, num_returns, local_vars, temp_vars) = self.generate_ast(
            &function.instructions[1..],
            function.index,
            stack,
            depth,
            &mut 0,
            &mut 0,
        )?;

        let ast_func = AstFunction {
            body,
            num_args: function.num_args,
            num_returns,
            index: function.index,
            local_vars,
            temp_vars,
        };

        Ok(ast_func)
    }

    fn generate_ast(
        &self,
        instructions: &[Opcode],
        function_index: usize,
        stack: &mut AstStack,
        mut depth: usize,
        local_vars: &mut u8,
        temp_vars: &mut u8,
    ) -> Result<(Ast, u8, u8, u8), Error> {
        let mut index = 0;
        let mut statements = vec![];
        let mut block_sizes = vec![];
        let mut returns = 0;

        while index < instructions.len() {
            let inst = &instructions[index];

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
                Opcode::PushConstU32 { one } => {
                    stack.push(Ast::ConstInt { val: *one as i32 });
                }
                Opcode::Nop => {}
                Opcode::Ilt => {
                    let args = stack.pop(&mut statements, temp_vars, 2)?;
                    stack.push(Ast::IntLessThan {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone()),
                    });
                }
                Opcode::Igt => {
                    let args = stack.pop(&mut statements, temp_vars, 2)?;
                    stack.push(Ast::IntGreaterThan {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone()),
                    });
                }
                Opcode::Ior => {
                    let args = stack.pop(&mut statements, temp_vars, 2)?;
                    stack.push(Ast::IntegerOr {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone()),
                    });
                }
                Opcode::Iadd => {
                    let args = stack.pop(&mut statements, temp_vars, 2)?;
                    stack.push(Ast::IntAdd {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone()),
                    });
                }
                Opcode::IaddS16 { num } => {
                    let arg = stack.pop(&mut statements, temp_vars, 1)?[0].clone();
                    stack.push(Ast::IntAdd {
                        lhs: Box::new(arg),
                        rhs: Box::new(Ast::ConstInt { val: *num as i32 }),
                    });
                }
                Opcode::IaddU8 { num } => {
                    let arg = stack.pop(&mut statements, temp_vars, 1)?[0].clone();
                    stack.push(Ast::IntAdd {
                        lhs: Box::new(arg),
                        rhs: Box::new(Ast::ConstInt { val: *num as i32 }),
                    });
                }
                Opcode::Isub => {
                    let args = stack.pop(&mut statements, temp_vars, 2)?;
                    stack.push(Ast::IntSub {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone()),
                    });
                }
                Opcode::Fsub => {
                    let args = stack.pop(&mut statements, temp_vars, 2)?;
                    stack.push(Ast::FloatSub {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone()),
                    });
                }
                Opcode::Fadd => {
                    let args = stack.pop(&mut statements, temp_vars, 2)?;
                    stack.push(Ast::FloatAdd {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone()),
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
                    let args = stack.pop(&mut statements, temp_vars, 2)?;
                    let (index, ptr) = (args[1].clone(), args[0].clone());
                    let ptr = match ptr {
                        Ast::Reference { val } => val,
                        _ => Box::new(ptr)
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
                    let args = stack.pop(&mut statements, temp_vars, 2)?;
                    let (index, ptr) = (args[1].clone(), args[0].clone());
                    let ptr = match ptr {
                        Ast::Reference { val } => val,
                        _ => Box::new(ptr)
                    };
                    stack.push(Ast::Array {
                        var: ptr,
                        at: Box::new(index),
                        size: *size as u32,
                    })
                }
                Opcode::ArrayU16 { size } => {
                    let args = stack.pop(&mut statements, temp_vars, 2)?;
                    let (index, ptr) = (args[1].clone(), args[0].clone());
                    let ptr = match ptr {
                        Ast::Reference { val } => val,
                        _ => Box::new(ptr)
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
                    let args = stack.pop(&mut statements, temp_vars, 2)?;
                    let (index, ptr) = (args[1].clone(), args[0].clone());
                    let ptr = match ptr {
                        Ast::Reference { val } => val,
                        _ => Box::new(ptr)
                    };
                    stack.push(Ast::Array {
                        var: ptr,
                        at: Box::new(index),
                        size: *size as u32,
                    })
                }

                Opcode::LocalU8 { frame_index } => {
                    self.register_local_var(function_index, *frame_index, local_vars);
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
                    self.register_local_var(function_index, *frame_index, local_vars);
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
                    self.register_local_var(function_index, *frame_index, local_vars);
                    let num_args = self.functions[function_index].num_args;

                    let local_var_index = if *frame_index > num_args {
                        Some(*frame_index as u32 - num_args as u32 - 1)
                    } else {
                        None
                    };
                    let arg = stack.pop(&mut statements, temp_vars, 1)?[0].clone();
                    statements.push(Ast::Store {
                        lhs: Box::new(Ast::Local {
                            index: *frame_index,
                            local_var_index,
                        }),
                        rhs: Box::new(arg),
                    });
                }
                Opcode::GlobalU24Load { index } => {
                    stack.push(Ast::Global { index: *index });
                }

                Opcode::IoffsetU8Store { offset } => {
                    let args = stack.pop(&mut statements, temp_vars, 2)?;
                    let var = match &args[0] {
                        Ast::Reference { val } => val.clone(),
                        _ => Box::new(args[0].clone()),
                    };

                    statements.push(Ast::Store {
                        lhs: Box::new(Ast::Offset {
                            var,
                            offset: *offset as u32,
                        }),
                        rhs: Box::new(args[1].clone()),
                    });
                }
                Opcode::IoffsetS16 { offset } => {
                    let arg = stack.pop(&mut statements, temp_vars, 1)?[0].clone();
                    let var = match arg {
                        Ast::Reference { val } => val,
                        _ => Box::new(arg.clone()),
                    };
                    stack.push(Ast::Reference {
                        val: Box::new(Ast::Offset {
                            var,
                            offset: *offset as u32,
                        }),
                    });
                }
                Opcode::IoffsetU8 { offset } => {
                    let arg = stack.pop(&mut statements, temp_vars, 1)?[0].clone();
                    let var = match arg {
                        Ast::Reference { val } => val,
                        _ => Box::new(arg.clone()),
                    };
                    stack.push(Ast::Reference {
                        val: Box::new(Ast::Offset {
                            var,
                            offset: *offset as u32,
                        }),
                    });
                }
                Opcode::IoffsetU8Load { offset } => {
                    let arg = stack.pop(&mut statements, temp_vars, 1)?[0].clone();
                    let var = match arg {
                        Ast::Reference { val } => val,
                        _ => Box::new(arg.clone()),
                    };
                    stack.push(Ast::Offset {
                        var,
                        offset: *offset as u32,
                    });
                }
                Opcode::IoffsetS16Load { offset } => {
                    let arg = stack.pop(&mut statements, temp_vars, 1)?[0].clone();
                    let var = match arg {
                        Ast::Reference { val } => val,
                        _ => Box::new(arg.clone()),
                    };
                    stack.push(Ast::Offset {
                        var,
                        offset: *offset as u32,
                    });
                }
                Opcode::IoffsetS16Store { offset } => {
                    let args = stack.pop(&mut statements, temp_vars, 2)?;
                    let var = args[0].clone();
                    let var = match var {
                        Ast::Reference { val } => val,
                        _ => Box::new(var),
                    };
                    statements.push(Ast::Store {
                        lhs: Box::new(Ast::Offset {
                            var,
                            offset: *offset as u32,
                        }),
                        rhs: Box::new(args[1].clone()),
                    });
                }
                Opcode::Leave { arg_count, .. } => {
                    if stack.is_empty() {
                        statements.push(Ast::Return { var: None });
                    } else {
                        let len = stack.len();
                        let items = stack.pop(&mut statements, temp_vars, len as u32)?;
                        if items.len() == 1 {
                            statements.push(Ast::Return {
                                var: Some(Box::new(items[0].clone())),
                            });
                        } else {
                            statements.push(Ast::Return {
                                var: Some(Box::new(Ast::StackVariableList { list: items })),
                            });
                        }
                        returns = (stack.len() - *arg_count as usize) as u8;
                    }
                }
                Opcode::IsBitSet => {
                    let args = Box::new(stack.pop(&mut statements, temp_vars, 2)?);

                    stack.push(Ast::IsBitSet {
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
                    let mut args_list = stack.pop(&mut statements, temp_vars, *num_args as u32)?;
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
                    if depth > 128 {
                        return Err(anyhow!("Function recursively calls it's self"));
                    }

                    let num_args = self.functions[index].num_args;
                    let mut args_list = stack.pop(&mut statements, temp_vars, num_args as u32)?;
                    args_list.reverse();
                    depth += 1;
                    let mut new_stack = AstStack::new();
                    for arg in &args_list {
                        new_stack.push(arg.clone());
                    }

                    let num_returns = self
                        .generate_function_with_stack(index, &mut new_stack, depth)?
                        .num_returns;
                    depth = 0;
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
                    let ast = stack.pop(&mut statements, temp_vars, 1)?[0].clone();
                    if ast.get_stack_size() == 1 {
                        statements.push(ast);
                    }
                }
                Opcode::Dup => {
                    let ast = stack.pop(&mut statements, temp_vars, 1)?[0].clone();
                    stack.push(ast.clone());
                    stack.push(ast);
                }
                Opcode::Ine => {
                    let args = stack.pop(&mut statements, temp_vars, 2)?;
                    stack.push(Ast::IntegerNotEqual {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone()),
                    })
                }
                Opcode::Ieq => {
                    let args = stack.pop(&mut statements, temp_vars, 2)?;
                    stack.push(Ast::IntegerEqual {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone()),
                    })
                }
                Opcode::String { value, .. } => {
                    let string_index = stack.pop(&mut statements, temp_vars, 1)?.pop().unwrap();
                    stack.push(Ast::String {
                        index: Box::new(string_index),
                        value: Some(value.clone()),
                    })
                }
                Opcode::LoadN => {
                    let args = stack.pop(&mut statements, temp_vars, 2)?;
                    let size;
                    if let Ast::ConstInt { val } = args[1] {
                        size = val;
                    } else {
                        panic!("LoadN called with non-const size.")
                    }

                    stack.push(Ast::LoadN {
                        address: Box::new(args[0].clone()),
                        size: size as u32,
                    })
                }
                Opcode::StoreN => {
                    let args = stack.pop(&mut statements, temp_vars, 2)?;
                    let lhs = args[0].clone();
                    let size;
                    if let Ast::ConstInt { val } = args[1] {
                        size = val;
                    } else {
                        panic!("StoreN called with non-const size.")
                    }
                    let mut stack_items = stack.pop(&mut statements, temp_vars, size as u32)?;
                    if stack_items.len() == 1 {
                        statements.push(Ast::Store {
                            lhs: Box::new(lhs),
                            rhs: Box::new(stack_items[0].clone()),
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
                    let args = stack.pop(&mut statements, temp_vars, 2)?;
                    let lhs = Ast::Dereference {
                        val: Box::new(args[0].clone()),
                    };
                    let rhs = args[1].clone();

                    statements.push(Ast::Store {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    })
                }
                Opcode::Inot => {
                    let arg = stack.pop(&mut statements, temp_vars, 1)?[0].clone();
                    stack.push(Ast::Not { val: Box::new(arg) });
                }
                Opcode::Load => {
                    let arg = stack.pop(&mut statements, temp_vars, 1)?[0].clone();
                    stack.push(Ast::Dereference { val: Box::new(arg) });
                }
                Opcode::Jz { offset } if *offset > 0 => {
                    let condition = stack.pop(&mut statements, temp_vars, 1)?[0].clone();

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

                    statements.push(Ast::If {
                        condition: Box::new(condition),
                        body: Box::new(
                            self.generate_ast(
                                &instructions_in_block[..],
                                function_index,
                                &mut stack.clone(),
                                depth,
                                local_vars,
                                temp_vars,
                            )?
                            .0,
                        ),
                    });
                }
                Opcode::IEqJz { offset } if *offset > 0 => {
                    let args = stack.pop(&mut statements, temp_vars, 2)?;
                    let condition = Box::new(Ast::IntegerEqual {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone()),
                    });

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

                    statements.push(Ast::If {
                        condition,
                        body: Box::new(
                            self.generate_ast(
                                &instructions_in_block[..],
                                function_index,
                                &mut stack.clone(),
                                depth,
                                local_vars,
                                temp_vars,
                            )?
                            .0,
                        ),
                    });
                }
                Opcode::INeJz { offset } if *offset > 0 => {
                    let args = stack.pop(&mut statements, temp_vars, 2)?;
                    let condition = Box::new(Ast::IntegerNotEqual {
                        lhs: Box::new(args[1].clone()),
                        rhs: Box::new(args[0].clone()),
                    });

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

                    statements.push(Ast::If {
                        condition,
                        body: Box::new(
                            self.generate_ast(
                                &instructions_in_block[..],
                                function_index,
                                &mut stack.clone(),
                                depth,
                                local_vars,
                                temp_vars,
                            )?
                            .0,
                        ),
                    });
                }
                _ => {
                    return Err(anyhow!("unsupported opcode: {inst:?}"));
                }
            }

            // let list = Ast::StatementList { list: statements.clone(), stack_size: 0 };
            // println!("INST: {inst:?} func_{function_index}");
            // println!("STACK: {:?}", stack.stack);
            // println!("ITER:\n{list:#?}\n\n");

            block_sizes.push(inst.get_size());
            index += 1;
        }

        Ok((
            Ast::StatementList {
                list: statements,
                stack_size: stack.len(),
            },
            returns,
            *local_vars,
            *temp_vars,
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

        for (instruction_end_index, (i, inst)) in instructions.iter().enumerate().enumerate() {
            if let Opcode::Enter {
                arg_count, index, ..
            } = inst
            {
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
        };

        Ok(ast_gen)
    }
}
