# ysc-utils
GTA V .ysc* utility suite.

The following binaries are provided by `ysc-utils`:

## basic_disassemble
Disassembles a ysc script, showing instructions with function & string annotations.

Usage: `basic_disassemble %ysc_script% (optional: function number/index)`

Example output:
```
ENTER            1 3 0  /* func_5905 */
    LOCAL_U8_LOAD    0 
    IOFFSET_S16_LOAD 770 
    SWITCH           1 
    J                14 
    PUSH_CONST_0      
    PUSH_CONST_0      
    CALL             1840583 
    LOCAL_U8_LOAD    0 
    IOFFSET_S16_STORE 786 
    J                9 
    PUSH_CONST_0      
    LOCAL_U8_LOAD    0 
    IOFFSET_S16_STORE 786 
    J                0 
    LEAVE            1 0
```

## global_updater
Updates a global (and optional offsets) from a ysc script to a different version of the script.

Usage: `global_updater %old_ysc_script% %new_ysc_script% %tokens%`

Example:
```
> global_updater freemode_old.ysc.full freemode.ysc.full Global_2652258.f_2649
Global_2652364.f_2649
```

## gui
A way to access `global_updater` via a GUI.

## native_table
Dumps the native table for a ysc script.

Usage: `native_table %ysc_script%`

## script_info
Dumps information about a ysc script.

Usage: `script_info %ysc_script%`
Example output:
```
> script_info freemode.ysc.full
Script name: freemode
String table size: 22029
Code size (bytes): 7019785
Native table size: 2742
No. code tables: 429
No. string tables: 22
```

## strings
Dumps the strings for a ysc script.

Usage: `strings %ysc_script%`

# Building
1. Install cargo.
2. `cargo build -r`
3. `cd target/release/`
4. All binaries are in your working directory.

## Future (ambitious) plans 
1. YSC assembler (ScriptVM instructions -> ScriptVM bytecode)
2. YSC decompiler (SriptVM bytecode -> ScriptVM instructions -> AST -> C)
3. YSC compiler (C -> ScriptVM instructions -> ScriptVM bytecode)
4. YSC function JIT compiler (during runtime, JIT compile functions when possible, to improve performance) (ScriptVM bytecode -> (disassemble) ScriptVM instructions -> (decompile) AST -> (compile) machine code)
5. GTA V runtime ScriptVM function hooker & patcher

I plan to avoid patching ScriptVM bytecode, but instead hook and re-implement the VM.
