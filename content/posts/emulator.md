---
title: "Building a Chip-8 emulator"
date: 2021-01-29T16:27:55+01:00
draft: true
tags:
 - Rust
---

## What is Chip-8?

Creating an emulator is an effective way to learn how computers work.
In this post, you are going to learn how to create your own Chip-8 emulator. But wait! What is even Chip-8?

> Chip-8 is a simple, interpreted, programming language which was first used on some do-it-yourself computer systems in the late 1970s and early 1980s.

The simple part means that is a relatively small project to create a complete emulator for it. And the fact that it is interpreted means that we can easily recreate it on modern hardware since it was already interpreted in the 70s. However, at the time a programming language was perhaps not what you would expect today. Indeed, it is unlike Python or JavaScript. In fact, the instructions are to be written in binary format, which means a text editor would not be able to create a valid Chip-8 program. Chip-8 is more of a [Bytecode](https://en.wikipedia.org/wiki/Bytecode) format than a programming language (arguably, the two definitions intersects). The instructions are laid sequentially in a binary file.

To create Chip-8 programs, it is prefered a higher level abstraction than to use directly an hex-editor even though this would be possible. Here are several tool you could use to create a Chip-8 program:

 - [Octo](https://internet-janitor.itch.io/octo) is a complete IDE to create Chip-8 programs

 - [Chipo](https://chipo.ber.gp/) (built following this guide or rather the other way around) provides an assembly like syntax to create Chip-8 programs.

But for now, it is not required to write new programs since we have not started to work on the emulator. Let's get going using the [Rust programming language](https://www.rust-lang.org/). If you don't know Rust, you should be able to follow since we won't be using advanced features like the mighty borrow checker.

## Emulating a computer

A Chip-8 computer, like every computer is composed of a few elementary bricks that are needed to execute the decoded instructions.
The first important block is the memory. It will be used to store the instructions for the programs as well as the sprite data to display images on the screen. Our virtual Chip-8 computer has a memory of 4096 bytes. Then there are the CPU registers. They are a small cell of data capable of storing only 1 bytes directly and they are located directly inside the CPU. To move data in or out of the register, we use special instructions. `ld v0, 0xAB` will set the register 0 to the numeric value `0xAB`. In the program file, the same instruction will by represented by two consecuting bytes `0x60AB`, we will explain later how to interpret this instruction. Some instructions are responsible for doing operations on the registers themselves, like adding two registers together and storing the result in the first register (`add v0, v1` will compute the sum of the values stored in `v0` and `v1` and store the result in `v0`). They are 16 registers plus a few extra with special behaviours. The special registers are `dt` for _delay timer_, it will be used to wait for certain durations in Chip-8 programs. Then there is `st`, the _sound timer_ which will be used almost like the delay timer but for playing sound for a precise amount of time. Next comes the program counter (`pc`) which is used to tell our CPU where we currently are in the execution of our process. As it is always pointing to an address in memory where our instructions is located, the CPU will use it to fetch the next instruction at this location. Usually, after executing an instructions, it will be incremented to allow fetching the next instruction in memory in the next CPU cycle. Since our address is composed of 4096 bytes, the program counter must be able to reference any point in the memory, it must be bigger than 8 bits to hold every address possible, it will then be 16 bits which can go as high as 65536 (2^16).

We can define a struct for our processor with the important fields:

```rust
struct Proc {
    memory: [u8; 4096],
    registers: [u8; 16], // Registers v0 to vF
    
    // Special registers
    delay_timer: u8,
    sound_timer: u8,
    pc: u16,
    
    // Call stack
    stack: Vec<u16>,
    
    // Pixel buffer
    pixels: [bool; 64 * 32]
}
```

And we can create a static method to create a new `Proc` instance:

```rust
impl Proc {
    fn new() -> Proc {
        let proc = Proc {
            memory: [0; 4096],
            rg: [0; 16],
            i: 0,
            delay_rg: 0,
            sound_rg: 0,
            pc: 0x200,
            stack: vec![],
            pixels: [false; 64 * 32],
        };
        proc
    }
}
```

## Decoding instructions

As previously said, the instructions written in the program are in binary. And binary data can be read as [hexadecimal representation](https://en.wikipedia.org/wiki/Hexadecimal) which packs 8 bits (a 0 or a 1) together in a character from 0 to F. In this representation, 0 means, well, 0 and F is 15 in good old decimal.
Since our human eyes are not easily reading hexadecimal values, we will be using the [Cowgod's Chip-8 Technical Reference](http://devernay.free.fr/hacks/chip8/C8TECH10.HTM) which is the goto reference for any Chip-8 emulator implementer. As said, the instructions are a group of 16 bits, meaning it is two bytes and it can be represented by for hexadecimal characters.

First, let's start by reading the stream of bytes from a file and convert it to 16 bits values instead of 8:

```rust 
use std::fs::read;
fn main() {
    let bytes = read("test.c8").unwrap(); // Vec<u8>
    let new_len = bytes.len() / 2;
    let instructions = Vec::with_capacity::<u16>(new_len);
    for i in 0..new_len {
        let instruction = ((bytes[i * 2] as u16) << 8) + // Upper part
            (bytes[i * 2 + 1] as u16); // Lower part
        instructions.push(instruction);
    }
}
```

The `instructions` vector now has 16 bits values representing our program's instructions. We can now start decoding instructions one by one:

```rust
fn main() {
    // Reading code from previously...
    for instruction in instructions {
        let (w, x, y, z) = ((instruction & 0xF000) >> 12, (instruction & 0xF00) >> 8, (instruction & 0xF0) >> 4, instruction & 0xF);
        match (w, x, y, z) {
            (0xF, _, _, 0x1) => {},
        }
    }
}
```

But for our instructions to take actions on our emulator we have a bit more work. In the next section, we will see how to create in software the different parts of a Chip-8 computer.

## Running instructions

## The loop

## Conclusion


