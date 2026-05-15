---
description: "Explain selected code or attached code files line-by-line in plain, beginner-friendly English — works for any programming language"
argument-hint: "Code file or snippet to explain"
---
Explain the attached code in plain, beginner-friendly English. Treat the reader as someone with minimal programming experience.

## Principles

- **Use the simplest language possible**. If technical terms are unavoidable, explain them the first time they appear.
- **Assume the reader is a beginner**. Do not assume any prior knowledge of the language, framework, or domain.
- **Start with WHAT, then WHY**: First describe what the code does, then explain why it's written that way.

## Output Structure

Strictly follow this structure:

### 1. Overview (2-3 sentences)

Summarize what this code does and which module/class/system it belongs to.

### 2. Block-by-Block Explanation

Split the code into logical blocks (functions, classes, loops, conditionals, etc. — **not** line by line). For each block:

- **Purpose**: What is this block trying to achieve?
- **Line-by-line walkthrough** (include the code snippet with line numbers):
  ```
  // Lines X-Y
  function trainModel(model, data) {
      // Define a training function...
  }
  ```
  - Explain each expression or statement
  - Explain what each variable name means and why it's named that way
- **Input & Output**: What goes in, what comes out?

### 3. Interactions Between Blocks

- Describe how the blocks connect to each other
- Point out the data flow (which function's output feeds into another function's input)
- If there are call relationships, explain the calling order and dependencies

### 4. Key Concepts, Syntax & Library Highlights

**Always** explain the following in extra detail when encountered:

- **Language-specific syntax** (e.g., decorators `@` in Python, generics `<T>` in Java/TypeScript, pointers `*` in Go/C, async/await, closures/lambdas, pattern matching, destructuring, etc.)
- **Key framework/library usage** (e.g., PyTorch `nn.Module`, React `useEffect`, Express.js routes, pandas `groupby`, etc.)
  - What does this library/framework function do?
  - Why is it used here?
  - Provide a short, independent mini-example
- **Design patterns or architectural concepts** (e.g., Singleton, Observer, Strategy, Dependency Injection, MVC, etc.)

Example format:

> **Generics `<T>` in Java/TypeScript**
> - A way to create reusable components that work with any type
> - Here it's used so the same function can handle both `User` and `Admin` objects
> ```typescript
> // Simple example
> function identity<T>(arg: T): T {
>     return arg;
> }
> ```

### 5. Summary (optional)

Highlight 3-5 key takeaways: the design philosophy, clever tricks, or important caveats in this code.
