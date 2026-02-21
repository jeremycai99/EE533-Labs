#!/bin/bash

# 1. Define Toolchain Path
TOOLCHAIN="/Applications/ArmGNUToolchain/14.2.rel1/arm-none-eabi/bin"
CC="$TOOLCHAIN/arm-none-eabi-gcc"
AS="$TOOLCHAIN/arm-none-eabi-as"
LD="$TOOLCHAIN/arm-none-eabi-ld"
OBJCOPY="$TOOLCHAIN/arm-none-eabi-objcopy"
OBJDUMP="$TOOLCHAIN/arm-none-eabi-objdump"

# 2. Check Input
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_c_file>"
    exit 1
fi

SOURCE_FILE="$1"

if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: File '$SOURCE_FILE' not found."
    exit 1
fi

BASENAME=$(basename "$SOURCE_FILE" .c)
DIRPATH=$(dirname "$SOURCE_FILE")

# Compiler Flags
CFLAGS="-mcpu=arm7tdmi -march=armv4t -marm -nostdlib -nostartfiles -ffreestanding -O0"

echo "Processing $SOURCE_FILE..."

# ---------------------------------------------------------
# Step 0: Create minimal linker script (code starts at 0x0)
# ---------------------------------------------------------
LINKER_SCRIPT="${BASENAME}.ld"
cat > "$LINKER_SCRIPT" << 'EOF'
ENTRY(_start)
SECTIONS
{
    . = 0x00000000;
    .text : {
        *(.text._start)
        *(.text*)
    }
    .rodata : { *(.rodata*) }
    .data : { *(.data*) }
    .bss : { *(.bss*) }
}
EOF

# ---------------------------------------------------------
# Step 1: Generate Assembly (.s)
# ---------------------------------------------------------
echo "Generating Assembly (.s)..."
$CC $CFLAGS -S "$SOURCE_FILE" -o "${BASENAME}.s"

# ---------------------------------------------------------
# Step 2: Compile to Object File (.o)
# ---------------------------------------------------------
echo "Compiling to object..."
$CC $CFLAGS -c "$SOURCE_FILE" -o "${BASENAME}.o"
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

# ---------------------------------------------------------
# Step 3: Link with linker script
# ---------------------------------------------------------
echo "Linking..."
$LD -T "$LINKER_SCRIPT" "${BASENAME}.o" -o "${BASENAME}.elf"
if [ $? -ne 0 ]; then
    echo "Linking failed."
    exit 1
fi

# ---------------------------------------------------------
# Step 4: Generate disassembly (human-readable verification)
# ---------------------------------------------------------
echo "Generating disassembly..."
$OBJDUMP -d "${BASENAME}.elf" > "${BASENAME}_disasm.txt"

# ---------------------------------------------------------
# Step 5: Extract raw binary
# ---------------------------------------------------------
echo "Extracting raw binary..."
$OBJCOPY -O binary "${BASENAME}.elf" "${BASENAME}.bin"

# ---------------------------------------------------------
# Step 6: Convert to Verilog-compatible hex formats
# ---------------------------------------------------------

# A. One 32-bit word per line (for $readmemh with word-addressed memory)
echo "Generating word-per-line hex (for \$readmemh)..."
xxd -e -c 4 "${BASENAME}.bin" | awk '{print $2}' > "${BASENAME}_imem.txt"

# # B. ELF readable dump (for debugging)
# echo "Generating ELF hex dump..."
# xxd "${BASENAME}.elf" > "${BASENAME}_elf.txt"

# ---------------------------------------------------------
# Step 7: Show code size info
# ---------------------------------------------------------
BYTE_COUNT=$(wc -c < "${BASENAME}.bin" | tr -d ' ')
WORD_COUNT=$(wc -l < "${BASENAME}_imem.txt" | tr -d ' ')
echo ""
echo "================================================"
echo "Success! Code size: ${BYTE_COUNT} bytes (${WORD_COUNT} words)"
echo "================================================"
echo "Generated files:"
echo "  ${BASENAME}.s             - Assembly source"
echo "  ${BASENAME}_disasm.txt    - Disassembly (verify instructions)"
echo "  ${BASENAME}_imem.txt       - Verilog \$readmemh format (1 word/line)"
echo "================================================"

# ---------------------------------------------------------
# Cleanup intermediate files
# ---------------------------------------------------------
rm -f "${BASENAME}.o" "${BASENAME}.elf" "$LINKER_SCRIPT"
# Optionally keep .bin:
rm -f "${BASENAME}.bin"