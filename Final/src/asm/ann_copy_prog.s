	.cpu arm7tdmi
	.arch armv4t
	.fpu softvfp
	.eabi_attribute 20, 1
	.eabi_attribute 21, 1
	.eabi_attribute 23, 3
	.eabi_attribute 24, 1
	.eabi_attribute 25, 1
	.eabi_attribute 26, 1
	.eabi_attribute 30, 4
	.eabi_attribute 34, 0
	.eabi_attribute 18, 4
	.file	"ann_copy_prog.c"
	.text
	.section	.text._start,"ax",%progbits
	.align	2
	.global	_start
	.syntax unified
	.arm
	.type	_start, %function
_start:
	@ Function supports interworking.
	@ Naked Function: prologue and epilogue provided by programmer.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	.syntax divided
@ 19 "/Users/jeremycai/Desktop/USC/Courses/Processing/EE533/Repo/EE533-Labs/Final/src/c/ann_copy_prog.c" 1
	mrc p10, 0, r11, c10, c0, 0
cmp r11, #0
bne 2f
mov r1, #16
mcr p10, 0, r1, c0, c0, 0
mov r0, #0
ldr r2, [r0, #0]
ldr r3, [r0, #4]
mcr p10, 0, r2, c1, c0, 0
mcr p10, 0, r3, c2, c0, 0
mov r4, #65
mcr p10, 0, r4, c3, c0, 0
mov r6, r3, lsl #2
add r6, r6, #16
1:
mrc p10, 0, r5, c9, c0, 0
cmp r5, r6
bne 1b
2:
b .
b .
nop

@ 0 "" 2
	.arm
	.syntax unified
	.size	_start, .-_start
	.ident	"GCC: (Arm GNU Toolchain 14.2.Rel1 (Build arm-14.52)) 14.2.1 20241119"
