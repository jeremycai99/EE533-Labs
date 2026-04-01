	.cpu arm7tdmi
	.arch armv4t
	.fpu softvfp
	.eabi_attribute 20, 1
	.eabi_attribute 21, 1
	.eabi_attribute 23, 3
	.eabi_attribute 24, 1
	.eabi_attribute 25, 1
	.eabi_attribute 26, 1
	.eabi_attribute 30, 6
	.eabi_attribute 34, 0
	.eabi_attribute 18, 4
	.file	"thread0_time_stats.c"
	.text
	.global	samples
	.section	.rodata
	.align	2
	.type	samples, %object
	.size	samples, 4
samples:
	.space	4
	.global	feat_out
	.align	2
	.type	feat_out, %object
	.size	feat_out, 4
feat_out:
	.word	512
	.text
	.align	2
	.syntax unified
	.arm
	.type	abs16, %function
abs16:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 8
	@ frame_needed = 1, uses_anonymous_args = 0
	@ link register save eliminated.
	str	fp, [sp, #-4]!
	add	fp, sp, #0
	sub	sp, sp, #12
	str	r0, [fp, #-8]
	ldr	r3, [fp, #-8]
	cmp	r3, #0
	rsblt	r3, r3, #0
	mov	r0, r3
	add	sp, fp, #0
	@ sp needed
	ldr	fp, [sp], #4
	bx	lr
	.size	abs16, .-abs16
	.align	2
	.syntax unified
	.arm
	.type	isqrt, %function
isqrt:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 16
	@ frame_needed = 1, uses_anonymous_args = 0
	@ link register save eliminated.
	str	fp, [sp, #-4]!
	add	fp, sp, #0
	sub	sp, sp, #20
	str	r0, [fp, #-16]
	mov	r3, #0
	str	r3, [fp, #-8]
	mov	r3, #1073741824
	str	r3, [fp, #-12]
	b	.L4
.L5:
	ldr	r3, [fp, #-12]
	lsr	r3, r3, #2
	str	r3, [fp, #-12]
.L4:
	ldr	r2, [fp, #-12]
	ldr	r3, [fp, #-16]
	cmp	r2, r3
	bhi	.L5
	b	.L6
.L9:
	ldr	r2, [fp, #-8]
	ldr	r3, [fp, #-12]
	add	r3, r2, r3
	ldr	r2, [fp, #-16]
	cmp	r2, r3
	bcc	.L7
	ldr	r2, [fp, #-8]
	ldr	r3, [fp, #-12]
	add	r3, r2, r3
	ldr	r2, [fp, #-16]
	sub	r3, r2, r3
	str	r3, [fp, #-16]
	ldr	r3, [fp, #-8]
	lsr	r3, r3, #1
	ldr	r2, [fp, #-12]
	add	r3, r2, r3
	str	r3, [fp, #-8]
	b	.L8
.L7:
	ldr	r3, [fp, #-8]
	lsr	r3, r3, #1
	str	r3, [fp, #-8]
.L8:
	ldr	r3, [fp, #-12]
	lsr	r3, r3, #2
	str	r3, [fp, #-12]
.L6:
	ldr	r3, [fp, #-12]
	cmp	r3, #0
	bne	.L9
	ldr	r3, [fp, #-8]
	mov	r0, r3
	add	sp, fp, #0
	@ sp needed
	ldr	fp, [sp], #4
	bx	lr
	.size	isqrt, .-isqrt
	.align	2
	.syntax unified
	.arm
	.type	idiv, %function
idiv:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 24
	@ frame_needed = 1, uses_anonymous_args = 0
	@ link register save eliminated.
	str	fp, [sp, #-4]!
	add	fp, sp, #0
	sub	sp, sp, #28
	str	r0, [fp, #-24]
	str	r1, [fp, #-28]
	ldr	r3, [fp, #-28]
	cmp	r3, #0
	bne	.L12
	mov	r3, #0
	b	.L13
.L12:
	mov	r3, #0
	str	r3, [fp, #-8]
	mov	r3, #0
	str	r3, [fp, #-12]
	ldr	r3, [fp, #-24]
	cmp	r3, #0
	bge	.L14
	ldr	r3, [fp, #-24]
	rsb	r3, r3, #0
	str	r3, [fp, #-24]
	ldr	r3, [fp, #-8]
	eor	r3, r3, #1
	str	r3, [fp, #-8]
.L14:
	ldr	r3, [fp, #-28]
	cmp	r3, #0
	bge	.L15
	ldr	r3, [fp, #-28]
	rsb	r3, r3, #0
	str	r3, [fp, #-28]
	ldr	r3, [fp, #-8]
	eor	r3, r3, #1
	str	r3, [fp, #-8]
.L15:
	mov	r3, #15
	str	r3, [fp, #-16]
	b	.L16
.L18:
	ldr	r2, [fp, #-28]
	ldr	r3, [fp, #-16]
	lsl	r3, r2, r3
	ldr	r2, [fp, #-24]
	cmp	r2, r3
	blt	.L17
	ldr	r2, [fp, #-28]
	ldr	r3, [fp, #-16]
	lsl	r3, r2, r3
	ldr	r2, [fp, #-24]
	sub	r3, r2, r3
	str	r3, [fp, #-24]
	mov	r2, #1
	ldr	r3, [fp, #-16]
	lsl	r3, r2, r3
	ldr	r2, [fp, #-12]
	add	r3, r2, r3
	str	r3, [fp, #-12]
.L17:
	ldr	r3, [fp, #-16]
	sub	r3, r3, #1
	str	r3, [fp, #-16]
.L16:
	ldr	r3, [fp, #-16]
	cmp	r3, #0
	bge	.L18
	ldr	r3, [fp, #-8]
	cmp	r3, #0
	beq	.L19
	ldr	r3, [fp, #-12]
	rsb	r3, r3, #0
	b	.L13
.L19:
	ldr	r3, [fp, #-12]
.L13:
	mov	r0, r3
	add	sp, fp, #0
	@ sp needed
	ldr	fp, [sp], #4
	bx	lr
	.size	idiv, .-idiv
	.align	2
	.global	thread0_time_stats
	.syntax unified
	.arm
	.type	thread0_time_stats, %function
thread0_time_stats:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 80
	@ frame_needed = 1, uses_anonymous_args = 0
	push	{fp, lr}
	add	fp, sp, #4
	sub	sp, sp, #80
	mov	r3, #0
	str	r3, [fp, #-8]
	mov	r3, #0
	str	r3, [fp, #-12]
	mov	r3, #0
	str	r3, [fp, #-16]
	mov	r3, #0
	str	r3, [fp, #-20]
	ldr	r3, .L30
	str	r3, [fp, #-24]
	ldr	r3, .L30+4
	str	r3, [fp, #-28]
	mov	r3, #0
	str	r3, [fp, #-32]
	mov	r3, #0
	str	r3, [fp, #-36]
	mov	r3, #0
	str	r3, [fp, #-40]
	mov	r3, #0
	str	r3, [fp, #-44]
	b	.L22
.L27:
	mov	r2, #0
	ldr	r3, [fp, #-44]
	lsl	r3, r3, #1
	add	r3, r2, r3
	ldrh	r3, [r3]	@ movhi
	lsl	r3, r3, #16
	asr	r3, r3, #16
	str	r3, [fp, #-72]
	ldr	r0, [fp, #-72]
	bl	abs16
	str	r0, [fp, #-76]
	ldr	r2, [fp, #-8]
	ldr	r3, [fp, #-72]
	add	r3, r2, r3
	str	r3, [fp, #-8]
	ldr	r2, [fp, #-12]
	ldr	r3, [fp, #-76]
	add	r3, r2, r3
	str	r3, [fp, #-12]
	ldr	r3, [fp, #-72]
	mov	r2, r3
	mul	r2, r3, r2
	mov	r3, r2
	asr	r3, r3, #8
	ldr	r2, [fp, #-16]
	add	r3, r2, r3
	str	r3, [fp, #-16]
	ldr	r3, [fp, #-72]
	mov	r2, r3
	mul	r2, r3, r2
	mov	r3, r2
	asr	r3, r3, #16
	ldr	r2, [fp, #-76]
	mul	r3, r2, r3
	ldr	r2, [fp, #-20]
	add	r3, r2, r3
	str	r3, [fp, #-20]
	ldr	r2, [fp, #-72]
	ldr	r3, [fp, #-24]
	cmp	r2, r3
	ble	.L23
	ldr	r3, [fp, #-72]
	str	r3, [fp, #-24]
.L23:
	ldr	r2, [fp, #-72]
	ldr	r3, [fp, #-28]
	cmp	r2, r3
	bge	.L24
	ldr	r3, [fp, #-72]
	str	r3, [fp, #-28]
.L24:
	ldr	r2, [fp, #-76]
	ldr	r3, [fp, #-32]
	cmp	r2, r3
	ble	.L25
	ldr	r3, [fp, #-76]
	str	r3, [fp, #-32]
.L25:
	ldr	r3, [fp, #-72]
	mvn	r3, r3
	lsr	r3, r3, #31
	and	r3, r3, #255
	str	r3, [fp, #-80]
	ldr	r3, [fp, #-44]
	cmp	r3, #0
	ble	.L26
	ldr	r2, [fp, #-80]
	ldr	r3, [fp, #-40]
	cmp	r2, r3
	beq	.L26
	ldr	r3, [fp, #-36]
	add	r3, r3, #1
	str	r3, [fp, #-36]
.L26:
	ldr	r3, [fp, #-80]
	str	r3, [fp, #-40]
	ldr	r3, [fp, #-44]
	add	r3, r3, #1
	str	r3, [fp, #-44]
.L22:
	ldr	r3, [fp, #-44]
	cmp	r3, #255
	ble	.L27
	ldr	r3, [fp, #-16]
	mov	r0, r3
	bl	isqrt
	str	r0, [fp, #-48]
	ldr	r3, [fp, #-12]
	asr	r3, r3, #8
	str	r3, [fp, #-52]
	ldr	r2, [fp, #-24]
	ldr	r3, [fp, #-28]
	sub	r3, r2, r3
	str	r3, [fp, #-56]
	ldr	r3, [fp, #-52]
	cmp	r3, #0
	ble	.L28
	ldr	r1, [fp, #-52]
	ldr	r0, [fp, #-32]
	bl	idiv
	mov	r3, r0
	b	.L29
.L28:
	mov	r3, #0
.L29:
	str	r3, [fp, #-60]
	ldr	r3, [fp, #-8]
	asr	r3, r3, #8
	str	r3, [fp, #-64]
	ldr	r3, [fp, #-64]
	mov	r2, r3
	mul	r2, r3, r2
	mov	r3, r2
	ldr	r2, [fp, #-16]
	sub	r3, r2, r3
	str	r3, [fp, #-68]
	mov	r2, #512
	ldr	r3, [fp, #-48]
	lsl	r3, r3, #16
	asr	r3, r3, #16
	strh	r3, [r2]	@ movhi
	mov	r3, #512
	add	r3, r3, #2
	ldr	r2, [fp, #-32]
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	mov	r3, #512
	add	r3, r3, #4
	ldr	r2, [fp, #-56]
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	mov	r3, #512
	add	r3, r3, #6
	ldr	r2, [fp, #-60]
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	mov	r3, #512
	add	r3, r3, #8
	ldr	r2, [fp, #-52]
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	mov	r3, #512
	add	r3, r3, #10
	ldr	r2, [fp, #-36]
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	ldr	r3, [fp, #-68]
	asr	r2, r3, #4
	mov	r3, #512
	add	r3, r3, #12
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	ldr	r3, [fp, #-20]
	asr	r2, r3, #8
	mov	r3, #512
	add	r3, r3, #14
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	nop
	sub	sp, fp, #4
	@ sp needed
	pop	{fp, lr}
	bx	lr
.L31:
	.align	2
.L30:
	.word	-32768
	.word	32767
	.size	thread0_time_stats, .-thread0_time_stats
	.ident	"GCC: (Arm GNU Toolchain 14.2.Rel1 (Build arm-14.52)) 14.2.1 20241119"
