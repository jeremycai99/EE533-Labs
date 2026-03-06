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
	.file	"thread3_envelope.c"
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
	bne	.L4
	mov	r3, #0
	b	.L5
.L4:
	mov	r3, #0
	str	r3, [fp, #-8]
	mov	r3, #0
	str	r3, [fp, #-12]
	ldr	r3, [fp, #-24]
	cmp	r3, #0
	bge	.L6
	ldr	r3, [fp, #-24]
	rsb	r3, r3, #0
	str	r3, [fp, #-24]
	ldr	r3, [fp, #-8]
	eor	r3, r3, #1
	str	r3, [fp, #-8]
.L6:
	ldr	r3, [fp, #-28]
	cmp	r3, #0
	bge	.L7
	ldr	r3, [fp, #-28]
	rsb	r3, r3, #0
	str	r3, [fp, #-28]
	ldr	r3, [fp, #-8]
	eor	r3, r3, #1
	str	r3, [fp, #-8]
.L7:
	mov	r3, #15
	str	r3, [fp, #-16]
	b	.L8
.L10:
	ldr	r2, [fp, #-28]
	ldr	r3, [fp, #-16]
	lsl	r3, r2, r3
	ldr	r2, [fp, #-24]
	cmp	r2, r3
	blt	.L9
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
.L9:
	ldr	r3, [fp, #-16]
	sub	r3, r3, #1
	str	r3, [fp, #-16]
.L8:
	ldr	r3, [fp, #-16]
	cmp	r3, #0
	bge	.L10
	ldr	r3, [fp, #-8]
	cmp	r3, #0
	beq	.L11
	ldr	r3, [fp, #-12]
	rsb	r3, r3, #0
	b	.L5
.L11:
	ldr	r3, [fp, #-12]
.L5:
	mov	r0, r3
	add	sp, fp, #0
	@ sp needed
	ldr	fp, [sp], #4
	bx	lr
	.size	idiv, .-idiv
	.align	2
	.global	thread3_envelope
	.syntax unified
	.arm
	.type	thread3_envelope, %function
thread3_envelope:
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
	mov	r3, #0
	str	r3, [fp, #-24]
	mov	r3, #0
	str	r3, [fp, #-28]
	sub	r3, fp, #84
	mov	r2, #0
	str	r2, [r3]
	str	r2, [r3, #4]
	str	r2, [r3, #8]
	str	r2, [r3, #12]
	mov	r3, #0
	str	r3, [fp, #-32]
	b	.L14
.L19:
	mov	r2, #0
	ldr	r3, [fp, #-32]
	lsl	r3, r3, #1
	add	r3, r2, r3
	ldrh	r3, [r3]	@ movhi
	lsl	r3, r3, #16
	asr	r3, r3, #16
	str	r3, [fp, #-48]
	ldr	r0, [fp, #-48]
	bl	abs16
	str	r0, [fp, #-52]
	ldr	r3, [fp, #-32]
	cmp	r3, #1
	ble	.L15
	mov	r2, #0
	ldr	r3, [fp, #-32]
	sub	r3, r3, #-2147483646
	lsl	r3, r3, #1
	add	r3, r2, r3
	ldrh	r3, [r3]	@ movhi
	lsl	r3, r3, #16
	asr	r3, r3, #16
	mov	r2, r3
	ldr	r3, [fp, #-48]
	sub	r3, r3, r2
	mov	r0, r3
	bl	abs16
	mov	r3, r0
	asr	r3, r3, #1
	b	.L16
.L15:
	mov	r3, #0
.L16:
	str	r3, [fp, #-56]
	ldr	r2, [fp, #-52]
	ldr	r3, [fp, #-56]
	add	r3, r2, r3
	str	r3, [fp, #-60]
	ldr	r2, [fp, #-8]
	ldr	r3, [fp, #-60]
	add	r3, r2, r3
	str	r3, [fp, #-8]
	ldr	r3, [fp, #-60]
	mov	r2, r3
	mul	r2, r3, r2
	mov	r3, r2
	asr	r3, r3, #8
	ldr	r2, [fp, #-12]
	add	r3, r2, r3
	str	r3, [fp, #-12]
	ldr	r2, [fp, #-60]
	ldr	r3, [fp, #-16]
	cmp	r2, r3
	ble	.L17
	ldr	r3, [fp, #-60]
	str	r3, [fp, #-16]
.L17:
	ldr	r3, [fp, #-32]
	cmp	r3, #1
	ble	.L18
	ldr	r2, [fp, #-60]
	ldr	r3, [fp, #-20]
	sub	r3, r2, r3
	str	r3, [fp, #-64]
	ldr	r2, [fp, #-20]
	ldr	r3, [fp, #-24]
	sub	r3, r2, r3
	str	r3, [fp, #-68]
	ldr	r2, [fp, #-64]
	ldr	r3, [fp, #-68]
	eor	r3, r3, r2
	cmp	r3, #0
	bge	.L18
	ldr	r3, [fp, #-28]
	add	r3, r3, #1
	str	r3, [fp, #-28]
.L18:
	ldr	r3, [fp, #-32]
	asr	r3, r3, #6
	lsl	r3, r3, #2
	sub	r3, r3, #4
	add	r3, r3, fp
	ldr	r1, [r3, #-80]
	ldr	r3, [fp, #-32]
	asr	r3, r3, #6
	ldr	r2, [fp, #-60]
	add	r2, r1, r2
	lsl	r3, r3, #2
	sub	r3, r3, #4
	add	r3, r3, fp
	str	r2, [r3, #-80]
	ldr	r3, [fp, #-20]
	str	r3, [fp, #-24]
	ldr	r3, [fp, #-60]
	str	r3, [fp, #-20]
	ldr	r3, [fp, #-32]
	add	r3, r3, #1
	str	r3, [fp, #-32]
.L14:
	ldr	r3, [fp, #-32]
	cmp	r3, #255
	ble	.L19
	ldr	r3, [fp, #-8]
	asr	r3, r3, #8
	str	r3, [fp, #-36]
	ldr	r3, [fp, #-36]
	cmp	r3, #0
	ble	.L20
	ldr	r1, [fp, #-36]
	ldr	r0, [fp, #-16]
	bl	idiv
	mov	r3, r0
	b	.L21
.L20:
	mov	r3, #0
.L21:
	str	r3, [fp, #-40]
	ldr	r3, [fp, #-36]
	mov	r2, r3
	mul	r2, r3, r2
	mov	r3, r2
	ldr	r2, [fp, #-12]
	sub	r3, r2, r3
	str	r3, [fp, #-44]
	mov	r3, #512
	add	r3, r3, #48
	ldr	r2, [fp, #-36]
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	mov	r3, #512
	add	r3, r3, #50
	ldr	r2, [fp, #-16]
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	mov	r3, #512
	add	r3, r3, #52
	ldr	r2, [fp, #-40]
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	ldr	r3, [fp, #-44]
	asr	r2, r3, #4
	mov	r3, #512
	add	r3, r3, #54
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	ldr	r3, [fp, #-84]
	asr	r2, r3, #6
	mov	r3, #512
	add	r3, r3, #56
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	ldr	r3, [fp, #-80]
	asr	r2, r3, #6
	mov	r3, #512
	add	r3, r3, #58
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	ldr	r3, [fp, #-76]
	asr	r2, r3, #6
	mov	r3, #512
	add	r3, r3, #60
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	ldr	r3, [fp, #-72]
	asr	r2, r3, #6
	mov	r3, #512
	add	r3, r3, #62
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	nop
	sub	sp, fp, #4
	@ sp needed
	pop	{fp, lr}
	bx	lr
	.size	thread3_envelope, .-thread3_envelope
	.ident	"GCC: (Arm GNU Toolchain 14.2.Rel1 (Build arm-14.52)) 14.2.1 20241119"
