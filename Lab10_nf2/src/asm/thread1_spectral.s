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
	.file	"thread1_spectral.c"
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
	.global	thread1_spectral
	.syntax unified
	.arm
	.type	thread1_spectral, %function
thread1_spectral:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 88
	@ frame_needed = 1, uses_anonymous_args = 0
	push	{fp, lr}
	add	fp, sp, #4
	sub	sp, sp, #88
	mov	r3, #0
	str	r3, [fp, #-8]
	b	.L14
.L19:
	mov	r3, #0
	str	r3, [fp, #-16]
	mov	r3, #0
	str	r3, [fp, #-20]
	ldr	r2, .L38
	ldr	r3, [fp, #-8]
	ldr	r3, [r2, r3, lsl #2]
	str	r3, [fp, #-72]
	mov	r3, #0
	str	r3, [fp, #-12]
	b	.L15
.L18:
	mov	r2, #0
	ldr	r3, [fp, #-12]
	lsl	r3, r3, #1
	add	r3, r2, r3
	ldrh	r3, [r3]	@ movhi
	lsl	r3, r3, #16
	asr	r3, r3, #16
	mov	r0, r3
	bl	abs16
	str	r0, [fp, #-76]
	ldr	r2, [fp, #-12]
	ldr	r3, [fp, #-72]
	asr	r3, r2, r3
	and	r3, r3, #1
	cmp	r3, #0
	beq	.L16
	ldr	r2, [fp, #-20]
	ldr	r3, [fp, #-76]
	add	r3, r2, r3
	str	r3, [fp, #-20]
	b	.L17
.L16:
	ldr	r2, [fp, #-16]
	ldr	r3, [fp, #-76]
	add	r3, r2, r3
	str	r3, [fp, #-16]
.L17:
	ldr	r3, [fp, #-12]
	add	r3, r3, #1
	str	r3, [fp, #-12]
.L15:
	ldr	r3, [fp, #-12]
	cmp	r3, #255
	ble	.L18
	ldr	r2, [fp, #-16]
	ldr	r3, [fp, #-20]
	sub	r3, r2, r3
	mov	r0, r3
	bl	abs16
	mov	r3, r0
	asr	r2, r3, #4
	ldr	r3, [fp, #-8]
	lsl	r3, r3, #2
	sub	r3, r3, #4
	add	r3, r3, fp
	str	r2, [r3, #-88]
	ldr	r3, [fp, #-8]
	add	r3, r3, #1
	str	r3, [fp, #-8]
.L14:
	ldr	r3, [fp, #-8]
	cmp	r3, #3
	ble	.L19
	mov	r3, #0
	str	r3, [fp, #-24]
	mov	r3, #0
	str	r3, [fp, #-28]
	mov	r3, #0
	str	r3, [fp, #-8]
	b	.L20
.L21:
	ldr	r3, [fp, #-8]
	lsl	r3, r3, #2
	sub	r3, r3, #4
	add	r3, r3, fp
	ldr	r3, [r3, #-88]
	ldr	r2, [fp, #-8]
	mul	r3, r2, r3
	ldr	r2, [fp, #-24]
	add	r3, r2, r3
	str	r3, [fp, #-24]
	ldr	r3, [fp, #-8]
	lsl	r3, r3, #2
	sub	r3, r3, #4
	add	r3, r3, fp
	ldr	r3, [r3, #-88]
	ldr	r2, [fp, #-28]
	add	r3, r2, r3
	str	r3, [fp, #-28]
	ldr	r3, [fp, #-8]
	add	r3, r3, #1
	str	r3, [fp, #-8]
.L20:
	ldr	r3, [fp, #-8]
	cmp	r3, #3
	ble	.L21
	ldr	r3, [fp, #-28]
	cmp	r3, #0
	ble	.L22
	ldr	r3, [fp, #-24]
	lsl	r3, r3, #4
	ldr	r1, [fp, #-28]
	mov	r0, r3
	bl	idiv
	mov	r3, r0
	b	.L23
.L22:
	mov	r3, #0
.L23:
	str	r3, [fp, #-52]
	mov	r3, #0
	str	r3, [fp, #-32]
	mov	r3, #0
	str	r3, [fp, #-8]
	b	.L24
.L25:
	ldr	r3, [fp, #-8]
	lsl	r2, r3, #4
	ldr	r3, [fp, #-52]
	sub	r3, r2, r3
	str	r3, [fp, #-68]
	ldr	r3, [fp, #-68]
	mul	r2, r3, r3
	ldr	r3, [fp, #-8]
	lsl	r3, r3, #2
	sub	r3, r3, #4
	add	r3, r3, fp
	ldr	r3, [r3, #-88]
	mul	r3, r2, r3
	asr	r3, r3, #8
	ldr	r2, [fp, #-32]
	add	r3, r2, r3
	str	r3, [fp, #-32]
	ldr	r3, [fp, #-8]
	add	r3, r3, #1
	str	r3, [fp, #-8]
.L24:
	ldr	r3, [fp, #-8]
	cmp	r3, #3
	ble	.L25
	ldr	r3, [fp, #-28]
	cmp	r3, #0
	ble	.L26
	ldr	r1, [fp, #-28]
	ldr	r0, [fp, #-32]
	bl	idiv
	mov	r3, r0
	b	.L27
.L26:
	mov	r3, #0
.L27:
	str	r3, [fp, #-56]
	ldr	r2, [fp, #-28]
	mov	r3, r2
	lsl	r3, r3, #3
	sub	r3, r3, r2
	asr	r3, r3, #3
	str	r3, [fp, #-60]
	mov	r3, #0
	str	r3, [fp, #-36]
	mov	r3, #3
	str	r3, [fp, #-40]
	mov	r3, #0
	str	r3, [fp, #-8]
	b	.L28
.L31:
	ldr	r3, [fp, #-8]
	lsl	r3, r3, #2
	sub	r3, r3, #4
	add	r3, r3, fp
	ldr	r3, [r3, #-88]
	ldr	r2, [fp, #-36]
	add	r3, r2, r3
	str	r3, [fp, #-36]
	ldr	r2, [fp, #-36]
	ldr	r3, [fp, #-60]
	cmp	r2, r3
	blt	.L29
	ldr	r3, [fp, #-8]
	str	r3, [fp, #-40]
	b	.L30
.L29:
	ldr	r3, [fp, #-8]
	add	r3, r3, #1
	str	r3, [fp, #-8]
.L28:
	ldr	r3, [fp, #-8]
	cmp	r3, #3
	ble	.L31
.L30:
	ldr	r3, [fp, #-92]
	str	r3, [fp, #-44]
	ldr	r3, [fp, #-92]
	str	r3, [fp, #-48]
	mov	r3, #1
	str	r3, [fp, #-8]
	b	.L32
.L35:
	ldr	r3, [fp, #-8]
	lsl	r3, r3, #2
	sub	r3, r3, #4
	add	r3, r3, fp
	ldr	r3, [r3, #-88]
	ldr	r2, [fp, #-44]
	cmp	r2, r3
	ble	.L33
	ldr	r3, [fp, #-8]
	lsl	r3, r3, #2
	sub	r3, r3, #4
	add	r3, r3, fp
	ldr	r3, [r3, #-88]
	str	r3, [fp, #-44]
.L33:
	ldr	r3, [fp, #-8]
	lsl	r3, r3, #2
	sub	r3, r3, #4
	add	r3, r3, fp
	ldr	r3, [r3, #-88]
	ldr	r2, [fp, #-48]
	cmp	r2, r3
	bge	.L34
	ldr	r3, [fp, #-8]
	lsl	r3, r3, #2
	sub	r3, r3, #4
	add	r3, r3, fp
	ldr	r3, [r3, #-88]
	str	r3, [fp, #-48]
.L34:
	ldr	r3, [fp, #-8]
	add	r3, r3, #1
	str	r3, [fp, #-8]
.L32:
	ldr	r3, [fp, #-8]
	cmp	r3, #3
	ble	.L35
	ldr	r3, [fp, #-48]
	cmp	r3, #0
	ble	.L36
	ldr	r3, [fp, #-44]
	lsl	r3, r3, #8
	ldr	r1, [fp, #-48]
	mov	r0, r3
	bl	idiv
	mov	r3, r0
	b	.L37
.L36:
	mov	r3, #0
.L37:
	str	r3, [fp, #-64]
	mov	r3, #512
	add	r3, r3, #16
	ldr	r2, [fp, #-52]
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	mov	r3, #512
	add	r3, r3, #18
	ldr	r2, [fp, #-56]
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	ldr	r2, [fp, #-92]
	mov	r3, #512
	add	r3, r3, #20
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	ldr	r2, [fp, #-88]
	mov	r3, #512
	add	r3, r3, #22
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	ldr	r2, [fp, #-84]
	mov	r3, #512
	add	r3, r3, #24
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	ldr	r2, [fp, #-80]
	mov	r3, #512
	add	r3, r3, #26
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	mov	r3, #512
	add	r3, r3, #28
	ldr	r2, [fp, #-64]
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	mov	r3, #512
	add	r3, r3, #30
	ldr	r2, [fp, #-40]
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	nop
	sub	sp, fp, #4
	@ sp needed
	pop	{fp, lr}
	bx	lr
.L39:
	.align	2
.L38:
	.word	hp_shift.0
	.size	thread1_spectral, .-thread1_spectral
	.section	.rodata
	.align	2
	.type	hp_shift.0, %object
	.size	hp_shift.0, 16
hp_shift.0:
	.word	6
	.word	4
	.word	2
	.word	0
	.ident	"GCC: (Arm GNU Toolchain 14.2.Rel1 (Build arm-14.52)) 14.2.1 20241119"
