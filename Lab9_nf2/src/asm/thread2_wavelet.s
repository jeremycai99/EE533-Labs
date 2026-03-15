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
	.file	"thread2_wavelet.c"
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
	.bss
	.align	2
wbuf:
	.space	512
	.size	wbuf, 512
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
	.global	thread2_wavelet
	.syntax unified
	.arm
	.type	thread2_wavelet, %function
thread2_wavelet:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 48
	@ frame_needed = 1, uses_anonymous_args = 0
	push	{fp, lr}
	add	fp, sp, #4
	sub	sp, sp, #48
	mov	r3, #256
	str	r3, [fp, #-16]
	mov	r3, #0
	str	r3, [fp, #-12]
	b	.L4
.L5:
	mov	r2, #0
	ldr	r3, [fp, #-12]
	lsl	r3, r3, #1
	add	r3, r2, r3
	ldrh	r3, [r3]	@ movhi
	lsl	r3, r3, #16
	asr	r2, r3, #16
	ldr	r1, .L11
	ldr	r3, [fp, #-12]
	lsl	r3, r3, #1
	add	r3, r1, r3
	strh	r2, [r3]	@ movhi
	ldr	r3, [fp, #-12]
	add	r3, r3, #1
	str	r3, [fp, #-12]
.L4:
	ldr	r3, [fp, #-12]
	cmp	r3, #255
	ble	.L5
	mov	r3, #0
	str	r3, [fp, #-8]
	b	.L6
.L10:
	ldr	r3, [fp, #-16]
	asr	r3, r3, #1
	str	r3, [fp, #-28]
	mov	r3, #0
	str	r3, [fp, #-20]
	mov	r3, #0
	str	r3, [fp, #-24]
	mov	r3, #0
	str	r3, [fp, #-12]
	b	.L7
.L9:
	ldr	r3, [fp, #-12]
	lsl	r3, r3, #1
	ldr	r2, .L11
	lsl	r3, r3, #1
	add	r3, r2, r3
	ldrsh	r3, [r3]
	str	r3, [fp, #-32]
	ldr	r3, [fp, #-12]
	lsl	r3, r3, #1
	add	r3, r3, #1
	ldr	r2, .L11
	lsl	r3, r3, #1
	add	r3, r2, r3
	ldrsh	r3, [r3]
	str	r3, [fp, #-36]
	ldr	r2, [fp, #-32]
	ldr	r3, [fp, #-36]
	add	r3, r2, r3
	asr	r3, r3, #1
	str	r3, [fp, #-40]
	ldr	r2, [fp, #-32]
	ldr	r3, [fp, #-36]
	sub	r3, r2, r3
	asr	r3, r3, #1
	str	r3, [fp, #-44]
	ldr	r0, [fp, #-44]
	bl	abs16
	str	r0, [fp, #-48]
	ldr	r2, [fp, #-20]
	ldr	r3, [fp, #-48]
	add	r3, r2, r3
	str	r3, [fp, #-20]
	ldr	r2, [fp, #-48]
	ldr	r3, [fp, #-24]
	cmp	r2, r3
	ble	.L8
	ldr	r3, [fp, #-48]
	str	r3, [fp, #-24]
.L8:
	ldr	r3, [fp, #-40]
	lsl	r3, r3, #16
	asr	r2, r3, #16
	ldr	r1, .L11
	ldr	r3, [fp, #-12]
	lsl	r3, r3, #1
	add	r3, r1, r3
	strh	r2, [r3]	@ movhi
	ldr	r3, [fp, #-12]
	add	r3, r3, #1
	str	r3, [fp, #-12]
.L7:
	ldr	r2, [fp, #-12]
	ldr	r3, [fp, #-28]
	cmp	r2, r3
	blt	.L9
	ldr	r3, [fp, #-20]
	asr	r2, r3, #4
	mov	r1, #512
	ldr	r3, [fp, #-8]
	add	r3, r3, #16
	lsl	r3, r3, #1
	add	r3, r1, r3
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	mov	r2, #512
	ldr	r3, [fp, #-8]
	add	r3, r3, #20
	lsl	r3, r3, #1
	add	r3, r2, r3
	ldr	r2, [fp, #-24]
	lsl	r2, r2, #16
	asr	r2, r2, #16
	strh	r2, [r3]	@ movhi
	ldr	r3, [fp, #-28]
	str	r3, [fp, #-16]
	ldr	r3, [fp, #-8]
	add	r3, r3, #1
	str	r3, [fp, #-8]
.L6:
	ldr	r3, [fp, #-8]
	cmp	r3, #3
	ble	.L10
	nop
	nop
	sub	sp, fp, #4
	@ sp needed
	pop	{fp, lr}
	bx	lr
.L12:
	.align	2
.L11:
	.word	wbuf
	.size	thread2_wavelet, .-thread2_wavelet
	.ident	"GCC: (Arm GNU Toolchain 14.2.Rel1 (Build arm-14.52)) 14.2.1 20241119"
