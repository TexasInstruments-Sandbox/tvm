; c7x_cxm.asm - C7x security mode helper functions
;
; Licensed to the Apache Software Foundation (ASF) under one
; or more contributor license agreements.

        .global tvm_dsp_get_cxm

        .sect ".text:l2_init"
        .clink
;
; uint32_t tvm_dsp_get_cxm(void)
; Returns CXM field (bits 6:4) from TSR
;
tvm_dsp_get_cxm:
        mvc.s1   TSR, a5         ; Read TSR into a5
        andd.l1  a5, 0x7, a4     ; Mask bits 2:0 to get CXM (3 bits)
||      ret.b1                   ; Return with result in a4

