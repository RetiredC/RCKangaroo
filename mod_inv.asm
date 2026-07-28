//(c) 2026, Retired Coder(RC)

FUNCTION _fast_find_first_bit()
{ //Ro_cnt=1, Rt_cnt=2, Rval, Rone, Rbc
////start _fast_find_first_bit
    [B------:R-:W-:-:S04]    SHF.L.U32 Rt0, Rone, Rbc, RZ
    [B------:R-:W-:-:S04]    LOP3.LUT Rt0, Rt0, Rval, RZ, 0xfc, !PT
    [B------:R-:W-:-:S04]    IADD3 Rt1, -Rt0, RZ, RZ //z=Rt0
    [B------:R-:W-:-:S04]    LOP3.LUT Rt0, Rt0, Rt1, RZ, 0xc0, !PT //(z & -z)
    [B------:R-:W-:-:S04]    I2FP.F32.S32 Rt0, Rt0
    [B------:R-:W-:-:S04]    LEA.HI Ro0, Rt0, 0xffffff81, RZ, 9 //(int >> 23) - 127
////end _fast_find_first_bit
}

FUNCTION _mul_256_by_i32(val=Rt8)
{ //Ro_cnt=9, Rt_cnt=6, Pt=[0..2]  Rval, Ri
////start _mul_256_by_i32
    [B------:R-:W-:-:S04] @Pt3 IABS val, Ri
    [B------:R-:W-:-:S01]    ISETP.LT.AND Pt1, PT, Ri, RZ, Pt3
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 Ro0, Rval, val.reuse, RZ
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 Rt0, Rval1, val.reuse, RZ
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 Ro2, Rval2, val.reuse, RZ
    [B------:R-:W-:-:S01] @Pt3 IADD3 Ro1, Pt2, Ro1, Rt0, RZ
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 Rt2, Rval3, val.reuse, RZ
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro2, Pt2, Ro2, Rt1, RZ, Pt2, !PT
    [B------:R-:W-:-:S01] @Pt1 IADD3.X Ro0, Pt0, PT, RZ, ~Ro0, RZ, PT, !PT
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 Ro4, Rval4, val.reuse, RZ
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro3, Pt2, Ro3, Rt2, RZ, Pt2, !PT
    [B------:R-:W-:-:S01] @Pt1 IADD3.X Ro1, Pt0, PT, RZ, ~Ro1, RZ, Pt0, !PT
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 Rt4, Rval5, val.reuse, RZ
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro4, Pt2, Ro4, Rt3, RZ, Pt2, !PT
    [B------:R-:W-:-:S01] @Pt1 IADD3.X Ro2, Pt0, PT, RZ, ~Ro2, RZ, Pt0, !PT
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 Ro6, Rval6, val.reuse, RZ
    [B------:R-:W-:-:S01] @Pt1 IADD3.X Ro3, Pt0, PT, RZ, ~Ro3, RZ, Pt0, !PT
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro5, Pt2, Ro5, Rt4, RZ, Pt2, !PT
    [B------:R-:W-:-:S01] @Pt1 IADD3.X Ro4, Pt0, PT, RZ, ~Ro4, RZ, Pt0, !PT
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 Rt0, Rval7, val, RZ
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro6, Pt2, Ro6, Rt5, RZ, Pt2, !PT
    [B------:R-:W-:-:S01] @Pt1 IADD3.X Ro5, Pt0, PT, RZ, ~Ro5, RZ, Pt0, !PT
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro7, Pt2, Ro7, Rt0, RZ, Pt2, !PT
    [B------:R-:W-:-:S04] @Pt1 IADD3.X Ro6, Pt0, PT, RZ, ~Ro6, RZ, Pt0, !PT
    [B------:R-:W-:-:S01] @Pt3 IMAD.X.U32 Ro8, Rval8, val, Rt1, Pt2
    [B------:R-:W-:-:S04] @Pt1 IADD3.X Ro7, Pt0, PT, RZ, ~Ro7, RZ, Pt0, !PT
    [B------:R-:W-:-:S01] @Pt1 IADD3.X Ro8, PT, PT, RZ, ~Ro8, RZ, Pt0, !PT
////end _mul_256_by_i32
}

FUNCTION _mul_256_by_u31()
{ //Ro_cnt=9, Rt_cnt=6, Pt=[0]  Rval, Ri
////start _mul_256_by_u31
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 Ro0, Rval, Ri.reuse, RZ
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 Rt0, Rval1, Ri.reuse, RZ
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 Ro2, Rval2, Ri, RZ
    [B------:R-:W-:-:S01] @Pt3 IADD3 Ro1, Pt0, Ro1, Rt0, RZ
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 Rt2, Rval3, Ri.reuse, RZ
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro2, Pt0, Ro2, Rt1, RZ, Pt0, !PT
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 Ro4, Rval4, Ri.reuse, RZ
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro3, Pt0, Ro3, Rt2, RZ, Pt0, !PT
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 Rt4, Rval5, Ri.reuse, RZ
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro4, Pt0, Ro4, Rt3, RZ, Pt0, !PT
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 Ro6, Rval6, Ri.reuse, RZ
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro5, Pt0, Ro5, Rt4, RZ, Pt0, !PT
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 Rt0, Rval7, Ri.reuse, RZ
    [B------:R-:W-:-:S04] @Pt3 IADD3.X Ro6, Pt0, Ro6, Rt5, RZ, Pt0, !PT
    [B------:R-:W-:Y:S05] @Pt3 IADD3.X Ro7, Pt0, Ro7, Rt0, RZ, Pt0, !PT
    [B------:R-:W-:Y:S01] @Pt3 IMAD.X.U32 Ro8, Rval8, Ri, Rt1, Pt0
////end _mul_256_by_u31
}

FUNCTION _mul_256_by_u31_add_shift(tmp=Rt6)
{ //Ro_cnt=9, Rt_cnt=15, Pt=[0,1]  Rval, Ri, Radd
////start _mul_256_by_u31_add_shift
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 tmp0, Rval, Ri.reuse, RZ
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 Rt0, Rval1, Ri.reuse, RZ
    [B------:R-:W-:-:S01] @Pt3 IADD3 tmp0, Pt0, PT, tmp0, RZ, Radd0, !PT, !PT
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 tmp2, Rval2, Ri, RZ
    [B------:R-:W-:-:S01] @Pt3 IADD3 tmp1, Pt0, Pt1, tmp1, Rt0, Radd1, Pt0, !PT
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 Rt2, Rval3, Ri.reuse, RZ
    [B------:R-:W-:-:S01] @Pt3 IADD3.X tmp2, Pt0, Pt1, tmp2, Rt1, Radd2, Pt0, Pt1
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 tmp4, Rval4, Ri.reuse, RZ
    [B------:R-:W-:-:S01] @Pt3 IADD3.X tmp3, Pt0, Pt1, tmp3, Rt2, Radd3, Pt0, Pt1
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 Rt4, Rval5, Ri.reuse, RZ
    [B------:R-:W-:-:S01] @Pt3 IADD3.X tmp4, Pt0, Pt1, tmp4, Rt3, Radd4, Pt0, Pt1
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 tmp6, Rval6, Ri.reuse, RZ
    [B------:R-:W-:-:S01] @Pt3 IADD3.X tmp5, Pt0, Pt1, tmp5, Rt4, Radd5, Pt0, Pt1
    [B------:R-:W-:-:S01] @Pt3 IMAD.WIDE.U32 Rt0, Rval7, Ri.reuse, RZ
    [B------:R-:W-:-:S01] @Pt3 IADD3.X tmp6, Pt0, Pt1, tmp6, Rt5, Radd6, Pt0, Pt1
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro0, tmp0, 30, tmp1
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro1, tmp1, 30, tmp2
    [B------:R-:W-:-:S01] @Pt3 IMAD.U32 tmp8, Rval8, Ri, Rt1
    [B------:R-:W-:-:S01] @Pt3 IADD3.X tmp7, Pt0, Pt1, tmp7, Rt0, Radd7, Pt0, Pt1
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro2, tmp2, 30, tmp3
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro3, tmp3, 30, tmp4
    [B------:R-:W-:-:S01] @Pt3 IADD3.X tmp8, PT, PT, tmp8, RZ, Radd8, Pt0, Pt1
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro4, tmp4, 30, tmp5
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro5, tmp5, 30, tmp6
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro6, tmp6, 30, tmp7
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro7, tmp7, 30, tmp8
    [B------:R-:W-:-:S01] @Pt3 SHF.R.S32.HI Ro8, RZ, 30, tmp8
////end _mul_256_by_u31_add_shift
}

FUNCTION _mul_P_by_32_add_shift(val=Rt0)
{ //Ro_cnt=9, Rt_cnt=1, Pt=[0,1],  Rval, RaddA
////start _mul_P_by_32_add_shift
    [B------:R-:W-:-:S05] @Pt3 IMAD.U32 val, Rval, 0xD2253531, RZ
    [B------:R-:W-:-:S05] @Pt3 LOP3.LUT val, val, 0x3fffffff, RZ, 0xc0, !PT
    [B------:R-:W-:-:S03] @Pt3 IMAD.WIDE.U32 Ro0, val, 0x000003D1, RZ
    [B------:R-:W-:-:S04] @Pt3 IADD3.X Ro0, Pt0, PT, RaddA0, ~Ro0, RZ, !PT, PT
    [B------:R-:W-:-:S04] @Pt3 IADD3.X Ro1, Pt0, Pt1, RaddA1, ~Ro1, ~val, Pt0, PT
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro2, Pt0, PT, RaddA2, 0xFFFFFFFE, RZ, Pt0, Pt1
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro0, Ro0, 30, Ro1
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro3, Pt0, PT, RaddA3, ~RZ, RZ, Pt0, !PT
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro1, Ro1, 30, Ro2
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro4, Pt0, PT, RaddA4, ~RZ, RZ, Pt0, !PT
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro2, Ro2, 30, Ro3
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro5, Pt0, PT, RaddA5, ~RZ, RZ, Pt0, !PT
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro3, Ro3, 30, Ro4
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro6, Pt0, PT, RaddA6, ~RZ, RZ, Pt0, !PT
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro4, Ro4, 30, Ro5
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro7, Pt0, PT, RaddA7, ~RZ, RZ, Pt0, !PT
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro5, Ro5, 30, Ro6
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro8, PT, PT, val, ~RZ, RaddA8, Pt0, !PT
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro6, Ro6, 30, Ro7
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro7, Ro7, 30, Ro8
    [B------:R-:W-:-:S01] @Pt3 SHF.R.S32.HI Ro8, RZ, 30, Ro8
////end _mul_P_by_32_add_shift
}

FUNCTION _mul_P_by_32_add3_shift(val=Rt0)
{ //Ro_cnt=9, Rt_cnt=1, Pt=[0..2],  Rval
////start _mul_P_by_32_add3_shift
    [B------:R-:W-:-:S05] @Pt3 IMAD.U32 val, Rval, 0xD2253531, RZ
    [B------:R-:W-:-:S05] @Pt3 LOP3.LUT val, val, 0x3fffffff, RZ, 0xc0, !PT
    [B------:R-:W-:-:S05] @Pt3 IMAD.WIDE.U32 Ro0, val, 0x000003D1, RZ
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro0, Pt0, Pt1, RaddA0, ~Ro0, RaddB0, !PT, PT
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro1, Pt2, PT, Ro1, val, RZ, !PT, !PT
    [B------:R-:W-:-:S01] @Pt3 IADD3 val, val, -0x1, RZ
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro1, Pt0, Pt1, RaddA1, ~Ro1, RaddB1, Pt0, Pt1
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro2, PT, PT, RZ, RZ, RZ, Pt2, !PT
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro0, Ro0, 30, Ro1
    [B------:R-:W-:-:S04] @Pt3 IADD3.X Ro2, Pt0, Pt1, RaddA2, ~Ro2, RaddB2, Pt0, Pt1
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro3, Pt0, Pt1, RaddA3, ~RZ, RaddB3, Pt0, Pt1
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro1, Ro1, 30, Ro2
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro4, Pt0, Pt1, RaddA4, ~RZ, RaddB4, Pt0, Pt1
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro2, Ro2, 30, Ro3
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro5, Pt0, Pt1, RaddA5, ~RZ, RaddB5, Pt0, Pt1
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro3, Ro3, 30, Ro4
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro6, Pt0, Pt1, RaddA6, ~RZ, RaddB6, Pt0, Pt1
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro4, Ro4, 30, Ro5
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro7, Pt0, Pt1, RaddA7, ~RZ, RaddB7, Pt0, Pt1
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro5, Ro5, 30, Ro6
    [B------:R-:W-:-:S01] @Pt3 IADD3.X Ro8, PT, PT, val, RaddA8, RaddB8, Pt0, Pt1
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro6, Ro6, 30, Ro7
    [B------:R-:W-:-:S01] @Pt3 SHF.R.U32 Ro7, Ro7, 30, Ro8
    [B------:R-:W-:-:S01] @Pt3 SHF.R.S32.HI Ro8, RZ, 30, Ro8
////end _mul_P_by_32_add3_shift
}

FUNCTION _mul_P_by_32_add3_shift_nopredicate(val=Rt0)
{ //Ro_cnt=9, Rt_cnt=1, Pt=[0..2],  Rval
////start _mul_P_by_32_add3_shift
    [B------:R-:W-:-:S05]    IMAD.U32 val, Rval, 0xD2253531, RZ
    [B------:R-:W-:-:S05]    LOP3.LUT val, val, 0x3fffffff, RZ, 0xc0, !PT
    [B------:R-:W-:-:S05]    IMAD.WIDE.U32 Ro0, val, 0x000003D1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Ro0, Pt0, Pt1, RaddA0, ~Ro0, RaddB0, !PT, PT
    [B------:R-:W-:-:S01]    IADD3.X Ro1, Pt2, PT, Ro1, val, RZ, !PT, !PT
    [B------:R-:W-:-:S01]    IADD3 val, val, -0x1, RZ
    [B------:R-:W-:-:S01]    IADD3.X Ro1, Pt0, Pt1, RaddA1, ~Ro1, RaddB1, Pt0, Pt1
    [B------:R-:W-:-:S01]    IADD3.X Ro2, PT, PT, RZ, RZ, RZ, Pt2, !PT
    [B------:R-:W-:-:S01]    SHF.R.U32 Ro0, Ro0, 30, Ro1
    [B------:R-:W-:-:S04]    IADD3.X Ro2, Pt0, Pt1, RaddA2, ~Ro2, RaddB2, Pt0, Pt1
    [B------:R-:W-:-:S01]    IADD3.X Ro3, Pt0, Pt1, RaddA3, ~RZ, RaddB3, Pt0, Pt1
    [B------:R-:W-:-:S01]    SHF.R.U32 Ro1, Ro1, 30, Ro2
    [B------:R-:W-:-:S01]    IADD3.X Ro4, Pt0, Pt1, RaddA4, ~RZ, RaddB4, Pt0, Pt1
    [B------:R-:W-:-:S01]    SHF.R.U32 Ro2, Ro2, 30, Ro3
    [B------:R-:W-:-:S01]    IADD3.X Ro5, Pt0, Pt1, RaddA5, ~RZ, RaddB5, Pt0, Pt1
    [B------:R-:W-:-:S01]    SHF.R.U32 Ro3, Ro3, 30, Ro4
    [B------:R-:W-:-:S01]    IADD3.X Ro6, Pt0, Pt1, RaddA6, ~RZ, RaddB6, Pt0, Pt1
    [B------:R-:W-:-:S01]    SHF.R.U32 Ro4, Ro4, 30, Ro5
    [B------:R-:W-:-:S01]    IADD3.X Ro7, Pt0, Pt1, RaddA7, ~RZ, RaddB7, Pt0, Pt1
    [B------:R-:W-:-:S01]    SHF.R.U32 Ro5, Ro5, 30, Ro6
    [B------:R-:W-:-:S01]    IADD3.X Ro8, PT, PT, val, RaddA8, RaddB8, Pt0, Pt1
    [B------:R-:W-:-:S01]    SHF.R.U32 Ro6, Ro6, 30, Ro7
    [B------:R-:W-:-:S01]    SHF.R.U32 Ro7, Ro7, 30, Ro8
    [B------:R-:W-:-:S01]    SHF.R.S32.HI Ro8, RZ, 30, Ro8
////end _mul_P_by_32_add3_shift
}

//time: about 56 MulMod256
//requires all active threads in warp!
FUNCTION InvMod256(val=Ri0, modp=Rt0, uu=Rt9, aa=Rt10, vv=Rt19, tmpA=Rt20, uv=Rt29, tmpB=Rt30, vu=Rt39, tmpC=Rt40, zeros=Rt20, _val=Rt49, _modp=Rt50, bc=Rt21, mulA=Rt22, mulB=Rt23, one=Rt51, two=URt0, tempA=Rt52, tempB=Rt53, tmpD=Rt54, tvars=Rt64)
{ //Ri_cnt=9(will be spoiled!), Ro_cnt=9, P=[0..3]
//Rt_cnt = 70,  URt=1
////start InvMod256
    [B------:R-:W-:-:S01]    MOV modp0, 0xFFFFFC2F
    [B------:R-:W-:-:S01]    IMAD modp1, RZ, RZ, 0xFFFFFFFE
    [B------:R-:W-:-:S01]    MOV modp2, 0xFFFFFFFF
    [B------:R-:W-:-:S01]    IMAD modp3, RZ, RZ, 0xFFFFFFFF
    [B------:R-:W-:-:S01]    MOV modp4, 0xFFFFFFFF
    [B------:R-:W-:-:S01]    IMAD modp5, RZ, RZ, 0xFFFFFFFF
    [B------:R-:W-:-:S01]    MOV modp6, 0xFFFFFFFF
    [B------:R-:W-:-:S01]    IMAD modp7, RZ, RZ, 0xFFFFFFFF
    [B------:R-:W-:-:S01]    MOV modp8, 0x00
    [B------:R-:W-:-:S01]    IMAD val8, RZ, RZ, RZ
    [B------:R-:W-:-:S01]    MOV one, 0x01
    [B------:R-:W-:-:S01]    UMOV two, 0x02
    [B------:R-:W-:-:S01]    MOV bc, 30
    [B------:R-:W-:-:S01]    MOV uu, 0x01
    [B------:R-:W-:-:S01]    IMAD vv, RZ, RZ, 0x01
    [B------:R-:W-:-:S01]    MOV uv, 0x00
    [B------:R-:W-:-:S01]    IMAD vu, RZ, RZ, 0x00
    [B------:R-:W-:-:S01]    MOV _val, val0
    [B------:R-:W-:-:S01]    IMAD _modp, RZ, RZ, modp0

inc_func _fast_find_first_bit(Ro=zeros, Rt=tvars, Rval=_val, Rone=one, Rbc=bc)

    [B------:R-:W-:-:S01]    ISETP.EQ.AND Pt3, PT, RZ, RZ, PT //set Pt3=PT

    [B------:R-:W-:-:S01]    IADD3 bc, bc, -zeros, RZ
    [B------:R-:W-:-:S01]    SHF.L.U32 uu, uu, zeros, RZ
    [B------:R-:W-:-:S03]    ISETP.GT.AND Pt0, PT, bc, RZ, PT
    [B------:R-:W-:-:S01]    SHF.L.U32 uv, uv, zeros, RZ
    [B------:R-:W-:-:S04]    SHF.R.S32.HI _val, RZ, zeros, _val
    [B------:R-:W-:-:S05]    LOP3.LUT tvars, _val, two, _modp, 0x48, !PT
.label_Start0:
    [B------:R-:W-:-:S02] @Pt0 IMAD mulA, tvars, one, -0x1
    [B------:R-:W-:-:S01] @Pt0 SHF.L.U32 tempA, one, bc, RZ
    [B------:R-:W-:-:S01]    BRA.CONV !Pt0, ~URZ `(.label_Done0)
    [B------:R-:W-:-:S01] @Pt0 IMAD _val, mulA, _modp, _val
    [B------:R-:W-:-:S04] @Pt0 IADD3 mulB, -mulA, RZ, RZ
    [B------:R-:W-:-:S01] @Pt0 LOP3.LUT tempA, tempA, _val, RZ, 0xfc, !PT
    [B------:R-:W-:-:S01] @Pt0 IMAD _modp, mulB, _val, _modp
    [B------:R-:W-:-:S01] @Pt0 IMAD vu, mulA, uu, vu
    [B------:R-:W-:-:S03] @Pt0 IADD3 tempB, -tempA, RZ, RZ
    [B------:R-:W-:-:S01] @Pt0 IMAD vv, mulA, uv, vv
    [B------:R-:W-:-:S03] @Pt0 LOP3.LUT tempA, tempA, tempB, RZ, 0xc0, !PT //(z & -z)
    [B------:R-:W-:-:S01] @Pt0 IMAD uu, mulB, vu, uu
    [B------:R-:W-:-:S03] @Pt0 I2FP.F32.S32 tempA, tempA
    [B------:R-:W-:-:S01] @Pt0 IMAD uv, mulB, vv, uv
    [B------:R-:W-:-:S04] @Pt0 LEA.HI zeros, tempA, 0xffffff81, RZ, 9 //(int >> 23) - 127
    [B------:R-:W-:-:S01] @Pt0 IADD3 bc, bc, -zeros, RZ
    [B------:R-:W-:-:S01] @Pt0 SHF.L.U32 uu, uu, zeros, RZ
    [B------:R-:W-:-:S01] @Pt0 SHF.L.U32 uv, uv, zeros, RZ
    [B------:R-:W-:-:S01] @Pt0 SHF.R.S32.HI _val, RZ, zeros, _val
    [B------:R-:W-:-:S01]    ISETP.GT.AND Pt0, PT, bc, RZ, Pt0
    [B------:R-:W-:-:S01] @Pt0 LOP3.LUT tvars, _val, two, _modp, 0x48, !PT // Rt = (Rv ^ Rm) & 0x2
    [B------:R-:W-:Y:S01]    BRA `(.label_Start0)
.label_Done0:

    //set uu and vv positive
    [B------:R-:W-:Y:S04]    ISETP.LT.AND Pt0, PT, uu, RZ, PT
    [B------:R-:W-:Y:S09]    ISETP.LT.AND Pt1, PT, vv, RZ, PT
    [B------:R-:W-:-:S01] @Pt0 IADD3 uv, RZ, RZ, -uv
    [B------:R-:W-:-:S01] @Pt0 IADD3 uu, RZ, RZ, -uu
    [B------:R-:W-:-:S01] @Pt1 IADD3 vv, RZ, RZ, -vv
    [B------:R-:W-:-:S01] @Pt1 IMAD vu, RZ, RZ, -vu

inc_func _mul_256_by_i32(Ro=tmpB, Pt=0, Rval=val, Ri=uv, Rt=tvars)
inc_func _mul_256_by_i32(Ro=tmpC, Pt=0, Rval=modp, Ri=vu, Rt=tvars)
inc_func _mul_256_by_u31_add_shift(Ro=val, Pt=0, Rval=val, Ri=vv, Radd=tmpC, Rt=tmpD)
inc_func _mul_256_by_u31_add_shift(Ro=modp, Pt=0, Rval=modp, Ri=uu, Radd=tmpB, Rt=tmpD)

//set_288_i32
    [B------:R-:W-:-:S04]    SHF.R.S32.HI tmpB1, RZ, 0x1f, uv  // res = (val < 0) ? 0xFFFFFFFF : 0
    [B------:R-:W-:-:S01]    MOV tmpB0, uv
    [B------:R-:W-:-:S01]    MOV tmpB2, tmpB1
    [B------:R-:W-:-:S01]    IMAD tmpB3, RZ, RZ, tmpB1
    [B------:R-:W-:-:S01]    MOV tmpB4, tmpB1
    [B------:R-:W-:-:S01]    IMAD tmpB5, RZ, RZ, tmpB1
    [B------:R-:W-:-:S01]    MOV tmpB6, tmpB1
    [B------:R-:W-:-:S01]    IMAD tmpB7, RZ, RZ, tmpB1
    [B------:R-:W-:-:S01]    MOV tmpB8, tmpB1
//set_288_i32, vv always positive
    [B------:R-:W-:-:S01]    IMAD tmpD0, RZ, RZ, vv
    [B------:R-:W-:-:S01]    MOV tmpD1, 0x00
    [B------:R-:W-:-:S01]    IMAD tmpD2, RZ, RZ, 0x00
    [B------:R-:W-:-:S01]    MOV tmpD3, 0x00
    [B------:R-:W-:-:S01]    IMAD tmpD4, RZ, RZ, 0x00
    [B------:R-:W-:-:S01]    MOV tmpD5, 0x00
    [B------:R-:W-:-:S01]    IMAD tmpD6, RZ, RZ, 0x00
    [B------:R-:W-:-:S01]    MOV tmpD7, 0x00
    [B------:R-:W-:-:S01]    MOV tmpD8, 0x00

inc_func _mul_P_by_32_add_shift(Ro=Ro0, Pt=0, Rval=tmpB, RaddA=tmpB, Rt=tvars)
inc_func _mul_P_by_32_add_shift(Ro=aa, Pt=0, Rval=tmpD, , RaddA=tmpD, Rt=tvars)

//START OF MAIN LOOP
.label_Start2:
    [B------:R-:W-:-:S02] @Pt3 MOV tvars0, 0x40000000
    [B------:R-:W-:-:S01] @Pt3 IMAD bc, RZ, RZ, 30
    [B------:R-:W-:-:S01] @Pt3 IMAD _modp, modp0, 0x01, RZ
    [B------:R-:W-:-:S03] @Pt3 LOP3.LUT tvars0, tvars0, val0, RZ, 0xfc, !PT //(val | R7)   R2=val
    [B------:R-:W-:-:S01] @Pt3 IMAD vu, RZ, RZ, RZ
    [B------:R-:W-:-:S03] @Pt3 IADD3 tvars1, -tvars0, RZ, RZ //z=Rt0
    [B------:R-:W-:-:S01] @Pt3 IMAD uv, RZ, RZ, RZ
    [B------:R-:W-:-:S02] @Pt3 LOP3.LUT tvars0, tvars0, tvars1, RZ, 0xc0, !PT //(z & -z)
    [B------:R-:W-:-:S04] @Pt3 MOV uu, 0x01
    [B------:R-:W-:-:S02] @Pt3 I2FP.F32.S32 tvars0, tvars0 //float to int
    [B------:R-:W-:-:S04] @Pt3 MOV vv, 0x01
    [B------:R-:W-:-:S02] @Pt3 LEA.HI zeros, tvars0, 0xffffff81, RZ, 9 //(int >> 23) - 127
    [B------:R-:W-:-:S04] @Pt3 MOV _val, val0
    [B------:R-:W-:-:S01] @Pt3 IADD3 bc, bc, -zeros, RZ
    [B------:R-:W-:-:S01] @Pt3 SHF.R.S32.HI _val, RZ, zeros, _val
    [B------:R-:W-:-:S01]    ISETP.GT.AND Pt0, PT, bc, RZ, Pt3
    [B------:R-:W-:-:S03] @Pt3 SHF.L.U32 uu, uu, zeros, RZ
    [B------:R-:W-:-:S04] @Pt3 LOP3.LUT tvars, _val, two, _modp, 0x48, !PT
    [B------:R-:W-:-:S05] @Pt3 SHF.L.U32 uv, uv, zeros, RZ
.label_Start1: //tight loop
    [B------:R-:W-:-:S02] @Pt0 IMAD mulA, tvars, one, -0x1
    [B------:R-:W-:-:S01] @Pt0 SHF.L.U32 tempA, one, bc, RZ
    [B------:R-:W-:-:S01]    BRA.CONV !Pt0, ~URZ `(.label_Done1)
    [B------:R-:W-:-:S01] @Pt0 IMAD _val, mulA, _modp, _val
    [B------:R-:W-:-:S04] @Pt0 IADD3 mulB, -mulA, RZ, RZ
    [B------:R-:W-:-:S01] @Pt0 LOP3.LUT tempA, tempA, _val, RZ, 0xfc, !PT
    [B------:R-:W-:-:S01] @Pt0 IMAD _modp, mulB, _val, _modp
    [B------:R-:W-:-:S01] @Pt0 IMAD vu, mulA, uu, vu
    [B------:R-:W-:-:S03] @Pt0 IADD3 tempB, -tempA, RZ, RZ
    [B------:R-:W-:-:S01] @Pt0 IMAD vv, mulA, uv, vv
    [B------:R-:W-:-:S03] @Pt0 LOP3.LUT tempA, tempA, tempB, RZ, 0xc0, !PT //(z & -z)
    [B------:R-:W-:-:S01] @Pt0 IMAD uu, mulB, vu, uu
    [B------:R-:W-:-:S03] @Pt0 I2FP.F32.S32 tempA, tempA
    [B------:R-:W-:-:S01] @Pt0 IMAD uv, mulB, vv, uv
    [B------:R-:W-:-:S04] @Pt0 LEA.HI zeros, tempA, 0xffffff81, RZ, 9 //(int >> 23) - 127
    [B------:R-:W-:-:S01] @Pt0 IADD3 bc, bc, -zeros, RZ
    [B------:R-:W-:-:S01] @Pt0 SHF.L.U32 uu, uu, zeros, RZ
    [B------:R-:W-:-:S01] @Pt0 SHF.L.U32 uv, uv, zeros, RZ
    [B------:R-:W-:-:S01] @Pt0 SHF.R.S32.HI _val, RZ, zeros, _val
    [B------:R-:W-:-:S01]    ISETP.GT.AND Pt0, PT, bc, RZ, Pt0
    [B------:R-:W-:-:S01] @Pt0 LOP3.LUT tvars, _val, two, _modp, 0x48, !PT // Rt = (Rv ^ Rm) & 0x2
    [B------:R-:W-:Y:S01]    BRA `(.label_Start1)
.label_Done1:

    //set uu and vv positive
    [B------:R-:W-:Y:S04]    ISETP.LT.AND Pt0, PT, uu, RZ, Pt3
    [B------:R-:W-:Y:S09]    ISETP.LT.AND Pt1, PT, vv, RZ, Pt3
    [B------:R-:W-:-:S01] @Pt0 IADD3 uu, -uu, RZ, RZ
    [B------:R-:W-:-:S01] @Pt0 IADD3 uv, -uv, RZ, RZ
    [B------:R-:W-:-:S01] @Pt1 IADD3 vv, -vv, RZ, RZ
    [B------:R-:W-:-:S01] @Pt1 IADD3 vu, -vu, RZ, RZ

inc_func _mul_256_by_i32(Ro=tmpB, Pt=0, Rval=val, Ri=uv, Rt=tvars)
inc_func _mul_256_by_i32(Ro=tmpC, Pt=0, Rval=modp, Ri=vu, Rt=tvars)
inc_func _mul_256_by_u31_add_shift(Ro=val, Pt=0, Rval=val, Ri=vv, Radd=tmpC, Rt=tmpD)
    [B------:R-:W-:-:S01] @Pt3 LOP3.LUT tempA, val0, val1, val2,  0xFE, !PT
    [B------:R-:W-:-:S01] @Pt3 LOP3.LUT tempB, val3, val4, val5,  0xFE, !PT
    [B------:R-:W-:-:S01] @Pt3 LOP3.LUT zeros, val6, val7, val8,  0xFE, !PT
inc_func _mul_256_by_u31_add_shift(Ro=modp, Pt=0, Rval=modp, Ri=uu, Radd=tmpB, Rt=tmpD)
    [B------:R-:W-:-:S01] @Pt3 LOP3.LUT tempA, tempA, tempB, zeros, 0xFE, !PT
inc_func _mul_256_by_i32(Ro=tmpB, Pt=0, Rval=aa, Ri=uv, Rt=tvars)
    [B------:R-:W-:-:S01]    ISETP.NE.AND Pt1, PT, tempA, RZ, Pt3
inc_func _mul_256_by_u31(Ro=tmpA, Pt=0, Rval=Ro0, Ri=uu, Rt=tvars)

//IF (val[0..8] == 0) THEN STOP MAIN LOOP
    [B------:R-:W-:-:S01]    BRA.CONV !Pt1, ~URZ `(.label_Done2)
    [B------:R-:W-:-:S01]    ISETP.EQ.AND Pt3, PT, RZ, RZ, Pt1 //set Pt3=Pt1

inc_func _mul_256_by_i32(Ro=tmpC, Pt=0, Rval=Ro0, Ri=vu, Rt=tvars)
    [B------:R-:W-:-:S01] @Pt3 IADD3 tempA, tmpA0, tmpB0, RZ
inc_func _mul_256_by_u31(Ro=tmpD, Pt=0, Rval=aa, Ri=vv, Rt=tvars)
    [B------:R-:W-:-:S01] @Pt3 IADD3 tempB, tmpC0, tmpD0, RZ

inc_func _mul_P_by_32_add3_shift(Ro=Ro0, Pt=0, Rval=tempA, RaddA=tmpA, RaddB=tmpB, Rt=tvars)
inc_func _mul_P_by_32_add3_shift(Ro=aa, Pt=0, Rval=tempB, RaddA=tmpC, RaddB=tmpD, Rt=tvars)
//END OF MAIN WHILE
    [B------:R-:W-:Y:S01]    BRA `(.label_Start2)
.label_Done2:

    [B------:R-:W-:Y:S04]    IADD3 tempA, tmpA0, tmpB0, RZ
    [B------:R-:W-:-:S01]    ISETP.LT.AND Pt3, PT, modp8, RZ, PT

inc_func _mul_P_by_32_add3_shift_nopredicate(Ro=Ro0, Pt=0, Rval=tempA, RaddA=tmpA, RaddB=tmpB, Rt=tvars)

//if ((int)modp[8] < 0) neg_288(res)
    [B------:R-:W-:-:S04] @Pt3 IADD3.X Ro0, Pt0, PT, RZ, ~Ro0, RZ, PT, !PT
    [B------:R-:W-:-:S04] @Pt3 IADD3.X Ro1, Pt0, PT, RZ, ~Ro1, RZ, Pt0, !PT
    [B------:R-:W-:-:S04] @Pt3 IADD3.X Ro2, Pt0, PT, RZ, ~Ro2, RZ, Pt0, !PT
    [B------:R-:W-:-:S04] @Pt3 IADD3.X Ro3, Pt0, PT, RZ, ~Ro3, RZ, Pt0, !PT
    [B------:R-:W-:-:S04] @Pt3 IADD3.X Ro4, Pt0, PT, RZ, ~Ro4, RZ, Pt0, !PT
    [B------:R-:W-:-:S04] @Pt3 IADD3.X Ro5, Pt0, PT, RZ, ~Ro5, RZ, Pt0, !PT
    [B------:R-:W-:-:S04] @Pt3 IADD3.X Ro6, Pt0, PT, RZ, ~Ro6, RZ, Pt0, !PT
    [B------:R-:W-:-:S04] @Pt3 IADD3.X Ro7, Pt0, PT, RZ, ~Ro7, RZ, Pt0, !PT
    [B------:R-:W-:-:S05] @Pt3 IADD3.X Ro8, PT, PT, RZ, ~Ro8, RZ, Pt0, !PT

//	while ((int)res[8] < 0) add_288_P(res);
.label_Start3:
    [B------:R-:W-:Y:S13]    ISETP.LT.AND Pt1, PT, Ro8, RZ, PT
    [B------:R-:W-:-:S01]    BRA.CONV !Pt1, ~URZ `(.label_Done3)
    [B------:R-:W-:-:S04] @Pt1 IADD3 Ro0, Pt0, Ro0, 0xFFFFFC2F, RZ
    [B------:R-:W-:-:S04] @Pt1 IADD3.X Ro1, Pt0, PT, Ro1, 0xFFFFFFFE, RZ, Pt0, !PT
    [B------:R-:W-:-:S04] @Pt1 IADD3.X Ro2, Pt0, PT, Ro2, 0xFFFFFFFF, RZ, Pt0, !PT
    [B------:R-:W-:-:S04] @Pt1 IADD3.X Ro3, Pt0, PT, Ro3, 0xFFFFFFFF, RZ, Pt0, !PT
    [B------:R-:W-:-:S04] @Pt1 IADD3.X Ro4, Pt0, PT, Ro4, 0xFFFFFFFF, RZ, Pt0, !PT
    [B------:R-:W-:-:S04] @Pt1 IADD3.X Ro5, Pt0, PT, Ro5, 0xFFFFFFFF, RZ, Pt0, !PT
    [B------:R-:W-:-:S04] @Pt1 IADD3.X Ro6, Pt0, PT, Ro6, 0xFFFFFFFF, RZ, Pt0, !PT
    [B------:R-:W-:-:S04] @Pt1 IADD3.X Ro7, Pt0, PT, Ro7, 0xFFFFFFFF, RZ, Pt0, !PT
    [B------:R-:W-:-:S04] @Pt1 IADD3.X Ro8, Pt0, PT, Ro8, RZ, RZ, Pt0, !PT
    [B------:R-:W-:Y:S01]    BRA `(.label_Start3)
.label_Done3:

// while ((int)res[8] > 0) sub_288_P(res);
.label_Start4:
    [B------:R-:W-:Y:S13]    ISETP.GT.AND Pt1, PT, Ro8, RZ, PT
    [B------:R-:W-:-:S01]    BRA.CONV !Pt1, ~URZ `(.label_Done4)
    [B------:R-:W-:-:S04] @Pt1 IADD3   Ro0, Pt0, Ro0, 0x000003D1, RZ
    [B------:R-:W-:-:S04] @Pt1 IADD3.X Ro1, Pt0, PT, Ro1, 0x00000001, RZ, Pt0, !PT
    [B------:R-:W-:-:S04] @Pt1 IADD3.X Ro2, Pt0, PT, Ro2, 0x00000000, RZ, Pt0, !PT
    [B------:R-:W-:-:S04] @Pt1 IADD3.X Ro3, Pt0, PT, Ro3, 0x00000000, RZ, Pt0, !PT
    [B------:R-:W-:-:S04] @Pt1 IADD3.X Ro4, Pt0, PT, Ro4, 0x00000000, RZ, Pt0, !PT
    [B------:R-:W-:-:S04] @Pt1 IADD3.X Ro5, Pt0, PT, Ro5, 0x00000000, RZ, Pt0, !PT
    [B------:R-:W-:-:S04] @Pt1 IADD3.X Ro6, Pt0, PT, Ro6, 0x00000000, RZ, Pt0, !PT
    [B------:R-:W-:-:S04] @Pt1 IADD3.X Ro7, Pt0, PT, Ro7, 0x00000000, RZ, Pt0, !PT
    [B------:R-:W-:-:S04] @Pt1 IADD3.X Ro8, PT, PT, Ro8, 0xFFFFFFFF, RZ, Pt0, !PT
    [B------:R-:W-:Y:S01]    BRA `(.label_Start4)
.label_Done4:
////end InvMod256
}