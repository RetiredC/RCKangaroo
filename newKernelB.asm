//(c) 2026, Retired Coder(RC)

CONST MD_LEN = 10
CONST MAX_DP_CNT = (256 * 1024)
CONST GPU_DP_SIZE = 48


FUNCTION SaveRegAB(Ind=Rt0)
{
    [B------:R-:W-:-:S05]    IMAD.U32 Ind, RCurA, 0x0000CCCD, RZ
    [B------:R-:W-:-:S05]    SHF.R.U32.HI Ind, RZ, 0x13, Ind
    [B------:R-:W-:-:S05]    IMAD.U32 Ind, Ind, 0xFFFFFFF6, RCurA //int ind = (i + MD_LEN - cur_indA) % MD_LEN;

    [B----4-:R-:W-:-:S05]    IMAD.U32 TmpB, Ind, {256}, ThrID
    [B------:R-:W-:-:S05]    IMAD.U32 TmpB, TmpB, 0x08, TmpD
    [B------:R4:W-:-:S02]    STG.E.64 [TmpB.U32 + LoopTableBase], RegA
////
    [B------:R-:W-:-:S05]    IMAD.U32 Ind, RCurB, 0x0000CCCD, RZ
    [B------:R-:W-:-:S05]    SHF.R.U32.HI Ind, RZ, 0x13, Ind
    [B------:R-:W-:-:S05]    IMAD.U32 Ind, Ind, 0xFFFFFFF6, RCurB //int ind = (i + MD_LEN - cur_indB) % MD_LEN;

    [B------:R-:W-:-:S05]    IMAD.U32 TmpA, Ind, {256}, ThrID
    [B-----5:R-:W-:-:S05]    IMAD.U32 TmpC, TmpA, 0x08, TmpD
    [B------:R5:W-:-:S02]    STG.E.64 [TmpC.U32 + LoopTableBase + {MD_LEN * 256 * 8}], RegB
    [B------:R-:W-:-:S04]    IADD3 RCurA, RCurA, 0x01, RZ
    [B------:R-:W-:-:S01]    IADD3 RCurB, RCurB, 0x01, RZ
}

FUNCTION PJD(LoopSize=Rt8, LS=Rt9, tmp=Rt10, \
ind_LastPnts=Rt12, var=Rt13, ind=Rt14, pos=Rt18, rx=Rt20)
{
//begin PJD

//// Kangaroo A
    [B------:R-:W-:-:S01]    MOV LoopSize, 0x04 //loop4
    [B------:R-:W-:Y:S13]    ISETP.EQ.U32.AND P1, PT, RvA_A0, dA0, PT
    [B------:R-:W-:Y:S13]    ISETP.EQ.U32.AND P0, PT, RvA_A1, dA1, P1

    [B------:R-:W-:-:S01] @!P0 IADD3 LoopSize, LoopSize, 0x02, RZ //loop6
    [B------:R-:W-:Y:S13] @!P0 ISETP.EQ.U32.AND P1, PT, RvB_A0, dA0, PT
    [B------:R-:W-:Y:S13] @!P0 ISETP.EQ.U32.AND P0, PT, RvB_A1, dA1, P1

    [B------:R-:W-:-:S01] @!P0 IADD3 LoopSize, LoopSize, 0x02, RZ //loop8
    [B------:R-:W-:Y:S13] @!P0 ISETP.EQ.U32.AND P1, PT, RvC_A0, dA0, PT
    [B------:R-:W-:Y:S13] @!P0 ISETP.EQ.U32.AND P0, PT, RvC_A1, dA1, P1

    [B------:R-:W-:-:S01] @!P0 IADD3 LoopSize, LoopSize, 0x02, RZ //loop10
    [B------:R-:W-:Y:S13] @!P0 ISETP.EQ.U32.AND P1, PT, RvD_A0, dA0, PT
    [B------:R-:W-:Y:S13] @!P0 ISETP.EQ.U32.AND P0, PT, RvD_A1, dA1, P1

    [B------:R-:W-:-:S01] @!P0 MOV LoopSize, 0xFFFFFFFF

// table[iter] = d[0]
    [B------:R-:W-:-:S01] @P5 MOV RvD_A0, dA0
    [B------:R-:W-:-:S01] @P5 MOV RvD_A1, dA1
    [B------:R-:W-:-:S01] @P5 MOV cur_indA, CiterNxt

//build DP
    [B------:R-:W-:Y:S13]    LOP3.LUT.PAND P3, RZ, cur_jA, {DP_FLAG}, RZ, 0xC0, P5
    [B------:R-:W5:-:S02] @P3 ATOMG.E.EL.ADD.STRONG_GPU PT, ind, [kang_indAF.U32 + DPTableBase], TenTh //TenTh = 0x10000
    [B-----5:R-:W-:-:S06]    SHF.R.U32.HI ind, RZ, 0x10, ind
    [B------:R-:W-:Y:S13]    ISETP.LT.AND P3, PT, ind, {DPTABLE_MAX_CNT}, P3
    [B------:R-:W-:-:S06]    IMAD.U32 tmp, kang_indA, {DPTABLE_MAX_CNT}, ind
    [B------:R-:W-:-:S06]    IMAD.U32 tmp, tmp, 0x10, KangCntF
    [B------:R-:W5:-:S02] @P3 LDG.E.128 rx, [tmp.U32 + DPTableBase]
    [B------:R-:W5:-:S02] @P3 ATOMG.E.EL.ADD.STRONG_GPU PT, pos, [RZ.U32 + DPsBase], One
    [B-----5:R-:W-:-:S06]    IMNMX pos, pos, {MAX_DP_CNT - 1}, PT
    [B------:R-:W-:-:S06]    IMAD.U32 pos, pos, {GPU_DP_SIZE}, RZ
    [B------:R4:W-:-:S02] @P3 STG.E.128 [pos.U32 + DPsBase + 0x10], rx
    [B------:R4:W-:-:S02] @P3 STG.E.128 [pos.U32 + DPsBase + 0x20], dA0
    [B------:R4:W-:-:S02] @P3 STG.E.64 [pos.U32 + DPsBase + 0x30], dA4
    [B------:R4:W-:-:S02] @P3 STG.E [pos.U32 + DPsBase + 0x38], kang_indA
//
    [B------:R-:W-:Y:S13]    ISETP.NE.U32.AND P4, PT, LoopSize, 0xFFFFFFFF, P5
    [B------:R-:W-:-:S06]    IMAD.U32 LS, LoopSize, 0x4, RZ
    [B------:R4:W-:-:S02] @P4 RED.E.EL.ADD.STRONG_GPU [LS.U32 + DbgBase], One //dbg
//ind_LastPnts = MD_LEN - 1 - ((STEP_CNT - 1 - step_ind) % LoopSize);
    [B------:R-:W-:-:S06]    IADD3 tmp, IterCnt, CHlpOfs, -StepLoopInd
    [B------:R-:W-:-:S06]    ISETP.EQ.U32.AND P0, PT, LoopSize, 0x6, PT
    [B------:R-:W-:-:S06]    ISETP.EQ.U32.AND P1, PT, LoopSize, 0xA, PT
    [B------:R-:W-:-:S06]    IADD3 var, LoopSize, 0xFFFFFFFF, RZ
    [B------:R-:W-:-:S06]    LOP3.LUT ind_LastPnts, tmp, var, RZ, 0xC0, !PT
    [B------:R-:W-:-:S06]    SHF.L.U32 ind_LastPnts, ind_LastPnts, 0x13, RZ
    [B------:R-:W-:-:S06] @P0 IMAD.U32 ind_LastPnts, tmp, 0x00015556, RZ
    [B------:R-:W-:-:S06] @P1 IMAD.U32 ind_LastPnts, tmp, 0x0000CCCD, RZ
    [B------:R-:W-:-:S06]    SHF.R.U32.HI ind_LastPnts, RZ, 0x13, ind_LastPnts
    [B------:R-:W-:-:S06] @P0 IMAD.U32 ind_LastPnts, ind_LastPnts, 0xFFFFFFFA, tmp
    [B------:R-:W-:-:S06] @P1 IMAD.U32 ind_LastPnts, ind_LastPnts, 0xFFFFFFF6, tmp
    [B------:R-:W-:-:S06]    IADD3 ind_LastPnts, RZ, {MD_LEN} - 1, -ind_LastPnts

    [B------:R-:W5:-:S02] @P4 ATOMG.E.EL.ADD.STRONG_GPU PT, ind, [RZ.U32 + LoopedKangsBase + 0x00], One
    [B-----5:R-:W-:-:S06]    IMAD.U32 ind, ind, 0x04, RZ

    [B------:R-:W-:-:S06]    SHF.L.U32 ind_LastPnts, ind_LastPnts, 0x1C, RZ
    [B------:R-:W-:-:S06]    IADD3 ind_LastPnts, ind_LastPnts, kang_indA, RZ
    [B------:R4:W-:-:S02] @P4 STG.E [ind.U32 + LoopedKangsBase + 0x08], ind_LastPnts
    [B----4-:R-:W-:Y:S13] @P4 ISETP.NE.AND P5, PT, RZ, RZ, !PT //if P4 then set P5=False

//// Kangaroo B
    [B------:R-:W-:-:S01]    MOV LoopSize, 0x04 //loop4
    [B------:R-:W-:Y:S13]    ISETP.EQ.U32.AND P1, PT, RvA_B0, dB0, PT
    [B------:R-:W-:Y:S13]    ISETP.EQ.U32.AND P0, PT, RvA_B1, dB1, P1

    [B------:R-:W-:-:S01] @!P0 IADD3 LoopSize, LoopSize, 0x02, RZ //loop6
    [B------:R-:W-:Y:S13] @!P0 ISETP.EQ.U32.AND P1, PT, RvB_B0, dB0, PT
    [B------:R-:W-:Y:S13] @!P0 ISETP.EQ.U32.AND P0, PT, RvB_B1, dB1, P1

    [B------:R-:W-:-:S01] @!P0 IADD3 LoopSize, LoopSize, 0x02, RZ //loop8
    [B------:R-:W-:Y:S13] @!P0 ISETP.EQ.U32.AND P1, PT, RvC_B0, dB0, PT
    [B------:R-:W-:Y:S13] @!P0 ISETP.EQ.U32.AND P0, PT, RvC_B1, dB1, P1

    [B------:R-:W-:-:S01] @!P0 IADD3 LoopSize, LoopSize, 0x02, RZ //loop10
    [B------:R-:W-:Y:S13] @!P0 ISETP.EQ.U32.AND P1, PT, RvD_B0, dB0, PT
    [B------:R-:W-:Y:S13] @!P0 ISETP.EQ.U32.AND P0, PT, RvD_B1, dB1, P1

    [B------:R-:W-:-:S01] @!P0 MOV LoopSize, 0xFFFFFFFF

// table[iter] = d[0]
    [B------:R-:W-:-:S01] @P6 MOV RvD_B0, dB0
    [B------:R-:W-:-:S01] @P6 MOV RvD_B1, dB1
    [B------:R-:W-:-:S01] @P6 MOV cur_indB, CiterNxt

//build DP
    [B------:R-:W-:Y:S13]    LOP3.LUT.PAND P3, RZ, cur_jB, {DP_FLAG}, RZ, 0xC0, P6
    [B------:R-:W5:-:S02] @P3 ATOMG.E.EL.ADD.STRONG_GPU PT, ind, [kang_indBF.U32 + DPTableBase], TenTh //TenTh = 0x10000
    [B-----5:R-:W-:-:S06]    SHF.R.U32.HI ind, RZ, 0x10, ind
    [B------:R-:W-:Y:S13]    ISETP.LT.AND P3, PT, ind, {DPTABLE_MAX_CNT}, P3
    [B------:R-:W-:-:S06]    IMAD.U32 tmp, kang_indB, {DPTABLE_MAX_CNT}, ind
    [B------:R-:W-:-:S06]    IMAD.U32 tmp, tmp, 0x10, KangCntF
    [B------:R-:W5:-:S02] @P3 LDG.E.128 rx, [tmp.U32 + DPTableBase]
    [B------:R-:W5:-:S02] @P3 ATOMG.E.EL.ADD.STRONG_GPU PT, pos, [RZ.U32 + DPsBase], One
    [B-----5:R-:W-:-:S06]    IMNMX pos, pos, {MAX_DP_CNT - 1}, PT
    [B------:R-:W-:-:S06]    IMAD.U32 pos, pos, {GPU_DP_SIZE}, RZ
    [B------:R4:W-:-:S02] @P3 STG.E.128 [pos.U32 + DPsBase + 0x10], rx
    [B------:R4:W-:-:S02] @P3 STG.E.128 [pos.U32 + DPsBase + 0x20], dB0
    [B------:R4:W-:-:S02] @P3 STG.E.64 [pos.U32 + DPsBase + 0x30], dB4
    [B------:R4:W-:-:S02] @P3 STG.E [pos.U32 + DPsBase + 0x38], kang_indB
////
    [B------:R-:W-:Y:S13]    ISETP.NE.U32.AND P4, PT, LoopSize, 0xFFFFFFFF, P6
    [B------:R-:W-:-:S06]    IMAD.U32 LS, LoopSize, 0x4, RZ
    [B------:R4:W-:-:S02] @P4 RED.E.EL.ADD.STRONG_GPU [LS.U32 + DbgBase], One //dbg
//ind_LastPnts = MD_LEN - 1 - ((STEP_CNT - 1 - step_ind) % LoopSize);
    [B------:R-:W-:-:S06]    IADD3 tmp, IterCnt, CHlpOfs, -StepLoopInd
    [B------:R-:W-:-:S06]    ISETP.EQ.U32.AND P0, PT, LoopSize, 0x6, PT
    [B------:R-:W-:-:S06]    ISETP.EQ.U32.AND P1, PT, LoopSize, 0xA, PT
    [B------:R-:W-:-:S06]    IADD3 var, LoopSize, 0xFFFFFFFF, RZ
    [B------:R-:W-:-:S06]    LOP3.LUT ind_LastPnts, tmp, var, RZ, 0xC0, !PT
    [B------:R-:W-:-:S06]    SHF.L.U32 ind_LastPnts, ind_LastPnts, 0x13, RZ
    [B------:R-:W-:-:S06] @P0 IMAD.U32 ind_LastPnts, tmp, 0x00015556, RZ
    [B------:R-:W-:-:S06] @P1 IMAD.U32 ind_LastPnts, tmp, 0x0000CCCD, RZ
    [B------:R-:W-:-:S06]    SHF.R.U32.HI ind_LastPnts, RZ, 0x13, ind_LastPnts
    [B------:R-:W-:-:S06] @P0 IMAD.U32 ind_LastPnts, ind_LastPnts, 0xFFFFFFFA, tmp
    [B------:R-:W-:-:S06] @P1 IMAD.U32 ind_LastPnts, ind_LastPnts, 0xFFFFFFF6, tmp
    [B------:R-:W-:-:S06]    IADD3 ind_LastPnts, RZ, {MD_LEN} - 1, -ind_LastPnts

    [B------:R-:W5:-:S02] @P4 ATOMG.E.EL.ADD.STRONG_GPU PT, ind, [RZ.U32 + LoopedKangsBase + 0x00], One
    [B-----5:R-:W-:-:S06]    IMAD.U32 ind, ind, 0x04, RZ

    [B------:R-:W-:-:S06]    SHF.L.U32 ind_LastPnts, ind_LastPnts, 0x1C, RZ
    [B------:R-:W-:-:S06]    IADD3 ind_LastPnts, ind_LastPnts, kang_indB, RZ
    [B------:R4:W-:-:S02] @P4 STG.E [ind.U32 + LoopedKangsBase + 0x08], ind_LastPnts
    [B----4-:R-:W-:Y:S13] @P4 ISETP.NE.AND P6, PT, RZ, RZ, !PT //if P4 then set P6=False
//end PJD
}


FUNCTION FastPath()
{
//get data and order next
    [B------:R-:W-:-:S04]    DEPBAR.LE SB0, 9
    [B------:R-:W-:-:S05]    MOV cur_jA, Rcur_jAB
    [B------:R5:W0:-:S02]    LDG.E.NA.STRONG_SM Rcur_jAB, [jAdr.U32 + JumpsListBaseCur]

// [B------:R-:W-:-:S09]    MOV cur_jAB, StepLoopInd //DEBUG
//Kang A and B
    [B------:R-:W-:-:S05]    LOP3.LUT PT, TmpA, cur_jA, {JMP_MASK_ADV}, RZ, 0xC0, !PT
    [B------:R-:W1:-:S02]    LDS.128 jmpSixA0, [TmpA.X16]    
    [B------:R-:W-:-:S05]    SHF.R.U32.HI cur_jB, RZ, 0x10, cur_jA
    [B------:R-:W-:-:S05]    LOP3.LUT PT, TmpB, cur_jB, {JMP_MASK_ADV}, RZ, 0xC0, !PT
    [B------:R-:W3:-:S02]    LDS.128 jmpSixB0, [TmpB.X16]
    [B------:R-:W2:-:S02]    LDS.64 jmpSixA4, [TmpA.X8 + {32 * 1024}]
    [B------:R-:W4:-:S02]    LDS.64 jmpSixB4, [TmpB.X8 + {32 * 1024}]
////
    [B-1----:R-:W-:-:S01] @P5 IADD3.X dA0, P0, dA0, jmpSixA0, RZ, !PT, !PT
    [B---3--:R-:W-:-:S01] @P6 IADD3.X dB0, P1, dB0, jmpSixB0, RZ, !PT, !PT
    [B------:R-:W-:-:S01] @P5 IADD3.X dA1, P0, dA1, jmpSixA1, RZ, P0, !PT
    [B------:R-:W-:-:S01] @P6 IADD3.X dB1, P1, dB1, jmpSixB1, RZ, P1, !PT
    [B------:R-:W-:-:S01] @P5 IADD3.X dA2, P0, dA2, jmpSixA2, RZ, P0, !PT
    [B------:R-:W-:-:S01] @P6 IADD3.X dB2, P1, dB2, jmpSixB2, RZ, P1, !PT
    [B------:R-:W-:-:S01] @P5 IADD3.X dA3, P0, dA3, jmpSixA3, RZ, P0, !PT
    [B------:R-:W-:-:S01] @P6 IADD3.X dB3, P1, dB3, jmpSixB3, RZ, P1, !PT
    [B--2---:R-:W-:-:S01] @P5 IADD3.X dA4, P0, dA4, jmpSixA4, RZ, P0, !PT
    [B----4-:R-:W-:-:S01] @P6 IADD3.X dB4, P1, dB4, jmpSixB4, RZ, P1, !PT
    [B------:R-:W-:-:S01] @P5 IADD3.X dA5, PT, dA5, jmpSixA5, RZ, P0, !PT
    [B------:R-:W-:-:S01] @P6 IADD3.X dB5, PT, dB5, jmpSixB5, RZ, P1, !PT
////
    [B------:R-:W-:-:S03]    ISETP.EQ.U32.AND P4, PT, RvA_A0, dA0, P5
    [B------:R-:W-:-:S03]    ISETP.EQ.U32.AND P3, PT, RvB_A0, dA0, P5
    [B------:R-:W-:-:S03]    ISETP.EQ.U32.AND P0, PT, RvA_A1, dA1, P4
    [B------:R-:W-:-:S03]    ISETP.EQ.U32.AND P1, PT, RvB_A1, dA1, P3
    [B------:R-:W-:-:S03]    ISETP.EQ.U32.AND P4, PT, RvC_A0, dA0, P5
    [B------:R-:W-:-:S03]    ISETP.EQ.U32.AND P3, PT, RvD_A0, dA0, P5
    [B------:R-:W-:-:S03]    ISETP.EQ.U32.AND P2, PT, RvC_A1, dA1, P4
    [B------:R-:W-:-:S03]    ISETP.EQ.U32.AND P3, PT, RvD_A1, dA1, P3

    [B------:R-:W-:-:S01]    LOP3.LUT.PAND P4, RZ, cur_jA, {DP_FLAG}, RZ, 0xC0, P5
    [B------:R-:W-:-:S04]    PLOP3.LUT P0, PT, P0, P1, P2, 0xFE, 0x0
    [B------:R-:W-:-:S04]    PLOP3.LUT P0, PT, P0, P3, P4, 0xFE, 0x0
    [B------:R-:W-:-:S01]    IADD3.X TmpA, RZ, RZ, RZ, P0, !PT

    [B------:R-:W-:-:S03]    ISETP.EQ.U32.AND P4, PT, RvA_B0, dB0, P6
    [B------:R-:W-:-:S03]    ISETP.EQ.U32.AND P3, PT, RvB_B0, dB0, P6
    [B------:R-:W-:-:S03]    ISETP.EQ.U32.AND P0, PT, RvA_B1, dB1, P4
    [B------:R-:W-:-:S03]    ISETP.EQ.U32.AND P1, PT, RvB_B1, dB1, P3
    [B------:R-:W-:-:S03]    ISETP.EQ.U32.AND P4, PT, RvC_B0, dB0, P6
    [B------:R-:W-:-:S03]    ISETP.EQ.U32.AND P3, PT, RvD_B0, dB0, P6
    [B------:R-:W-:-:S03]    ISETP.EQ.U32.AND P2, PT, RvC_B1, dB1, P4
    [B------:R-:W-:-:S03]    ISETP.EQ.U32.AND P3, PT, RvD_B1, dB1, P3

    [B------:R-:W-:-:S03]    LOP3.LUT.PAND P4, RZ, cur_jB, {DP_FLAG}, RZ, 0xC0, P6
    [B------:R-:W-:-:S04]    PLOP3.LUT P0, PT, P0, P1, P2, 0xFE, 0x0
    [B------:R-:W-:-:S04]    PLOP3.LUT P0, PT, P0, P3, P4, 0xFE, 0x0

    [B------:R-:W-:Y:S12]    LOP3.LUT P0, RZ, TmpA, 0x01, RZ, 0xC0, P0
    [B-----5:R-:W-:-:S01]    IADD3 jAdr, jAdr, {256 * GROUP_SIZE * 2}, RZ
}


KERNEL KernelB(regcnt=255, ThrID=R0, BlockID=R1, gID=R2, TmpA=R3, TmpB=R4, TmpC=R5, TmpD=R6, \
RegsA=R8, RegsB=R28, cur_indA=R48, cur_indB=R49, dA=R52, dB=R60, \
kang_indA=R68, kang_indB=R69, cur_jA=R71, cur_jB=R72, ThrIDx=R73, KangCnt=R74, KangCntF=R75, RInd=R76, TenTh=R77, IterCnt=R78, StepLoopInd=R79, RegOfs=R80, One=R81, \
kang_indAF=R82, kang_indBF=R83, rx=R84, jAdr=R85, jmpSixA=R88, jmpSixB=R96, cur_jAB=R104, /*cur_jAB takes 10regs!*/  \
TmpTmp=R200, \
CPolicy=UR0, uIterCnt=UR2, uKangCnt=UR3, DPsBase=UR4, DPsBase=UR6, LoopTableBase=UR8, LoopedKangsBase=UR10, \
distsBase=UR12, JmpDists12Base=UR14, JumpsListBase=UR16, uTmpA=UR18, uMainLoopInd=UR19, \
DbgBase=UR20, uMainLoopPnt=UR22, uMainLoopHlp=UR24, JumpsListBaseCur=UR26, uBlockCnt=UR28, \
uStepLoopInd=UR29, uStepLoopPnt=UR30, uStepLoopHlp=UR32, uOne=UR33, uTmpB=UR34, uIterCntP=UR35, uBlockID=UR36, DPTableBase=UR38, uRetAdr=UR40, uTmpTmp=UR52)
{
////start KernelB
    [B------:R-:W0:-:S01]    S2R ThrID, SR_TID.X //ThrID
    [B------:R-:W0:-:S01]    S2R BlockID, SR_CTAID.X //BlockID
    [B------:R-:W0:-:S01]    S2UR uBlockID, SR_CTAID.X //BlockID
#IF {SM_VER} == 89
    [B------:R-:W-:-:S01]    ULDC.64 CPolicy, c[0x0][0x118] //zero cache policy UR0..UR1
    [B------:R-:W-:-:S01]    ULDC.64 DPTableBase, c[0x0][0x170] //DPTable
    [B------:R-:W-:-:S01]    ULDC.64 JumpsListBase, c[0x0][0x180] //JumpsList
    [B------:R-:W-:-:S01]    ULDC.64 DbgBase, c[0x0][0x190] //pDbg
    [B------:R-:W-:-:S01]    ULDC uBlockCnt, c[0x0][0x1AC]
    [B------:R-:W-:-:S01]    ULDC.64 DPsBase, c[0x0][0x1B8] //DPs_out
    [B------:R-:W-:-:S01]    ULDC.64 LoopTableBase, c[0x0][0x1C0] //LoopTable
    [B------:R-:W-:-:S01]    ULDC.64 LoopedKangsBase, c[0x0][0x1C8] //LoopedKangs
    [B------:R-:W-:-:S01]    ULDC.64 distsBase, c[0x0][0x1D0] //dists
    [B------:R-:W-:-:S01]    ULDC.64 JmpDists12Base, c[0x0][0x1D8] //JmpDists12
    [B------:R-:W-:-:S01]    ULDC uIterCnt, c[0x0][0x1A8]
    [B------:R-:W-:-:S01]    ULDC uKangCnt, c[0x0][0x1E0]
#ELSE
    [B------:R-:W-:-:S01]    UMOV URZ, 0x00 //for sm_120 we have limited UR support, same 63 instead of 255
    [B------:R-:W0:-:S01]    LDCU.64 CPolicy, c[0x0][0x358] //zero cache policy UR0..UR1
    [B------:R-:W0:-:S01]    LDCU.64 DPTableBase, c[0x0][{0x170 + 0x220}] //DPTable
    [B------:R-:W0:-:S01]    LDCU.64 JumpsListBase, c[0x0][{0x180 + 0x220}] //JumpsList
    [B------:R-:W0:-:S01]    LDCU.64 DbgBase, c[0x0][{0x190 + 0x220}] //pDbg
    [B------:R-:W0:-:S01]    LDCU uBlockCnt, c[0x0][{0x1AC + 0x220}]
    [B------:R-:W0:-:S01]    LDCU.64 DPsBase, c[0x0][{0x1B8 + 0x220}] //DPs_out
    [B------:R-:W0:-:S01]    LDCU.64 LoopTableBase, c[0x0][{0x1C0 + 0x220}] //LoopTable
    [B------:R-:W0:-:S01]    LDCU.64 LoopedKangsBase, c[0x0][{0x1C8 + 0x220}] //LoopedKangs
    [B------:R-:W0:-:S01]    LDCU.64 distsBase, c[0x0][{0x1D0 + 0x220}] //dists
    [B------:R-:W0:-:S01]    LDCU.64 JmpDists12Base, c[0x0][{0x1D8 + 0x220}] //JmpDists12
    [B------:R-:W0:-:S01]    LDCU uIterCnt, c[0x0][{0x1A8 + 0x220}]
    [B------:R-:W0:-:S01]    LDCU uKangCnt, c[0x0][{0x1E0 + 0x220}]
#ENDIF
    [B0-----:R-:W-:-:S01]    IMAD gID, BlockID, 0x100, ThrID
    [B------:R-:W-:-:S01]    UMOV uOne, 0x01
    [B------:R-:W-:-:S01]    MOV One, 0x01
    [B------:R-:W-:-:S01]    MOV IterCnt, uIterCnt

//write JmpDists12Base to LDS (48KB)
    [B------:R-:W-:-:S09]    MOV TmpA, {6 - 1} //we need 6 repeats for 48KB
    [B------:R-:W-:-:S09]    IMAD TmpB, ThrID, 0x20, RZ
    [B------:R-:W-:-:S09]    IMAD TmpC, ThrID, 0x2, RZ
.label_ldsB_loop:
    [B------:R-:W-:-:S01]    ISETP.EQ.AND P0, PT, TmpA, 0x00, PT
    [B------:R4:W0:-:S02]    LDG.E.NA.EFL2.256.STRONG_GPU TmpTmp4, TmpTmp0, [TmpB.U32 + JmpDists12Base], 0xFF
    [B------:R-:W-:-:S01]    IADD3 TmpA, TmpA, -0x01, RZ
    [B----4-:R-:W-:-:S02]    IADD3 TmpB, TmpB, 0x2000, RZ
    [B0-----:R5:W-:-:S02]    STS.128 [TmpC.X16], TmpTmp0
    [B------:R5:W-:-:S02]    STS.128 [TmpC.X16 + 0x10], TmpTmp4
    [B-----5:R-:W-:-:S02]    IADD3 TmpC, TmpC, 0x200, RZ
    [B------:R-:W-:Y:S01] @!P0 BRA.U `(.label_ldsB_loop)
.label_ldsB_end:
    [B------:R-:W-:-:S09]    IMAD.U32 ThrIDx, ThrID, 0x04, RZ
    [B------:R-:W-:Y:S06]    BAR.SYNC.DEFER_BLOCKING 0x0
    [B------:R-:W-:-:S06]    UIADD3 uTmpTmp1, uIterCnt, {MD_LEN}, URZ
    [B------:R-:W-:-:S09]    UIMAD.U32 uTmpA, uTmpTmp1, {GROUP_SIZE * 256 * 2}, URZ
    [B------:R-:W-:-:S09]    UIMAD.WIDE.U32 JumpsListBase, uTmpA, uBlockID, JumpsListBase

// [B------:R-:W-:-:S09]    UMOV uIterCnt, 0x01 //DEBUG

    [B------:R-:W-:-:S09]    UIADD3 uIterCntP, uIterCnt, -1, URZ

    [B------:R-:W-:-:S09]    UMOV uRetAdr1, 0xFFFFFFFF
    [B------:R-:W-:-:S06]    MOV KangCnt, uKangCnt
    [B------:R-:W-:-:S01]    IMAD.U32 KangCntF, KangCnt, 0x04, RZ
    [B------:R-:W-:-:S01]    MOV TenTh, 0x10000

//main iter loop prolog
    [B------:R-:W-:-:S09]    UMOV uMainLoopInd, 0x00
    [B------:R-:W-:-:S09]    UMOV uMainLoopPnt0, 0x00
    [B------:R-:W-:-:S09]    UMOV uMainLoopPnt1, 0x00
    [B------:R-:W-:-:S09]    UMOV uMainLoopHlp, `(.relP_main_loop_end) - 0x10
.main_loop_beg:
//load regs
    [B------:R-:W-:-:S01]    IMAD.U32 TmpA, BlockID, {MD_LEN * 256 * GROUP_SIZE * 8}, RZ //MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X
    [B------:R-:W-:-:S05]    MOV TmpB, {2 * MD_LEN * 256 * 8}    //2 * MD_LEN * BLOCK_SIZE * gr_ind2
    [B------:R-:W-:-:S05]    IMAD.U32 TmpA, TmpB, uMainLoopInd, TmpA
    [B------:R-:W-:-:S06]    IMAD.U32 TmpD, ThrID, 0x08, TmpA
    [B------:R-:W0:-:S02]    LDG.E.64 RegsA0, [TmpD.U32 + LoopTableBase + {0*8*256}]
    [B------:R-:W0:-:S02]    LDG.E.64 RegsB0, [TmpD.U32 + LoopTableBase + {0*8*256 + MD_LEN * 256 * 8}]
    [B------:R-:W0:-:S02]    LDG.E.64 RegsA2, [TmpD.U32 + LoopTableBase + {1*8*256}]
    [B------:R-:W0:-:S02]    LDG.E.64 RegsB2, [TmpD.U32 + LoopTableBase + {1*8*256 + MD_LEN * 256 * 8}]
    [B------:R-:W0:-:S02]    LDG.E.64 RegsA4, [TmpD.U32 + LoopTableBase + {2*8*256}]
    [B------:R-:W0:-:S02]    LDG.E.64 RegsB4, [TmpD.U32 + LoopTableBase + {2*8*256 + MD_LEN * 256 * 8}]
    [B------:R-:W0:-:S02]    LDG.E.64 RegsA6, [TmpD.U32 + LoopTableBase + {3*8*256}]
    [B------:R-:W0:-:S02]    LDG.E.64 RegsB6, [TmpD.U32 + LoopTableBase + {3*8*256 + MD_LEN * 256 * 8}]
    [B------:R-:W0:-:S02]    LDG.E.64 RegsA8, [TmpD.U32 + LoopTableBase + {4*8*256}]
    [B------:R-:W0:-:S02]    LDG.E.64 RegsB8, [TmpD.U32 + LoopTableBase + {4*8*256 + MD_LEN * 256 * 8}]
    [B------:R-:W0:-:S02]    LDG.E.64 RegsA10, [TmpD.U32 + LoopTableBase + {5*8*256}]
    [B------:R-:W0:-:S02]    LDG.E.64 RegsB10, [TmpD.U32 + LoopTableBase + {5*8*256 + MD_LEN * 256 * 8}]
    [B------:R-:W0:-:S02]    LDG.E.64 RegsA12, [TmpD.U32 + LoopTableBase + {6*8*256}]
    [B------:R-:W0:-:S02]    LDG.E.64 RegsB12, [TmpD.U32 + LoopTableBase + {6*8*256 + MD_LEN * 256 * 8}]
    [B------:R-:W0:-:S02]    LDG.E.64 RegsA14, [TmpD.U32 + LoopTableBase + {7*8*256}]
    [B------:R-:W0:-:S02]    LDG.E.64 RegsB14, [TmpD.U32 + LoopTableBase + {7*8*256 + MD_LEN * 256 * 8}]
    [B------:R-:W0:-:S02]    LDG.E.64 RegsA16, [TmpD.U32 + LoopTableBase + {8*8*256}]
    [B------:R-:W0:-:S02]    LDG.E.64 RegsB16, [TmpD.U32 + LoopTableBase + {8*8*256 + MD_LEN * 256 * 8}]
    [B------:R-:W0:-:S02]    LDG.E.64 RegsA18, [TmpD.U32 + LoopTableBase + {9*8*256}]
    [B------:R-:W0:-:S02]    LDG.E.64 RegsB18, [TmpD.U32 + LoopTableBase + {9*8*256 + MD_LEN * 256 * 8}]
////
    [B------:R-:W-:-:S01]    MOV cur_indA, 0x00
    [B------:R-:W-:-:S01]    MOV cur_indB, 0x00

    [B------:R-:W-:-:S01]    UIMAD.WIDE.U32 JumpsListBaseCur, uMainLoopInd, {256 * 4}, JumpsListBase
//calc original kang_ind
    [B------:R-:W-:-:S06]    UIMAD.U32 uTmpA, uMainLoopInd, 256, URZ
    [B------:R-:W-:-:S06]    IADD3 TmpA, ThrID, uTmpA, RZ //tind
    [B------:R-:W-:-:S06]    IMAD.U32 TmpB, TmpA, 0x02, RZ
    [B------:R-:W-:-:S06]    LOP3.LUT TmpB, TmpB, 0xFF, RZ, 0xC0, !PT //thr_ind
    [B------:R-:W-:-:S06]    SHF.R.U32.HI TmpC, RZ, 0x07, TmpA //gr_ind

    [B------:R-:W-:-:S06]    UIMAD.U32 uTmpA, uBlockCnt, 0x100, URZ
    [B------:R-:W-:-:S06]    UIMAD.U32 uTmpB, uBlockID, 0x100, URZ
    [B------:R-:W-:-:S06]    IMAD.U32 kang_indA, TmpC, uTmpA, TmpB
    [B------:R-:W-:-:S06]    IADD3 kang_indA, kang_indA, uTmpB, RZ
    [B------:R-:W-:-:S06]    IADD3 kang_indB, kang_indA, 0x01, RZ
    [B------:R-:W-:-:S06]    IMAD.U32 kang_indAF, kang_indA, 0x04, RZ
    [B------:R-:W-:-:S06]    IMAD.U32 kang_indBF, kang_indB, 0x04, RZ
//
// [B------:R-:W-:-:S09]    MOV TmpA, kang_indA
// [B------:R4:W-:-:S09]    RED.E.EL.MAX.STRONG_GPU [RZ.U32 + DbgBase + {0}], TmpA
// [B----4-:R-:W-:-:S09]    NOP

    [B------:R-:W-:-:S01]    ISETP.EQ.AND P5, PT, RZ, RZ, PT //LoopedA
    [B------:R-:W-:-:S01]    ISETP.EQ.AND P6, PT, RZ, RZ, PT //LoopedB

    [B------:R-:W-:-:S06]    IMAD.U32 TmpA, kang_indA, 0x20, RZ
    [B------:R4:W0:-:S02]    LDG.E.NA.EFL2.256.STRONG_GPU dA4, dA0, [TmpA.U32 + distsBase + 0x00], 0x3F
    [B------:R-:W-:-:S06]    IMAD.U32 TmpB, kang_indB, 0x20, RZ
    [B------:R4:W0:-:S02]    LDG.E.NA.EFL2.256.STRONG_GPU dB4, dB0, [TmpB.U32 + distsBase + 0x00], 0x3F
    [B0-----:R-:W-:-:S01]    NOP

    [B------:R-:W-:-:S05]    MOV jAdr, ThrIDx

//order data in advance
    [B------:R4:W0:-:S02]    LDG.E.NA.STRONG_SM cur_jAB0, [jAdr.U32 + JumpsListBaseCur + {0 * 256 * GROUP_SIZE * 2}]
    [B------:R4:W0:-:S02]    LDG.E.NA.STRONG_SM cur_jAB1, [jAdr.U32 + JumpsListBaseCur + {1 * 256 * GROUP_SIZE * 2}]
    [B------:R4:W0:-:S02]    LDG.E.NA.STRONG_SM cur_jAB2, [jAdr.U32 + JumpsListBaseCur + {2 * 256 * GROUP_SIZE * 2}]
    [B------:R4:W0:-:S02]    LDG.E.NA.STRONG_SM cur_jAB3, [jAdr.U32 + JumpsListBaseCur + {3 * 256 * GROUP_SIZE * 2}]
    [B------:R4:W0:-:S02]    LDG.E.NA.STRONG_SM cur_jAB4, [jAdr.U32 + JumpsListBaseCur + {4 * 256 * GROUP_SIZE * 2}]
    [B------:R4:W0:-:S02]    LDG.E.NA.STRONG_SM cur_jAB5, [jAdr.U32 + JumpsListBaseCur + {5 * 256 * GROUP_SIZE * 2}]
    [B------:R4:W0:-:S02]    LDG.E.NA.STRONG_SM cur_jAB6, [jAdr.U32 + JumpsListBaseCur + {6 * 256 * GROUP_SIZE * 2}]
    [B------:R4:W0:-:S02]    LDG.E.NA.STRONG_SM cur_jAB7, [jAdr.U32 + JumpsListBaseCur + {7 * 256 * GROUP_SIZE * 2}]
    [B------:R4:W0:-:S02]    LDG.E.NA.STRONG_SM cur_jAB8, [jAdr.U32 + JumpsListBaseCur + {8 * 256 * GROUP_SIZE * 2}]
    [B------:R4:W0:-:S02]    LDG.E.NA.STRONG_SM cur_jAB9, [jAdr.U32 + JumpsListBaseCur + {9 * 256 * GROUP_SIZE * 2}]
    [B----4-:R-:W-:-:S01]    IADD3 jAdr, jAdr, {10 * 256 * GROUP_SIZE * 2}, RZ

//step iter loop prolog
    [B------:R-:W-:-:S01]    MOV StepLoopInd, 0x00
    [B------:R-:W-:-:S01]    UMOV uStepLoopInd, 0x00
    [B------:R-:W-:-:S01]    UMOV uStepLoopPnt0, 0x00
    [B------:R-:W-:-:S01]    UMOV uStepLoopPnt1, 0x00
    [B------:R-:W-:-:S01]    UMOV uStepLoopHlp, `(.relP_step_loop_end) - 0x10
.step_loop_beg:

//RvA = 2 * ((iter + MD_LEN - 4) % MD_LEN)
//RvB = 2 * ((iter + MD_LEN - 6) % MD_LEN)
//RvC = 2 * ((iter + MD_LEN - 8) % MD_LEN)
//RvD = iter

//DO_ITER(0)
include FastPath(Rt=TmpTmp, CiterNxt=0x01, RvA_A=RegsA12, RvB_A=RegsA8, RvC_A=RegsA4, RvD_A=RegsA0, RvA_B=RegsB12, RvB_B=RegsB8, RvC_B=RegsB4, RvD_B=RegsB0, Rcur_jAB=cur_jAB0)
    [B------:R-:W-:-:S01]    BRA.DIV P0, ~URZ `(.slow_0)
.ret_slow_0:
    [B------:R-:W-:-:S01] @P5 MOV RegsA0, dA0
    [B------:R-:W-:-:S01] @P5 MOV RegsA1, dA1
    [B------:R-:W-:-:S01] @P5 MOV cur_indA, 0x01
    [B------:R-:W-:-:S01] @P6 MOV RegsB0, dB0
    [B------:R-:W-:-:S01] @P6 MOV RegsB1, dB1
    [B------:R-:W-:-:S01] @P6 MOV cur_indB, 0x01

//DO_ITER(1)
include FastPath(Rt=TmpTmp, CiterNxt=0x02, RvA_A=RegsA14, RvB_A=RegsA10, RvC_A=RegsA6, RvD_A=RegsA2, RvA_B=RegsB14, RvB_B=RegsB10, RvC_B=RegsB6, RvD_B=RegsB2, Rcur_jAB=cur_jAB1)
    [B------:R-:W-:-:S01]    BRA.DIV P0, ~URZ `(.slow_1)
.ret_slow_1:
    [B------:R-:W-:-:S01] @P5 MOV RegsA2, dA0
    [B------:R-:W-:-:S01] @P5 MOV RegsA3, dA1
    [B------:R-:W-:-:S01] @P5 MOV cur_indA, 0x02
    [B------:R-:W-:-:S01] @P6 MOV RegsB2, dB0
    [B------:R-:W-:-:S01] @P6 MOV RegsB3, dB1
    [B------:R-:W-:-:S01] @P6 MOV cur_indB, 0x02

//DO_ITER(2)
include FastPath(Rt=TmpTmp, CiterNxt=0x03, RvA_A=RegsA16, RvB_A=RegsA12, RvC_A=RegsA8, RvD_A=RegsA4, RvA_B=RegsB16, RvB_B=RegsB12, RvC_B=RegsB8, RvD_B=RegsB4, Rcur_jAB=cur_jAB2)
    [B------:R-:W-:-:S01]    BRA.DIV P0, ~URZ `(.slow_2)
.ret_slow_2:
    [B------:R-:W-:-:S01] @P5 MOV RegsA4, dA0
    [B------:R-:W-:-:S01] @P5 MOV RegsA5, dA1
    [B------:R-:W-:-:S01] @P5 MOV cur_indA, 0x03
    [B------:R-:W-:-:S01] @P6 MOV RegsB4, dB0
    [B------:R-:W-:-:S01] @P6 MOV RegsB5, dB1
    [B------:R-:W-:-:S01] @P6 MOV cur_indB, 0x03

//DO_ITER(3)
include FastPath(Rt=TmpTmp, CiterNxt=0x04, RvA_A=RegsA18, RvB_A=RegsA14, RvC_A=RegsA10, RvD_A=RegsA6, RvA_B=RegsB18, RvB_B=RegsB14, RvC_B=RegsB10, RvD_B=RegsB6, Rcur_jAB=cur_jAB3)
    [B------:R-:W-:-:S01]    BRA.DIV P0, ~URZ `(.slow_3)
.ret_slow_3:
    [B------:R-:W-:-:S01] @P5 MOV RegsA6, dA0
    [B------:R-:W-:-:S01] @P5 MOV RegsA7, dA1
    [B------:R-:W-:-:S01] @P5 MOV cur_indA, 0x04
    [B------:R-:W-:-:S01] @P6 MOV RegsB6, dB0
    [B------:R-:W-:-:S01] @P6 MOV RegsB7, dB1
    [B------:R-:W-:-:S01] @P6 MOV cur_indB, 0x04

//DO_ITER(4)
include FastPath(Rt=TmpTmp, CiterNxt=0x05, RvA_A=RegsA0, RvB_A=RegsA16, RvC_A=RegsA12, RvD_A=RegsA8, RvA_B=RegsB0, RvB_B=RegsB16, RvC_B=RegsB12, RvD_B=RegsB8, Rcur_jAB=cur_jAB4)
    [B------:R-:W-:-:S01]    BRA.DIV P0, ~URZ `(.slow_4)
.ret_slow_4:
    [B------:R-:W-:-:S01] @P5 MOV RegsA8, dA0
    [B------:R-:W-:-:S01] @P5 MOV RegsA9, dA1
    [B------:R-:W-:-:S01] @P5 MOV cur_indA, 0x05
    [B------:R-:W-:-:S01] @P6 MOV RegsB8, dB0
    [B------:R-:W-:-:S01] @P6 MOV RegsB9, dB1
    [B------:R-:W-:-:S01] @P6 MOV cur_indB, 0x05

//DO_ITER(5)
include FastPath(Rt=TmpTmp, CiterNxt=0x06, RvA_A=RegsA2, RvB_A=RegsA18, RvC_A=RegsA14, RvD_A=RegsA10, RvA_B=RegsB2, RvB_B=RegsB18, RvC_B=RegsB14, RvD_B=RegsB10, Rcur_jAB=cur_jAB5)
    [B------:R-:W-:-:S01]    BRA.DIV P0, ~URZ `(.slow_5)
.ret_slow_5:
    [B------:R-:W-:-:S01] @P5 MOV RegsA10, dA0
    [B------:R-:W-:-:S01] @P5 MOV RegsA11, dA1
    [B------:R-:W-:-:S01] @P5 MOV cur_indA, 0x06
    [B------:R-:W-:-:S01] @P6 MOV RegsB10, dB0
    [B------:R-:W-:-:S01] @P6 MOV RegsB11, dB1
    [B------:R-:W-:-:S01] @P6 MOV cur_indB, 0x06

//DO_ITER(6)
include FastPath(Rt=TmpTmp, CiterNxt=0x07, RvA_A=RegsA4, RvB_A=RegsA0, RvC_A=RegsA16, RvD_A=RegsA12, RvA_B=RegsB4, RvB_B=RegsB0, RvC_B=RegsB16, RvD_B=RegsB12, Rcur_jAB=cur_jAB6)
    [B------:R-:W-:-:S01]    BRA.DIV P0, ~URZ `(.slow_6)
.ret_slow_6:
    [B------:R-:W-:-:S01] @P5 MOV RegsA12, dA0
    [B------:R-:W-:-:S01] @P5 MOV RegsA13, dA1
    [B------:R-:W-:-:S01] @P5 MOV cur_indA, 0x07
    [B------:R-:W-:-:S01] @P6 MOV RegsB12, dB0
    [B------:R-:W-:-:S01] @P6 MOV RegsB13, dB1
    [B------:R-:W-:-:S01] @P6 MOV cur_indB, 0x07

//DO_ITER(7)
include FastPath(Rt=TmpTmp, CiterNxt=0x08, RvA_A=RegsA6, RvB_A=RegsA2, RvC_A=RegsA18, RvD_A=RegsA14, RvA_B=RegsB6, RvB_B=RegsB2, RvC_B=RegsB18, RvD_B=RegsB14, Rcur_jAB=cur_jAB7)
    [B------:R-:W-:-:S01]    BRA.DIV P0, ~URZ `(.slow_7)
.ret_slow_7:
    [B------:R-:W-:-:S01] @P5 MOV RegsA14, dA0
    [B------:R-:W-:-:S01] @P5 MOV RegsA15, dA1
    [B------:R-:W-:-:S01] @P5 MOV cur_indA, 0x08
    [B------:R-:W-:-:S01] @P6 MOV RegsB14, dB0
    [B------:R-:W-:-:S01] @P6 MOV RegsB15, dB1
    [B------:R-:W-:-:S01] @P6 MOV cur_indB, 0x08

//DO_ITER(8)
include FastPath(Rt=TmpTmp, CiterNxt=0x09, RvA_A=RegsA8, RvB_A=RegsA4, RvC_A=RegsA0, RvD_A=RegsA16, RvA_B=RegsB8, RvB_B=RegsB4, RvC_B=RegsB0, RvD_B=RegsB16, Rcur_jAB=cur_jAB8)
    [B------:R-:W-:-:S01]    BRA.DIV P0, ~URZ `(.slow_8)
.ret_slow_8:
    [B------:R-:W-:-:S01] @P5 MOV RegsA16, dA0
    [B------:R-:W-:-:S01] @P5 MOV RegsA17, dA1
    [B------:R-:W-:-:S01] @P5 MOV cur_indA, 0x09
    [B------:R-:W-:-:S01] @P6 MOV RegsB16, dB0
    [B------:R-:W-:-:S01] @P6 MOV RegsB17, dB1
    [B------:R-:W-:-:S01] @P6 MOV cur_indB, 0x09

//DO_ITER(9)
include FastPath(Rt=TmpTmp, CiterNxt=0x00, RvA_A=RegsA10, RvB_A=RegsA6, RvC_A=RegsA2, RvD_A=RegsA18, RvA_B=RegsB10, RvB_B=RegsB6, RvC_B=RegsB2, RvD_B=RegsB18, Rcur_jAB=cur_jAB9)
    [B------:R-:W-:-:S01]    BRA.DIV P0, ~URZ `(.slow_9)
.ret_slow_9:
    [B------:R-:W-:-:S01] @P5 MOV RegsA18, dA0
    [B------:R-:W-:-:S01] @P5 MOV RegsA19, dA1
    [B------:R-:W-:-:S01] @P5 MOV cur_indA, 0x00
    [B------:R-:W-:-:S01] @P6 MOV RegsB18, dB0
    [B------:R-:W-:-:S01] @P6 MOV RegsB19, dB1
    [B------:R-:W-:-:S01] @P6 MOV cur_indB, 0x00



    [B------:R-:W-:-:S09]    UIADD3 uStepLoopInd, uStepLoopInd, 0x0A, URZ
    [B------:R-:W-:-:S09]    IADD3 StepLoopInd, StepLoopInd, 0x0A, RZ
// handle step iter loop
    [B------:R-:W-:Y:S13]    UISETP.GE.AND UP0, UPT, uStepLoopInd, uIterCntP, UPT
    [B------:R-:W-:-:S09] @UP0 UMOV uStepLoopPnt0, uStepLoopHlp
    [B------:R-:W-:-:S09]    BRXU.U uStepLoopPnt, `(.step_loop_beg)
.relP_step_loop_end:
//
    [B------:R-:W-:-:S04]    IMAD.U32 kang_indA, kang_indA, 32, RZ
    [B------:R-:W-:-:S01]    IMAD.U32 kang_indB, kang_indB, 32, RZ
    [B------:R3:W-:-:S02]    STG.E.64 [kang_indA.U32 + distsBase + 0x00], dA0
    [B------:R3:W-:-:S02]    STG.E.64 [kang_indA.U32 + distsBase + 0x08], dA2
    [B------:R3:W-:-:S02]    STG.E.64 [kang_indA.U32 + distsBase + 0x10], dA4
    [B------:R3:W-:-:S02]    STG.E.64 [kang_indB.U32 + distsBase + 0x00], dB0
    [B------:R3:W-:-:S02]    STG.E.64 [kang_indB.U32 + distsBase + 0x08], dB2
    [B------:R3:W-:-:S02]    STG.E.64 [kang_indB.U32 + distsBase + 0x10], dB4

//store so cur_ind is 0 at next loading
    [B------:R-:W-:-:S01]    IADD3 cur_indA, RZ, {MD_LEN}, -cur_indA
    [B------:R-:W-:-:S01]    IADD3 cur_indB, RZ, {MD_LEN}, -cur_indB
    [B------:R-:W-:-:S01]    IMAD.U32 TmpA, BlockID, {MD_LEN * 256 * GROUP_SIZE * 8}, RZ  //MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X
    [B------:R-:W-:-:S05]    MOV TmpB, {2 * MD_LEN * 256 * 8}    //2 * MD_LEN * BLOCK_SIZE * gr_ind2
    [B------:R-:W-:-:S06]    IMAD.U32 TmpD, TmpB, uMainLoopInd, TmpA
    [B------:R-:W-:-:S01]    UISETP.GE.AND UP0, UPT, uMainLoopInd, {GROUP_SIZE / 2 - 1}, UPT
include SaveRegAB(RegA=RegsA0, RegB=RegsB0, RCurA=cur_indA, RCurB=cur_indB, Rt=TmpTmp)
include SaveRegAB(RegA=RegsA2, RegB=RegsB2, RCurA=cur_indA, RCurB=cur_indB, Rt=TmpTmp)
include SaveRegAB(RegA=RegsA4, RegB=RegsB4, RCurA=cur_indA, RCurB=cur_indB, Rt=TmpTmp)
include SaveRegAB(RegA=RegsA6, RegB=RegsB6, RCurA=cur_indA, RCurB=cur_indB, Rt=TmpTmp)
include SaveRegAB(RegA=RegsA8, RegB=RegsB8, RCurA=cur_indA, RCurB=cur_indB, Rt=TmpTmp)
include SaveRegAB(RegA=RegsA10, RegB=RegsB10, RCurA=cur_indA, RCurB=cur_indB, Rt=TmpTmp)
include SaveRegAB(RegA=RegsA12, RegB=RegsB12, RCurA=cur_indA, RCurB=cur_indB, Rt=TmpTmp)
include SaveRegAB(RegA=RegsA14, RegB=RegsB14, RCurA=cur_indA, RCurB=cur_indB, Rt=TmpTmp)
include SaveRegAB(RegA=RegsA16, RegB=RegsB16, RCurA=cur_indA, RCurB=cur_indB, Rt=TmpTmp)
include SaveRegAB(RegA=RegsA18, RegB=RegsB18, RCurA=cur_indA, RCurB=cur_indB, Rt=TmpTmp)
    [B---345:R-:W-:-:S01]    NOP

// handle main iter loop
    [B------:R-:W-:-:S01] @UP0 UMOV uMainLoopPnt0, uMainLoopHlp
    [B------:R-:W-:-:S01]    UIADD3 uMainLoopInd, uMainLoopInd, 0x01, URZ
    [B------:R-:W-:-:S01]    BRXU.U uMainLoopPnt, `(.main_loop_beg)
.relP_main_loop_end:

    [B------:R-:W-:-:S05]    EXIT

.slow_0:
include PJD(Rt=TmpTmp, CiterNxt=0x01, RvA_A=RegsA12, RvB_A=RegsA8, RvC_A=RegsA4, RvD_A=RegsA0, RvA_B=RegsB12, RvB_B=RegsB8, RvC_B=RegsB4, RvD_B=RegsB0, CHlpOfs=0xFFFFFFFF)
    [B------:R-:W-:Y:S01]    BRA.U `(.ret_slow_0)

.slow_1:
include PJD(Rt=TmpTmp, CiterNxt=0x02, RvA_A=RegsA14, RvB_A=RegsA10, RvC_A=RegsA6, RvD_A=RegsA2, RvA_B=RegsB14, RvB_B=RegsB10, RvC_B=RegsB6, RvD_B=RegsB2, CHlpOfs=0xFFFFFFFE)
    [B------:R-:W-:Y:S01]    BRA.U `(.ret_slow_1)

.slow_2:
include PJD(Rt=TmpTmp, CiterNxt=0x03, RvA_A=RegsA16, RvB_A=RegsA12, RvC_A=RegsA8, RvD_A=RegsA4, RvA_B=RegsB16, RvB_B=RegsB12, RvC_B=RegsB8, RvD_B=RegsB4, CHlpOfs=0xFFFFFFFD)
    [B------:R-:W-:Y:S01]    BRA.U `(.ret_slow_2)

.slow_3:
include PJD(Rt=TmpTmp, CiterNxt=0x04, RvA_A=RegsA18, RvB_A=RegsA14, RvC_A=RegsA10, RvD_A=RegsA6, RvA_B=RegsB18, RvB_B=RegsB14, RvC_B=RegsB10, RvD_B=RegsB6, CHlpOfs=0xFFFFFFFC)
    [B------:R-:W-:Y:S01]    BRA.U `(.ret_slow_3)

.slow_4:
include PJD(Rt=TmpTmp, CiterNxt=0x05, RvA_A=RegsA0, RvB_A=RegsA16, RvC_A=RegsA12, RvD_A=RegsA8, RvA_B=RegsB0, RvB_B=RegsB16, RvC_B=RegsB12, RvD_B=RegsB8, CHlpOfs=0xFFFFFFFB)
    [B------:R-:W-:Y:S01]    BRA.U `(.ret_slow_4)

.slow_5:
include PJD(Rt=TmpTmp, CiterNxt=0x06, RvA_A=RegsA2, RvB_A=RegsA18, RvC_A=RegsA14, RvD_A=RegsA10, RvA_B=RegsB2, RvB_B=RegsB18, RvC_B=RegsB14, RvD_B=RegsB10, CHlpOfs=0xFFFFFFFA)
    [B------:R-:W-:Y:S01]    BRA.U `(.ret_slow_5)

.slow_6:
include PJD(Rt=TmpTmp, CiterNxt=0x07, RvA_A=RegsA4, RvB_A=RegsA0, RvC_A=RegsA16, RvD_A=RegsA12, RvA_B=RegsB4, RvB_B=RegsB0, RvC_B=RegsB16, RvD_B=RegsB12, CHlpOfs=0xFFFFFFF9)
    [B------:R-:W-:Y:S01]    BRA.U `(.ret_slow_6)

.slow_7:
include PJD(Rt=TmpTmp, CiterNxt=0x08, RvA_A=RegsA6, RvB_A=RegsA2, RvC_A=RegsA18, RvD_A=RegsA14, RvA_B=RegsB6, RvB_B=RegsB2, RvC_B=RegsB18, RvD_B=RegsB14, CHlpOfs=0xFFFFFFF8)
    [B------:R-:W-:Y:S01]    BRA.U `(.ret_slow_7)

.slow_8:
include PJD(Rt=TmpTmp, CiterNxt=0x09, RvA_A=RegsA8, RvB_A=RegsA4, RvC_A=RegsA0, RvD_A=RegsA16, RvA_B=RegsB8, RvB_B=RegsB4, RvC_B=RegsB0, RvD_B=RegsB16, CHlpOfs=0xFFFFFFF7)
    [B------:R-:W-:Y:S01]    BRA.U `(.ret_slow_8)

.slow_9:
include PJD(Rt=TmpTmp, CiterNxt=0x00, RvA_A=RegsA10, RvB_A=RegsA6, RvC_A=RegsA2, RvD_A=RegsA18, RvA_B=RegsB10, RvB_B=RegsB6, RvC_B=RegsB2, RvD_B=RegsB18, CHlpOfs=0xFFFFFFF6)
    [B------:R-:W-:Y:S01]    BRA.U `(.ret_slow_9)

////end KernelB
}